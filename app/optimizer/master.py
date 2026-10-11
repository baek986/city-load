"""Exact heterogeneous VRPTW master.

CP-SAT minimizes transport cost and driving time under weight, volume, and time
windows. Extra stacking implied by floor area and cargo height is in the
objective, so the bound sees the same handling layers as a packed plan. Packing
cuts forbid a set of stops from sharing a vehicle. Circuit constraints replace
MTZ after a first incumbent exists on large instances; they are used from the
start on small ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from app.domain.matrix import TravelMatrix
from app.domain.models import CustomerNode, Dataset, Scales, Weights

SCALE = 1_000_000
DEPOT = "depot"


@dataclass(frozen=True)
class SetCut:
    node_ids: frozenset[str]
    vehicle_ids: frozenset[str] | None = None


@dataclass(frozen=True)
class ArcCut:
    arcs: tuple[tuple[str, str], ...]
    vehicle_ids: frozenset[str] | None = None


@dataclass(frozen=True)
class AssignmentCut:
    """Forbid one exact node-to-vehicle partition, so route optima can be enumerated."""

    pairs: tuple[tuple[str, str], ...]


@dataclass
class MasterOutcome:
    status: str
    assignment: dict[str, list[str]] = field(default_factory=dict)
    bound_j: float | None = None
    objective_j: float | None = None


def sequence_arcs(node_ids: list[str]) -> tuple[tuple[str, str], ...]:
    points = [DEPOT, *node_ids, DEPOT]
    return tuple((points[index], points[index + 1]) for index in range(len(points) - 1))


def _canonical_hint(hint: dict[str, list[str]], vehicles: list) -> dict[str, list[str]]:
    """Move each hinted route onto the lowest-index identical vehicle so symmetry constraints accept it."""
    groups: dict[tuple, list[str]] = {}
    signature_of: dict[str, tuple] = {}
    for vehicle in vehicles:
        signature = (
            vehicle.type_name,
            vehicle.length_mm,
            vehicle.width_mm,
            vehicle.height_mm,
            vehicle.max_weight_g,
            vehicle.cost_per_km,
            vehicle.fixed_cost,
        )
        groups.setdefault(signature, []).append(vehicle.id)
        signature_of[vehicle.id] = signature
    pools = {signature: list(ids) for signature, ids in groups.items()}
    rewritten = {vehicle.id: [] for vehicle in vehicles}
    for vehicle in vehicles:
        sequence = hint.get(vehicle.id) or []
        if not sequence:
            continue
        target = pools[signature_of[vehicle.id]].pop(0)
        rewritten[target] = list(sequence)
    return rewritten


def _arc_reaches(
    origin: int,
    dest: int,
    nodes: list[CustomerNode],
    dataset: Dataset,
    matrix: TravelMatrix,
    locations: list[str],
) -> bool:
    """False when even the earliest departure misses the destination window."""
    arc = matrix.get(locations[origin], locations[dest])
    if origin == 0:
        depart = dataset.depot.window_open_s
    else:
        node = nodes[origin - 1]
        depart = node.window_open_s + node.service_time_s
    arrive = depart + arc.travel_time_s
    if dest == 0:
        return arrive <= dataset.depot.window_close_s
    return arrive <= nodes[dest - 1].window_close_s


def _arc_points(
    vehicle_cost_per_km: float,
    distance_m: int,
    travel_time_s: int,
    weights: Weights,
    scales: Scales,
) -> int:
    cost = distance_m / 1000.0 * vehicle_cost_per_km
    score = weights.cost * cost / scales.cost + weights.time * travel_time_s / scales.time
    return int(round(score * SCALE))


def solve_master(
    nodes: list[CustomerNode],
    dataset: Dataset,
    matrix: TravelMatrix,
    weights: Weights,
    scales: Scales,
    set_cuts: list[SetCut],
    arc_cuts: list[ArcCut],
    time_limit_s: float,
    seed: int,
    hint: dict[str, list[str]] | None = None,
    fixed_nodes: dict[str, str] | None = None,
    assignment_cuts: list[AssignmentCut] | None = None,
    search_workers: int = 1,
    use_circuit: bool = False,
) -> MasterOutcome:
    if not nodes:
        return MasterOutcome(status="optimal", assignment={}, bound_j=0.0, objective_j=0.0)
    model = cp_model.CpModel()
    vehicles = list(dataset.vehicles)
    vehicle_index_of = {vehicle.id: index for index, vehicle in enumerate(vehicles)}
    node_index = {node.node_id: index + 1 for index, node in enumerate(nodes)}
    customer_indexes = list(range(1, len(nodes) + 1))
    all_indexes = [0, *customer_indexes]
    horizon = max(dataset.depot.window_close_s, max(node.window_close_s for node in nodes))
    y: dict[tuple[int, int], cp_model.IntVar] = {}
    arcs: dict[tuple[int, int, int], cp_model.IntVar] = {}
    used: dict[int, cp_model.IntVar] = {}
    arrival: dict[tuple[int, int], cp_model.IntVar] = {}
    late: dict[tuple[int, int], cp_model.IntVar] = {}
    position: dict[tuple[int, int], cp_model.IntVar] = {}
    skip_node: dict[tuple[int, int], cp_model.IntVar] = {}
    tend: dict[int, cp_model.IntVar] = {}
    locations = [dataset.depot.id, *[node.store_id for node in nodes]]
    possible = [
        (origin, dest)
        for origin in all_indexes
        for dest in all_indexes
        if origin != dest and _arc_reaches(origin, dest, nodes, dataset, matrix, locations)
    ]

    for vehicle_index, _vehicle in enumerate(vehicles):
        used[vehicle_index] = model.NewBoolVar(f"used_{vehicle_index}")
        for index in all_indexes:
            arrival[vehicle_index, index] = model.NewIntVar(0, horizon, f"t_{vehicle_index}_{index}")
        for customer in customer_indexes:
            y[vehicle_index, customer] = model.NewBoolVar(f"y_{vehicle_index}_{customer}")
            late[vehicle_index, customer] = model.NewIntVar(0, horizon, f"late_{vehicle_index}_{customer}")
            if not use_circuit:
                position[vehicle_index, customer] = model.NewIntVar(0, len(nodes), f"u_{vehicle_index}_{customer}")
        for origin, dest in possible:
            arcs[vehicle_index, origin, dest] = model.NewBoolVar(f"a_{vehicle_index}_{origin}_{dest}")

    for customer in customer_indexes:
        model.Add(sum(y[vehicle_index, customer] for vehicle_index in range(len(vehicles))) == 1)

    stop_weight = int(round(weights.handling / scales.handling * SCALE))
    late_weight = int(round(weights.late / scales.late / 60.0 * SCALE))
    split_weight = int(round(weights.split / scales.split * SCALE))
    objective_terms = []

    for vehicle_index, vehicle in enumerate(vehicles):
        fixed = int(round(weights.cost * vehicle.fixed_cost / scales.cost * SCALE))
        if fixed:
            objective_terms.append(fixed * used[vehicle_index])
        departures = [arcs[vehicle_index, 0, customer] for customer in customer_indexes if (vehicle_index, 0, customer) in arcs]
        returns = [arcs[vehicle_index, customer, 0] for customer in customer_indexes if (vehicle_index, customer, 0) in arcs]
        model.Add(sum(departures) == used[vehicle_index])
        model.Add(sum(returns) == used[vehicle_index])
        model.Add(arrival[vehicle_index, 0] >= dataset.depot.window_open_s)
        model.Add(arrival[vehicle_index, 0] <= dataset.depot.window_close_s)
        return_time = model.NewIntVar(0, horizon, f"tend_{vehicle_index}")
        tend[vehicle_index] = return_time
        model.Add(return_time <= dataset.depot.window_close_s)
        for customer in customer_indexes:
            node = nodes[customer - 1]
            inbound = [arcs[vehicle_index, other, customer] for other in all_indexes if (vehicle_index, other, customer) in arcs]
            outbound = [arcs[vehicle_index, customer, other] for other in all_indexes if (vehicle_index, customer, other) in arcs]
            model.Add(sum(inbound) == y[vehicle_index, customer])
            model.Add(sum(outbound) == y[vehicle_index, customer])
            if not use_circuit:
                model.Add(position[vehicle_index, customer] == 0).OnlyEnforceIf(y[vehicle_index, customer].Not())
                model.Add(position[vehicle_index, customer] >= 1).OnlyEnforceIf(y[vehicle_index, customer])
            model.Add(arrival[vehicle_index, customer] >= node.window_open_s).OnlyEnforceIf(y[vehicle_index, customer])
            model.Add(arrival[vehicle_index, customer] <= node.window_close_s).OnlyEnforceIf(y[vehicle_index, customer])
            if node.soft_deadline_s is None:
                model.Add(late[vehicle_index, customer] == 0)
            else:
                model.Add(
                    late[vehicle_index, customer] >= arrival[vehicle_index, customer] - node.soft_deadline_s
                ).OnlyEnforceIf(y[vehicle_index, customer])
                model.Add(late[vehicle_index, customer] == 0).OnlyEnforceIf(y[vehicle_index, customer].Not())
            if stop_weight:
                objective_terms.append(stop_weight * y[vehicle_index, customer])
            if late_weight:
                objective_terms.append(late_weight * late[vehicle_index, customer])
        model.Add(
            sum(node.weight_g * y[vehicle_index, index] for index, node in enumerate(nodes, start=1))
            <= vehicle.max_weight_g
        )
        model.Add(
            sum(node.volume_mm3 * y[vehicle_index, index] for index, node in enumerate(nodes, start=1))
            <= vehicle.volume_mm3
        )
        floor_mm2 = max(vehicle.length_mm * vehicle.width_mm, 1)
        total_footprint = sum(max(node.footprint_mm2, 0) for node in nodes)
        extra_cap = max(1, len(nodes), (total_footprint + floor_mm2 - 1) // floor_mm2)
        layer_h = max((getattr(node, "max_extent_mm", 0) or 0) for node in nodes) if nodes else 0
        layer_vol = 0
        if layer_h > 0:
            layer_h = min(layer_h, vehicle.height_mm)
            layer_vol = floor_mm2 * max(layer_h, 1)
            extra_cap = max(
                extra_cap,
                (sum(node.volume_mm3 for node in nodes) + layer_vol - 1) // layer_vol,
            )
        extra_layers = model.NewIntVar(0, extra_cap, f"extra_{vehicle_index}")
        assigned_footprint = sum(
            max(node.footprint_mm2, 0) * y[vehicle_index, index] for index, node in enumerate(nodes, start=1)
        )
        model.Add(assigned_footprint <= floor_mm2 * extra_layers + floor_mm2)
        if layer_vol:
            assigned_volume = sum(
                node.volume_mm3 * y[vehicle_index, index] for index, node in enumerate(nodes, start=1)
            )
            model.Add(assigned_volume <= extra_layers * layer_vol + layer_vol)
        if stop_weight:
            objective_terms.append(stop_weight * extra_layers)

        service = [0] + [node.service_time_s for node in nodes]
        for origin in all_indexes:
            for dest in all_indexes:
                if origin == dest or (vehicle_index, origin, dest) not in arcs:
                    continue
                arc = matrix.get(locations[origin], locations[dest])
                travel = service[origin] + arc.travel_time_s
                if dest == 0:
                    model.Add(return_time >= arrival[vehicle_index, origin] + travel).OnlyEnforceIf(
                        arcs[vehicle_index, origin, dest]
                    )
                else:
                    model.Add(
                        arrival[vehicle_index, dest] >= arrival[vehicle_index, origin] + travel
                    ).OnlyEnforceIf(arcs[vehicle_index, origin, dest])
                points = _arc_points(vehicle.cost_per_km, arc.distance_m, arc.travel_time_s, weights, scales)
                if points:
                    objective_terms.append(points * arcs[vehicle_index, origin, dest])
                if not use_circuit and origin != 0 and dest != 0:
                    model.Add(
                        position[vehicle_index, dest] >= position[vehicle_index, origin] + 1
                    ).OnlyEnforceIf(arcs[vehicle_index, origin, dest])
        if use_circuit:
            circuit = [(origin, dest, arcs[vehicle_index, origin, dest]) for origin, dest in possible]
            circuit.append((0, 0, used[vehicle_index].Not()))
            for customer in customer_indexes:
                skip = model.NewBoolVar(f"skip_{vehicle_index}_{customer}")
                model.Add(skip + y[vehicle_index, customer] == 1)
                skip_node[vehicle_index, customer] = skip
                circuit.append((customer, customer, skip))
            model.AddCircuit(circuit)

    order_nodes: dict[str, list[CustomerNode]] = {}
    for node in nodes:
        for order_id in node.order_ids:
            order_nodes.setdefault(order_id, []).append(node)
    for order_id, members in order_nodes.items():
        if len(members) < 2:
            continue
        presents = []
        for vehicle_index in range(len(vehicles)):
            present = model.NewBoolVar(f"ord_{order_id}_{vehicle_index}")
            member_vars = [y[vehicle_index, node_index[member.node_id]] for member in members]
            for member_var in member_vars:
                model.AddImplication(member_var, present)
            model.Add(present <= sum(member_vars))
            presents.append(present)
        excess = model.NewIntVar(0, len(vehicles), f"split_{order_id}")
        model.Add(excess >= sum(presents) - 1)
        if split_weight:
            objective_terms.append(split_weight * excess)

    for cut in set_cuts:
        indexes = [node_index[node_id] for node_id in cut.node_ids if node_id in node_index]
        if len(indexes) < 2:
            continue
        for vehicle_index, vehicle in enumerate(vehicles):
            if cut.vehicle_ids is not None and vehicle.id not in cut.vehicle_ids:
                continue
            model.Add(sum(y[vehicle_index, index] for index in indexes) <= len(indexes) - 1)

    for cut in arc_cuts:
        for vehicle_index, vehicle in enumerate(vehicles):
            if cut.vehicle_ids is not None and vehicle.id not in cut.vehicle_ids:
                continue
            literals = []
            valid = True
            for origin_id, dest_id in cut.arcs:
                origin = 0 if origin_id == DEPOT else node_index.get(origin_id)
                dest = 0 if dest_id == DEPOT else node_index.get(dest_id)
                if origin is None or dest is None or (vehicle_index, origin, dest) not in arcs:
                    valid = False
                    break
                literals.append(arcs[vehicle_index, origin, dest])
            if valid and literals:
                model.Add(sum(literals) <= len(literals) - 1)

    for cut in assignment_cuts or []:
        literals = []
        valid = True
        for vehicle_id, node_id in cut.pairs:
            vehicle_index = vehicle_index_of.get(vehicle_id)
            customer = node_index.get(node_id)
            if vehicle_index is None or customer is None:
                valid = False
                break
            literals.append(y[vehicle_index, customer])
        if valid and len(literals) > 1:
            model.Add(sum(literals) <= len(literals) - 1)

    for node_id, vehicle_id in (fixed_nodes or {}).items():
        if node_id not in node_index or vehicle_id not in vehicle_index_of:
            continue
        model.Add(y[vehicle_index_of[vehicle_id], node_index[node_id]] == 1)

    if not fixed_nodes:
        groups: dict[tuple, list[int]] = {}
        for vehicle_index, vehicle in enumerate(vehicles):
            signature = (
                vehicle.type_name,
                vehicle.length_mm,
                vehicle.width_mm,
                vehicle.height_mm,
                vehicle.max_weight_g,
                vehicle.cost_per_km,
                vehicle.fixed_cost,
            )
            groups.setdefault(signature, []).append(vehicle_index)
        for indexes in groups.values():
            for left, right in zip(indexes, indexes[1:]):
                model.Add(used[left] >= used[right])

    if hint:
        # Complete hints: every variable gets a value, so CP-SAT can reconstruct
        # the incumbent exactly instead of repairing a partial assignment.
        hint = _canonical_hint(hint, vehicles)
        service = [0] + [node.service_time_s for node in nodes]
        for vehicle_index, vehicle in enumerate(vehicles):
            sequence = hint.get(vehicle.id) or []
            if any(node_id not in node_index for node_id in sequence):
                sequence = []
            indexed = [node_index[node_id] for node_id in sequence]
            arc_keys = [(0, indexed[0]), *zip(indexed, indexed[1:]), (indexed[-1], 0)] if indexed else []
            if any((vehicle_index, origin, dest) not in arcs for origin, dest in arc_keys):
                indexed = []
                arc_keys = []
            served = set(indexed)
            on_route = set(arc_keys)
            model.AddHint(used[vehicle_index], 1 if indexed else 0)
            for customer in customer_indexes:
                if customer in served:
                    continue
                model.AddHint(y[vehicle_index, customer], 0)
                model.AddHint(late[vehicle_index, customer], 0)
                if not use_circuit:
                    model.AddHint(position[vehicle_index, customer], 0)
                else:
                    model.AddHint(skip_node[vehicle_index, customer], 1)
                model.AddHint(arrival[vehicle_index, customer], nodes[customer - 1].window_open_s)
            for origin, dest in possible:
                model.AddHint(arcs[vehicle_index, origin, dest], 1 if (origin, dest) in on_route else 0)
            if not indexed:
                model.AddHint(arrival[vehicle_index, 0], dataset.depot.window_open_s)
                model.AddHint(tend[vehicle_index], dataset.depot.window_open_s)
                continue
            clock = dataset.depot.window_open_s
            model.AddHint(arrival[vehicle_index, 0], min(horizon, clock))
            previous = 0
            rank = 0
            for current in indexed:
                rank += 1
                model.AddHint(y[vehicle_index, current], 1)
                if not use_circuit:
                    model.AddHint(position[vehicle_index, current], rank)
                else:
                    model.AddHint(skip_node[vehicle_index, current], 0)
                arc = matrix.get(locations[previous], locations[current])
                clock += service[previous] + arc.travel_time_s
                node = nodes[current - 1]
                if clock < node.window_open_s:
                    clock = node.window_open_s
                if clock > horizon:
                    break
                model.AddHint(arrival[vehicle_index, current], clock)
                if node.soft_deadline_s is None:
                    model.AddHint(late[vehicle_index, current], 0)
                else:
                    model.AddHint(late[vehicle_index, current], min(horizon, max(0, clock - node.soft_deadline_s)))
                previous = current
            else:
                back = matrix.get(locations[previous], locations[0])
                clock += service[previous] + back.travel_time_s
            model.AddHint(tend[vehicle_index], min(horizon, clock))

    if objective_terms:
        model.Minimize(sum(objective_terms))
    else:
        model.Minimize(0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(time_limit_s, 0.1)
    solver.parameters.random_seed = seed
    solver.parameters.num_search_workers = max(1, search_workers)
    solver.parameters.use_optimization_hints = True
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        return MasterOutcome(status="infeasible")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return MasterOutcome(status="unknown", bound_j=solver.BestObjectiveBound() / SCALE)
    assignment: dict[str, list[str]] = {}
    for vehicle_index, vehicle in enumerate(vehicles):
        if solver.Value(used[vehicle_index]) == 0:
            continue
        sequence: list[str] = []
        current = 0
        seen: set[int] = set()
        while True:
            nxt = None
            for dest in all_indexes:
                key = (vehicle_index, current, dest)
                if dest != current and key in arcs and solver.Value(arcs[key]) == 1:
                    nxt = dest
                    break
            if nxt is None or nxt == 0 or nxt in seen:
                break
            seen.add(nxt)
            sequence.append(nodes[nxt - 1].node_id)
            current = nxt
        if sequence:
            assignment[vehicle.id] = sequence
    return MasterOutcome(
        status="optimal" if status == cp_model.OPTIMAL else "feasible",
        assignment=assignment,
        bound_j=solver.BestObjectiveBound() / SCALE,
        objective_j=solver.ObjectiveValue() / SCALE,
    )

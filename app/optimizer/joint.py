"""Joint search: exact routes, 3D loading cuts, optimality gap."""

from __future__ import annotations

import threading
import time
from itertools import combinations, permutations, product

from app.domain.errors import CargoExceedsVehicleError, MissingTravelArcError
from app.domain.matrix import TravelMatrix
from app.domain.models import (
    CustomerNode,
    Dataset,
    JointResult,
    PackingUnit,
    Placement,
    RoutePlan,
    Vehicle,
    VehicleStat,
    Weights,
)
from app.domain.units import assert_units_fit_fleet, build_units, group_units_by_store, make_node
from app.optimizer.adaptive import adaptive_improve
from app.optimizer.initial_agent import _evaluate, _joint_insertion
from app.optimizer.instructions import build_instructions
from app.optimizer.master import ArcCut, AssignmentCut, SetCut, sequence_arcs, solve_master
from app.optimizer.packing import (
    can_pack_nodes,
    clear_pack_cache,
    later_blocks_earlier,
    minimal_infeasible_nodes,
    pack_stops,
    stops_from_nodes,
)
from app.optimizer.routing import cheapest_insertion, clarke_wright, materialize_routes
from app.optimizer.scoring import build_scales, fast_lower_bound, on_time_rate, score_routes, split_order_count
from app.optimizer.shape import banks_for, shapes_from_nodes

# CP-SAT scores are integers / 1e6; packed scores are floats. Treat a bound
# this close to the incumbent as closed, so scale rounding cannot reopen a gap.
BOUND_TOL = 1e-5


def _failed(
    code: str,
    message: str,
    store_ids: list[str] | None = None,
    unit_ids: list[str] | None = None,
) -> JointResult:
    return JointResult(
        status="failed",
        error_code=code,
        error_message=message,
        infeasible_stores=list(store_ids or []),
        failed_unit_ids=list(unit_ids or []),
        gap=None,
    )


def _pack_assignment(
    assignment: dict[str, list[CustomerNode]],
    dataset: Dataset,
    units_by_id: dict[str, PackingUnit],
) -> tuple[list[Placement], str | None]:
    vehicles = {vehicle.id: vehicle for vehicle in dataset.vehicles}
    placements: list[Placement] = []
    for vehicle_id, nodes in assignment.items():
        if not nodes:
            continue
        result = pack_stops(vehicles[vehicle_id], stops_from_nodes(nodes, units_by_id), dataset.loading_rule)
        if not result.ok:
            return placements, vehicle_id
        placements.extend(result.placements)
    return placements, None


def _repair(
    assignment: dict[str, list[CustomerNode]],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
) -> dict[str, list[CustomerNode]] | None:
    vehicles = {vehicle.id: vehicle for vehicle in dataset.vehicles}
    current = {vehicle_id: list(nodes) for vehicle_id, nodes in assignment.items()}
    for vehicle in dataset.vehicles:
        current.setdefault(vehicle.id, [])
    limit = max(len(units_by_id), 1) * 2
    for _ in range(limit):
        failed_id = None
        for vehicle in dataset.vehicles:
            nodes = current[vehicle.id]
            if nodes and not pack_stops(vehicle, stops_from_nodes(nodes, units_by_id), dataset.loading_rule).ok:
                failed_id = vehicle.id
                break
        if failed_id is None:
            return current
        sequence = current[failed_id]
        moving = max(sequence, key=lambda node: (node.volume_mm3, node.node_id))
        current[failed_id] = [node for node in sequence if node.node_id != moving.node_id]
        best: tuple[float, str] | None = None
        for vehicle in dataset.vehicles:
            if vehicle.id == failed_id:
                continue
            trial = current[vehicle.id] + [moving]
            if sum(node.weight_g for node in trial) > vehicle.max_weight_g:
                continue
            if sum(node.volume_mm3 for node in trial) > vehicle.volume_mm3:
                continue
            if materialize_routes({vehicle.id: trial}, dataset, matrix) is None:
                continue
            if not pack_stops(vehicle, stops_from_nodes(trial, units_by_id), dataset.loading_rule).ok:
                continue
            plan = materialize_routes({vehicle.id: trial}, dataset, matrix)
            assert plan is not None
            cost = plan[0].travel_cost + plan[0].fixed_cost
            candidate = (cost, vehicle.id)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return None
        current[best[1]].append(moving)
    return None


def _bridge_groups(nodes: list[CustomerNode], dataset: Dataset) -> tuple[list[CustomerNode], list[CustomerNode]] | None:
    """Split a large road-network day across the river. Small Euclidean cases stay intact."""
    if len(nodes) <= 12:
        return None
    banks = banks_for(nodes, dataset)
    west = [node for node in nodes if banks.get(node.node_id) == "west"]
    east = [node for node in nodes if banks.get(node.node_id) == "east"]
    if len(west) < 3 or len(east) < 3:
        return None
    return west, east


def _allocate_bank_vehicles(
    dataset: Dataset,
    west: list[CustomerNode],
    east: list[CustomerNode],
) -> tuple[set[str], set[str]]:
    vehicles = sorted(dataset.vehicles, key=lambda vehicle: (-vehicle.volume_mm3, -vehicle.max_weight_g, vehicle.id))
    west_need = (sum(node.volume_mm3 for node in west), sum(node.weight_g for node in west))
    east_need = (sum(node.volume_mm3 for node in east), sum(node.weight_g for node in east))
    heavier_east = east_need >= west_need
    heavier_need = east_need if heavier_east else west_need
    heavier_ids: set[str] = set()
    volume = weight = 0
    for vehicle in vehicles:
        if len(heavier_ids) >= len(vehicles) - 1:
            break
        if heavier_ids and volume >= heavier_need[0] and weight >= heavier_need[1]:
            break
        heavier_ids.add(vehicle.id)
        volume += vehicle.volume_mm3
        weight += vehicle.max_weight_g
    if not heavier_ids:
        heavier_ids.add(vehicles[0].id)
    lighter_ids = {vehicle.id for vehicle in dataset.vehicles if vehicle.id not in heavier_ids}
    if not lighter_ids:
        smallest = min(
            (vehicle for vehicle in dataset.vehicles if vehicle.id in heavier_ids),
            key=lambda vehicle: (vehicle.volume_mm3, vehicle.id),
        )
        heavier_ids.remove(smallest.id)
        lighter_ids.add(smallest.id)
    if heavier_east:
        return lighter_ids, heavier_ids
    return heavier_ids, lighter_ids


def _construct_subset(
    nodes: list[CustomerNode],
    dataset: Dataset,
    matrix: TravelMatrix,
    shapes,
    banks,
    vehicle_ids: set[str],
) -> dict[str, list[CustomerNode]] | None:
    inserted = cheapest_insertion(nodes, dataset, matrix, shapes=shapes, banks=banks, vehicle_ids=vehicle_ids)
    if inserted is not None:
        return inserted
    return clarke_wright(nodes, dataset, matrix, shapes=shapes, banks=banks, vehicle_ids=vehicle_ids)


def _bridge_seed(
    nodes: list[CustomerNode],
    dataset: Dataset,
    matrix: TravelMatrix,
    shapes,
    banks,
) -> dict[str, list[CustomerNode]] | None:
    groups = _bridge_groups(nodes, dataset)
    if groups is None:
        return None
    west, east = groups
    west_ids, east_ids = _allocate_bank_vehicles(dataset, west, east)
    west_assignment = _construct_subset(west, dataset, matrix, shapes, banks, west_ids)
    east_assignment = _construct_subset(east, dataset, matrix, shapes, banks, east_ids)
    if west_assignment is None or east_assignment is None:
        return None
    merged = {vehicle.id: [] for vehicle in dataset.vehicles}
    for assignment in (west_assignment, east_assignment):
        for vehicle_id, sequence in assignment.items():
            if not sequence:
                continue
            if merged[vehicle_id]:
                return None
            merged[vehicle_id] = sequence
    covered = {node.node_id for sequence in merged.values() for node in sequence}
    if covered != {node.node_id for node in nodes}:
        return None
    return merged


def _best_construction(
    nodes: list[CustomerNode],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
) -> dict[str, list[CustomerNode]] | None:
    shapes = shapes_from_nodes(nodes, units_by_id)
    banks = banks_for(nodes, dataset) if len(nodes) > 12 else None
    candidates = []
    inserted = cheapest_insertion(nodes, dataset, matrix, shapes=shapes, banks=banks)
    if inserted is not None:
        candidates.append(inserted)
    saved = clarke_wright(nodes, dataset, matrix, shapes=shapes, banks=banks)
    if saved is not None:
        candidates.append(saved)
    split = _bridge_seed(nodes, dataset, matrix, shapes, banks)
    if split is not None:
        candidates.append(split)
    best_assignment = None
    best_score = None
    for candidate in candidates:
        repaired = _repair(candidate, dataset, matrix, units_by_id)
        if repaired is None:
            continue
        plans = materialize_routes(repaired, dataset, matrix)
        placements, failed = _pack_assignment(repaired, dataset, units_by_id)
        if plans is None or failed is not None:
            continue
        score = score_routes(plans, placements, 0, weights, scales, include_extra=True)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = repaired
    return best_assignment


def _infeasible_vehicle_ids(
    subset: list[CustomerNode],
    groups: dict[tuple, list[Vehicle]],
    rule,
    units_by_id: dict[str, PackingUnit],
    skip: set[str] | None = None,
) -> list[str]:
    """Vehicle ids whose cargo bay cannot hold this subset.

    Weight and volume overflows are left to the master. skip is a set of
    vehicle ids already forbidden by a subset cut (a pair inside a triple).
    """
    blocked: list[str] = []
    skipped = skip or set()
    weight = sum(node.weight_g for node in subset)
    volume = sum(node.volume_mm3 for node in subset)
    for group in groups.values():
        representative = group[0]
        if all(vehicle.id in skipped for vehicle in group):
            continue
        if weight > representative.max_weight_g or volume > representative.volume_mm3:
            continue
        if not can_pack_nodes(subset, representative, rule, units_by_id):
            blocked.extend(vehicle.id for vehicle in group if vehicle.id not in skipped)
    return blocked


def _pair_cuts(
    nodes: list[CustomerNode],
    vehicles: list[Vehicle],
    dataset: Dataset,
    units_by_id: dict[str, PackingUnit],
    budget_s: float | None = None,
) -> list[SetCut]:
    """Forbid stop pairs and triples that cannot share a vehicle.

    Runs at any problem size. Identical vehicles share one packing check, subsets
    that already violate weight or volume are left to the master's capacity
    constraints, and a time budget keeps very large instances cheap. Pairs run
    first; leftover budget is spent on triples that are not implied by a pair
    cut. The budget has a wide margin so same-seed runs stay deterministic.
    """
    if budget_s is None:
        budget_s = 10.0 if len(nodes) <= 12 else 5.0
    deadline = time.monotonic() + budget_s
    groups: dict[tuple, list[Vehicle]] = {}
    for vehicle in vehicles:
        signature = (vehicle.length_mm, vehicle.width_mm, vehicle.height_mm, vehicle.max_weight_g)
        groups.setdefault(signature, []).append(vehicle)
    cuts: list[SetCut] = []
    rule = dataset.loading_rule
    for left, right in combinations(nodes, 2):
        if time.monotonic() > deadline:
            break
        blocked = _infeasible_vehicle_ids([left, right], groups, rule, units_by_id)
        if not blocked:
            continue
        vehicle_ids = None if len(blocked) == len(vehicles) else frozenset(blocked)
        cuts.append(SetCut(frozenset({left.node_id, right.node_id}), vehicle_ids))
    pair_index: dict[frozenset[str], frozenset[str] | None] = {
        cut.node_ids: cut.vehicle_ids for cut in cuts if len(cut.node_ids) == 2
    }
    if len(nodes) >= 3:
        for first, second, third in combinations(nodes, 3):
            if time.monotonic() > deadline:
                break
            subset = [first, second, third]
            skipped: set[str] = set()
            implied = False
            for left, right in combinations(subset, 2):
                key = frozenset({left.node_id, right.node_id})
                if key not in pair_index:
                    continue
                vids = pair_index[key]
                if vids is None:
                    implied = True
                    break
                skipped.update(vids)
            if implied or len(skipped) == len(vehicles):
                continue
            blocked = _infeasible_vehicle_ids(subset, groups, rule, units_by_id, skipped)
            if not blocked:
                continue
            vehicle_ids = None if len(blocked) + len(skipped) == len(vehicles) else frozenset(blocked)
            cuts.append(SetCut(frozenset({first.node_id, second.node_id, third.node_id}), vehicle_ids))
    return cuts


def _vehicle_signature(vehicle: Vehicle) -> tuple:
    return (
        vehicle.type_name,
        vehicle.length_mm,
        vehicle.width_mm,
        vehicle.height_mm,
        vehicle.max_weight_g,
        vehicle.cost_per_km,
        vehicle.fixed_cost,
    )


def _partition_key(assignment: dict[str, list[CustomerNode]], dataset: Dataset) -> tuple:
    """Vehicle-identity-free key for a node partition, used to spot repeats."""
    signature_of = {vehicle.id: _vehicle_signature(vehicle) for vehicle in dataset.vehicles}
    entries = [
        (signature_of[vehicle_id], tuple(sorted(node.node_id for node in nodes)))
        for vehicle_id, nodes in assignment.items()
        if nodes
    ]
    return tuple(sorted(entries))


def _assignment_exclusions(
    assignment: dict[str, list[CustomerNode]],
    dataset: Dataset,
    cap: int = 256,
) -> list[AssignmentCut]:
    """Forbid this exact partition, including permutations over identical vehicles."""
    groups: dict[tuple, list[str]] = {}
    for vehicle in dataset.vehicles:
        groups.setdefault(_vehicle_signature(vehicle), []).append(vehicle.id)
    served = {vehicle_id: tuple(node.node_id for node in nodes) for vehicle_id, nodes in assignment.items() if nodes}
    group_ids: list[list[str]] = []
    group_variants: list[list[tuple[tuple[str, ...], ...]]] = []
    for ids in groups.values():
        sets = [served[vehicle_id] for vehicle_id in ids if vehicle_id in served]
        if not sets:
            continue
        padded = sets + [()] * (len(ids) - len(sets))
        variants = sorted(set(permutations(padded)))
        group_ids.append(ids)
        group_variants.append(variants)
    cuts: list[AssignmentCut] = []
    for combo in product(*group_variants):
        pairs: list[tuple[str, str]] = []
        for ids, variant in zip(group_ids, combo):
            for vehicle_id, node_ids in zip(ids, variant):
                pairs.extend((vehicle_id, node_id) for node_id in node_ids)
        cuts.append(AssignmentCut(tuple(pairs)))
        if len(cuts) >= cap:
            break
    return cuts


def _split_node(node: CustomerNode, units_by_id: dict[str, PackingUnit], dataset: Dataset) -> list[CustomerNode] | None:
    units = [units_by_id[unit_id] for unit_id in node.unit_ids]
    if len(units) < 2:
        return None
    groups: dict[str, list[PackingUnit]] = {}
    for unit in units:
        groups.setdefault(unit.order_line_id, []).append(unit)
    largest_line = max(groups, key=lambda line_id: (sum(unit.volume_mm3 for unit in groups[line_id]), line_id))
    if len(groups) == 1:
        ordered = sorted(units, key=lambda unit: (-unit.volume_mm3, unit.unit_id))
        pivot = max(1, len(ordered) // 2)
        parts = [ordered[:pivot], ordered[pivot:]]
    else:
        moved = groups[largest_line]
        stayed = [unit for unit in units if unit.order_line_id != largest_line]
        parts = [stayed, moved]
    created = []
    for index, part in enumerate(parts):
        if not part:
            continue
        created.append(make_node(f"{node.node_id}~{index}", node.store_id, part, dataset))
    return created


def _apply_split(
    nodes: list[CustomerNode],
    target_id: str,
    units_by_id: dict[str, PackingUnit],
    dataset: Dataset,
) -> list[CustomerNode] | None:
    target = next(node for node in nodes if node.node_id == target_id)
    pieces = _split_node(target, units_by_id, dataset)
    if pieces is None:
        return None
    return [node for node in nodes if node.node_id != target_id] + pieces


def _vehicle_stats(dataset: Dataset, routes: list[RoutePlan], placements: list[Placement]) -> list[VehicleStat]:
    used = {route.vehicle_id: route for route in routes}
    volume_by_vehicle: dict[str, int] = {}
    for placement in placements:
        volume_by_vehicle[placement.vehicle_id] = volume_by_vehicle.get(placement.vehicle_id, 0) + (
            placement.dx * placement.dy * placement.dz
        )
    stats: list[VehicleStat] = []
    for vehicle in dataset.vehicles:
        route = used.get(vehicle.id)
        if route is None:
            stats.append(
                VehicleStat(vehicle_id=vehicle.id, used=False, volume_ratio=0.0, weight_ratio=0.0, distance_m=0)
            )
            continue
        stats.append(
            VehicleStat(
                vehicle_id=vehicle.id,
                used=True,
                volume_ratio=volume_by_vehicle.get(vehicle.id, 0) / vehicle.volume_mm3,
                weight_ratio=route.total_weight_g / vehicle.max_weight_g,
                distance_m=route.distance_m,
            )
        )
    return stats


def _reshuffle_count(routes: list[RoutePlan], placements: list[Placement]) -> int:
    """Count boxes a later delivery leaves blocking an earlier stop.

    Depth overlap between stops is not a reshuffle when the earlier stop rides
    on top of the later one. Only true blocking counts: a later box on top of
    an earlier box, or between it and the door.
    """
    by_unit = {placement.unit_id: placement for placement in placements}
    count = 0
    for route in routes:
        stops = [
            [by_unit[unit_id] for unit_id in stop.unit_ids if unit_id in by_unit]
            for stop in route.stops
        ]
        for earlier_index, earlier_boxes in enumerate(stops):
            for later_boxes in stops[earlier_index + 1 :]:
                for earlier in earlier_boxes:
                    for later in later_boxes:
                        if later_blocks_earlier(later, earlier):
                            count += 1
    return count


def _finish(
    routes: list[RoutePlan],
    placements: list[Placement],
    dataset: Dataset,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
    proven: bool,
    lower_bound: float | None,
    rounds: int,
    iteration_limit_reached: bool,
) -> JointResult:
    unit_orders = {unit.unit_id: unit.order_id for unit in units_by_id.values()}
    split_count = split_order_count(routes, unit_orders)
    upper = score_routes(routes, placements, split_count, weights, scales, include_extra=True)
    if proven:
        lower_bound = upper
        gap = 0.0
    elif lower_bound is None:
        gap = 1.0
    elif upper > 1e-9:
        gap = max(0.0, (upper - lower_bound) / upper)
    else:
        gap = 0.0
    return JointResult(
        status="completed",
        proven_optimal=proven,
        lower_bound=lower_bound,
        upper_bound=upper,
        gap=gap,
        bound_established=bool(proven or (lower_bound is not None and lower_bound > 1e-6)),
        score=upper,
        split_order_count=split_count,
        on_time_rate=on_time_rate(routes),
        iteration_limit_reached=iteration_limit_reached,
        rounds=rounds,
        reshuffle_count=_reshuffle_count(routes, placements),
        routes=routes,
        placements=sorted(placements, key=lambda item: item.unit_id),
        steps=build_instructions(routes, placements),
        vehicle_stats=_vehicle_stats(dataset, routes, placements),
    )


def solve_joint(
    dataset: Dataset,
    weights: Weights | None = None,
    seed: int = 1,
    time_limit_s: float = 180.0,
    fixed_nodes: dict[str, str] | None = None,
    prove_optimal: bool = False,
    search_workers: int = 1,
    model_proposal: bool = False,
) -> JointResult:
    """Solve the joint problem.

    locked_vehicles maps node_id to vehicle_id for orders that must stay put.
    The caller resolves order locks into node locks before calling.

    prove_optimal spends the budget on closing the gap: more master rounds,
    longer slices once the packing cuts stabilize, and enumeration of route
    optima until the master bound reaches the packed incumbent. The default
    large-instance path keeps short master slices and stops after two rounds
    that do not improve the packed score, so leftover time is not spent only
    on the bound. search_workers above one speeds up CP-SAT but gives up
    same-seed reproducibility. Packing feasibility verdicts come from the
    constructive packer, so a proven gap of zero is exact up to packings that
    packer cannot find. Extra stacking implied by floor area and cargo height
    is in the master. Small instances use circuit constraints from the first round.
    """
    weights = weights or Weights()
    clear_pack_cache()
    deadline = time.monotonic() + time_limit_s
    try:
        if not dataset.vehicles:
            return _failed("infeasible_route", "No vehicles are available")
        matrix = TravelMatrix(dataset.arcs)
        location_ids = [dataset.depot.id, *[store.id for store in dataset.stores]]
        matrix.require_complete(location_ids)
        units = build_units(dataset)
        if not units:
            return JointResult(status="completed", proven_optimal=True, lower_bound=0.0, upper_bound=0.0, gap=0.0, score=0.0)
        assert_units_fit_fleet(units, dataset.vehicles)
        units_by_id = {unit.unit_id: unit for unit in units}
        nodes = group_units_by_store(dataset, units)
        scales = build_scales(dataset, matrix)
    except CargoExceedsVehicleError as exc:
        return _failed(exc.code, exc.message, unit_ids=exc.unit_ids)
    except MissingTravelArcError as exc:
        return _failed(exc.code, exc.message)

    node_by_id = {node.node_id: node for node in nodes}
    model_thread: threading.Thread | None = None
    model_ready_by = time.monotonic()
    model_lock = threading.Lock()
    model_box: dict = {"done": not model_proposal, "consumed": False, "assignment": None, "note": None}
    if model_proposal:
        from app.agent.proposal import MODEL_TIMEOUT_S, propose_model_assignment

        model_ready_by = time.monotonic() + MODEL_TIMEOUT_S

        def _fetch_model() -> None:
            try:
                drafted, note = propose_model_assignment(
                    nodes,
                    dataset,
                    matrix,
                    units_by_id,
                    fixed_nodes=fixed_nodes,
                    timeout_s=MODEL_TIMEOUT_S,
                    weights=weights,
                )
            except Exception:
                drafted, note = None, "模型初稿失败"
            with model_lock:
                model_box["assignment"] = drafted
                model_box["note"] = note
                model_box["done"] = True

        model_thread = threading.Thread(target=_fetch_model, name="route-proposal", daemon=True)
        model_thread.start()
    set_cuts = _pair_cuts(nodes, dataset.vehicles, dataset, units_by_id)
    arc_cuts: list[ArcCut] = []
    assignment_cuts: list[AssignmentCut] = []
    excluded_partitions: set[tuple] = set()
    best_routes: list[RoutePlan] | None = None
    best_placements: list[Placement] | None = None
    best_score: float | None = None
    lower_bound = fast_lower_bound(nodes, dataset, matrix, weights, scales, units_by_id)
    master_bound: float | None = None
    proven = False
    rounds = 0
    stable_rounds = 0
    iteration_limit_reached = False

    def _keep(assignment: dict[str, list[CustomerNode]], evaluated) -> bool:
        nonlocal constructed, best_routes, best_placements, best_score
        routes, placements, score = evaluated
        if best_score is not None and score >= best_score - 1e-9:
            return False
        constructed = assignment
        best_routes, best_placements, best_score = routes, placements, score
        return True

    constructed = _best_construction(nodes, dataset, matrix, units_by_id, weights, scales)
    if constructed is not None:
        evaluated = _evaluate(constructed, dataset, matrix, units_by_id, weights, scales)
        if evaluated is not None:
            _keep(constructed, evaluated)

    proposal_note: str | None = None
    proposal_accepted: bool | None = None

    agent_budget = 2.0 if len(nodes) <= 12 else (12.0 if prove_optimal else 8.0)
    agent_end = min(deadline - 1.0, time.monotonic() + agent_budget)
    if agent_end > time.monotonic() + 0.2:
        joint = _joint_insertion(nodes, dataset, matrix, units_by_id, weights, scales, agent_end, fixed_nodes)
        if joint is not None:
            evaluated = _evaluate(joint, dataset, matrix, units_by_id, weights, scales)
            if evaluated is not None:
                _keep(joint, evaluated)
    if constructed is not None and time.monotonic() < agent_end - 0.05:
        improved = adaptive_improve(
            constructed,
            dataset,
            matrix,
            units_by_id,
            weights,
            scales,
            agent_end,
            seed,
            fixed_nodes,
        )
        if improved is not None:
            improved_assignment, improved_routes, improved_placements, improved_score = improved
            _keep(improved_assignment, (improved_routes, improved_placements, improved_score))

    hint = None
    if constructed is not None:
        hint = {vehicle_id: [node.node_id for node in sequence] for vehicle_id, sequence in constructed.items()}
    last_route_objective: float | None = None
    large = len(nodes) > 12
    unknown_streak = 0
    idle_rounds = 0

    def _model_pending() -> bool:
        if model_thread is None:
            return False
        with model_lock:
            return not model_box["done"]

    def _hint_from(assignment: dict[str, list[CustomerNode]]) -> dict[str, list[str]] | None:
        current = set(node_by_id)
        hint_map = {vehicle.id: [] for vehicle in dataset.vehicles}
        covered: list[str] = []
        for vehicle_id, sequence in assignment.items():
            if vehicle_id not in hint_map:
                return None
            ids = [node.node_id for node in sequence]
            if any(node_id not in current for node_id in ids):
                return None
            hint_map[vehicle_id] = ids
            covered.extend(ids)
        if set(covered) != current:
            return None
        return hint_map

    def _absorb_model() -> bool:
        """If the parallel draft is ready and better, make it the incumbent."""
        nonlocal proposal_note, proposal_accepted, hint, idle_rounds, unknown_streak, proven
        if model_thread is None:
            return False
        with model_lock:
            if not model_box["done"] or model_box["consumed"]:
                return False
            model_box["consumed"] = True
            drafted = model_box["assignment"]
            note = model_box["note"]
        proposal_note = note
        if drafted is None:
            proposal_accepted = False
            return False
        evaluated = _evaluate(drafted, dataset, matrix, units_by_id, weights, scales)
        if evaluated is None:
            proposal_accepted = False
            proposal_note = "模型初稿没有通过时间窗或装箱"
            return False
        if not _keep(drafted, evaluated):
            proposal_accepted = False
            proposal_note = f"{note}，但分数不如确定性搜索当前解"
            return False
        proposal_accepted = True
        proposal_note = f"{note}，优于确定性搜索当前解"
        refreshed = _hint_from(drafted)
        if refreshed is not None:
            hint = refreshed
        idle_rounds = 0
        unknown_streak = 0
        if proven and lower_bound is not None and best_score is not None and best_score < lower_bound - BOUND_TOL:
            proven = False
        return True

    def _polish() -> bool:
        nonlocal best_score, best_routes, best_placements, hint
        if not large or hint is None or time.monotonic() > deadline - 6:
            return False
        covered = [node_id for node_ids in hint.values() for node_id in node_ids]
        if len(covered) != len(nodes):
            return False
        assignment = {vehicle.id: [] for vehicle in dataset.vehicles}
        for vehicle_id, node_ids in hint.items():
            if vehicle_id not in assignment:
                continue
            assignment[vehicle_id] = [node_by_id[node_id] for node_id in node_ids if node_id in node_by_id]
        polish_end = min(deadline - 6.0, time.monotonic() + (30.0 if prove_optimal else 12.0))
        if polish_end - time.monotonic() < 0.4:
            return False
        polished = adaptive_improve(
            assignment,
            dataset,
            matrix,
            units_by_id,
            weights,
            scales,
            polish_end,
            seed + 1000 + rounds,
            fixed_nodes,
        )
        if polished is None:
            return False
        improved, routes, placements, score = polished
        if best_score is not None and score >= best_score - 1e-9:
            return False
        best_score = score
        best_routes = routes
        best_placements = placements
        hint = {vehicle_id: [node.node_id for node in sequence] for vehicle_id, sequence in improved.items()}
        return True

    if not large:
        max_rounds = 40 if prove_optimal else 20
        master_slice = None
    elif prove_optimal:
        max_rounds = 40
        master_slice = 30.0
    else:
        max_rounds = 8
        master_slice = 12.0
    effective_workers = search_workers if search_workers > 1 else (8 if large else 1)
    while time.monotonic() < deadline and rounds < max_rounds:
        _absorb_model()
        rounds += 1
        remaining = deadline - time.monotonic()
        if remaining < 0.2:
            iteration_limit_reached = True
            break
        if prove_optimal and (master_slice is None or stable_rounds >= 1):
            # Prove mode spends leftover time on the bound. Default mode keeps
            # short slices so a stalled incumbent does not eat the whole budget.
            round_limit = remaining
        elif master_slice is None:
            round_limit = remaining
        else:
            first_slice = 20.0 if rounds == 1 else master_slice
            round_limit = min(first_slice, remaining)
        outcome = solve_master(
            nodes,
            dataset,
            matrix,
            weights,
            scales,
            set_cuts,
            arc_cuts,
            time_limit_s=round_limit,
            seed=seed + rounds,
            hint=hint,
            fixed_nodes=fixed_nodes,
            assignment_cuts=assignment_cuts,
            search_workers=effective_workers,
            use_circuit=(not large) or rounds > 1,
        )
        if outcome.bound_j is not None and outcome.bound_j > 1e-6:
            master_bound = outcome.bound_j if master_bound is None else max(master_bound, outcome.bound_j)
            lower_bound = outcome.bound_j if lower_bound is None else max(lower_bound, outcome.bound_j)
        if best_score is not None and master_bound is not None and master_bound >= best_score - BOUND_TOL:
            # Nothing outside the excluded partitions can beat the incumbent.
            proven = True
            break
        if outcome.status == "infeasible":
            if best_routes is None:
                return _failed(
                    "infeasible_route",
                    "No route satisfies the time windows and packing cuts",
                    store_ids=sorted({node.store_id for node in nodes}),
                )
            # Every remaining assignment is excluded or unpackable: the
            # incumbent is the joint optimum (up to packer false negatives).
            proven = True
            break
        if outcome.status == "unknown" or not outcome.assignment:
            if outcome.status == "unknown":
                iteration_limit_reached = time.monotonic() >= deadline - 0.2
            if not outcome.assignment:
                unknown_streak += 1
                polished = _polish()
                if polished:
                    unknown_streak = 0
                    idle_rounds = 0
                if not large or unknown_streak >= (6 if prove_optimal else 2):
                    if large and (_model_pending() or _absorb_model()):
                        continue
                    break
                continue
        unknown_streak = 0
        node_by_id = {node.node_id: node for node in nodes}
        assignment = {
            vehicle_id: [node_by_id[node_id] for node_id in sequence]
            for vehicle_id, sequence in outcome.assignment.items()
        }
        routes = materialize_routes(assignment, dataset, matrix)
        if routes is None:
            stable_rounds = 0
            for vehicle_id, sequence in assignment.items():
                arc_cuts.append(ArcCut(sequence_arcs([node.node_id for node in sequence]), frozenset({vehicle_id})))
            continue
        placements, failed_vehicle = _pack_assignment(assignment, dataset, units_by_id)
        if failed_vehicle is None and routes is not None:
            unit_orders = {unit.unit_id: unit.order_id for unit in units}
            full = score_routes(routes, placements, split_order_count(routes, unit_orders), weights, scales, True)
            improved = best_score is None or full < best_score - 1e-9
            if improved:
                best_score = full
                best_routes = routes
                best_placements = placements
                hint = {vehicle_id: [node.node_id for node in sequence] for vehicle_id, sequence in assignment.items()}
                idle_rounds = 0
            if outcome.status == "optimal":
                # Keep floor/volume layers already priced in the master. Dropping
                # them to the route-only score re-opens a fake handling gap.
                if outcome.bound_j is not None and (
                    best_score is None or outcome.bound_j <= best_score + BOUND_TOL
                ):
                    lower_bound = max(lower_bound, outcome.bound_j)
                    master_bound = (
                        outcome.bound_j if master_bound is None else max(master_bound, outcome.bound_j)
                    )
                if master_bound is not None and best_score is not None and master_bound >= best_score - BOUND_TOL:
                    proven = True
                    break
                # Packed extras still sit above the master's layer estimate:
                # forbid this exact partition and enumerate the next route
                # optimum until the bound reaches the packed incumbent.
                key = _partition_key(assignment, dataset)
                if key in excluded_partitions:
                    iteration_limit_reached = True
                    break
                excluded_partitions.add(key)
                assignment_cuts.extend(_assignment_exclusions(assignment, dataset))
                stable_rounds = 0
                continue
            stable_rounds += 1
            stalled = (
                last_route_objective is not None
                and outcome.objective_j is not None
                and outcome.objective_j >= last_route_objective - 1e-9
            )
            last_route_objective = outcome.objective_j
            allow_polish = prove_optimal or improved or idle_rounds == 0
            if allow_polish and _polish():
                idle_rounds = 0
            elif not improved:
                idle_rounds += 1
            if stalled and not large:
                break
            if not prove_optimal and best_score is not None and idle_rounds >= (2 if large else 1):
                if _absorb_model() or _model_pending():
                    continue
                break
            continue
        vehicle = next(item for item in dataset.vehicles if item.id == failed_vehicle)
        sequence = assignment[failed_vehicle]
        stable_rounds = 0
        if len(sequence) <= 4 and can_pack_nodes(sequence, vehicle, dataset.loading_rule, units_by_id):
            arc_cuts.append(ArcCut(sequence_arcs([node.node_id for node in sequence]), frozenset({failed_vehicle})))
            continue
        if len(sequence) <= 4:
            subset = minimal_infeasible_nodes(sequence, vehicle, dataset.loading_rule, units_by_id)
        else:
            subset = sequence
        if len(subset) <= 1:
            target = subset[0] if subset else sequence[0]
            split_nodes = _apply_split(nodes, target.node_id, units_by_id, dataset)
            if split_nodes is None:
                return _failed(
                    "packing_failed",
                    "A package set cannot be split any further",
                    unit_ids=target.unit_ids,
                    store_ids=[target.store_id],
                )
            nodes = split_nodes
            node_by_id = {node.node_id: node for node in nodes}
            set_cuts = [cut for cut in set_cuts if cut.node_ids <= set(node_by_id)]
            hint = None
            continue
        new_cut = SetCut(frozenset(node.node_id for node in subset))
        if new_cut in set_cuts:
            iteration_limit_reached = True
            break
        set_cuts.append(new_cut)
        # The incumbent still packs, so it cannot contain the forbidden subset:
        # keep it as the hint instead of restarting the master blind.
    else:
        if rounds >= max_rounds:
            iteration_limit_reached = True

    if model_thread is not None and _model_pending():
        # The call itself stops at model_ready_by. Packing the draft takes a
        # few more seconds, so wait past the clock before reading the verdict.
        model_thread.join(timeout=max(0.0, model_ready_by - time.monotonic()) + 20.0)
    _absorb_model()

    if best_routes is None or best_placements is None:
        return _failed(
            "packing_failed",
            "No packable route was found before the time limit",
            store_ids=sorted({node.store_id for node in nodes}),
        )
    finished = _finish(
        best_routes,
        best_placements,
        dataset,
        units_by_id,
        weights,
        scales,
        proven,
        lower_bound,
        rounds,
        iteration_limit_reached,
    )
    if proposal_note is None:
        return finished
    return finished.model_copy(
        update={"model_proposal_accepted": proposal_accepted, "model_proposal_note": proposal_note}
    )


def replan(
    dataset: Dataset,
    previous: JointResult,
    locked_order_ids: list[str],
    weights: Weights | None = None,
    seed: int = 1,
    time_limit_s: float = 180.0,
    locked_vehicle_ids: list[str] | None = None,
    prove_optimal: bool = False,
    search_workers: int = 1,
    model_proposal: bool = False,
) -> JointResult:
    """Keep locked vehicles unchanged and keep locked orders on their previous vehicles."""
    weights = weights or Weights()
    if previous.status != "completed" or (not locked_order_ids and not locked_vehicle_ids):
        return solve_joint(
            dataset,
            weights=weights,
            seed=seed,
            time_limit_s=time_limit_s,
            prove_optimal=prove_optimal,
            search_workers=search_workers,
            model_proposal=model_proposal,
        )
    frozen_ids = set(locked_vehicle_ids or [])
    frozen_routes = [route for route in previous.routes if route.vehicle_id in frozen_ids]
    frozen_vehicles = {route.vehicle_id for route in frozen_routes}
    frozen_stores = {stop.store_id for route in frozen_routes for stop in route.stops}
    store_vehicle = {
        stop.store_id: route.vehicle_id for route in previous.routes for stop in route.stops
    }
    locked = set(locked_order_ids)
    fixed = {
        f"store:{order.store_id}": store_vehicle[order.store_id]
        for order in dataset.orders
        if order.id in locked and order.store_id in store_vehicle and order.store_id not in frozen_stores
    }
    reduced = dataset.model_copy(
        update={
            "stores": [store for store in dataset.stores if store.id not in frozen_stores],
            "orders": [order for order in dataset.orders if order.store_id not in frozen_stores],
            "vehicles": [vehicle for vehicle in dataset.vehicles if vehicle.id not in frozen_vehicles],
        }
    )
    if not reduced.orders:
        units_by_id = {unit.unit_id: unit for unit in build_units(dataset)}
        matrix = TravelMatrix(dataset.arcs)
        frozen_placements = [item for item in previous.placements if item.vehicle_id in frozen_vehicles]
        return _finish(
            frozen_routes,
            frozen_placements,
            dataset,
            units_by_id,
            weights,
            build_scales(dataset, matrix),
            proven=False,
            lower_bound=None,
            rounds=0,
            iteration_limit_reached=False,
        )
    partial = solve_joint(
        reduced,
        weights=weights,
        seed=seed,
        time_limit_s=time_limit_s,
        fixed_nodes=fixed or None,
        prove_optimal=prove_optimal,
        search_workers=search_workers,
        model_proposal=model_proposal,
    )
    if partial.status != "completed" or not frozen_routes:
        return partial
    units_by_id = {unit.unit_id: unit for unit in build_units(dataset)}
    matrix = TravelMatrix(dataset.arcs)
    frozen_placements = [item for item in previous.placements if item.vehicle_id in frozen_vehicles]
    return _finish(
        frozen_routes + partial.routes,
        frozen_placements + partial.placements,
        dataset,
        units_by_id,
        weights,
        build_scales(dataset, matrix),
        proven=False,
        lower_bound=None,
        rounds=partial.rounds,
        iteration_limit_reached=partial.iteration_limit_reached,
    )

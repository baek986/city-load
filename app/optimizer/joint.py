"""Joint search: exact routes, 3D loading cuts, optimality gap."""

from __future__ import annotations

import time
from itertools import combinations

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
from app.optimizer.instructions import build_instructions
from app.optimizer.master import ArcCut, SetCut, sequence_arcs, solve_master
from app.optimizer.packing import (
    can_pack_nodes,
    clear_pack_cache,
    later_blocks_earlier,
    minimal_infeasible_nodes,
    pack_stops,
    stops_from_nodes,
)
from app.optimizer.routing import cheapest_insertion, clarke_wright, materialize_routes
from app.optimizer.scoring import build_scales, on_time_rate, score_routes, split_order_count


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


def _best_construction(
    nodes: list[CustomerNode],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
) -> dict[str, list[CustomerNode]] | None:
    candidates = []
    inserted = cheapest_insertion(nodes, dataset, matrix)
    if inserted is not None:
        candidates.append(inserted)
    saved = clarke_wright(nodes, dataset, matrix)
    if saved is not None:
        candidates.append(saved)
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


def _pair_cuts(
    nodes: list[CustomerNode],
    vehicles: list[Vehicle],
    dataset: Dataset,
    units_by_id: dict[str, PackingUnit],
) -> list[SetCut]:
    if len(nodes) > 10:
        return []
    cuts: list[SetCut] = []
    rule = dataset.loading_rule
    for left, right in combinations(nodes, 2):
        blocked = [
            vehicle.id
            for vehicle in vehicles
            if not can_pack_nodes([left, right], vehicle, rule, units_by_id)
        ]
        if not blocked:
            continue
        vehicle_ids = None if len(blocked) == len(vehicles) else frozenset(blocked)
        cuts.append(SetCut(frozenset({left.node_id, right.node_id}), vehicle_ids))
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
) -> JointResult:
    """Solve the joint problem.

    locked_vehicles maps node_id to vehicle_id for orders that must stay put.
    The caller resolves order locks into node locks before calling.
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
    set_cuts = _pair_cuts(nodes, dataset.vehicles, dataset, units_by_id)
    arc_cuts: list[ArcCut] = []
    best_routes: list[RoutePlan] | None = None
    best_placements: list[Placement] | None = None
    best_score: float | None = None
    lower_bound: float | None = None
    proven = False
    rounds = 0
    iteration_limit_reached = False

    constructed = _best_construction(nodes, dataset, matrix, units_by_id, weights, scales)
    if constructed is not None:
        routes = materialize_routes(constructed, dataset, matrix)
        placements, failed = _pack_assignment(constructed, dataset, units_by_id)
        if routes is not None and failed is None:
            best_routes = routes
            best_placements = placements
            best_score = score_routes(
                routes,
                placements,
                split_order_count(routes, {unit.unit_id: unit.order_id for unit in units}),
                weights,
                scales,
            )

    hint = None
    if constructed is not None:
        hint = {vehicle_id: [node.node_id for node in sequence] for vehicle_id, sequence in constructed.items()}
    last_route_objective: float | None = None

    if len(nodes) <= 12:
        max_rounds = 20
        master_slice = None
    elif best_routes is not None:
        max_rounds = 1
        master_slice = 12.0
    else:
        max_rounds = 2
        master_slice = 20.0
    while time.monotonic() < deadline and rounds < max_rounds:
        rounds += 1
        remaining = deadline - time.monotonic()
        if remaining < 0.2:
            iteration_limit_reached = True
            break
        outcome = solve_master(
            nodes,
            dataset,
            matrix,
            weights,
            scales,
            set_cuts,
            arc_cuts,
            time_limit_s=remaining if master_slice is None else min(master_slice, remaining),
            seed=seed + rounds,
            hint=hint,
            fixed_nodes=fixed_nodes,
        )
        if outcome.bound_j is not None:
            lower_bound = outcome.bound_j if lower_bound is None else max(lower_bound, outcome.bound_j)
        if outcome.status == "infeasible":
            if best_routes is None:
                return _failed(
                    "infeasible_route",
                    "No route satisfies the time windows and packing cuts",
                    store_ids=sorted({node.store_id for node in nodes}),
                )
            break
        if outcome.status == "unknown" or not outcome.assignment:
            if outcome.status == "unknown":
                iteration_limit_reached = time.monotonic() >= deadline - 0.2
            if not outcome.assignment:
                break
        node_by_id = {node.node_id: node for node in nodes}
        assignment = {
            vehicle_id: [node_by_id[node_id] for node_id in sequence]
            for vehicle_id, sequence in outcome.assignment.items()
        }
        routes = materialize_routes(assignment, dataset, matrix)
        if routes is None:
            for vehicle_id, sequence in assignment.items():
                arc_cuts.append(ArcCut(sequence_arcs([node.node_id for node in sequence]), frozenset({vehicle_id})))
            continue
        placements, failed_vehicle = _pack_assignment(assignment, dataset, units_by_id)
        if failed_vehicle is None and routes is not None:
            unit_orders = {unit.unit_id: unit.order_id for unit in units}
            full = score_routes(routes, placements, split_order_count(routes, unit_orders), weights, scales, True)
            route_only = score_routes(routes, placements, split_order_count(routes, unit_orders), weights, scales, False)
            if best_score is None or full < best_score - 1e-9:
                best_score = full
                best_routes = routes
                best_placements = placements
                hint = {vehicle_id: [node.node_id for node in sequence] for vehicle_id, sequence in assignment.items()}
            if outcome.status == "optimal":
                lower_bound = route_only
                if abs(full - route_only) <= 1e-6 and (best_score is None or abs(best_score - route_only) <= 1e-6):
                    proven = True
                    best_routes = routes
                    best_placements = placements
                    best_score = full
                break
            if (
                last_route_objective is not None
                and outcome.objective_j is not None
                and outcome.objective_j >= last_route_objective - 1e-9
            ):
                break
            last_route_objective = outcome.objective_j
            continue
        vehicle = next(item for item in dataset.vehicles if item.id == failed_vehicle)
        sequence = assignment[failed_vehicle]
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
        hint = None
    else:
        if rounds >= max_rounds:
            iteration_limit_reached = True

    if best_routes is None or best_placements is None:
        return _failed(
            "packing_failed",
            "No packable route was found before the time limit",
            store_ids=sorted({node.store_id for node in nodes}),
        )
    return _finish(
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


def replan(
    dataset: Dataset,
    previous: JointResult,
    locked_order_ids: list[str],
    weights: Weights | None = None,
    seed: int = 1,
    time_limit_s: float = 180.0,
    locked_vehicle_ids: list[str] | None = None,
) -> JointResult:
    """Keep locked vehicles unchanged and keep locked orders on their previous vehicles."""
    weights = weights or Weights()
    if previous.status != "completed" or (not locked_order_ids and not locked_vehicle_ids):
        return solve_joint(dataset, weights=weights, seed=seed, time_limit_s=time_limit_s)
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

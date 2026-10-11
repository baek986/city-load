"""Tool-using start for the joint search.

Route-only construction ignores the loaded geometry. This agent places each
store only where the vehicle still packs, scores the full objective, and then
hands that assignment to the existing iteration as the incumbent and the hint.
"""

from __future__ import annotations

import time

from app.domain.matrix import TravelMatrix
from app.domain.models import CustomerNode, Dataset, PackingUnit, Placement, RoutePlan, Weights
from app.optimizer.adaptive import adaptive_improve
from app.optimizer.packing import pack_stops, stops_from_nodes
from app.optimizer.routing import materialize_routes
from app.optimizer.scoring import score_routes, split_order_count
from app.optimizer.shape import banks_for, route_bank


def _blank(dataset: Dataset) -> dict[str, list[CustomerNode]]:
    return {vehicle.id: [] for vehicle in dataset.vehicles}


def _copy(assignment: dict[str, list[CustomerNode]]) -> dict[str, list[CustomerNode]]:
    return {vehicle_id: list(nodes) for vehicle_id, nodes in assignment.items()}


def _evaluate(
    assignment: dict[str, list[CustomerNode]],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
) -> tuple[list[RoutePlan], list[Placement], float] | None:
    routes = materialize_routes(assignment, dataset, matrix)
    if routes is None:
        return None
    vehicles = {vehicle.id: vehicle for vehicle in dataset.vehicles}
    placements: list[Placement] = []
    for vehicle_id, nodes in assignment.items():
        if not nodes:
            continue
        packed = pack_stops(vehicles[vehicle_id], stops_from_nodes(nodes, units_by_id), dataset.loading_rule)
        if not packed.ok:
            return None
        placements.extend(packed.placements)
    unit_orders = {unit.unit_id: unit.order_id for unit in units_by_id.values()}
    score = score_routes(routes, placements, split_order_count(routes, unit_orders), weights, scales, True)
    return routes, placements, score


def _joint_insertion(
    nodes: list[CustomerNode],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
    deadline: float,
    fixed_nodes: dict[str, str] | None,
) -> dict[str, list[CustomerNode]] | None:
    """Insert stores farthest-first, accepting only positions that pack and lower the joint score."""
    assignment = _blank(dataset)
    locked = fixed_nodes or {}
    banks = banks_for(nodes, dataset) if len(nodes) > 12 else None
    ordered = sorted(
        nodes,
        key=lambda node: (-matrix.get(dataset.depot.id, node.store_id).distance_m, node.node_id),
    )
    for node in ordered:
        if time.monotonic() >= deadline:
            return None
        best: tuple[float, str, int] | None = None
        candidates = list(dataset.vehicles)
        forced = locked.get(node.node_id)
        if forced is not None:
            candidates = [vehicle for vehicle in dataset.vehicles if vehicle.id == forced]
        elif banks is not None:
            node_bank = banks.get(node.node_id)
            candidates.sort(
                key=lambda vehicle: (
                    0 if route_bank(assignment[vehicle.id], banks) in (None, node_bank) else 1,
                    vehicle.id,
                )
            )
        for vehicle in candidates:
            if time.monotonic() >= deadline:
                return None
            current = assignment[vehicle.id]
            for position in range(len(current) + 1):
                trial_nodes = current[:position] + [node] + current[position:]
                if sum(item.weight_g for item in trial_nodes) > vehicle.max_weight_g:
                    continue
                if sum(item.volume_mm3 for item in trial_nodes) > vehicle.volume_mm3:
                    continue
                packed = pack_stops(vehicle, stops_from_nodes(trial_nodes, units_by_id), dataset.loading_rule)
                if not packed.ok:
                    continue
                trial = _copy(assignment)
                trial[vehicle.id] = trial_nodes
                evaluated = _evaluate(trial, dataset, matrix, units_by_id, weights, scales)
                if evaluated is None:
                    continue
                choice = (evaluated[2], vehicle.id, position)
                if best is None or choice < best:
                    best = choice
        if best is None:
            return None
        _score, vehicle_id, position = best
        current = assignment[vehicle_id]
        assignment[vehicle_id] = current[:position] + [node] + current[position:]
        if materialize_routes({vehicle_id: assignment[vehicle_id]}, dataset, matrix) is None:
            return None
    return assignment


def propose_initial(
    nodes: list[CustomerNode],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
    starting: dict[str, list[CustomerNode]] | None,
    deadline: float,
    seed: int,
    fixed_nodes: dict[str, str] | None = None,
) -> tuple[dict[str, list[CustomerNode]], list[RoutePlan], list[Placement], float] | None:
    """Best packed assignment from route-only seeds, joint insertion, and a short improvement."""
    seeds: list[dict[str, list[CustomerNode]]] = []
    if starting is not None:
        seeds.append(starting)
    if time.monotonic() < deadline:
        joint = _joint_insertion(nodes, dataset, matrix, units_by_id, weights, scales, deadline, fixed_nodes)
        if joint is not None:
            seeds.append(joint)
    best: tuple[dict[str, list[CustomerNode]], list[RoutePlan], list[Placement], float] | None = None
    for seed_assignment in seeds:
        if time.monotonic() >= deadline:
            break
        evaluated = _evaluate(seed_assignment, dataset, matrix, units_by_id, weights, scales)
        if evaluated is None:
            continue
        routes, placements, score = evaluated
        if best is None or score < best[3]:
            best = (_copy(seed_assignment), routes, placements, score)
    if best is None or time.monotonic() >= deadline:
        return best
    improved = adaptive_improve(
        best[0],
        dataset,
        matrix,
        units_by_id,
        weights,
        scales,
        deadline,
        seed,
        fixed_nodes,
    )
    if improved is not None and improved[3] < best[3] - 1e-9:
        return improved
    return best

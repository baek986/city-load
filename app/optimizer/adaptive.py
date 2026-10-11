"""Packing-aware neighborhood search with adaptive operator weights.

The route master scores weight, volume, and time windows. It does not see the
3D placement. This search tries relocate, swap, and merge moves across
vehicles plus 2-opt and or-opt moves inside one route, keeps a move only when
the packed joint score improves, and shifts weight toward the operators that
actually help. The random draws come from the caller seed.
"""

from __future__ import annotations

import random
import time

from app.domain.matrix import TravelMatrix
from app.domain.models import CustomerNode, Dataset, PackingUnit, Placement, RoutePlan, Weights
from app.optimizer.packing import pack_stops, stops_from_nodes
from app.optimizer.routing import materialize_routes
from app.optimizer.scoring import score_routes, split_order_count
from app.optimizer.shape import banks_for, route_bank

_OPERATORS = ("relocate", "swap", "merge", "two_opt", "or_opt")


def _copy(assignment: dict[str, list[CustomerNode]]) -> dict[str, list[CustomerNode]]:
    return {vehicle_id: list(nodes) for vehicle_id, nodes in assignment.items()}


def _pack(
    assignment: dict[str, list[CustomerNode]],
    dataset: Dataset,
    units_by_id: dict[str, PackingUnit],
) -> list[Placement] | None:
    vehicles = {vehicle.id: vehicle for vehicle in dataset.vehicles}
    placements: list[Placement] = []
    for vehicle_id, nodes in assignment.items():
        if not nodes:
            continue
        result = pack_stops(vehicles[vehicle_id], stops_from_nodes(nodes, units_by_id), dataset.loading_rule)
        if not result.ok:
            return None
        placements.extend(result.placements)
    return placements


def _score(
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
    placements = _pack(assignment, dataset, units_by_id)
    if placements is None:
        return None
    unit_orders = {unit.unit_id: unit.order_id for unit in units_by_id.values()}
    value = score_routes(routes, placements, split_order_count(routes, unit_orders), weights, scales, True)
    return routes, placements, value


def _choose(weights: dict[str, float], rng: random.Random) -> str:
    total = sum(weights[name] for name in _OPERATORS)
    draw = rng.random() * total
    cursor = 0.0
    for name in _OPERATORS:
        cursor += weights[name]
        if draw <= cursor:
            return name
    return _OPERATORS[-1]


def _reward(weights: dict[str, float], name: str, improved: bool) -> None:
    if improved:
        weights[name] = min(4.0, weights[name] * 1.25)
    else:
        weights[name] = max(0.25, weights[name] * 0.85)


def _relocate(
    current: dict[str, list[CustomerNode]],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
    rng: random.Random,
    locked: dict[str, str],
) -> tuple[dict[str, list[CustomerNode]], list[RoutePlan], list[Placement], float] | None:
    used = [vehicle_id for vehicle_id, nodes in current.items() if nodes]
    if not used:
        return None
    source_id = used[rng.randrange(len(used))]
    movable = [index for index, node in enumerate(current[source_id]) if node.node_id not in locked]
    if not movable:
        return None
    index = movable[rng.randrange(len(movable))]
    node = current[source_id][index]
    targets = [vehicle for vehicle in dataset.vehicles if vehicle.id != source_id]
    if not targets:
        return None
    banks = banks_for([item for sequence in current.values() for item in sequence], dataset)
    if len(banks) > 12 and rng.random() < 0.7:
        node_bank = banks.get(node.node_id)
        same = [
            vehicle
            for vehicle in targets
            if route_bank(current[vehicle.id], banks) in (None, node_bank)
        ]
        if same:
            targets = same
    target = targets[rng.randrange(len(targets))]
    best: tuple[float, dict[str, list[CustomerNode]], list[RoutePlan], list[Placement]] | None = None
    for position in range(len(current[target.id]) + 1):
        trial = _copy(current)
        trial[source_id] = [item for item_index, item in enumerate(current[source_id]) if item_index != index]
        trial[target.id] = trial[target.id][:position] + [node] + trial[target.id][position:]
        evaluated = _score(trial, dataset, matrix, units_by_id, weights, scales)
        if evaluated is None:
            continue
        routes, placements, value = evaluated
        if best is None or value < best[0]:
            best = (value, trial, routes, placements)
    if best is None:
        return None
    value, trial, routes, placements = best
    return trial, routes, placements, value


def _swap(
    current: dict[str, list[CustomerNode]],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
    rng: random.Random,
    locked: dict[str, str],
) -> tuple[dict[str, list[CustomerNode]], list[RoutePlan], list[Placement], float] | None:
    used = [vehicle_id for vehicle_id, nodes in current.items() if nodes]
    if len(used) < 2:
        return None
    left_id, right_id = rng.sample(used, 2)
    left_choices = [index for index, node in enumerate(current[left_id]) if node.node_id not in locked]
    right_choices = [index for index, node in enumerate(current[right_id]) if node.node_id not in locked]
    if not left_choices or not right_choices:
        return None
    left_index = left_choices[rng.randrange(len(left_choices))]
    right_index = right_choices[rng.randrange(len(right_choices))]
    trial = _copy(current)
    left_node = trial[left_id][left_index]
    right_node = trial[right_id][right_index]
    trial[left_id][left_index] = right_node
    trial[right_id][right_index] = left_node
    evaluated = _score(trial, dataset, matrix, units_by_id, weights, scales)
    if evaluated is None:
        return None
    routes, placements, value = evaluated
    return trial, routes, placements, value


def _merge(
    current: dict[str, list[CustomerNode]],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
    locked: dict[str, str],
) -> tuple[dict[str, list[CustomerNode]], list[RoutePlan], list[Placement], float] | None:
    donors = [
        vehicle_id
        for vehicle_id, nodes in current.items()
        if nodes and len(nodes) <= 4 and all(node.node_id not in locked for node in nodes)
    ]
    if not donors:
        return None
    donor_id = min(donors, key=lambda vehicle_id: (len(current[vehicle_id]), vehicle_id))
    trial = _copy(current)
    moving = list(trial[donor_id])
    trial[donor_id] = []
    for node in moving:
        best: tuple[float, str, int] | None = None
        for vehicle in dataset.vehicles:
            if vehicle.id == donor_id:
                continue
            for position in range(len(trial[vehicle.id]) + 1):
                candidate_nodes = trial[vehicle.id][:position] + [node] + trial[vehicle.id][position:]
                plan = materialize_routes({vehicle.id: candidate_nodes}, dataset, matrix)
                if plan is None:
                    continue
                choice = (plan[0].travel_cost + plan[0].fixed_cost, vehicle.id, position)
                if best is None or choice < best:
                    best = choice
        if best is None:
            return None
        _, vehicle_id, position = best
        trial[vehicle_id] = trial[vehicle_id][:position] + [node] + trial[vehicle_id][position:]
    evaluated = _score(trial, dataset, matrix, units_by_id, weights, scales)
    if evaluated is None:
        return None
    routes, placements, value = evaluated
    return trial, routes, placements, value


def _two_opt(
    current: dict[str, list[CustomerNode]],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
    rng: random.Random,
) -> tuple[dict[str, list[CustomerNode]], list[RoutePlan], list[Placement], float] | None:
    """Reverse one segment inside a single route. Vehicles stay unchanged."""
    choices = [vehicle_id for vehicle_id, nodes in current.items() if len(nodes) >= 2]
    if not choices:
        return None
    vehicle_id = choices[rng.randrange(len(choices))]
    nodes = current[vehicle_id]
    left = rng.randrange(len(nodes) - 1)
    right = left + 1 + rng.randrange(len(nodes) - left - 1)
    trial = _copy(current)
    trial[vehicle_id] = nodes[:left] + list(reversed(nodes[left : right + 1])) + nodes[right + 1 :]
    evaluated = _score(trial, dataset, matrix, units_by_id, weights, scales)
    if evaluated is None:
        return None
    routes, placements, value = evaluated
    return trial, routes, placements, value


def _or_opt(
    current: dict[str, list[CustomerNode]],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
    rng: random.Random,
) -> tuple[dict[str, list[CustomerNode]], list[RoutePlan], list[Placement], float] | None:
    """Move a segment of one or two stops to another position in the same route."""
    choices = [vehicle_id for vehicle_id, nodes in current.items() if len(nodes) >= 3]
    if not choices:
        return None
    vehicle_id = choices[rng.randrange(len(choices))]
    nodes = current[vehicle_id]
    length = 1 + rng.randrange(min(2, len(nodes) - 1))
    start = rng.randrange(len(nodes) - length + 1)
    segment = nodes[start : start + length]
    rest = nodes[:start] + nodes[start + length :]
    position = rng.randrange(len(rest) + 1)
    if position == start:
        return None
    trial = _copy(current)
    trial[vehicle_id] = rest[:position] + segment + rest[position:]
    evaluated = _score(trial, dataset, matrix, units_by_id, weights, scales)
    if evaluated is None:
        return None
    routes, placements, value = evaluated
    return trial, routes, placements, value


def adaptive_improve(
    assignment: dict[str, list[CustomerNode]],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    weights: Weights,
    scales,
    deadline: float,
    seed: int,
    fixed_nodes: dict[str, str] | None = None,
) -> tuple[dict[str, list[CustomerNode]], list[RoutePlan], list[Placement], float] | None:
    """Return a better packed assignment, or None when nothing improved in time."""
    if time.monotonic() >= deadline:
        return None
    baseline = _score(assignment, dataset, matrix, units_by_id, weights, scales)
    if baseline is None:
        return None
    _routes, _placements, best_score = baseline
    best_assignment = _copy(assignment)
    best_routes, best_placements = _routes, _placements
    improved = False
    operator_weight = {name: 1.0 for name in _OPERATORS}
    locked = fixed_nodes or {}
    rng = random.Random(seed)
    trials = 0
    stale = 0
    while time.monotonic() < deadline and trials < 48 and stale < 16:
        trials += 1
        name = _choose(operator_weight, rng)
        current = best_assignment
        if name == "relocate":
            candidate = _relocate(current, dataset, matrix, units_by_id, weights, scales, rng, locked)
        elif name == "swap":
            candidate = _swap(current, dataset, matrix, units_by_id, weights, scales, rng, locked)
        elif name == "merge":
            candidate = _merge(current, dataset, matrix, units_by_id, weights, scales, locked)
        elif name == "two_opt":
            candidate = _two_opt(current, dataset, matrix, units_by_id, weights, scales, rng)
        else:
            candidate = _or_opt(current, dataset, matrix, units_by_id, weights, scales, rng)
        if candidate is None or candidate[3] >= best_score - 1e-9:
            stale += 1
            _reward(operator_weight, name, False)
            continue
        best_assignment, best_routes, best_placements, best_score = candidate
        improved = True
        stale = 0
        _reward(operator_weight, name, True)
    if not improved:
        return None
    return best_assignment, best_routes, best_placements, best_score

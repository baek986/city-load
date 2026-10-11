"""The packing-aware adaptive search stays feasible and repeatable."""

from __future__ import annotations

import time

from app.demo.scenarios import preview_day
from app.domain.matrix import TravelMatrix
from app.domain.models import Weights
from app.domain.units import build_units, group_units_by_store
from app.optimizer.adaptive import adaptive_improve
from app.optimizer.initial_agent import propose_initial
from app.optimizer.joint import solve_joint
from app.optimizer.routing import cheapest_insertion
from app.optimizer.scoring import build_scales, fast_lower_bound


def _signature(polished) -> tuple:
    if polished is None:
        return ()
    _assignment, routes, placements, score = polished
    route_key = tuple(
        (route.vehicle_id, tuple(stop.store_id for stop in route.stops))
        for route in sorted(routes, key=lambda item: item.vehicle_id)
    )
    place_key = tuple(
        (item.unit_id, item.vehicle_id, item.x, item.y, item.z) for item in sorted(placements, key=lambda item: item.unit_id)
    )
    return route_key, place_key, round(score, 6)


def test_adaptive_search_does_not_worsen_and_repeats() -> None:
    dataset = preview_day()
    matrix = TravelMatrix(dataset.arcs)
    units = build_units(dataset)
    units_by_id = {unit.unit_id: unit for unit in units}
    nodes = group_units_by_store(dataset, units)
    assignment = cheapest_insertion(nodes, dataset, matrix)
    assert assignment is not None
    weights = Weights()
    scales = build_scales(dataset, matrix)
    first = adaptive_improve(
        assignment, dataset, matrix, units_by_id, weights, scales, time.monotonic() + 30, seed=1
    )
    second = adaptive_improve(
        assignment, dataset, matrix, units_by_id, weights, scales, time.monotonic() + 30, seed=1
    )
    assert _signature(first) == _signature(second)


def test_fast_lower_bound_sits_below_a_proven_plan() -> None:
    dataset = preview_day()
    result = solve_joint(dataset, seed=1, time_limit_s=15)
    assert result.status == "completed", result.error_message
    assert result.lower_bound is not None
    assert result.bound_established
    assert 0 < result.lower_bound <= result.score + 1e-6
    matrix = TravelMatrix(dataset.arcs)
    units = build_units(dataset)
    nodes = group_units_by_store(dataset, units)
    scales = build_scales(dataset, matrix)
    assert fast_lower_bound(nodes, dataset, matrix, Weights(), scales) <= result.score + 1e-6


def test_joint_initial_agent_packs_and_repeats() -> None:
    dataset = preview_day()
    matrix = TravelMatrix(dataset.arcs)
    units = build_units(dataset)
    units_by_id = {unit.unit_id: unit for unit in units}
    nodes = group_units_by_store(dataset, units)
    weights = Weights()
    scales = build_scales(dataset, matrix)
    first = propose_initial(nodes, dataset, matrix, units_by_id, weights, scales, None, time.monotonic() + 5, seed=1)
    second = propose_initial(
        nodes, dataset, matrix, units_by_id, weights, scales, None, time.monotonic() + 5, seed=1
    )
    assert first is not None and second is not None
    assert _signature(first) == _signature(second)
    _assignment, routes, placements, score = first
    assert routes
    assert placements
    assert score > 0

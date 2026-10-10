"""Acceptance checks for the pharmacy day and the published gap."""

from __future__ import annotations

import time
from collections import defaultdict

from pathlib import Path

from app.domain.models import Dataset, Placement
from app.optimizer.joint import solve_joint
from app.optimizer.packing import boxes_overlap


def _problems(dataset: Dataset, result) -> list[str]:
    problems: list[str] = []
    vehicles = {vehicle.id: vehicle for vehicle in dataset.vehicles}
    if result.reshuffle_count != 0:
        problems.append("reshuffle")
    for route in result.routes:
        vehicle = vehicles[route.vehicle_id]
        if route.total_weight_g > vehicle.max_weight_g:
            problems.append(f"weight {route.vehicle_id}")
        for stop in route.stops:
            store = next(item for item in dataset.stores if item.id == stop.store_id)
            if stop.arrival_s < store.window_open_s or stop.arrival_s > store.window_close_s:
                problems.append(f"window {stop.store_id}")
    grouped: dict[str, list[Placement]] = defaultdict(list)
    for placement in result.placements:
        grouped[placement.vehicle_id].append(placement)
        vehicle = vehicles[placement.vehicle_id]
        if (
            placement.x < 0
            or placement.y < 0
            or placement.z < 0
            or placement.x + placement.dx > vehicle.length_mm
            or placement.y + placement.dy > vehicle.width_mm
            or placement.z + placement.dz > vehicle.height_mm
        ):
            problems.append(f"bounds {placement.unit_id}")
    for items in grouped.values():
        for left_index, left in enumerate(items):
            for right in items[left_index + 1 :]:
                if boxes_overlap(left, right):
                    problems.append(f"overlap {left.unit_id} {right.unit_id}")
            if not left.fragile:
                continue
            for right in items:
                if right.unit_id == left.unit_id:
                    continue
                above = right.z >= left.z + left.dz
                overlap_x = min(left.x + left.dx, right.x + right.dx) - max(left.x, right.x)
                overlap_y = min(left.y + left.dy, right.y + right.dy) - max(left.y, right.y)
                if above and overlap_x > 0 and overlap_y > 0:
                    problems.append(f"fragile {left.unit_id}")
    return problems


def _signature(result) -> tuple:
    routes = tuple(
        (route.vehicle_id, tuple(stop.store_id for stop in route.stops))
        for route in sorted(result.routes, key=lambda item: item.vehicle_id)
    )
    placements = tuple(
        (item.unit_id, item.vehicle_id, item.x, item.y, item.z, item.dx, item.dy, item.dz, item.rotation)
        for item in result.placements
    )
    return routes, placements


def test_pharmacy_day_is_feasible_and_reports_gap() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "pharmacy_day.json"
    dataset = Dataset.model_validate_json(path.read_text(encoding="utf-8"))
    assert len(dataset.stores) == 30
    assert sum(len(order.lines) for order in dataset.orders) == 400
    assert len(dataset.vehicles) == 8
    started = time.perf_counter()
    first = solve_joint(dataset, seed=7, time_limit_s=50)
    elapsed = time.perf_counter() - started
    assert elapsed < 180
    assert first.status == "completed", first.error_message
    assert first.gap is not None
    assert first.split_order_count / len(dataset.orders) <= 0.10
    assert _problems(dataset, first) == []
    second = solve_joint(dataset, seed=7, time_limit_s=50)
    assert _signature(first) == _signature(second)
    if first.gap == 0:
        assert first.proven_optimal

"""Scaled objective shared by the route master and the packed incumbent."""

from __future__ import annotations

from collections import defaultdict

from app.domain.matrix import TravelMatrix
from app.domain.models import Dataset, Placement, RoutePlan, Scales, Weights


def build_scales(dataset: Dataset, matrix: TravelMatrix) -> Scales:
    cheapest = min(vehicle.cost_per_km for vehicle in dataset.vehicles)
    cheapest_fixed = min(vehicle.fixed_cost for vehicle in dataset.vehicles)
    cost = cheapest_fixed
    travel = 0
    for store in dataset.stores:
        out_arc = matrix.get(dataset.depot.id, store.id)
        back_arc = matrix.get(store.id, dataset.depot.id)
        cost += (out_arc.distance_m + back_arc.distance_m) / 1000.0 * cheapest
        travel += out_arc.travel_time_s + back_arc.travel_time_s
    return Scales(
        cost=max(cost, 1.0),
        time=max(float(travel), 1.0),
        late=max(len(dataset.stores) * 60.0, 1.0),
        handling=max(float(len(dataset.stores)), 1.0),
        split=float(max(len(dataset.orders), 1)),
    )


def split_order_count(routes: list[RoutePlan], unit_orders: dict[str, str]) -> int:
    vehicles_by_order: dict[str, set[str]] = defaultdict(set)
    for route in routes:
        for stop in route.stops:
            for unit_id in stop.unit_ids:
                order_id = unit_orders.get(unit_id)
                if order_id is not None:
                    vehicles_by_order[order_id].add(route.vehicle_id)
    return sum(1 for vehicle_ids in vehicles_by_order.values() if len(vehicle_ids) > 1)


def extra_handling(placements: list[Placement]) -> int:
    """Layers above the floor, and fragile goods left on the floor beside sturdier goods."""
    by_vehicle: dict[str, list[Placement]] = defaultdict(list)
    for placement in placements:
        by_vehicle[placement.vehicle_id].append(placement)
    extra = 0
    for group in by_vehicle.values():
        extra += max(0, len({item.z for item in group}) - 1)
        by_store: dict[str, list[Placement]] = defaultdict(list)
        for placement in group:
            by_store[placement.store_id].append(placement)
        for items in by_store.values():
            if not any(not item.fragile for item in items):
                continue
            extra += sum(1 for item in items if item.fragile and item.z == 0)
    return extra


def score_routes(
    routes: list[RoutePlan],
    placements: list[Placement],
    split_count: int,
    weights: Weights,
    scales: Scales,
    include_extra: bool = True,
) -> float:
    cost = sum(route.travel_cost + route.fixed_cost for route in routes)
    travel_time = sum(route.travel_time_s for route in routes)
    late = sum(stop.late_minutes for route in routes for stop in route.stops)
    stops = sum(len(route.stops) for route in routes)
    handling = stops + (extra_handling(placements) if include_extra else 0)
    return (
        weights.cost * cost / scales.cost
        + weights.time * travel_time / scales.time
        + weights.late * late / scales.late
        + weights.handling * handling / scales.handling
        + weights.split * split_count / scales.split
    )


def on_time_rate(routes: list[RoutePlan]) -> float:
    stops = [stop for route in routes for stop in route.stops]
    if not stops:
        return 1.0
    return sum(1 for stop in stops if stop.late_minutes <= 1e-9) / len(stops)

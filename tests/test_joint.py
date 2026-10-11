"""Joint optimality on a case where the volume-shortest routes do not pack."""

import math

from app.domain.models import Dataset, LoadingRule, Location, Order, OrderLine, Sku, TravelArc, Vehicle, Weights
from app.domain.matrix import TravelMatrix
from app.domain.units import build_units, group_units_by_store
from app.optimizer.joint import solve_joint
from app.optimizer.master import solve_master
from app.optimizer.routing import materialize_routes
from app.optimizer.scoring import build_scales


def _arc(origin: str, dest: str, points: dict[str, tuple[float, float]]) -> list[TravelArc]:
    ax, ay = points[origin]
    bx, by = points[dest]
    distance = int(round(math.hypot(ax - bx, ay - by)))
    return [
        TravelArc(from_id=origin, to_id=dest, distance_m=distance, travel_time_s=max(distance // 10, 0)),
    ]


def cube_dataset() -> Dataset:
    points = {
        "depot": (0.0, 0.0),
        "s1": (1000.0, 0.0),
        "s2": (1200.0, 0.0),
        "s3": (0.0, 1000.0),
        "s4": (0.0, 1200.0),
        "s5": (-1000.0, 0.0),
        "s6": (-1200.0, 0.0),
    }
    depot = Location(id="depot", name="depot", kind="depot", x_m=0, y_m=0, window_open_s=0, window_close_s=100_000)
    stores = [
        Location(id=store_id, name=store_id, kind="store", x_m=x, y_m=y, window_open_s=0, window_close_s=100_000)
        for store_id, (x, y) in points.items()
        if store_id != "depot"
    ]
    sku = Sku(id="cube", name="cube", length_mm=600, width_mm=600, height_mm=600, weight_g=1000, fragile=False)
    orders = [
        Order(
            id=f"o-{store.id}",
            store_id=store.id,
            service_date="2026-10-10",
            lines=[OrderLine(id=f"l-{store.id}", sku_id="cube", qty=1)],
        )
        for store in stores
    ]
    vehicles = [
        Vehicle(
            id=f"v{index}",
            type_name="cube-van",
            length_mm=1000,
            width_mm=1000,
            height_mm=1000,
            max_weight_g=100_000,
            cost_per_km=1.0,
            fixed_cost=0.0,
        )
        for index in range(1, 7)
    ]
    arcs: list[TravelArc] = []
    ids = list(points)
    for origin in ids:
        for dest in ids:
            if origin == dest:
                continue
            arcs.extend(_arc(origin, dest, points))
    return Dataset(depot=depot, stores=stores, skus=[sku], orders=orders, vehicles=vehicles, arcs=arcs, loading_rule=LoadingRule())


def test_joint_search_proves_the_packable_optimum() -> None:
    dataset = cube_dataset()
    result = solve_joint(dataset, weights=Weights(), seed=1, time_limit_s=60)
    assert result.status == "completed", result.error_message
    assert result.proven_optimal
    assert result.gap == 0
    assert result.reshuffle_count == 0
    used = [route for route in result.routes if route.stops]
    assert len(used) == 6
    assert all(len(route.stops) == 1 for route in used)
    assert sum(route.distance_m for route in used) == 13200

    matrix = TravelMatrix(dataset.arcs)
    nodes = group_units_by_store(dataset, build_units(dataset))
    scales = build_scales(dataset, matrix)
    plain = solve_master(nodes, dataset, matrix, Weights(), scales, [], [], time_limit_s=20, seed=1)
    plain_routes = materialize_routes(
        {vehicle_id: [next(node for node in nodes if node.node_id == node_id) for node_id in sequence] for vehicle_id, sequence in plain.assignment.items()},
        dataset,
        matrix,
    )
    assert plain.status == "optimal"
    assert plain_routes is not None
    assert sum(route.distance_m for route in plain_routes) < 13200


def test_split_penalty_prefers_the_intact_order() -> None:
    from app.optimizer.scoring import score_routes
    from app.domain.models import RoutePlan, Scales, StopPlan

    def route(vehicle_id: str, store_id: str, units: list[str]) -> RoutePlan:
        return RoutePlan(
            vehicle_id=vehicle_id,
            stops=[
                StopPlan(
                    store_id=store_id,
                    node_ids=[store_id],
                    unit_ids=units,
                    arrival_s=0,
                    depart_s=0,
                    late_minutes=0,
                    sequence=1,
                )
            ],
            distance_m=1000,
            travel_time_s=100,
            travel_cost=1,
            fixed_cost=0,
            depart_depot_s=0,
            return_depot_s=100,
            total_weight_g=1,
            total_volume_mm3=1,
        )

    weights = Weights()
    scales = Scales(cost=10, time=1000, late=60, handling=2, split=2)
    intact = score_routes([route("v1", "s", ["a", "b"])], [], 0, weights, scales)
    split = score_routes([route("v1", "s", ["a"]), route("v2", "s", ["b"])], [], 1, weights, scales)
    assert intact < split


def test_triple_packing_cuts_forbid_three_cubes_that_fit_in_pairs() -> None:
    from app.optimizer.joint import _pair_cuts
    from app.optimizer.packing import can_pack_nodes

    points = {"depot": (0.0, 0.0), "s1": (100.0, 0.0), "s2": (200.0, 0.0), "s3": (300.0, 0.0)}
    depot = Location(id="depot", name="depot", kind="depot", x_m=0, y_m=0, window_open_s=0, window_close_s=100_000)
    stores = [
        Location(id=store_id, name=store_id, kind="store", x_m=x, y_m=y, window_open_s=0, window_close_s=100_000)
        for store_id, (x, y) in points.items()
        if store_id != "depot"
    ]
    sku = Sku(id="cube", name="cube", length_mm=400, width_mm=400, height_mm=400, weight_g=100)
    orders = [
        Order(
            id=f"o-{store.id}",
            store_id=store.id,
            service_date="2026-10-10",
            lines=[OrderLine(id=f"l-{store.id}", sku_id="cube", qty=1)],
        )
        for store in stores
    ]
    vehicles = [
        Vehicle(
            id="v1",
            type_name="van",
            length_mm=1000,
            width_mm=500,
            height_mm=400,
            max_weight_g=100_000,
            cost_per_km=1.0,
            fixed_cost=0.0,
        )
    ]
    dataset = Dataset(
        depot=depot,
        stores=stores,
        skus=[sku],
        orders=orders,
        vehicles=vehicles,
        arcs=[],
        loading_rule=LoadingRule(),
    )
    units = build_units(dataset)
    units_by_id = {unit.unit_id: unit for unit in units}
    nodes = group_units_by_store(dataset, units)
    vehicle = vehicles[0]
    assert can_pack_nodes(nodes[:2], vehicle, dataset.loading_rule, units_by_id)
    assert not can_pack_nodes(nodes, vehicle, dataset.loading_rule, units_by_id)
    cuts = _pair_cuts(nodes, dataset.vehicles, dataset, units_by_id, budget_s=5.0)
    pairs = [cut for cut in cuts if len(cut.node_ids) == 2]
    triples = [cut for cut in cuts if len(cut.node_ids) == 3]
    assert pairs == []
    assert len(triples) == 1
    assert triples[0].vehicle_ids is None

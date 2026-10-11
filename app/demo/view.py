"""Turn a solved day into the geometry the demo draws."""

from __future__ import annotations

from app.domain.matrix import TravelMatrix
from app.domain.models import Dataset, JointResult
from app.network.roads import ARTERIAL_SPEED_MPS, BLOCK_M, LOCAL_SPEED_MPS, RoadNetwork

_COLORS = (
    "#b45309",
    "#0369a1",
    "#15803d",
    "#7c3aed",
    "#be123c",
    "#0f766e",
    "#c2410c",
    "#4338ca",
)


def build_scene(dataset: Dataset, result: JointResult, scenario: dict) -> dict:
    network = RoadNetwork()
    matrix = TravelMatrix(dataset.arcs)
    names = {dataset.depot.id: dataset.depot.name}
    coordinates = {dataset.depot.id: (dataset.depot.x_m, dataset.depot.y_m)}
    for store in dataset.stores:
        names[store.id] = store.name
        coordinates[store.id] = (store.x_m, store.y_m)
    vehicles = {vehicle.id: vehicle for vehicle in dataset.vehicles}
    used = [route for route in result.routes if route.stops]
    colors = {route.vehicle_id: _COLORS[index % len(_COLORS)] for index, route in enumerate(used)}

    routes = []
    for route in used:
        vehicle = vehicles[route.vehicle_id]
        stop_ids = [dataset.depot.id, *[stop.store_id for stop in route.stops], dataset.depot.id]
        legs = []
        for origin_id, dest_id in zip(stop_ids, stop_ids[1:]):
            ox, oy = coordinates[origin_id]
            dx, dy = coordinates[dest_id]
            points = network.route_points(network.snap(ox, oy), network.snap(dx, dy))
            arc = matrix.get(origin_id, dest_id)
            legs.append(
                {
                    "from_id": origin_id,
                    "to_id": dest_id,
                    "distance_m": arc.distance_m,
                    "travel_time_s": arc.travel_time_s,
                    "points": [[x_m, y_m] for x_m, y_m in points],
                }
            )
        routes.append(
            {
                "vehicle_id": route.vehicle_id,
                "type_name": vehicle.type_name,
                "color": colors[route.vehicle_id],
                "length_mm": vehicle.length_mm,
                "width_mm": vehicle.width_mm,
                "height_mm": vehicle.height_mm,
                "distance_m": route.distance_m,
                "travel_time_s": route.travel_time_s,
                "travel_cost": route.travel_cost,
                "fixed_cost": route.fixed_cost,
                "legs": legs,
                "stops": [
                    {
                        "store_id": stop.store_id,
                        "name": names[stop.store_id],
                        "sequence": stop.sequence,
                        "arrival_s": stop.arrival_s,
                        "late_minutes": stop.late_minutes,
                        "x_m": coordinates[stop.store_id][0],
                        "y_m": coordinates[stop.store_id][1],
                    }
                    for stop in route.stops
                ],
            }
        )

    assigned: dict[str, str] = {}
    for route in routes:
        for stop in route["stops"]:
            assigned.setdefault(stop["store_id"], route["vehicle_id"])

    locations = [
        {
            "id": dataset.depot.id,
            "name": dataset.depot.name,
            "kind": "depot",
            "x_m": dataset.depot.x_m,
            "y_m": dataset.depot.y_m,
            "vehicle_id": None,
        }
    ]
    for store in dataset.stores:
        vehicle_id = assigned.get(store.id)
        locations.append(
            {
                "id": store.id,
                "name": store.name,
                "kind": "store",
                "x_m": store.x_m,
                "y_m": store.y_m,
                "vehicle_id": vehicle_id,
                "color": colors.get(vehicle_id) if vehicle_id else "#78716c",
                "window_open_s": store.window_open_s,
                "window_close_s": store.window_close_s,
            }
        )

    return {
        "scenario": scenario,
        "summary": {
            "proven_optimal": result.proven_optimal,
            "gap": result.gap,
            "score": result.score,
            "on_time_rate": result.on_time_rate,
            "split_order_count": result.split_order_count,
            "store_count": len(dataset.stores),
            "used_vehicle_count": len(routes),
            "reshuffle_count": result.reshuffle_count,
        },
        "network": {
            "block_m": BLOCK_M,
            "half_blocks": network.half_blocks,
            "river_x_m": int(1.5 * BLOCK_M),
            "bridge_y_m": 0,
            "arterial_every": 4,
            "local_speed_mps": LOCAL_SPEED_MPS,
            "arterial_speed_mps": ARTERIAL_SPEED_MPS,
        },
        "locations": locations,
        "routes": routes,
        "placements": [item.model_dump() for item in result.placements],
    }

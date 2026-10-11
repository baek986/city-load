"""Road-network distances follow streets, including the river detour."""

import math

from app.network.roads import BLOCK_M, Intersection, RoadNetwork, straight_line_m
from tests.generate_fixture import build_pharmacy_dataset


def test_adjacent_intersections_use_the_street_length() -> None:
    network = RoadNetwork()
    distance_m, travel_time_s = network.fastest_path(Intersection(1, 1), Intersection(1, 2))
    assert distance_m == BLOCK_M
    assert travel_time_s == max(int(round(BLOCK_M / 8)), 1)


def test_river_path_crosses_the_bridge() -> None:
    network = RoadNetwork()
    origin = Intersection(0, 4)
    dest = Intersection(4, 4)
    points = network.route_points(origin, dest)
    distance_m, _travel_time_s = network.fastest_path(origin, dest)
    length = sum(abs(right[0] - left[0]) + abs(right[1] - left[1]) for left, right in zip(points, points[1:]))
    assert points[0] == (origin.x_m, origin.y_m)
    assert points[-1] == (dest.x_m, dest.y_m)
    assert length == distance_m
    assert (BLOCK_M, 0) in points
    assert (2 * BLOCK_M, 0) in points


def test_river_forces_a_detour_longer_than_the_straight_line() -> None:
    network = RoadNetwork()
    origin = Intersection(0, 4)
    dest = Intersection(4, 4)
    distance_m, _travel_time_s = network.fastest_path(origin, dest)
    straight = straight_line_m(origin.x_m, origin.y_m, dest.x_m, dest.y_m)
    assert distance_m > straight
    assert distance_m > abs(dest.x_m - origin.x_m)


def test_pharmacy_arcs_follow_the_road_network() -> None:
    dataset = build_pharmacy_dataset()
    network = RoadNetwork()
    points = [(dataset.depot.id, dataset.depot.x_m, dataset.depot.y_m)]
    points.extend((store.id, store.x_m, store.y_m) for store in dataset.stores)
    expected = {(arc.from_id, arc.to_id): arc for arc in network.travel_arcs(points)}
    coordinates = {location_id: (x_m, y_m) for location_id, x_m, y_m in points}
    longer = 0
    for arc in dataset.arcs:
        road = expected[(arc.from_id, arc.to_id)]
        assert arc.distance_m == road.distance_m
        assert arc.travel_time_s == road.travel_time_s
        ox, oy = coordinates[arc.from_id]
        dx, dy = coordinates[arc.to_id]
        straight = int(round(math.hypot(ox - dx, oy - dy)))
        assert arc.distance_m >= straight
        if arc.distance_m > straight:
            longer += 1
    assert longer > 0

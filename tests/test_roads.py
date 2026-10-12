"""Road-network distances follow streets, including the river detour."""

import math

from app.network.roads import BLOCK_M, Intersection, RoadNetwork, node_xy, straight_line_m
from tests.generate_fixture import build_pharmacy_dataset


def _segment_length(points: list[tuple[int, int]]) -> int:
    return sum(int(round(math.hypot(right[0] - left[0], right[1] - left[1]))) for left, right in zip(points, points[1:]))


def test_school_campus_is_not_split_by_a_road() -> None:
    network = RoadNetwork()
    distance_m, _travel_time_s = network.fastest_path(Intersection(5, -2), Intersection(5, -1))
    assert distance_m > BLOCK_M


def test_a_direct_street_is_not_left_for_a_parallel_road() -> None:
    network = RoadNetwork()
    north = network.route_points(Intersection(6, -6), Intersection(6, 10))
    assert all(abs(x_m - 6 * BLOCK_M) <= 120 for x_m, _y_m in north)
    east = network.route_points(Intersection(-10, 5), Intersection(-3, 5))
    assert all(abs(y_m - 5 * BLOCK_M) <= 120 for _x_m, y_m in east)


def test_a_street_bends_without_zigzagging() -> None:
    worst = 0.0
    for i in range(-8, 9):
        for j in range(-8, 7):
            for horizontal in (False, True):
                if horizontal:
                    a, b, c = node_xy(i, j), node_xy(i + 1, j), node_xy(i + 2, j)
                else:
                    a, b, c = node_xy(i, j), node_xy(i, j + 1), node_xy(i, j + 2)
                if (0, 0) in (a, b, c):
                    continue
                v1 = (b[0] - a[0], b[1] - a[1])
                v2 = (c[0] - b[0], c[1] - b[1])
                l1 = math.hypot(*v1) or 1
                l2 = math.hypot(*v2) or 1
                dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)))
                worst = max(worst, math.degrees(math.acos(dot)))
    assert worst < 20


def test_local_corners_leave_the_square_grid() -> None:
    corner = Intersection(3, 1)
    assert corner.x_m != 3 * BLOCK_M
    assert corner.y_m != BLOCK_M
    assert Intersection(0, 0).x_m == 0 and Intersection(0, 0).y_m == 0
    assert Intersection(1, 8).x_m == BLOCK_M and Intersection(1, 8).y_m == 8 * BLOCK_M
    assert node_xy(2, -8) == (2 * BLOCK_M, -8 * BLOCK_M)


def test_adjacent_intersections_use_the_street_length() -> None:
    network = RoadNetwork()
    left, right = Intersection(1, 1), Intersection(1, 2)
    distance_m, travel_time_s = network.fastest_path(left, right)
    straight = max(int(round(math.hypot(right.x_m - left.x_m, right.y_m - left.y_m))), 1)
    assert distance_m == straight
    assert travel_time_s == max(int(round(straight / 8)), 1)
    back_distance, back_time = network.fastest_path(right, left)
    assert back_distance == distance_m
    assert back_time == travel_time_s


def test_river_path_crosses_the_bridge() -> None:
    network = RoadNetwork()
    origin = Intersection(0, 4)
    dest = Intersection(4, 4)
    points = network.route_points(origin, dest)
    distance_m, _travel_time_s = network.fastest_path(origin, dest)
    assert points[0] == (origin.x_m, origin.y_m)
    assert points[-1] == (dest.x_m, dest.y_m)
    assert _segment_length(points) == distance_m
    crossings = [point[1] for point in points if point[0] in (BLOCK_M, 2 * BLOCK_M)]
    assert 0 in crossings or 8 * BLOCK_M in crossings or -8 * BLOCK_M in crossings


def test_north_bank_uses_the_north_bridge() -> None:
    network = RoadNetwork()
    points = network.route_points(Intersection(0, 12), Intersection(4, 12))
    assert (BLOCK_M, 8 * BLOCK_M) in points
    assert (2 * BLOCK_M, 8 * BLOCK_M) in points
    assert (BLOCK_M, 0) not in points


def test_park_blocks_the_shortcut() -> None:
    network = RoadNetwork()
    origin, dest = Intersection(-4, 0), Intersection(-2, 0)
    distance_m, _travel_time_s = network.fastest_path(origin, dest)
    points = network.route_points(origin, dest)
    assert node_xy(-3, 0) not in points
    assert distance_m > straight_line_m(origin.x_m, origin.y_m, dest.x_m, dest.y_m)


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
        assert arc.distance_m >= straight - 1
        if arc.distance_m > straight:
            longer += 1
    assert longer > 0

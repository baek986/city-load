"""Simulated urban road network.

Local streets are offset so blocks are not a perfect square grid. Arterials
and the river stay straight. A river cuts local streets except at three
bridges. Parks have no through roads. Depot and stores sit on intersections;
travel arcs are shortest paths on this graph.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from app.domain.models import TravelArc

BLOCK_M = 400
LOCAL_SPEED_MPS = 8
ARTERIAL_SPEED_MPS = 14
# Local streets cannot cross i = 2 except at these bridges. Depot at the origin
# sits on the west bank; stores with i >= RIVER_I must use a bridge.
RIVER_I = 2
BRIDGES_J = (0, 8, -8)
# Cell rectangles [i, i+w) x [j, j+h). Interior streets are closed.
PARKS = (
    (-4, -1, 2, 2),
    (8, 10, 2, 2),
    (-8, 12, 2, 2),
    (8, -8, 2, 2),
)
# Campuses and other blocks that occupy more than one cell. Same closure as parks.
BLOCKS = (
    (4, -2, 2, 1),
    (3, 5, 1, 2),
    (-10, 4, 2, 1),
    (-12, -6, 2, 1),
    (14, -2, 2, 1),
)


def bank_of(x_m: float) -> str:
    """'west' is the depot side of the river; 'east' must cross a bridge."""
    i = int(round(x_m / BLOCK_M))
    return "west" if i < RIVER_I else "east"


_SPAN = 4


def _ctrl(seed: int, k: int, amp: int) -> int:
    n = (seed * 374761393 + k * 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return int(n % (2 * amp + 1)) - amp


def _along(seed: int, t: int, amp: int) -> int:
    """Offset that changes smoothly along a street, not independently at each corner."""
    cell = math.floor(t / _SPAN)
    frac = t - cell * _SPAN
    a = _ctrl(seed, cell, amp)
    b = _ctrl(seed, cell + 1, amp)
    return math.floor((a * (_SPAN - frac) + b * frac) / _SPAN)


def node_xy(i: int, j: int) -> tuple[int, int]:
    """Intersection position.

    The depot stays at the origin. River banks stay vertical and the three
    bridge rows stay level. Other streets bend gradually, so a route along one
    street does not zigzag at every block.
    """
    x = i * BLOCK_M
    y = j * BLOCK_M
    if i == 0 and j == 0:
        return 0, 0
    lock_x = i in (1, 2)
    lock_y = j in BRIDGES_J
    if not lock_x:
        x += _along(10_000 + i, j, 36 if i % 4 == 0 else 96)
    if not lock_y:
        y += _along(80_000 + j, i, 36 if j % 4 == 0 else 96)
    return x, y


def _arterial_edge(left: Intersection, right: Intersection) -> bool:
    return (left.i % 4 == 0 and right.i % 4 == 0) or (left.j % 4 == 0 and right.j % 4 == 0)


def _through_block(left: Intersection, right: Intersection) -> bool:
    """True when the edge runs through a park or a multi-cell campus."""
    footprints = PARKS + BLOCKS
    if left.i == right.i:
        i = left.i
        j = min(left.j, right.j)
        return any(pi < i < pi + w and pj <= j < pj + h for pi, pj, w, h in footprints)
    i = min(left.i, right.i)
    j = left.j
    return any(pj < j < pj + h and pi <= i < pi + w for pi, pj, w, h in footprints)


@dataclass(frozen=True)
class Intersection:
    i: int
    j: int

    @property
    def x_m(self) -> int:
        return node_xy(self.i, self.j)[0]

    @property
    def y_m(self) -> int:
        return node_xy(self.i, self.j)[1]


class RoadNetwork:
    def __init__(self, half_blocks: int = 32) -> None:
        self.half_blocks = half_blocks
        self._neighbors: dict[Intersection, list[tuple[Intersection, int, int]]] = {}
        self._build()

    def _build(self) -> None:
        nodes = [
            Intersection(i, j)
            for i in range(-self.half_blocks, self.half_blocks + 1)
            for j in range(-self.half_blocks, self.half_blocks + 1)
        ]
        for node in nodes:
            self._neighbors[node] = []
        for node in nodes:
            for di, dj in ((1, 0), (0, 1)):
                other = Intersection(node.i + di, node.j + dj)
                if other not in self._neighbors:
                    continue
                if self._blocked(node, other):
                    continue
                self._link(node, other)

    def _link(self, left: Intersection, right: Intersection) -> None:
        arterial = _arterial_edge(left, right)
        speed = ARTERIAL_SPEED_MPS if arterial else LOCAL_SPEED_MPS
        meters = max(int(round(math.hypot(right.x_m - left.x_m, right.y_m - left.y_m))), 1)
        seconds = max(int(round(meters / speed)), 1)
        self._neighbors[left].append((right, meters, seconds))
        self._neighbors[right].append((left, meters, seconds))

    @staticmethod
    def _blocked(left: Intersection, right: Intersection) -> bool:
        """No road through a park, and no river crossing except on a bridge."""
        if _through_block(left, right):
            return True
        crosses_river = left.i < RIVER_I <= right.i or right.i < RIVER_I <= left.i
        if not crosses_river or left.j != right.j:
            return False
        return left.j not in BRIDGES_J

    def snap(self, x_m: float, y_m: float) -> Intersection:
        i = max(-self.half_blocks, min(self.half_blocks, int(round(x_m / BLOCK_M))))
        j = max(-self.half_blocks, min(self.half_blocks, int(round(y_m / BLOCK_M))))
        return Intersection(i, j)

    def _search(self, origin: Intersection) -> tuple[dict[Intersection, tuple[int, int]], dict[Intersection, Intersection]]:
        """Shortest paths from origin. Equal length prefers the faster street.

        Reached values are (distance_m, travel_time_s). Parents rebuild the same path.
        A parallel arterial is not worth leaving the direct street and coming back.
        """
        best: dict[Intersection, tuple[int, int]] = {origin: (0, 0)}
        parent: dict[Intersection, Intersection] = {}
        queue: list[tuple[int, int, int, Intersection]] = [(0, 0, 0, origin)]
        sequence = 0
        while queue:
            distance, elapsed, _order, node = heapq.heappop(queue)
            if (distance, elapsed) != best[node]:
                continue
            for neighbor, meters, seconds in self._neighbors[node]:
                candidate = (distance + meters, elapsed + seconds)
                known = best.get(neighbor)
                if known is None or candidate < known:
                    sequence += 1
                    best[neighbor] = candidate
                    parent[neighbor] = node
                    heapq.heappush(queue, (candidate[0], candidate[1], sequence, neighbor))
        return best, parent

    def fastest_path(self, origin: Intersection, dest: Intersection) -> tuple[int, int]:
        if origin == dest:
            return 0, 0
        found = self._search(origin)[0].get(dest)
        if found is None:
            raise ValueError(f"No road path from {origin} to {dest}")
        return found

    def route_points(self, origin: Intersection, dest: Intersection) -> list[tuple[int, int]]:
        """Intersection coordinates along the shortest path, including both ends."""
        if origin == dest:
            return [(origin.x_m, origin.y_m)]
        _reached, parent = self._search(origin)
        if dest not in parent and dest != origin:
            raise ValueError(f"No road path from {origin} to {dest}")
        points: list[tuple[int, int]] = []
        node: Intersection | None = dest
        while node is not None:
            points.append((node.x_m, node.y_m))
            node = parent.get(node)
        points.reverse()
        return points

    def travel_arcs(self, locations: list[tuple[str, float, float]]) -> list[TravelArc]:
        snapped = [(location_id, self.snap(x_m, y_m)) for location_id, x_m, y_m in locations]
        arcs: list[TravelArc] = []
        for from_id, origin in snapped:
            reached = self._search(origin)[0]
            for to_id, dest in snapped:
                if from_id == to_id:
                    continue
                found = reached.get(dest)
                if found is None:
                    raise ValueError(f"No road path from {from_id} to {to_id}")
                distance_m, travel_time_s = found
                arcs.append(
                    TravelArc(
                        from_id=from_id,
                        to_id=to_id,
                        distance_m=distance_m,
                        travel_time_s=travel_time_s,
                    )
                )
        return arcs


def straight_line_m(x1: float, y1: float, x2: float, y2: float) -> int:
    return int(round(math.hypot(x1 - x2, y1 - y2)))

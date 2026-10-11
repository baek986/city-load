"""Simulated urban road network.

Streets form a grid. Arterials are faster. A river cuts local streets, so the
fastest path can be longer than the straight line. Depot and stores sit on
intersections; travel arcs are shortest paths on this graph.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from app.domain.models import TravelArc

BLOCK_M = 400
LOCAL_SPEED_MPS = 8
ARTERIAL_SPEED_MPS = 14
# Local streets cannot cross i = 2 except at j = 0. Depot at the origin sits on
# the west bank; stores with i >= RIVER_I must use the bridge.
RIVER_I = 2


def bank_of(x_m: float) -> str:
    """'west' is the depot side of the river; 'east' must cross the bridge."""
    i = int(round(x_m / BLOCK_M))
    return "west" if i < RIVER_I else "east"


@dataclass(frozen=True)
class Intersection:
    i: int
    j: int

    @property
    def x_m(self) -> int:
        return self.i * BLOCK_M

    @property
    def y_m(self) -> int:
        return self.j * BLOCK_M


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
                arterial = node.i % 4 == 0 and other.i % 4 == 0 or node.j % 4 == 0 and other.j % 4 == 0
                speed = ARTERIAL_SPEED_MPS if arterial else LOCAL_SPEED_MPS
                seconds = max(int(round(BLOCK_M / speed)), 1)
                self._neighbors[node].append((other, BLOCK_M, seconds))
                self._neighbors[other].append((node, BLOCK_M, seconds))

    @staticmethod
    def _blocked(left: Intersection, right: Intersection) -> bool:
        """Local streets cannot cross the river except on the arterial bridge at j = 0."""
        crosses_river = left.i < 2 <= right.i or right.i < 2 <= left.i
        if not crosses_river or left.j != right.j:
            return False
        return left.j != 0

    def snap(self, x_m: float, y_m: float) -> Intersection:
        i = max(-self.half_blocks, min(self.half_blocks, int(round(x_m / BLOCK_M))))
        j = max(-self.half_blocks, min(self.half_blocks, int(round(y_m / BLOCK_M))))
        return Intersection(i, j)

    def _search(self, origin: Intersection) -> tuple[dict[Intersection, tuple[int, int]], dict[Intersection, Intersection]]:
        """Fastest paths from origin.

        Reached values are (distance_m, travel_time_s). Parents rebuild the same path.
        """
        best_time: dict[Intersection, tuple[int, int]] = {origin: (0, 0)}
        parent: dict[Intersection, Intersection] = {}
        queue: list[tuple[int, int, int, Intersection]] = [(0, 0, 0, origin)]
        sequence = 0
        while queue:
            elapsed, distance, _order, node = heapq.heappop(queue)
            if (elapsed, distance) != best_time[node]:
                continue
            for neighbor, meters, seconds in self._neighbors[node]:
                candidate = (elapsed + seconds, distance + meters)
                known = best_time.get(neighbor)
                if known is None or candidate < known:
                    sequence += 1
                    best_time[neighbor] = candidate
                    parent[neighbor] = node
                    heapq.heappush(queue, (candidate[0], candidate[1], sequence, neighbor))
        reached = {node: (distance, elapsed) for node, (elapsed, distance) in best_time.items()}
        return reached, parent

    def fastest_path(self, origin: Intersection, dest: Intersection) -> tuple[int, int]:
        if origin == dest:
            return 0, 0
        found = self._search(origin)[0].get(dest)
        if found is None:
            raise ValueError(f"No road path from {origin} to {dest}")
        return found

    def route_points(self, origin: Intersection, dest: Intersection) -> list[tuple[int, int]]:
        """Intersection coordinates along the fastest path, including both ends."""
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

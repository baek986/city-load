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

    def _search(self, origin: Intersection) -> dict[Intersection, tuple[int, int]]:
        """Map each reached intersection to (distance_m, travel_time_s) on the fastest path."""
        best_time: dict[Intersection, tuple[int, int]] = {origin: (0, 0)}
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
                    heapq.heappush(queue, (candidate[0], candidate[1], sequence, neighbor))
        return {node: (distance, elapsed) for node, (elapsed, distance) in best_time.items()}

    def fastest_path(self, origin: Intersection, dest: Intersection) -> tuple[int, int]:
        if origin == dest:
            return 0, 0
        found = self._search(origin).get(dest)
        if found is None:
            raise ValueError(f"No road path from {origin} to {dest}")
        return found

    def travel_arcs(self, locations: list[tuple[str, float, float]]) -> list[TravelArc]:
        snapped = [(location_id, self.snap(x_m, y_m)) for location_id, x_m, y_m in locations]
        arcs: list[TravelArc] = []
        for from_id, origin in snapped:
            reached = self._search(origin)
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

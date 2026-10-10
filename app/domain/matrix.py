"""Travel-time matrix. A missing arc is unreachable."""

from __future__ import annotations

from app.domain.errors import MissingTravelArcError
from app.domain.models import TravelArc


class TravelMatrix:
    def __init__(self, arcs: list[TravelArc]) -> None:
        self._arcs = {(arc.from_id, arc.to_id): arc for arc in arcs}

    def get(self, from_id: str, to_id: str) -> TravelArc:
        if from_id == to_id:
            return TravelArc(from_id=from_id, to_id=to_id, distance_m=0, travel_time_s=0)
        arc = self._arcs.get((from_id, to_id))
        if arc is None:
            raise MissingTravelArcError(from_id, to_id)
        return arc

    def require_complete(self, location_ids: list[str]) -> None:
        for origin in location_ids:
            for dest in location_ids:
                if origin == dest:
                    continue
                if (origin, dest) not in self._arcs:
                    raise MissingTravelArcError(origin, dest)

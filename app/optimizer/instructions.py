"""Load and unload steps. Loading is the reverse of the delivery sequence."""

from __future__ import annotations

from app.domain.models import HandlingStep, Placement, RoutePlan, StepPlacement


def _step_placements(items: list[Placement]) -> list[StepPlacement]:
    return [
        StepPlacement(
            unit_id=item.unit_id,
            x=item.x,
            y=item.y,
            z=item.z,
            dx=item.dx,
            dy=item.dy,
            dz=item.dz,
            rotation=item.rotation,
        )
        for item in items
    ]


def build_instructions(routes: list[RoutePlan], placements: list[Placement]) -> list[HandlingStep]:
    by_stop: dict[tuple[str, str], list[Placement]] = {}
    for placement in placements:
        by_stop.setdefault((placement.vehicle_id, placement.store_id), []).append(placement)
    unload: list[HandlingStep] = []
    for route in routes:
        for stop in route.stops:
            items = by_stop.get((route.vehicle_id, stop.store_id), [])
            ordered = sorted(items, key=lambda item: (-item.z, item.x, item.y, item.unit_id))
            unload.append(
                HandlingStep(
                    step_no=0,
                    action="unload",
                    vehicle_id=route.vehicle_id,
                    store_id=stop.store_id,
                    unit_ids=[item.unit_id for item in ordered],
                    placements=_step_placements(ordered),
                )
            )
    load: list[HandlingStep] = []
    for step in reversed(unload):
        reversed_items = list(reversed(step.placements))
        load.append(
            HandlingStep(
                step_no=0,
                action="load",
                vehicle_id=step.vehicle_id,
                store_id=step.store_id,
                unit_ids=[item.unit_id for item in reversed_items],
                placements=reversed_items,
            )
        )
    steps = unload + load
    for index, step in enumerate(steps, start=1):
        step.step_no = index
    return steps

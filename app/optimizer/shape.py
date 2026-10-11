"""Cargo-shape signatures so routing can prefer stores that tessellate.

Geography still dominates. These scores only break ties and slightly bias
merges: a sturdy, tall stop can carry a fragile, short one even if it sits a
bit farther away. The packer remains the feasibility judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.models import CustomerNode, Dataset, PackingUnit
from app.domain.units import min_footprint_mm2
from app.network.roads import bank_of

TALL_MM = 350


@dataclass(frozen=True)
class CargoShape:
    count: int
    footprint_mm2: int
    volume_mm3: int
    fragile: int
    sturdy: int
    upright: int
    height_sum: int
    tall: int

    @property
    def empty(self) -> bool:
        return self.count <= 0


def shape_from_units(units: list[PackingUnit]) -> CargoShape:
    if not units:
        return CargoShape(0, 0, 0, 0, 0, 0, 0, 0)
    fragile = sum(1 for unit in units if unit.fragile)
    upright = sum(1 for unit in units if unit.upright_only)
    return CargoShape(
        count=len(units),
        footprint_mm2=sum(min_footprint_mm2(unit) for unit in units),
        volume_mm3=sum(unit.volume_mm3 for unit in units),
        fragile=fragile,
        sturdy=len(units) - fragile,
        upright=upright,
        height_sum=sum(unit.height_mm for unit in units),
        tall=sum(1 for unit in units if unit.height_mm >= TALL_MM),
    )


def combine_shapes(shapes: list[CargoShape]) -> CargoShape:
    if not shapes:
        return CargoShape(0, 0, 0, 0, 0, 0, 0, 0)
    return CargoShape(
        count=sum(item.count for item in shapes),
        footprint_mm2=sum(item.footprint_mm2 for item in shapes),
        volume_mm3=sum(item.volume_mm3 for item in shapes),
        fragile=sum(item.fragile for item in shapes),
        sturdy=sum(item.sturdy for item in shapes),
        upright=sum(item.upright for item in shapes),
        height_sum=sum(item.height_sum for item in shapes),
        tall=sum(item.tall for item in shapes),
    )


def shapes_from_nodes(nodes: list[CustomerNode], units_by_id: dict[str, PackingUnit]) -> dict[str, CargoShape]:
    return {
        node.node_id: shape_from_units([units_by_id[unit_id] for unit_id in node.unit_ids if unit_id in units_by_id])
        for node in nodes
    }


def complementarity(left: CargoShape, right: CargoShape) -> float:
    """1 means the two loads fill different roles in the same bay; 0 means they compete."""
    if left.empty or right.empty:
        return 0.0
    nest = min(
        1.0,
        (left.sturdy / left.count) * (right.fragile / right.count)
        + (right.sturdy / right.count) * (left.fragile / left.count),
    )
    height = abs(left.tall / left.count - right.tall / right.count)
    return 0.55 * nest + 0.45 * height


def banks_for(nodes: list[CustomerNode], dataset: Dataset) -> dict[str, str]:
    stores = {store.id: store for store in dataset.stores}
    return {node.node_id: bank_of(stores[node.store_id].x_m) for node in nodes if node.store_id in stores}


def route_bank(nodes: list[CustomerNode], banks: dict[str, str]) -> str | None:
    seen = {banks[node.node_id] for node in nodes if node.node_id in banks}
    if len(seen) == 1:
        return next(iter(seen))
    return None

"""Turn order lines into physical cartons and one routing node per store group."""

from __future__ import annotations

from collections import defaultdict

from app.domain.errors import CargoExceedsVehicleError
from app.domain.models import CustomerNode, Dataset, PackingUnit, Vehicle


def build_units(dataset: Dataset) -> list[PackingUnit]:
    skus = {sku.id: sku for sku in dataset.skus}
    units: list[PackingUnit] = []
    for order in dataset.orders:
        for line in order.lines:
            sku = skus[line.sku_id]
            rotations = list(sku.allowed_rotations)
            if sku.upright_only:
                rotations = [name for name in rotations if name in ("LWH", "WLH")]
            for index in range(line.qty):
                units.append(
                    PackingUnit(
                        unit_id=f"{line.id}#{index + 1}",
                        order_id=order.id,
                        order_line_id=line.id,
                        store_id=order.store_id,
                        sku_id=sku.id,
                        length_mm=sku.length_mm,
                        width_mm=sku.width_mm,
                        height_mm=sku.height_mm,
                        weight_g=sku.weight_g,
                        fragile=sku.fragile,
                        stackable=sku.stackable,
                        max_stack_load_g=sku.max_stack_load_g,
                        upright_only=sku.upright_only,
                        allowed_rotations=rotations or ["LWH"],
                    )
                )
    return units


def _rotations(unit: PackingUnit) -> list[tuple[int, int, int]]:
    dims = (unit.length_mm, unit.width_mm, unit.height_mm)
    perms = {
        "LWH": (0, 1, 2),
        "WLH": (1, 0, 2),
        "LHW": (0, 2, 1),
        "HLW": (2, 0, 1),
        "WHL": (1, 2, 0),
        "HWL": (2, 1, 0),
    }
    boxes: list[tuple[int, int, int]] = []
    for name in unit.allowed_rotations:
        perm = perms.get(name)
        if perm is None:
            continue
        boxes.append((dims[perm[0]], dims[perm[1]], dims[perm[2]]))
    return boxes


def min_footprint_mm2(unit: PackingUnit) -> int:
    """Smallest base the carton can stand on under its allowed rotations."""
    faces = [dx * dy for dx, dy, _dz in _rotations(unit)]
    return min(faces) if faces else unit.length_mm * unit.width_mm


def max_standing_height_mm(unit: PackingUnit) -> int:
    """Tallest the carton can stand under its allowed rotations."""
    heights = [dz for _dx, _dy, dz in _rotations(unit)]
    return max(heights) if heights else unit.height_mm


def unit_fits_vehicle(unit: PackingUnit, vehicle: Vehicle) -> bool:
    if unit.weight_g > vehicle.max_weight_g or unit.volume_mm3 > vehicle.volume_mm3:
        return False
    return any(
        dx <= vehicle.length_mm and dy <= vehicle.width_mm and dz <= vehicle.height_mm
        for dx, dy, dz in _rotations(unit)
    )


def assert_units_fit_fleet(units: list[PackingUnit], vehicles: list[Vehicle]) -> None:
    bad = [unit.unit_id for unit in units if not any(unit_fits_vehicle(unit, vehicle) for vehicle in vehicles)]
    if bad:
        raise CargoExceedsVehicleError(bad)


def make_node(node_id: str, store_id: str, units: list[PackingUnit], dataset: Dataset) -> CustomerNode:
    stores = {store.id: store for store in dataset.stores}
    store = stores[store_id]
    order_ids = sorted({unit.order_id for unit in units})
    return CustomerNode(
        node_id=node_id,
        store_id=store_id,
        unit_ids=[unit.unit_id for unit in units],
        order_ids=order_ids,
        weight_g=sum(unit.weight_g for unit in units),
        volume_mm3=sum(unit.volume_mm3 for unit in units),
        service_time_s=store.service_time_s,
        window_open_s=store.window_open_s,
        window_close_s=store.window_close_s,
        soft_deadline_s=store.soft_deadline_s,
        footprint_mm2=sum(min_footprint_mm2(unit) for unit in units),
        max_extent_mm=max((max_standing_height_mm(unit) for unit in units), default=0),
    )


def group_units_by_store(dataset: Dataset, units: list[PackingUnit]) -> list[CustomerNode]:
    grouped: dict[str, list[PackingUnit]] = defaultdict(list)
    for unit in units:
        grouped[unit.store_id].append(unit)
    nodes: list[CustomerNode] = []
    for store_id in sorted(grouped):
        nodes.append(make_node(f"store:{store_id}", store_id, grouped[store_id], dataset))
    return nodes

"""Packing geometry tests."""

import time

from app.domain.models import CustomerNode, Dataset, LoadingRule, Location, Order, OrderLine, PackingUnit, Placement, Sku, Vehicle
from app.domain.units import build_units
from app.optimizer.packing import PackStop, boxes_overlap, later_blocks_earlier, minimal_infeasible_nodes, pack_stops


def _unit(unit_id: str, length: int, width: int, height: int, weight: int = 100, fragile: bool = False, store: str = "s") -> PackingUnit:
    return PackingUnit(
        unit_id=unit_id,
        order_id=store,
        order_line_id=unit_id,
        store_id=store,
        sku_id="sku",
        length_mm=length,
        width_mm=width,
        height_mm=height,
        weight_g=weight,
        fragile=fragile,
        stackable=not fragile,
        max_stack_load_g=0 if fragile else 100_000,
        upright_only=False,
        allowed_rotations=["LWH", "WLH", "LHW", "HLW", "WHL", "HWL"],
    )


def _vehicle() -> Vehicle:
    return Vehicle(
        id="v1",
        type_name="van",
        length_mm=1000,
        width_mm=400,
        height_mm=400,
        max_weight_g=10_000,
        cost_per_km=1,
        fixed_cost=0,
    )


def test_touching_boxes_do_not_overlap() -> None:
    left = Placement(
        unit_id="a",
        vehicle_id="v",
        store_id="s",
        order_id="o",
        x=0,
        y=0,
        z=0,
        dx=10,
        dy=10,
        dz=10,
        rotation="LWH",
        fragile=False,
        stackable=True,
        max_stack_load_g=1,
        weight_g=1,
    )
    touching = left.model_copy(update={"unit_id": "b", "x": 10})
    crossing = left.model_copy(update={"unit_id": "c", "x": 9})
    assert not boxes_overlap(left, touching)
    assert boxes_overlap(left, crossing)


def test_lifo_blocks_keep_earlier_stop_near_the_door() -> None:
    vehicle = _vehicle()
    stops = [
        PackStop("s1", ["n1"], [_unit("a", 400, 400, 400, store="s1")]),
        PackStop("s2", ["n2"], [_unit("b", 400, 400, 400, store="s2")]),
    ]
    result = pack_stops(vehicle, stops, LoadingRule())
    assert result.ok
    first = next(item for item in result.placements if item.store_id == "s1")
    second = next(item for item in result.placements if item.store_id == "s2")
    assert first.x + first.dx <= second.x


def test_overweight_route_is_rejected_before_geometry() -> None:
    vehicle = _vehicle().model_copy(update={"max_weight_g": 1000})
    stops = [
        PackStop("s1", ["n1"], [_unit("a", 100, 100, 100, weight=800, store="s1")]),
        PackStop("s2", ["n2"], [_unit("b", 100, 100, 100, weight=800, store="s2")]),
    ]
    result = pack_stops(vehicle, stops, LoadingRule())
    assert not result.ok
    assert result.reason == "capacity"
    assert set(result.failed_unit_ids) == {"a", "b"}


def test_nothing_is_placed_on_a_fragile_item() -> None:
    vehicle = _vehicle().model_copy(update={"length_mm": 400, "width_mm": 400, "height_mm": 800})
    stops = [
        PackStop(
            "s1",
            ["n1"],
            [
                _unit("solid", 400, 400, 200, store="s1"),
                _unit("glass", 400, 400, 200, fragile=True, store="s1"),
            ],
        )
    ]
    result = pack_stops(vehicle, stops, LoadingRule())
    assert result.ok
    glass = next(item for item in result.placements if item.unit_id == "glass")
    solid = next(item for item in result.placements if item.unit_id == "solid")
    assert glass.z >= solid.z + solid.dz or not (
        glass.x < solid.x + solid.dx and solid.x < glass.x + glass.dx and glass.z >= solid.z + solid.dz
    )
    for item in result.placements:
        if item.unit_id == "glass":
            continue
        assert not (item.z >= glass.z + glass.dz and item.x < glass.x + glass.dx and glass.x < item.x + item.dx)


def test_minimal_infeasible_pair_cannot_share_a_vehicle() -> None:
    vehicle = Vehicle(
        id="v",
        type_name="van",
        length_mm=1000,
        width_mm=1000,
        height_mm=1000,
        max_weight_g=100_000,
        cost_per_km=1,
        fixed_cost=0,
    )
    units = {
        "a": _unit("a", 600, 600, 600, store="s1"),
        "b": _unit("b", 600, 600, 600, store="s2"),
    }
    nodes = [
        CustomerNode(
            node_id="n1",
            store_id="s1",
            unit_ids=["a"],
            order_ids=["o1"],
            weight_g=100,
            volume_mm3=600**3,
            service_time_s=0,
            window_open_s=0,
            window_close_s=100_000,
        ),
        CustomerNode(
            node_id="n2",
            store_id="s2",
            unit_ids=["b"],
            order_ids=["o2"],
            weight_g=100,
            volume_mm3=600**3,
            service_time_s=0,
            window_open_s=0,
            window_close_s=100_000,
        ),
    ]
    subset = minimal_infeasible_nodes(nodes, vehicle, LoadingRule(), units)
    assert {node.node_id for node in subset} == {"n1", "n2"}


def _pharmacy_units(qty: int) -> tuple[Vehicle, list[PackingUnit]]:
    """Same 20 pharmacy SKUs as the fixture, with more than one carton per line."""
    skus: list[Sku] = []
    for index in range(20):
        fragile = index % 7 == 0
        upright = index % 11 == 0
        skus.append(
            Sku(
                id=f"sku-{index + 1:02d}",
                name=f"SKU{index + 1:02d}",
                length_mm=180 + (index % 4) * 20,
                width_mm=140 + (index % 3) * 15,
                height_mm=100 + (index % 5) * 10,
                weight_g=400 + index * 40,
                fragile=fragile,
                stackable=not upright,
                upright_only=upright,
                max_stack_load_g=0 if upright or fragile else 20_000,
            )
        )
    depot = Location(id="depot", name="中心仓", kind="depot", x_m=0, y_m=0)
    store = Location(id="store-01", name="药店", kind="store", x_m=400, y_m=400)
    order = Order(
        id="order-store-01",
        store_id=store.id,
        service_date="2026-10-10",
        lines=[OrderLine(id=f"line-{index + 1:02d}", sku_id=sku.id, qty=qty) for index, sku in enumerate(skus)],
    )
    vehicle = Vehicle(
        id="van-1",
        type_name="van",
        length_mm=3200,
        width_mm=1700,
        height_mm=1700,
        max_weight_g=1_500_000,
        cost_per_km=4.5,
        fixed_cost=80,
    )
    dataset = Dataset(depot=depot, stores=[store], skus=skus, orders=[order], vehicles=[vehicle], arcs=[])
    return vehicle, build_units(dataset)


def _assert_packed(vehicle: Vehicle, result) -> None:
    assert result.ok, result.failed_unit_ids
    assert result.reason == "ok"
    for item in result.placements:
        assert item.x >= 0 and item.y >= 0 and item.z >= 0
        assert item.x + item.dx <= vehicle.length_mm
        assert item.y + item.dy <= vehicle.width_mm
        assert item.z + item.dz <= vehicle.height_mm
    for index, left in enumerate(result.placements):
        for right in result.placements[index + 1 :]:
            assert not boxes_overlap(left, right)
        if not left.fragile:
            continue
        for right in result.placements:
            if right.unit_id == left.unit_id:
                continue
            stacked = (
                right.z >= left.z + left.dz
                and right.x < left.x + left.dx
                and left.x < right.x + right.dx
                and right.y < left.y + left.dy
                and left.y < right.y + right.dy
            )
            assert not stacked


def test_tenfold_cartons_pack_in_one_pass() -> None:
    vehicle, units = _pharmacy_units(10)
    assert len(units) == 200
    started = time.perf_counter()
    result = pack_stops(vehicle, [PackStop("store-01", ["n1"], units)], LoadingRule())
    elapsed = time.perf_counter() - started
    assert len(result.placements) == 200
    _assert_packed(vehicle, result)
    assert elapsed < 2


def test_two_stops_of_many_cartons_keep_lifo() -> None:
    vehicle, units = _pharmacy_units(10)
    first = [unit.model_copy(update={"store_id": "s1", "order_id": "o1"}) for unit in units[:100]]
    second = [unit.model_copy(update={"store_id": "s2", "order_id": "o2"}) for unit in units[100:]]
    started = time.perf_counter()
    result = pack_stops(
        vehicle,
        [PackStop("s1", ["n1"], first), PackStop("s2", ["n2"], second)],
        LoadingRule(),
    )
    elapsed = time.perf_counter() - started
    assert len(result.placements) == 200
    _assert_packed(vehicle, result)
    early = [item for item in result.placements if item.store_id == "s1"]
    late = [item for item in result.placements if item.store_id == "s2"]
    for earlier in early:
        for later in late:
            assert not later_blocks_earlier(later, earlier)
    assert elapsed < 2


def test_short_stack_of_one_store_can_carry_the_next() -> None:
    vehicle = _vehicle().model_copy(update={"height_mm": 400})
    result = pack_stops(
        vehicle,
        [
            PackStop("s1", ["n1"], [_unit("early", 600, 400, 200, store="s1")]),
            PackStop("s2", ["n2"], [_unit("late", 600, 400, 200, store="s2")]),
        ],
        LoadingRule(),
    )
    assert result.ok
    early = next(item for item in result.placements if item.store_id == "s1")
    late = next(item for item in result.placements if item.store_id == "s2")
    assert early.z >= late.z + late.dz
    assert not later_blocks_earlier(late, early)
    _assert_packed(vehicle, result)


def test_sparse_stop_leaves_a_platform_for_another_store() -> None:
    vehicle = _vehicle().model_copy(update={"height_mm": 400})
    early_units = [_unit(f"e{index}", 400, 400, 200, store="s1") for index in range(4)]
    result = pack_stops(
        vehicle,
        [
            PackStop("s1", ["n1"], early_units),
            PackStop("s2", ["n2"], [_unit("late", 400, 400, 200, store="s2")]),
        ],
        LoadingRule(),
    )
    assert result.ok
    late = next(item for item in result.placements if item.unit_id == "late")
    parked = [
        item
        for item in result.placements
        if item.store_id == "s1"
        and item.z >= late.z + late.dz
        and item.x < late.x + late.dx
        and late.x < item.x + item.dx
        and item.y < late.y + late.dy
        and late.y < item.y + item.dy
    ]
    assert parked
    for item in result.placements:
        if item.store_id == "s2":
            continue
        assert not later_blocks_earlier(late, item)
    _assert_packed(vehicle, result)


def test_narrow_stop_can_sit_beside_another_store() -> None:
    vehicle = _vehicle().model_copy(update={"length_mm": 500, "width_mm": 800, "height_mm": 400})
    result = pack_stops(
        vehicle,
        [
            PackStop("s1", ["n1"], [_unit("early", 400, 400, 400, store="s1")]),
            PackStop("s2", ["n2"], [_unit("late", 400, 400, 400, store="s2")]),
        ],
        LoadingRule(),
    )
    assert result.ok
    early = next(item for item in result.placements if item.store_id == "s1")
    late = next(item for item in result.placements if item.store_id == "s2")
    assert early.x < late.x + late.dx and late.x < early.x + early.dx
    assert not later_blocks_earlier(late, early)
    _assert_packed(vehicle, result)


def test_other_stores_are_not_stacked_on_fragile() -> None:
    vehicle = _vehicle().model_copy(update={"length_mm": 400, "height_mm": 400})
    result = pack_stops(
        vehicle,
        [
            PackStop("s1", ["n1"], [_unit("solid", 400, 400, 200, store="s1")]),
            PackStop("s2", ["n2"], [_unit("glass", 400, 400, 200, fragile=True, store="s2")]),
        ],
        LoadingRule(),
    )
    assert not result.ok
    assert result.reason == "geometry"

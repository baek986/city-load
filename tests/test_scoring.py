"""Handling lower bound stays below a packed extra_handling score."""

from app.domain.models import LoadingRule, Placement, Weights
from app.domain.units import min_footprint_mm2
from app.optimizer.packing import PackStop, pack_stops
from app.optimizer.scoring import LAYER_MM, extra_handling, extra_layers_lower_bound, extra_layers_lower_bound_for_fleet
from tests.test_packing import _unit, _vehicle


def _place(unit_id: str, z: int, fragile: bool = False) -> Placement:
    return Placement(
        unit_id=unit_id,
        vehicle_id="v",
        store_id="s",
        order_id="o",
        x=0,
        y=0,
        z=z,
        dx=10,
        dy=10,
        dz=10,
        rotation="LWH",
        fragile=fragile,
        stackable=not fragile,
        max_stack_load_g=0 if fragile else 1,
        weight_g=1,
    )


def test_loading_rule_aligns_layers_by_default() -> None:
    rule = LoadingRule()
    assert rule.align_levels
    assert rule.fragile_avoid_floor


def test_extra_layers_count_overflow_floors() -> None:
    assert extra_layers_lower_bound(0, 100) == 0
    assert extra_layers_lower_bound(100, 100) == 0
    assert extra_layers_lower_bound(101, 100) == 1
    assert extra_layers_lower_bound(250, 100) == 2
    vehicle = _vehicle()
    floor = vehicle.length_mm * vehicle.width_mm
    assert extra_layers_lower_bound_for_fleet(floor, [vehicle]) == 0
    assert extra_layers_lower_bound_for_fleet(floor * 2 + 1, [vehicle]) == 2
    assert extra_layers_lower_bound_for_fleet(floor * 2, [_vehicle(), _vehicle()]) == 0


def test_extra_layers_count_volume_overflow() -> None:
    assert extra_layers_lower_bound(50, 100, volume_mm3=1000, layer_height_mm=10) == 0
    assert extra_layers_lower_bound(50, 100, volume_mm3=1500, layer_height_mm=10) == 1
    assert extra_layers_lower_bound(250, 100, volume_mm3=1500, layer_height_mm=10) == 2
    vehicle = _vehicle()
    floor = vehicle.length_mm * vehicle.width_mm
    layer_vol = floor * 100
    assert extra_layers_lower_bound_for_fleet(0, [vehicle], volume_mm3=layer_vol, layer_height_mm=100) == 0
    assert extra_layers_lower_bound_for_fleet(0, [vehicle], volume_mm3=layer_vol + 1, layer_height_mm=100) == 1


def test_nearby_z_share_one_handling_layer() -> None:
    assert extra_handling([_place("a", 0), _place("b", LAYER_MM - 1)]) == 0
    assert extra_handling([_place("a", 0), _place("b", LAYER_MM)]) == 1


def test_mixed_store_fragile_on_floor_still_counts() -> None:
    assert extra_handling([_place("a", 0, fragile=False), _place("b", 0, fragile=True)]) == 1


def test_packed_extras_are_at_least_the_footprint_bound() -> None:
    vehicle = _vehicle().model_copy(update={"length_mm": 800, "width_mm": 400, "height_mm": 800})
    units = [_unit(f"b{index}", 400, 400, 200, store="s1") for index in range(4)]
    result = pack_stops(vehicle, [PackStop("s1", ["n1"], units)], LoadingRule())
    assert result.ok, result.reason
    footprint = sum(min_footprint_mm2(unit) for unit in units)
    floor = vehicle.length_mm * vehicle.width_mm
    volume = sum(unit.volume_mm3 for unit in units)
    layer_h = max(unit.height_mm for unit in units)
    assert extra_handling(result.placements) >= extra_layers_lower_bound(footprint, floor, volume, layer_h)


def test_weights_still_sum_to_one() -> None:
    weights = Weights()
    assert abs(weights.cost + weights.time + weights.late + weights.handling + weights.split - 1.0) < 1e-9

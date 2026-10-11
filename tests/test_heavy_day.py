"""A full 30-store day whose cartons no longer fit in one truck."""

from __future__ import annotations

import time

from app.demo.scenarios import heavy_day
from app.domain.units import build_units, group_units_by_store
from app.optimizer.joint import solve_joint
from app.optimizer.packing import can_pack_nodes
from tests.test_mvp import _problems


def _used_routes(result):
    return [route for route in result.routes if route.stops]


def test_thirty_store_day_needs_several_vehicles() -> None:
    dataset = heavy_day()
    assert len(dataset.stores) == 30
    units = build_units(dataset)
    nodes = group_units_by_store(dataset, units)
    units_by_id = {unit.unit_id: unit for unit in units}
    truck = max(dataset.vehicles, key=lambda vehicle: vehicle.volume_mm3)
    van = min(dataset.vehicles, key=lambda vehicle: vehicle.volume_mm3)
    assert len(units) == 8 * 6 + 12 * 4 + 10 * 3
    assert sum(unit.volume_mm3 for unit in units) > truck.volume_mm3
    assert not can_pack_nodes(nodes, truck, dataset.loading_rule, units_by_id)
    for node in nodes:
        assert can_pack_nodes([node], van, dataset.loading_rule, units_by_id)

    started = time.perf_counter()
    result = solve_joint(dataset, seed=3, time_limit_s=50)
    assert time.perf_counter() - started < 120
    assert result.status == "completed", result.error_message
    assert len(_used_routes(result)) >= 2
    assert result.split_order_count / len(dataset.orders) <= 0.10
    assert _problems(dataset, result) == []
    served = {stop.store_id for route in result.routes for stop in route.stops}
    assert served == {store.id for store in dataset.stores}

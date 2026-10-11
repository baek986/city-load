"""Built-in days the demo can solve without loading master data first."""

from __future__ import annotations

from pathlib import Path

from app.domain.models import Dataset, Location, Order, OrderLine, Sku, Vehicle
from app.network.roads import RoadNetwork

_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "pharmacy_day.json"

# Wholesale cases, about 80–94 cm on the long side. Still a carton, not a pallet.
_BULK = (
    (900, 600, 480, 14_000, False, True, False),
    (860, 560, 460, 12_000, False, True, False),
    (820, 540, 440, 11_000, False, True, False),
    (940, 580, 400, 13_000, False, True, False),
    (780, 520, 420, 8_000, True, True, False),
    (880, 500, 450, 12_000, False, False, True),
    (760, 500, 420, 10_000, False, True, False),
    (840, 540, 400, 11_500, False, True, False),
)

SCENARIOS = (
    {
        "id": "preview",
        "name": "短途预览",
        "detail": "3 家店，沿真实路网走一趟，用来先看路线和车厢。",
        "seed": 1,
        "time_limit_s": 15,
    },
    {
        "id": "heavy",
        "name": "30 店大药箱",
        "detail": "30 家药店，批发箱，早班、全天和午后窗口错开，一辆货车装不下。",
        "seed": 3,
        "time_limit_s": 90,
    },
    {
        "id": "pharmacy",
        "name": "30 店小药箱",
        "detail": "原来的 30 店日配，箱子较小，窗口是 8:00–20:00。",
        "seed": 7,
        "time_limit_s": 90,
    },
)


def scenario_by_id(scenario_id: str) -> dict:
    for scenario in SCENARIOS:
        if scenario["id"] == scenario_id:
            return scenario
    known = ", ".join(item["id"] for item in SCENARIOS)
    raise KeyError(f"Unknown scenario {scenario_id}. Known: {known}")


def pharmacy_day() -> Dataset:
    return Dataset.model_validate_json(_FIXTURE.read_text(encoding="utf-8"))


def _bulk_skus() -> list[Sku]:
    skus: list[Sku] = []
    for index, (length, width, height, weight, fragile, stackable, upright) in enumerate(_BULK):
        skus.append(
            Sku(
                id=f"bulk-{index + 1:02d}",
                name=f"药箱{index + 1:02d}",
                length_mm=length,
                width_mm=width,
                height_mm=height,
                weight_g=weight,
                fragile=fragile,
                stackable=stackable,
                upright_only=upright,
                max_stack_load_g=0 if fragile or upright else 80_000,
            )
        )
    return skus


def _window(index: int) -> tuple[int, int, int]:
    if index % 3 == 0:
        return 8 * 3600, 15 * 3600, 13 * 3600
    if index % 3 == 1:
        return 8 * 3600, 20 * 3600, 18 * 3600
    return 11 * 3600, 20 * 3600, 18 * 3600


def _lines_for(store_index: int, store_id: str, skus: list[Sku]) -> list[OrderLine]:
    if store_index < 8:
        pattern = (2, 2, 2)
    elif store_index < 20:
        pattern = (2, 2)
    else:
        pattern = (1, 1, 1)
    lines = []
    for line_index, qty in enumerate(pattern):
        sku = skus[(store_index + line_index) % len(skus)]
        lines.append(OrderLine(id=f"heavy-{store_id}-{line_index}", sku_id=sku.id, qty=qty))
    return lines


def heavy_day() -> Dataset:
    """All 30 stores. Busy shops take six cartons, light shops three."""
    base = pharmacy_day()
    skus = _bulk_skus()
    stores: list[Location] = []
    orders: list[Order] = []
    for index, store in enumerate(base.stores):
        open_s, close_s, soft_s = _window(index)
        stores.append(
            store.model_copy(update={"window_open_s": open_s, "window_close_s": close_s, "soft_deadline_s": soft_s})
        )
        orders.append(
            Order(
                id=f"order-{store.id}",
                store_id=store.id,
                service_date="2026-10-10",
                lines=_lines_for(index, store.id, skus),
            )
        )
    return base.model_copy(update={"stores": stores, "skus": skus, "orders": orders})


def preview_day() -> Dataset:
    """Three stores on the same simulated streets, small enough to solve immediately."""
    network = RoadNetwork()
    depot_point = network.snap(0, 0)
    depot = Location(
        id="depot",
        name="中心仓",
        kind="depot",
        x_m=depot_point.x_m,
        y_m=depot_point.y_m,
        window_open_s=6 * 3600,
        window_close_s=22 * 3600,
    )
    spots = (("东店", 2400, 800), ("北店", -800, 2800), ("南店", 1200, -2000))
    stores: list[Location] = []
    for index, (name, x_m, y_m) in enumerate(spots):
        point = network.snap(x_m, y_m)
        stores.append(
            Location(
                id=f"preview-{index + 1}",
                name=name,
                kind="store",
                x_m=point.x_m,
                y_m=point.y_m,
                service_time_s=600,
                window_open_s=8 * 3600,
                window_close_s=20 * 3600,
                soft_deadline_s=18 * 3600,
            )
        )
    skus = [
        Sku(id="preview-plain", name="普通药箱", length_mm=500, width_mm=400, height_mm=400, weight_g=8_000),
        Sku(
            id="preview-fragile",
            name="易碎药箱",
            length_mm=480,
            width_mm=360,
            height_mm=300,
            weight_g=4_000,
            fragile=True,
            max_stack_load_g=0,
        ),
    ]
    orders = []
    for index, store in enumerate(stores):
        sku = skus[index % len(skus)]
        orders.append(
            Order(
                id=f"order-{store.id}",
                store_id=store.id,
                service_date="2026-10-10",
                lines=[OrderLine(id=f"line-{store.id}", sku_id=sku.id, qty=4 if index == 0 else 3)],
            )
        )
    vehicles = [
        Vehicle(
            id="van-1",
            type_name="van",
            length_mm=3200,
            width_mm=1700,
            height_mm=1700,
            max_weight_g=1_500_000,
            cost_per_km=4.5,
            fixed_cost=80,
        ),
        Vehicle(
            id="truck-1",
            type_name="truck",
            length_mm=4200,
            width_mm=2100,
            height_mm=2000,
            max_weight_g=3_000_000,
            cost_per_km=6.0,
            fixed_cost=120,
        ),
    ]
    points = [(depot.id, depot.x_m, depot.y_m), *[(store.id, store.x_m, store.y_m) for store in stores]]
    return Dataset(
        depot=depot,
        stores=stores,
        skus=skus,
        orders=orders,
        vehicles=vehicles,
        arcs=network.travel_arcs(points),
    )


def load_scenario(scenario_id: str) -> Dataset:
    scenario_by_id(scenario_id)
    if scenario_id == "preview":
        return preview_day()
    if scenario_id == "heavy":
        return heavy_day()
    if scenario_id == "pharmacy":
        return pharmacy_day()
    raise KeyError(scenario_id)

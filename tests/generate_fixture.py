"""Build the 30-store pharmacy fixture."""

from __future__ import annotations

import json
import math
from pathlib import Path

from app.domain.models import Dataset, LoadingRule, Location, Order, OrderLine, Sku, Vehicle
from app.network.roads import BLOCK_M, RoadNetwork


def build_pharmacy_dataset() -> Dataset:
    depot = Location(
        id="depot",
        name="中心仓",
        kind="depot",
        x_m=0,
        y_m=0,
        service_time_s=0,
        window_open_s=6 * 3600,
        window_close_s=22 * 3600,
    )
    network = RoadNetwork()
    occupied = {(0, 0)}
    stores: list[Location] = []
    for index in range(30):
        angle = 2 * math.pi * index / 30
        radius = 4000 + (index % 5) * 1800
        cell_i = int(round(radius * math.cos(angle) / BLOCK_M))
        cell_j = int(round(radius * math.sin(angle) / BLOCK_M))
        while (cell_i, cell_j) in occupied:
            cell_i += 1
        occupied.add((cell_i, cell_j))
        point = network.snap(cell_i * BLOCK_M, cell_j * BLOCK_M)
        stores.append(
            Location(
                id=f"store-{index + 1:02d}",
                name=f"药店{index + 1:02d}",
                kind="store",
                x_m=point.x_m,
                y_m=point.y_m,
                service_time_s=600,
                window_open_s=8 * 3600,
                window_close_s=20 * 3600,
                soft_deadline_s=18 * 3600,
            )
        )
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
    orders: list[Order] = []
    line_no = 0
    for store_index, store in enumerate(stores):
        count = 14 if store_index < 10 else 13
        lines = []
        for offset in range(count):
            line_no += 1
            lines.append(
                OrderLine(id=f"line-{line_no:04d}", sku_id=skus[(store_index + offset) % len(skus)].id, qty=1)
            )
        orders.append(
            Order(id=f"order-{store.id}", store_id=store.id, service_date="2026-10-10", priority=0, lines=lines)
        )
    vehicles = []
    for index in range(4):
        vehicles.append(
            Vehicle(
                id=f"van-{index + 1}",
                type_name="van",
                length_mm=3200,
                width_mm=1700,
                height_mm=1700,
                max_weight_g=1_500_000,
                cost_per_km=4.5,
                fixed_cost=80,
            )
        )
    for index in range(4):
        vehicles.append(
            Vehicle(
                id=f"truck-{index + 1}",
                type_name="truck",
                length_mm=4200,
                width_mm=2100,
                height_mm=2000,
                max_weight_g=3_000_000,
                cost_per_km=6.0,
                fixed_cost=120,
            )
        )
    arcs = network.travel_arcs(
        [(depot.id, depot.x_m, depot.y_m), *[(store.id, store.x_m, store.y_m) for store in stores]]
    )
    return Dataset(
        depot=depot,
        stores=stores,
        skus=skus,
        orders=orders,
        vehicles=vehicles,
        arcs=arcs,
        loading_rule=LoadingRule(),
    )


def main() -> None:
    dataset = build_pharmacy_dataset()
    path = Path(__file__).resolve().parent / "fixtures" / "pharmacy_day.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {path} orders={len(dataset.orders)} lines={sum(len(o.lines) for o in dataset.orders)} arcs={len(dataset.arcs)}")


if __name__ == "__main__":
    main()

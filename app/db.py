"""SQLAlchemy persistence. SQLite is the default; PostgreSQL works with the same schema."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from app.domain.models import (
    Dataset,
    JointResult,
    LoadingRule,
    Location,
    Order,
    OrderLine,
    Sku,
    TravelArc,
    Vehicle,
    Weights,
)


class Base(DeclarativeBase):
    pass


class DepotRow(Base):
    __tablename__ = "depot"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    x_m: Mapped[float] = mapped_column(Float)
    y_m: Mapped[float] = mapped_column(Float)
    service_time_s: Mapped[int] = mapped_column(Integer, default=0)
    window_open_s: Mapped[int] = mapped_column(Integer)
    window_close_s: Mapped[int] = mapped_column(Integer)
    soft_deadline_s: Mapped[int | None] = mapped_column(Integer, nullable=True)


class StoreRow(Base):
    __tablename__ = "stores"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    x_m: Mapped[float] = mapped_column(Float)
    y_m: Mapped[float] = mapped_column(Float)
    service_time_s: Mapped[int] = mapped_column(Integer, default=0)
    window_open_s: Mapped[int] = mapped_column(Integer)
    window_close_s: Mapped[int] = mapped_column(Integer)
    soft_deadline_s: Mapped[int | None] = mapped_column(Integer, nullable=True)


class SkuRow(Base):
    __tablename__ = "skus"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    length_mm: Mapped[int] = mapped_column(Integer)
    width_mm: Mapped[int] = mapped_column(Integer)
    height_mm: Mapped[int] = mapped_column(Integer)
    weight_g: Mapped[int] = mapped_column(Integer)
    allowed_rotations: Mapped[str] = mapped_column(Text)
    fragile: Mapped[int] = mapped_column(Integer, default=0)
    stackable: Mapped[int] = mapped_column(Integer, default=1)
    max_stack_load_g: Mapped[int] = mapped_column(Integer)
    upright_only: Mapped[int] = mapped_column(Integer, default=0)


class VehicleRow(Base):
    __tablename__ = "vehicles"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    type_name: Mapped[str] = mapped_column(String)
    length_mm: Mapped[int] = mapped_column(Integer)
    width_mm: Mapped[int] = mapped_column(Integer)
    height_mm: Mapped[int] = mapped_column(Integer)
    max_weight_g: Mapped[int] = mapped_column(Integer)
    cost_per_km: Mapped[float] = mapped_column(Float)
    fixed_cost: Mapped[float] = mapped_column(Float)
    door: Mapped[str] = mapped_column(String, default="rear")


class OrderRow(Base):
    __tablename__ = "orders"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    store_id: Mapped[str] = mapped_column(String, index=True)
    service_date: Mapped[str] = mapped_column(String, index=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)


class OrderLineRow(Base):
    __tablename__ = "order_lines"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    order_id: Mapped[str] = mapped_column(String, index=True)
    sku_id: Mapped[str] = mapped_column(String)
    qty: Mapped[int] = mapped_column(Integer)


class TravelArcRow(Base):
    __tablename__ = "travel_arcs"
    __table_args__ = (UniqueConstraint("from_id", "to_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    from_id: Mapped[str] = mapped_column(String, index=True)
    to_id: Mapped[str] = mapped_column(String, index=True)
    distance_m: Mapped[int] = mapped_column(Integer)
    travel_time_s: Mapped[int] = mapped_column(Integer)


class LoadingRuleRow(Base):
    __tablename__ = "loading_rule"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    body: Mapped[str] = mapped_column(Text)


class PlanRow(Base):
    __tablename__ = "plans"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    service_date: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, index=True)
    seed: Mapped[int] = mapped_column(Integer)
    weights_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)


def make_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite") and ":memory:" in database_url:
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    if database_url.startswith("sqlite"):
        return create_engine(database_url, connect_args={"check_same_thread": False})
    return create_engine(database_url)


def _location_from_store(row: StoreRow) -> Location:
    return Location(
        id=row.id,
        name=row.name,
        kind="store",
        x_m=row.x_m,
        y_m=row.y_m,
        service_time_s=row.service_time_s,
        window_open_s=row.window_open_s,
        window_close_s=row.window_close_s,
        soft_deadline_s=row.soft_deadline_s,
    )


class Repository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def set_depot(self, depot: Location) -> Location:
        row = self.session.get(DepotRow, depot.id)
        if row is None:
            row = DepotRow(
                id=depot.id,
                name=depot.name,
                x_m=depot.x_m,
                y_m=depot.y_m,
                service_time_s=depot.service_time_s,
                window_open_s=depot.window_open_s,
                window_close_s=depot.window_close_s,
                soft_deadline_s=depot.soft_deadline_s,
            )
            self.session.add(row)
        else:
            row.name = depot.name
            row.x_m = depot.x_m
            row.y_m = depot.y_m
            row.service_time_s = depot.service_time_s
            row.window_open_s = depot.window_open_s
            row.window_close_s = depot.window_close_s
            row.soft_deadline_s = depot.soft_deadline_s
        self.session.flush()
        return depot.model_copy(update={"kind": "depot"})

    def get_depot(self) -> Location | None:
        row = self.session.scalars(select(DepotRow)).first()
        if row is None:
            return None
        return Location(
            id=row.id,
            name=row.name,
            kind="depot",
            x_m=row.x_m,
            y_m=row.y_m,
            service_time_s=row.service_time_s,
            window_open_s=row.window_open_s,
            window_close_s=row.window_close_s,
            soft_deadline_s=row.soft_deadline_s,
        )

    def add_store(self, store: Location) -> Location:
        self.session.add(
            StoreRow(
                id=store.id,
                name=store.name,
                x_m=store.x_m,
                y_m=store.y_m,
                service_time_s=store.service_time_s,
                window_open_s=store.window_open_s,
                window_close_s=store.window_close_s,
                soft_deadline_s=store.soft_deadline_s,
            )
        )
        self.session.flush()
        return store.model_copy(update={"kind": "store"})

    def list_stores(self) -> list[Location]:
        return [_location_from_store(row) for row in self.session.scalars(select(StoreRow).order_by(StoreRow.id))]

    def delete_store(self, store_id: str) -> bool:
        row = self.session.get(StoreRow, store_id)
        if row is None:
            return False
        self.session.delete(row)
        return True

    def add_sku(self, sku: Sku) -> Sku:
        self.session.add(
            SkuRow(
                id=sku.id,
                name=sku.name,
                length_mm=sku.length_mm,
                width_mm=sku.width_mm,
                height_mm=sku.height_mm,
                weight_g=sku.weight_g,
                allowed_rotations=json.dumps(sku.allowed_rotations),
                fragile=int(sku.fragile),
                stackable=int(sku.stackable),
                max_stack_load_g=sku.max_stack_load_g,
                upright_only=int(sku.upright_only),
            )
        )
        self.session.flush()
        return sku

    def list_skus(self) -> list[Sku]:
        rows = self.session.scalars(select(SkuRow).order_by(SkuRow.id))
        return [
            Sku(
                id=row.id,
                name=row.name,
                length_mm=row.length_mm,
                width_mm=row.width_mm,
                height_mm=row.height_mm,
                weight_g=row.weight_g,
                allowed_rotations=json.loads(row.allowed_rotations),
                fragile=bool(row.fragile),
                stackable=bool(row.stackable),
                max_stack_load_g=row.max_stack_load_g,
                upright_only=bool(row.upright_only),
            )
            for row in rows
        ]

    def delete_sku(self, sku_id: str) -> bool:
        row = self.session.get(SkuRow, sku_id)
        if row is None:
            return False
        self.session.delete(row)
        return True

    def add_vehicle(self, vehicle: Vehicle) -> Vehicle:
        self.session.add(VehicleRow(**vehicle.model_dump()))
        self.session.flush()
        return vehicle

    def list_vehicles(self) -> list[Vehicle]:
        rows = self.session.scalars(select(VehicleRow).order_by(VehicleRow.id))
        return [Vehicle.model_validate(row, from_attributes=True) for row in rows]

    def delete_vehicle(self, vehicle_id: str) -> bool:
        row = self.session.get(VehicleRow, vehicle_id)
        if row is None:
            return False
        self.session.delete(row)
        return True

    def add_order(self, order: Order) -> Order:
        self.session.add(
            OrderRow(id=order.id, store_id=order.store_id, service_date=order.service_date, priority=order.priority)
        )
        for line in order.lines:
            self.session.add(OrderLineRow(id=line.id, order_id=order.id, sku_id=line.sku_id, qty=line.qty))
        self.session.flush()
        return order

    def list_orders(self, service_date: str | None = None) -> list[Order]:
        query = select(OrderRow).order_by(OrderRow.id)
        if service_date is not None:
            query = query.where(OrderRow.service_date == service_date)
        orders = []
        for row in self.session.scalars(query):
            lines = self.session.scalars(select(OrderLineRow).where(OrderLineRow.order_id == row.id)).all()
            orders.append(
                Order(
                    id=row.id,
                    store_id=row.store_id,
                    service_date=row.service_date,
                    priority=row.priority,
                    lines=[OrderLine(id=line.id, sku_id=line.sku_id, qty=line.qty) for line in lines],
                )
            )
        return orders

    def delete_order(self, order_id: str) -> bool:
        row = self.session.get(OrderRow, order_id)
        if row is None:
            return False
        for line in self.session.scalars(select(OrderLineRow).where(OrderLineRow.order_id == order_id)):
            self.session.delete(line)
        self.session.delete(row)
        return True

    def add_arc(self, arc: TravelArc) -> TravelArc:
        existing = self.session.scalars(
            select(TravelArcRow).where(TravelArcRow.from_id == arc.from_id, TravelArcRow.to_id == arc.to_id)
        ).first()
        if existing is None:
            self.session.add(
                TravelArcRow(
                    from_id=arc.from_id,
                    to_id=arc.to_id,
                    distance_m=arc.distance_m,
                    travel_time_s=arc.travel_time_s,
                )
            )
        else:
            existing.distance_m = arc.distance_m
            existing.travel_time_s = arc.travel_time_s
        self.session.flush()
        return arc

    def list_arcs(self) -> list[TravelArc]:
        rows = self.session.scalars(select(TravelArcRow))
        return [
            TravelArc(from_id=row.from_id, to_id=row.to_id, distance_m=row.distance_m, travel_time_s=row.travel_time_s)
            for row in rows
        ]

    def set_loading_rule(self, rule: LoadingRule) -> LoadingRule:
        row = self.session.get(LoadingRuleRow, 1)
        body = rule.model_dump_json()
        if row is None:
            self.session.add(LoadingRuleRow(id=1, body=body))
        else:
            row.body = body
        self.session.flush()
        return rule

    def get_loading_rule(self) -> LoadingRule:
        row = self.session.get(LoadingRuleRow, 1)
        if row is None:
            return LoadingRule()
        return LoadingRule.model_validate_json(row.body)

    def load_dataset(self, service_date: str) -> Dataset:
        depot = self.get_depot()
        if depot is None:
            raise ValueError("Depot is not configured")
        return Dataset(
            depot=depot,
            stores=self.list_stores(),
            skus=self.list_skus(),
            orders=self.list_orders(service_date),
            vehicles=self.list_vehicles(),
            arcs=self.list_arcs(),
            loading_rule=self.get_loading_rule(),
        )

    def create_plan(self, service_date: str, weights: Weights, seed: int) -> str:
        plan_id = uuid.uuid4().hex
        self.session.add(
            PlanRow(
                id=plan_id,
                service_date=service_date,
                status="running",
                seed=seed,
                weights_json=weights.model_dump_json(),
            )
        )
        self.session.flush()
        return plan_id

    def save_result(self, plan_id: str, result: JointResult) -> None:
        row = self.session.get(PlanRow, plan_id)
        if row is None:
            raise KeyError(plan_id)
        row.status = result.status
        row.result_json = result.model_dump_json()
        self.session.flush()

    def get_plan_row(self, plan_id: str) -> PlanRow | None:
        return self.session.get(PlanRow, plan_id)

    def get_result(self, plan_id: str) -> JointResult | None:
        row = self.get_plan_row(plan_id)
        if row is None or not row.result_json:
            return None
        return JointResult.model_validate_json(row.result_json)

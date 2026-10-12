"""HTTP schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.models import Weights


class PlanCreate(BaseModel):
    date: str
    weights: Weights = Field(default_factory=Weights)
    seed: int = 1
    time_limit_s: float = 180.0
    prove_optimal: bool = False


class DemoSolve(BaseModel):
    scenario: str = "heavy"
    seed: int = 3
    time_limit_s: float = 90.0


class ReplanCreate(BaseModel):
    locked_order_ids: list[str] = Field(default_factory=list)
    locked_vehicle_ids: list[str] = Field(default_factory=list)
    weights: Weights | None = None
    seed: int | None = None
    time_limit_s: float = 180.0
    prove_optimal: bool = False


class CustomSelection(BaseModel):
    """One SKU line chosen by the user: which SKU and how many per store."""

    sku_id: str
    qty: int = 1


class CustomStorePick(BaseModel):
    """A store the user selected, with the SKU lines to deliver."""

    store_id: str
    lines: list[CustomSelection] = Field(default_factory=list)


class CustomSolve(BaseModel):
    """Solve a custom day built from user-selected stores and SKU lines."""

    stores: list[CustomStorePick]
    seed: int = 1
    time_limit_s: float = 30.0


class TaskCreate(BaseModel):
    """Create a background solve task for a preset or custom scenario."""

    scenario: str = "heavy"
    seed: int | None = None
    time_limit_s: float | None = None
    custom: CustomSolve | None = None

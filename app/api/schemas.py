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

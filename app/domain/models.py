"""Pydantic entities. Volume is always derived from length, width, and height."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Location(BaseModel):
    id: str
    name: str
    kind: Literal["depot", "store"]
    x_m: float
    y_m: float
    service_time_s: int = 0
    window_open_s: int = 0
    window_close_s: int = 86_400
    soft_deadline_s: int | None = None


class Sku(BaseModel):
    id: str
    name: str
    length_mm: int
    width_mm: int
    height_mm: int
    weight_g: int
    allowed_rotations: list[str] = Field(
        default_factory=lambda: ["LWH", "WLH", "LHW", "HLW", "WHL", "HWL"]
    )
    fragile: bool = False
    stackable: bool = True
    max_stack_load_g: int = 1_000_000_000
    upright_only: bool = False

    @property
    def volume_mm3(self) -> int:
        return self.length_mm * self.width_mm * self.height_mm


class OrderLine(BaseModel):
    id: str
    sku_id: str
    qty: int

    @model_validator(mode="after")
    def qty_positive(self) -> "OrderLine":
        if self.qty < 1:
            raise ValueError("qty must be at least 1")
        return self


class Order(BaseModel):
    id: str
    store_id: str
    service_date: str
    priority: int = 0
    lines: list[OrderLine]


class Vehicle(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    type_name: str
    length_mm: int
    width_mm: int
    height_mm: int
    max_weight_g: int
    cost_per_km: float
    fixed_cost: float
    door: str = "rear"

    @property
    def volume_mm3(self) -> int:
        return self.length_mm * self.width_mm * self.height_mm


class TravelArc(BaseModel):
    from_id: str
    to_id: str
    distance_m: int
    travel_time_s: int


class LoadingRule(BaseModel):
    min_support_ratio: float = 0.75
    max_overhang_mm: int = 0
    fragile_no_stack: bool = True
    split_penalty: float = 1.0
    # Packing preferences used by the per-item placer. On so extra_handling
    # stays close to geometric layers instead of counting every jagged z.
    # fragile_avoid_floor: try every non-floor pose for fragile units before z=0.
    # align_levels: prefer z-levels that already hold many boxes.
    fragile_avoid_floor: bool = True
    align_levels: bool = True


class Weights(BaseModel):
    cost: float = 0.45
    time: float = 0.35
    late: float = 0.10
    handling: float = 0.05
    split: float = 0.05


class Dataset(BaseModel):
    depot: Location
    stores: list[Location]
    skus: list[Sku]
    orders: list[Order]
    vehicles: list[Vehicle]
    arcs: list[TravelArc]
    loading_rule: LoadingRule = Field(default_factory=LoadingRule)


class PackingUnit(BaseModel):
    unit_id: str
    order_id: str
    order_line_id: str
    store_id: str
    sku_id: str
    length_mm: int
    width_mm: int
    height_mm: int
    weight_g: int
    fragile: bool
    stackable: bool
    max_stack_load_g: int
    upright_only: bool
    allowed_rotations: list[str]

    @property
    def volume_mm3(self) -> int:
        return self.length_mm * self.width_mm * self.height_mm


class CustomerNode(BaseModel):
    node_id: str
    store_id: str
    unit_ids: list[str]
    order_ids: list[str]
    weight_g: int
    volume_mm3: int
    service_time_s: int
    window_open_s: int
    window_close_s: int
    soft_deadline_s: int | None = None
    footprint_mm2: int = 0
    max_extent_mm: int = 0


class StopPlan(BaseModel):
    store_id: str
    node_ids: list[str]
    unit_ids: list[str]
    arrival_s: int
    depart_s: int
    late_minutes: float
    sequence: int


class RoutePlan(BaseModel):
    vehicle_id: str
    stops: list[StopPlan]
    distance_m: int
    travel_time_s: int
    travel_cost: float
    fixed_cost: float
    depart_depot_s: int
    return_depot_s: int
    total_weight_g: int
    total_volume_mm3: int


class Placement(BaseModel):
    unit_id: str
    vehicle_id: str
    store_id: str
    order_id: str
    x: int
    y: int
    z: int
    dx: int
    dy: int
    dz: int
    rotation: str
    fragile: bool
    stackable: bool
    max_stack_load_g: int
    weight_g: int
    load_on_top_g: int = 0


class StepPlacement(BaseModel):
    unit_id: str
    x: int
    y: int
    z: int
    dx: int
    dy: int
    dz: int
    rotation: str


class HandlingStep(BaseModel):
    step_no: int
    action: Literal["load", "unload"]
    vehicle_id: str
    store_id: str
    unit_ids: list[str]
    placements: list[StepPlacement]


class VehicleStat(BaseModel):
    vehicle_id: str
    used: bool
    volume_ratio: float
    weight_ratio: float
    distance_m: int


class Scales(BaseModel):
    cost: float
    time: float
    late: float
    handling: float
    split: float


class JointResult(BaseModel):
    status: Literal["completed", "failed"]
    proven_optimal: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    gap: float | None = None
    bound_established: bool = False
    score: float | None = None
    split_order_count: int = 0
    on_time_rate: float = 1.0
    iteration_limit_reached: bool = False
    rounds: int = 0
    reshuffle_count: int = 0
    routes: list[RoutePlan] = Field(default_factory=list)
    placements: list[Placement] = Field(default_factory=list)
    steps: list[HandlingStep] = Field(default_factory=list)
    vehicle_stats: list[VehicleStat] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    infeasible_stores: list[str] = Field(default_factory=list)
    failed_unit_ids: list[str] = Field(default_factory=list)
    model_proposal_accepted: bool | None = None
    model_proposal_note: str | None = None

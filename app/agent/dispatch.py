"""Turn one dispatch sentence into a single replan.

This interpreter is rules, not a model. It only accepts locks, one weight
bias, quantity edits, and cancellations. A sentence it does not understand
is rejected. Route drafts come from the model slot in app.agent.proposal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.llm import LlmSettings
from app.db import Repository
from app.domain.models import Dataset, JointResult, Weights
from app.optimizer.joint import replan

BiasName = Literal["on_time", "fewer_vehicles", "less_split", "less_handling"]

_BIASES: dict[BiasName, Weights] = {
    "on_time": Weights(cost=0.25, time=0.30, late=0.35, handling=0.05, split=0.05),
    "fewer_vehicles": Weights(cost=0.70, time=0.15, late=0.05, handling=0.05, split=0.05),
    "less_split": Weights(cost=0.35, time=0.25, late=0.05, handling=0.05, split=0.30),
    "less_handling": Weights(cost=0.35, time=0.25, late=0.10, handling=0.25, split=0.05),
}
_BIAS_LABEL: dict[BiasName, str] = {
    "on_time": "更准时",
    "fewer_vehicles": "少出车",
    "less_split": "少拆单",
    "less_handling": "更好卸",
}
_BIAS_RULES: tuple[tuple[BiasName, re.Pattern[str]], ...] = (
    ("on_time", re.compile(r"准时|别迟到|不要迟到|赶时间")),
    ("fewer_vehicles", re.compile(r"少出车|少用车|少派车|省车|降低成本")),
    ("less_split", re.compile(r"别再?拆|不要拆|少拆|不拆单")),
    ("less_handling", re.compile(r"好卸|少装卸|装卸少|轻装卸")),
)
_CN_QTY = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_LOCK_WORDS = re.compile(r"出门|出发|别动|不要动|锁住|锁定")


class DispatchError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class LineAdd(BaseModel):
    model_config = ConfigDict(extra="ignore")

    store: str
    sku: str | None = None
    qty: int

    @field_validator("sku", mode="before")
    @classmethod
    def blank_sku(cls, value: object) -> object:
        if value in ("", "null", None):
            return None
        return value

    @field_validator("qty")
    @classmethod
    def qty_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("qty must be at least 1")
        return value


class RawDispatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    locked_vehicles: list[str] = Field(default_factory=list)
    locked_orders: list[str] = Field(default_factory=list)
    weight_bias: BiasName | None = None
    add_lines: list[LineAdd] = Field(default_factory=list)
    cancel_stores: list[str] = Field(default_factory=list)
    summary: str = ""
    error: str | None = None

    @field_validator("locked_vehicles", "locked_orders", "cancel_stores", "add_lines", mode="before")
    @classmethod
    def null_list(cls, value: object) -> object:
        if value is None:
            return []
        return value

    @field_validator("weight_bias", mode="before")
    @classmethod
    def blank_bias(cls, value: object) -> object:
        if value in ("", "null", None):
            return None
        return value


@dataclass
class PreparedDispatch:
    interpreter: str
    summary: str
    weights: Weights
    locked_order_ids: list[str]
    locked_vehicle_ids: list[str]
    qty_edits: list[tuple[str, str | None, int]]
    cancel_order_ids: list[str]

    def public(self) -> dict:
        return {
            "interpreter": self.interpreter,
            "summary": self.summary,
            "weights": self.weights.model_dump(),
            "locked_vehicle_ids": self.locked_vehicle_ids,
            "locked_order_ids": self.locked_order_ids,
        }


def interpret(
    text: str,
    dataset: Dataset | None = None,
    settings: LlmSettings | None = None,
    complete: Callable[[LlmSettings, str, str], dict] | None = None,
) -> tuple[RawDispatch, str]:
    del dataset, settings, complete
    return interpret_locally(text), "cursor"


def interpret_locally(text: str) -> RawDispatch:
    """Rule interpreter for one dispatch sentence. It does not call a model."""
    raw = RawDispatch()
    biases: list[BiasName] = []
    for clause in _clauses(text):
        if not clause or _noise(clause):
            continue
        store = _store_ref(clause)
        vehicle = _vehicle_ref(clause)
        for name, pattern in _BIAS_RULES:
            if pattern.search(clause) and name not in biases:
                biases.append(name)
        if "取消" in clause and store is not None:
            raw.cancel_stores.append(store)
            continue
        added = _line_add(clause, store)
        if added is not None:
            raw.add_lines.append(added)
            continue
        if vehicle is not None and _LOCK_WORDS.search(clause):
            raw.locked_vehicles.append(vehicle)
        elif store is not None and re.search(r"锁|别动|不要动", clause):
            raw.locked_orders.append(store)
    if len(biases) > 1:
        raw.error = "一次只接受一种偏好：更准时、少出车、少拆单或更好卸"
    elif biases:
        raw.weight_bias = biases[0]
    if not _actionable(raw) and raw.error is None:
        raw.error = "没有听出锁车、偏好或改单。例如：v2已经出门，其余更准时"
    return raw


def resolve_dispatch(
    raw: RawDispatch,
    dataset: Dataset,
    current_weights: Weights,
    store_on_vehicle: dict[str, str],
) -> PreparedDispatch:
    if raw.error:
        raise DispatchError("not_understood", raw.error)
    if not _actionable(raw):
        raise DispatchError("not_understood", "没有听出锁车、偏好或改单。例如：v2已经出门，其余更准时")

    vehicles = [(vehicle.id, vehicle.id) for vehicle in dataset.vehicles]
    stores = [(store.id, store.name) for store in dataset.stores]
    skus = [(sku.id, sku.name) for sku in dataset.skus]
    orders_by_store: dict[str, list[str]] = {}
    for order in dataset.orders:
        orders_by_store.setdefault(order.store_id, []).append(order.id)

    locked_vehicle_ids = _unique(_require_one(ref, _match_numbered(ref, vehicles), "车辆") for ref in raw.locked_vehicles)
    locked_stores = {store_id for store_id, vehicle_id in store_on_vehicle.items() if vehicle_id in locked_vehicle_ids}

    locked_order_ids: list[str] = []
    for ref in raw.locked_orders:
        if any(order.id == ref for order in dataset.orders):
            locked_order_ids.append(ref)
            continue
        store_id = _require_one(ref, _match_numbered(ref, stores), "门店")
        found = orders_by_store.get(store_id, [])
        if not found:
            raise DispatchError("unknown_order", f"门店 {store_id} 没有订单")
        locked_order_ids.extend(found)
    locked_order_ids = _unique(locked_order_ids)

    cancel_order_ids: list[str] = []
    for ref in raw.cancel_stores:
        store_id = _require_one(ref, _match_numbered(ref, stores), "门店")
        _reject_frozen_store(store_id, store_on_vehicle, locked_vehicle_ids, "取消")
        found = orders_by_store.get(store_id, [])
        if not found:
            raise DispatchError("unknown_order", f"门店 {store_id} 没有订单")
        cancel_order_ids.extend(found)

    qty_edits: list[tuple[str, str | None, int]] = []
    for line in raw.add_lines:
        store_id = _require_one(line.store, _match_numbered(line.store, stores), "门店")
        _reject_frozen_store(store_id, store_on_vehicle, locked_vehicle_ids, "改数量")
        found = orders_by_store.get(store_id, [])
        if len(found) != 1:
            raise DispatchError("ambiguous_order", f"门店 {store_id} 不是恰好一张订单，不能直接加箱")
        sku_id = None if line.sku is None else _require_one(line.sku, _match_numbered(line.sku, skus), "货品")
        qty_edits.append((found[0], sku_id, line.qty))

    if set(cancel_order_ids) & set(locked_order_ids):
        raise DispatchError("conflict", "同一张订单不能又锁定又取消")
    if set(cancel_order_ids) & {order_id for order_id, _sku_id, _qty in qty_edits}:
        raise DispatchError("conflict", "同一张订单不能又加箱又取消")

    weights = _BIASES[raw.weight_bias].model_copy() if raw.weight_bias is not None else current_weights
    return PreparedDispatch(
        interpreter="",
        summary=_summary(locked_vehicle_ids, locked_order_ids, raw.weight_bias, qty_edits, cancel_order_ids),
        weights=weights,
        locked_order_ids=locked_order_ids,
        locked_vehicle_ids=locked_vehicle_ids,
        qty_edits=qty_edits,
        cancel_order_ids=_unique(cancel_order_ids),
    )


def prepare_dispatch(
    repo: Repository,
    plan_id: str,
    text: str,
    settings: LlmSettings | None = None,
) -> tuple[PreparedDispatch, JointResult, str, int]:
    row = repo.get_plan_row(plan_id)
    if row is None:
        raise DispatchError("not_found", "找不到这张计划", status_code=404)
    previous = repo.get_result(plan_id)
    if previous is None or previous.status != "completed":
        raise DispatchError("plan_not_ready", "计划还没有可用结果", status_code=409)
    dataset = repo.load_dataset(row.service_date)
    raw, interpreter = interpret(text, dataset, settings)
    current = Weights.model_validate_json(row.weights_json)
    prepared = resolve_dispatch(raw, dataset, current, _store_on_vehicle(previous))
    prepared.interpreter = interpreter
    return prepared, previous, row.service_date, row.seed


def apply_edits(repo: Repository, prepared: PreparedDispatch) -> None:
    for order_id in prepared.cancel_order_ids:
        repo.delete_order(order_id)
    for order_id, sku_id, qty in prepared.qty_edits:
        repo.increase_order_qty(order_id, sku_id, qty)


def run_dispatch(
    repo: Repository,
    plan_id: str,
    text: str,
    *,
    time_limit_s: float = 180.0,
    prove_optimal: bool = False,
    settings: LlmSettings | None = None,
) -> dict:
    prepared, previous, service_date, seed = prepare_dispatch(repo, plan_id, text, settings)
    apply_edits(repo, prepared)
    dataset = repo.load_dataset(service_date)
    from app.agent.proposal import model_proposal_enabled

    result = replan(
        dataset,
        previous,
        locked_order_ids=prepared.locked_order_ids,
        weights=prepared.weights,
        seed=seed,
        time_limit_s=time_limit_s,
        locked_vehicle_ids=prepared.locked_vehicle_ids,
        prove_optimal=prove_optimal,
        model_proposal=model_proposal_enabled(),
    )
    new_id = repo.create_plan(service_date, prepared.weights, seed)
    repo.save_result(new_id, result)
    body = prepared.public()
    body["plan_id"] = new_id
    body["status"] = result.status
    body["score"] = result.score
    body["error_code"] = result.error_code
    body["error_message"] = result.error_message
    return body


def _clauses(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[，,。；;\n]|其余|剩下", text) if part.strip()]


def _noise(clause: str) -> bool:
    return clause in {"其余", "剩下", "其他", "然后", "并且", "同时", "一下", "今天"}


def _vehicle_ref(clause: str) -> str | None:
    match = re.search(r"van-(\d+)", clause, flags=re.IGNORECASE)
    if match:
        return str(int(match.group(1)))
    match = re.search(r"(?<![A-Za-z0-9])v(\d+)(?!\d)", clause, flags=re.IGNORECASE)
    if match:
        return str(int(match.group(1)))
    match = re.search(r"车辆?\s*(\d+)", clause)
    if match:
        return str(int(match.group(1)))
    if re.search(r"(?<![A-Za-z0-9])van(?![A-Za-z0-9])", clause, flags=re.IGNORECASE):
        return "van"
    return None


def _store_ref(clause: str) -> str | None:
    match = re.search(r"(?:药店|门店)\s*(\d+)", clause)
    if match:
        return match.group(1)
    match = re.search(r"store-(\d+)", clause, flags=re.IGNORECASE)
    if match:
        return f"store-{int(match.group(1))}"
    match = re.search(r"(?<![A-Za-z0-9])s(\d+)(?!\d)", clause, flags=re.IGNORECASE)
    if match:
        return f"s{int(match.group(1))}"
    return None


def _line_add(clause: str, store: str | None) -> LineAdd | None:
    if "加" not in clause or store is None:
        return None
    match = re.search(r"加上?\s*(\d+|[一两二三四五六七八九十])\s*箱(?:\s*([^\s，,。；;]+))?", clause)
    if match is None:
        return None
    token = match.group(1)
    qty = int(token) if token.isdigit() else _CN_QTY[token]
    sku = match.group(2)
    if sku in {None, "", "的", "了", "吧"}:
        sku = None
    return LineAdd(store=store, sku=sku, qty=qty)


def _actionable(raw: RawDispatch) -> bool:
    return bool(raw.locked_vehicles or raw.locked_orders or raw.weight_bias or raw.add_lines or raw.cancel_stores)


def _match_numbered(ref: str, items: list[tuple[str, str]]) -> list[str]:
    exact = [item_id for item_id, name in items if item_id == ref or name == ref]
    if exact:
        return _unique(exact)
    number = _number_of(ref)
    if number is None:
        return []
    hits = [item_id for item_id, name in items if _tail_number(item_id) == number or _tail_number(name) == number]
    return _unique(hits)


def _number_of(ref: str) -> int | None:
    if re.fullmatch(r"\d+", ref):
        return int(ref)
    match = re.fullmatch(r"v(?:an)?-?(\d+)", ref, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _tail_number(value: str) -> int | None:
    match = re.search(r"(\d+)$", value)
    if match is None:
        return None
    return int(match.group(1))


def _require_one(ref: str, hits: list[str], kind: str) -> str:
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise DispatchError("unknown_ref", f"没有找到{kind} {ref}")
    raise DispatchError("ambiguous_ref", f"{kind} {ref} 对应多条记录")


def _reject_frozen_store(store_id: str, store_on_vehicle: dict[str, str], locked_vehicle_ids: list[str], action: str) -> None:
    vehicle_id = store_on_vehicle.get(store_id)
    if vehicle_id is not None and vehicle_id in locked_vehicle_ids:
        raise DispatchError("frozen_store", f"门店 {store_id} 在已锁定的 {vehicle_id} 上，不能{action}")


def _store_on_vehicle(result: JointResult) -> dict[str, str]:
    return {stop.store_id: route.vehicle_id for route in result.routes for stop in route.stops}


def _summary(
    vehicles: list[str],
    orders: list[str],
    bias: BiasName | None,
    edits: list[tuple[str, str | None, int]],
    cancelled: list[str],
) -> str:
    parts: list[str] = []
    if vehicles:
        parts.append("锁住 " + "、".join(vehicles))
    if orders:
        parts.append("订单留在原车 " + "、".join(orders))
    if bias is not None:
        parts.append("偏好" + _BIAS_LABEL[bias])
    for order_id, sku_id, qty in edits:
        goods = order_id if sku_id is None else f"{order_id} 的 {sku_id}"
        parts.append(f"{goods} 加 {qty} 箱")
    if cancelled:
        parts.append("取消 " + "、".join(cancelled))
    return "；".join(parts)


def _unique(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen

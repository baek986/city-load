"""Ask the configured model for a first route draft.

The draft is only an assignment of stops to vehicles, in unload order.
Time windows and the packer decide whether it can be the incumbent.
The solver may still change it. No key is stored in the repository.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from app.agent.llm import LlmSettings, ModelCallError, complete_json
from app.domain.errors import MissingTravelArcError
from app.domain.matrix import TravelMatrix
from app.domain.models import CustomerNode, Dataset, PackingUnit, Weights
from app.optimizer.packing import pack_stops, stops_from_nodes

_SYSTEM = """你是配送调度员，只给一版初始路线，不声称最优。
用户消息里是求解器用的同一份原数据，没有做汇总。时间都是从当天 0 点起的秒。
只输出一个 JSON 对象，不要解释，不要写坐标：
{"routes":[{"vehicle_id":"给定的车辆id","stops":["门店id", "..."]}]}
规则：
- 每个停靠点恰好出现一次
- stops 从左到右是卸货顺序，先写的先卸
- 不用的车可以不写
- vehicle_id 和门店 id 必须来自数据，不要编造
- fixed_vehicle_by_node 里的店必须留在指定车上
- 从仓库 window_open_s 出发。每一段用 arcs 里对应的 travel_time_s。到达早于门店 window_open_s 就等到窗开；晚于 window_close_s 则这条路线不可行。卸货加上该店 service_time_s，再去下一家。最后用返回仓库的弧，回到仓库的时刻不能晚于仓库 window_close_s
- 订单每一行是 qty 只该 SKU 的箱子，长宽高和重量在 skus 里。同车的箱子总重不得超过车辆 max_weight_g，并且要能放进车厢 length_mm、width_mm、height_mm
- 车厢门在 x=0。后卸的货先装、靠里；后卸的箱子不能压在先卸的上面，也不能挡在先卸的和门之间。箱子底面至少要有 loading_rule.min_support_ratio 的面积被支撑。fragile_no_stack 为真时，易碎箱上面不能再放。upright_only 的箱子不能倒放
- 分数按 weights 权衡出车费、里程、行驶时间、软截止迟到、装卸和拆单。软截止不是硬窗
装不下或赶不上硬时间窗时，会把哪辆车、哪一段没通过发回来。请改正后重新输出完整路线，不要解释。
"""


# The model clock is separate from the CP-SAT limit. A slow draft must not
# shorten the search. The budget covers several repair rounds.
MODEL_TIMEOUT_S = 180.0
MAX_MODEL_ROUNDS = 2


def model_proposal_enabled() -> bool:
    return LlmSettings.from_env().configured


def propose_model_assignment(
    nodes: list[CustomerNode],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
    fixed_nodes: dict[str, str] | None = None,
    settings: LlmSettings | None = None,
    complete: Callable[..., dict] | None = None,
    timeout_s: float = MODEL_TIMEOUT_S,
    weights: Weights | None = None,
) -> tuple[dict[str, list[CustomerNode]] | None, str]:
    """Return a feasible assignment, or None with a reason.

    Infeasible drafts are sent back with the failing vehicle and cause.
    Rounds share timeout_s and do not touch the CP-SAT clock.
    """
    chosen = settings if settings is not None else LlmSettings.from_env()
    if not chosen.configured:
        return None, "未配置模型，不生成初稿"
    caller = complete or complete_json
    deadline = time.monotonic() + timeout_s
    base = _prompt(dataset, weights, fixed_nodes)
    feedback = ""
    last_note = "模型没有给出初稿"
    for index in range(1, MAX_MODEL_ROUNDS + 1):
        budget = _attempt_timeout(deadline, MAX_MODEL_ROUNDS - index + 1)
        if budget < 8:
            break
        user = base if not feedback else base + "\n" + feedback
        try:
            payload = caller(chosen, _SYSTEM, user, timeout_s=budget)
        except TypeError:
            payload = caller(chosen, _SYSTEM, user)
        except ModelCallError as exc:
            message = str(exc)
            if "超时" not in message:
                return None, message
            if "没有通过" in last_note or "无法使用" in last_note:
                last_note = last_note + "；后面一轮超时"
            else:
                last_note = f"模型第 {index} 轮超时"
            continue
        try:
            assignment, note = assignment_from_payload(payload, nodes, dataset, fixed_nodes)
        except (TypeError, ValueError, KeyError):
            assignment, note = None, "模型初稿无法解析"
        if assignment is None:
            feedback = "上一版没有通过，请改正后重新输出完整 JSON，不要解释。\n" + note
            last_note = f"模型第 {index} 版无法使用：{note}"
            continue
        reasons = _rejection_reasons(assignment, dataset, matrix, units_by_id)
        if not reasons:
            return assignment, f"模型第 {index} 版通过时间窗和装箱"
        shown = "\n".join(reasons[:6])
        feedback = (
            "上一版没有通过，请改正后重新输出完整 JSON，不要解释。\n"
            + _routes_text(assignment)
            + "\n问题：\n"
            + shown
        )
        last_note = f"模型 {index} 轮都没有通过时间窗或装箱"
    return None, last_note


def assignment_from_payload(
    payload: dict,
    nodes: list[CustomerNode],
    dataset: Dataset,
    fixed_nodes: dict[str, str] | None = None,
) -> tuple[dict[str, list[CustomerNode]] | None, str]:
    routes = payload.get("routes") if isinstance(payload, dict) else None
    if not isinstance(routes, list):
        return None, "模型初稿没有 routes"
    by_node = {node.node_id: node for node in nodes}
    by_store: dict[str, list[CustomerNode]] = {}
    for node in nodes:
        by_store.setdefault(node.store_id, []).append(node)
    vehicles = {vehicle.id for vehicle in dataset.vehicles}
    assignment: dict[str, list[CustomerNode]] = {vehicle_id: [] for vehicle_id in vehicles}
    used: set[str] = set()
    for route in routes:
        if not isinstance(route, dict):
            return None, "模型初稿的路线格式不对"
        vehicle_id = str(route.get("vehicle_id", ""))
        if vehicle_id not in vehicles:
            return None, f"模型初稿使用了未知车辆 {vehicle_id}"
        stops = route.get("stops") or []
        if not isinstance(stops, list):
            return None, "模型初稿的停靠不是列表"
        for stop in stops:
            chosen_nodes = _nodes_for_stop(stop, by_node, by_store)
            if chosen_nodes is None:
                return None, f"模型初稿使用了未知停靠 {stop}"
            for node in chosen_nodes:
                if node.node_id in used:
                    return None, f"停靠 {node.store_id} 被重复分配"
                used.add(node.node_id)
                assignment[vehicle_id].append(node)
    missing = [node.store_id for node in nodes if node.node_id not in used]
    if missing:
        return None, "模型初稿漏了 " + "、".join(missing[:6])
    locked = fixed_nodes or {}
    for node_id, vehicle_id in locked.items():
        node = by_node.get(node_id)
        if node is None:
            continue
        if not any(item.node_id == node_id for item in assignment.get(vehicle_id, [])):
            return None, f"模型初稿把必须留在 {vehicle_id} 的 {node.store_id} 放错了车"
    used_vehicles = [vehicle_id for vehicle_id, sequence in assignment.items() if sequence]
    return assignment, "模型给出 " + str(len(used_vehicles)) + " 条路线"


def _nodes_for_stop(
    stop: object,
    by_node: dict[str, CustomerNode],
    by_store: dict[str, list[CustomerNode]],
) -> list[CustomerNode] | None:
    if isinstance(stop, dict):
        token = stop.get("node_id") or stop.get("store_id") or stop.get("id")
    else:
        token = stop
    if not isinstance(token, str):
        return None
    if token in by_node:
        return [by_node[token]]
    if token in by_store:
        return list(by_store[token])
    prefixed = f"store:{token}"
    if prefixed in by_node:
        return [by_node[prefixed]]
    return None


def _prompt(dataset: Dataset, weights: Weights | None, fixed_nodes: dict[str, str] | None) -> str:
    """The solver's own records, not a summary."""
    body = {
        "depot": dataset.depot.model_dump(),
        "stores": [store.model_dump() for store in dataset.stores],
        "vehicles": [vehicle.model_dump() for vehicle in dataset.vehicles],
        "skus": [sku.model_dump() for sku in dataset.skus],
        "orders": [order.model_dump() for order in dataset.orders],
        "arcs": [arc.model_dump() for arc in dataset.arcs],
        "loading_rule": dataset.loading_rule.model_dump(),
        "weights": (weights or Weights()).model_dump(),
        "fixed_vehicle_by_node": fixed_nodes or {},
    }
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def _attempt_timeout(deadline: float, attempts_left: int) -> float:
    """Leave time for later repairs instead of spending the whole clock on one call."""
    remaining = deadline - time.monotonic()
    if attempts_left <= 1 or remaining <= 50:
        return max(0.0, remaining)
    return max(0.0, min(90.0, remaining - 70.0))


def _routes_text(assignment: dict[str, list[CustomerNode]]) -> str:
    routes = [
        {"vehicle_id": vehicle_id, "stops": [node.store_id for node in sequence]}
        for vehicle_id, sequence in assignment.items()
        if sequence
    ]
    return json.dumps({"routes": routes}, ensure_ascii=False, separators=(",", ":"))


def _rejection_reasons(
    assignment: dict[str, list[CustomerNode]],
    dataset: Dataset,
    matrix: TravelMatrix,
    units_by_id: dict[str, PackingUnit],
) -> list[str]:
    vehicles = {vehicle.id: vehicle for vehicle in dataset.vehicles}
    reasons: list[str] = []
    for vehicle_id, sequence in assignment.items():
        if not sequence:
            continue
        vehicle = vehicles.get(vehicle_id)
        if vehicle is None:
            reasons.append(f"未知车辆 {vehicle_id}")
            continue
        window = _window_reason(sequence, matrix, dataset)
        if window:
            reasons.append(f"{vehicle_id} {window}")
        packed = pack_stops(vehicle, stops_from_nodes(sequence, units_by_id), dataset.loading_rule)
        if packed.ok:
            continue
        names = "、".join(node.store_id for node in sequence)
        if packed.reason == "capacity":
            reasons.append(f"{vehicle_id} 装不下 {names}：超载重或体积")
        else:
            reasons.append(f"{vehicle_id} 装不下 {names}：车厢里放不下，后卸的不能挡先卸的")
    return reasons


def _window_reason(sequence: list[CustomerNode], matrix: TravelMatrix, dataset: Dataset) -> str | None:
    depot = dataset.depot
    clock = depot.window_open_s
    previous = depot.id
    try:
        for node in sequence:
            clock += matrix.get(previous, node.store_id).travel_time_s
            if clock > node.window_close_s:
                return f"到达 {node.store_id} 时为 {_clock(clock)}，晚于硬窗 {_clock(node.window_close_s)}"
            if clock < node.window_open_s:
                clock = node.window_open_s
            clock += node.service_time_s
            previous = node.store_id
        clock += matrix.get(previous, depot.id).travel_time_s
    except MissingTravelArcError as exc:
        return f"缺少 {exc.from_id} 到 {exc.to_id} 的行驶弧"
    if clock > depot.window_close_s:
        return f"回到仓库时刻 {_clock(clock)}，晚于仓库关门 {_clock(depot.window_close_s)}"
    return None


def _clock(seconds: int) -> str:
    seconds = seconds % 86_400
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"

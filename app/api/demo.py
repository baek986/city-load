"""Demo endpoints that solve a built-in day and return drawing geometry."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException

from app.api.schemas import CustomSolve, DemoSolve, TaskCreate
from app.demo.scenarios import (
    SCENARIOS,
    custom_catalogue,
    custom_day,
    load_scenario,
    scenario_by_id,
)
from app.demo.view import build_scene
from app.optimizer.joint import solve_joint

router = APIRouter()


# ---------------------------------------------------------------------------
# In-memory task store for background solves.
# ---------------------------------------------------------------------------

@dataclass
class _Task:
    id: str
    name: str
    detail: str
    status: str = "pending"  # pending | running | completed | failed
    scene: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


_tasks: dict[str, _Task] = {}
_tasks_lock = threading.Lock()


def _list_tasks() -> list[_Task]:
    with _tasks_lock:
        return sorted(_tasks.values(), key=lambda t: t.created_at, reverse=True)


def _get_task(task_id: str) -> _Task | None:
    with _tasks_lock:
        return _tasks.get(task_id)


def _set_task(task_id: str, **kwargs: object) -> None:
    with _tasks_lock:
        t = _tasks.get(task_id)
        if t is not None:
            for k, v in kwargs.items():
                setattr(t, k, v)


def _task_to_dict(t: _Task) -> dict:
    return {
        "id": t.id,
        "name": t.name,
        "detail": t.detail,
        "status": t.status,
        "error": t.error,
        "created_at": t.created_at,
        "completed_at": t.completed_at,
    }


def _run_solve(task_id: str, dataset, seed: int, time_limit_s: float, scenario_meta: dict) -> None:
    _set_task(task_id, status="running")
    try:
        result = solve_joint(dataset, seed=seed, time_limit_s=time_limit_s)
        if result.status != "completed":
            _set_task(
                task_id,
                status="failed",
                error=result.error_message or "No feasible plan",
                completed_at=time.time(),
            )
            return
        scene = build_scene(dataset, result, scenario_meta)
        _set_task(task_id, status="completed", scene=scene, completed_at=time.time())
    except Exception as exc:  # noqa: BLE001
        _set_task(task_id, status="failed", error=str(exc), completed_at=time.time())


# ---------------------------------------------------------------------------
# Preset scenario endpoints (kept for backward compatibility).
# ---------------------------------------------------------------------------

@router.get("/demo/scenarios")
def get_scenarios() -> dict:
    return {"scenarios": list(SCENARIOS)}


@router.post("/demo/solve")
def post_demo(body: DemoSolve) -> dict:
    try:
        scenario = scenario_by_id(body.scenario)
        dataset = load_scenario(body.scenario)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": str(exc)}) from exc
    result = solve_joint(dataset, seed=body.seed, time_limit_s=body.time_limit_s)
    if result.status != "completed":
        raise HTTPException(
            status_code=409,
            detail={
                "code": result.error_code or "failed",
                "message": result.error_message or "No feasible plan",
                "unit_ids": result.failed_unit_ids,
                "store_ids": result.infeasible_stores,
            },
        )
    return build_scene(dataset, result, scenario)


# ---------------------------------------------------------------------------
# Custom scenario catalogue + solve (kept for backward compatibility).
# ---------------------------------------------------------------------------

@router.get("/demo/catalog")
def get_catalog() -> dict:
    """Return the catalogue of selectable stores, SKUs, and vehicles for the custom scenario."""
    return custom_catalogue()


@router.post("/demo/solve-custom")
def post_solve_custom(body: CustomSolve) -> dict:
    """Solve a custom day built from user-selected stores and SKU lines."""
    if not body.stores:
        raise HTTPException(status_code=422, detail={"code": "no_stores", "message": "请至少选择一个门店"})
    has_lines = any(line.qty > 0 for store in body.stores for line in store.lines)
    if not has_lines:
        raise HTTPException(status_code=422, detail={"code": "no_skus", "message": "请至少为一个门店选择药箱"})
    store_picks = [
        {"store_id": store.store_id, "lines": [{"sku_id": line.sku_id, "qty": line.qty} for line in store.lines]}
        for store in body.stores
    ]
    dataset = custom_day(store_picks)
    result = solve_joint(dataset, seed=body.seed, time_limit_s=body.time_limit_s)
    if result.status != "completed":
        raise HTTPException(
            status_code=409,
            detail={
                "code": result.error_code or "failed",
                "message": result.error_message or "No feasible plan",
                "unit_ids": result.failed_unit_ids,
                "store_ids": result.infeasible_stores,
            },
        )
    scenario = {"id": "custom", "name": "自定义", "detail": f"{len(dataset.stores)}家店", "seed": body.seed, "time_limit_s": body.time_limit_s}
    return build_scene(dataset, result, scenario)


# ---------------------------------------------------------------------------
# Background task endpoints: create / list / poll / get result.
# ---------------------------------------------------------------------------

@router.post("/demo/tasks")
def post_task(body: TaskCreate) -> dict:
    """Create a background solve task. Returns immediately with task metadata."""
    task_id = uuid.uuid4().hex[:12]

    if body.custom is not None:
        cs = body.custom
        if not cs.stores:
            raise HTTPException(status_code=422, detail={"code": "no_stores", "message": "请至少选择一个门店"})
        has_lines = any(line.qty > 0 for store in cs.stores for line in store.lines)
        if not has_lines:
            raise HTTPException(status_code=422, detail={"code": "no_skus", "message": "请至少为一个门店选择药箱"})
        store_picks = [
            {"store_id": s.store_id, "lines": [{"sku_id": l.sku_id, "qty": l.qty} for l in s.lines]}
            for s in cs.stores
        ]
        dataset = custom_day(store_picks)
        seed = cs.seed
        time_limit = cs.time_limit_s
        store_count = len(dataset.stores)
        scenario_meta = {
            "id": "custom",
            "name": f"自定义方案 ({store_count}店)",
            "detail": f"{store_count}家店 · 自选药箱",
            "seed": seed,
            "time_limit_s": time_limit,
        }
        task = _Task(
            id=task_id,
            name=scenario_meta["name"],
            detail=scenario_meta["detail"],
        )
    else:
        try:
            scenario_meta = scenario_by_id(body.scenario)
            dataset = load_scenario(body.scenario)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"code": "not_found", "message": str(exc)}) from exc
        seed = body.seed if body.seed is not None else scenario_meta.get("seed", 1)
        time_limit = body.time_limit_s if body.time_limit_s is not None else scenario_meta.get("time_limit_s", 30.0)
        task = _Task(
            id=task_id,
            name=scenario_meta["name"],
            detail=scenario_meta["detail"],
        )

    with _tasks_lock:
        _tasks[task_id] = task

    thread = threading.Thread(
        target=_run_solve,
        args=(task_id, dataset, seed, time_limit, scenario_meta),
        daemon=True,
    )
    thread.start()

    return _task_to_dict(task)


@router.get("/demo/tasks")
def get_tasks() -> dict:
    """List all tasks, newest first."""
    return {"tasks": [_task_to_dict(t) for t in _list_tasks()]}


@router.get("/demo/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    """Poll a single task's status. If completed, includes the full scene."""
    t = _get_task(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "任务不存在"})
    d = _task_to_dict(t)
    if t.status == "completed" and t.scene is not None:
        d["scene"] = t.scene
    return d


@router.delete("/demo/tasks/{task_id}")
def delete_task(task_id: str) -> dict:
    """Delete a task from the in-memory store."""
    with _tasks_lock:
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail={"code": "not_found", "message": "任务不存在"})
        del _tasks[task_id]
    return {"ok": True}
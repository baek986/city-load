"""Demo endpoints that solve a built-in day and return drawing geometry."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import DemoSolve
from app.demo.scenarios import SCENARIOS, load_scenario, scenario_by_id
from app.demo.view import build_scene
from app.optimizer.joint import solve_joint

router = APIRouter()


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

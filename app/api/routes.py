"""Planning and master-data routes."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.api.schemas import PlanCreate, ReplanCreate
from app.db import Repository
from app.domain.models import LoadingRule, Location, Order, Sku, TravelArc, Vehicle, Weights
from app.optimizer.joint import replan, solve_joint

router = APIRouter()


def _engine(request: Request):
    return request.app.state.engine


def get_repo(request: Request):
    session = Session(_engine(request))
    try:
        yield Repository(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _error_detail(code: str, message: str, unit_ids: list[str] | None = None, store_ids: list[str] | None = None) -> dict:
    return {"code": code, "message": message, "unit_ids": unit_ids or [], "store_ids": store_ids or []}


def _run_plan(engine, plan_id: str, time_limit_s: float, prove_optimal: bool = False) -> None:
    session = Session(engine)
    repo = Repository(session)
    try:
        row = repo.get_plan_row(plan_id)
        if row is None:
            return
        dataset = repo.load_dataset(row.service_date)
        weights = Weights.model_validate_json(row.weights_json)
        result = solve_joint(dataset, weights=weights, seed=row.seed, time_limit_s=time_limit_s, prove_optimal=prove_optimal)
        repo.save_result(plan_id, result)
        session.commit()
    except Exception as exc:
        session.rollback()
        row = repo.get_plan_row(plan_id)
        if row is not None:
            from app.domain.models import JointResult

            repo.save_result(
                plan_id,
                JointResult(status="failed", error_code="internal_error", error_message=str(exc)),
            )
            session.commit()
    finally:
        session.close()


def _summary(plan_id: str, status: str, result) -> dict:
    if result is None:
        return {
            "plan_id": plan_id,
            "status": status,
            "proven_optimal": False,
            "lower_bound": None,
            "gap": None,
            "score": None,
            "split_order_count": None,
            "on_time_rate": None,
            "vehicle_utilization": [],
            "iteration_limit_reached": False,
            "error": None,
            "infeasible_stores": [],
            "failed_unit_ids": [],
        }
    error = None
    if result.error_code:
        error = _error_detail(result.error_code, result.error_message or "", result.failed_unit_ids, result.infeasible_stores)
    return {
        "plan_id": plan_id,
        "status": status,
        "proven_optimal": result.proven_optimal,
        "lower_bound": result.lower_bound,
        "gap": result.gap,
        "score": result.score,
        "split_order_count": result.split_order_count,
        "on_time_rate": result.on_time_rate,
        "vehicle_utilization": [stat.model_dump() for stat in result.vehicle_stats],
        "iteration_limit_reached": result.iteration_limit_reached,
        "reshuffle_count": result.reshuffle_count,
        "rounds": result.rounds,
        "error": error,
        "infeasible_stores": result.infeasible_stores,
        "failed_unit_ids": result.failed_unit_ids,
    }


@router.put("/depot")
def put_depot(body: Location, repo: Repository = Depends(get_repo)) -> Location:
    return repo.set_depot(body.model_copy(update={"kind": "depot"}))


@router.get("/depot")
def get_depot(repo: Repository = Depends(get_repo)) -> Location:
    depot = repo.get_depot()
    if depot is None:
        raise HTTPException(status_code=404, detail=_error_detail("not_found", "Depot is not configured"))
    return depot


@router.post("/stores", status_code=201)
def post_store(body: Location, repo: Repository = Depends(get_repo)) -> Location:
    return repo.add_store(body.model_copy(update={"kind": "store"}))


@router.get("/stores")
def get_stores(repo: Repository = Depends(get_repo)) -> list[Location]:
    return repo.list_stores()


@router.delete("/stores/{store_id}")
def remove_store(store_id: str, repo: Repository = Depends(get_repo)) -> dict:
    if not repo.delete_store(store_id):
        raise HTTPException(status_code=404, detail=_error_detail("not_found", "Store not found"))
    return {"deleted": store_id}


@router.post("/skus", status_code=201)
def post_sku(body: Sku, repo: Repository = Depends(get_repo)) -> Sku:
    return repo.add_sku(body)


@router.get("/skus")
def get_skus(repo: Repository = Depends(get_repo)) -> list[Sku]:
    return repo.list_skus()


@router.delete("/skus/{sku_id}")
def remove_sku(sku_id: str, repo: Repository = Depends(get_repo)) -> dict:
    if not repo.delete_sku(sku_id):
        raise HTTPException(status_code=404, detail=_error_detail("not_found", "SKU not found"))
    return {"deleted": sku_id}


@router.post("/vehicles", status_code=201)
def post_vehicle(body: Vehicle, repo: Repository = Depends(get_repo)) -> Vehicle:
    return repo.add_vehicle(body)


@router.get("/vehicles")
def get_vehicles(repo: Repository = Depends(get_repo)) -> list[Vehicle]:
    return repo.list_vehicles()


@router.delete("/vehicles/{vehicle_id}")
def remove_vehicle(vehicle_id: str, repo: Repository = Depends(get_repo)) -> dict:
    if not repo.delete_vehicle(vehicle_id):
        raise HTTPException(status_code=404, detail=_error_detail("not_found", "Vehicle not found"))
    return {"deleted": vehicle_id}


@router.post("/orders", status_code=201)
def post_order(body: Order, repo: Repository = Depends(get_repo)) -> Order:
    return repo.add_order(body)


@router.get("/orders")
def get_orders(date: str | None = None, repo: Repository = Depends(get_repo)) -> list[Order]:
    return repo.list_orders(date)


@router.delete("/orders/{order_id}")
def remove_order(order_id: str, repo: Repository = Depends(get_repo)) -> dict:
    if not repo.delete_order(order_id):
        raise HTTPException(status_code=404, detail=_error_detail("not_found", "Order not found"))
    return {"deleted": order_id}


@router.post("/travel-arcs", status_code=201)
def post_arc(body: TravelArc, repo: Repository = Depends(get_repo)) -> TravelArc:
    return repo.add_arc(body)


@router.get("/travel-arcs")
def get_arcs(repo: Repository = Depends(get_repo)) -> list[TravelArc]:
    return repo.list_arcs()


@router.put("/loading-rules")
def put_rules(body: LoadingRule, repo: Repository = Depends(get_repo)) -> LoadingRule:
    return repo.set_loading_rule(body)


@router.post("/plans", status_code=202)
def post_plan(
    body: PlanCreate,
    background: BackgroundTasks,
    request: Request,
    repo: Repository = Depends(get_repo),
) -> dict:
    try:
        repo.load_dataset(body.date)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=_error_detail("missing_depot", str(exc))) from exc
    plan_id = repo.create_plan(body.date, body.weights, body.seed)
    background.add_task(_run_plan, request.app.state.engine, plan_id, body.time_limit_s, body.prove_optimal)
    return {"plan_id": plan_id, "status": "running"}


@router.get("/plans/{plan_id}")
def get_plan(plan_id: str, repo: Repository = Depends(get_repo)) -> dict:
    row = repo.get_plan_row(plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail=_error_detail("not_found", "Plan not found"))
    return _summary(plan_id, row.status, repo.get_result(plan_id))


@router.get("/plans/{plan_id}/routes")
def get_routes(plan_id: str, repo: Repository = Depends(get_repo)) -> dict:
    result = _require_result(plan_id, repo)
    return {"routes": [route.model_dump() for route in result.routes]}


@router.get("/plans/{plan_id}/placements")
def get_placements(
    plan_id: str,
    vehicle_id: str | None = Query(default=None),
    repo: Repository = Depends(get_repo),
) -> dict:
    result = _require_result(plan_id, repo)
    placements = result.placements
    if vehicle_id is not None:
        placements = [item for item in placements if item.vehicle_id == vehicle_id]
    return {"placements": [item.model_dump() for item in placements]}


@router.get("/plans/{plan_id}/instructions")
def get_instructions(
    plan_id: str,
    vehicle_id: str | None = Query(default=None),
    repo: Repository = Depends(get_repo),
) -> dict:
    result = _require_result(plan_id, repo)
    steps = result.steps
    if vehicle_id is not None:
        steps = [step for step in steps if step.vehicle_id == vehicle_id]
    return {"instructions": [step.model_dump() for step in steps]}


@router.post("/plans/{plan_id}/replan", status_code=202)
def post_replan(
    plan_id: str,
    body: ReplanCreate,
    background: BackgroundTasks,
    request: Request,
    repo: Repository = Depends(get_repo),
) -> dict:
    previous_row = repo.get_plan_row(plan_id)
    if previous_row is None:
        raise HTTPException(status_code=404, detail=_error_detail("not_found", "Plan not found"))
    previous = repo.get_result(plan_id)
    if previous is None:
        raise HTTPException(status_code=409, detail=_error_detail("plan_not_ready", "Plan has no result yet"))
    weights = body.weights or Weights.model_validate_json(previous_row.weights_json)
    seed = body.seed if body.seed is not None else previous_row.seed
    new_id = repo.create_plan(previous_row.service_date, weights, seed)

    def _job() -> None:
        session = Session(request.app.state.engine)
        local = Repository(session)
        try:
            dataset = local.load_dataset(previous_row.service_date)
            result = replan(
                dataset,
                previous,
                locked_order_ids=body.locked_order_ids,
                weights=weights,
                seed=seed,
                time_limit_s=body.time_limit_s,
                locked_vehicle_ids=body.locked_vehicle_ids,
                prove_optimal=body.prove_optimal,
            )
            local.save_result(new_id, result)
            session.commit()
        except Exception as exc:
            session.rollback()
            from app.domain.models import JointResult

            local.save_result(new_id, JointResult(status="failed", error_code="internal_error", error_message=str(exc)))
            session.commit()
        finally:
            session.close()

    background.add_task(_job)
    return {"plan_id": new_id, "status": "running"}


def _require_result(plan_id: str, repo: Repository):
    row = repo.get_plan_row(plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail=_error_detail("not_found", "Plan not found"))
    result = repo.get_result(plan_id)
    if result is None:
        raise HTTPException(status_code=409, detail=_error_detail("plan_not_ready", "Plan is still running"))
    if result.status == "failed":
        raise HTTPException(
            status_code=409,
            detail=_error_detail(
                result.error_code or "failed",
                result.error_message or "Plan failed",
                result.failed_unit_ids,
                result.infeasible_stores,
            ),
        )
    return result

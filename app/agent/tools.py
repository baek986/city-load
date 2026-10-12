"""Agent tools. Coordinates and routes come only from the optimizer."""

from __future__ import annotations

from app.db import Repository
from app.domain.models import Weights
from app.optimizer.joint import replan, solve_joint


class AgentToolbox:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    def list_orders(self, date: str) -> list[dict]:
        skus = {sku.id: sku for sku in self.repo.list_skus()}
        stores = {store.id: store for store in self.repo.list_stores()}
        rows = []
        for order in self.repo.list_orders(date):
            store = stores.get(order.store_id)
            for line in order.lines:
                sku = skus[line.sku_id]
                rows.append(
                    {
                        "order_id": order.id,
                        "line_id": line.id,
                        "store_id": order.store_id,
                        "window_open_s": None if store is None else store.window_open_s,
                        "window_close_s": None if store is None else store.window_close_s,
                        "sku_id": sku.id,
                        "qty": line.qty,
                        "weight_g": sku.weight_g * line.qty,
                        "volume_mm3": sku.volume_mm3 * line.qty,
                    }
                )
        return rows

    def list_fleet(self) -> list[dict]:
        return [
            {
                "vehicle_id": vehicle.id,
                "type_name": vehicle.type_name,
                "length_mm": vehicle.length_mm,
                "width_mm": vehicle.width_mm,
                "height_mm": vehicle.height_mm,
                "max_weight_g": vehicle.max_weight_g,
                "cost_per_km": vehicle.cost_per_km,
                "fixed_cost": vehicle.fixed_cost,
            }
            for vehicle in self.repo.list_vehicles()
        ]

    def get_travel_time(self, from_id: str, to_id: str) -> dict:
        for arc in self.repo.list_arcs():
            if arc.from_id == from_id and arc.to_id == to_id:
                return {"from_id": from_id, "to_id": to_id, "distance_m": arc.distance_m, "travel_time_s": arc.travel_time_s}
        if from_id == to_id:
            return {"from_id": from_id, "to_id": to_id, "distance_m": 0, "travel_time_s": 0}
        return {"from_id": from_id, "to_id": to_id, "error": "missing_travel_arc"}

    def create_plan(self, date: str, weights: dict | None = None, seed: int = 1) -> dict:
        parsed = Weights.model_validate(weights or {})
        dataset = self.repo.load_dataset(date)
        from app.agent.proposal import model_proposal_enabled

        result = solve_joint(dataset, weights=parsed, seed=seed, model_proposal=model_proposal_enabled())
        plan_id = self.repo.create_plan(date, parsed, seed)
        self.repo.save_result(plan_id, result)
        return self.get_plan_summary(plan_id)

    def get_plan_summary(self, plan_id: str) -> dict:
        row = self.repo.get_plan_row(plan_id)
        result = self.repo.get_result(plan_id)
        if row is None or result is None:
            return {"plan_id": plan_id, "error": "not_found"}
        idle = [stat.vehicle_id for stat in result.vehicle_stats if not stat.used]
        return {
            "plan_id": plan_id,
            "status": result.status,
            "proven_optimal": result.proven_optimal,
            "lower_bound": result.lower_bound,
            "gap": result.gap,
            "score": result.score,
            "on_time_rate": result.on_time_rate,
            "split_order_count": result.split_order_count,
            "idle_vehicles": idle,
            "utilization": [stat.model_dump() for stat in result.vehicle_stats],
            "error_code": result.error_code,
            "error_message": result.error_message,
        }

    def explain_route(self, plan_id: str, vehicle_id: str) -> dict:
        result = self.repo.get_result(plan_id)
        if result is None:
            return {"error": "not_found"}
        route = next((item for item in result.routes if item.vehicle_id == vehicle_id), None)
        if route is None:
            return {"vehicle_id": vehicle_id, "stops": [], "reason": "vehicle is idle"}
        stops = []
        weight = 0
        for stop in route.stops:
            weight += sum(1 for _ in stop.unit_ids)
            stops.append(
                {
                    "store_id": stop.store_id,
                    "arrival_s": stop.arrival_s,
                    "depart_s": stop.depart_s,
                    "late_minutes": stop.late_minutes,
                    "unit_count": len(stop.unit_ids),
                    "reason": "arrival is inside the hard window"
                    if stop.late_minutes <= 0
                    else "arrival is inside the hard window and after the soft deadline",
                }
            )
        return {
            "vehicle_id": vehicle_id,
            "distance_m": route.distance_m,
            "travel_time_s": route.travel_time_s,
            "travel_cost": route.travel_cost,
            "fixed_cost": route.fixed_cost,
            "total_weight_g": route.total_weight_g,
            "stops": stops,
        }

    def explain_placement(self, plan_id: str, vehicle_id: str) -> dict:
        result = self.repo.get_result(plan_id)
        if result is None:
            return {"error": "not_found"}
        items = [item for item in result.placements if item.vehicle_id == vehicle_id]
        if not items:
            failed = result.failed_unit_ids
            return {
                "vehicle_id": vehicle_id,
                "blocks": [],
                "unplaced_unit_ids": failed,
                "reason": result.error_message,
            }
        by_store: dict[str, list] = {}
        for item in items:
            by_store.setdefault(item.store_id, []).append(item)
        blocks = []
        for store_id, group in by_store.items():
            blocks.append(
                {
                    "store_id": store_id,
                    "x_min": min(item.x for item in group),
                    "x_max": max(item.x + item.dx for item in group),
                    "unit_count": len(group),
                    "placements": [item.model_dump() for item in group],
                }
            )
        blocks.sort(key=lambda block: block["x_min"])
        return {
            "vehicle_id": vehicle_id,
            "blocks": blocks,
            "unplaced_unit_ids": [],
            "reason": "blocks are ordered from the door toward the deep end",
        }

    def get_handling_steps(self, plan_id: str, vehicle_id: str) -> list[dict]:
        result = self.repo.get_result(plan_id)
        if result is None:
            return []
        return [step.model_dump() for step in result.steps if step.vehicle_id == vehicle_id]

    def dispatch_plan(self, plan_id: str, text: str, time_limit_s: float = 180.0) -> dict:
        from app.agent.dispatch import DispatchError, run_dispatch

        try:
            return run_dispatch(self.repo, plan_id, text, time_limit_s=time_limit_s)
        except DispatchError as exc:
            return {"error": exc.code, "message": exc.message}

    def replan_with_changes(
        self,
        plan_id: str,
        locked_orders: list[str] | None = None,
        locked_vehicles: list[str] | None = None,
        weight_overrides: dict | None = None,
        seed: int | None = None,
    ) -> dict:
        row = self.repo.get_plan_row(plan_id)
        previous = self.repo.get_result(plan_id)
        if row is None or previous is None:
            return {"error": "not_found"}
        weights = Weights.model_validate(weight_overrides) if weight_overrides else Weights.model_validate_json(row.weights_json)
        dataset = self.repo.load_dataset(row.service_date)
        result = replan(
            dataset,
            previous,
            locked_order_ids=list(locked_orders or []),
            weights=weights,
            seed=row.seed if seed is None else seed,
            locked_vehicle_ids=list(locked_vehicles or []),
        )
        new_id = self.repo.create_plan(row.service_date, weights, row.seed if seed is None else seed)
        self.repo.save_result(new_id, result)
        return self.get_plan_summary(new_id)

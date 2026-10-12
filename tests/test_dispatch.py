"""Dispatch sentences stay structured. The model slot stays empty unless configured."""

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.agent.dispatch import interpret, interpret_locally, resolve_dispatch
from app.agent.llm import LlmSettings, chat_completions_url
from app.api.main import create_app
from app.db import Repository
from app.domain.models import (
    Dataset,
    JointResult,
    Location,
    Order,
    OrderLine,
    RoutePlan,
    Sku,
    TravelArc,
    StopPlan,
    Vehicle,
    Weights,
)
from tests.test_api import _load


def _day() -> Dataset:
    depot = Location(id="depot", name="仓", kind="depot", x_m=0, y_m=0)
    stores = [
        Location(id="store-12", name="药店12", kind="store", x_m=1, y_m=1),
        Location(id="store-08", name="药店08", kind="store", x_m=2, y_m=2),
    ]
    sku = Sku(id="cold", name="感冒药", length_mm=10, width_mm=10, height_mm=10, weight_g=10)
    orders = [
        Order(
            id="o12",
            store_id="store-12",
            service_date="2026-10-10",
            lines=[OrderLine(id="l12", sku_id="cold", qty=1)],
        ),
        Order(
            id="o08",
            store_id="store-08",
            service_date="2026-10-10",
            lines=[OrderLine(id="l08", sku_id="cold", qty=1)],
        ),
    ]
    vehicles = [
        Vehicle(
            id=f"van-{index}",
            type_name="van",
            length_mm=1000,
            width_mm=1000,
            height_mm=1000,
            max_weight_g=100_000,
            cost_per_km=1,
            fixed_cost=1,
        )
        for index in (2, 3)
    ]
    return Dataset(depot=depot, stores=stores, skus=[sku], orders=orders, vehicles=vehicles, arcs=[])


def test_chat_url_matches_openai_style_base() -> None:
    assert chat_completions_url("https://api.deepseek.com") == "https://api.deepseek.com/chat/completions"
    assert chat_completions_url("https://api.openai.com/v1") == "https://api.openai.com/v1/chat/completions"


def test_empty_slot_uses_the_local_interpreter() -> None:
    raw, source = interpret("v2已经出门，其余更准时", _day(), LlmSettings())
    assert source == "cursor"
    prepared = resolve_dispatch(raw, _day(), Weights(), {"store-12": "van-2"})
    assert prepared.locked_vehicle_ids == ["van-2"]
    assert prepared.weights.late == 0.35


def test_local_add_and_reject_unknown_sentence() -> None:
    raw = interpret_locally("药店12加上两箱感冒药")
    prepared = resolve_dispatch(raw, _day(), Weights(), {})
    assert prepared.qty_edits == [("o12", "cold", 2)]
    rejected = interpret_locally("你好")
    assert rejected.error


def test_two_biases_are_rejected() -> None:
    raw = interpret_locally("更准时，同时少出车")
    assert raw.error


def test_model_route_draft_covers_each_store_once() -> None:
    from app.agent.proposal import assignment_from_payload

    from app.domain.units import build_units, group_units_by_store

    day = _day()
    nodes = group_units_by_store(day, build_units(day))
    assignment, note = assignment_from_payload(
        {"routes": [{"vehicle_id": "van-2", "stops": ["store-12", "store-08"]}]},
        nodes,
        day,
        None,
    )
    assert assignment is not None
    assert [node.store_id for node in assignment["van-2"]] == ["store-12", "store-08"]
    assert assignment["van-3"] == []
    assert "1 条路线" in note
    rejected, reason = assignment_from_payload(
        {"routes": [{"vehicle_id": "van-2", "stops": ["store-12"]}]},
        nodes,
        day,
        None,
    )
    assert rejected is None
    assert "漏了" in reason


def test_model_timeout_is_separate_from_the_solver() -> None:
    from app.agent.proposal import MODEL_TIMEOUT_S, propose_model_assignment
    from app.domain.matrix import TravelMatrix

    seen: dict = {}

    def complete(_settings, _system, user, timeout_s=None):
        seen["timeout_s"] = timeout_s
        seen["user"] = user
        return {"routes": []}

    day = _day()
    day.arcs.append(TravelArc(from_id="depot", to_id="store-12", distance_m=1000, travel_time_s=300))
    assignment, _note = propose_model_assignment(
        [],
        day,
        TravelMatrix(day.arcs),
        {},
        settings=LlmSettings(api_key="k", base_url="https://example.test", model="m"),
        complete=complete,
    )
    assert MODEL_TIMEOUT_S == 180
    assert 60 <= seen["timeout_s"] < MODEL_TIMEOUT_S
    assert assignment is not None
    raw = seen["user"]
    assert "service_time_s" in raw
    assert "travel_time_s" in raw
    assert "loading_rule" in raw
    assert "length_mm" in raw
    assert "store-12" in raw


def test_model_repairs_a_rejected_draft() -> None:
    from app.agent.proposal import propose_model_assignment
    from app.domain.matrix import TravelMatrix
    from app.domain.units import build_units, group_units_by_store

    day = _day()
    pairs = (
        ("depot", "store-12"),
        ("depot", "store-08"),
        ("store-12", "store-08"),
        ("store-08", "store-12"),
        ("store-12", "depot"),
        ("store-08", "depot"),
    )
    day.arcs = [TravelArc(from_id=origin, to_id=dest, distance_m=1000, travel_time_s=300) for origin, dest in pairs]
    units = build_units(day)
    calls = {"n": 0, "users": []}

    def complete(_settings, _system, user, timeout_s=None):
        calls["n"] += 1
        calls["users"].append(user)
        if calls["n"] == 1:
            return {"routes": [{"vehicle_id": "van-2", "stops": ["store-12"]}]}
        return {"routes": [{"vehicle_id": "van-2", "stops": ["store-12", "store-08"]}]}

    assignment, note = propose_model_assignment(
        group_units_by_store(day, units),
        day,
        TravelMatrix(day.arcs),
        {unit.unit_id: unit for unit in units},
        settings=LlmSettings(api_key="k", base_url="https://example.test", model="m"),
        complete=complete,
        timeout_s=30,
    )
    assert calls["n"] == 2
    assert "漏了" in calls["users"][1]
    assert assignment is not None
    assert [node.store_id for node in assignment["van-2"]] == ["store-12", "store-08"]
    assert "第 2 版" in note


def test_dispatch_endpoint_uses_local_interpreter_when_slot_is_empty(monkeypatch) -> None:
    monkeypatch.setattr("app.agent.dispatch.LlmSettings.from_env", classmethod(lambda cls: LlmSettings()))
    seen: dict = {}

    def fake_replan(*_args, **kwargs):
        seen.update(kwargs)
        return JointResult(status="completed", score=0.2)

    monkeypatch.setattr("app.api.routes.replan", fake_replan)
    app = create_app("sqlite:///:memory:")
    client = TestClient(app)
    _load(client)
    session = Session(app.state.engine)
    repo = Repository(session)
    plan_id = repo.create_plan("2026-10-10", Weights(), 1)
    repo.save_result(
        plan_id,
        JointResult(
            status="completed",
            routes=[
                RoutePlan(
                    vehicle_id="van",
                    stops=[
                        StopPlan(
                            store_id="s1",
                            node_ids=["s1"],
                            unit_ids=["u"],
                            arrival_s=0,
                            depart_s=60,
                            late_minutes=0,
                            sequence=1,
                        )
                    ],
                    distance_m=1000,
                    travel_time_s=100,
                    travel_cost=1,
                    fixed_cost=1,
                    depart_depot_s=0,
                    return_depot_s=200,
                    total_weight_g=500,
                    total_volume_mm3=8_000_000,
                )
            ],
        ),
    )
    session.commit()
    session.close()

    refused = client.post(f"/v1/plans/{plan_id}/dispatch", json={"text": "你好"})
    assert refused.status_code == 422
    accepted = client.post(f"/v1/plans/{plan_id}/dispatch", json={"text": "更准时", "time_limit_s": 5})
    assert accepted.status_code == 202
    body = accepted.json()
    assert body["interpreter"] == "cursor"
    assert body["status"] == "running"
    assert seen["weights"].late == 0.35

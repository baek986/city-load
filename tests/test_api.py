"""API and agent tool coverage on a two-store instance."""

from fastapi.testclient import TestClient

from app.agent.tools import AgentToolbox
from app.api.main import create_app
from app.db import Repository
from app.domain.models import LoadingRule, Location, Order, OrderLine, Sku, TravelArc, Vehicle
from sqlalchemy.orm import Session


def _load(client: TestClient) -> None:
    client.put(
        "/v1/depot",
        json=Location(
            id="depot",
            name="depot",
            kind="depot",
            x_m=0,
            y_m=0,
            window_open_s=0,
            window_close_s=100_000,
        ).model_dump(),
    )
    for store_id, x in (("s1", 1000.0), ("s2", 2000.0)):
        response = client.post(
            "/v1/stores",
            json=Location(
                id=store_id,
                name=store_id,
                kind="store",
                x_m=x,
                y_m=0,
                service_time_s=60,
                window_open_s=0,
                window_close_s=100_000,
            ).model_dump(),
        )
        assert response.status_code == 201
    assert client.post(
        "/v1/skus",
        json=Sku(id="sku", name="box", length_mm=200, width_mm=200, height_mm=200, weight_g=500).model_dump(),
    ).status_code == 201
    assert client.post(
        "/v1/vehicles",
        json=Vehicle(
            id="van",
            type_name="van",
            length_mm=2000,
            width_mm=1000,
            height_mm=1000,
            max_weight_g=100_000,
            cost_per_km=2,
            fixed_cost=10,
        ).model_dump(),
    ).status_code == 201
    for store_id in ("s1", "s2"):
        assert client.post(
            "/v1/orders",
            json=Order(
                id=f"order-{store_id}",
                store_id=store_id,
                service_date="2026-10-10",
                lines=[OrderLine(id=f"line-{store_id}", sku_id="sku", qty=1)],
            ).model_dump(),
        ).status_code == 201
    points = {"depot": (0, 0), "s1": (1000, 0), "s2": (2000, 0)}
    for origin, (ox, oy) in points.items():
        for dest, (dx, dy) in points.items():
            if origin == dest:
                continue
            distance = abs(ox - dx) + abs(oy - dy)
            assert client.post(
                "/v1/travel-arcs",
                json=TravelArc(from_id=origin, to_id=dest, distance_m=distance, travel_time_s=distance // 10).model_dump(),
            ).status_code == 201
    assert client.put("/v1/loading-rules", json=LoadingRule().model_dump()).status_code == 200


def test_plan_endpoints_and_agent_tools() -> None:
    app = create_app("sqlite:///:memory:")
    client = TestClient(app)
    _load(client)
    created = client.post("/v1/plans", json={"date": "2026-10-10", "seed": 1, "time_limit_s": 20})
    assert created.status_code == 202
    plan_id = created.json()["plan_id"]
    summary = client.get(f"/v1/plans/{plan_id}")
    assert summary.status_code == 200
    body = summary.json()
    assert body["status"] == "completed"
    assert body["gap"] is not None
    assert "proven_optimal" in body
    routes = client.get(f"/v1/plans/{plan_id}/routes").json()["routes"]
    assert routes
    vehicle_id = routes[0]["vehicle_id"]
    placements = client.get(f"/v1/plans/{plan_id}/placements", params={"vehicle_id": vehicle_id}).json()["placements"]
    assert placements
    steps = client.get(f"/v1/plans/{plan_id}/instructions", params={"vehicle_id": vehicle_id}).json()["instructions"]
    assert steps[0]["action"] == "unload"
    assert any(step["action"] == "load" for step in steps)

    session = Session(app.state.engine)
    tools = AgentToolbox(Repository(session))
    listed = tools.list_orders("2026-10-10")
    assert len(listed) == 2
    assert tools.list_fleet()[0]["vehicle_id"] == "van"
    travel = tools.get_travel_time("depot", "s1")
    assert travel["distance_m"] == 1000
    explanation = tools.explain_placement(plan_id, vehicle_id)
    explained = {
        (item["unit_id"], item["x"], item["y"], item["z"])
        for block in explanation["blocks"]
        for item in block["placements"]
    }
    returned = {(item["unit_id"], item["x"], item["y"], item["z"]) for item in placements}
    assert explained == returned
    session.close()

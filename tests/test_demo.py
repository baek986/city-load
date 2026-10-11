"""The demo page and the geometry returned for a short road trip."""

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.demo.scenarios import preview_day
from app.demo.view import build_scene
from app.optimizer.joint import solve_joint


def test_demo_page_links_route_and_loading() -> None:
    client = TestClient(create_app("sqlite:///:memory:"))
    page = client.get("/demo")
    assert page.status_code == 200
    assert "路线" in page.text
    assert "装载" in page.text
    assert "清河" in page.text
    assert "清河桥" in page.text
    assert "中心公园" in page.text
    assert 'id="map"' in page.text
    assert 'id="cargo"' in page.text
    home = client.get("/")
    assert home.status_code == 200
    assert 'href="/demo"' in home.text
    listed = client.get("/v1/demo/scenarios")
    assert listed.status_code == 200
    ids = {item["id"] for item in listed.json()["scenarios"]}
    assert {"preview", "heavy", "pharmacy"} <= ids


def test_preview_scene_follows_the_streets() -> None:
    dataset = preview_day()
    result = solve_joint(dataset, seed=1, time_limit_s=15)
    assert result.status == "completed", result.error_message
    scene = build_scene(dataset, result, {"id": "preview", "name": "短途预览"})
    assert scene["summary"]["store_count"] == 3
    assert scene["routes"]
    assert scene["placements"]
    for route in scene["routes"]:
        assert route["stops"]
        for leg in route["legs"]:
            points = leg["points"]
            length = sum(
                abs(right[0] - left[0]) + abs(right[1] - left[1]) for left, right in zip(points, points[1:])
            )
            assert length == leg["distance_m"]
            assert len(points) >= 2
    client = TestClient(create_app("sqlite:///:memory:"))
    response = client.post("/v1/demo/solve", json={"scenario": "preview", "seed": 1, "time_limit_s": 15})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["store_count"] == 3
    assert body["placements"]
    assert body["network"]["block_m"] == 400

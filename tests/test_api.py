"""HTTP API and static frontend tests."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from app import analyzer
from app.main import app
from app.tiers import tier_for_score


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"]
    assert payload["model"].endswith(".onnx")


def test_tiers_endpoint(client: TestClient) -> None:
    response = client.get("/api/tiers")
    assert response.status_code == 200
    tiers = response.json()
    assert len(tiers) == 7
    assert tiers[0]["id"] == "sub3"
    assert tiers[-1]["id"] == "true_adam"
    assert tiers[-1]["name"] == "TRUE ADAM"


def test_home_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "TIER CHECK" in response.text
    assert "/styles.css" in response.text
    assert "/app.js" in response.text


@pytest.mark.parametrize("path", ["/styles.css", "/app.js", "/favicon.svg"])
def test_static_assets(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert response.content


def test_rate_rejects_short_payload(client: TestClient) -> None:
    response = client.post("/api/rate", json={"image": "short"})
    assert response.status_code == 422


def test_rate_rejects_broken_image(client: TestClient) -> None:
    payload = base64.b64encode(b"<html>not an image</html>" * 4).decode()
    response = client.post("/api/rate", json={"image": payload})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


def test_rate_response_contract(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    tier = tier_for_score(73.4)
    fixed = analyzer.Analysis(
        score=73.4,
        tier=tier,
        stats=(
            analyzer.Stat("jaw", "ЧЕЛЮСТЬ", 77),
            analyzer.Stat("symmetry", "СИММЕТРИЯ", 81),
            analyzer.Stat("harmony", "ПРОПОРЦИИ", 69),
            analyzer.Stat("skin", "КОЖА", 58),
            analyzer.Stat("hair", "ОБЪЁМ", 72),
            analyzer.Stat("light", "СВЕТ", 90),
        ),
        notes=("Тестовая заметка.", "Это мем, а не диагноз."),
        face_box=(0.2, 0.15, 0.4, 0.5),
        confidence=0.97,
    )
    monkeypatch.setattr(analyzer, "analyze_data_url", lambda _: fixed)

    image = base64.b64encode(b"fake-but-long-enough-image-payload" * 4).decode()
    response = client.post("/api/rate", json={"image": image})
    assert response.status_code == 200
    payload = response.json()

    assert payload["score"] == 73.4
    assert payload["tier"]["id"] == "htn"
    assert payload["tier"]["name"] == "HTN"
    assert len(payload["stats"]) == 6
    assert payload["stats"][0] == {"key": "jaw", "label": "ЧЕЛЮСТЬ", "value": 77}
    assert payload["notes"] == ["Тестовая заметка.", "Это мем, а не диагноз."]
    assert payload["face_box"] == {"x": 0.2, "y": 0.15, "width": 0.4, "height": 0.5}
    assert payload["confidence"] == 0.97
    assert "мем" in payload["disclaimer"].lower()
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_rate_rejects_oversized_request(client: TestClient) -> None:
    response = client.post(
        "/api/rate",
        content=b"{}",
        headers={
            "content-type": "application/json",
            "content-length": str(9_000_001),
        },
    )
    assert response.status_code == 413

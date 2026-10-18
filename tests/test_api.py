from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
PAYLOAD = {
    "Time": 10000,
    **{f"V{i}": 0.0 for i in range(1, 29)},
    "Amount": 499.0,
}


def test_health_and_demo_metrics_are_explicit() -> None:
    health = client.get("/health")
    metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["mode"] in {"synthetic_demo", "kaggle_trained"}
    if health.json()["mode"] == "synthetic_demo":
        assert "unavailable" in metrics.json()["message"]


def test_prediction_contract_and_transaction_feed(monkeypatch, tmp_path) -> None:
    import src.store

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "predictions.db"))
    response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert body["explanation_type"] in {"synthetic_demo_signals", "shap"}
    assert body["prediction_id"] > 0
    assert client.get("/transactions").json()["items"][0]["id"] == body["prediction_id"]


def test_prediction_rejects_negative_amount() -> None:
    payload = {**PAYLOAD, "Amount": -1}

    assert client.post("/predict", json=payload).status_code == 422

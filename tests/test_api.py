import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "random_forest.pkl")),
    reason="Trained model artifacts not found — run the Day 5-7 notebooks/scripts first.",
)


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


VALID_PAYLOAD = {
    "sender_bank": "HDFC", "receiver_bank": "SBI", "transaction_amount": 850.0,
    "timestamp": "2026-07-18T14:30:00", "transaction_location": "Lucknow",
    "device_type": "Android", "is_new_beneficiary": False,
    "location_mismatch_flag": False, "device_change_flag": False,
    "transactions_last_24h": 2,
}


def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"Fraud", "No Fraud"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert "RandomForest" in body["model_used"]



def test_predict_high_risk_transaction_flags_fraud(client):
    high_risk = {
        **VALID_PAYLOAD,
        "transaction_amount": 48000.0,
        "timestamp": "2026-07-18T02:15:00",
        "is_new_beneficiary": True,
        "location_mismatch_flag": True,
        "device_change_flag": True,
        "transactions_last_24h": 9,
    }
    r = client.post("/predict", json=high_risk)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "Fraud"
    assert "explanation" in body
    assert len(body["explanation"]["risk_drivers"]) > 0


def test_predict_explain_endpoint(client):
    r = client.post("/predict/explain", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "explanation" in body
    assert "summary" in body["explanation"]
    assert isinstance(body["explanation"]["risk_drivers"], list)
    assert isinstance(body["explanation"]["trust_factors"], list)


def test_rule_engine_blacklist_blocks_transaction(client):
    blacklisted_payload = {**VALID_PAYLOAD, "sender_bank": "SUSPICIOUS_BANK_TEST"}
    r = client.post("/predict", json=blacklisted_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "Fraud"
    assert body["rule_summary"]["max_severity"] == "CRITICAL"
    assert body["rule_summary"]["triggered_rules"][0]["rule_id"] == "RULE_001_BLACKLIST"


def test_get_and_update_rules_endpoints(client):
    r_get = client.get("/rules")
    assert r_get.status_code == 200
    cfg = r_get.json()
    assert "max_24h_velocity" in cfg

    updated_payload = {**cfg, "max_24h_velocity": 4}
    r_post = client.post("/rules/update", json=updated_payload)
    assert r_post.status_code == 200
    assert r_post.json()["max_24h_velocity"] == 4


def test_live_stream_with_attack_scenario(client):
    with client.websocket_connect("/ws/live?speed=0.01&scenario=blacklist") as ws:
        data = ws.receive_json()
        assert data["prediction"] == "Fraud"
        assert data["transaction"]["sender_bank"] == "SUSPICIOUS_BANK_TEST"





def test_predict_rejects_invalid_device_type(client):
    bad_payload = {**VALID_PAYLOAD, "device_type": "Fax Machine"}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422  # Pydantic validation error


def test_predict_rejects_negative_amount(client):
    bad_payload = {**VALID_PAYLOAD, "transaction_amount": -100}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_live_stream_returns_scored_transactions(client):
    with client.websocket_connect("/ws/live?speed=0.01") as ws:
        seen_ids = []
        for _ in range(5):
            data = ws.receive_json()
            assert data["prediction"] in {"Fraud", "No Fraud"}
            assert 0.0 <= data["fraud_probability"] <= 1.0
            assert "transaction" in data
            assert "transaction_amount" in data["transaction"]
            seen_ids.append(data["id"])
        # IDs should be sequential and increasing
        assert seen_ids == sorted(seen_ids)
        assert len(set(seen_ids)) == len(seen_ids)


def test_analytics_summary_reflects_logged_predictions(client, monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "analytics_test.db"))
    client.post("/predict", json=VALID_PAYLOAD)
    client.post("/predict", json=VALID_PAYLOAD)

    r = client.get("/analytics/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total_scanned"] >= 2
    assert "fraud_rate" in body
    assert "avg_fraud_probability" in body


def test_analytics_recent_returns_predictions(client, monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "analytics_recent_test.db"))
    client.post("/predict", json=VALID_PAYLOAD)

    r = client.get("/analytics/recent?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) >= 1
    assert body["predictions"][0]["sender_bank"] == "HDFC"


def test_analytics_recent_rejects_invalid_limit(client):
    r = client.get("/analytics/recent?limit=1000")
    assert r.status_code == 422


def test_analytics_locations_endpoint(client, monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "analytics_loc_test.db"))
    client.post("/predict", json=VALID_PAYLOAD)
    r = client.get("/analytics/locations")
    assert r.status_code == 200
    body = r.json()
    assert "locations" in body
    assert isinstance(body["locations"], list)
    assert len(body["locations"]) >= 1
    assert "coordinates" in body["locations"][0]


def test_analytics_hourly_endpoint(client, monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "analytics_hourly_test.db"))
    client.post("/predict", json=VALID_PAYLOAD)
    r = client.get("/analytics/hourly")
    assert r.status_code == 200
    body = r.json()
    assert "hourly" in body
    assert len(body["hourly"]) == 24


def test_analytics_export_csv_endpoint(client, monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "analytics_export_test.db"))
    client.post("/predict", json=VALID_PAYLOAD)
    r = client.get("/analytics/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "id,logged_at" in r.text
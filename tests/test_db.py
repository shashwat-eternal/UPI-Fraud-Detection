import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import app.db as db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test_predictions.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_file))
    db.init_db()
    return db_file


SAMPLE_TXN = {
    "sender_bank": "HDFC", "receiver_bank": "SBI", "transaction_amount": 850.0,
    "timestamp": "2026-07-18T14:30:00", "transaction_location": "Lucknow",
    "device_type": "Android", "is_new_beneficiary": False,
    "location_mismatch_flag": False, "device_change_flag": False,
    "transactions_last_24h": 2,
}
SAMPLE_RESULT = {
    "prediction": "No Fraud", "fraud_probability": 0.05,
    "message": "No Fraud: Details Verified and Processed", "model_used": "RandomForest",
}


def test_init_db_creates_table(temp_db):
    assert os.path.exists(temp_db)


def test_log_and_retrieve_prediction(temp_db):
    db.log_prediction(SAMPLE_TXN, SAMPLE_RESULT, source="predict")
    rows = db.get_recent_predictions(limit=10)
    assert len(rows) == 1
    assert rows[0]["sender_bank"] == "HDFC"
    assert rows[0]["prediction"] == "No Fraud"
    assert rows[0]["source"] == "predict"


def test_boolean_flags_stored_as_integers(temp_db):
    txn = {**SAMPLE_TXN, "is_new_beneficiary": True, "location_mismatch_flag": True}
    db.log_prediction(txn, SAMPLE_RESULT, source="predict")
    row = db.get_recent_predictions(limit=1)[0]
    assert row["is_new_beneficiary"] == 1
    assert row["location_mismatch_flag"] == 1
    assert row["device_change_flag"] == 0


def test_recent_predictions_newest_first(temp_db):
    for i in range(3):
        db.log_prediction({**SAMPLE_TXN, "transaction_amount": 100.0 + i}, SAMPLE_RESULT, source="predict")
    rows = db.get_recent_predictions(limit=10)
    assert len(rows) == 3
    # Most recently inserted (amount=102) should come first
    assert rows[0]["transaction_amount"] == 102.0
    assert rows[2]["transaction_amount"] == 100.0


def test_recent_predictions_respects_limit(temp_db):
    for i in range(10):
        db.log_prediction(SAMPLE_TXN, SAMPLE_RESULT, source="predict")
    rows = db.get_recent_predictions(limit=3)
    assert len(rows) == 3


def test_summary_stats_empty_db(temp_db):
    stats = db.get_summary_stats()
    assert stats == {
        "total_scanned": 0, "total_fraud": 0,
        "fraud_rate": 0.0, "avg_fraud_probability": 0.0,
    }


def test_summary_stats_computed_correctly(temp_db):
    fraud_result = {**SAMPLE_RESULT, "prediction": "Fraud", "fraud_probability": 0.95}
    safe_result = {**SAMPLE_RESULT, "prediction": "No Fraud", "fraud_probability": 0.05}

    db.log_prediction(SAMPLE_TXN, fraud_result, source="predict")
    db.log_prediction(SAMPLE_TXN, safe_result, source="predict")
    db.log_prediction(SAMPLE_TXN, safe_result, source="stream")
    db.log_prediction(SAMPLE_TXN, safe_result, source="stream")

    stats = db.get_summary_stats()
    assert stats["total_scanned"] == 4
    assert stats["total_fraud"] == 1
    assert stats["fraud_rate"] == 0.25
    assert stats["avg_fraud_probability"] == pytest.approx(0.275, abs=0.001)


def test_clear_all_removes_everything(temp_db):
    db.log_prediction(SAMPLE_TXN, SAMPLE_RESULT, source="predict")
    assert db.get_summary_stats()["total_scanned"] == 1
    db.clear_all()
    assert db.get_summary_stats()["total_scanned"] == 0


def test_source_field_distinguishes_predict_vs_stream(temp_db):
    db.log_prediction(SAMPLE_TXN, SAMPLE_RESULT, source="predict")
    db.log_prediction(SAMPLE_TXN, SAMPLE_RESULT, source="stream")
    rows = db.get_recent_predictions(limit=10)
    sources = {row["source"] for row in rows}
    assert sources == {"predict", "stream"}
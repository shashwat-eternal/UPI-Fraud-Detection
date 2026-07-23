import os
import sqlite3
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "predictions.db")


def _db_path() -> str:
    """Resolved fresh on every call so tests can override via env var
    without needing to reload this module."""
    return os.environ.get("DATABASE_PATH", DEFAULT_DB_PATH)


def _ensure_schema(conn: sqlite3.Connection):
    """Idempotent schema creation, run on every connection. This means logging
    and reading work correctly even if init_db() was never explicitly called —
    important because some test setups and ASGI server configurations don't
    reliably fire FastAPI's startup event."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT NOT NULL,
            sender_bank TEXT,
            receiver_bank TEXT,
            transaction_amount REAL,
            transaction_timestamp TEXT,
            transaction_location TEXT,
            device_type TEXT,
            is_new_beneficiary INTEGER,
            location_mismatch_flag INTEGER,
            device_change_flag INTEGER,
            transactions_last_24h INTEGER,
            prediction TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            model_used TEXT,
            source TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_logged_at ON predictions(logged_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions(prediction)")
    conn.commit()


def _seed_initial_telemetry(conn: sqlite3.Connection):
    """Seed initial sample telemetry records so Threat Analytics and Map display
    data immediately upon deployment even before user predictions are logged."""
    if os.environ.get("DATABASE_PATH") or os.environ.get("PYTEST_CURRENT_TEST"):
        return

    cursor = conn.execute("SELECT COUNT(*) FROM predictions")
    if cursor.fetchone()[0] > 0:
        return

    sample_locations = [
        ("Delhi", 18, 4, 0.22),
        ("Mumbai", 22, 3, 0.14),
        ("Bengaluru", 16, 2, 0.08),
        ("Lucknow", 12, 5, 0.42),
        ("Hyderabad", 15, 3, 0.18),
        ("Kolkata", 11, 2, 0.15),
        ("Pune", 13, 1, 0.07),
        ("Jaipur", 10, 3, 0.30),
        ("Rural-UP", 14, 7, 0.50),
        ("Rural-Bihar", 12, 6, 0.50),
        ("Chennai", 14, 2, 0.12),
        ("Patna", 10, 4, 0.40),
    ]

    records = []
    now = datetime.now(timezone.utc)
    for loc, count, fraud_cnt, base_prob in sample_locations:
        for i in range(count):
            is_fraud = i < fraud_cnt
            prob = base_prob + (0.15 if is_fraud else -0.05)
            prob = max(0.01, min(0.99, round(prob, 2)))
            pred = "Fraud" if is_fraud else "No Fraud"
            records.append((
                now.isoformat(),
                "HDFC", "SBI", 3500.0 if is_fraud else 450.0,
                now.isoformat(), loc, "Android",
                1 if is_fraud else 0,
                1 if is_fraud else 0,
                1 if is_fraud else 0,
                8 if is_fraud else 2,
                pred, prob, "Random Forest", "seed"
            ))

    conn.executemany("""
        INSERT INTO predictions (
            logged_at, sender_bank, receiver_bank, transaction_amount,
            transaction_timestamp, transaction_location, device_type,
            is_new_beneficiary, location_mismatch_flag, device_change_flag,
            transactions_last_24h, prediction, fraud_probability, model_used, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()


def _get_connection() -> sqlite3.Connection:
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def init_db():
    """Explicit convenience wrapper — schema creation actually happens lazily
    on every _get_connection() call, so this is safe to call any number of
    times (e.g. once on FastAPI startup) purely for clarity/documentation."""
    conn = _get_connection()
    _seed_initial_telemetry(conn)
    conn.close()


def log_prediction(txn: dict, result: dict, source: str = "predict"):
    """Insert one prediction record. `txn` is the raw transaction payload
    (dict, e.g. from a Pydantic model's .model_dump()), `result` is
    predict_transaction()'s output, `source` is 'predict' or 'stream'."""
    conn = _get_connection()
    try:
        conn.execute("""
            INSERT INTO predictions (
                logged_at, sender_bank, receiver_bank, transaction_amount,
                transaction_timestamp, transaction_location, device_type,
                is_new_beneficiary, location_mismatch_flag, device_change_flag,
                transactions_last_24h, prediction, fraud_probability, model_used, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            txn.get("sender_bank"),
            txn.get("receiver_bank"),
            txn.get("transaction_amount"),
            str(txn.get("timestamp")),
            txn.get("transaction_location"),
            txn.get("device_type"),
            int(bool(txn.get("is_new_beneficiary"))),
            int(bool(txn.get("location_mismatch_flag"))),
            int(bool(txn.get("device_change_flag"))),
            txn.get("transactions_last_24h"),
            result.get("prediction"),
            result.get("fraud_probability"),
            result.get("model_used"),
            source,
        ))
        conn.commit()
    finally:
        conn.close()


def get_recent_predictions(limit: int = 50) -> list[dict]:
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_summary_stats() -> dict:
    conn = _get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        fraud = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE prediction = 'Fraud'"
        ).fetchone()["c"]
        avg_prob_row = conn.execute("SELECT AVG(fraud_probability) AS a FROM predictions").fetchone()
        avg_prob = avg_prob_row["a"] if avg_prob_row["a"] is not None else 0.0

        return {
            "total_scanned": total,
            "total_fraud": fraud,
            "fraud_rate": round(fraud / total, 4) if total else 0.0,
            "avg_fraud_probability": round(avg_prob, 4),
        }
    finally:
        conn.close()


def clear_all():
    """Wipe all logged predictions. Used by tests; also handy for demos
    where you want to reset the counters before a fresh run."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    finally:
        conn.close()


LOCATION_COORDS = {
    "Lucknow": [26.8467, 80.9462],
    "Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Bengaluru": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
    "Chennai": [13.0827, 80.2707],
    "Kolkata": [22.5726, 88.3639],
    "Pune": [18.5204, 73.8567],
    "Jaipur": [26.9124, 75.7873],
    "Patna": [25.5941, 85.1376],
    "Bhopal": [23.2599, 77.4126],
    "Kanpur": [26.4499, 80.3319],
    "Rural-UP": [27.1767, 78.0081],
    "Rural-Bihar": [26.1542, 85.8918],
    "Rural-MP": [23.1815, 75.7772],
    "Rural-Rajasthan": [26.2389, 73.0243],
}


def get_location_stats() -> list[dict]:
    """Aggregate total scanned and fraud count per location with lat/lon coords."""
    conn = _get_connection()
    try:
        rows = conn.execute("""
            SELECT 
                transaction_location AS location,
                COUNT(*) AS total_scanned,
                SUM(CASE WHEN prediction = 'Fraud' THEN 1 ELSE 0 END) AS total_fraud,
                AVG(fraud_probability) AS avg_fraud_probability
            FROM predictions
            WHERE transaction_location IS NOT NULL
            GROUP BY transaction_location
        """).fetchall()

        results = []
        for r in rows:
            loc = r["location"]
            total = r["total_scanned"]
            fraud = r["total_fraud"] or 0
            coords = LOCATION_COORDS.get(loc, [20.5937, 78.9629])
            results.append({
                "location": loc,
                "coordinates": coords,
                "total_scanned": total,
                "total_fraud": fraud,
                "fraud_rate": round(fraud / total, 4) if total else 0.0,
                "avg_fraud_probability": round(r["avg_fraud_probability"] or 0.0, 4),
            })
        return results
    finally:
        conn.close()


def get_hourly_stats() -> list[dict]:
    """Computes hourly transaction volume and fraud rate distribution."""
    conn = _get_connection()
    try:
        rows = conn.execute("""
            SELECT 
                CAST(strftime('%H', logged_at) AS INTEGER) AS hour,
                COUNT(*) AS total_scanned,
                SUM(CASE WHEN prediction = 'Fraud' THEN 1 ELSE 0 END) AS total_fraud
            FROM predictions
            GROUP BY hour
            ORDER BY hour ASC
        """).fetchall()

        hourly_map = {r["hour"]: dict(r) for r in rows}
        results = []
        for h in range(24):
            item = hourly_map.get(h, {"hour": h, "total_scanned": 0, "total_fraud": 0})
            total = item["total_scanned"]
            fraud = item["total_fraud"]
            results.append({
                "hour": h,
                "total_scanned": total,
                "total_fraud": fraud,
                "fraud_rate": round(fraud / total, 4) if total else 0.0,
            })
        return results
    finally:
        conn.close()


def export_predictions_csv() -> str:
    """Generate CSV text formatted string of all logged predictions."""
    conn = _get_connection()
    try:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id ASC").fetchall()
        if not rows:
            return "id,logged_at,sender_bank,receiver_bank,transaction_amount,transaction_location,prediction,fraud_probability,model_used,source\n"

        headers = list(rows[0].keys())
        csv_lines = [",".join(headers)]
        for r in rows:
            line = [str(r[h]) if r[h] is not None else "" for h in headers]
            csv_lines.append(",".join(line))

        return "\n".join(csv_lines)
    finally:
        conn.close()
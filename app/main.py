from contextlib import asynccontextmanager

import os
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas import TransactionRequest, PredictionResponse, RuleConfigSchema
from app.predict import predict_transaction, load_artifacts
from app.rules import get_rule_config, update_rule_config, RuleConfig
from app.stream import transaction_stream
from app.db import (
    init_db, log_prediction, get_recent_predictions, get_summary_stats,
    get_location_stats, get_hourly_stats, export_predictions_csv
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if not os.path.exists(FRONTEND_DIR):
    FRONTEND_DIR = os.path.join(os.getcwd(), "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_artifacts()  # Preload models into RAM at server startup
    yield



app = FastAPI(
    title="UPI Fraud Detection API",
    description="Classifies UPI transactions as Fraud / No Fraud using a "
                 "Random Forest model trained on engineered behavioral, "
                 "time-based, and anomaly-detection features.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/console")
@app.get("/index.html")
def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "UPI Fraud Detection API is running.",
        "docs": "/docs",
        "health": "/health",
        "analytics": "/analytics.html",
        "live": "/live.html",
    }


@app.get("/analytics")
@app.get("/analytics.html")
def analytics_page():
    analytics_path = os.path.join(FRONTEND_DIR, "analytics.html")
    if os.path.exists(analytics_path):
        return FileResponse(analytics_path)
    raise HTTPException(status_code=404, detail="analytics.html not found")


@app.get("/live")
@app.get("/live.html")
def live_page():
    live_path = os.path.join(FRONTEND_DIR, "live.html")
    if os.path.exists(live_path):
        return FileResponse(live_path)
    raise HTTPException(status_code=404, detail="live.html not found")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    try:
        payload = transaction.model_dump()
        result = predict_transaction(payload)
        log_prediction(payload, result, source="predict")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/explain", response_model=PredictionResponse)
def predict_explain(transaction: TransactionRequest):
    """Generates a prediction with explicit XAI feature attributions and reason codes."""
    try:
        payload = transaction.model_dump()
        result = predict_transaction(payload, include_explanation=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")


@app.get("/rules", response_model=RuleConfigSchema)

def get_rules():
    """Retrieve active business rule threshold configuration."""
    config = get_rule_config()
    return config.model_dump()


@app.post("/rules/update", response_model=RuleConfigSchema)
def update_rules(config: RuleConfigSchema):
    """Dynamically update active business rule limits and blacklists."""
    new_cfg = RuleConfig(**config.model_dump())
    updated = update_rule_config(new_cfg)
    return updated.model_dump()




@app.websocket("/ws/live")
async def live_transaction_feed(websocket: WebSocket, speed: float = 1.5, scenario: str = "random"):
    """Streams simulated transactions, each scored in real time by the same
    model and pipeline used by /predict. Connect with e.g.:
        ws://127.0.0.1:8000/ws/live?speed=1.0&scenario=sim_swap
    `speed` is the delay in seconds between transactions (default 1.5s).
    """
    await websocket.accept()
    try:
        scen_param = scenario if scenario != "random" else None
        async for record in transaction_stream(interval_seconds=speed, scenario=scen_param):
            await websocket.send_json(record)
    except WebSocketDisconnect:
        pass



@app.get("/analytics/summary")
def analytics_summary():
    """Aggregate stats across every prediction logged so far (both from
    /predict and the live stream). Foundation for the analytics dashboard."""
    return get_summary_stats()


@app.get("/analytics/recent")
def analytics_recent(limit: int = 50):
    """Most recent logged predictions, newest first."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    return {"predictions": get_recent_predictions(limit)}


@app.get("/analytics/locations")
def analytics_locations():
    """Geographic breakdown of scanned transactions and fraud rates per location."""
    return {"locations": get_location_stats()}


@app.get("/analytics/hourly")
def analytics_hourly():
    """Hourly distribution of scanned transactions and fraud rates."""
    return {"hourly": get_hourly_stats()}


@app.get("/analytics/export")
def analytics_export():
    """Export prediction logs as a CSV file for compliance auditing."""
    csv_data = export_predictions_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=upi_fraud_audit_report.csv"}
    )


if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
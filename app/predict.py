
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd

from features.build_features import engineer_features, add_anomaly_scores
from app.explain import explain_prediction
from app.rules import evaluate_rules

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_model = None
_preprocessor = None
_isolation_forest = None
_outlier_thresholds = None


def load_artifacts():
    """Lazy-load all model artifacts once, cache in module-level globals."""
    global _model, _preprocessor, _isolation_forest, _outlier_thresholds
    if _model is None:
        _model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
        _preprocessor = joblib.load(os.path.join(MODELS_DIR, "final_preprocessor.pkl"))
        _isolation_forest = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
        _outlier_thresholds = joblib.load(os.path.join(MODELS_DIR, "outlier_thresholds.pkl"))
    return _model, _preprocessor, _isolation_forest, _outlier_thresholds


def build_feature_row(txn: dict) -> pd.DataFrame:
    """Reconstruct the same columns Day 3's basic_clean() produces, for one row."""
    df = pd.DataFrame([txn])
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.day_name()
    df["transaction_amount_log"] = np.log1p(df["transaction_amount"])
    df["is_new_beneficiary"] = df["is_new_beneficiary"].astype(int)
    df["location_mismatch_flag"] = df["location_mismatch_flag"].astype(int)
    df["device_change_flag"] = df["device_change_flag"].astype(int)
    df = df.drop(columns=["timestamp", "transaction_amount"])
    return df


def predict_transaction(txn: dict, include_explanation: bool = True) -> dict:
    model, preprocessor, iso_forest, thresholds = load_artifacts()

    X = build_feature_row(txn)
    X_fe, _ = engineer_features(X, outlier_thresholds=thresholds)
    X_fe = add_anomaly_scores(X_fe, iso_forest)

    feature_names = list(preprocessor.get_feature_names_out())
    X_proc = preprocessor.transform(X_fe)
    X_proc = pd.DataFrame(X_proc, columns=feature_names)
    ml_fraud_probability = float(model.predict_proba(X_proc)[0, 1])

    # Evaluate Hybrid Rule Engine
    rule_res = evaluate_rules(txn)
    rule_score = rule_res["rule_risk_score"]

    # Composite risk score taking max of ML score & Rule score
    composite_risk = max(ml_fraud_probability, rule_score)
    prediction = "Fraud" if composite_risk >= 0.5 else "No Fraud"

    if rule_res["max_severity"] == "CRITICAL":
        message = f"Fraud: Hard Policy Violation ({rule_res['triggered_rules'][0]['rule_name']})"
    elif prediction == "Fraud":
        message = "Fraud: Unusual Activity Detected"
    else:
        message = "No Fraud: Details Verified and Processed"

    result = {
        "prediction": prediction,
        "fraud_probability": round(composite_risk, 4),
        "composite_risk_score": round(composite_risk, 4),
        "message": message,
        "model_used": "RandomForest + HybridRuleEngine",
        "rule_summary": rule_res,
    }

    if include_explanation:
        result["explanation"] = explain_prediction(
            txn=txn,
            X_fe=X_fe,
            X_proc=X_proc,
            model=model,
            feature_names=feature_names,
            fraud_probability=composite_risk,
        )

    return result



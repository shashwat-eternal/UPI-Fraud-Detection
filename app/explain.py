from typing import Any, Dict, List
import pandas as pd
import numpy as np


HUMAN_REASON_MAP = {
    "is_new_beneficiary": {
        "risk": "First-time transfer to an unverified beneficiary",
        "trust": "Transfer to a known/verified beneficiary",
    },
    "location_mismatch_flag": {
        "risk": "Geographic location mismatch detected (IP vs Home location)",
        "trust": "Transaction location matches regular user pattern",
    },
    "device_change_flag": {
        "risk": "Transaction executed from a new or un-registered device",
        "trust": "Executed from a verified primary device",
    },
    "is_night": {
        "risk": "High-risk late night transaction (23:00 - 04:00)",
        "trust": "Standard daytime transaction window",
    },
    "is_large_amount": {
        "risk": "Unusually large transfer amount relative to baseline",
        "trust": "Transaction amount within normal threshold",
    },
    "isolation_forest_flag": {
        "risk": "Flagged as an anomaly by unsupervised Isolation Forest",
        "trust": "Normal behavioral distribution across unsupervised features",
    },
    "transactions_last_24h": {
        "risk": "High 24-hour transaction velocity",
        "trust": "Low 24-hour transaction frequency",
    },
    "anomaly_score": {
        "risk": "Elevated statistical anomaly index",
        "trust": "Low anomaly index score",
    },
    "amount_velocity_ratio": {
        "risk": "High amount-to-velocity ratio anomaly",
        "trust": "Balanced transaction amount and frequency",
    },
    "risk_flag_count": {
        "risk": "Multiple concurrent security flags triggered",
        "trust": "Clean risk flag profile",
    },
}


def explain_prediction(
    txn: Dict[str, Any],
    X_fe: pd.DataFrame,
    X_proc: pd.DataFrame,
    model: Any,
    feature_names: List[str],
    fraud_probability: float,
) -> Dict[str, Any]:
    """
    Computes feature contributions for a transaction and maps them to human-readable
    reason codes, categorized into risk_drivers and trust_factors.
    """

    # Get model feature importances if available
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        importances = np.ones(len(feature_names)) / len(feature_names)

    feature_imp_map = dict(zip(feature_names, importances))

    risk_drivers = []
    trust_factors = []

    # 1. Analyze Flag Features
    flags = [
        "is_new_beneficiary",
        "location_mismatch_flag",
        "device_change_flag",
        "is_night",
        "is_large_amount",
        "isolation_forest_flag",
    ]

    for flag in flags:
        if flag in X_fe.columns:
            is_active = bool(X_fe[flag].iloc[0] == 1)
            # Find matching encoded feature weight in preprocessor output
            imp_keys = [k for k in feature_names if flag in k]
            weight = sum(feature_imp_map[k] for k in imp_keys) if imp_keys else 0.05

            if is_active:
                impact = round(min(0.99, float(weight * 3.5 + 0.15)), 2)
                risk_drivers.append({
                    "feature_name": flag,
                    "impact_score": impact,
                    "reason_code": HUMAN_REASON_MAP.get(flag, {}).get("risk", f"High risk on {flag}"),
                    "category": "risk",
                })
            else:
                impact = round(min(0.99, float(weight * 2.0 + 0.05)), 2)
                trust_factors.append({
                    "feature_name": flag,
                    "impact_score": impact,
                    "reason_code": HUMAN_REASON_MAP.get(flag, {}).get("trust", f"Normal state for {flag}"),
                    "category": "trust",
                })

    # 2. Analyze Velocity & Numeric Features
    txns_24h = int(txn.get("transactions_last_24h", 0))
    if txns_24h >= 5:
        risk_drivers.append({
            "feature_name": "transactions_last_24h",
            "impact_score": round(min(0.95, 0.2 + (txns_24h * 0.06)), 2),
            "reason_code": f"High 24h frequency: {txns_24h} transactions logged in last 24h",
            "category": "risk",
        })
    elif txns_24h <= 2:
        trust_factors.append({
            "feature_name": "transactions_last_24h",
            "impact_score": 0.15,
            "reason_code": f"Low 24h frequency: only {txns_24h} transaction(s) logged",
            "category": "trust",
        })

    amount = float(txn.get("transaction_amount", 0))
    if amount >= 25000:
        risk_drivers.append({
            "feature_name": "transaction_amount",
            "impact_score": round(min(0.98, 0.3 + (amount / 100000.0)), 2),
            "reason_code": f"High value transfer: ₹{amount:,.2f} exceeds standard safety threshold",
            "category": "risk",
        })
    elif amount <= 5000:
        trust_factors.append({
            "feature_name": "transaction_amount",
            "impact_score": 0.20,
            "reason_code": f"Moderate transfer amount: ₹{amount:,.2f} within standard limits",
            "category": "trust",
        })

    if "anomaly_score" in X_fe.columns:
        anom_val = float(X_fe["anomaly_score"].iloc[0])
        if anom_val > 0.4:
            risk_drivers.append({
                "feature_name": "anomaly_score",
                "impact_score": round(min(0.99, anom_val), 2),
                "reason_code": f"Isolation Forest anomaly score elevated ({anom_val:.2f})",
                "category": "risk",
            })

    # Sort drivers & trust factors by impact score (descending)
    risk_drivers.sort(key=lambda x: x["impact_score"], reverse=True)
    trust_factors.sort(key=lambda x: x["impact_score"], reverse=True)

    summary_statement = (
        f"Fraud risk ({fraud_probability*100:.1f}%) driven primarily by {len(risk_drivers)} security risk flags."
        if fraud_probability >= 0.5
        else f"Transaction verified safe ({fraud_probability*100:.1f}% risk) backed by {len(trust_factors)} trust indicators."
    )

    return {
        "risk_drivers": risk_drivers[:4],
        "trust_factors": trust_factors[:3],
        "summary": summary_statement,
    }

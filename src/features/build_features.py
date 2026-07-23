

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

RANDOM_SEED = 42

RAW_NUMERIC_FOR_ANOMALY = [
    "transaction_amount_log", "transactions_last_24h",
    "is_new_beneficiary", "location_mismatch_flag", "device_change_flag",
]


def add_time_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["is_night"] = X["hour"].apply(lambda h: 1 if (h >= 23 or h <= 4) else 0)
    X["is_weekend"] = X["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return X


def add_behavioral_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["risk_flag_count"] = (
        X["is_new_beneficiary"] + X["location_mismatch_flag"] + X["device_change_flag"]
    )
    X["amount_velocity_ratio"] = X["transaction_amount_log"] / (X["transactions_last_24h"] + 1)
    return X


def add_outlier_flags(X: pd.DataFrame, thresholds: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Percentile-based statistical outlier flags. Thresholds are computed on
    train and reused on test (pass `thresholds` back in for the test set)."""
    X = X.copy()
    if thresholds is None:
        thresholds = {
            "large_amount": X["transaction_amount_log"].quantile(0.95),
            "micro_amount": X["transaction_amount_log"].quantile(0.05),
        }
    X["is_large_amount"] = (X["transaction_amount_log"] >= thresholds["large_amount"]).astype(int)
    X["is_micro_amount"] = (X["transaction_amount_log"] <= thresholds["micro_amount"]).astype(int)
    return X, thresholds


def engineer_features(X: pd.DataFrame, outlier_thresholds: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Apply all deterministic (non-fitted) feature engineering steps."""
    X = add_time_features(X)
    X = add_behavioral_features(X)
    X, thresholds = add_outlier_flags(X, outlier_thresholds)
    return X, thresholds


def fit_anomaly_detector(X_train: pd.DataFrame, contamination: float = 0.08) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(X_train[RAW_NUMERIC_FOR_ANOMALY])
    return model


def add_anomaly_scores(X: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    X = X.copy()
    # Higher score_samples = more normal, so negate for an intuitive "anomaly score"
    X["anomaly_score"] = -model.score_samples(X[RAW_NUMERIC_FOR_ANOMALY])
    # predict(): -1 = outlier, 1 = inlier -> convert to 1 = anomalous flag
    X["isolation_forest_flag"] = (model.predict(X[RAW_NUMERIC_FOR_ANOMALY]) == -1).astype(int)
    return X


FINAL_NUMERIC_COLS = [
    "transaction_amount_log", "hour_sin", "hour_cos", "transactions_last_24h",
    "risk_flag_count", "amount_velocity_ratio", "anomaly_score",
]
FINAL_CATEGORICAL_COLS = [
    "sender_bank", "receiver_bank", "transaction_location", "device_type", "day_of_week",
]
FINAL_FLAG_COLS = [
    "is_new_beneficiary", "location_mismatch_flag", "device_change_flag",
    "is_night", "is_weekend", "is_large_amount", "is_micro_amount", "isolation_forest_flag",
]


def build_final_preprocessor():
    """ColumnTransformer over the full engineered feature set (Day 3 + Day 4 features)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    flag_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, FINAL_NUMERIC_COLS),
        ("cat", categorical_pipeline, FINAL_CATEGORICAL_COLS),
        ("flag", flag_pipeline, FINAL_FLAG_COLS),
    ])

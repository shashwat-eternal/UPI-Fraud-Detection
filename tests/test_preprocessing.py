"""
test_preprocessing.py
-----------------------
Unit tests for src/data/preprocess.py and src/features/build_features.py.
Run with: pytest tests/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from data.preprocess import load_raw, basic_clean, build_preprocessor
from features.build_features import (
    add_time_features, add_behavioral_features, add_outlier_flags,
    engineer_features, fit_anomaly_detector, add_anomaly_scores,
)

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "upi_transactions.csv")


@pytest.fixture(scope="module")
def raw_df():
    if not os.path.exists(RAW_PATH):
        pytest.skip("Raw dataset not found — run src/data/generate_dataset.py first.")
    return load_raw(RAW_PATH).head(2000)  # small slice, tests should be fast


@pytest.fixture(scope="module")
def cleaned(raw_df):
    return basic_clean(raw_df)


def test_basic_clean_no_id_columns(cleaned):
    X, y = cleaned
    for col in ["transaction_id", "sender_account_name", "receiver_account_name", "timestamp"]:
        assert col not in X.columns


def test_basic_clean_target_is_binary(cleaned):
    X, y = cleaned
    assert set(y.unique()).issubset({0, 1})


def test_basic_clean_no_missing_values(cleaned):
    X, y = cleaned
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


def test_time_features_add_expected_columns(cleaned):
    X, y = cleaned
    X_fe = add_time_features(X)
    for col in ["hour_sin", "hour_cos", "is_night", "is_weekend"]:
        assert col in X_fe.columns


def test_hour_sin_cos_bounded(cleaned):
    X, y = cleaned
    X_fe = add_time_features(X)
    assert X_fe["hour_sin"].between(-1, 1).all()
    assert X_fe["hour_cos"].between(-1, 1).all()


def test_behavioral_features_risk_flag_range(cleaned):
    X, y = cleaned
    X_fe = add_behavioral_features(X)
    assert X_fe["risk_flag_count"].between(0, 3).all()


def test_outlier_flags_are_binary(cleaned):
    X, y = cleaned
    X_fe, thresholds = add_outlier_flags(X)
    assert set(X_fe["is_large_amount"].unique()).issubset({0, 1})
    assert set(X_fe["is_micro_amount"].unique()).issubset({0, 1})
    assert "large_amount" in thresholds and "micro_amount" in thresholds


def test_outlier_thresholds_reused_consistently(cleaned):
    """Thresholds computed on one slice should be reusable on another without recomputation."""
    X, y = cleaned
    _, thresholds = add_outlier_flags(X.iloc[:1000])
    X_fe2, thresholds2 = add_outlier_flags(X.iloc[1000:], thresholds)
    assert thresholds == thresholds2


def test_engineer_features_end_to_end(cleaned):
    X, y = cleaned
    X_fe, thresholds = engineer_features(X)
    expected_new_cols = [
        "hour_sin", "hour_cos", "is_night", "is_weekend",
        "risk_flag_count", "amount_velocity_ratio", "is_large_amount", "is_micro_amount",
    ]
    for col in expected_new_cols:
        assert col in X_fe.columns
    assert X_fe.isnull().sum().sum() == 0


def test_anomaly_detector_fits_and_scores(cleaned):
    X, y = cleaned
    X_fe, _ = engineer_features(X)
    model = fit_anomaly_detector(X_fe, contamination=0.08)
    X_scored = add_anomaly_scores(X_fe, model)
    assert "anomaly_score" in X_scored.columns
    assert "isolation_forest_flag" in X_scored.columns
    assert set(X_scored["isolation_forest_flag"].unique()).issubset({0, 1})
    assert not X_scored["anomaly_score"].isnull().any()


def test_preprocessor_output_shape(cleaned):
    X, y = cleaned
    X_fe, _ = engineer_features(X)
    model = fit_anomaly_detector(X_fe, contamination=0.08)
    X_fe = add_anomaly_scores(X_fe, model)

    # Use the Day 4 final column set via build_features' own preprocessor builder
    from features.build_features import build_final_preprocessor
    preprocessor = build_final_preprocessor()
    X_proc = preprocessor.fit_transform(X_fe)

    assert X_proc.shape[0] == len(X_fe)
    assert not np.isnan(X_proc).any()

"""
test_generic_pipeline.py
--------------------------
Confirms the config-driven pipeline (src/data/generic_pipeline.py) produces
valid, leakage-free output for BOTH registered datasets — proving the
pipeline is genuinely schema-agnostic, not just working by coincidence on
the one dataset it was designed for.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from data.dataset_config import DATASET_REGISTRY
from data.generic_pipeline import load_raw, clean_dataset, engineer_features, run_pipeline

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _dataset_available(config):
    path = config.path if os.path.isabs(config.path) else os.path.join(PROJECT_ROOT, config.path)
    return os.path.exists(path)


@pytest.mark.parametrize("dataset_key", ["upi", "creditcard"])
def test_clean_dataset_produces_binary_target(dataset_key):
    config = DATASET_REGISTRY[dataset_key]
    if not _dataset_available(config):
        pytest.skip(f"{config.path} not found")
    df = load_raw(config).head(2000)
    X, y, schema = clean_dataset(df, config)
    assert set(y.unique()).issubset({0, 1})
    assert config.target_column not in X.columns


@pytest.mark.parametrize("dataset_key", ["upi", "creditcard"])
def test_clean_dataset_no_missing_values(dataset_key):
    config = DATASET_REGISTRY[dataset_key]
    if not _dataset_available(config):
        pytest.skip(f"{config.path} not found")
    df = load_raw(config).head(2000)
    X, y, schema = clean_dataset(df, config)
    assert X.isnull().sum().sum() == 0


@pytest.mark.parametrize("dataset_key", ["upi", "creditcard"])
def test_engineer_features_respects_schema(dataset_key):
    """A dataset with no flag_columns (e.g. creditcard) should not get a
    risk_flag_count feature; a dataset with flags (upi) should."""
    config = DATASET_REGISTRY[dataset_key]
    if not _dataset_available(config):
        pytest.skip(f"{config.path} not found")
    df = load_raw(config).head(2000)
    X, y, schema = clean_dataset(df, config)
    X_fe, schema_fe, thresholds = engineer_features(X, schema, config)

    if config.flag_columns:
        assert "risk_flag_count" in X_fe.columns
    else:
        assert "risk_flag_count" not in X_fe.columns


@pytest.mark.parametrize("dataset_key", ["upi", "creditcard"])
def test_full_pipeline_end_to_end(dataset_key):
    config = DATASET_REGISTRY[dataset_key]
    if not _dataset_available(config):
        pytest.skip(f"{config.path} not found")

    X_train_bal, X_test_proc, y_train_bal, y_test, feature_names, artifacts = run_pipeline(
        config, save=False
    )

    # Balanced training set should be ~50/50
    assert abs(y_train_bal.mean() - 0.5) < 0.01
    # Test set should NOT be balanced (SMOTE must only touch training data)
    assert y_test.mean() < 0.5
    # No leakage-induced NaNs
    assert not np.isnan(X_train_bal).any()
    assert not np.isnan(X_test_proc).any()
    # Feature count matches between train/test
    assert X_train_bal.shape[1] == X_test_proc.shape[1] == len(feature_names)

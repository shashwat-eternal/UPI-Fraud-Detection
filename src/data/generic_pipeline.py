
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from data.dataset_config import DatasetConfig

RANDOM_SEED = 42


@dataclass
class ResolvedSchema:
    """Tracks which columns actually exist after cleaning/engineering, since not
    every dataset has every feature category (e.g. creditcard.csv has no
    behavioral flags, UPI has no PCA components)."""
    numeric_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)
    flag_cols: list = field(default_factory=list)


import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


def load_raw(config: DatasetConfig) -> pd.DataFrame:
    path = config.path
    if not _os.path.isabs(path) and not _os.path.exists(path):
        path = _os.path.join(_PROJECT_ROOT, config.path)
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.Series, ResolvedSchema]:
    """Generic version of Day 3's basic_clean(): drop IDs, binarize target,
    extract time/amount features, and report back exactly which columns are
    available for downstream feature engineering."""
    df = df.copy()
    y = df[config.target_column].apply(lambda v: 1 if v in config.fraud_labels else 0)

    drop_cols = list(config.id_columns) + [config.target_column]
    numeric_cols = list(config.numeric_columns)
    categorical_cols = list(config.categorical_columns)
    flag_cols = list(config.flag_columns)

    if config.amount_column:
        df["amount_log"] = np.log1p(df[config.amount_column].astype(float).clip(lower=0))
        numeric_cols.append("amount_log")
        drop_cols.append(config.amount_column)

    if config.timestamp_column:
        ts = pd.to_datetime(df[config.timestamp_column])
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.day_name()
        numeric_cols.append("hour")
        categorical_cols.append("day_of_week")
        drop_cols.append(config.timestamp_column)
    elif config.seconds_column:
        # No wall-clock timestamp, but a seconds-since-start column (like the
        # real credit card dataset's "Time") still lets us derive hour-of-day.
        df["hour"] = ((df[config.seconds_column] % 86400) // 3600).astype(int)
        numeric_cols.append("hour")
        drop_cols.append(config.seconds_column)

    if config.velocity_column:
        numeric_cols.append(config.velocity_column)

    X = df.drop(columns=drop_cols)
    schema = ResolvedSchema(numeric_cols=numeric_cols, categorical_cols=categorical_cols, flag_cols=flag_cols)
    return X, y, schema


def engineer_features(X: pd.DataFrame, schema: ResolvedSchema, config: DatasetConfig,
                       outlier_thresholds: dict | None = None) -> tuple[pd.DataFrame, ResolvedSchema, dict]:
    """Adds whichever engineered features are meaningful for this schema."""
    X = X.copy()
    schema = ResolvedSchema(list(schema.numeric_cols), list(schema.categorical_cols), list(schema.flag_cols))

    # Cyclical time encoding (only if we derived an 'hour' column)
    if "hour" in schema.numeric_cols:
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
        schema.numeric_cols = [c for c in schema.numeric_cols if c != "hour"] + ["hour_sin", "hour_cos"]

    # Behavioral aggregate (only if the dataset has pre-existing binary risk flags)
    if schema.flag_cols:
        X["risk_flag_count"] = X[schema.flag_cols].sum(axis=1)
        schema.numeric_cols.append("risk_flag_count")

    # Amount-velocity ratio (only if both amount and a velocity column exist)
    if "amount_log" in schema.numeric_cols and config.velocity_column:
        X["amount_velocity_ratio"] = X["amount_log"] / (X[config.velocity_column] + 1)
        schema.numeric_cols.append("amount_velocity_ratio")

    # Statistical outlier flags on amount (only if an amount column exists)
    if "amount_log" in schema.numeric_cols:
        if outlier_thresholds is None:
            outlier_thresholds = {
                "large_amount": X["amount_log"].quantile(0.95),
                "micro_amount": X["amount_log"].quantile(0.05),
            }
        X["is_large_amount"] = (X["amount_log"] >= outlier_thresholds["large_amount"]).astype(int)
        X["is_micro_amount"] = (X["amount_log"] <= outlier_thresholds["micro_amount"]).astype(int)
        schema.flag_cols += ["is_large_amount", "is_micro_amount"]
    else:
        outlier_thresholds = outlier_thresholds or {}

    return X, schema, outlier_thresholds


def fit_anomaly_detector(X: pd.DataFrame, schema: ResolvedSchema, contamination: float = 0.05) -> IsolationForest:
    """Fit on whatever numeric + flag columns exist — works for any schema
    with at least one numeric feature."""
    cols = schema.numeric_cols + schema.flag_cols
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X[cols])
    return model


def add_anomaly_scores(X: pd.DataFrame, schema: ResolvedSchema, model: IsolationForest) -> tuple[pd.DataFrame, ResolvedSchema]:
    X = X.copy()
    cols = schema.numeric_cols + schema.flag_cols
    X["anomaly_score"] = -model.score_samples(X[cols])
    X["isolation_forest_flag"] = (model.predict(X[cols]) == -1).astype(int)

    schema = ResolvedSchema(list(schema.numeric_cols) + ["anomaly_score"],
                             list(schema.categorical_cols),
                             list(schema.flag_cols) + ["isolation_forest_flag"])
    return X, schema


def build_preprocessor(schema: ResolvedSchema) -> ColumnTransformer:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    flag_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

    transformers = []
    if schema.numeric_cols:
        transformers.append(("num", numeric_pipeline, schema.numeric_cols))
    if schema.categorical_cols:
        transformers.append(("cat", categorical_pipeline, schema.categorical_cols))
    if schema.flag_cols:
        transformers.append(("flag", flag_pipeline, schema.flag_cols))

    return ColumnTransformer(transformers=transformers)


def run_pipeline(config: DatasetConfig, contamination: float = 0.05, save: bool = True):
    """Full pipeline, driven entirely by `config`. Returns everything needed
    to train and evaluate a model, plus the fitted artifacts for reuse."""
    df = load_raw(config)
    X, y, schema = clean_dataset(df, config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    X_train_fe, schema_engineered, thresholds = engineer_features(X_train, schema, config)
    X_test_fe, _, _ = engineer_features(X_test, schema, config, outlier_thresholds=thresholds)
    schema = schema_engineered

    iso_forest = fit_anomaly_detector(X_train_fe, schema, contamination=contamination)
    X_train_fe, schema_scored = add_anomaly_scores(X_train_fe, schema, iso_forest)
    X_test_fe, _ = add_anomaly_scores(X_test_fe, schema, iso_forest)
    schema = schema_scored

    preprocessor = build_preprocessor(schema)
    X_train_proc = preprocessor.fit_transform(X_train_fe)
    X_test_proc = preprocessor.transform(X_test_fe)
    feature_names = list(preprocessor.get_feature_names_out())

    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

    artifacts = {
        "preprocessor": preprocessor,
        "isolation_forest": iso_forest,
        "outlier_thresholds": thresholds,
        "schema": schema,
        "feature_names": feature_names,
    }

    if save:
        import os
        safe_name = config.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_dir = _os.path.join(_PROJECT_ROOT, "models", "generic", safe_name)
        os.makedirs(out_dir, exist_ok=True)
        for key in ["preprocessor", "isolation_forest", "outlier_thresholds", "schema"]:
            joblib.dump(artifacts[key], f"{out_dir}/{key}.pkl")

    return X_train_bal, X_test_proc, y_train_bal, y_test, feature_names, artifacts


if __name__ == "__main__":
    import sys
    from data.dataset_config import DATASET_REGISTRY

    dataset_key = sys.argv[1] if len(sys.argv) > 1 else "upi"
    config = DATASET_REGISTRY[dataset_key]
    print(f"Running generic pipeline on: {config.name}")
    X_train_bal, X_test_proc, y_train_bal, y_test, feature_names, _ = run_pipeline(config)
    print(f"Train (balanced): {X_train_bal.shape}, fraud rate: {y_train_bal.mean():.3f}")
    print(f"Test (untouched): {X_test_proc.shape}, fraud rate: {y_test.mean():.3f}")
    print(f"Feature count: {len(feature_names)}")

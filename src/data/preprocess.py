
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

RAW_PATH = "data/raw/upi_transactions.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
RANDOM_SEED = 42

ID_COLUMNS = ["transaction_id", "sender_account_name", "receiver_account_name"]
NUMERIC_COLS = ["transaction_amount_log", "hour", "transactions_last_24h"]
CATEGORICAL_COLS = ["sender_bank", "receiver_bank", "transaction_location", "device_type", "day_of_week"]
FLAG_COLS = ["is_new_beneficiary", "location_mismatch_flag", "device_change_flag"]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def basic_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop IDs, extract time features, log-transform amount, build target."""
    df = df.copy()

    # Missing values: fill any gaps before feature extraction.
    # (This dataset has 0 nulls by construction, but real UPI logs won't be this clean.)
    df["transaction_amount"] = df["transaction_amount"].fillna(df["transaction_amount"].median())

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()

    # Log-transform amount (heavily right-skewed, per EDA)
    df["transaction_amount_log"] = np.log1p(df["transaction_amount"])

    # Target
    y = (df["transaction_status"] == "Fraud").astype(int)

    # Drop columns not used as model features
    drop_cols = ID_COLUMNS + ["timestamp", "transaction_amount", "transaction_status"]
    X = df.drop(columns=drop_cols)

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer with imputation + scaling/encoding baked in per feature group."""
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

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ("flag", flag_pipeline, FLAG_COLS),
    ])
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def run_pipeline(save: bool = True):
    """Full pipeline: load -> clean -> split -> fit/transform -> SMOTE -> save."""
    df = load_raw()
    X, y = basic_clean(df)

    # Split BEFORE balancing, so the test set reflects real-world class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    # Balance ONLY the training set (test set must stay representative)
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

    if save:
        import os
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        pd.DataFrame(X_train_bal, columns=feature_names).to_csv(
            f"{PROCESSED_DIR}/X_train_balanced.csv", index=False)
        pd.Series(y_train_bal, name="is_fraud").to_csv(
            f"{PROCESSED_DIR}/y_train_balanced.csv", index=False)
        pd.DataFrame(X_test_proc, columns=feature_names).to_csv(
            f"{PROCESSED_DIR}/X_test.csv", index=False)
        pd.Series(y_test.values, name="is_fraud").to_csv(
            f"{PROCESSED_DIR}/y_test.csv", index=False)

        joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.pkl")

    return X_train_bal, X_test_proc, y_train_bal, y_test, feature_names


if __name__ == "__main__":
    X_train_bal, X_test_proc, y_train_bal, y_test, feature_names = run_pipeline()
    print(f"Train (balanced): {X_train_bal.shape}, fraud rate: {y_train_bal.mean():.3f}")
    print(f"Test (untouched): {X_test_proc.shape}, fraud rate: {y_test.mean():.3f}")
    print(f"Feature count: {len(feature_names)}")

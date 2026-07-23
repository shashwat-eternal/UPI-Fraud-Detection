

import sys
sys.path.append("src")

import time
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from models.evaluate import evaluate_model, print_summary

RANDOM_SEED = 42
DATA_DIR = "data/processed"
MODELS_DIR = "models"
KNN_REFERENCE_SIZE = 25_000  # size of the training reference set KNN searches against

PARAM_GRID = [
    {"n_neighbors": k, "weights": w}
    for k in [3, 5, 7, 9, 11]
    for w in ["uniform", "distance"]
]


def load_data():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train_final.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train_final.csv")["is_fraud"]
    X_test = pd.read_csv(f"{DATA_DIR}/X_test_final.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test_final.csv")["is_fraud"]
    return X_train, y_train, X_test, y_test


def train(save: bool = True):
    X_train, y_train, X_test, y_test = load_data()

    # Reference (search) set: a manageable subsample, since KNN's cost is all
    # at prediction time (distance computation against every reference point).
    X_ref = X_train.sample(n=KNN_REFERENCE_SIZE, random_state=RANDOM_SEED)
    y_ref = y_train.loc[X_ref.index]

    # Small internal validation split (carved from the reference pool's
    # complement) to pick hyperparameters WITHOUT touching the real test set.
    X_val_pool = X_train.drop(X_ref.index).sample(n=5000, random_state=RANDOM_SEED)
    y_val_pool = y_train.loc[X_val_pool.index]

    best_score, best_params = -1, None
    for params in PARAM_GRID:
        model = KNeighborsClassifier(n_jobs=-1, **params)
        model.fit(X_ref, y_ref)
        val_pred = model.predict(X_val_pool)
        score = f1_score(y_val_pool, val_pred)
        if score > best_score:
            best_score, best_params = score, params

    print("Best params (validation F1 = %.4f):" % best_score, best_params)

    best_model = KNeighborsClassifier(n_jobs=-1, **best_params)
    best_model.fit(X_ref, y_ref)

    metrics = evaluate_model(best_model, X_test, y_test, model_name="KNN", save=save)
    print_summary(metrics)
    print(f"Reference (training) set size used: {KNN_REFERENCE_SIZE:,}")

    if save:
        import os
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(best_model, f"{MODELS_DIR}/knn.pkl")

    return best_model, metrics, best_params


if __name__ == "__main__":
    train()

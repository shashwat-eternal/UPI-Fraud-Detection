

import sys
sys.path.append("src")

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from models.evaluate import evaluate_model, print_summary

RANDOM_SEED = 42
DATA_DIR = "data/processed"
MODELS_DIR = "models"

PARAM_DIST = {
    "n_estimators": [100, 150, 200],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
}


def load_data():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train_final.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train_final.csv")["is_fraud"]
    X_test = pd.read_csv(f"{DATA_DIR}/X_test_final.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test_final.csv")["is_fraud"]
    return X_train, y_train, X_test, y_test


def train(save: bool = True):
    X_train, y_train, X_test, y_test = load_data()

    # Hyperparameter search on a stratified subsample (Random Forest ensembles are
    # expensive to fit repeatedly on 220k rows x 61 features x many CV folds).
    # The winning params are then refit on the FULL balanced training set below.
    search_sample = X_train.sample(frac=0.25, random_state=RANDOM_SEED)
    y_search_sample = y_train.loc[search_sample.index]

    base_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    search = RandomizedSearchCV(
        base_model, param_distributions=PARAM_DIST, n_iter=8, cv=2,
        scoring="f1", random_state=RANDOM_SEED, n_jobs=-1, verbose=1,
    )
    search.fit(search_sample, y_search_sample)
    print("Best params (from subsample search):", search.best_params_)

    # Refit best params on the full balanced training set
    best_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **search.best_params_)
    best_model.fit(X_train, y_train)

    metrics = evaluate_model(best_model, X_test, y_test, model_name="RandomForest", save=save)
    print_summary(metrics)

    if save:
        import os
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(best_model, f"{MODELS_DIR}/random_forest.pkl")

    return best_model, metrics, search.best_params_


if __name__ == "__main__":
    train()

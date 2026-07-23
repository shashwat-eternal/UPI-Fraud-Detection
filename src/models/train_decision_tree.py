
import sys
sys.path.append("src")

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

from models.evaluate import evaluate_model, print_summary

RANDOM_SEED = 42
DATA_DIR = "data/processed"
MODELS_DIR = "models"

PARAM_DIST = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 8, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
}


def load_data():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train_final.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train_final.csv")["is_fraud"]
    X_test = pd.read_csv(f"{DATA_DIR}/X_test_final.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test_final.csv")["is_fraud"]
    return X_train, y_train, X_test, y_test


def train(save: bool = True):
    X_train, y_train, X_test, y_test = load_data()

    base_model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        base_model, param_distributions=PARAM_DIST, n_iter=20, cv=3,
        scoring="f1", random_state=RANDOM_SEED, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test, model_name="DecisionTree", save=save)
    print_summary(metrics)
    print("Best params:", search.best_params_)

    if save:
        import os
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(best_model, f"{MODELS_DIR}/decision_tree.pkl")

    return best_model, metrics, search.best_params_


if __name__ == "__main__":
    train()

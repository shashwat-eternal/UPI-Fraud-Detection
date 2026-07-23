

import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)

METRICS_DIR = "reports/metrics"
FIGURES_DIR = "reports/figures"


def evaluate_model(model, X_test, y_test, model_name: str, save: bool = True) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    if save:
        os.makedirs(METRICS_DIR, exist_ok=True)
        with open(f"{METRICS_DIR}/{model_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def print_report(y_test, y_pred, model_name: str):
    print(f"=== {model_name} — Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["No Fraud", "Fraud"]))


def plot_confusion_matrix(metrics: dict, save_path: str | None = None):
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"])
    ax.set_title(f"Confusion Matrix — {metrics['model_name']}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax


def print_summary(metrics: dict):
    print(f"Model: {metrics['model_name']}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

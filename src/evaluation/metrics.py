from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)


def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Youden's J statistic clipped to [0.30, 0.85]."""
    try:
        fpr, tpr, thresholds = roc_curve(labels, probs)
        j = tpr - fpr
        best = float(thresholds[np.argmax(j)])
        return float(np.clip(best, 0.30, 0.85))
    except Exception:
        return 0.5


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    if threshold is None:
        threshold = find_best_threshold(labels, probs)

    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        "threshold": float(threshold),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def summarize_folds(fold_results: list[dict]) -> dict[str, dict[str, float]]:
    keys = ["accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "auc",
            "fps", "training_time_sec"]
    summary = {}
    for key in keys:
        values = np.array(
            [fr["metrics"][key] for fr in fold_results if key in fr.get("metrics", {})],
            dtype=np.float64,
        )
        if len(values) == 0:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary

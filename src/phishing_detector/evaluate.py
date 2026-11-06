"""Metrics tailored to phishing detection (with explicit false-positive rate)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, make_scorer,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import cross_validate


def false_positive_rate(y_true, y_pred) -> float:
    """FPR = FP / (FP + TN): legitimate sites wrongly flagged as phishing."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) else 0.0


def recall_at_fpr(y_true, y_score, target_fpr: float = 0.01) -> float:
    """Highest recall (TPR) achievable while keeping FPR <= ``target_fpr``."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def compute_metrics(
    y_true, y_pred, y_score: Optional[np.ndarray] = None, target_fpr: float = 0.01
) -> Dict[str, float]:
    """Compute classification metrics. Positive class = phishing (1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out[f"recall_at_fpr_{target_fpr:g}"] = recall_at_fpr(y_true, y_score, target_fpr)
    return out


def cross_validate_model(pipeline, X, y, cv, groups=None, n_jobs: int = 1) -> Dict[str, float]:
    """Cross-validated mean/std for the key metrics (including FPR)."""
    scoring = {
        "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc",
        "fpr": make_scorer(false_positive_rate),
    }
    res = cross_validate(pipeline, X, y, cv=cv, groups=groups, scoring=scoring, n_jobs=n_jobs)
    out: Dict[str, float] = {}
    for name in scoring:
        vals = res[f"test_{name}"]
        out[f"cv_{name}_mean"] = float(np.mean(vals))
        out[f"cv_{name}_std"] = float(np.std(vals))
    return out


def results_table(rows: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Turn ``{model: metrics}`` into a tidy DataFrame sorted by F1."""
    df = pd.DataFrame(rows).T
    df.index.name = "model"
    return df.sort_values("f1", ascending=False)

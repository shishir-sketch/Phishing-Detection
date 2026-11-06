"""Matplotlib figures for model comparison (headless-safe)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

Scores = Dict[str, Tuple[np.ndarray, np.ndarray]]  # name -> (y_true, y_score)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(scores: Scores, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, s) in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_roc_low_fpr(scores: Scores, path: Path, max_fpr: float = 0.05) -> None:
    """Zoom on the low-FPR region, which is where deployment decisions are made."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, s) in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=name)
    ax.set(xlim=(0, max_fpr), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate (recall)", title=f"ROC (FPR <= {max_fpr:g})")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_pr(scores: Scores, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, s) in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, path)


def plot_confusion_matrices(preds: Dict[str, Tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    n = len(preds)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, (y, p)) in zip(axes.ravel(), preds.items()):
        cm = confusion_matrix(y, p, labels=[0, 1])
        ax.axis("on")
        ax.imshow(cm, cmap="Blues")
        ax.set(title=name, xticks=[0, 1], yticks=[0, 1],
               xticklabels=["legit", "phish"], yticklabels=["legit", "phish"],
               xlabel="Predicted", ylabel="Actual")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
    _save(fig, path)


def plot_metric_comparison(table, path: Path, metrics: Sequence[str] = ("precision", "recall", "f1", "fpr")) -> None:
    metrics = [m for m in metrics if m in table.columns]
    x = np.arange(len(table))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(table)), 4.5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, table[m].values, width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(table.index, rotation=25, ha="right")
    ax.set(ylabel="Score", title="Model comparison (test set; lower FPR is better)")
    ax.legend()
    _save(fig, path)


def plot_feature_importance(estimator, feature_names: Sequence[str], path: Path, top: int = 20) -> bool:
    """Plot importances if the final estimator exposes them; returns True if drawn."""
    final = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if hasattr(final, "feature_importances_"):
        imp = np.asarray(final.feature_importances_)
    elif hasattr(final, "coef_"):
        imp = np.abs(np.ravel(final.coef_))
    else:
        return False
    idx = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(idx) + 1.5))
    ax.barh([feature_names[i] for i in idx], imp[idx])
    ax.set(title="Top feature importances", xlabel="Importance")
    _save(fig, path)
    return True

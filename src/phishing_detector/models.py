"""Model zoo: scikit-learn pipelines compared in this project."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:  # optional dependency
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - depends on environment
    HAS_XGBOOST = False

DEFAULT_MODELS = [
    "logistic_regression",
    "linear_svm",
    "naive_bayes",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
] + (["xgboost"] if HAS_XGBOOST else [])
# 'svm_rbf' is opt-in: O(n^2) training time makes it slow on large datasets.
ALL_MODELS = DEFAULT_MODELS + ["svm_rbf"]


def _build(name: str, seed: int):
    if name == "logistic_regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ])
    if name == "linear_svm":
        base = LinearSVC(dual=False, class_weight="balanced", max_iter=5000, random_state=seed)
        return Pipeline([("scale", StandardScaler()), ("clf", CalibratedClassifierCV(base, cv=3))])
    if name == "svm_rbf":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        ])
    if name == "naive_bayes":
        return Pipeline([("clf", GaussianNB())])
    if name == "decision_tree":
        return Pipeline([("clf", DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=3, class_weight="balanced", random_state=seed))])
    if name == "random_forest":
        return Pipeline([("clf", RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced_subsample", random_state=seed))])
    if name == "gradient_boosting":
        return Pipeline([("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=seed))])
    if name == "xgboost" and HAS_XGBOOST:
        return Pipeline([("clf", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1, random_state=seed))])
    raise ValueError(f"Unknown or unavailable model '{name}'. Options: {ALL_MODELS}")


def get_models(names: Optional[Iterable[str]] = None, seed: int = 42) -> Dict[str, Pipeline]:
    """Return ``{name: unfitted Pipeline}`` for the requested model names."""
    names = list(names) if names else DEFAULT_MODELS
    return {n: _build(n, seed) for n in names}

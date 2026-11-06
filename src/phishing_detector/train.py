"""Train and compare phishing-detection models.

Examples
--------
    # Zero-setup demo on synthetic URLs
    python -m phishing_detector.train --synthetic

    # URL dataset (CSV with url,label columns)
    python -m phishing_detector.train --data data/urls.csv --mode url \
        --url-col url --label-col label --phishing-label phishing

    # Pre-featurised dataset (e.g. UCI Phishing Websites, -1 = phishing)
    python -m phishing_detector.train --data data/uci.csv --mode tabular \
        --label-col Result --phishing-label -1
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, train_test_split

from . import __version__
from .data import generate_synthetic_urls, load_tabular_dataset, load_url_dataset
from .evaluate import compute_metrics, cross_validate_model, results_table
from .features import extract_features_df, get_hostname
from .models import ALL_MODELS, get_models
from . import plots

LOG = logging.getLogger("phishing_detector")
LOWER_IS_BETTER = {"fpr", "fnr"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train & compare phishing detection models.")
    src = p.add_argument_group("data")
    src.add_argument("--data", type=Path, help="Path to CSV dataset.")
    src.add_argument("--synthetic", action="store_true", help="Use generated demo data (no file needed).")
    src.add_argument("--n-synthetic", type=int, default=6000)
    src.add_argument("--mode", choices=["url", "tabular"], default="url")
    src.add_argument("--url-col", default="url")
    src.add_argument("--label-col", default="label")
    src.add_argument("--phishing-label", default=None,
                     help="Raw label value meaning phishing (e.g. 'phishing', 1, -1, 0).")
    src.add_argument("--max-rows", type=int, default=None, help="Subsample large datasets.")
    exp = p.add_argument_group("experiment")
    exp.add_argument("--models", nargs="+", default=None, choices=ALL_MODELS,
                     help="Models to train (default: all fast models).")
    exp.add_argument("--test-size", type=float, default=0.2)
    exp.add_argument("--cv-folds", type=int, default=5, help="0 disables cross-validation.")
    exp.add_argument("--seed", type=int, default=42)
    exp.add_argument("--no-group-split", action="store_true",
                     help="Disable hostname-grouped splitting (url mode).")
    exp.add_argument("--select-by", default="f1",
                     choices=["f1", "precision", "recall", "roc_auc", "pr_auc", "fpr", "recall_at_fpr"])
    exp.add_argument("--target-fpr", type=float, default=0.01,
                     help="FPR budget used for the recall@FPR metric.")
    out = p.add_argument_group("output")
    out.add_argument("--output-dir", type=Path, default=Path("results"))
    out.add_argument("--model-dir", type=Path, default=Path("models"))
    out.add_argument("--no-plots", action="store_true")
    return p


def _load(args):
    """Return X (DataFrame), y (ndarray), groups (ndarray|None), mode."""
    if args.synthetic:
        df = generate_synthetic_urls(n=args.n_synthetic, seed=args.seed)
        mode = "url"
    elif args.data:
        mode = args.mode
        if mode == "url":
            df = load_url_dataset(args.data, args.url_col, args.label_col,
                                  args.phishing_label, args.max_rows, args.seed)
        else:
            X, y = load_tabular_dataset(args.data, args.label_col, args.phishing_label)
            return X, y.to_numpy(), None, mode
    else:
        raise SystemExit("Provide --data PATH or --synthetic.")

    LOG.info("Extracting features from %d URLs ...", len(df))
    X = extract_features_df(df["url"])
    groups = None if args.no_group_split else df["url"].map(get_hostname).to_numpy()
    return X, df["label"].to_numpy(), groups, mode


def _split(X, y, groups, test_size, seed):
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(X, y, groups))
        return tr, te
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    return tr, te


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def run(argv: Optional[List[str]] = None) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    X, y, groups, mode = _load(args)
    LOG.info("Dataset: %d samples, %d features, %.1f%% phishing",
             len(y), X.shape[1], 100 * y.mean())

    tr, te = _split(X, y, groups, args.test_size, args.seed)
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]
    g_tr = groups[tr] if groups is not None else None
    LOG.info("Train=%d  Test=%d  (hostname-grouped split: %s)", len(tr), len(te), groups is not None)

    if args.cv_folds and args.cv_folds > 1:
        cv = (StratifiedGroupKFold(args.cv_folds, shuffle=True, random_state=args.seed)
              if g_tr is not None else
              StratifiedKFold(args.cv_folds, shuffle=True, random_state=args.seed))
    else:
        cv = None

    fitted, test_rows, cv_rows = {}, {}, {}
    curves, preds = {}, {}
    for name, pipe in get_models(args.models, args.seed).items():
        LOG.info("Training %s ...", name)
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        y_score = _scores(pipe, X_te)
        y_pred = pipe.predict(X_te)
        metrics = compute_metrics(y_te, y_pred, y_score, args.target_fpr)
        metrics["fit_seconds"] = round(fit_s, 3)
        test_rows[name] = metrics
        fitted[name], curves[name], preds[name] = pipe, (y_te, y_score), (y_te, y_pred)
        if cv is not None:
            cv_rows[name] = cross_validate_model(pipe, X_tr, y_tr, cv, groups=g_tr)
        LOG.info("  %s | P=%.4f R=%.4f F1=%.4f FPR=%.4f", name, metrics["precision"],
                 metrics["recall"], metrics["f1"], metrics["fpr"])

    table = results_table(test_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "metrics.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(test_rows, indent=2))
    if cv_rows:
        pd.DataFrame(cv_rows).T.to_csv(args.output_dir / "cv_metrics.csv", index_label="model")

    key = args.select_by if args.select_by != "recall_at_fpr" else f"recall_at_fpr_{args.target_fpr:g}"
    ascending = args.select_by in LOWER_IS_BETTER
    best = table[key].astype(float).sort_values(ascending=ascending).index[0]
    LOG.info("Best model by %s: %s", key, best)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": fitted[best], "model_name": best, "mode": mode,
        "feature_names": list(X.columns), "threshold": 0.5,
        "metrics": test_rows[best], "version": __version__,
    }
    joblib.dump(bundle, args.model_dir / "best_model.joblib")

    if not args.no_plots:
        fig_dir = args.output_dir / "figures"
        plots.plot_roc(curves, fig_dir / "roc_curves.png")
        plots.plot_roc_low_fpr(curves, fig_dir / "roc_low_fpr.png")
        plots.plot_pr(curves, fig_dir / "pr_curves.png")
        plots.plot_confusion_matrices(preds, fig_dir / "confusion_matrices.png")
        plots.plot_metric_comparison(table, fig_dir / "metric_comparison.png")
        for cand in ["random_forest", best, *fitted]:
            if cand in fitted and plots.plot_feature_importance(
                    fitted[cand], list(X.columns), fig_dir / "feature_importance.png"):
                break

    show = [c for c in ["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", key, "fit_seconds"]
            if c in table.columns]
    print("\n=== Test-set results (positive class = phishing) ===")
    print(table[list(dict.fromkeys(show))].round(4).to_string())
    print(f"\nBest model by {key}: {best}  ->  {args.model_dir / 'best_model.joblib'}")
    return table


def main() -> None:
    run()


if __name__ == "__main__":
    main()

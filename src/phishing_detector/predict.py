"""Score URLs with a trained model bundle.

    python -m phishing_detector.predict --url "http://paypal.secure-login.xyz/verify"
    python -m phishing_detector.predict --file urls.txt --threshold 0.7
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import joblib
import pandas as pd

from .features import extract_features_df


def load_bundle(path):
    return joblib.load(path)


def predict_urls(bundle: dict, urls: Sequence[str], threshold: Optional[float] = None) -> pd.DataFrame:
    if bundle.get("mode") != "url":
        raise ValueError("This model was trained on pre-computed tabular features; "
                         "URL prediction needs a model trained with --mode url.")
    thr = bundle.get("threshold", 0.5) if threshold is None else threshold
    X = extract_features_df(urls)[bundle["feature_names"]]
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    return pd.DataFrame({
        "url": list(urls),
        "phishing_probability": proba.round(4),
        "prediction": ["phishing" if p >= thr else "legitimate" for p in proba],
    })


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Classify URLs as phishing or legitimate.")
    ap.add_argument("--model", type=Path, default=Path("models/best_model.joblib"))
    ap.add_argument("--url", action="append", default=[], help="URL to score (repeatable).")
    ap.add_argument("--file", type=Path, help="Text file with one URL per line.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Decision threshold on phishing probability (default 0.5). "
                         "Raise it to reduce false positives.")
    args = ap.parse_args(argv)

    urls = list(args.url)
    if args.file:
        urls += [u.strip() for u in args.file.read_text().splitlines() if u.strip()]
    if not urls:
        ap.error("Provide --url and/or --file.")
    out = predict_urls(load_bundle(args.model), urls, args.threshold)
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()

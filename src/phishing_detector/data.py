"""Dataset loading, label normalisation and a synthetic data generator.

Two dataset styles are supported:

* ``url``     - a CSV with a URL column and a label column; features are
                extracted by :mod:`phishing_detector.features`.
* ``tabular`` - a CSV of pre-computed numeric features plus a label column
                (e.g. the UCI "Phishing Websites" dataset).

Internally the label convention is always ``1 = phishing`` and ``0 = legitimate``.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PHISHING_ALIASES = {"phishing", "phish", "bad", "malicious", "1", "true", "yes"}
LEGIT_ALIASES = {"legitimate", "legit", "good", "benign", "safe", "0", "false", "no"}


def normalize_labels(series: pd.Series, phishing_label: Optional[Union[str, int, float]] = None) -> pd.Series:
    """Map a raw label column to ints where 1 = phishing and 0 = legitimate.

    Numeric labels are ambiguous across datasets (UCI uses -1 for phishing,
    PhiUSIIL uses 0), so pass ``phishing_label`` explicitly when in doubt.
    """
    if phishing_label is not None:
        numeric = pd.to_numeric(series, errors="coerce")
        try:
            target = float(phishing_label)
            mask = numeric == target
        except (TypeError, ValueError):
            mask = series.astype(str).str.strip().str.lower() == str(phishing_label).strip().lower()
        return mask.astype(int)

    if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        cleaned = series.astype(str).str.strip().str.lower()
        unknown = set(cleaned.unique()) - PHISHING_ALIASES - LEGIT_ALIASES
        if unknown:
            raise ValueError(
                f"Unrecognised label values {sorted(unknown)}; pass phishing_label explicitly."
            )
        return cleaned.isin(PHISHING_ALIASES).astype(int)

    values = set(pd.unique(series.dropna()))
    if not values <= {0, 1}:
        raise ValueError(
            f"Numeric labels {sorted(values)} are ambiguous; pass phishing_label "
            "(e.g. -1 for UCI Phishing Websites, 0 for PhiUSIIL)."
        )
    return series.astype(int)


def load_url_dataset(
    path: Union[str, Path],
    url_col: str = "url",
    label_col: str = "label",
    phishing_label: Optional[Union[str, int, float]] = None,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load a URL dataset and return a de-duplicated frame with ``url`` and ``label``."""
    df = pd.read_csv(path)
    for col in (url_col, label_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)}")
    df = df[[url_col, label_col]].dropna().rename(columns={url_col: "url", label_col: "label"})
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = normalize_labels(df["label"], phishing_label)
    # Duplicate URLs inflate test scores through train/test leakage.
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return df


def load_tabular_dataset(
    path: Union[str, Path],
    label_col: str = "Result",
    phishing_label: Optional[Union[str, int, float]] = None,
    drop_cols: Sequence[str] = (),
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a pre-featurised dataset; returns ``(X, y)`` with y in {0, 1}."""
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise KeyError(f"Column '{label_col}' not found. Available: {list(df.columns)}")
    y = normalize_labels(df[label_col], phishing_label)
    X = df.drop(columns=[label_col, *[c for c in drop_cols if c in df.columns]])
    X = X.select_dtypes(include=[np.number]).copy()
    return X, y


# --------------------------------------------------------------------------- #
# Synthetic data (for demos, CI and tests only - NOT a substitute for real data)
# --------------------------------------------------------------------------- #
_WORDS = [
    "garden", "photo", "travel", "bank", "market", "school", "news", "cloud", "music",
    "health", "sports", "code", "books", "store", "city", "design", "food", "game",
    "tech", "media", "energy", "studio", "labs", "works", "home", "pixel", "river",
]
_LEGIT_TLDS = ["com", "org", "net", "edu", "gov", "io", "co.uk", "de"]
_PHISH_TLDS = ["xyz", "top", "tk", "ml", "click", "icu", "work", "com", "net", "info"]
_BRANDS = ["paypal", "apple", "amazon", "microsoft", "google", "netflix", "ebay"]
_PHISH_PATHS = ["login", "verify", "account", "secure", "update", "signin", "confirm", "billing"]


def _rand_token(rng: random.Random, n: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(alphabet) for _ in range(n))


def _legit_url(rng: random.Random) -> str:
    host = rng.choice(_WORDS) + (rng.choice(["", "-" + rng.choice(_WORDS)]) if rng.random() < 0.15 else "")
    host = f"{'www.' if rng.random() < 0.6 else ''}{host}.{rng.choice(_LEGIT_TLDS)}"
    scheme = "https" if rng.random() < 0.85 else "http"
    segs = [rng.choice(_WORDS) for _ in range(rng.choice([0, 1, 1, 2, 3]))]
    path = "/" + "/".join(segs) + (".html" if segs and rng.random() < 0.2 else "")
    query = f"?id={rng.randint(1, 9999)}&page={rng.randint(1, 20)}" if rng.random() < 0.2 else ""
    return f"{scheme}://{host}{path if segs else ''}{query}"


def _phish_url(rng: random.Random) -> str:
    style = rng.choice(["ip", "brand_sub", "long_hyphen", "at", "keyword_path", "random"])
    scheme = "https" if rng.random() < 0.45 else "http"
    brand, kw = rng.choice(_BRANDS), rng.choice(_PHISH_PATHS)
    tld = rng.choice(_PHISH_TLDS)
    path = f"/{rng.choice(['wp-includes', 'cgi-bin', 'images'])}/{kw}/{_rand_token(rng, rng.randint(6, 14))}.php"
    query = f"?cmd=_{kw}&id={_rand_token(rng, 16)}" if rng.random() < 0.6 else ""
    if style == "ip":
        host = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        return f"http://{host}{path}{query}"
    if style == "brand_sub":
        host = f"{brand}.{kw}.{_rand_token(rng, rng.randint(5, 10))}.{tld}"
    elif style == "long_hyphen":
        host = f"{brand}-{kw}-{rng.choice(_WORDS)}-{rng.randint(10, 99)}.{tld}"
    elif style == "at":
        return f"{scheme}://{brand}.com@{_rand_token(rng, 8)}.{tld}{path}"
    elif style == "keyword_path":
        host = f"{rng.choice(_WORDS)}{rng.randint(1, 99)}.{tld}"
    else:
        host = f"{_rand_token(rng, rng.randint(10, 18))}.{tld}"
    return f"{scheme}://{host}{path}{query}"


def generate_synthetic_urls(
    n: int = 6000, phishing_ratio: float = 0.4, noise: float = 0.04, seed: int = 42
) -> pd.DataFrame:
    """Generate a labelled URL dataset with overlapping classes.

    ``noise`` is the fraction of samples whose URL looks like the *other* class,
    so models cannot reach 100 % and the metric trade-offs stay visible.
    This is purely for demos/tests; use a real dataset for any real claims.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        is_phish = rng.random() < phishing_ratio
        looks_phish = is_phish if rng.random() >= noise else not is_phish
        rows.append((_phish_url(rng) if looks_phish else _legit_url(rng), int(is_phish)))
    df = pd.DataFrame(rows, columns=["url", "label"]).drop_duplicates("url").reset_index(drop=True)
    return df

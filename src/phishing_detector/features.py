"""Lexical and host-based URL feature extraction.

Everything here is computed from the URL string alone (no DNS, WHOIS or HTTP
requests), so extraction is fast, deterministic, safe to run on malicious URLs
and reproducible offline.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List
from urllib.parse import urlparse

import pandas as pd

SUSPICIOUS_KEYWORDS = (
    "login", "signin", "sign-in", "verify", "verification", "update", "secure",
    "account", "banking", "confirm", "password", "wallet", "webscr", "support",
    "billing", "suspend", "unlock", "bonus", "invoice", "recover", "validate",
)
BRANDS = (
    "paypal", "apple", "amazon", "google", "microsoft", "facebook", "netflix",
    "ebay", "instagram", "dropbox", "linkedin", "whatsapp", "outlook", "office365",
)
SHORTENERS = (
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
)
SUSPICIOUS_TLDS = {
    "zip", "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work",
    "support", "country", "loan", "men", "icu", "buzz", "rest", "cam", "monster",
}
SUSPICIOUS_EXTENSIONS = (".exe", ".apk", ".scr", ".zip", ".rar", ".js", ".jar", ".bat")
# Second-level labels that form part of a public suffix (co.uk, com.au, ...).
_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "edu", "ac"}

_IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_HEX_IP = re.compile(r"^0x[0-9a-f]+$", re.IGNORECASE)
_PERCENT_ENC = re.compile(r"%[0-9a-fA-F]{2}")
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _entropy(text: str) -> float:
    """Shannon entropy (bits/char). Random-looking strings score higher."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _parse(url: str):
    """Parse a URL, tolerating a missing scheme and malformed input."""
    url = (url or "").strip()
    has_scheme = "://" in url
    candidate = url if has_scheme else "http://" + url.lstrip("/")
    try:
        parsed = urlparse(candidate)
        parsed.hostname  # noqa: B018 - may raise on malformed IPv6 brackets
    except ValueError:
        parsed = urlparse("http://invalid.invalid/")
    return url, parsed


def get_hostname(url: str) -> str:
    """Return the lower-cased hostname of ``url`` (empty string if unparsable)."""
    _, parsed = _parse(url)
    return (parsed.hostname or "").lower()


def _split_host(host: str):
    """Return (subdomain_labels, registered_label, tld) using a light heuristic."""
    parts = [p for p in host.split(".") if p]
    if len(parts) < 2:
        return [], host, ""
    tld = parts[-1]
    k = 2
    if len(tld) == 2 and len(parts) >= 3 and parts[-2] in _SECOND_LEVEL:
        k = 3
    registered = parts[-k] if len(parts) >= k else parts[0]
    return parts[: max(len(parts) - k, 0)], registered, tld


def extract_url_features(url: str) -> Dict[str, float]:
    """Extract numeric features from a single URL."""
    raw, parsed = _parse(url)
    lower = raw.lower()
    host = (parsed.hostname or "").lower()
    path, query = parsed.path or "", parsed.query or ""
    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 1

    is_ip = int(bool(_IPV4.match(host)) or bool(_HEX_IP.match(host)))
    subdomains, registered, tld = ([], host, "") if is_ip else _split_host(host)

    n = max(len(raw), 1)
    letters = sum(c.isalpha() for c in raw)
    digits = sum(c.isdigit() for c in raw)
    host_digits = sum(c.isdigit() for c in host)
    tokens = [t for t in _TOKEN_SPLIT.split(lower) if t]
    path_segments = [s for s in path.split("/") if s]

    sub_and_path = ".".join(subdomains) + "/" + path.lower()
    brand_misuse = int(any(b in sub_and_path and b not in registered for b in BRANDS))

    return {
        # --- length features
        "url_length": len(raw),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "domain_length": len(registered),
        # --- character counts
        "count_dots": raw.count("."),
        "count_hyphens": raw.count("-"),
        "count_underscores": raw.count("_"),
        "count_slashes": raw.count("/"),
        "count_at": raw.count("@"),
        "count_question": raw.count("?"),
        "count_equals": raw.count("="),
        "count_ampersand": raw.count("&"),
        "count_percent": raw.count("%"),
        "count_digits": digits,
        "count_special": sum(not c.isalnum() for c in raw),
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "host_digit_ratio": host_digits / max(len(host), 1),
        # --- host / structure
        "has_ip_host": is_ip,
        "uses_https": int(lower.startswith("https://")),
        "has_port": has_port,
        "num_subdomains": len(subdomains),
        "tld_length": len(tld),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "hyphen_in_domain": int("-" in registered),
        "is_shortener": int(any(host == s or host.endswith("." + s) for s in SHORTENERS)),
        "has_punycode": int("xn--" in host),
        "path_depth": len(path_segments),
        "num_params": len([p for p in query.split("&") if p]),
        "double_slash_in_path": int("//" in path),
        "http_in_path_or_query": int("http" in (path + query).lower()),
        "suspicious_extension": int(path.lower().endswith(SUSPICIOUS_EXTENSIONS)),
        # --- semantics
        "suspicious_keyword_count": sum(lower.count(k) for k in SUSPICIOUS_KEYWORDS),
        "brand_in_subdomain_or_path": brand_misuse,
        "hex_encoded_count": len(_PERCENT_ENC.findall(raw)),
        # --- token / randomness
        "longest_token_length": max((len(t) for t in tokens), default=0),
        "avg_token_length": (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0,
        "url_entropy": _entropy(lower),
        "host_entropy": _entropy(host),
    }


FEATURE_NAMES: List[str] = list(extract_url_features("http://example.com").keys())


def extract_features_df(urls: Iterable[str]) -> pd.DataFrame:
    """Vectorise many URLs into a DataFrame with a stable column order."""
    rows = [extract_url_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES).astype(float)

# Data

Datasets are **not** committed to this repo. Download one and place it here.

## Supported formats

| Mode | File layout | Example command |
|------|-------------|-----------------|
| `url` | CSV with a URL column and a label column | `--data data/urls.csv --mode url --url-col url --label-col label --phishing-label phishing` |
| `tabular` | CSV of pre-computed numeric features + label | `--data data/uci.csv --mode tabular --label-col Result --phishing-label -1` |

> **Label conventions differ between datasets.** UCI "Phishing Websites" uses `-1` for phishing,
> PhiUSIIL uses `0` for phishing, many Kaggle URL sets use the strings `bad`/`good`.
> Always pass `--phishing-label` for numeric labels. Internally `1 = phishing`.

## Suggested sources (check each licence/terms before use)

- **UCI Machine Learning Repository** - "Phishing Websites" (pre-computed features) and
  "PhiUSIIL Phishing URL" (URLs + features).
- **Kaggle** - search "phishing site URLs" for URL/label CSVs.
- **PhishTank / OpenPhish** - live feeds of verified phishing URLs (positives).
- **Tranco list** - ranked popular domains, a reasonable source of legitimate (negative) URLs.

When combining feeds, build negatives and positives from the same time window, de-duplicate URLs,
and avoid features that leak the source (e.g. every phishing URL being `http://`, every benign URL
coming from a top-site list). The loader de-duplicates exact URLs and, in `url` mode, splits by
hostname so that the same domain never appears in both train and test.

## Quick start without any download

`python -m phishing_detector.train --synthetic` generates a labelled demo dataset. It exists for
smoke tests and CI only; **do not report its scores as real results.**

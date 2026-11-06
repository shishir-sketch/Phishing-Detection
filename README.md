# 🛡️ ML-Based Phishing Website Detection

Classify **phishing vs. legitimate websites** from URL patterns, and compare classic ML models on
**precision, recall, F1-score and false-positive rate (FPR)** - the metrics that actually matter
when a security filter sits in front of real users.

![python](https://img.shields.io/badge/python-3.9%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![status](https://img.shields.io/badge/status-active-brightgreen)

## Why this project

A phishing detector that blocks 99% of attacks but also flags 5% of legitimate sites is unusable:
users lose trust and start ignoring warnings. So this repo goes beyond accuracy and reports:

| Metric | Meaning in this context |
|---|---|
| **Precision** | Of the sites flagged as phishing, how many really were |
| **Recall (TPR)** | Of all real phishing sites, how many were caught |
| **F1-score** | Harmonic mean of precision and recall |
| **FPR** | Share of *legitimate* sites wrongly flagged (lower is better) |
| **Recall @ FPR ≤ 1%** | How many attacks you catch under a strict false-alarm budget |
| ROC-AUC / PR-AUC | Threshold-independent ranking quality |

## Features

- **40 URL features, no network calls** - length, character counts, IP-as-host, subdomain depth,
  suspicious TLDs, brand names in subdomain/path, phishing keywords, URL shorteners, punycode,
  percent-encoding, Shannon entropy, and more (`src/phishing_detector/features.py`).
- **Model zoo** - Logistic Regression, Linear SVM, Naive Bayes, Decision Tree, Random Forest,
  Histogram Gradient Boosting, optional RBF-SVM and XGBoost.
- **Leakage-aware evaluation** - exact-duplicate URL removal, **hostname-grouped train/test split**
  and grouped cross-validation, so a domain never appears on both sides.
- **Two input modes** - raw URLs (`url`) or pre-computed feature tables like UCI Phishing Websites (`tabular`).
- **Deployment-oriented outputs** - metrics tables, cross-validation, ROC/PR curves (incl. a low-FPR zoom),
  confusion matrices, feature importances, and a saved best-model bundle with a CLI for scoring URLs.
- Tests, CI (GitHub Actions) and a zero-download demo mode.

## Repository structure

```
phishing-detection/
├── src/phishing_detector/
│   ├── features.py     # URL -> numeric feature vector
│   ├── data.py         # loaders, label normalisation, synthetic demo data
│   ├── models.py       # model zoo (sklearn pipelines)
│   ├── evaluate.py     # precision/recall/F1/FPR, recall@FPR, cross-validation
│   ├── plots.py        # ROC, PR, confusion matrices, importances
│   ├── train.py        # CLI: train + compare + save best model
│   └── predict.py      # CLI: score new URLs
├── tests/              # pytest suite
├── data/               # put datasets here (git-ignored) + sources guide
├── results/            # metrics + figures (generated, git-ignored)
├── models/             # saved model bundle (generated, git-ignored)
├── .github/workflows/ci.yml
├── pyproject.toml  requirements.txt  Makefile  LICENSE
```

## Installation

```bash
git clone https://github.com/<your-username>/phishing-detection.git
cd phishing-detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # add ",xgboost" for XGBoost support
```

## Quick start (no dataset needed)

```bash
python -m phishing_detector.train --synthetic
```

This runs the whole pipeline on generated demo URLs and writes `results/` and `models/best_model.joblib`.
**Synthetic scores are for smoke-testing only and say nothing about real-world performance.**

## Train on a real dataset

See [`data/README.md`](data/README.md) for sources and label conventions.

```bash
# URL dataset (CSV with url,label columns)
python -m phishing_detector.train \
  --data data/urls.csv --mode url \
  --url-col url --label-col label --phishing-label phishing

# Pre-computed features (UCI Phishing Websites uses -1 for phishing)
python -m phishing_detector.train \
  --data data/uci_phishing.csv --mode tabular \
  --label-col Result --phishing-label -1
```

Useful options:

| Flag | Purpose |
|---|---|
| `--models random_forest xgboost ...` | choose which models to compare |
| `--select-by fpr` / `recall_at_fpr` | pick the "best" model by a different criterion |
| `--target-fpr 0.005` | FPR budget for the recall@FPR metric |
| `--cv-folds 5` | cross-validation folds (`0` to disable) |
| `--no-group-split` | use a plain stratified split instead of hostname-grouped |
| `--max-rows 200000` | subsample very large datasets |

Console aliases are also installed: `phish-train` and `phish-predict`.

## Score new URLs

```bash
python -m phishing_detector.predict \
  --url "http://paypal.secure-login.example-xyz.top/verify/account.php" \
  --url "https://www.wikipedia.org/"

python -m phishing_detector.predict --file urls.txt --threshold 0.8   # fewer false positives
```

Raising `--threshold` trades recall for a lower FPR - choose it from the low-FPR ROC plot rather than
defaulting to 0.5.

## Outputs

```
results/metrics.csv            test-set metrics per model
results/cv_metrics.csv         cross-validated mean/std per model
results/metrics.json
results/figures/               roc_curves, roc_low_fpr, pr_curves, confusion_matrices,
                               metric_comparison, feature_importance
models/best_model.joblib       pipeline + feature list + metadata
```

## Results

Add your table here after running on a real dataset, for example:

| Model | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | … | … | … | … | … |
| Gradient Boosting | … | … | … | … | … |
| Logistic Regression | … | … | … | … | … |

Then add `results/figures/metric_comparison.png` and `roc_low_fpr.png` and discuss the
precision/recall/FPR trade-off between models.

## Methodology notes

1. **Features** are purely lexical/host-based so scoring works offline and never touches a malicious site.
2. **Splits** are grouped by hostname in `url` mode to prevent near-duplicate leakage.
3. **Class imbalance** is handled with class weights; judge models by FPR and PR-AUC, not accuracy.
4. **Model selection** defaults to F1 but can optimise FPR or recall under an FPR budget.

## Limitations & ethics

- URL-only features miss attacks on compromised legitimate domains and can be evaded by attackers who
  mimic benign URL structure. Content, DNS/WHOIS, certificate and page-behaviour features would help.
- Public datasets can carry sampling bias (e.g. phishing from one feed, benign from top-site lists), which
  inflates scores. Validate on fresh, time-separated data before trusting any number.
- Intended for defensive research and education. Do not treat output as a sole blocking decision.

## Roadmap

- [ ] Content-based features (HTML/forms, favicon, redirects) and TLS/WHOIS signals
- [ ] Character-level CNN / transformer URL baselines
- [ ] Time-based train/test split and drift evaluation
- [ ] Probability calibration and cost-sensitive threshold selection
- [ ] REST API / browser-extension demo

## Contributing

Issues and PRs are welcome. Run `make test lint` before submitting.

## License

MIT - see [LICENSE](LICENSE). Replace `<YOUR NAME>` in the licence file.

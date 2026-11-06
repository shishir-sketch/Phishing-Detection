from phishing_detector.predict import load_bundle, predict_urls
from phishing_detector.train import run


def test_end_to_end_synthetic(tmp_path):
    table = run([
        "--synthetic", "--n-synthetic", "1200", "--models", "logistic_regression", "decision_tree",
        "--cv-folds", "2", "--output-dir", str(tmp_path / "res"), "--model-dir", str(tmp_path / "mod"),
        "--no-plots",
    ])
    assert {"precision", "recall", "f1", "fpr"} <= set(table.columns)
    assert table["f1"].min() > 0.7
    bundle = load_bundle(tmp_path / "mod" / "best_model.joblib")
    out = predict_urls(bundle, ["http://paypal.secure-login.evil.xyz/verify/account.php",
                                "https://www.garden.com/news"])
    assert out["phishing_probability"].iloc[0] > out["phishing_probability"].iloc[1]
    assert (tmp_path / "res" / "metrics.csv").exists()

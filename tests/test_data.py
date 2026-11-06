import pandas as pd
import pytest

from phishing_detector.data import generate_synthetic_urls, load_url_dataset, normalize_labels


def test_synthetic_has_both_classes_and_is_deterministic():
    a = generate_synthetic_urls(500, seed=1)
    b = generate_synthetic_urls(500, seed=1)
    assert set(a["label"]) == {0, 1}
    assert a.equals(b)


def test_normalize_string_labels():
    s = pd.Series(["phishing", "Legitimate", "bad", "good"])
    assert normalize_labels(s).tolist() == [1, 0, 1, 0]


def test_numeric_labels_need_explicit_phishing_value():
    s = pd.Series([-1, 1, 1, -1])
    with pytest.raises(ValueError):
        normalize_labels(s)
    assert normalize_labels(s, phishing_label=-1).tolist() == [1, 0, 0, 1]


def test_load_url_dataset_dedupes(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"url": ["http://a.com", "http://a.com", "http://b.com"],
                  "label": ["good", "good", "bad"]}).to_csv(p, index=False)
    df = load_url_dataset(p)
    assert len(df) == 2 and df["label"].sum() == 1

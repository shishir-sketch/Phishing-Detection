import numpy as np
import pytest

from phishing_detector.evaluate import compute_metrics, false_positive_rate, recall_at_fpr


def test_known_confusion_matrix():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])  # tp=3 fn=1 fp=1 tn=5
    m = compute_metrics(y_true, y_pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(1 / 6)
    assert m["specificity"] == pytest.approx(5 / 6)
    assert false_positive_rate(y_true, y_pred) == pytest.approx(1 / 6)


def test_recall_at_fpr_perfect_ranking():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    assert recall_at_fpr(y, s, 0.0) == pytest.approx(1.0)


def test_no_negatives_predicted_positive():
    y = np.array([0, 0, 1, 1])
    assert compute_metrics(y, np.array([0, 0, 1, 1]))["fpr"] == 0.0

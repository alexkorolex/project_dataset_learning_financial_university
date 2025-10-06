import numpy as np

from bank.metrics import compute_metrics, find_best_threshold


def test_compute_and_tune():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert 0.0 <= m["roc_auc"] <= 1.0
    t = find_best_threshold(y_true, y_prob, metric="f1")
    assert 0.0 <= t <= 1.0

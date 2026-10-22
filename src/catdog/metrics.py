"""Metrics that go beyond accuracy: bootstrap confidence intervals, calibration, and paired model tests."""
import numpy as np
from scipy.stats import binomtest
from sklearn import metrics as skm


def point_metrics(y, p, threshold=0.5):
    """Classification + probability-quality metrics. y: 0=Cat, 1=Dog; p: predicted P(Dog)."""
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": skm.accuracy_score(y, pred),
        "precision_dog": skm.precision_score(y, pred, zero_division=0),
        "recall_dog": skm.recall_score(y, pred, zero_division=0),
        "recall_cat": skm.recall_score(1 - y, 1 - pred, zero_division=0),
        "f1": skm.f1_score(y, pred, zero_division=0),
        "roc_auc": skm.roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "log_loss": skm.log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "brier": skm.brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
    }


def bootstrap_ci(y, p, metric_fn, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for any metric_fn(y, p) -> float."""
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric_fn(y[idx], p[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def expected_calibration_error(y, p, n_bins=10):
    """Average |accuracy - confidence| over equal-width confidence bins, weighted by bin size."""
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def best_threshold(y, p):
    """Threshold that maximizes accuracy. Pick it on validation, then apply it to test."""
    candidates = np.linspace(0.05, 0.95, 91)
    accs = [skm.accuracy_score(y, (p >= t).astype(int)) for t in candidates]
    return float(candidates[int(np.argmax(accs))])


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test: do two classifiers make different errors on the same examples?"""
    a_right = pred_a == y
    b_right = pred_b == y
    only_a = int(np.sum(a_right & ~b_right))
    only_b = int(np.sum(~a_right & b_right))
    n = only_a + only_b
    p_value = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p_value)}

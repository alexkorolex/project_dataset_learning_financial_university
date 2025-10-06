from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve

def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(threshold),
    }

def find_best_threshold(y_true, y_prob, metric: str = "f1") -> float:
    # scan thresholds by unique probabilities (bounded)
    thresholds = np.unique(y_prob)
    if thresholds.size > 2000:
        thresholds = np.quantile(y_prob, q=np.linspace(0,1,2001))
    best_t, best_s = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            s = f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError("Only f1 supported in tuner")
        if s > best_s:
            best_s, best_t = s, float(t)
    return float(best_t)

def curves(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return (fpr, tpr), (prec, rec)

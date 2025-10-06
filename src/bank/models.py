from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_estimator(kind: str, params: Dict[str, Any]):
    k = kind.lower()
    if k == "logreg":
        return LogisticRegression(**params)
    if k == "rf":
        return RandomForestClassifier(**params)
    if k == "hgbt":
        return HistGradientBoostingClassifier(**params)
    raise ValueError(f"Unknown model type: {kind}")

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def make_estimator(kind: str, params: Dict[str, Any]):
    k = kind.lower()
    if k == "logreg":
        return LogisticRegression(**params)
    if k == "rf":
        return RandomForestClassifier(**params)
    if k == "hgbt":
        return HistGradientBoostingClassifier(**params)
    raise ValueError(f"Unknown model type: {kind}")

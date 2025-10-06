from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _map_yes_no(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    mapping = {"yes": 1, "no": 0}
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(mapping).astype("Int64")
    return df


def prepare_frames(train_path: str | Path, test_path: str | Path, cfg: Dict[str, Any]):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    drop_cols = cfg["data"].get("drop_cols", [])
    train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    test = test.drop(columns=[c for c in drop_cols if c in test.columns])

    yn_cols = cfg["data"].get("yn_binary_cols", [])
    train = _map_yes_no(train, yn_cols)
    test = _map_yes_no(test, yn_cols)

    numeric = list(cfg["features"]["numeric"])
    if not cfg["data"].get("use_duration", False):
        if "duration" in numeric:
            numeric.remove("duration")
        for df in (train, test):
            if "duration" in df.columns:
                df.drop(columns=["duration"], inplace=True)

    if cfg["features"].get("add_pdays_indicator", True) and "pdays" in train.columns:
        for df in (train, test):
            df["pdays_is_never"] = (df["pdays"] == -1).astype(int)
        if "pdays_is_never" not in numeric:
            numeric = numeric + ["pdays_is_never"]

    target = cfg["data"]["target"]
    X = train.drop(columns=[target])
    y = train[target].astype(int)
    X_test = test.copy()

    nsub = cfg["data"].get("subsample_rows")
    if nsub is not None and len(X) > nsub:
        pos = y[y == 1].index
        neg = y[y == 0].index
        keep_pos = int(nsub * y.mean())
        keep_neg = nsub - keep_pos
        pos_idx = np.random.RandomState(42).choice(pos, size=keep_pos, replace=False)
        neg_idx = np.random.RandomState(42).choice(neg, size=keep_neg, replace=False)
        idx = np.concatenate([pos_idx, neg_idx])
        X = X.loc[idx].reset_index(drop=True)
        y = y.loc[idx].reset_index(drop=True)

    categorical_cfg = list(cfg["features"]["categorical"])
    categorical = [c for c in categorical_cfg if c in X.columns]
    numeric = [c for c in numeric if c in X.columns]
    overlap = set(numeric) & set(categorical)
    if overlap:
        categorical = [c for c in categorical if c not in overlap]

    return X, y, X_test, numeric, categorical

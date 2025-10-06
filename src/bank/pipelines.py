from __future__ import annotations
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

def make_linear_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),  # sparse OK for LR
        ],
        remainder="drop",
    )

def make_tree_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    # Trees don't benefit from scaling; OrdinalEncoder is compact & works well.
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ],
        remainder="drop",
    )

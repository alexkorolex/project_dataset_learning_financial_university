from typing import List, Union

from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)


def create_pipeline(
    model: Union[
        LogisticRegression,
        RandomForestClassifier,
        CatBoostClassifier,
    ],
    num_features: List[str],
    cat_features: List[str],
    use_splines: bool = True,
) -> Pipeline:
    num_transformer = Pipeline(
        steps=[
            (
                "splines",
                SplineTransformer(
                    n_knots=5,
                    degree=3,
                    include_bias=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
        if use_splines
        else [("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                cat_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                model,
            ),
        ]
    )

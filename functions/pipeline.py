from typing import List, Union

import pandas as pd
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
from sklearn.utils import resample


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


def balance_with_resample(X: pd.DataFrame, y: pd.Series, method: str = "over") -> tuple:
    """
    Балансировка классов через sklearn.utils.resample

    method:
        - "over"  → Random Oversampling (увеличиваем миноритарный класс)
        - "under" → Random Undersampling (срезаем мажоритарный класс)
    """
    df = pd.concat([X, y], axis=1)
    target_col = y.name

    df_majority = df[df[target_col] == 0]
    df_minority = df[df[target_col] == 1]

    if method == "over":
        df_minority_resampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42,
        )
        df_resampled = pd.concat([df_majority, df_minority_resampled])

    elif method == "under":
        df_majority_resampled = resample(
            df_majority,
            replace=False,
            n_samples=len(df_minority),
            random_state=42,
        )
        df_resampled = pd.concat([df_majority_resampled, df_minority])

    else:
        raise ValueError("method должен быть 'over' или 'under'")

    df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

    X_res = df_resampled.drop(columns=[target_col])
    y_res = df_resampled[target_col]

    return X_res, y_res

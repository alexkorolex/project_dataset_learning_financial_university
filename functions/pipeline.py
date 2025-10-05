from typing import List, Tuple, Union

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
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
        steps=(
            [
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


def balance_with_resample(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "over",
    num_features: List[str] = None,
    cat_features: List[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Балансировка классов через resample или SMOTE.

    Args:
        X: DataFrame с признаками.
        y: Series с целевой переменной.
        method: "over", "under" или "smote".
        num_features: Список числовых признаков (для SMOTE).
        cat_features: Список категориальных признаков (для SMOTE).

    Returns:
        X_res, y_res: Сбалансированные данные.
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

    elif method == "smote":
        if num_features is None or cat_features is None:
            raise ValueError("num_features and cat_features must be provided for SMOTE")

        # Предобработка для SMOTE
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant", fill_value="unknown"
                                ),
                            ),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    cat_features,
                ),
            ]
        )

        # Применяем предобработку
        X_transformed = preprocessor.fit_transform(X)
        # Получаем имена столбцов после OneHotEncoding
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_encoded_names = ohe.get_feature_names_out(cat_features)
        all_feature_names = num_features + list(cat_encoded_names)

        # Преобразуем в DataFrame
        X_transformed = pd.DataFrame(
            X_transformed, columns=all_feature_names, index=X.index
        )

        # Применяем SMOTE
        sampler = SMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X_transformed, y)

        return pd.DataFrame(X_res, columns=all_feature_names), pd.Series(
            y_res, name=y.name
        )

    else:
        raise ValueError("method должен быть 'over', 'under' или 'smote'")

    df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
    X_res = df_resampled.drop(columns=[target_col])
    y_res = df_resampled[target_col]

    return X_res, y_res

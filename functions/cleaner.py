import numpy as np
from pandas import DataFrame


def clean_numeric_features(df: DataFrame) -> DataFrame:
    df = df.copy()

    # 1. Age — оставляем в пределах 18–90 (очень редко встречаются >90)
    df = df[(df["age"] >= 18) & (df["age"] <= 90)]

    # 2. Balance — убираем крайние 1% выбросов
    bal_low, bal_high = df["balance"].quantile([0.01, 0.99])
    df = df[(df["balance"] >= bal_low) & (df["balance"] <= bal_high)]
    df["balance_log"] = np.log1p(df["balance"].clip(lower=0))

    # 3. Duration — убираем верхние 1% строк с аномально долгими звонками
    dur_high = df["duration"].quantile(0.99)
    df = df[df["duration"] <= dur_high]

    # 4. Campaign — тоже лучше не "clip", а выкинуть слишком большие
    df = df[df["campaign"] <= 30]

    return df

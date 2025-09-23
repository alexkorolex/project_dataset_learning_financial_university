import numpy as np
import pandas as pd
from pandas import DataFrame


def clean_numeric_features(df: DataFrame) -> DataFrame:
    """
    Очистка числовых признаков с обработкой выбросов, пропусков и добавлением новых фич.

    Args:
        df: DataFrame с числовыми признаками (age, balance, duration, campaign, pdays, previous).

    Returns:
        DataFrame: Очищенный DataFrame с новыми признаками (balance_log, balance_group).
    """
    df = df.copy()
    initial_rows = len(df)

    # 1. Проверка пропусков в числовых признаках
    num_features = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    missing = df[num_features].isna().sum()
    if missing.any():
        print(f"Missing values in numeric features:\n{missing[missing > 0]}")
        # Импутация медианой для числовых признаков
        df[num_features] = df[num_features].fillna(df[num_features].median())

    # 2. Age: Ограничение 18–90 лет
    df = df[(df["age"] >= 18) & (df["age"] <= 90)]
    print(f"Removed {initial_rows - len(df)} rows with age <18 or >90")

    # 3. Balance: Обработка отрицательных значений и выбросов
    bal_low, bal_high = df["balance"].quantile([0.01, 0.99])
    df = df[(df["balance"] >= bal_low) & (df["balance"] <= bal_high)]
    print(f"Removed {initial_rows - len(df)} rows with extreme balance values")
    # Signed log для сохранения знака
    df["balance_log"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
    # Биннинг balance
    balance_bins = [-np.inf, 0, 500, 2000, np.inf]
    balance_labels = ["negative", "low", "medium", "high"]
    df["balance_group"] = pd.cut(
        df["balance"], bins=balance_bins, labels=balance_labels
    )

    # 4. Duration: Проверка корреляции и удаление верхних 1%
    corr_duration = df["duration"].corr(df["y"]) if "y" in df.columns else None
    if corr_duration is not None:
        print(f"Correlation of duration with target: {corr_duration:.3f}")
        if corr_duration > 0.3:
            print(
                "Warning: High correlation of duration with target. Consider excluding from features."
            )
    dur_high = df["duration"].quantile(0.99)
    df = df[df["duration"] <= dur_high]
    print(f"Removed {initial_rows - len(df)} rows with extreme duration values")

    # 5. Campaign: Ограничение <=30
    df = df[df["campaign"] <= 30]
    print(f"Removed {initial_rows - len(df)} rows with campaign >30")

    # 6. Pdays: Замена -1 на 999 (индикатор "нет контакта")
    df["pdays"] = df["pdays"].replace(-1, 999)
    print("Replaced pdays=-1 with 999")

    # 7. Previous: Удаление верхних 1% выбросов
    prev_high = df["previous"].quantile(0.99)
    df = df[df["previous"] <= prev_high]
    print(f"Removed {initial_rows - len(df)} rows with extreme previous values")

    # 8. Логирование итогов
    print(
        f"Total rows removed: {initial_rows - len(df)} ({(initial_rows - len(df)) / initial_rows:.2%})"
    )
    print(f"Final shape: {df.shape}")

    return df

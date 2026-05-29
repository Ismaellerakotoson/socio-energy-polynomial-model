import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame, datetime_col="Datetime"):
    df = df.copy()

    # ======================
    # FEATURES CALENDAIRES
    # ======================
    df["hour"] = df[datetime_col].dt.hour
    df["dow"] = df[datetime_col].dt.dayofweek
    df["month"] = df[datetime_col].dt.month
    df["is_weekend"] = df["dow"] >= 5

    # ======================
    # LAGS (dépendance temporelle)
    # ======================
    target_col = [c for c in df.columns if c not in ["Datetime", datetime_col]][0]

    df["lag_1"] = df[target_col].shift(1)      # t-1
    df["lag_24"] = df[target_col].shift(24)    # t-24

    # ======================
    # INERTIE (moyenne mobile)
    # ======================
    df["rolling_24"] = df[target_col].rolling(window=24).mean()
    df["rolling_168"] = df[target_col].rolling(window=168).mean()

    # ======================
    # CLEAN
    # ======================
    df = df.dropna().reset_index(drop=True)

    return df
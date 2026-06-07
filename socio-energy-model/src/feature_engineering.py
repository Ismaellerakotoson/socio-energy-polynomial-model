import pandas as pd

def add_calendar_features(df: pd.DataFrame, datetime_col="Datetime"):
    df = df.copy()

    # ======================
    # FEATURES CALENDAIRES
    # ======================
    df["hour"]       = df[datetime_col].dt.hour
    df["dow"]        = df[datetime_col].dt.dayofweek
    df["month"]      = df[datetime_col].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = 0  # tu peux enrichir avec la lib holidays si tu veux

    # ======================
    # LAGS
    # ======================
    target_col = [c for c in df.columns if c not in [datetime_col]][0]

    df["lag_1"]   = df[target_col].shift(1)    # t-1h
    df["lag_24"]  = df[target_col].shift(24)   # même heure hier ✅
    df["lag_168"] = df[target_col].shift(168)  # même heure semaine dernière ✅ (manquait)

    # ======================
    # MOYENNES MOBILES
    # ======================
    df["rolling_24"]  = df[target_col].rolling(window=24).mean()
    df["rolling_168"] = df[target_col].rolling(window=168).mean()

    # ======================
    # CLEAN
    # ======================
    df = df.dropna().reset_index(drop=True)

    return df
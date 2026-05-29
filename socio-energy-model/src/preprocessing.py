from pathlib import Path
import pandas as pd
from feature_engineering import add_calendar_features


def main():
    root = Path(__file__).resolve().parent.parent
    data_raw = root / "data" / "raw"
    data_processed = root / "data" / "processed"
    data_processed.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1. Charger CSV brut
    # -------------------------
    csv_files = list(data_raw.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("Aucun CSV trouvé dans data/raw")

    df = pd.read_csv(csv_files[0], parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # -------------------------
    # 2. Feature engineering (CENTRALISÉ)
    # -------------------------
    df = add_calendar_features(df, datetime_col="Datetime")

    # -------------------------
    # 3. Sauvegarde
    # -------------------------
    out_path = data_processed / "PJME_hourly_with_features.csv"
    df.to_csv(out_path, index=False)

    print("✔ Feature engineering terminé")
    print("Sortie :", out_path)
    print("Colonnes :", list(df.columns))


if __name__ == "__main__":
    main()
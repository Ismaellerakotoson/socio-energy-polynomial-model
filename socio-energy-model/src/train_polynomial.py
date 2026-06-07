import pandas as pd
import numpy as np
import os
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# 1. Chargement
# -------------------------
ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "PJME_hourly_with_features.csv"
df        = pd.read_csv(DATA_PATH)

# -------------------------
# 2. Features / target
# -------------------------
# ✅ CORRECTION 1 : on utilise TOUTES les features disponibles
features = [
    "hour", "dow", "month", "is_weekend",
    "lag_1", "lag_24", "lag_168",
    "rolling_24", "rolling_168"
]
target = "PJME_MW"

X = df[features]
y = df[target]

# -------------------------
# 3. Split chronologique
# -------------------------
# ✅ CORRECTION 2 : split temporel, pas aléatoire
split    = int(len(df) * 0.8)
X_train  = X.iloc[:split]
X_test   = X.iloc[split:]
y_train  = y.iloc[:split]
y_test   = y.iloc[split:]

print(f"Train : {len(X_train)} lignes | Test : {len(X_test)} lignes")

# -------------------------
# 4. Modèle
# -------------------------
degree = 2  # ✅ degré 2 suffit avec les lags, degré 3 est trop lent

model = Pipeline([
    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
    ("reg",  LinearRegression())
])

# -------------------------
# 5. Entraînement
# -------------------------
model.fit(X_train, y_train)

# -------------------------
# 6. Évaluation
# -------------------------
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n===== PERFORMANCE MODELE =====")
print(f"Degree : {degree}")
print(f"RMSE   : {rmse:.2f}")
print(f"R²     : {r2:.4f}")

# -------------------------
# 7. Sauvegarde
# -------------------------
os.makedirs("results/models", exist_ok=True)
joblib.dump(model, "results/models/polynomial_model.pkl")

# ✅ Sauvegarde aussi le split index pour que evaluate.py soit cohérent
with open("results/models/split_index.txt", "w") as f:
    f.write(str(split))

print("\n✔ Modèle sauvegardé")
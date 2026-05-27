import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# 1. Chargement data
# -------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "PJME_hourly_with_features.csv"
df = pd.read_csv(DATA_PATH)

# -------------------------
# 2. Features / target
# -------------------------
features = ["hour", "dow", "month"]
target = "PJME_MW"

X = df[features]
y = df[target]

# -------------------------
# 3. Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 4. Modèle polynomial
# -------------------------
degree = 3

model = Pipeline([
    ("poly", PolynomialFeatures(degree=degree)),
    ("reg", LinearRegression())
])

# -------------------------
# 5. Train
# -------------------------
model.fit(X_train, y_train)

# -------------------------
# 6. Predict
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# 7. Evaluation
# -------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== PERFORMANCE MODELE =====")
print("Degree :", degree)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)

# -------------------------
# 8. Save model
# -------------------------
os.makedirs("results/models", exist_ok=True)

joblib.dump(model, "results/models/polynomial_model.pkl")

print("\n✔ Modèle sauvegardé : results/models/polynomial_model.pkl")
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# 1. Load data
# -------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "PJME_hourly_with_features.csv"
df = pd.read_csv(DATA_PATH)

features = ["hour", "dow", "month"]
target = "PJME_MW"

X = df[features]
y = df[target]

# -------------------------
# 2. Load model
# -------------------------
model = joblib.load("results/models/polynomial_model.pkl")

# -------------------------
# 3. Predict
# -------------------------
y_pred = model.predict(X)

# -------------------------
# 4. Metrics
# -------------------------
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\n===== EVALUATION =====")
print("MAE  :", mae)
print("RMSE :", rmse)
print("R2   :", r2)

# -------------------------
# 5. Plot comparaison
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(y.values[:200], label="True")
plt.plot(y_pred[:200], label="Predicted")
plt.title("Régression polynomiale - comparaison")
plt.legend()
plt.grid()
plt.show()

# -------------------------
# 6. Scatter plot
# -------------------------
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, alpha=0.3)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("True vs Predicted")
plt.grid()
plt.show()
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# 1. Chargement
# -------------------------
ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "PJME_hourly_with_features.csv"
df        = pd.read_csv(DATA_PATH)

features = [
    "hour", "dow", "month", "is_weekend",
    "lag_1", "lag_24", "lag_168",
    "rolling_24", "rolling_168"
]
target = "PJME_MW"

# -------------------------
# 2. Récupérer le même split
# -------------------------
with open("results/models/split_index.txt") as f:
    split = int(f.read())

X_test = df[features].iloc[split:]
y_test = df[target].iloc[split:]

# -------------------------
# 3. Chargement modèle + prédiction
# -------------------------
model  = joblib.load("results/models/polynomial_model.pkl")
y_pred = model.predict(X_test)

# -------------------------
# 4. Métriques
# -------------------------
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n===== EVALUATION (test set uniquement) =====")
print(f"MAE  : {mae:.2f} MW")
print(f"RMSE : {rmse:.2f} MW")
print(f"R²   : {r2:.4f}")

# -------------------------
# 5. Graphique comparaison
# -------------------------
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:300], label="Réel",   color="steelblue")
plt.plot(y_pred[:300],        label="Prédit", color="orange", alpha=0.8)
plt.title("Valeurs réelles vs prédites (300 premières heures du test)")
plt.xlabel("Heure")
plt.ylabel("MW")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/actual_vs_predicted.png")
plt.show()

# -------------------------
# 6. Distribution des erreurs
# -------------------------
errors = y_test.values - y_pred

plt.figure(figsize=(8, 4))
plt.hist(errors, bins=60, color="steelblue", edgecolor="white")
plt.axvline(0, color="red", linestyle="--", linewidth=2)
plt.title("Distribution des erreurs")
plt.xlabel("Erreur (MW)")
plt.ylabel("Fréquence")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/error_distribution.png")
plt.show()

# -------------------------
# 7. Scatter plot actual vs predicted
# -------------------------
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.figure(figsize=(8, 8))
plt.scatter(
    y_test, y_pred,
    alpha=0.2,
    s=5,                  # points plus petits car beaucoup d'observations
    color="steelblue",
    label="Prédictions"
)
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="red", linestyle="--", linewidth=2,
    label="Prédiction parfaite"
)
plt.xlabel("Valeurs réelles (MW)")
plt.ylabel("Valeurs prédites (MW)")
plt.title(f"Actual vs Predicted — R² = {r2:.4f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/scatter_actual_vs_predicted.png")
plt.show()
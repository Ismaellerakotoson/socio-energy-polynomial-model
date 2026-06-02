from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 1. FIND PROJECT ROOT
# =========================================================
CURRENT = Path(__file__).resolve()

ROOT = None
for parent in CURRENT.parents:
    if (parent / "results").exists() and (parent / "socio-energy-model").exists():
        ROOT = parent
        break

if ROOT is None:
    raise FileNotFoundError("❌ Impossible de trouver la racine du projet")

print("✔ ROOT =", ROOT)

# =========================================================
# 2. PATHS
# =========================================================
DATA_PATH = ROOT / "socio-energy-model" / "data" / "processed" / "PJME_hourly_with_features.csv"

MODELS_DIR = ROOT / "results" / "models"
PLOTS_DIR = ROOT / "results" / "plots"
REPORT_PATH = ROOT / "results" / "report.html"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 3. LOAD DATA
# =========================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ Dataset introuvable: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# =========================================================
# 4. LOAD MODEL
# =========================================================
model_files = list(MODELS_DIR.glob("*.pkl"))

if len(model_files) == 0:
    raise FileNotFoundError(f"❌ Aucun modèle trouvé dans {MODELS_DIR}")

MODEL_PATH = model_files[0]
print("✔ MODEL =", MODEL_PATH)

model = joblib.load(MODEL_PATH)

# =========================================================
# 5. FEATURES SAFE
# =========================================================
if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)
else:
    features = [
        "hour",
        "dow",
        "month",
        "is_weekend",
        "lag_1",
        "lag_24",
        "rolling_24",
        "rolling_168"
    ]

features = [f for f in features if f in df.columns]

X = df[features]
y = df["PJME_MW"]

# =========================================================
# 6. PREDICTIONS
# =========================================================
y_pred = model.predict(X)

# =========================================================
# 7. METRICS
# =========================================================
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# =========================================================
# 8. PLOTS (SAVE IMAGES)
# =========================================================

plt.figure(figsize=(12, 5))
plt.plot(y.iloc[:500].values, label="Actual")
plt.plot(y_pred[:500], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Energy")

plot1 = PLOTS_DIR / "actual_vs_predicted.png"
plt.savefig(plot1, bbox_inches="tight")
plt.close()

errors = y - y_pred

plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50)
plt.title("Error Distribution")

plot2 = PLOTS_DIR / "error_distribution.png"
plt.savefig(plot2, bbox_inches="tight")
plt.close()

hourly = df.groupby("hour")["PJME_MW"].mean()

plt.figure(figsize=(10, 5))
hourly.plot()
plt.title("Hourly Consumption Pattern")

plot3 = PLOTS_DIR / "hourly_consumption.png"
plt.savefig(plot3, bbox_inches="tight")
plt.close()

print("✔ IMAGES GENERATED")

# =========================================================
# 9. IMPORTANT FIX (HTML IMAGE PATHS)
# =========================================================

# 👉 IMPORTANT: use RELATIVE PATHS (browser friendly)
plot1_html = "plots/actual_vs_predicted.png"
plot2_html = "plots/error_distribution.png"
plot3_html = "plots/hourly_consumption.png"

# =========================================================
# 10. HTML REPORT
# =========================================================
html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Socio-Energy Report</title>

<style>
body {{
    font-family: Arial;
    margin: 40px;
    background: #f4f4f4;
}}

h1 {{
    color: #222;
}}

.card {{
    background: white;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 10px;
}}

img {{
    max-width: 100%;
}}
</style>
</head>

<body>

<h1>Socio-Energy Forecast Report</h1>

<div class="card">
    <h2>Dataset</h2>
    <p><b>Rows:</b> {len(df)}</p>
    <p><b>Columns:</b> {len(df.columns)}</p>
</div>

<div class="card">
    <h2>Model Performance</h2>
    <p>MAE: {mae:.2f}</p>
    <p>RMSE: {rmse:.2f}</p>
    <p>R²: {r2:.4f}</p>
</div>

<div class="card">
    <h2>Features Used</h2>
    <ul>
        {''.join(f"<li>{f}</li>" for f in features)}
    </ul>
</div>

<div class="card">
    <h2>Actual vs Predicted</h2>
    <img src="{plot1_html}">
</div>

<div class="card">
    <h2>Error Distribution</h2>
    <img src="{plot2_html}">
</div>

<div class="card">
    <h2>Hourly Pattern</h2>
    <img src="{plot3_html}">
</div>

<div class="card">
    <h2>Model File</h2>
    <p>{MODEL_PATH}</p>
</div>

</body>
</html>
"""

# =========================================================
# 11. SAVE REPORT
# =========================================================
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("✔ REPORT GENERATED :", REPORT_PATH)
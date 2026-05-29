from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "PJME_hourly_with_features.csv"
MODEL_PATH = ROOT / "results" / "models" / "polynomial_model.pkl"
REPORT_PATH = ROOT / "results" / "report.html"

# =========================
# Load data
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# Basic info
# =========================
n_rows = len(df)
columns = list(df.columns)

# =========================
# Generate plot
# =========================
plot_path = ROOT / "results" / "plots"
plot_path.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 4))

target_col = "PJME_MW"

ax.plot(df[target_col].head(500))
ax.set_title("Energy Consumption Sample")
ax.set_xlabel("Time")
ax.set_ylabel("MW")

img_path = plot_path / "sample_plot.png"

fig.savefig(img_path)
plt.close()

# =========================
# HTML Report
# =========================
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

        code {{
            background: #eee;
            padding: 2px 5px;
        }}
    </style>
</head>

<body>

<h1>Socio-Energy Polynomial Forecasting Report</h1>

<div class="card">
    <h2>Dataset</h2>

    <p><b>Rows:</b> {n_rows}</p>

    <p><b>Columns:</b></p>

    <ul>
        {''.join(f"<li>{c}</li>" for c in columns)}
    </ul>
</div>

<div class="card">
    <h2>Model</h2>

    <p>Polynomial Regression</p>

    <p>Saved model:</p>

    <code>{MODEL_PATH}</code>
</div>

<div class="card">
    <h2>Features Used</h2>

    <ul>
        <li>hour</li>
        <li>dow</li>
        <li>month</li>
        <li>is_weekend</li>
        <li>lag_1</li>
        <li>lag_24</li>
        <li>rolling_24</li>
        <li>rolling_168</li>
    </ul>
</div>

<div class="card">
    <h2>Visualization</h2>

    <img src="plots/sample_plot.png" width="900">
</div>

</body>
</html>
"""

# =========================
# Save report
# =========================
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("✔ Rapport généré :", REPORT_PATH)
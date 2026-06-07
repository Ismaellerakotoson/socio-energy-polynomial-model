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
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport de Modélisation Socio-Énergétique</title>

<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background: #f4f4f4;
    line-height: 1.7;
}}

h1,h2 {{
    color: #1f3c88;
}}

.card {{
    background: white;
    padding: 25px;
    margin-bottom: 25px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}

img {{
    max-width: 100%;
    border-radius: 8px;
}}

p {{
    text-align: justify;
}}
</style>
</head>

<body>

<h1>Rapport de Modélisation et d’Analyse des Données Socio-Énergétiques</h1>

<div class="card">
    <h2>1. Présentation du projet</h2>

    <p>
    Ce projet a pour objectif d'étudier les comportements de consommation
    énergétique à partir de données horaires de charge électrique.
    Une approche de régression polynomiale a été utilisée afin de modéliser
    les relations non linéaires entre les variables temporelles et la
    consommation énergétique observée.
    </p>

    <p>
    Les données proviennent du jeu de données PJME Hourly Energy Consumption
    et contiennent des mesures horaires de consommation électrique.
    </p>
</div>

<div class="card">
    <h2>2. Description du jeu de données</h2>

    <p><b>Nombre d'observations :</b> {len(df):,}</p>
    <p><b>Nombre de variables :</b> {len(df.columns)}</p>

    <p>
    Après le nettoyage et le prétraitement des données, plusieurs variables
    temporelles ont été extraites afin d'améliorer les capacités prédictives
    du modèle.
    </p>
</div>

<div class="card">
    <h2>3. Variables utilisées</h2>

    <p>
    Les variables explicatives utilisées pour entraîner le modèle sont :
    </p>

    <ul>
        {''.join(f"<li>{f}</li>" for f in features)}
    </ul>

    <p>
    Ces variables permettent de capturer les effets liés aux heures de la
    journée, aux jours de la semaine, aux mois ainsi qu'aux tendances
    temporelles de la consommation électrique.
    </p>
</div>

<div class="card">
    <h2>4. Performance du modèle</h2>

    <p>
    Les performances du modèle de régression polynomiale ont été évaluées
    à l'aide de plusieurs métriques statistiques.
    </p>

    <ul>
        <li><b>MAE :</b> {mae:.2f}</li>
        <li><b>RMSE :</b> {rmse:.2f}</li>
        <li><b>R² :</b> {r2:.4f}</li>
    </ul>

    <p>
    Le coefficient de détermination R² mesure la proportion de la variance
    expliquée par le modèle. Plus sa valeur est proche de 1, meilleure est
    la qualité des prédictions.
    </p>
</div>

<div class="card">
    <h2>5. Comparaison entre valeurs réelles et prédites</h2>

    <img src="{plot1_html}">

    <p>
    Ce graphique compare les valeurs réelles de consommation électrique
    aux valeurs prédites par le modèle. Une proximité importante entre les
    deux courbes indique que le modèle reproduit correctement les variations
    de la charge énergétique.
    </p>
</div>

<div class="card">
    <h2>6. Distribution des erreurs</h2>

    <img src="{plot2_html}">

    <p>
    L'histogramme des erreurs permet d'analyser les écarts entre les valeurs
    observées et les prédictions du modèle. Une distribution centrée autour
    de zéro suggère l'absence de biais systématique dans les prédictions.
    </p>
</div>

<div class="card">
    <h2>7. Profil horaire de consommation</h2>

    <img src="{plot3_html}">

    <p>
    Cette représentation met en évidence l'évolution moyenne de la
    consommation électrique selon l'heure de la journée.
    Elle permet d'identifier les périodes de forte demande énergétique
    ainsi que les périodes creuses.
    </p>
</div>

<div class="card">
    <h2>8. Conclusion</h2>

    <p>
    Les résultats obtenus montrent que la régression polynomiale constitue
    une approche pertinente pour modéliser les comportements énergétiques
    non linéaires. Les variables temporelles extraites permettent de capturer
    une part importante de la variabilité de la consommation électrique.
    </p>

    <p>
    Des améliorations futures pourraient inclure l'intégration de variables
    météorologiques, économiques ou démographiques afin d'accroître la
    précision des prédictions et d'enrichir l'analyse socio-énergétique.
    </p>
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
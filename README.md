# ⚡ Socio-Energy Model — Prévision de Charge Électrique par Régression Polynomiale

> Modélisation non-linéaire de la consommation énergétique (données PJM) avec feature engineering temporel, régression polynomiale et pipeline ML complet.

---

## 📋 Table des matières

- [Aperçu du projet](#aperçu-du-projet)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Données](#données)
- [Pipeline complet](#pipeline-complet)
- [Fichiers principaux](#fichiers-principaux)
- [Configuration](#configuration)
- [Résultats](#résultats)
- [Auteur](#auteur)

---

## 🔍 Aperçu du projet

Ce projet implémente un pipeline de **prévision de la demande électrique horaire** sur les données historiques du réseau PJM (Pennsylvania-New Jersey-Maryland Interconnection).

L'approche repose sur :
- Une **analyse exploratoire** (EDA) des séries temporelles de consommation
- Un **feature engineering** calendaire (heure, jour de la semaine, mois, lags, moyennes mobiles)
- Un **modèle de régression polynomiale** (degré 3) entraîné via scikit-learn
- Une **évaluation quantitative** (MAE, RMSE, R²) et des visualisations comparatives
- La **génération automatique d'un rapport HTML**

---

## 📁 Structure du projet

```
socio-energy-model/
│
├── config.yaml                        # Configuration centrale (chemins, hyperparamètres)
│
├── data/
│   ├── raw/                           # Données brutes (PJME_hourly.csv)
│   └── processed/                     # Données nettoyées et enrichies
│       ├── PJME_hourly_clean.csv
│       ├── PJME_hourly_with_features.csv
│       ├── train_with_features.csv
│       ├── val_with_features.csv
│       └── test_with_features.csv
│
├── energy_forecast/
│   └── notebooks/
│       └── 01_eda_pjm_hourly_load.ipynb   # Analyse exploratoire (EDA)
│
├── src/
│   ├── feature_engineering.py         # Génération des features temporelles
│   ├── preprocessing.py               # Pipeline de prétraitement complet
│   ├── train_polynomial.py            # Entraînement du modèle polynomial
│   ├── evaluate.py                    # Évaluation et visualisations
│   └── generate_report.py             # Génération du rapport HTML
│
├── results/
│   ├── models/
│   │   └── polynomial_model.pkl       # Modèle entraîné (joblib)
│   └── predictions/
│       └── rapport.html               # Rapport final
│
└── requirements.txt                   # Dépendances Python
```

---

## ⚙️ Prérequis

- **Python** ≥ 3.10
- **Anaconda** (recommandé pour la gestion de l'environnement)
- **VS Code** avec l'extension Jupyter
- **Git Bash** ou **PowerShell** (Windows)

---

## 🚀 Installation

### Option A — Environnement Anaconda (recommandé)

```bash
# 1. Créer un environnement conda dédié
conda create -n socio-energy python=3.10 -y
conda activate socio-energy

# 2. Installer les dépendances via conda-forge
conda install -c conda-forge pandas
conda install -c conda-forge numpy
conda install -c conda-forge matplotlib
conda install -c conda-forge seaborn
conda install -c conda-forge plotly
conda install -c conda-forge scikit-learn
conda install -c conda-forge holidays
conda install -c conda-forge pyyaml
conda install -c conda-forge jupyter
conda install -c conda-forge tqdm
```

### Option B — Environnement virtuel Python (venv)

```bash
# 1. Créer et activer l'environnement
python -m venv .venv

# Git Bash (Windows)
source .venv/Scripts/activate

# PowerShell (Windows)
.\.venv\Scripts\Activate.ps1

# 2. Mettre à jour pip et installer les dépendances
python -m pip install -U pip
python -m pip install -r requirements.txt

# 3. Vérifier l'installation
python -c "import torch, pandas, sklearn, yaml; print('OK')"
```

### Contenu de `requirements.txt`

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
scikit-learn>=1.3.0
holidays>=0.30
pyyaml>=6.0
jupyter>=1.0.0
tqdm>=4.65.0
```

---

## 📊 Données

Placer le fichier CSV brut dans `data/raw/` :

| Fichier attendu         | Source                          |
|-------------------------|---------------------------------|
| `PJME_hourly.csv`       | Kaggle — PJM Hourly Energy Consumption |
| `PJM_Load_hourly.csv`   | Alternative acceptée            |

Le fichier doit contenir au minimum les colonnes :
- `Datetime` — horodatage horaire (ex : `2002-01-01 01:00:00`)
- `PJME_MW` — consommation en mégawatts

---

## 🔄 Pipeline complet

Exécuter les étapes dans l'ordre suivant :

### Étape 1 — Analyse exploratoire (EDA)

```bash
jupyter notebook energy_forecast/notebooks/01_eda_pjm_hourly_load.ipynb
```

> Charge le CSV brut, inspecte les types, détecte les valeurs manquantes, gère les doublons DST, puis sauvegarde un fichier propre dans `data/processed/`.

---

### Étape 2 — Feature engineering & prétraitement

```bash
python energy_forecast/src/preprocessing.py --all
```

> Appelle `feature_engineering.py` pour générer les features calendaires (heure, jour, mois, weekend, lags, moyennes mobiles) et exporte `PJME_hourly_with_features.csv`.

**Features générées :**

| Feature        | Description                            |
|----------------|----------------------------------------|
| `hour`         | Heure de la journée (0–23)             |
| `dow`          | Jour de la semaine (0 = lundi)         |
| `month`        | Mois de l'année (1–12)                 |
| `is_weekend`   | Booléen week-end                       |
| `lag_1`        | Consommation à t-1 heure               |
| `lag_24`       | Consommation à t-24 heures             |
| `rolling_24`   | Moyenne mobile sur 24h                 |
| `rolling_168`  | Moyenne mobile sur 7 jours (168h)      |

---

### Étape 3 — Entraînement du modèle polynomial

```bash
python socio-energy-model/src/train_polynomial.py --config socio-energy-model/config.yaml
```

> Entraîne un pipeline `PolynomialFeatures (degré 3) → LinearRegression` sur les features `[hour, dow, month]`. Le modèle est sauvegardé dans `results/models/polynomial_model.pkl`.

---

### Étape 4 — Évaluation

```bash
python socio-energy-model/src/evaluate.py --config socio-energy-model/config.yaml
```

> Calcule MAE, RMSE et R² sur l'ensemble de test. Génère deux visualisations :
> - Comparaison valeurs réelles vs prédites (200 premiers points)
> - Nuage de points True vs Predicted

---

### Étape 5 — Génération du rapport HTML

```bash
python socio-energy-model/src/generate_report.py
```

> Produit `results/predictions/rapport.html` avec les métriques, graphiques et synthèse du modèle.

---

## 📄 Fichiers principaux

### `src/feature_engineering.py`

Fonction centrale `add_calendar_features(df, datetime_col)` :
- Features temporelles (heure, jour, mois, weekend)
- Lags temporels (t-1, t-24)
- Moyennes mobiles (24h, 168h)
- Suppression automatique des NaN générés

### `src/preprocessing.py`

Script de prétraitement complet :
- Chargement du CSV brut depuis `data/raw/`
- Appel de `add_calendar_features`
- Sauvegarde dans `data/processed/PJME_hourly_with_features.csv`

### `src/train_polynomial.py`

Pipeline d'entraînement :
- Split train/test (80/20, `random_state=42`)
- `sklearn.pipeline.Pipeline` avec `PolynomialFeatures` + `LinearRegression`
- Affichage des métriques (MSE, RMSE, R²)
- Sauvegarde du modèle avec `joblib`

### `src/evaluate.py`

Évaluation post-entraînement :
- Chargement du modèle `.pkl`
- Calcul MAE, RMSE, R²
- Graphiques matplotlib (courbe temporelle + scatter)

---

## ⚙️ Configuration (`config.yaml`)

```yaml
paths:
  train_csv: data/processed/train_with_features.csv
  val_csv:   data/processed/val_with_features.csv
  test_csv:  data/processed/test_with_features.csv
  model_dir: results/models
  outputs_dir: results/predictions

model:
  name: polynomial_regression
  degree: 3          # Degré du polynôme (à ajuster)
  features:
    - hour
    - dayofweek
    - month

target: load_MW

training:
  test_size: 0.2
  random_state: 42

device: cpu
```

---

## 📈 Résultats attendus

| Métrique | Valeur typique (degré 3) |
|----------|--------------------------|
| MAE      | ~1 500 – 2 500 MW        |
| RMSE     | ~2 000 – 3 500 MW        |
| R²       | ~0.55 – 0.75             |

> **Note :** Les performances varient selon le degré polynomial choisi et les features utilisées. L'ajout des lags (`lag_1`, `lag_24`) améliore significativement le R².

---

## 👤 Auteur

Projet de modélisation socio-énergétique non-linéaire — Régression polynomiale sur données PJM.

---

*Généré avec Python · scikit-learn · pandas · matplotlib*

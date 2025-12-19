# GameProofer ML Proof-of-Concept

## Project summary

This repository documents a forward-facing walkthrough of the GameProofer proof-of-concept for Latitude 64. The Kaggle “Disc Flight Numbers and Dimensions” catalog acts as a proxy for the throw-level GameProofer data that will ultimately be used in production.

Because the proxy dataset is **specs-only** (no throws/players/timestamps/quality labels), the goal is **methodology validation + clear data requirements**, not a final production model.

**Key proxy results (from notebooks):**

- `01_exploratory_analysis.ipynb`: data audit, domain checks, correlation analysis, hypothesis tests.
- `02_clustering_methodology.ipynb`: K-Means clustering with **K=4** and **silhouette=0.307** (+ PCA visualization).
- `03_classification_model.ipynb`: Random Forest classifier with **test accuracy=0.957** and 5-fold CV **0.959 ± 0.009**.
- `04_quality_prediction.ipynb`: regression proxy (predicting disc `SPEED`) with **R²=0.950**, **MAE=0.585**, **RMSE=0.797**.

All figures and intermediate datasets are written to `data/processed/` so that they can be reviewed in one directory for supporting evidence.

---

## Quickstart (reproduce results)

1. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download the Kaggle dataset into `data/raw/`:
   - Ensure Kaggle credentials exist at `~/.kaggle/kaggle.json`.
   - Run:

     ```bash
     python src/data_loading/download_kaggle_data.py
     ```

3. Run notebooks in order:
   - `notebooks/01_exploratory_analysis.ipynb`
   - `notebooks/02_clustering_methodology.ipynb`
   - `notebooks/03_classification_model.ipynb`
   - `notebooks/04_quality_prediction.ipynb`

When executed, notebooks write intermediate outputs to `data/processed/` and persist trained artifacts to `models/`.

---

## Repository structure

```
GameProofer-sample/
├── README.md
├── docs/
│   ├── 01_BUSINESS_UNDERSTANDING.md
│   ├── 02_DATA_UNDERSTANDING.md
│   ├── 03_EDA_FINDINGS_AND_ML_FEASIBILITY.md
│   ├── 04_ML_SOLUTIONS_RESEARCH.md
│   └── 05_GAMEPROOFER_ML_FEASIBILITY_ANALYSIS.md
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_clustering_methodology.ipynb
│   ├── 03_classification_model.ipynb
│   └── 04_quality_prediction.ipynb
├── data/
│   ├── raw/               # Kaggle CSV (after download)
│   └── processed/         # generated outputs (after running notebooks)
├── models/                # persisted sklearn artifacts (joblib)
├── src/
│   ├── analysis/
│   ├── common/
│   ├── data_loading/
│   └── preprocessing/
├── requirements.txt
└── .gitignore
```

---

## Documentation map

Suggested reading order (short complements to the notebooks):

- `docs/01_BUSINESS_UNDERSTANDING.md`: Business framing and updated scope after data discovery.
- `docs/02_DATA_UNDERSTANDING.md`: Kaggle catalog audit, data quality notes, cleaning steps.
- `docs/03_EDA_FINDINGS_AND_ML_FEASIBILITY.md`: Summary of exploratory analysis and feasibility decisions.
- `docs/04_ML_SOLUTIONS_RESEARCH.md`: Results-driven write-up for the five ML concepts.
- `docs/05_GAMEPROOFER_ML_FEASIBILITY_ANALYSIS.md`: Proof-of-concept evidence + data requirements for real GameProofer integration.

---

## Notebooks at a glance

- **`01_exploratory_analysis.ipynb`** – Data audit, newline fix, summary statistics, correlation analysis.
- **`02_clustering_methodology.ipynb`** – Feature scaling, elbow + silhouette search, PCA visualization, saved K-Means + scaler.
- **`03_classification_model.ipynb`** – Random Forest classifier on cluster labels, evaluation metrics, confusion matrix, persisted model.
- **`04_quality_prediction.ipynb`** – Regression proxy predicting `SPEED`, residual diagnostics, saved regressor.

Run notebooks in order (they assume outputs from previous steps). All dependencies are listed in `requirements.txt`.

---

## Saved artifacts

The `models/` directory contains joblib files for every trained estimator, plus feature manifests and scalers:

- `disc_cluster_model.pkl`, `disc_cluster_scaler.pkl`, `disc_cluster_features.pkl`
- `skill_classifier.pkl`, `feature_scaler.pkl`
- `quality_regressor.pkl`, `quality_feature_columns.json`, `quality_metrics.json`

Use them for quick demos or to verify notebook outputs without re-training.

---

## Next steps with real GameProofer data

The documentation spells out the throw-level dataset we need (player IDs, release metrics, quality labels, timestamps) and how we will extend these proofs-of-concept into production-ready solutions once Latitude 64 provides access.

---

*Last updated: 2025-12-19*

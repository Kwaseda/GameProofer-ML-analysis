# EDA Findings & ML Solution Feasibility

This file summarizes what the **proxy notebooks (01–04)** establish for the GameProofer project. It is intentionally short; detailed reasoning, plots, and code live in the notebooks.

---

## What we can (and cannot) prove with the proxy dataset

- The Kaggle dataset is a **disc catalog** (one row per mold). It has **no throws, no players, no timestamps, and no `quality` labels**.
- Therefore:
  - We **can** validate the workflow: cleaning → EDA → clustering → classification → regression.
  - We **cannot** claim final performance for real GameProofer features/targets until throw-level data is available.

---

## Key observations from Notebook 01

| Theme | Evidence | Why it matters |
|-------|----------|----------------|
| Data type reality check | 1,175 disc specs (15 columns), not sensor logs. | Sets correct expectations for what is demo-able today. |
| Basic data quality | No missing values; 1 domain exception flagged (`GLIDE=0`). | Cleaning/validation templates are ready for the real dataset. |
| Redundant features | Example: `SPEED` ↔ `RIM WIDTH (cm)` correlation **0.964**. | Informs feature selection and motivates engineered ratios in regression. |
| Artifacts exported | Cleaned CSV, correlation heatmap, distribution plots, hypothesis test table. | Makes the analysis reproducible and easy to review. | 

---

## Feasibility evidence (Notebooks 02–04)

| Workstream | Notebook | Evidence produced |
|-----------|----------|------------------|
| Disc archetype clustering (proxy for skill clustering) | `02_clustering_methodology.ipynb` | K-Means selects **K=4** with **silhouette=0.307** and PCA visualization (85.6% variance explained by first 2 PCs). |
| Cluster-label classification (proxy for “assign skill tier”) | `03_classification_model.ipynb` | Random Forest reaches **test accuracy=0.957** and 5-fold CV **0.959 ± 0.009**. Feature importance is dominated by `STABILITY` and `SPEED`. |
| Regression workflow (proxy for quality prediction) | `04_quality_prediction.ipynb` | Random Forest predicting disc `SPEED` achieves **R²=0.950**, **MAE=0.585**, **RMSE=0.797** (5-fold CV R² ≈ **0.950 ± 0.010**). Engineered ratio features dominate importance. |

---

## What changes with real GameProofer data

- **Inputs change:** flight numbers/dimensions → release metrics (e.g., armSpeed/spin/wobble/angles).
- **Targets change:** proxy labels/`SPEED` → skill tiers and `quality` (1–5).
- **New capabilities:** time series enables anomaly detection and trend tracking.

---

## Next actions when throw-level data arrives

1. Re-run Notebook 01 templates on the new exports (validate ranges, missingness, correlations).
2. Retrain the 2-stage pipeline (clustering → classifier) on player/throw features.
3. Retrain regression to predict `quality` using release features; benchmark against the proxy evaluation structure.

---

*Last updated: 2025-12-15*
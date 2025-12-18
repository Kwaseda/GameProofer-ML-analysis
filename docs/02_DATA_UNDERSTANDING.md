# Data Understanding – Kaggle Disc Catalog (Proxy)

`notebooks/01_exploratory_analysis.ipynb` contains the authoritative work for this phase. This Markdown file is only a short summary that clarifies what data we actually had, what checks we ran, and which artifacts were produced.

---

## Data source summary

- **Dataset:** Kaggle “Disc Golf Disc Flight Numbers and Dimensions” (CSV)
- **Raw file (expected path):** `data/raw/disc-data.csv` (download script: `python src/data_loading/download_kaggle_data.py`)
- **Granularity:** one row per disc model (no throws, players, or timestamps)
- **Shape (raw):** **1,175 rows × 15 columns** (3 categorical, 12 numeric)
- **Cleaned output:** `data/processed/disc_golf_cleaned.csv` (same columns + `DISC_TYPE_CLEAN` for consistent grouping)

Because this is catalog data, not GameProofer throw logs, we use it to validate methodology only. The structure and quality checks described below will be reused when true throw-level data is delivered.

---

## Cleaning and validation performed (Notebook 01)

- **Text normalization:** strip whitespace and replace newline characters (`\n`, `\r`) across all string columns.
- **Basic quality checks:** missing-value scan plus simple domain bounds.
- **Flagged issue:** **1 record has `GLIDE=0`**, which violates the expected 1–7 range (kept as-is but explicitly flagged).
- **Exports:** the notebook persists the cleaned CSV plus key plots and hypothesis-test outputs to `data/processed/`.

---

## Key statistics (cleaned data)

- `SPEED` min/max: **1.0 – 14.5**, mean **6.99**
- `GLIDE` min/max: **0 – 7**, mean **4.29**
- `TURN` min/max: **-5 – 2**, mean **-0.84**
- `FADE` min/max: **0 – 6**, mean **2.01**
- Geometry is tightly constrained (e.g., `DIAMETER (cm)` mean **21.30**, std **0.33**), which matters for downstream feature selection.

These numbers provide sanity checks and benchmark the code that will later run on GameProofer metrics.

---

## Correlation highlights (Pearson)

| Pair | Correlation | Why it matters |
|------|-------------|----------------|
| `SPEED` ↔ `RIM WIDTH (cm)` | **0.964** | strong redundancy; be careful using both in linear models |
| `RIM WIDTH (cm)` ↔ `INSIDE RIM DIAMETER (cm)` | **0.951** | geometry variables are highly coupled |
| `RIM DEPTH (cm)` ↔ `RIM DEPTH / DIAMETER RATION (%)` | **0.982** | derived ratios can dominate unless you control multicollinearity |
| `TURN` ↔ `STABILITY` | **0.835** | stability metadata is closely tied to the turn rating |

The correlation plot produced in `01_exploratory_analysis.ipynb` is a direct template for analyzing sensor metrics once throw data arrives.

---

## Insights that drive later notebooks

- **Disc archetypes exist in flight numbers**, which supports the clustering demonstration in `02_clustering_methodology.ipynb`.
- **The dataset has no throw-level target**, so `04_quality_prediction.ipynb` uses **disc `SPEED` as a regression proxy** to validate the end-to-end workflow.
- **Feature redundancy is real**, so later notebooks either standardize carefully (K-Means) or rely on engineered ratios (regression).

---
 
## Reuse checklist for GameProofer data
 
- Apply the same string normalization to categorical fields (e.g., `shotType`).
- Validate numeric ranges early (sensor bounds, angle limits, physically plausible values).
- Re-run summary statistics + correlation heatmaps to detect redundancy before modeling.
- Persist cleaned datasets and plots to `data/processed/` so modeling is fully reproducible. 

---

## Deliverables produced at this stage

- `data/processed/disc_golf_cleaned.csv`
- `data/processed/correlation_heatmap.png`
- `data/processed/feature_distributions_violin_z_score.png`
- `data/processed/skewed_feature_violins_z_score.png`
- `data/processed/hypothesis_test_results.csv`

This completes CRISP-DM Phase 2 for the proxy study and primes the pipeline for real GameProofer sensor data.

---

*Last updated: 2025-12-15*

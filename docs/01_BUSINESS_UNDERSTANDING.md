# Business Understanding – GameProofer proof-of-concept

## Purpose

Latitude 64 wants to use some planned GameProofer machine-learning solutions for their purposes, and to choose which approach better suits their tasks, we conduct a proxy study. Because that GameProofer dataset is not yet available due to some security and privacy reasons, we executed the proxy study using the Kaggle disc catalog. This document re-states the stakeholders, objectives, and success criteria after completing the proxy analysis.

---

## Stakeholders and goals

- **Latitude 64 leadership** – confirm technical feasibility before committing engineering resources; understand data requirements and the staged roadmap.
- **GameProofer product team** – review the methodology for skill classification, quality prediction, and recommendation features, and plan how those models will be integrated when throw data arrives.
- **Players and coaches** – ultimately receive objective, real-time feedback that shortens the improvement cycle.
- **Project study team (Me)** – demonstrate disciplined CRISP-DM execution, reproducible code, and professor-ready documentation.

---

## Updated business context

1. The original scope assumed access to throw-level metrics (armSpeed, spin, wobble, launch, nose, hyzer, quality ratings, timestamps).
2. Data discovery confirmed only catalog-level information is available today (disc flight numbers and dimensions). No player identifiers, targets, or timestamps exist in the source file.
3. We pivoted to a proof-of-concept track:
   - Cluster disc archetypes to mimic skill segmentation.
   - Train a classifier on those clusters to validate the two-stage pipeline.
   - Predict disc speed from physical dimensions as a regression proxy for quality prediction.
4. Each notebook documents the adapted scope and preserves every artifact required to re-run the analysis when true GameProofer data is supplied.

---

## Business objectives (proxy study)

1. **Demonstrate methodology** – show that clustering, classification, and regression workflows execute cleanly and produce defensible metrics on the proxy dataset.
2. **Quantify data requirements** – translate lessons from the catalog data into concrete throw-count targets for the GameProofer MVP and production phases.
3. **Provide decision support** – equip the professor and Latitude 64 stakeholders with a structured narrative, saved artifacts, and a phased roadmap.

---

## Success criteria

| Objective | Proxy Evidence | Real-Data Requirement |
|-----------|----------------|-----------------------|
| Skill clustering | K-Means selects **K=4** with **silhouette=0.307** (`02_clustering_methodology.ipynb`) | ≥50 players × 50 throws with armSpeed/spin/wobble
| Skill classification | Random Forest **test accuracy=0.957**, 5-fold CV **0.959 ± 0.009** (`03_classification_model.ipynb`) | ≥2,500 labeled throws for training + validation
| Quality prediction | Regression proxy (predicting disc SPEED) reaches **R²=0.950**, **MAE=0.585**, **RMSE=0.797** (`04_quality_prediction.ipynb`) | Quality scores 1–5 with release metrics; target MAE <0.5
| Stakeholder communication | README (quickstart with reading order) and internal meeting notes finalized for professor review | Maintain the same structure when GameProofer data arrives

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Delayed access to throw-level data | Slows MVP delivery | Continue refining proxy pipelines and finalize requirements to accelerate onboarding once data arrives | 
| Data quality gaps (missing quality labels, inconsistent metrics) | Reduces model accuracy | Apply the validated cleaning/validation steps captured in `src/preprocessing/clean_data.py` and Notebook 01 before modeling |
| Stakeholder misalignment on priorities | Duplicated effort | Use the meeting notes and documentation map to keep the review order clear and confirm Solution 1 + Solution 2 remain the focus |

---

## Next steps with real GameProofer data

1. Secure throw-level dataset (minimum 2,500 labeled throws) via NDA.
2. Re-run the cleaning and EDA workflow (Notebook 01) on the new measurements, documenting any deviations from the proxy results.
3. Retrain the clustering/classification/quality models with real features; compare metrics against the proxy baselines listed above.
4. Package findings for Latitude 64 leadership with updated timelines and resource requests.

---

*Last updated: 2025-12-15*

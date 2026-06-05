---

editor_options: 
  markdown: 
    wrap: 72
---

# SPHERE-PPL NHS Forecasting Competition — Methods Summary

**By: Emmanuel Oparaku & Michael Bakalov**

------------------------------------------------------------------------

## Repository Structure

The pipeline scripts are located in the `scripts/` folder:

| File | Description |
|------------------------------------|------------------------------------|
| `scripts/01_eda.R` | Data cleaning and feature selection |
| `scripts/02_feature_eng.R` | Feature engineering and CV fold creation |
| `scripts/03_modeling.R` | Cross-validation and model comparison |
| `scripts/04_residual_analysis.R` | Residual diagnostics and bias correction estimation |
| `scripts/05_forecast.R` | Full refit, rolling forecasts, and submission file generation |
| `scripts/aggregation_map.R` | Aggregation rules for raw sub-daily data |

Submission files are in the `submission/` folder. To reproduce results, run scripts in order (01 through 05) from the project root directory.

------------------------------------------------------------------------

## Overview

This submission forecasts daily estimated avoidable deaths in Bristol NHS (BNSSG ICS) over rolling 10-day windows using a direct multi-horizon Random Forest approach with post-hoc bias correction. The pipeline is implemented entirely in R and runs end-to-end in under one hour on a MacBook M1.

------------------------------------------------------------------------

## Feature Engineering

Raw data (16.7M rows, long format) was aggregated to daily and pivoted wide. Features fall into four groups:

**Outcome lags** — lags 3–14 of the outcome, rolling means, and momentum. Only lags ≥ 3 are used, enforcing the 3-day reporting lag constraint.

**Calendar features** — day of week, sine/cosine annual seasonality, and binary flags for structurally distinct periods (New Year surge, Christmas week, Easter, summer low).

**Clinical predictor lags** — 10 curated predictors selected for clinical distinctiveness rather than raw correlation rank. Each receives lag 1–3, 3-day rolling mean, and 7-day rolling mean features. Predictors span four clinical domains: community NCtR capacity (Sirona), acute-to-community transfer backlog (DtA), acute hospital pressure (DTA counts, ED \>12hr), and emergency demand (999 waiting calls, Severnside IUC calls).

**Derived aggregates** — four system-level composite features collapsing correlated predictors: `total_NCtR_patients`, `total_DtA_waiting`, `total_DTA_acute`, and `ed_pressure_roll3` (3-day rolling mean of ED \>12hr at BRI).

Feature selection used maximum absolute Spearman correlation against the outcome at horizons h=3:10, computed on pre-2025 data only to prevent leakage. The final feature matrix contains 133 numeric features.

------------------------------------------------------------------------

## Cross-Validation

Expanding window CV with 8 folds, 10-day validation windows, and a 10-day gap between training end and validation start (matching the forecast horizon). Median imputation was applied inside each fold using training-row medians only, applied to both training and validation slices.

------------------------------------------------------------------------

## Models

**Direct multi-horizon forecasting** — one model per horizon h=1:10, trained to predict the outcome h days ahead.

**Random Forest** (ranger) with tuned hyperparameters: `num.trees=500`, `mtry=floor(p/3)≈44`, `min.node.size=5`.

**XGBoost** was also evaluated with tuned hyperparameters (`max_depth=3`, `eta=0.1`, `subsample=0.7`, `colsample_bytree=0.7`, `min_child_weight=5`, early stopping via inner 80/20 holdout).

**CV results:**

| Model             | MSE days 1–5 | MSE days 6–10 |
|-------------------|--------------|---------------|
| Random Forest     | **0.0834**   | **0.0790**    |
| XGBoost           | 0.0927       | 0.0860        |
| Linear regression | 0.114        | 0.109         |
| Rolling mean      | 0.111        | 0.180         |
| Naive (lag-7)     | 0.153        | 0.140         |

Random Forest was selected for both horizon groups. For submission, models were refit on all 927 development rows.

------------------------------------------------------------------------

## Residual Analysis and Bias Correction

Out-of-fold residual analysis revealed two systematic patterns in RF predictions: a weekly cycle (Saturday underpredicted most, Thursday least) and a seasonal component (January and October underpredicted, consistent with winter surge periods). Both patterns motivated post-hoc bias correction.

**Day-of-week correction (primary)** — applied at the target date for each horizon. For each of the 10 horizons, the correction corresponding to the day of week that horizon lands on is added to that prediction cell. Estimated from \~100 out-of-fold observations per day, spread across all seasons.

**Monthly correction (secondary)** — applied at the origin date uniformly across all 10 horizons. Captures residual seasonal signal not explained by day of week. Applied additively on top of the DOW correction. Note: the February correction is estimated from only 10 observations and should be treated with caution.

Both corrections are derived entirely from out-of-fold residuals (no assessment data used) and are applied before submission file generation.

------------------------------------------------------------------------

## Known Limitations

- Fold 5 (January 2025 post-holiday surge) dominates CV variance and is largely irreducible — the model lacks sufficient surge training examples.
- Holiday binary flags added negligible improvement for the same reason.
- The monthly bias correction for January and February is estimated from very few CV observations (fold 5 only) and carries high uncertainty.
- Mean reversion in predictions: the model compresses extreme values toward the centre, visible in the actual vs predicted scatter at horizon 1.

------------------------------------------------------------------------

## Reproducibility

Assessment forecasts are generated by running `scripts/05_forecast.R` after the assessment dataset is released. The script reloads the same CSV, replicates the cleaning pipeline without the development period filter, builds features using the same `build_features()` function, imputes using development-period medians, and applies pre-computed bias corrections before saving `submission/pred_matrix.csv`.

# SPHERE-PPL NHS Forecasting Competition — Progress Summary

**Date:** May 2026  
**Goal:** Forecast daily estimated avoidable deaths in Bristol NHS (BNSSG ICS) over rolling 10-day windows, evaluated on MSE for days 1–5 and days 6–10 separately.  
**Deadline:** Initial submission 5 June 2026 | Assessment data released 6 June | Final forecasts due 20 June 2026

---

## What We Have Built

### Pipeline overview

The pipeline is structured across four scripts:

| Script | Status | Purpose |
|---|---|---|
| `01_eda.R` | ✅ Complete | Data cleaning, aggregation, feature selection |
| `02_feature_eng.R` | ✅ Complete | Feature matrix construction, CV fold creation |
| `03_modeling.R` | ✅ Complete | Baseline models, Random Forest, XGBoost |
| `04_forecast.R` | 🔲 Not started | Final model refit and submission generation |

### Data

- Source: 16.7M row long-format dataset, March 2023 – September 2025 (development)
- After aggregation, pivoting wide, and cleaning: **927 rows × 123 feature columns**
- Outcome: `estimated_avoidable_deaths -- BNSSG` (daily, system-level)
- Key constraint: 3-day reporting lag on outcome — only `outcome_lag3` and beyond are safe as features

### Feature engineering (`02_feature_eng.R`)

Features fall into four groups:

1. **Outcome lags** — lags 3–14, rolling means, momentum (trend direction)
2. **Calendar features** — day of week, sine/cosine seasonality encoding, holiday period flags
3. **Predictor lag features** — 8 clinically curated predictors, each with lags 1–3 and 7-day rolls
4. **Derived clinical aggregates** — `total_NCtR_patients` (Sirona community capacity), `total_DtA_waiting` (transfer backlog), `total_DTA_acute` (acute DTA pressure), `ed_pressure_roll3` (sustained ED stress)

**Key design decisions made:**
- Feature selection uses max Spearman correlation against future outcome at horizons h=3:10 (not same-day), computed on pre-2025 data only to prevent leakage
- Redundant correlated predictors (e.g. NCtR P1 vs P2, Calls Received vs Answered) collapsed into aggregates rather than included separately
- Imputation (median) is handled inside the CV fold loop — training medians applied to validation — not globally

### Cross-validation (`02_feature_eng.R` + `03_modeling.R`)

- **Expanding window CV, 8 folds**, gap of 10 days between training end and validation start
- Validation windows are 10 days each, matching the competition forecast horizon
- **Direct forecasting**: one model per horizon (h=1 through h=10), targets constructed via `lead(outcome, h)`

### Current CV results

| Model | Mean MSE (days 1–5) | Mean MSE (days 6–10) |
|---|---|---|
| **Random Forest** | **0.0878** | 0.0903 |
| **XGBoost** | 0.0922 | **0.0844** |
| Linear regression | 0.114 | 0.109 |
| Rolling mean | 0.111 | 0.180 |
| Naive (same weekday) | 0.153 | 0.140 |

Both RF and XGBoost beat all baselines on both horizon groups. RF leads on days 1–5; XGBoost leads on days 6–10.

**Notable:** Fold 5 (January 2025) is an outlier — both models produce MSE ~0.24 on days 1–5 due to a post-holiday surge the model cannot anticipate from the preceding low December. Adding explicit holiday flags made negligible difference — this appears to be an irreducible problem given only two January surges in the training history.

---

## Key Decisions Still to Make

### 1. Submission strategy — one model or two?

Since days 1–5 and days 6–10 are separate prizes, we could submit RF for the near horizon and XGBoost for the far horizon. Alternatively, pick one model for simplicity. The CV gap between them is small enough that this may not matter much.

### 2. Hyperparameter tuning

Current RF and XGBoost hyperparameters are conservative defaults. Systematic tuning (tree depth, learning rate, number of trees) could improve performance but risks overfitting at N~927. Worth attempting but needs careful CV-based evaluation rather than tuning on the full dataset.

### 3. Feature importance review

RF stores impurity-based feature importance. We haven't yet inspected which features are actually driving predictions. This is worth doing before submission — it may reveal that some features are unused or counterproductive, and could point to additional engineered features worth adding.

### 4. Final model refit (`04_forecast.R`)

The submission model should be refitted on the full development dataset (all 927 rows) rather than the final CV fold alone. The `results/model_summary.rds` file stores the final fold models and imputation medians as a starting point, but a full refit is needed.

### 5. Assessment data handling

When assessment data is released on 6 June, the pipeline needs to run `build_features()` on the assessment rows using the same `core_predictors` and then generate predictions for each horizon. The imputation medians from the full development fit must be used — not recomputed from assessment data.

---

## Repository Structure

```
data/
  modeling_data_clean.rds   — cleaned development dataset
  df_feat_raw.rds           — engineered feature matrix (927 × 123)
  df_folds.rds              — CV fold specifications
  core_predictors.rds       — curated predictor vector
results/
  baseline_summary.rds      — baseline CV results
  model_summary.rds         — RF/XGBoost CV results + final fold models
  comparison_table.rds      — combined comparison table
scripts/
  01_eda.R
  02_feature_eng.R
  03_modeling.R
  04_forecast.R             — not yet written
```

---

*Pipeline built in R using tidyverse, ranger, and xgboost. All scripts are self-contained and run sequentially.*

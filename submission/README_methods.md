------------------------------------------------------------------------

editor_options: markdown: wrap: 72 ---

# SPHERE-PPL NHS Forecasting Competition — Methods Summary

**Modellers:** Emmanuel Oparaku & Michael Bakalov

------------------------------------------------------------------------

## Repository Structure

The pipeline scripts are located in the `scripts/` folder:

| File | Description |
|------------------------------------|------------------------------------|
| `scripts/01_eda.R` | Data cleaning and feature selection |
| `scripts/02_feature_eng.R` | Feature engineering and CV fold creation |
| `scripts/03_modeling.R` | Cross-validation and model comparison |
| `scripts/04_residual_analysis.R` | Residual diagnostics and bias correction estimation |
| `scripts/05_forecast.R` | Full refit, rolling forecasts, bias correction, and submission file generation |
| `scripts/aggregation_map.R` | Aggregation rules for raw sub-daily data |

Submission files are in the `submission/` folder. To reproduce results, run scripts in order (01 through 05) from the project root directory.

------------------------------------------------------------------------

## Overview

This submission forecasts daily estimated avoidable deaths in Bristol NHS (BNSSG ICS) over rolling 10-day windows using a direct multi-horizon Random Forest approach with post-hoc bias correction. The pipeline is implemented entirely in R and runs end-to-end in under one hour on a MacBook M1.

The assessment period runs from 1 October 2025 to 17 February 2026, yielding 131 sliding 10-day forecast periods (origin dates 30 September 2025 to 7 February 2026).

------------------------------------------------------------------------

## Data Loading and Cleaning

Raw data (\~16.7M rows, long format) is loaded with `data.table::fread()` for performance. The development and assessment datasets are loaded separately and combined:

``` r
dev_raw    <- data.table::fread("data/turingAI_forecasting_challenge_dataset.csv")
assess_raw <- data.table::fread("data/turingAI_forecasting_challenge_validation_dataset.csv")

dev_raw[, date := as.POSIXct(dt, origin = "1970-01-01", tz = "UTC") |> as.Date()]
assess_raw[, date := as.POSIXct(dt, origin = "1970-01-01", tz = "UTC") |> as.Date()]

dev_raw  <- dev_raw[date < as.Date("2025-10-01")]
raw_data <- rbind(dev_raw, assess_raw)
```

Sub-daily records are aggregated to daily resolution using a metric-specific aggregation map (mean, max, sum, last, or as-is). Metrics flagged as YESTERDAY in their name are shifted back one day to align with the observation date. Negative values (-9999 dummy codes) are replaced with NA. Columns with more than 50% missing values over the development period are dropped, computed on development rows only to avoid contaminating the filter with assessment data patterns.

**Known data issue:** Origin date 2025-10-03 is absent from the assessment dataset. Period 4 (predicting October 4–13) therefore produces NA predictions. These are excluded from MSE calculations via `na.rm = TRUE`.

------------------------------------------------------------------------

## Feature Engineering

Feature engineering is implemented as a deterministic, leakage-free function `build_features()` in `02_feature_eng.R`. Imputation is deliberately excluded from this step and applied inside each CV fold using training-period medians only.

**Outcome lags**

Outcome lag features capture recent history of the target variable at the time of prediction. Lags at days 3 through 14 are included, alongside 3-day and 7-day rolling means and a momentum feature computed as the difference between the lag-3 and lag-6 values. Only lags of 3 days or more are used, enforcing the 3-day reporting lag constraint — lag-3 represents the most recent observable outcome value at prediction time:

``` r
df <- df %>%
  mutate(
    outcome_lag3  = lag(.data[[outcome_col]], 3),
    outcome_lag7  = lag(.data[[outcome_col]], 7),
    outcome_lag14 = lag(.data[[outcome_col]], 14),
    outcome_roll3 = rowMeans(cbind(outcome_lag3, outcome_lag4, outcome_lag5)),
    outcome_momentum = outcome_lag3 - outcome_lag6
  )
```

**Calendar features**

Calendar features capture temporal structure at multiple scales. Day of week is included as both a numeric variable and an ordered factor, alongside sine and cosine encodings of day-of-year which provide a smooth, continuous representation of annual seasonality. ISO week, month, and a linear time index are also included. Four binary flags mark structurally distinct periods where NHS demand patterns differ systematically from baseline: the New Year surge window (1–14 January), Christmas week (24–31 December), the Easter period (days 85–105), and the summer low (July–August).

**Clinical predictor lags**

Ten clinically curated predictors are selected across four domains: community NCtR capacity (Sirona), acute-to-community transfer backlog (DtA), acute hospital pressure (DTA counts and ED attendance over 12 hours), and emergency demand (999 waiting calls and Severnside IUC calls). Predictors are chosen for clinical distinctiveness rather than raw correlation rank, avoiding redundant correlated pairs. Each predictor receives individual lags at 1, 2, and 3 days, a 3-day rolling mean, and a 7-day rolling mean, giving a short-term level signal and a weekly-level signal for each clinical domain.

**Feature selection** used maximum absolute Spearman correlation against the outcome at horizons h=3:10, computed on pre-2025 data only to prevent leakage:

``` r
cors_across_horizons <- map_dbl(3:10, function(h) {
  outcome_future <- lead(outcome_vec, h)
  cor(feature_vec, outcome_future,
      use    = "pairwise.complete.obs",
      method = "spearman")
})
max(abs(cors_across_horizons), na.rm = TRUE)
```

**Derived aggregates**

Four system-level composite features collapse clinically related but correlated predictors into single signals: `total_NCtR_patients` (P1 + P2 NCtR patients at NBT), `total_DtA_waiting` (P2 + P3 DtA patients), `total_DTA_acute` (delayed transfers at BRI, NBT, and WGH combined), and `ed_pressure_roll3` (a 3-day rolling mean of ED attendance over 12 hours at BRI). Binary missingness indicators are created for all lagged predictor and derived aggregate columns, treating a system's failure to report as an operationally informative signal in its own right. The final feature matrix contains 133 numeric features.

------------------------------------------------------------------------

## Cross-Validation

Temporal cross-validation uses an expanding window with 8 folds, fixed 10-day validation windows, and a 10-day gap between training end and validation start to match the forecast horizon. This structure ensures no information from the validation period contaminates training. Median imputation is computed exclusively from training rows within each fold and applied to both the training and validation slices. This mirrors the procedure used at submission time, where development-period medians are applied to impute assessment features, ensuring consistency between the CV evaluation and the final forecast:

``` r
medians <- sapply(df_train[feat_cols], median, na.rm = TRUE)
for (col in feat_cols) {
  df_train[[col]] <- ifelse(is.na(df_train[[col]]), medians[col], df_train[[col]])
  df_val[[col]]   <- ifelse(is.na(df_val[[col]]),   medians[col], df_val[[col]])
}
```

------------------------------------------------------------------------

## Models

Three baseline models provide reference points for evaluation. The naive baseline predicts the outcome from the same weekday of the previous week (`outcome_lag7`), capturing the strong weekly periodicity in the data. The rolling mean baseline predicts using a 3-day rolling mean of the outcome (`outcome_roll3`), representing a simple recent-average forecast. The linear regression baseline fits one model per horizon using calendar features and outcome lags, providing a parametric reference.

The main models are Random Forest and XGBoost, both using direct multi-horizon forecasting — one model is trained per horizon from 1 to 10 days ahead. This approach is preferred over recursive forecasting because it avoids compounding prediction errors across horizons; each model is trained directly against its target lead time. Random Forest was fitted using the ranger package with tuned hyperparameters of `num.trees=500`, `mtry=floor(p/3)≈44`, and `min.node.size=5`. XGBoost was fitted with `max_depth=3`, `eta=0.1`, `subsample=0.7`, `colsample_bytree=0.7`, and `min_child_weight=5`, with early stopping determined via an inner 80/20 holdout split on the training data.

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

To diagnose systematic model errors, out-of-fold residuals were reconstructed by refitting models on each fold's training slice and collecting validation predictions across all 8 folds. Diagnostic plots revealed two systematic patterns: a weekly cycle (Saturday predictions consistently low) and a seasonal bias (January persistently underpredicted, coinciding with the post-holiday surge).

Two additive post-hoc corrections were computed from out-of-fold residuals:

**Day-of-week correction (primary)** — applied at the target date (day) for each horizon. Each of the 10 horizons lands on a specific day of week; the correction for that day is applied to that prediction cell only. Estimated from \~100 out-of-fold observations per day, spread across all seasons:

``` r
dow_corrections <- resid_df %>%
  mutate(dow_num = wday(date, week_start = 1)) %>%
  group_by(dow_num) %>%
  summarise(correction = -mean(rf_resid, na.rm = TRUE), .groups = "drop")
```

**Monthly correction (secondary)** — applied at the origin date uniformly across all 10 horizons. Captures residual seasonal signal not explained by day of week. The February correction is estimated from only 10 observations and carries high uncertainty.

**Assessment period finding:** On the actual assessment data, the bias correction worsened MSE relative to uncorrected predictions:

| Version                        | MSE days 1–5 | MSE days 6–10 |
|--------------------------------|--------------|---------------|
| Uncorrected (primary)          | **0.0980**   | **0.110**     |
| Bias-corrected (supplementary) | 0.103        | 0.127         |

The correction overcorrects on the winter-dominated assessment window because it was estimated from predominantly non-winter development data. We therefore submit uncorrected predictions as primary (`pred_matrix.csv`) and bias-corrected predictions as supplementary (`pred_matrix_corrected.csv`), with both files saved by `05_forecast.R`.

------------------------------------------------------------------------

## Known Limitations

- Fold 5 (January 2025 post-holiday surge) dominates CV variance and is largely irreducible — the model lacks sufficient surge training examples.
- Holiday binary flags added negligible improvement for the same reason.
- The monthly bias correction for January and February is estimated from very few CV observations (fold 5 only) and carries high uncertainty.
- Mean reversion in predictions: Random Forest averages across many trees, which dampens extreme predicted values toward the centre of the distribution. The model therefore tends to underpredict on high-mortality days and overpredict slightly on low-mortality days, which is visible in the actual vs predicted scatter at horizon 1.
- Origin date 2025-10-03 is missing from the assessment dataset; period 4 predictions are NA and excluded from MSE calculations.

------------------------------------------------------------------------

## Reproducibility

The assessment forecast pipeline (`05_forecast.R`) loads both the development and validation datasets, combines them, replicates the cleaning pipeline without the development period filter, builds features using `build_features()`, imputes using development-period medians, applies pre-computed bias corrections, and computes MSE against observed outcomes before saving submission files:

``` r
# Primary submission (uncorrected — lower MSE on assessment data)
write.csv(pred_out,           "submission/pred_matrix.csv",           row.names = FALSE)

# Supplementary (bias-corrected)
write.csv(pred_out_corrected, "submission/pred_matrix_corrected.csv", row.names = FALSE)
```

The MSE comparison between corrected and uncorrected predictions runs automatically before Section 6 (`validate and save`), so the rationale for the primary submission choice is documented in the console output on every run.

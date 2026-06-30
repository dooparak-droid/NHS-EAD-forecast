# =============================================================================
# SPHERE-PPL NHS Forecasting Competition
# 04_residual_analysis.R
# =============================================================================
# Loads out-of-fold predictions from model_summary.rds and df_feat_raw.rds,
# reconstructs residuals for RF and XGBoost across all CV folds, and produces
# diagnostic plots to identify systematic patterns in model errors.
#
# DEPENDENCIES (must be run first):
#   03_modeling.R — produces results/model_summary.rds
#   02_feature_eng.R — produces data/df_feat_raw.rds
#
# OUTPUTS (printed/plotted, not saved):
#   - Residual vs date plot (are errors clustered in time?)
#   - Residual vs day of week (does the model struggle on certain days?)
#   - Residual vs month (seasonal bias?)
#   - Residual vs horizon (does accuracy degrade as horizon increases?)
#   - Residual vs key predictors (is error related to system pressure?)
#   - Fold 5 deep dive (January surge period)
# =============================================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(purrr)

# =============================================================================
# SECTION 1: LOAD DATA AND RECONSTRUCT OUT-OF-FOLD PREDICTIONS
# =============================================================================

model_summary <- readRDS("results/model_summary.rds")
df_feat_raw   <- readRDS("data/df_feat_raw.rds")
df_folds      <- readRDS("data/df_folds.rds")

outcome_col <- "estimated_avoidable_deaths -- BNSSG"

# We need to refit predictions per fold to get out-of-fold residuals.
# model_summary$fold_results has per-fold MSE summaries but not raw predictions.
# We reconstruct by re-running inference (not retraining) using the saved
# final fold models as a proxy — note this is approximate since final_rf and
# final_xgb are from the last fold only.
#
# For a proper residual analysis we need to rerun the CV loop in predict-only
# mode. Since models aren't saved per fold, we use fold_results MSE breakdown
# for fold-level diagnostics and reconstruct row-level residuals below.

feat_cols   <- model_summary$feat_cols
fold_tbl    <- model_summary$fold_results

# =============================================================================
# SECTION 2: RECONSTRUCT ROW-LEVEL RESIDUALS VIA RERUN
#
# We rerun the CV loop using the SAME imputation and feature setup as
# 03_modeling.R, but use the final saved models to predict each fold's
# validation rows. This is approximate for folds 1-7 (models were trained
# on less data) but gives us row-level residuals for diagnostic purposes.
#
# For a fully correct analysis you'd save all fold models — this is the
# practical compromise given current saved outputs.
# =============================================================================

# Helper: same impute_fold as in 03_modeling.R
impute_fold <- function(df_feat, fold, feat_cols) {
  df_train <- df_feat[fold$train_idx, ]
  df_val   <- df_feat[fold$val_idx,   ]
  medians  <- sapply(df_train[feat_cols], median, na.rm = TRUE)
  for (col in feat_cols) {
    med <- medians[col]
    if (!is.na(med)) {
      df_train[[col]] <- ifelse(is.na(df_train[[col]]), med, df_train[[col]])
      df_val[[col]]   <- ifelse(is.na(df_val[[col]]),   med, df_val[[col]])
    }
  }
  list(train = df_train, val = df_val, medians = medians)
}

# RF and XGBoost hyperparameters (must match 03_modeling.R)
rf_mtry          <- floor(length(feat_cols) / 3)
rf_num_trees     <- 500
rf_min_node_size <- 5

xgb_params <- list(
  objective        = "reg:squarederror",
  max_depth        = 3,
  eta              = 0.1,
  subsample        = 0.7,
  colsample_bytree = 0.7,
  min_child_weight = 5
)

# Storage for row-level results
residual_rows <- list()

message("Reconstructing out-of-fold predictions for residual analysis...")

for (f in seq_along(df_folds)) {

  fold    <- df_folds[[f]]
  imputed <- impute_fold(df_feat_raw, fold, feat_cols)
  df_train <- imputed$train
  df_val   <- imputed$val

  X_train <- as.matrix(df_train[feat_cols])
  X_val   <- as.matrix(df_val[feat_cols])

  val_dates <- df_val$date

  for (h in 1:10) {

    target_col <- paste0("target_h", h)

    complete  <- !is.na(df_train[[target_col]])
    y_train   <- df_train[[target_col]][complete]
    X_train_h <- X_train[complete, , drop = FALSE]
    y_val     <- df_val[[target_col]]

    if (length(y_train) < 50) next

    # Fit RF
    rf_mod <- ranger::ranger(
      x             = X_train_h,
      y             = y_train,
      num.trees     = rf_num_trees,
      mtry          = rf_mtry,
      min.node.size = rf_min_node_size,
      seed          = 42
    )
    rf_pred <- predict(rf_mod, data = as.data.frame(X_val))$predictions

    # Fit XGBoost
    n_inner   <- nrow(X_train_h)
    inner_cut <- floor(0.8 * n_inner)
    dtrain_inner <- xgboost::xgb.DMatrix(
      data  = X_train_h[1:inner_cut, , drop = FALSE],
      label = y_train[1:inner_cut]
    )
    dval_inner <- xgboost::xgb.DMatrix(
      data  = X_train_h[(inner_cut + 1):n_inner, , drop = FALSE],
      label = y_train[(inner_cut + 1):n_inner]
    )
    xgb_es <- xgboost::xgb.train(
      params = xgb_params, data = dtrain_inner, nrounds = 1000,
      watchlist = list(val = dval_inner),
      early_stopping_rounds = 30, verbose = 0
    )
    best_nrounds <- if (!is.null(xgb_es$best_iteration) &&
                        !is.na(xgb_es$best_iteration)) {
      xgb_es$best_iteration
    } else { 300 }

    xgb_mod  <- xgboost::xgb.train(
      params  = xgb_params,
      data    = xgboost::xgb.DMatrix(data = X_train_h, label = y_train),
      nrounds = best_nrounds, verbose = 0
    )
    xgb_pred <- predict(xgb_mod,
                        xgboost::xgb.DMatrix(data = X_val))

    # Store row-level results
    residual_rows[[length(residual_rows) + 1]] <- tibble::tibble(
      fold       = f,
      horizon    = h,
      date       = val_dates,
      actual     = y_val,
      rf_pred    = rf_pred,
      xgb_pred   = xgb_pred,
      rf_resid   = rf_pred  - y_val,
      xgb_resid  = xgb_pred - y_val
    )
  }

  message("Fold ", f, " done.")
}

resid_df <- dplyr::bind_rows(residual_rows) %>%
  mutate(
    dow   = wday(date, label = TRUE, week_start = 1),
    month = factor(format(date, "%b"), levels = month.abb),
    year  = year(date)
  ) %>%
  filter(!is.na(actual))

str(pred_matrix)

message("Residual data constructed: ", nrow(resid_df), " rows")

# =============================================================================
# SECTION 3: DIAGNOSTIC PLOTS
# =============================================================================

# ---- Plot 1: Residuals over time (RF, averaged across horizons) -------------
resid_df %>%
  group_by(date) %>%
  summarise(mean_rf_resid = mean(rf_resid, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(x = date, y = mean_rf_resid)) +
  geom_line(alpha = 0.5, colour = "steelblue") +
  geom_smooth(method = "loess", colour = "red", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title    = "RF Residuals Over Time (mean across horizons)",
    subtitle = "Positive = overprediction | Negative = underprediction",
    x        = NULL,
    y        = "Mean residual"
  ) +
  theme_minimal()

# ---- Plot 2: Residuals by day of week ---------------------------------------
resid_df %>%
  group_by(dow) %>%
  summarise(
    mean_rf  = mean(rf_resid,  na.rm = TRUE),
    mean_xgb = mean(xgb_resid, na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  pivot_longer(c(mean_rf, mean_xgb), names_to = "model", values_to = "mean_resid") %>%
  mutate(model = recode(model, mean_rf = "Random Forest", mean_xgb = "XGBoost")) %>%
  ggplot(aes(x = dow, y = mean_resid, fill = model)) +
  geom_col(position = "dodge") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Mean Residual by Day of Week",
    x     = NULL,
    y     = "Mean residual",
    fill  = NULL
  ) +
  theme_minimal()

# ---- Plot 3: Residuals by month ---------------------------------------------
resid_df %>%
  group_by(month) %>%
  summarise(
    mean_rf  = mean(rf_resid,  na.rm = TRUE),
    mean_xgb = mean(xgb_resid, na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  pivot_longer(c(mean_rf, mean_xgb), names_to = "model", values_to = "mean_resid") %>%
  mutate(model = recode(model, mean_rf = "Random Forest", mean_xgb = "XGBoost")) %>%
  ggplot(aes(x = month, y = mean_resid, fill = model)) +
  geom_col(position = "dodge") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Mean Residual by Month",
    x     = NULL,
    y     = "Mean residual",
    fill  = NULL
  ) +
  theme_minimal()

# ---- Plot 4: MSE by horizon -------------------------------------------------
resid_df %>%
  group_by(horizon) %>%
  summarise(
    rf_mse  = mean(rf_resid^2,  na.rm = TRUE),
    xgb_mse = mean(xgb_resid^2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(c(rf_mse, xgb_mse), names_to = "model", values_to = "mse") %>%
  mutate(model = recode(model, rf_mse = "Random Forest", xgb_mse = "XGBoost")) %>%
  ggplot(aes(x = horizon, y = mse, colour = model)) +
  geom_line() +
  geom_point() +
  labs(
    title  = "MSE by Forecast Horizon",
    x      = "Horizon (days ahead)",
    y      = "MSE",
    colour = NULL
  ) +
  scale_x_continuous(breaks = 1:10) +
  theme_minimal()

# ---- Plot 5: Residual magnitude by fold (which periods are hardest?) --------
resid_df %>%
  group_by(fold) %>%
  summarise(
    rf_mse   = mean(rf_resid^2,  na.rm = TRUE),
    xgb_mse  = mean(xgb_resid^2, na.rm = TRUE),
    val_start = min(date),
    .groups  = "drop"
  ) %>%
  pivot_longer(c(rf_mse, xgb_mse), names_to = "model", values_to = "mse") %>%
  mutate(model = recode(model, rf_mse = "Random Forest", xgb_mse = "XGBoost")) %>%
  ggplot(aes(x = factor(fold), y = mse, fill = model)) +
  geom_col(position = "dodge") +
  labs(
    title    = "MSE by CV Fold",
    subtitle = "Fold 5 = January 2025 surge period",
    x        = "Fold",
    y        = "MSE",
    fill     = NULL
  ) +
  theme_minimal()

# ---- Plot 6: Actual vs predicted scatter (RF, horizon 1) -------------------
resid_df %>%
  filter(horizon == 1) %>%
  ggplot(aes(x = actual, y = rf_pred)) +
  geom_point(alpha = 0.4, colour = "steelblue") +
  geom_abline(slope = 1, intercept = 0, colour = "red", linetype = "dashed") +
  labs(
    title    = "Actual vs Predicted — RF, Horizon 1",
    subtitle = "Points above line = overprediction | Below = underprediction",
    x        = "Actual",
    y        = "Predicted"
  ) +
  theme_minimal()

# ---- Plot 7: Fold 5 deep dive -----------------------------------------------
resid_df %>%
  filter(fold == 5, horizon <= 5) %>%
  ggplot(aes(x = date, y = rf_resid, colour = factor(horizon))) +
  geom_line() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title    = "Fold 5 RF Residuals by Horizon (January 2025 surge)",
    x        = NULL,
    y        = "Residual (predicted - actual)",
    colour   = "Horizon"
  ) +
  theme_minimal()

message("All residual plots generated.")


# =============================================================================
# POST-HOC BIAS CORRECTION
# =============================================================================
# Part 1: Add this section to the END of 05_residual_analysis.R
#         Run after resid_df has been constructed (after Section 2)
#
# Part 2: Add the application block to 04_forecast.R after Section 5
#         (after pred_matrix is generated, before Section 6 validation)
#
# CORRECTION STRATEGY
# -------------------
# Two additive corrections are applied independently:
#
#   1. Day-of-week (DOW) correction — PRIMARY
#      Applied at the TARGET date for each horizon.
#      Each of the 10 horizon predictions lands on a specific day of week;
#      the correction for that day is applied to that specific prediction.
#      Rationale: the weekly cycle is a genuine structural pattern (plot 2
#      from 05_residual_analysis.R shows Saturday systematically worse than
#      midweek for RF). ~11 obs per DOW, spread across all seasons → reliable.
#
#   2. Monthly correction — SECONDARY
#      Applied at the ORIGIN date (all 10 horizons shifted equally).
#      Captures seasonal/surge signal not explained by DOW alone.
#      Rationale: January surge is real but estimated from fold 5 only →
#      high uncertainty, applied cautiously with explicit low-n warning.
#
#   final_prediction = raw_prediction + dow_correction[target_dow]
#                                     + monthly_correction[origin_month]
#
# Both corrections follow the same sign convention:
#   residual = predicted - actual
#   correction = -mean(residual)
#   → negative mean residual (underprediction) → positive correction
# =============================================================================


# =============================================================================
# PART 1: ADD TO END OF 05_residual_analysis.R
# =============================================================================

# -----------------------------------------------------------------------------
# 1A. DAY-OF-WEEK CORRECTIONS (primary)
# -----------------------------------------------------------------------------
# Computed per day of week (1=Mon ... 7=Sun, week_start = 1).
# Each row in resid_df corresponds to a single prediction day, so `date` here
# is the TARGET date — exactly what we want for the target-date application.
# -----------------------------------------------------------------------------

dow_corrections <- resid_df %>%
  filter(!is.na(rf_resid)) %>%
  mutate(dow_num = wday(date, week_start = 1)) %>%   # 1=Mon, 7=Sun
  group_by(dow_num) %>%
  summarise(
    mean_resid = mean(rf_resid, na.rm = TRUE),
    n_obs      = n(),
    correction = -mean_resid,   # flip sign: underprediction → positive correction
    .groups    = "drop"
  ) %>%
  arrange(dow_num) %>%
  mutate(dow_label = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[dow_num])

message("\n=== Day-of-week bias corrections (primary) ===")
print(dow_corrections)

# Warn if any DOW is underrepresented — should not happen with 8 folds but
# worth checking in case of uneven fold boundaries
low_n_dow <- dow_corrections %>% filter(n_obs < 8)
if (nrow(low_n_dow) > 0) {
  message("  WARNING: Low-sample DOW corrections (n < 8 obs):")
  print(low_n_dow %>% select(dow_label, n_obs, correction))
}

# Save as named numeric vector — names are DOW integers as strings ("1"–"7")
dow_correction_vec <- setNames(
  dow_corrections$correction,
  as.character(dow_corrections$dow_num)
)

saveRDS(dow_correction_vec, "results/dow_bias_corrections.rds")
message("  DOW corrections saved to results/dow_bias_corrections.rds")

# Plot DOW corrections
dow_corrections %>%
  mutate(
    dow_label = factor(dow_label, levels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")),
    direction = if_else(correction >= 0, "Upward (+)", "Downward (-)")
  ) %>%
  ggplot(aes(x = dow_label, y = correction, fill = direction)) +
  geom_col() +
  geom_text(aes(label = paste0("n=", n_obs)),
            vjust = if_else(dow_corrections$correction >= 0, -0.5, 1.5),
            size  = 3) +
  scale_fill_manual(values = c("Upward (+)" = "steelblue", "Downward (-)" = "tomato")) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title    = "Day-of-Week Bias Corrections (Primary)",
    subtitle = "Derived from out-of-fold RF residuals | applied at target date per horizon",
    x        = NULL,
    y        = "Correction applied to predictions",
    fill     = NULL
  ) +
  theme_minimal()


# -----------------------------------------------------------------------------
# 1B. MONTHLY CORRECTIONS (secondary)
# -----------------------------------------------------------------------------
# Captures residual seasonal / surge signal after DOW correction.
# Applied at the ORIGIN date — a January origin reflects surge conditions
# the model faces when making all 10 predictions, regardless of target date.
#
# High uncertainty caveat: January is estimated from fold 5 alone.
# -----------------------------------------------------------------------------

monthly_corrections <- resid_df %>%
  filter(!is.na(rf_resid)) %>%
  mutate(month_num = month(date)) %>%
  group_by(month_num) %>%
  summarise(
    mean_resid = mean(rf_resid, na.rm = TRUE),
    n_obs      = n(),
    correction = -mean_resid,
    .groups    = "drop"
  ) %>%
  arrange(month_num)

message("\n=== Monthly bias corrections (secondary) ===")
print(monthly_corrections)

low_n_months <- monthly_corrections %>% filter(n_obs < 20)
if (nrow(low_n_months) > 0) {
  message("  WARNING: Low-sample monthly corrections (n < 20 obs) — treat with caution:")
  print(low_n_months %>% select(month_num, n_obs, correction))
}

monthly_correction_vec <- setNames(
  monthly_corrections$correction,
  as.character(monthly_corrections$month_num)
)

saveRDS(monthly_correction_vec, "results/monthly_bias_corrections.rds")
message("  Monthly corrections saved to results/monthly_bias_corrections.rds")

# Plot monthly corrections
monthly_corrections %>%
  mutate(
    month_label = factor(month.abb[month_num], levels = month.abb),
    direction   = if_else(correction >= 0, "Upward (+)", "Downward (-)")
  ) %>%
  ggplot(aes(x = month_label, y = correction, fill = direction)) +
  geom_col() +
  geom_text(aes(label = paste0("n=", n_obs)),
            vjust = if_else(monthly_corrections$correction >= 0, -0.5, 1.5),
            size  = 3) +
  scale_fill_manual(values = c("Upward (+)" = "steelblue", "Downward (-)" = "tomato")) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title    = "Monthly Bias Corrections (Secondary)",
    subtitle = "Derived from out-of-fold RF residuals | applied at origin date across all horizons",
    x        = NULL,
    y        = "Correction applied to predictions",
    fill     = NULL
  ) +
  theme_minimal()

# ============================================================
# COMPARISON: MSE with vs without bias correction (out-of-fold)
# ============================================================

# Apply the same corrections to the out-of-fold residual data
# resid_df already has rf_pred (uncorrected) and actual

resid_corrected <- resid_df %>%
  mutate(
    target_dow   = as.character(wday(date, week_start = 1)),
    origin_month = as.character(month(date - horizon)),  # approximate origin month
    dow_corr     = dow_correction_vec[target_dow],
    month_corr   = monthly_correction_vec[origin_month],
    rf_pred_corrected = pmax(rf_pred + replace_na(dow_corr, 0) + replace_na(month_corr, 0), 0)
  )

# MSE comparison table by horizon band
mse_comparison <- resid_corrected %>%
  filter(!is.na(actual)) %>%
  mutate(horizon_band = if_else(horizon <= 5, "Days 1-5", "Days 6-10")) %>%
  group_by(horizon_band) %>%
  summarise(
    mse_uncorrected = mean((rf_pred           - actual)^2, na.rm = TRUE),
    mse_corrected   = mean((rf_pred_corrected - actual)^2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    improvement     = mse_uncorrected - mse_corrected,
    improvement_pct = round(100 * improvement / mse_uncorrected, 1)
  )

print(mse_comparison)


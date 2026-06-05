# =============================================================================
# SPHERE-PPL NHS Forecasting Competition
# 03_modeling.R
# =============================================================================
# Loads the feature matrix and CV folds saved by 02_feature_eng.R, runs
# baseline models and main models (Random Forest, XGBoost) through the CV
# framework, and produces a comparison table of MSE results.
#
# SECTIONS:
#   1. Setup and imputation helper
#   2. Feature column definition
#   3. Baseline models (naive, rolling mean, linear regression)
#   4. Main models (Random Forest, XGBoost)
#   5. Results comparison
#
# CHANGES FROM ORIGINAL:
#   - RF: mtry set to floor(p/3) based on hyperparameter (HPT) tuning results
#   - XGBoost: max_depth=3, eta=0.1, subsample=0.7, colsample_bytree=0.7
#     based on HPT tuning results; early stopping via inner holdout
#   - Added all-NA column guard in Section 2
#   - Fixed duplicate RF prediction line
#
# DEPENDENCIES (must be run first):
#   02_feature_eng.R — produces data/df_feat_raw.rds, data/df_folds.rds,
#                      data/core_predictors.rds
#
# OUTPUTS:
#   results/baseline_summary.rds  — baseline MSE per fold and summary
#   results/model_summary.rds     — RF and XGBoost MSE per fold and summary
#   results/comparison_table.rds  — combined comparison of all approaches
# =============================================================================

library(dplyr)
library(tidyr)
library(purrr)
library(ranger)
library(xgboost)

# Create results directory if it doesn't exist
if (!dir.exists("results")) dir.create("results")

# =============================================================================
# SECTION 1: SETUP AND IMPUTATION HELPER
# =============================================================================

# Load outputs from 02_feature_eng.R
df_feat_raw     <- readRDS("data/df_feat_raw.rds")
df_folds        <- readRDS("data/df_folds.rds")
core_predictors <- readRDS("data/core_predictors.rds")

outcome_col <- "estimated_avoidable_deaths -- BNSSG"

message("Loaded data: ", nrow(df_feat_raw), " rows, ", ncol(df_feat_raw), " columns")
message("Number of CV folds: ", length(df_folds))

# -----------------------------------------------------------------------------
# impute_fold()
#
# Takes the full feature matrix and a single fold specification.
# Returns training and validation slices with median imputation applied.
# Medians are computed from training rows only and applied to both slices —
# this is the correct approach to prevent leakage across the fold boundary.
#
# Factors (dow_factor, month_fct) and non-numeric columns are excluded from
# imputation and passed through unchanged.
#
# @param df_feat  Full feature matrix (df_feat_raw).
# @param fold     Single fold object from df_folds.
# @param feat_cols Character vector of numeric feature column names to impute.
#
# @return Named list:
#   $train       — imputed training matrix (data frame)
#   $val         — imputed validation matrix (data frame)
#   $medians     — named numeric vector of training medians (for submission use)
# -----------------------------------------------------------------------------

impute_fold <- function(df_feat, fold, feat_cols) {

  df_train <- df_feat[fold$train_idx, ]
  df_val   <- df_feat[fold$val_idx,   ]

  # Compute medians from training rows only
  medians <- sapply(df_train[feat_cols], median, na.rm = TRUE)

  # Apply to training slice
  for (col in feat_cols) {
    med <- medians[col]
    if (!is.na(med)) {
      df_train[[col]] <- ifelse(is.na(df_train[[col]]), med, df_train[[col]])
      df_val[[col]]   <- ifelse(is.na(df_val[[col]]),   med, df_val[[col]])
    }
  }

  list(
    train   = df_train,
    val     = df_val,
    medians = medians
  )
}

# =============================================================================
# SECTION 2: FEATURE COLUMN DEFINITION
# =============================================================================
# Define once, reuse throughout. Feature columns are all numeric columns
# excluding date, raw outcome, target columns, and factor columns.
# These are the columns passed as inputs to every model.
# =============================================================================

exclude_cols <- c(
  "date",
  outcome_col,
  paste0("target_h", 1:10),
  "dow_factor",
  "month_fct"
)

feat_cols <- names(df_feat_raw)[
  sapply(df_feat_raw, is.numeric) & !names(df_feat_raw) %in% exclude_cols
]

# Guard: drop any columns that are entirely NA across all rows
all_na_cols <- feat_cols[
  sapply(df_feat_raw[feat_cols], function(x) all(is.na(x)))
]

if (length(all_na_cols) > 0) {
  message("Dropping all-NA columns: ",
          paste(all_na_cols, collapse = ", "))
  feat_cols <- base::setdiff(feat_cols, all_na_cols)
}

message("Feature columns available to models: ", length(feat_cols))

# =============================================================================
# SECTION 3: BASELINE MODELS
# =============================================================================
# Three baselines evaluated through all CV folds.
# No hyperparameter tuning — these are fixed reference points.
#
# Naive:        predict outcome_lag7 for all horizons (same weekday last week)
# Rolling mean: predict outcome_roll3 for all horizons (recent average)
# LM:           one lm() per horizon on calendar + outcome lag features
# =============================================================================

# LM uses a small fixed feature set — interpretable, no predictor lags
lm_features <- c("outcome_lag3", "outcome_lag7", "dow", "yday_sin", "yday_cos")

# Storage
baseline_fold_results <- vector("list", length(df_folds))

message("\n=== Running baseline models ===")

for (f in seq_along(df_folds)) {

  fold <- df_folds[[f]]
  message("\n-- Baseline fold ", fold$fold_number, " --")

  # Impute this fold
  imputed  <- impute_fold(df_feat_raw, fold, feat_cols)
  df_train <- imputed$train
  df_val   <- imputed$val
  n_val    <- nrow(df_val)

  # Actuals matrix: n_val x 10
  actuals <- sapply(paste0("target_h", 1:10), function(tc) df_val[[tc]])

  # ---- Baseline 1: Naive same-weekday ----------------------------------------
  naive_preds <- matrix(
    rep(df_val$outcome_lag7, times = 10),
    nrow  = n_val,
    ncol  = 10
  )

  # ---- Baseline 2: Rolling mean ----------------------------------------------
  roll_preds <- matrix(
    rep(df_val$outcome_roll3, times = 10),
    nrow  = n_val,
    ncol  = 10
  )

  # ---- Baseline 3: Linear regression, one model per horizon -----------------
  lm_preds <- matrix(NA_real_, nrow = n_val, ncol = 10)

  lm_feats_available <- base::intersect(lm_features, names(df_train))

  for (h in 1:10) {

    target_col <- paste0("target_h", h)

    # Complete cases only for training
    df_train_lm <- df_train %>%
      select(all_of(c(lm_feats_available, target_col))) %>%
      tidyr::drop_na()

    if (nrow(df_train_lm) < 30) {
      warning("Fold ", f, " h", h, ": fewer than 30 training rows for lm — skipping")
      next
    }

    lm_formula <- as.formula(
      paste(target_col, "~", paste(lm_feats_available, collapse = " + "))
    )

    lm_mod <- tryCatch(
      lm(lm_formula, data = df_train_lm),
      error = function(e) { warning("lm failed: ", e$message); NULL }
    )

    if (!is.null(lm_mod)) {
      lm_preds[, h] <- tryCatch(
        predict(lm_mod, newdata = df_val),
        error = function(e) rep(NA_real_, n_val)
      )
    }
  }

  # ---- Compute MSE per baseline ----------------------------------------------
  mse_block <- function(preds, actuals, h_range) {
    mean((as.vector(preds[, h_range]) - as.vector(actuals[, h_range]))^2,
         na.rm = TRUE)
  }

  message("fold class before tibble: ", class(fold), " | fold_number: ", fold$fold_number)

  baseline_fold_results[[f]] <- tibble::tibble(
    fold_number       = fold$fold_number,
    train_end         = as.character(fold$train_end_date),
    val_start         = as.character(fold$val_start_date),
    naive_mse_1_5     = mse_block(naive_preds, actuals, 1:5),
    naive_mse_6_10    = mse_block(naive_preds, actuals, 6:10),
    rolling_mse_1_5   = mse_block(roll_preds,  actuals, 1:5),
    rolling_mse_6_10  = mse_block(roll_preds,  actuals, 6:10),
    lm_mse_1_5        = mse_block(lm_preds,    actuals, 1:5),
    lm_mse_6_10       = mse_block(lm_preds,    actuals, 6:10)
  )

  message(
    "  Naive   — MSE(1-5): ", round(baseline_fold_results[[f]]$naive_mse_1_5,   4),
    " | MSE(6-10): ",          round(baseline_fold_results[[f]]$naive_mse_6_10,  4), "\n",
    "  Rolling — MSE(1-5): ", round(baseline_fold_results[[f]]$rolling_mse_1_5, 4),
    " | MSE(6-10): ",          round(baseline_fold_results[[f]]$rolling_mse_6_10,4), "\n",
    "  LM      — MSE(1-5): ", round(baseline_fold_results[[f]]$lm_mse_1_5,      4),
    " | MSE(6-10): ",          round(baseline_fold_results[[f]]$lm_mse_6_10,    4)
  )
}

# Summarise baselines across folds
baseline_fold_tbl <- dplyr::bind_rows(baseline_fold_results)

baseline_summary <- tibble::tibble(
  model     = c("naive", "rolling_mean", "lm"),
  mse_1_5   = c(
    mean(baseline_fold_tbl$naive_mse_1_5,   na.rm = TRUE),
    mean(baseline_fold_tbl$rolling_mse_1_5, na.rm = TRUE),
    mean(baseline_fold_tbl$lm_mse_1_5,      na.rm = TRUE)
  ),
  sd_1_5    = c(
    sd(baseline_fold_tbl$naive_mse_1_5,   na.rm = TRUE),
    sd(baseline_fold_tbl$rolling_mse_1_5, na.rm = TRUE),
    sd(baseline_fold_tbl$lm_mse_1_5,      na.rm = TRUE)
  ),
  mse_6_10  = c(
    mean(baseline_fold_tbl$naive_mse_6_10,   na.rm = TRUE),
    mean(baseline_fold_tbl$rolling_mse_6_10, na.rm = TRUE),
    mean(baseline_fold_tbl$lm_mse_6_10,      na.rm = TRUE)
  ),
  sd_6_10   = c(
    sd(baseline_fold_tbl$naive_mse_6_10,   na.rm = TRUE),
    sd(baseline_fold_tbl$rolling_mse_6_10, na.rm = TRUE),
    sd(baseline_fold_tbl$lm_mse_6_10,      na.rm = TRUE)
  )
)

message("\n=== Baseline summary ===")
print(baseline_summary)

saveRDS(
  list(fold_results = baseline_fold_tbl, summary = baseline_summary),
  "results/baseline_summary.rds"
)

# =============================================================================
# SECTION 4: MAIN MODELS — RANDOM FOREST AND XGBOOST
# =============================================================================
# One model per horizon (direct forecasting) on each CV fold.
# Feature columns: feat_cols defined in Section 2.
# Models from the final fold are stored for submission use.
#
# Hyperparameters are fixed to the best values from HPT tuning:
#   RF:     num.trees=500, mtry=floor(p/3), min.node.size=5
#   XGBoost: max_depth=3, eta=0.1, subsample=0.7, colsample_bytree=0.7,
#            min_child_weight=5, with early stopping via inner holdout
# =============================================================================

model_fold_results <- vector("list", length(df_folds))
final_rf_models    <- vector("list", 10)   # one per horizon, from last fold
final_xgb_models   <- vector("list", 10)
final_medians      <- NULL

# RF tuned hyperparameters
rf_mtry          <- floor(length(feat_cols) / 3)
rf_num_trees     <- 500
rf_min_node_size <- 5

# XGBoost tuned hyperparameters
xgb_params <- list(
  objective        = "reg:squarederror",
  max_depth        = 3,
  eta              = 0.1,
  subsample        = 0.7,
  colsample_bytree = 0.7,
  min_child_weight = 5
)

message("\n=== Running Random Forest and XGBoost ===")
message("RF params:  num.trees=", rf_num_trees,
        ", mtry=", rf_mtry,
        ", min.node.size=", rf_min_node_size)
message("XGB params: max_depth=", xgb_params$max_depth,
        ", eta=", xgb_params$eta,
        ", subsample=", xgb_params$subsample,
        ", colsample_bytree=", xgb_params$colsample_bytree,
        ", min_child_weight=", xgb_params$min_child_weight)

for (f in seq_along(df_folds)) {

  fold <- df_folds[[f]]
  message("\n-- Model fold ", fold$fold_number, " --")

  # Impute this fold
  imputed  <- impute_fold(df_feat_raw, fold, feat_cols)
  df_train <- imputed$train
  df_val   <- imputed$val

  # Store medians from final fold for submission
  if (f == length(df_folds)) final_medians <- imputed$medians

  stopifnot(all(sapply(df_train[feat_cols], is.numeric)))

  # Feature matrices
  X_train <- as.matrix(df_train[feat_cols])
  X_val   <- as.matrix(df_val[feat_cols])

  # Storage for this fold's predictions
  rf_preds  <- matrix(NA_real_, nrow = nrow(X_val), ncol = 10)
  xgb_preds <- matrix(NA_real_, nrow = nrow(X_val), ncol = 10)

  for (h in 1:10) {

    target_col <- paste0("target_h", h)

    # Drop training rows where this horizon's target is NA
    complete   <- !is.na(df_train[[target_col]])
    y_train    <- df_train[[target_col]][complete]
    X_train_h  <- X_train[complete, , drop = FALSE]

    if (length(y_train) < 50) {
      warning("Fold ", f, " h", h, ": fewer than 50 training rows — skipping")
      next
    }

    # ---- Random Forest (ranger) ----------------------------------------------
    rf_mod <- ranger::ranger(
      x             = X_train_h,
      y             = y_train,
      num.trees     = rf_num_trees,
      mtry          = rf_mtry,
      min.node.size = rf_min_node_size,
      importance    = "impurity",
      seed          = 42
    )

    rf_preds[, h] <- predict(rf_mod, data = as.data.frame(X_val))$predictions

    # Store model from final fold
    if (f == length(df_folds)) final_rf_models[[h]] <- rf_mod

    # ---- XGBoost with inner holdout early stopping ---------------------------
    # Split training data 80/20 for early stopping only (not tuning —
    # hyperparameters are fixed). The inner holdout tells xgb.train when
    # to stop adding trees.
    n_inner     <- nrow(X_train_h)
    inner_cut   <- floor(0.8 * n_inner)
    X_inner_tr  <- X_train_h[1:inner_cut, , drop = FALSE]
    y_inner_tr  <- y_train[1:inner_cut]
    X_inner_val <- X_train_h[(inner_cut + 1):n_inner, , drop = FALSE]
    y_inner_val <- y_train[(inner_cut + 1):n_inner]

    dtrain_inner <- xgboost::xgb.DMatrix(data = X_inner_tr,  label = y_inner_tr)
    dval_inner   <- xgboost::xgb.DMatrix(data = X_inner_val, label = y_inner_val)

    # Find optimal nrounds via early stopping on inner holdout
    xgb_es <- xgboost::xgb.train(
      params                = xgb_params,
      data                  = dtrain_inner,
      nrounds               = 1000,
      watchlist             = list(val = dval_inner),
      early_stopping_rounds = 30,
      verbose               = 0
    )

    best_nrounds <- if (!is.null(xgb_es$best_iteration) &&
                        !is.na(xgb_es$best_iteration)) {
      xgb_es$best_iteration
    } else {
      300  # safe fallback
    }

    # Refit on full training slice with optimal nrounds
    dtrain_full <- xgboost::xgb.DMatrix(data = X_train_h, label = y_train)
    dval_outer  <- xgboost::xgb.DMatrix(data = X_val)

    xgb_mod <- xgboost::xgb.train(
      params  = xgb_params,
      data    = dtrain_full,
      nrounds = best_nrounds,
      verbose = 0
    )

    xgb_preds[, h] <- predict(xgb_mod, dval_outer)

    # Store model from final fold
    if (f == length(df_folds)) final_xgb_models[[h]] <- xgb_mod
  }

  # ---- Compute MSE -----------------------------------------------------------
  actuals <- sapply(paste0("target_h", 1:10), function(tc) df_val[[tc]])

  mse_block <- function(preds, actuals, h_range) {
    mean((as.vector(preds[, h_range]) - as.vector(actuals[, h_range]))^2,
         na.rm = TRUE)
  }

  model_fold_results[[f]] <- tibble::tibble(
    fold_number     = fold$fold_number,
    train_end       = as.character(fold$train_end_date),
    val_start       = as.character(fold$val_start_date),
    rf_mse_1_5      = mse_block(rf_preds,  actuals, 1:5),
    rf_mse_6_10     = mse_block(rf_preds,  actuals, 6:10),
    xgb_mse_1_5     = mse_block(xgb_preds, actuals, 1:5),
    xgb_mse_6_10    = mse_block(xgb_preds, actuals, 6:10)
  )

  message(
    "  RF  — MSE(1-5): ", round(model_fold_results[[f]]$rf_mse_1_5,  4),
    " | MSE(6-10): ",      round(model_fold_results[[f]]$rf_mse_6_10, 4), "\n",
    "  XGB — MSE(1-5): ", round(model_fold_results[[f]]$xgb_mse_1_5,  4),
    " | MSE(6-10): ",      round(model_fold_results[[f]]$xgb_mse_6_10, 4)
  )
}

# Summarise main models across folds
model_fold_tbl <- dplyr::bind_rows(model_fold_results)

model_summary_tbl <- tibble::tibble(
  model    = c("random_forest", "xgboost"),
  mse_1_5  = c(
    mean(model_fold_tbl$rf_mse_1_5,  na.rm = TRUE),
    mean(model_fold_tbl$xgb_mse_1_5, na.rm = TRUE)
  ),
  sd_1_5   = c(
    sd(model_fold_tbl$rf_mse_1_5,  na.rm = TRUE),
    sd(model_fold_tbl$xgb_mse_1_5, na.rm = TRUE)
  ),
  mse_6_10 = c(
    mean(model_fold_tbl$rf_mse_6_10,  na.rm = TRUE),
    mean(model_fold_tbl$xgb_mse_6_10, na.rm = TRUE)
  ),
  sd_6_10  = c(
    sd(model_fold_tbl$rf_mse_6_10,  na.rm = TRUE),
    sd(model_fold_tbl$xgb_mse_6_10, na.rm = TRUE)
  )
)

message("\n=== Model summary ===")
print(model_summary_tbl)

saveRDS(
  list(
    fold_results  = model_fold_tbl,
    summary       = model_summary_tbl,
    final_rf      = final_rf_models,
    final_xgb     = final_xgb_models,
    final_medians = final_medians,
    feat_cols     = feat_cols
  ),
  "results/model_summary.rds"
)

# =============================================================================
# SECTION 5: RESULTS COMPARISON
# =============================================================================
# Combines baseline and model summaries into a single comparison table.
# This is the primary output — tells you whether RF/XGBoost beat the baselines.
# =============================================================================

comparison_table <- dplyr::bind_rows(baseline_summary, model_summary_tbl) %>%
  arrange(mse_1_5)

message("\n=== Full comparison table (sorted by MSE 1-5) ===")
print(comparison_table, n = Inf)

# Also print sorted by MSE 6-10 since that is a separate prize
message("\n=== Full comparison table (sorted by MSE 6-10) ===")
print(comparison_table %>% arrange(mse_6_10), n = Inf)

saveRDS(comparison_table, "results/comparison_table.rds")

# Quick visual summary
message("\n=== KEY RESULTS ===")
message("Baseline to beat (naive, MSE 1-5):  ",
        round(baseline_summary$mse_1_5[baseline_summary$model == "naive"], 4))
message("Baseline to beat (naive, MSE 6-10): ",
        round(baseline_summary$mse_6_10[baseline_summary$model == "naive"], 4))
message("Best model MSE(1-5):  ",
        round(min(comparison_table$mse_1_5),  4),
        " (", comparison_table$model[which.min(comparison_table$mse_1_5)], ")")
message("Best model MSE(6-10): ",
        round(min(comparison_table$mse_6_10), 4),
        " (", comparison_table$model[which.min(comparison_table$mse_6_10)], ")")

message("\nAll results saved to results/")

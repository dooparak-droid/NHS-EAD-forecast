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
#   4. Main models (Random Forest, XGBoost, SVM)
#   5. Results comparison
#   6. Hyperparameter tuning summaries
#
# DEPENDENCIES (must be run first):
#   02_feature_eng.R — produces data/df_feat_raw.rds, data/df_folds.rds,
#                      data/core_predictors.rds
#
# OUTPUTS:
#   results/baseline_summary.rds  — baseline MSE per fold and summary
#   results/model_summary.rds     — RF, XGBoost, and SVM MSE per fold, summary,
#                                   and hyperparameter tuning logs
#   results/comparison_table.rds  — combined comparison of all approaches
# =============================================================================

library(dplyr)
library(tidyr)
library(purrr)
library(ranger)
library(xgboost)
library(e1071)

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

# SVMs are sensitive to feature scale, so need to scale after imputation
scale_fold <- function(df_train, df_val, feat_cols) {
  
  col_means <- sapply(df_train[feat_cols], mean, na.rm = TRUE)
  col_sds   <- sapply(df_train[feat_cols], sd,   na.rm = TRUE)
  
  # Handle zero or missing SDs
  col_sds[is.na(col_sds) | col_sds == 0] <- 1
  
  # Handle missing means
  col_means[is.na(col_means)] <- 0
  
  scale_matrix <- function(mat) {
    scaled <- sweep(sweep(mat, 2, col_means, "-"), 2, col_sds, "/")
    
    # Final safety cleanup
    scaled[!is.finite(scaled)] <- 0
    
    scaled
  }
  
  list(
    X_train_scaled = scale_matrix(as.matrix(df_train[feat_cols])),
    X_val_scaled   = scale_matrix(as.matrix(df_val[feat_cols]))
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

all_na_cols <- feat_cols[
  sapply(df_feat_raw[feat_cols], function(x) all(is.na(x)))
]

if (length(all_na_cols) > 0) {
  message("Dropping all-NA columns: ",
          paste(all_na_cols, collapse = ", "))
}

feat_cols <- setdiff(feat_cols, all_na_cols)

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
    fold_number              = fold$fold_number,
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
# SECTION 4: MAIN MODELS — RANDOM FOREST, XGBOOST, AND SVM
# =============================================================================
# One model per horizon (direct forecasting) on each CV fold.
# Feature columns: feat_cols defined in Section 2.
# Models from the final fold are stored for submission use.
#
# Inner holdout: last 20% of training rows used as a validation proxy for
# hyperparameter selection. Medians from the final fold are stored for use
# at submission time.
#
# Tuning logs: every (fold, horizon, hyperparameter combo) result is stored
# in rf_tuning_log, xgb_tuning_log, and svm_tuning_log. These are saved into
# model_summary.rds and can be inspected in Section 6 below.
# =============================================================================

model_fold_results <- vector("list", length(df_folds))
final_rf_models    <- vector("list", 10)
final_xgb_models   <- vector("list", 10)
final_svm_models   <- vector("list", 10)
final_medians      <- NULL

# Initialise tuning logs — accumulated across all folds and horizons
rf_tuning_log  <- tibble::tibble()
xgb_tuning_log <- tibble::tibble()
svm_tuning_log <- tibble::tibble()

message("\n=== Running Random Forest, XGBoost, and SVM ===")

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
  
  # Feature matrices (unscaled — used by RF and XGBoost)
  X_train <- as.matrix(df_train[feat_cols])
  X_val   <- as.matrix(df_val[feat_cols])
  
  # Scaled matrices (used by SVM) — computed once per fold from the full
  # training slice; the per-horizon complete-case subsetting is applied below
  scaled      <- scale_fold(df_train, df_val, feat_cols)
  X_train_svm_full <- scaled$X_train_scaled   # all training rows, scaled
  X_val_svm        <- scaled$X_val_scaled
  
  # Storage for this fold's predictions
  rf_preds  <- matrix(NA_real_, nrow = nrow(X_val), ncol = 10)
  xgb_preds <- matrix(NA_real_, nrow = nrow(X_val), ncol = 10)
  svm_preds <- matrix(NA_real_, nrow = nrow(X_val), ncol = 10)
  
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
    
    # ------------------------------------------------------------------
    # Shared inner holdout split (used by RF, XGBoost, and SVM)
    # Last 20% of training rows serve as a proxy validation set for
    # hyperparameter selection. Splitting here once avoids duplication
    # and guarantees all models are tuned on identical data.
    # ------------------------------------------------------------------
    n_inner     <- nrow(X_train_h)
    inner_cut   <- floor(0.8 * n_inner)
    X_inner_tr  <- X_train_h[1:inner_cut, ]
    y_inner_tr  <- y_train[1:inner_cut]
    X_inner_val <- X_train_h[(inner_cut + 1):n_inner, ]
    y_inner_val <- y_train[(inner_cut + 1):n_inner]
    
    # ---- Random Forest (ranger) ------------------------------------------
    
    rf_grid <- expand.grid(
      num.trees     = c(300, 500),
      mtry          = c(floor(sqrt(length(feat_cols))),
                        floor(length(feat_cols) / 3),
                        floor(length(feat_cols) / 2)),
      min.node.size = c(3, 5, 10)
    )
    
    best_rf_mse    <- Inf
    best_rf_params <- rf_grid[1, ]
    
    rf_grid_results <- vector("list", nrow(rf_grid))
    
    for (i in seq_len(nrow(rf_grid))) {
      params <- rf_grid[i, ]
      
      mod <- ranger::ranger(
        x             = X_inner_tr,
        y             = y_inner_tr,
        num.trees     = params$num.trees,
        mtry          = params$mtry,
        min.node.size = params$min.node.size,
        seed          = 42
      )
      
      preds <- predict(mod, data = as.data.frame(X_inner_val))$predictions
      mse   <- mean((preds - y_inner_val)^2, na.rm = TRUE)
      
      rf_grid_results[[i]] <- tibble::tibble(
        fold          = f,
        horizon       = h,
        num.trees     = params$num.trees,
        mtry          = params$mtry,
        min.node.size = params$min.node.size,
        inner_mse     = mse
      )
      
      if (mse < best_rf_mse) {
        best_rf_mse    <- mse
        best_rf_params <- params
      }
    }
    
    rf_tuning_log <- dplyr::bind_rows(rf_tuning_log, dplyr::bind_rows(rf_grid_results))
    
    # Refit on full training slice with best params
    rf_mod <- ranger::ranger(
      x             = X_train_h,
      y             = y_train,
      num.trees     = best_rf_params$num.trees,
      mtry          = best_rf_params$mtry,
      min.node.size = best_rf_params$min.node.size,
      importance    = "impurity",
      seed          = 42
    )
    
    rf_preds[, h] <- predict(rf_mod, data = as.data.frame(X_val))$predictions
    
    if (f == length(df_folds)) final_rf_models[[h]] <- rf_mod
    
    # ---- XGBoost ---------------------------------------------------------
    
    dtrain_inner <- xgboost::xgb.DMatrix(data = X_inner_tr, label = y_inner_tr)
    dval_inner   <- xgboost::xgb.DMatrix(data = X_inner_val, label = y_inner_val)
    dtrain_full  <- xgboost::xgb.DMatrix(data = X_train_h, label = y_train)
    dval_outer   <- xgboost::xgb.DMatrix(data = X_val)
    
    xgb_grid <- expand.grid(
      max_depth        = c(3, 4, 6),
      eta              = c(0.03, 0.05, 0.1),
      subsample        = c(0.7, 0.85),
      colsample_bytree = c(0.7, 0.85),
      min_child_weight = c(3, 5)
    )
    
    best_xgb_mse    <- Inf
    best_xgb_params <- xgb_grid[1, ]
    
    xgb_grid_results <- vector("list", nrow(xgb_grid))
    
    # Data-driven nrounds fallback via xgb.cv on first grid combination
    cv_probe <- tryCatch(
      xgboost::xgb.cv(
        params = list(
          objective        = "reg:squarederror",
          max_depth        = xgb_grid$max_depth[1],
          eta              = xgb_grid$eta[1],
          subsample        = xgb_grid$subsample[1],
          colsample_bytree = xgb_grid$colsample_bytree[1],
          min_child_weight = xgb_grid$min_child_weight[1]
        ),
        data                  = dtrain_inner,
        nrounds               = 1000,
        nfold                 = 3,
        early_stopping_rounds = 30,
        verbose               = 0
      ),
      error = function(e) NULL
    )
    best_nrounds <- if (!is.null(cv_probe) &&
                        !is.null(cv_probe$best_iteration) &&
                        !is.na(cv_probe$best_iteration)) {
      cv_probe$best_iteration
    } else {
      300
    }
    
    for (i in seq_len(nrow(xgb_grid))) {
      params <- xgb_grid[i, ]
      
      mod <- tryCatch(
        xgboost::xgb.train(
          params = list(
            objective        = "reg:squarederror",
            max_depth        = params$max_depth,
            eta              = params$eta,
            subsample        = params$subsample,
            colsample_bytree = params$colsample_bytree,
            min_child_weight = params$min_child_weight
          ),
          data                  = dtrain_inner,
          nrounds               = 1000,
          watchlist             = list(val = dval_inner),
          early_stopping_rounds = 30,
          verbose               = 0
        ),
        error = function(e) { warning("XGB grid iter ", i, " failed: ", e$message); NULL }
      )
      
      if (is.null(mod)) next
      
      preds <- predict(mod, dval_inner)
      mse   <- mean((preds - y_inner_val)^2, na.rm = TRUE)
      
      nrounds_used <- if (!is.null(mod$best_iteration) &&
                          length(mod$best_iteration) > 0 &&
                          !is.na(mod$best_iteration)) {
        mod$best_iteration
      } else {
        NA_integer_
      }
      
      xgb_grid_results[[i]] <- tibble::tibble(
        fold             = f,
        horizon          = h,
        max_depth        = params$max_depth,
        eta              = params$eta,
        subsample        = params$subsample,
        colsample_bytree = params$colsample_bytree,
        min_child_weight = params$min_child_weight,
        nrounds_used     = nrounds_used,
        inner_mse        = mse
      )
      
      if (!is.na(mse) && mse < best_xgb_mse) {
        best_xgb_mse    <- mse
        best_xgb_params <- params
        best_nrounds    <- if (!is.na(nrounds_used)) nrounds_used else best_nrounds
      }
    }
    
    xgb_tuning_log <- dplyr::bind_rows(xgb_tuning_log, dplyr::bind_rows(xgb_grid_results))
    
    # Refit on full training slice with best params
    xgb_mod <- xgboost::xgb.train(
      params = list(
        objective        = "reg:squarederror",
        max_depth        = best_xgb_params$max_depth,
        eta              = best_xgb_params$eta,
        subsample        = best_xgb_params$subsample,
        colsample_bytree = best_xgb_params$colsample_bytree,
        min_child_weight = best_xgb_params$min_child_weight
      ),
      data    = dtrain_full,
      nrounds = best_nrounds,
      verbose = 0
    )
    
    xgb_preds[, h] <- predict(xgb_mod, dval_outer)
    
    if (f == length(df_folds)) final_xgb_models[[h]] <- xgb_mod
    
    # ---- SVM (e1071) -------------------------------------------------------
    # Apply the same complete-case row mask used by RF/XGBoost so all three
    # models train on identical observations for this horizon.
    X_train_svm_h   <- X_train_svm_full[complete, , drop = FALSE]
    
    # Inner holdout split — scaled versions of the shared split above
    X_svm_inner_tr  <- X_train_svm_h[1:inner_cut, ]
    X_svm_inner_val <- X_train_svm_h[(inner_cut + 1):n_inner, ]
    
    svm_grid <- expand.grid(
      cost    = c(0.1, 1, 10),
      epsilon = c(0.05, 0.1, 0.2),
      gamma   = c(1 / ncol(X_train_svm_h), 0.01, 0.001)
    )
    
    best_svm_mse    <- Inf
    best_svm_params <- svm_grid[1, ]
    svm_grid_results <- vector("list", nrow(svm_grid))
    
    for (i in seq_len(nrow(svm_grid))) {
      params <- svm_grid[i, ]
      
      mod <- tryCatch(
        e1071::svm(
          x       = X_svm_inner_tr,
          y       = y_inner_tr,
          kernel  = "radial",
          cost    = params$cost,
          epsilon = params$epsilon,
          gamma   = params$gamma
        ),
        error = function(e) { warning("SVM grid iter ", i, " failed: ", e$message); NULL }
      )
      
      if (is.null(mod)) next
      
      preds <- predict(mod, X_svm_inner_val)
      mse   <- mean((preds - y_inner_val)^2, na.rm = TRUE)
      
      svm_grid_results[[i]] <- tibble::tibble(
        fold      = f,
        horizon   = h,
        cost      = params$cost,
        epsilon   = params$epsilon,
        gamma     = params$gamma,
        inner_mse = mse
      )
      
      if (!is.na(mse) && mse < best_svm_mse) {
        best_svm_mse    <- mse
        best_svm_params <- params
      }
    }
    
    svm_tuning_log <- dplyr::bind_rows(svm_tuning_log, dplyr::bind_rows(svm_grid_results))
    
    if (any(!is.finite(X_train_svm_h))) {
      stop("Non-finite values in X_train_svm_h")
    }
    
    if (any(!is.finite(X_val_svm))) {
      stop("Non-finite values in X_val_svm")
    }
    
    if (any(!is.finite(y_train))) {
      stop("Non-finite values in y_train")
    }
    
    # Refit on full training slice with best params
    svm_mod <- e1071::svm(
      x       = X_train_svm_h,
      y       = y_train,
      kernel  = "radial",
      cost    = best_svm_params$cost,
      epsilon = best_svm_params$epsilon,
      gamma   = best_svm_params$gamma
    )
    
    svm_preds[, h] <- predict(svm_mod, X_val_svm)
    
    if (f == length(df_folds)) final_svm_models[[h]] <- svm_mod
    
  } # ---- end horizon loop --------------------------------------------------
  
  # ---- Compute MSE across all horizons for this fold -----------------------
  actuals <- sapply(paste0("target_h", 1:10), function(tc) df_val[[tc]])
  
  mse_block <- function(preds, actuals, h_range) {
    mean((as.vector(preds[, h_range]) - as.vector(actuals[, h_range]))^2,
         na.rm = TRUE)
  }
  
  model_fold_results[[f]] <- tibble::tibble(
    fold_number  = fold$fold_number,
    train_end    = as.character(fold$train_end_date),
    val_start    = as.character(fold$val_start_date),
    rf_mse_1_5   = mse_block(rf_preds,  actuals, 1:5),
    rf_mse_6_10  = mse_block(rf_preds,  actuals, 6:10),
    xgb_mse_1_5  = mse_block(xgb_preds, actuals, 1:5),
    xgb_mse_6_10 = mse_block(xgb_preds, actuals, 6:10),
    svm_mse_1_5  = mse_block(svm_preds, actuals, 1:5),
    svm_mse_6_10 = mse_block(svm_preds, actuals, 6:10)
  )
  
  message(
    "  RF  — MSE(1-5): ", round(model_fold_results[[f]]$rf_mse_1_5,   4),
    " | MSE(6-10): ",      round(model_fold_results[[f]]$rf_mse_6_10,  4), "\n",
    "  XGB — MSE(1-5): ", round(model_fold_results[[f]]$xgb_mse_1_5,  4),
    " | MSE(6-10): ",      round(model_fold_results[[f]]$xgb_mse_6_10, 4), "\n",
    "  SVM — MSE(1-5): ", round(model_fold_results[[f]]$svm_mse_1_5,  4),
    " | MSE(6-10): ",      round(model_fold_results[[f]]$svm_mse_6_10, 4)
  )
  
} # ---- end fold loop --------------------------------------------------------

# Summarise main models across folds
model_fold_tbl <- dplyr::bind_rows(model_fold_results)
model_summary <- tibble::tibble(
  model    = c("random_forest", "xgboost", "svm"),
  mse_1_5  = c(
    mean(model_fold_tbl$rf_mse_1_5,  na.rm = TRUE),
    mean(model_fold_tbl$xgb_mse_1_5, na.rm = TRUE),
    mean(model_fold_tbl$svm_mse_1_5, na.rm = TRUE)
  ),
  sd_1_5   = c(
    sd(model_fold_tbl$rf_mse_1_5,  na.rm = TRUE),
    sd(model_fold_tbl$xgb_mse_1_5, na.rm = TRUE),
    sd(model_fold_tbl$svm_mse_1_5, na.rm = TRUE)
  ),
  mse_6_10 = c(
    mean(model_fold_tbl$rf_mse_6_10,  na.rm = TRUE),
    mean(model_fold_tbl$xgb_mse_6_10, na.rm = TRUE),
    mean(model_fold_tbl$svm_mse_6_10, na.rm = TRUE)
  ),
  sd_6_10  = c(
    sd(model_fold_tbl$rf_mse_6_10,  na.rm = TRUE),
    sd(model_fold_tbl$xgb_mse_6_10, na.rm = TRUE),
    sd(model_fold_tbl$svm_mse_6_10, na.rm = TRUE)
  )
)

message("\n=== Model summary ===")
print(model_summary)

saveRDS(
  list(
    fold_results   = model_fold_tbl,
    summary        = model_summary,
    final_rf       = final_rf_models,
    final_xgb      = final_xgb_models,
    final_svm      = final_svm_models,
    final_medians  = final_medians,
    feat_cols      = feat_cols,
    rf_tuning_log  = rf_tuning_log,
    xgb_tuning_log = xgb_tuning_log,
    svm_tuning_log = svm_tuning_log
  ),
  "results/model_summary.rds"
)

# =============================================================================
# SECTION 5: RESULTS COMPARISON
# =============================================================================
# Combines baseline and model summaries into a single comparison table.
# This is the primary output — tells you whether RF/XGBoost/SVM beat baselines.
# =============================================================================

comparison_table <- dplyr::bind_rows(baseline_summary, model_summary) %>%
  arrange(mse_1_5)

message("\n=== Full comparison table (sorted by MSE 1-5) ===")
print(comparison_table, n = Inf)

message("\n=== Full comparison table (sorted by MSE 6-10) ===")
print(comparison_table %>% arrange(mse_6_10), n = Inf)

saveRDS(comparison_table, "results/comparison_table.rds")

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

# =============================================================================
# SECTION 6: HYPERPARAMETER TUNING SUMMARIES
# =============================================================================
# Inspect which hyperparameter combinations performed best, averaged across
# all folds and horizons. Useful for narrowing the grid in future runs.
#
# All three tuning logs are also saved inside model_summary.rds so you can
# reload and re-examine them without re-running the full script:
#
#   saved <- readRDS("results/model_summary.rds")
#   rf_tuning_log  <- saved$rf_tuning_log
#   xgb_tuning_log <- saved$xgb_tuning_log
#   svm_tuning_log <- saved$svm_tuning_log
# =============================================================================

message("\n=== Random Forest hyperparameter summary (mean inner MSE across folds & horizons) ===")
rf_hp_summary <- rf_tuning_log %>%
  group_by(num.trees, mtry, min.node.size) %>%
  summarise(
    mean_inner_mse = mean(inner_mse, na.rm = TRUE),
    sd_inner_mse   = sd(inner_mse,   na.rm = TRUE),
    n_runs         = n(),
    .groups        = "drop"
  ) %>%
  arrange(mean_inner_mse)
print(rf_hp_summary, n = Inf)

message("\n=== XGBoost hyperparameter summary (mean inner MSE across folds & horizons) ===")
xgb_hp_summary <- xgb_tuning_log %>%
  group_by(max_depth, eta, subsample, colsample_bytree, min_child_weight) %>%
  summarise(
    mean_inner_mse  = mean(inner_mse,    na.rm = TRUE),
    sd_inner_mse    = sd(inner_mse,      na.rm = TRUE),
    mean_nrounds    = mean(nrounds_used, na.rm = TRUE),
    n_runs          = n(),
    .groups         = "drop"
  ) %>%
  arrange(mean_inner_mse)
print(xgb_hp_summary, n = Inf)

message("\n=== SVM hyperparameter summary (mean inner MSE across folds & horizons) ===")
svm_hp_summary <- svm_tuning_log %>%
  group_by(cost, epsilon, gamma) %>%
  summarise(
    mean_inner_mse = mean(inner_mse, na.rm = TRUE),
    sd_inner_mse   = sd(inner_mse,   na.rm = TRUE),
    n_runs         = n(),
    .groups        = "drop"
  ) %>%
  arrange(mean_inner_mse)
print(svm_hp_summary, n = Inf)

# -----------------------------------------------------------------------------
# Fold timeline diagnostics
# -----------------------------------------------------------------------------
df_folds[[5]]$val_start_date
df_folds[[5]]$train_end_date
df_folds[[7]]$val_start_date

library(ggplot2)
ggplot(df_feat_raw, aes(x = date, y = .data[[outcome_col]])) +
  geom_line() +
  geom_vline(xintercept = as.Date(sapply(df_folds, \(f) f$val_start_date)),
             linetype = "dashed", colour = "red") +
  labs(title = "Target with fold validation boundaries")
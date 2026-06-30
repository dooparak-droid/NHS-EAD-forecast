# =============================================================================
# SPHERE-PPL NHS Forecasting Competition
# 02_feature_eng.R
# =============================================================================
# This script contains three functions and a run section.
#
# FUNCTIONS:
#   build_features()  — deterministic feature engineering (no imputation).
#                       Takes raw clean data, returns engineered feature matrix.
#                       Imputation is intentionally excluded — it belongs inside
#                       the CV fold loop in 03_modeling.R where training medians
#                       can be computed from training rows only.
#
#   make_targets()    — adds 10 horizon target columns to the feature matrix.
#                       target_h1 = outcome 1 day ahead, ..., target_h10 = 10 days ahead.
#
#   make_cv_folds()   — splits the feature matrix into temporal CV fold
#                       specifications using an expanding window with a gap.
#
# PREDICTOR SELECTION PHILOSOPHY:
#   Rather than using the raw top-N by correlation (which includes redundant
#   correlated pairs), we define a curated set of clinically distinct predictors.
#   These fall into four groups:
#
#   GROUP 1 — Community capacity (Sirona NCtR):
#     Individual lags for NBT P1 NCtR Patients and Beddays, % beds NCtR at
#     BRI and WGH. Collapsed into total_NCtR_patients aggregate (P1 + P2).
#
#   GROUP 2 — Acute-to-community transfer backlog (DtA):
#     DtA P2 UNBOOKED as anchor individual predictor.
#     Collapsed into total_DtA_waiting aggregate (P2 + P3).
#
#   GROUP 3 — Acute hospital pressure (DTA + ED):
#     No. of DTAs at BRI and NBT collapsed into total_DTA_acute.
#     ED >12hr at BRI collapsed into ed_pressure_roll3 (3-day rolling mean).
#     ED >12hr at WGH retained individually (distinct hospital signal).
#
#   GROUP 4 — Emergency demand:
#     999 Waiting Calls (unmet demand) and Medical Outliers at BRI.
#
# RUN SECTION:
#   Loads outputs from 01_eda.R, builds the feature matrix, adds targets,
#   creates folds, and saves outputs for 03_modeling.R:
#     data/df_feat_raw.rds     — engineered feature matrix with target columns
#     data/df_folds.rds        — list of CV fold specifications
#     data/core_predictors.rds — character vector of individual predictors used
#
# DEPENDENCIES (must be run first):
#   01_eda.R — produces data/modeling_data_clean.rds
# =============================================================================

library(dplyr)
library(tidyr)
library(lubridate)
library(purrr)
library(rlang)

# =============================================================================
# FUNCTION 1: build_features()
# =============================================================================

#' Build deterministic feature matrix for NHS avoidable deaths forecasting
#'
#' @param df              A tibble: clean wide-format daily data, sorted by date.
#' @param core_predictors Character vector of individual predictor column names
#'                        to lag. Use the curated set defined in the run section.
#' @param outcome_col     Name of the outcome column.
#'
#' @return A named list with one element:
#'   $data — tibble of engineered features. Early rows where outcome_lag3
#'           is NA are dropped. No imputation applied.

build_features <- function(
    df,
    core_predictors = NULL,
    outcome_col     = "estimated_avoidable_deaths -- BNSSG"
) {

  # ---------------------------------------------------------------------------
  # 0. Input validation
  # ---------------------------------------------------------------------------

  stopifnot(
    "`df` must be a data frame"         = is.data.frame(df),
    "`df` must contain a `date` column" = "date" %in% names(df),
    "outcome_col must exist in df"      = outcome_col %in% names(df)
  )

  if (is.null(core_predictors)) {
    stop(
      "core_predictors cannot be NULL. ",
      "Supply the character vector defined in the run section of this script."
    )
  }

  missing_preds <- base::setdiff(core_predictors, names(df))
  if (length(missing_preds) > 0) {
    warning(
      length(missing_preds), " core_predictors not found in df and will be skipped: ",
      paste(missing_preds, collapse = ", ")
    )
    core_predictors <- base::intersect(core_predictors, names(df))
  }

  df <- df %>% arrange(date)

  # ---------------------------------------------------------------------------
  # 1. OUTCOME LAG FEATURES
  #
  # Only lags >= 3 are safe due to the 3-day reporting lag on the outcome.
  # lag3  = most recent observable outcome at prediction time D.
  # lag7  = same weekday last week (ACF peak at lag 7, r~0.53).
  # lag14 = same weekday two weeks ago.
  # Momentum encodes whether the outcome has been rising or falling recently.
  # ---------------------------------------------------------------------------

  df <- df %>%
    mutate(
      outcome_lag3  = lag(.data[[outcome_col]], 3),
      outcome_lag4  = lag(.data[[outcome_col]], 4),
      outcome_lag5  = lag(.data[[outcome_col]], 5),
      outcome_lag6  = lag(.data[[outcome_col]], 6),
      outcome_lag7  = lag(.data[[outcome_col]], 7),
      outcome_lag8  = lag(.data[[outcome_col]], 8),
      outcome_lag9  = lag(.data[[outcome_col]], 9),
      outcome_lag14 = lag(.data[[outcome_col]], 14),

      outcome_roll3 = rowMeans(
        cbind(outcome_lag3, outcome_lag4, outcome_lag5),
        na.rm = FALSE
      ),
      outcome_roll7 = rowMeans(
        cbind(
          outcome_lag3, outcome_lag4, outcome_lag5,
          outcome_lag6, outcome_lag7, outcome_lag8, outcome_lag9
        ),
        na.rm = FALSE
      ),
      outcome_momentum = outcome_lag3 - outcome_lag6
    )

  # ---------------------------------------------------------------------------
  # 2. CALENDAR AND TEMPORAL FEATURES
  # ---------------------------------------------------------------------------

  df <- df %>%
    mutate(
      dow          = wday(date, week_start = 1),   # 1 = Monday, 7 = Sunday
      dow_factor   = factor(
        wday(date, label = TRUE, week_start = 1),
        levels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")
      ),
      is_monday    = as.integer(dow == 1),
      is_weekend   = as.integer(dow %in% c(6, 7)),
      yday         = yday(date),
      yday_sin     = sin(2 * pi * yday / 365),
      yday_cos     = cos(2 * pi * yday / 365),
      month_int    = month(date),
      month_fct    = factor(month(date), levels = 1:12, labels = month.abb),
      week_of_year = isoweek(date),
      time_index   = as.integer(date - min(date))
    )

  # Holiday and structural period flags
  # These encode periods where NHS demand patterns differ structurally
  # from normal weeks. A model trained on normal periods will systematically
  # underpredict surges in these windows without explicit flags.
  #
  # is_new_year_surge: first two weeks of January — post-holiday surge period
  # is_christmas_week: 24-31 December — reduced capacity, delayed presentations
  # is_easter_week:    Easter week — minor but consistent effect
  # is_bank_holiday_adj: day after a bank holiday — backlog effect

  df <- df %>%
    mutate(
      # New Year surge: 1-14 January
      is_new_year_surge = as.integer(
        month_int == 1 & yday <= 14
      ),

      # Christmas week: 24-31 December
      is_christmas_week = as.integer(
        month_int == 12 & yday >= 358
      ),

      # Easter: approximate — week containing day 85-105 (late March/early April)
      # This is an approximation since Easter moves — good enough for a tree model
      is_easter_period = as.integer(
        yday >= 85 & yday <= 105
      ),

      # Summer low: July-August — outcome consistently lowest in this period
      is_summer = as.integer(
        month_int %in% c(7, 8)
      )
    )

  # ---------------------------------------------------------------------------
  # 3. PREDICTOR LAG FEATURES
  #
  # Applied only to core_predictors — the curated clinically distinct set.
  # For each predictor X:
  #   X_lag1, X_lag2, X_lag3 — individual recent lags
  #   X_roll3                — mean of lags 1-3 (short-term level)
  #   X_roll7                — mean of lags 1-7 (weekly level)
  # Lags 4-7 are computed internally to build roll7, then dropped.
  #
  # NOTE: the pipeline in 01_eda.R already applies a 1-day lag to all
  # "yesterday" metrics. So _lag1 here = 2 days ago in calendar terms.
  # ---------------------------------------------------------------------------

  for (pred in core_predictors) {

    l1 <- paste0(pred, "_lag1")
    l2 <- paste0(pred, "_lag2")
    l3 <- paste0(pred, "_lag3")
    l4 <- paste0(pred, "_lag4")
    l5 <- paste0(pred, "_lag5")
    l6 <- paste0(pred, "_lag6")
    l7 <- paste0(pred, "_lag7")
    r3 <- paste0(pred, "_roll3")
    r7 <- paste0(pred, "_roll7")

    df <- df %>%
      mutate(
        !!l1 := lag(.data[[pred]], 1),
        !!l2 := lag(.data[[pred]], 2),
        !!l3 := lag(.data[[pred]], 3),
        !!l4 := lag(.data[[pred]], 4),
        !!l5 := lag(.data[[pred]], 5),
        !!l6 := lag(.data[[pred]], 6),
        !!l7 := lag(.data[[pred]], 7)
      ) %>%
      mutate(
        !!r3 := rowMeans(
          cbind(.data[[l1]], .data[[l2]], .data[[l3]]),
          na.rm = FALSE
        ),
        !!r7 := rowMeans(
          cbind(
            .data[[l1]], .data[[l2]], .data[[l3]],
            .data[[l4]], .data[[l5]], .data[[l6]], .data[[l7]]
          ),
          na.rm = FALSE
        )
      ) %>%
      select(-all_of(c(l4, l5, l6, l7)))
  }

  # ---------------------------------------------------------------------------
  # 4. DERIVED CLINICAL AGGREGATES
  #
  # Four aggregate features collapsing clinically related but correlated
  # predictors into single system-level signals. Each aggregate uses
  # safe_rowsum() which returns NA only when ALL inputs are NA — preserving
  # the distinction between "zero pressure" and "missing data".
  #
  # These aggregates are built from the RAW columns in df (before lagging),
  # then the aggregates themselves are lagged in the missingness section
  # below via the _missing flags. The raw aggregate values here represent
  # the same-day system state; downstream imputation in 03_modeling.R will
  # handle missing values.
  # ---------------------------------------------------------------------------

  safe_rowsum <- function(df, cols) {
    present <- base::intersect(cols, names(df))
    if (length(present) == 0) return(rep(NA_real_, nrow(df)))
    mat     <- as.matrix(df[present])
    result  <- rowSums(mat, na.rm = TRUE)
    all_na  <- apply(mat, 1, function(x) all(is.na(x)))
    result[all_na] <- NA_real_
    result
  }

  # GROUP 1: Community capacity — total NCtR patients (P1 + P2) at NBT
  df$total_NCtR_patients <- safe_rowsum(df, c(
    "NBT P1 NCtR Patients -- Sirona",
    "NBT P2 NCtR Patients -- Sirona"
  ))

  # GROUP 2: Transfer backlog — total DtA patients waiting (P2 + P3)
  df$total_DtA_waiting <- safe_rowsum(df, c(
    "DtA P2 UNBOOKED Waiting for capacity, medically fit and ready to leave acute -- Sirona",
    "DtA P3 BOOKED AND UN-BOOKED Waiting for capacity, medically fit and ready to leave ACUTE -- Sirona"
  ))

  # GROUP 3: Acute DTA pressure — total delayed transfers (BRI + NBT)
  df$total_DTA_acute <- safe_rowsum(df, c(
    "No. of DTAs -- BRI",
    "No. of DTAs -- NBT",
    "No. of DTAs -- WGH"
  ))

  # GROUP 4: Sustained ED pressure — rolling 3-day mean of ED >12hr at BRI
  # Built from lagged values to respect the prediction-time constraint.
  ed_col <- "% of patients spending >12 hours in ED -- BRI"

  if (ed_col %in% names(df)) {
    df <- df %>%
      mutate(
        ed_bri_l1 = lag(.data[[ed_col]], 1),
        ed_bri_l2 = lag(.data[[ed_col]], 2),
        ed_bri_l3 = lag(.data[[ed_col]], 3),
        ed_pressure_roll3 = rowMeans(
          cbind(ed_bri_l1, ed_bri_l2, ed_bri_l3),
          na.rm = FALSE
        )
      ) %>%
      select(-ed_bri_l1, -ed_bri_l2, -ed_bri_l3)
  } else {
    warning("ED >12hr BRI column not found — ed_pressure_roll3 set to NA.")
    df$ed_pressure_roll3 <- NA_real_
  }

  derived_cols <- c(
    "total_NCtR_patients",
    "total_DtA_waiting",
    "total_DTA_acute",
    "ed_pressure_roll3"
  )

  # ---------------------------------------------------------------------------
  # 5. MISSINGNESS INDICATORS
  #
  # Binary flags marking NA positions before imputation occurs.
  # Created for all predictor lag columns and all derived aggregates.
  # A system failing to report is itself operationally informative.
  # ---------------------------------------------------------------------------

  lag_suffixes <- c("_lag1", "_lag2", "_lag3", "_roll3", "_roll7")

  cols_to_flag <- c(
    as.vector(outer(core_predictors, lag_suffixes, paste0)),
    derived_cols,
    "outcome_lag3"
  )
  cols_to_flag <- base::intersect(cols_to_flag, names(df))

  for (col in cols_to_flag) {
    df[[paste0(col, "_missing")]] <- as.integer(is.na(df[[col]]))
  }

  # ---------------------------------------------------------------------------
  # 6. DROP EARLY ROWS
  #
  # Rows where outcome_lag3 is NA are the earliest rows in the dataset.
  # They have no observable outcome history and cannot be used for training.
  # ---------------------------------------------------------------------------

  View(df)
  n_before  <- nrow(df)
  first_valid <- min(df$date[!is.na(df$outcome_lag3)])
  df <- df %>% filter(date >= first_valid)
  n_dropped <- n_before - nrow(df)

  if (n_dropped > 0) {
    message(n_dropped, " rows dropped — outcome_lag3 is NA (expected for early rows).")
  }

  # ---------------------------------------------------------------------------
  # 7. SELECT AND ORDER OUTPUT COLUMNS
  #
  # Uses startsWith() to match predictor lag columns — avoids regex breakage
  # from special characters in column names (parentheses, commas, % signs).
  # Raw same-day predictor columns are excluded (not available at prediction
  # time and their presence would cause leakage).
  # ---------------------------------------------------------------------------

  pred_lag_cols <- names(df)[
    sapply(names(df), function(col) {
      any(startsWith(col, paste0(core_predictors, "_")))
    })
  ]

  cols_to_keep <- c(
    "date",
    outcome_col,
    grep("^outcome_", names(df), value = TRUE),
    c("dow", "dow_factor", "is_monday", "is_weekend",
      "yday", "yday_sin", "yday_cos",
      "month_int", "month_fct", "week_of_year", "time_index",
      "is_new_year_surge", "is_christmas_week",
      "is_easter_period", "is_summer"),
    pred_lag_cols,
    derived_cols,
    names(df)[grepl("_missing$", names(df))]
  )

  cols_to_keep <- unique(base::intersect(cols_to_keep, names(df)))
  df_out       <- df %>% select(all_of(cols_to_keep))

  # ---------------------------------------------------------------------------
  # 8. RETURN
  # ---------------------------------------------------------------------------

  message(
    "build_features() complete. ",
    "Rows: ", nrow(df_out), " | ",
    "Feature columns: ", ncol(df_out) - 2, " (excl. date and outcome)"
  )

  list(data = df_out)
}


# =============================================================================
# FUNCTION 2: make_targets()
# =============================================================================

#' Add forecast horizon target columns to the feature matrix
#'
#' Creates target_h1 through target_h10 by leading the outcome forward.
#' These are labels only — never used as features. Last 10 rows will have
#' NA targets (excluded from training, retained for submission prediction).
#'
#' @param df          Feature matrix from build_features()$data, sorted by date.
#' @param outcome_col Name of the outcome column.
#' @return The same tibble with 10 additional target columns.

make_targets <- function(
    df,
    outcome_col = "estimated_avoidable_deaths -- BNSSG"
) {

  stopifnot(
    "`df` must be a data frame"    = is.data.frame(df),
    "outcome_col must exist in df" = outcome_col %in% names(df),
    "`df` must be sorted by date"  = !is.unsorted(df$date)
  )

  df <- df %>%
    mutate(
      target_h1  = lead(.data[[outcome_col]], 1),
      target_h2  = lead(.data[[outcome_col]], 2),
      target_h3  = lead(.data[[outcome_col]], 3),
      target_h4  = lead(.data[[outcome_col]], 4),
      target_h5  = lead(.data[[outcome_col]], 5),
      target_h6  = lead(.data[[outcome_col]], 6),
      target_h7  = lead(.data[[outcome_col]], 7),
      target_h8  = lead(.data[[outcome_col]], 8),
      target_h9  = lead(.data[[outcome_col]], 9),
      target_h10 = lead(.data[[outcome_col]], 10)
    )

  message("make_targets() complete. Last 10 rows will have NA targets — expected.")
  df
}


# =============================================================================
# FUNCTION 3: make_cv_folds()
# =============================================================================

#' Create temporal CV fold specifications using an expanding window
#'
#' Returns a list of fold objects — no model fitting happens here.
#' Validation windows are fixed at 10 days to match the competition structure.
#' The gap enforces a buffer between training end and validation start.
#'
#' @param df             Feature matrix with targets (output of make_targets()).
#' @param n_folds        Number of CV folds. Default 8.
#' @param gap            Gap in days between training end and validation start.
#'                       Default 10 (= forecast horizon).
#' @param min_train_days Minimum training rows before the first fold. Default 300.
#'
#' @return List of fold objects each containing:
#'   $fold_number, $train_end_date, $val_start_date, $val_end_date,
#'   $train_idx, $val_idx

make_cv_folds <- function(
    df,
    n_folds        = 8,
    gap            = 10,
    min_train_days = 300
) {

  stopifnot(
    "`df` must be a data frame"   = is.data.frame(df),
    "`df` must contain `date`"    = "date" %in% names(df),
    "`df` must be sorted by date" = !is.unsorted(df$date)
  )

  n       <- nrow(df)
  horizon <- 10

  earliest_val_start <- min_train_days + gap + 1
  latest_val_end     <- n
  available_range    <- latest_val_end - horizon - earliest_val_start + 1

  if (available_range < n_folds * horizon) {
    stop(
      "Not enough data to place ", n_folds, " folds of ", horizon, " days each. ",
      "Reduce n_folds or min_train_days."
    )
  }

  val_start_indices <- round(
    seq(earliest_val_start, latest_val_end - horizon + 1, length.out = n_folds)
  )

  folds <- vector("list", n_folds)

  for (i in seq_len(n_folds)) {

    val_start <- val_start_indices[i]
    val_end   <- val_start + horizon - 1
    train_end <- val_start - gap - 1

    if (train_end < min_train_days) {
      warning("Fold ", i, " has fewer than min_train_days rows. Skipping.")
      next
    }

    folds[[i]] <- list(
      fold_number    = i,
      train_end_date = df$date[train_end],
      val_start_date = df$date[val_start],
      val_end_date   = df$date[val_end],
      train_idx      = seq_len(train_end),
      val_idx        = seq(val_start, val_end)
    )

    message(
      "Fold ", i, ": ",
      "train = [", df$date[1], " to ", df$date[train_end], "] ",
      "(", train_end, " rows) | ",
      "gap = ", gap, " days | ",
      "val = [", df$date[val_start], " to ", df$date[val_end], "]"
    )
  }

  folds <- purrr::compact(folds)
  message("\n", length(folds), " folds created successfully.")
  folds
}


# =============================================================================
# RUN SECTION
# =============================================================================
# Defines the curated predictor set, builds the feature matrix, adds targets,
# creates folds, and saves all outputs for 03_modeling.R.
#
# top20_vars from 01_eda.R is no longer used directly here. The predictor
# set below is derived from the correlation analysis but deduplicated and
# grouped by clinical meaning into core_predictors (individual lags) and
# derived aggregates (handled inside build_features()).
# =============================================================================

# Load clean dataset from 01_eda.R
df <- readRDS("data/modeling_data_clean.rds")

# -----------------------------------------------------------------------------
# DEFINE CORE PREDICTORS
#
# Individual predictors that receive lag and roll features in Section 3.
# Chosen to be clinically distinct — one representative per correlated
# cluster. Redundant correlated pairs (e.g. P1 vs P2 NCtR Patients,
# Calls Received vs Calls Answered) are collapsed into derived aggregates
# in Section 4 rather than duplicated here.
# -----------------------------------------------------------------------------

core_predictors <- c(

  # --- Community capacity: NCtR ---
  # P1 NCtR Patients retained as individual predictor (P2 collapsed into aggregate)
  "NBT P1 NCtR Patients -- Sirona",
  # Bed-days: distinct dimension from patient count
  "NBT P1 NCtR Beddays -- Sirona",
  # Proportion metrics at WGH and BRI — different type from raw counts
  "% of beds occupied by patients with NCtR -- WGH",
  "% of beds occupied by patients with NCtR -- BRI",

  # --- Transfer backlog: DtA ---
  # P2 UNBOOKED retained as anchor (P3 and TOTAL collapsed into aggregate)
  "DtA P2 UNBOOKED Waiting for capacity, medically fit and ready to leave acute -- Sirona",

  # --- Acute hospital pressure ---
  # Medical outliers: patients in wrong specialty beds — distinct signal
  "Number of Medical Outliers -- BRI",
  # ED pressure at WGH — distinct from BRI (which goes into ed_pressure_roll3)
  "% of patients spending >12 hours in ED -- WGH",

  # --- Emergency demand ---
  # Waiting calls: unmet 999 demand (distinct from calls received/answered)
  "Number of Waiting Calls on the 999 Call Stack -- BNSSG",

  # --- Upstream demand: IUC/111 service ---
  "(Severnside) Calls Received -- SevernSide",

  # --- Bed occupancy: available capacity dimension ---
  "G&A Bed occupancy -- BRI"
)

# Verify all core predictors exist
missing_check <- base::setdiff(core_predictors, names(df))
if (length(missing_check) > 0) {
  warning(
    "These core_predictors are not in df and will cause issues: ",
    paste(missing_check, collapse = ", ")
  )
}

# -----------------------------------------------------------------------------
# BUILD PIPELINE
# -----------------------------------------------------------------------------

# Step 1: Build deterministic feature matrix (no imputation)
result      <- build_features(df, core_predictors = core_predictors)
df_feat_raw <- result$data

# Step 2: Add horizon target columns
df_feat_raw <- make_targets(df_feat_raw)

# Step 3: Create CV fold specifications
df_folds <- make_cv_folds(
  df_feat_raw,
  n_folds        = 8,
  gap            = 10,
  min_train_days = 300
)

# Step 4: Inspect output
glimpse(df_feat_raw)
message("Rows: ", nrow(df_feat_raw), " | Columns: ", ncol(df_feat_raw))

# Step 5: Save for 03_modeling.R
saveRDS(df_feat_raw,     "data/df_feat_raw.rds")
saveRDS(df_folds,        "data/df_folds.rds")
saveRDS(core_predictors, "data/core_predictors.rds")

message("Saved df_feat_raw.rds, df_folds.rds, and core_predictors.rds to data/")

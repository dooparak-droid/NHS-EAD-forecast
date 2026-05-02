# =============================================================================
# SPHERE-PPL NHS Forecasting Competition
# Feature Engineering Function
# =============================================================================
# build_features() takes the clean wide-format daily dataset and returns a
# fully engineered feature matrix ready for modelling, plus training medians
# for safe imputation across CV folds.
#
# Usage:
#   # On a training fold (computes and returns medians):
#   result_train <- build_features(df_train, top_predictors = top20_vars)
#   feat_train   <- result_train$data
#   medians      <- result_train$medians
#
#   # On a validation or assessment fold (uses training medians):
#   result_val   <- build_features(df_val, top_predictors = top20_vars,
#                                  training_medians = medians)
#   feat_val     <- result_val$data
# =============================================================================

library(dplyr)
library(tidyr)
library(lubridate)
library(purrr)
library(rlang)

# -----------------------------------------------------------------------------
# Helper: safe rolling mean over a set of already-computed lag columns
# Returns NA if all inputs are NA (does not impute here — that happens later).
# -----------------------------------------------------------------------------
roll_mean_cols <- function(df, lag_cols) {
  df %>%
    mutate(
      result = rowMeans(across(all_of(lag_cols)), na.rm = FALSE)
    ) %>%
    pull(result)
}

# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------

#' Build feature matrix for NHS avoidable deaths forecasting
#'
#' @param df             A tibble: clean wide-format daily data, sorted by date.
#'                       Must contain columns: `date`, `estimated_avoidable_deaths -- BNSSG`,
#'                       and all predictor columns.
#' @param top_predictors Character vector of predictor column names to lag.
#'                       Defaults to NULL — you MUST supply this (see note below).
#' @param training_medians Named numeric vector from a previous build_features()
#'                       call on a training fold. Pass NULL (default) to compute
#'                       medians from `df` itself (training fold behaviour).
#' @param outcome_col    Name of the outcome column. Defaults to the competition target.
#'
#' @return A named list:
#'   $data    — tibble of engineered features (outcome lags, calendar, predictor
#'              lags, derived features, missingness flags), with rows dropped
#'              where outcome_lag3 is NA.
#'   $medians — named numeric vector of column medians used for imputation.
#'              Always computed from `df` when training_medians is NULL;
#'              equals training_medians otherwise (passed through unchanged for
#'              bookkeeping).

build_features <- function(
    df,
    top_predictors   = NULL,
    training_medians = NULL,
    outcome_col      = "estimated_avoidable_deaths -- BNSSG"
) {

  # ---------------------------------------------------------------------------
  # 0. Input validation
  # ---------------------------------------------------------------------------

  stopifnot(
    "`df` must be a data frame"          = is.data.frame(df),
    "`df` must contain a `date` column"  = "date" %in% names(df),
    "outcome_col must exist in df"       = outcome_col %in% names(df)
  )

  if (is.null(top_predictors)) {
    stop(
      "top_predictors cannot be NULL. ",
      "Supply a character vector of your top ~20 predictor column names ",
      "(e.g. from Spearman |r| ranking computed on the full training set)."
    )
  }

  missing_preds <- base::setdiff(top_predictors, names(df))
  if (length(missing_preds) > 0) {
    warning(
      length(missing_preds), " top_predictors not found in df and will be skipped: ",
      paste(missing_preds, collapse = ", ")
    )
    top_predictors <- base::intersect(top_predictors, names(df))
  }

  # Ensure chronological order — critical for lag correctness
  df <- df %>% arrange(date)

  n <- nrow(df)

  # ---------------------------------------------------------------------------
  # 1. OUTCOME LAG FEATURES
  #
  # Only lags ≥ 3 are safe due to the 3-day reporting lag on the outcome.
  # lag3 = most recent observable value of the outcome at prediction time D.
  # lag7 = same weekday last week (aligns with your ACF peak at lag 7).
  # lag14 = same weekday two weeks ago (second-order weekly anchor).
  # Momentum = recent trend direction (rising vs. falling).
  # ---------------------------------------------------------------------------

  outcome_sym <- sym(outcome_col)

  df <- df %>%
    mutate(
      # Individual lags
      outcome_lag3  = lag(!!outcome_sym, 3),
      outcome_lag4  = lag(!!outcome_sym, 4),
      outcome_lag5  = lag(!!outcome_sym, 5),
      outcome_lag6  = lag(!!outcome_sym, 6),
      outcome_lag7  = lag(!!outcome_sym, 7),   # same weekday last week
      outcome_lag8  = lag(!!outcome_sym, 8),
      outcome_lag9  = lag(!!outcome_sym, 9),
      outcome_lag14 = lag(!!outcome_sym, 14),  # same weekday two weeks ago

      # Rolling mean: lags 3–5 (very recent observable window)
      outcome_roll3 = rowMeans(
        cbind(outcome_lag3, outcome_lag4, outcome_lag5),
        na.rm = FALSE
      ),

      # Rolling mean: lags 3–9 (full weekly observable window)
      outcome_roll7 = rowMeans(
        cbind(
          outcome_lag3, outcome_lag4, outcome_lag5,
          outcome_lag6, outcome_lag7, outcome_lag8, outcome_lag9
        ),
        na.rm = FALSE
      ),

      # Momentum: direction of recent observable trend
      # Positive = outcome has been rising; negative = falling
      outcome_momentum = outcome_lag3 - outcome_lag6
    )

  # ---------------------------------------------------------------------------
  # 2. CALENDAR AND TEMPORAL FEATURES
  #
  # Encodes day-of-week effects and annual seasonality.
  # Sine/cosine encoding wraps the year correctly (day 365 ≈ day 1).
  # Binary DOW flags give tree models easy splits for the strongest effects.
  # ---------------------------------------------------------------------------

  df <- df %>%
    mutate(
      # Day of week
      dow         = wday(date, week_start = 1),          # 1 = Mon, 7 = Sun
      dow_factor  = factor(wday(date, label = TRUE, week_start = 1),
                           levels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")),

      # Binary DOW flags for the extreme effects found in EDA
      is_monday   = as.integer(dow == 1),
      is_weekend  = as.integer(dow %in% c(6, 7)),

      # Annual seasonality via sine-cosine pair
      yday        = yday(date),
      yday_sin    = sin(2 * pi * yday / 365),
      yday_cos    = cos(2 * pi * yday / 365),

      # Month as integer and as factor (backup for tree models)
      month_int   = month(date),
      month_fct   = factor(month(date), levels = 1:12,
                           labels = month.abb),

      # Week of year (captures holiday effects etc.)
      week_of_year = isoweek(date),

      # Linear time index — useful if there is a long-term secular trend
      # Centred at first date so it starts at 0
      time_index  = as.integer(date - min(date))
    )

  # ---------------------------------------------------------------------------
  # 3. PREDICTOR LAG FEATURES
  #
  # Applied only to top_predictors (your top ~20 by Spearman |r|).
  # NOTE: your pipeline already applies a 1-day lag to all predictors
  # (yesterday's metrics aligned to today). So _lag1 here = 2 days ago
  # in calendar terms. This is correct — do not re-lag at submission.
  #
  # For each predictor X:
  #   X_lag1, X_lag2, X_lag3  — individual recent lags
  #   X_roll3                 — mean of lags 1–3 (short-term level)
  #   X_roll7                 — mean of lags 1–7 (weekly level)
  # ---------------------------------------------------------------------------

  for (pred in top_predictors) {
    pred_sym <- sym(pred)

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
        !!l1 := lag(!!pred_sym, 1),
        !!l2 := lag(!!pred_sym, 2),
        !!l3 := lag(!!pred_sym, 3),
        !!l4 := lag(!!pred_sym, 4),
        !!l5 := lag(!!pred_sym, 5),
        !!l6 := lag(!!pred_sym, 6),
        !!l7 := lag(!!pred_sym, 7)
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
      # Drop intermediate lags 4–7 (keep only lag1–3 and the two rolls)
      # to avoid excessive collinearity; retain them in rolls
      select(-all_of(c(l4, l5, l6, l7)))
  }

  # ---------------------------------------------------------------------------
  # 4. DERIVED CLINICAL FEATURES
  #
  # Hand-crafted aggregates based on clinical logic and EDA findings.
  # Kept intentionally minimal — trees learn interactions, and N=1,111 is small.
  # ---------------------------------------------------------------------------

  # Column name lookup helpers (partial match to handle naming variation)
  find_col <- function(df, pattern) {
    hits <- grep(pattern, names(df), value = TRUE, ignore.case = TRUE)
    if (length(hits) == 0) return(NULL)
    hits[1]  # take first match
  }

  dta_bri <- find_col(df, "DTA.*BRI|No.*DTA.*BRI")
  dta_nbt <- find_col(df, "DTA.*NBT|No.*DTA.*NBT")
  dta_wgh <- find_col(df, "DTA.*WGH|No.*DTA.*WGH")
  ed_12hr  <- find_col(df, "12.*hr.*ED|ED.*12.*hr|>12")

  # System-level delayed transfer pressure (sum across hospitals)
  if (!is.null(dta_bri) && !is.null(dta_nbt) && !is.null(dta_wgh)) {
    df <- df %>%
      mutate(
        total_DTA = rowSums(
          cbind(.data[[dta_bri]], .data[[dta_nbt]], .data[[dta_wgh]]),
          na.rm = TRUE  # sum with na.rm: if one hospital missing, use others
        ),
        total_DTA = if_else(
          is.na(.data[[dta_bri]]) & is.na(.data[[dta_nbt]]) & is.na(.data[[dta_wgh]]),
          NA_real_,   # all three missing → true NA, not 0
          total_DTA
        )
      )
  } else {
    warning("Could not find DTA columns for BRI/NBT/WGH — total_DTA not created.")
    df <- df %>% mutate(total_DTA = NA_real_)
  }

  # Sustained ED pressure: rolling 3-day mean of ED >12hr metric (lagged)
  if (!is.null(ed_12hr)) {
    ed_sym <- sym(ed_12hr)
    df <- df %>%
      mutate(
        ed_12hr_lag1 = lag(!!ed_sym, 1),
        ed_12hr_lag2 = lag(!!ed_sym, 2),
        ed_12hr_lag3 = lag(!!ed_sym, 3),
        ed_pressure_roll3 = rowMeans(
          cbind(ed_12hr_lag1, ed_12hr_lag2, ed_12hr_lag3),
          na.rm = FALSE
        )
      ) %>%
      select(-ed_12hr_lag1, -ed_12hr_lag2, -ed_12hr_lag3)
  } else {
    warning("Could not find ED >12hr column — ed_pressure_roll3 not created.")
    df <- df %>% mutate(ed_pressure_roll3 = NA_real_)
  }

  # ---------------------------------------------------------------------------
  # 5. MISSINGNESS INDICATORS
  #
  # For each top predictor and its lag columns, create a binary _missing flag
  # BEFORE imputation. A missing report can itself be informative (e.g. a
  # hospital system under stress may fail to report).
  # ---------------------------------------------------------------------------

  # Columns to flag: the lag columns we created for top predictors
  lag_suffixes  <- c("_lag1", "_lag2", "_lag3", "_roll3", "_roll7")
  cols_to_flag  <- c(
    outer(top_predictors, lag_suffixes, paste0),  # predictor lag cols
    "outcome_lag3", "total_DTA", "ed_pressure_roll3"
  )
  cols_to_flag  <- base::intersect(cols_to_flag, names(df))  # keep only those that exist

  for (col in cols_to_flag) {
    flag_name    <- paste0(col, "_missing")
    df[[flag_name]] <- as.integer(is.na(df[[col]]))
  }

  # ---------------------------------------------------------------------------
  # 6. IMPUTATION
  #
  # Median imputation on all predictor and feature columns (NOT the outcome).
  # If training_medians is provided (validation/assessment fold), use those.
  # If NULL (training fold), compute from current data and return them.
  #
  # Rows where outcome_lag3 is NA are dropped — these are early rows that
  # cannot safely be used for training or evaluation.
  # ---------------------------------------------------------------------------

  # Columns to impute: everything except date, the raw outcome, and _missing flags
  do_not_impute <- c(
    "date",
    outcome_col,
    names(df)[grepl("_missing$", names(df))],
    "dow_factor", "month_fct"  # factors handled separately
  )

  numeric_feature_cols <- names(df)[
    sapply(df, is.numeric) & !names(df) %in% do_not_impute
  ]

  if (is.null(training_medians)) {
    # Training fold: compute medians from this data
    medians_out <- sapply(df[numeric_feature_cols], median, na.rm = TRUE)
  } else {
    # Validation/assessment fold: use supplied training medians
    # Only impute columns that exist in both df and training_medians
    medians_out <- training_medians
  }

  # Apply imputation
  for (col in numeric_feature_cols) {
    med <- medians_out[col]
    if (!is.null(med) && !is.na(med)) {
      df[[col]] <- if_else(is.na(df[[col]]), med, df[[col]])
    }
  }

  # ---------------------------------------------------------------------------
  # 7. DROP ROWS WITH MISSING OUTCOME LAG (cannot train or evaluate on these)
  # ---------------------------------------------------------------------------

  n_before <- nrow(df)
  df <- df %>% filter(!is.na(outcome_lag3))
  n_dropped <- n_before - nrow(df)

  if (n_dropped > 0) {
    message(n_dropped, " rows dropped due to NA outcome_lag3 (expected for early rows).")
  }

  # ---------------------------------------------------------------------------
  # 8. SELECT AND ORDER OUTPUT COLUMNS
  #
  # Return a clean tibble with date first, then outcome (for reference),
  # then all engineered features. The raw predictor columns (non-lag) are
  # dropped to avoid target leakage from same-day values.
  # ---------------------------------------------------------------------------


  # Identify engineered feature columns
  raw_pred_cols <- base::setdiff(
    top_predictors,
    names(df)[grepl("_lag|_roll|_missing|_momentum|total_DTA|ed_pressure", names(df))]
  )

  # Keep: date, outcome, all engineered features; drop raw same-day predictors
  cols_to_keep <- c(
    "date",
    outcome_col,
    # Outcome lags
    grep("^outcome_", names(df), value = TRUE),
    # Calendar features
    c("dow", "dow_factor", "is_monday", "is_weekend",
      "yday", "yday_sin", "yday_cos",
      "month_int", "month_fct", "week_of_year", "time_index"),
    # Predictor lags and rolls
    names(df)[grepl(paste0("^(", paste(top_predictors, collapse="|"), ")_"), names(df))],
    # Derived clinical features
    c("total_DTA", "ed_pressure_roll3"),
    # Missingness flags
    names(df)[grepl("_missing$", names(df))]
  )

  # Deduplicate and keep only columns that actually exist
  cols_to_keep <- unique(base::intersect(cols_to_keep, names(df)))

  df_out <- df %>% select(all_of(cols_to_keep))

  # ---------------------------------------------------------------------------
  # 9. RETURN
  # ---------------------------------------------------------------------------

  message(
    "build_features() complete. ",
    "Rows: ", nrow(df_out), " | ",
    "Feature columns: ", ncol(df_out) - 2, " (excl. date and outcome)"
  )

  list(
    data    = df_out,
    medians = medians_out
  )
}


# =============================================================================
# CODE RUN
# =============================================================================

# Load saved dataset from EDA
df <- readRDS("data/modeling_data_clean.rds")
#
# Load top 20 predictors
top20 <- readRDS("data/top20_vars.rds")
#
# Build features on full training data (development period only)
result <- build_features(df, top_predictors = top20)
features     <- result$data
train_medians <- result$medians
#
# # 4. Inspect output
glimpse(features)
names(features)
#
# # 5. On a validation fold (pass training medians to prevent leakage):
# result_val <- build_features(df_val, top_predictors = top20,
#                              training_medians = train_medians)
# features_val <- result_val$data

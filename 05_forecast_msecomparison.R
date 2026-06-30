# =============================================================================
# SPHERE-PPL NHS Forecasting Competition
# 05_forecast.R
# =============================================================================
#
# READING GUIDE
# =============================================================================
# Read this before running the script. It explains the libraries, method
# choices, and structure so the code itself is easier to follow.
#
# ---- LIBRARIES ----
#
# data.table    — used only for fread(), which reads CSV files 5-10x faster
#                 than read.csv(). We use it in Section 3 to load the raw
#                 16M-row dataset. After loading, we switch to dplyr for
#                 everything else.
#
# dplyr         — the primary data manipulation library. All filtering,
#                 grouping, mutating, and joining is done via dplyr pipes.
#                 We call base::setdiff() and base::intersect() explicitly
#                 in a few places because dplyr overrides those base functions
#                 with versions that behave differently on data frames.
#
# tidyr         — used for pivot_wider() in Section 3 when reshaping the
#                 raw long-format data to wide format (one column per metric).
#
# purrr         — used for map_dbl() and map_int() in the missing-data audit.
#                 These are type-safe versions of sapply() — they guarantee
#                 the output is a numeric or integer vector, which prevents
#                 subtle bugs when sapply() sometimes returns a list.
#
# lubridate     — used for date manipulation throughout. Functions like
#                 wday(), month(), yday(), isoweek() are from lubridate and
#                 handle edge cases better than base R date functions.
#
# rlang         — provides .data[[col]] syntax used inside build_features().
#                 .data is a pronoun that tells dplyr "look up this column
#                 name from the string variable `col`". The alternative is
#                 !!sym(col) which converts a string to a symbol and then
#                 unquotes it — same result, more complex syntax.
#
# ranger        — the Random Forest implementation. Faster than randomForest
#                 and supports the matrix interface (x = matrix, y = vector)
#                 which avoids formula parsing issues with special characters
#                 in column names.
#
# xgboost       — loaded but not used for prediction in this script (RF won
#                 both horizons). Kept in case you want to add XGBoost as
#                 a comparison or fallback.
#
# ---- METHOD CHOICES ----
#
# source("02_feature_eng.R")
#   Loads the build_features() function. Note that sourcing also executes
#   the run section at the bottom of that file, which re-creates
#   df_feat_raw.rds and df_folds.rds. This is harmless — the outputs are
#   identical to the saved versions. We source rather than copy-pasting
#   build_features() to guarantee the function is always in sync.
#
# Replicating 01_eda.R cleaning inline (Section 3)
#   We cannot simply source 01_eda.R because it filters to the development
#   period only (date <= 2025-09-30). The forecast script needs the FULL
#   dataset including the assessment period. So we replicate the same
#   cleaning steps — aggregation, pivoting, yesterday lag, negative value
#   handling, high-missing column drop — but without the date filter.
#
# as.matrix() for model prediction
#   ranger's predict() expects a data.frame, while xgboost expects an
#   xgb.DMatrix. We convert feat_cols to matrix form and wrap in
#   as.data.frame() for ranger. This is the same pattern as 03_modeling.R.
#
# base::intersect() and base::setdiff()
#   Used instead of dplyr's versions because dplyr's intersect() and
#   setdiff() are designed for data frame operations and behave differently
#   on character vectors. The base versions do simple set operations on
#   character vectors, which is what we need for column name matching.
#
# pmax(pred, 0) for clipping
#   Avoidable deaths cannot be negative. pmax() is a vectorised "parallel
#   maximum" — it compares each element to 0 and returns whichever is larger.
#   This clips any negative predictions to 0 in a single operation.
#
# ---- STRUCTURE ----
#
# Section 1: Setup — load libraries and saved objects from prior scripts
# Section 2: Refit RF on full development data (one model per horizon)
# Section 3: Load and clean full dataset (dev + assessment)
# Section 4: Build features for assessment period and impute
# Section 5: Generate rolling 10-day forecasts
# Section 6: Validate and save submission files
#
# Sections 1-2 can run immediately (before assessment data release).
# Sections 3-6 require the assessment dataset (released 6 June 2026).
#
# DEPENDENCIES:
#   01_eda.R             — must have been run at least once (produces saved
#                          objects and scripts/aggregation_map.R)
#   02_feature_eng.R     — sourced for build_features() function
#   03_modeling.R        — must have been run (produces results/model_summary.rds)
#   scripts/aggregation_map.R — aggregation rules for raw data
#
# OUTPUTS:
#   submission/pred_matrix.csv   — 173 rows x 11 columns (forecast_id + 10 horizons)
#   submission/pred_matrix.rds   — same data as RDS for reproducibility
# =============================================================================

library(data.table)
library(dplyr)
library(tidyr)
library(purrr)
library(lubridate)
library(rlang)
library(ranger)
library(xgboost)

# =============================================================================
# SECTION 1: SETUP
# =============================================================================
# Load saved objects from prior scripts. These give us:
#   - df_feat_raw:     the development feature matrix (927 rows x 147 cols)
#   - core_predictors: the 10 curated predictor names
#   - model_summary:   contains feat_cols (the column names used by models)
#
# We also source 02_feature_eng.R to load the build_features() and
# make_targets() functions. The run section at the bottom of that file
# will execute — this is harmless and just re-saves identical outputs.
# =============================================================================

# Source build_features() from the feature engineering script
source("scripts/02_feature_eng.R")

# Load saved objects
df_feat_raw     <- readRDS("data/df_feat_raw.rds")
core_predictors <- readRDS("data/core_predictors.rds")
model_summary   <- readRDS("results/model_summary.rds")

feat_cols   <- model_summary$feat_cols
outcome_col <- "estimated_avoidable_deaths -- BNSSG"

# Create submission directory if it doesn't exist
if (!dir.exists("submission")) dir.create("submission")

message("Setup complete.")
message("  Development data: ", nrow(df_feat_raw), " rows, ", ncol(df_feat_raw), " cols")
message("  Feature columns:  ", length(feat_cols))
message("  Core predictors:  ", length(core_predictors))

# =============================================================================
# SECTION 2: REFIT RF ON FULL DEVELOPMENT DATA
# =============================================================================
# In 03_modeling.R, models were trained on CV fold training slices (subsets).
# For submission, we refit on ALL 927 development rows to use maximum data.
#
# Steps:
#   1. Compute imputation medians from the full development feature matrix
#   2. Apply median imputation to fill NAs
#   3. Create target columns (lead of outcome)
#   4. For each horizon h=1:10, fit one RF model on complete rows
#
# The imputation medians from this step are saved and reused in Section 4
# to impute assessment data. This mirrors the CV logic: training medians
# applied to validation data.
#
# RF hyperparameters are the tuned values from the HPT analysis:
#   num.trees=500, mtry=floor(p/3), min.node.size=5
# =============================================================================

message("\n=== Section 2: Refitting RF on full development data ===")

# --- Step 1: Compute full-dataset imputation medians ---
# These medians represent the "training" distribution and will be applied
# to both development data (here) and assessment data (Section 4).

dev_medians <- sapply(df_feat_raw[feat_cols], median, na.rm = TRUE)

# --- Step 2: Apply imputation to development data ---
# We work on a copy to keep df_feat_raw untouched.

df_dev <- df_feat_raw

for (col in feat_cols) {
  med <- dev_medians[col]
  if (!is.na(med)) {
    df_dev[[col]] <- ifelse(is.na(df_dev[[col]]), med, df_dev[[col]])
  }
}

# Verify no NAs remain in feature columns
na_remaining <- sum(sapply(df_dev[feat_cols], function(x) sum(is.na(x))))
if (na_remaining > 0) {
  warning(na_remaining, " NAs remain after imputation — check feat_cols.")
} else {
  message("  Imputation complete. No NAs in feature columns.")
}

# --- Step 3: Create target columns ---
# make_targets() adds target_h1 through target_h10 via lead(outcome, h).
# These are only needed for training — they're the labels the model learns from.

df_dev <- make_targets(df_dev, outcome_col = outcome_col)

# --- Step 4: Fit one RF model per horizon ---
# RF hyperparameters (from HPT tuning)
rf_num_trees     <- 500
rf_mtry          <- floor(length(feat_cols) / 3)
rf_min_node_size <- 5

message("  RF params: num.trees=", rf_num_trees,
        ", mtry=", rf_mtry,
        ", min.node.size=", rf_min_node_size)

# Feature matrix for training
X_dev <- as.matrix(df_dev[feat_cols])

# Storage for fitted models
final_rf <- vector("list", 10)

for (h in 1:10) {
  
  target_col <- paste0("target_h", h)
  
  # Rows where target is NA (last h rows of dataset) cannot be used for training
  complete  <- !is.na(df_dev[[target_col]])
  y_train   <- df_dev[[target_col]][complete]
  X_train_h <- X_dev[complete, , drop = FALSE]
  
  final_rf[[h]] <- ranger::ranger(
    x             = X_train_h,
    y             = y_train,
    num.trees     = rf_num_trees,
    mtry          = rf_mtry,
    min.node.size = rf_min_node_size,
    seed          = 42
  )
  
  message("  Horizon ", h, ": trained on ", sum(complete), " rows")
}

message("  All 10 RF models fitted.")

# Save medians and models for reproducibility
saveRDS(
  list(final_rf = final_rf, dev_medians = dev_medians, feat_cols = feat_cols),
  "results/forecast_models.rds"
)

message("  Models and medians saved to results/forecast_models.rds")

# =============================================================================
# SECTION 3: LOAD AND CLEAN FULL DATASET (DEV + ASSESSMENT)
# =============================================================================
# This section replicates the cleaning from 01_eda.R but WITHOUT the date
# filter, so the result includes both development and assessment periods.
#
# The assessment dataset is released on 6 June 2026 by replacing the -9999
# dummy values in the original CSV with real data. So we reload the same
# CSV file — it now contains real values for Oct 2025 - Mar 2026.
#
# Steps (mirroring 01_eda.R):
#   1. Load raw CSV with fread() for speed
#   2. Aggregate sub-daily records to daily using the aggregation map
#   3. Pivot from long to wide format (one column per metric)
#   4. Apply 1-day lag to "YESTERDAY" metrics
#   5. Replace negative values (dummy -9999) with NA
#   6. Drop columns with >50% missing (using dev-period thresholds)
#
# The output is modeling_data_full — same structure as modeling_data_clean
# from 01_eda.R but covering the full time range.
# =============================================================================

message("\n=== Section 3: Loading and cleaning full dataset ===")

# --- Step 1: Load raw data ---
# raw_data <- data.table::fread("data/turingAI_forecasting_challenge_dataset.csv")
# raw_data[, date := as.Date(dt)]

dev_raw    <- data.table::fread("data/turingAI_forecasting_challenge_dataset.csv")
assess_raw <- data.table::fread("data/turingAI_forecasting_challenge_validation_dataset.csv")

dev_raw[, date := as.POSIXct(dt, origin = "1970-01-01", tz = "UTC") |> as.Date()]
assess_raw[, date := as.POSIXct(dt, origin = "1970-01-01", tz = "UTC") |> as.Date()]

dev_raw <- dev_raw[date < as.Date("2025-10-01")]

raw_data <- rbind(dev_raw, assess_raw)

message("  Raw data loaded: ", nrow(raw_data), " rows")

max(dev_raw$date)
min(dev_raw$date)

max(assess_raw$date)
min(assess_raw$date)

# --- Step 2: Aggregate to daily ---
# The aggregation map defines how sub-daily records (e.g. every 15 minutes)
# are collapsed to daily values — mean, max, sum, or last value depending
# on the metric type. This is sourced from the same file used by 01_eda.R.

source("scripts/aggregation_map.R")

raw_data <- raw_data %>%
  left_join(aggregation_map, by = "metric_name")

daily_data <- raw_data %>%
  group_by(date, metric_name, coverage_label, variable_type) %>%
  summarise(
    value = case_when(
      first(agg_method) == "mean" ~ mean(value, na.rm = TRUE),
      first(agg_method) == "max"  ~ max(value, na.rm = TRUE),
      first(agg_method) == "sum"  ~ sum(value, na.rm = TRUE),
      first(agg_method) == "last" ~ last(value[!is.na(value)]),
      first(agg_method) == "asis" ~ first(value[!is.na(value)]),
      TRUE ~ mean(value, na.rm = TRUE)
    ),
    .groups = "drop"
  )

# --- Step 3: Pivot to wide format ---
# Features and outcome are pivoted separately then joined. Column names
# follow the pattern "metric_name -- coverage_label" to match what
# core_predictors and build_features() expect.

features_daily <- daily_data %>% filter(variable_type == "feature")
outcome_daily  <- daily_data %>% filter(variable_type == "outcome")

features_wide <- features_daily %>%
  pivot_wider(
    names_from  = c(metric_name, coverage_label),
    values_from = value,
    id_cols     = date,
    names_sep   = " -- "
  )

outcome_wide <- outcome_daily %>%
  pivot_wider(
    names_from  = c(metric_name, coverage_label),
    values_from = value,
    id_cols     = date,
    names_sep   = " -- "
  )

modeling_data_full <- features_wide %>%
  left_join(outcome_wide, by = "date") %>%
  arrange(date)

message("  Wide format: ", nrow(modeling_data_full), " rows, ",
        ncol(modeling_data_full), " cols")
message("  Date range:  ", min(modeling_data_full$date), " to ",
        max(modeling_data_full$date))

# --- Step 4: Apply YESTERDAY lags ---
# Metrics with "YESTERDAY" in their name report yesterday's value as today's
# record. We shift them back by 1 day so they align correctly with the date.
# This is identical to what 01_eda.R does.

modeling_data_full <- modeling_data_full %>%
  arrange(date) %>%
  mutate(across(
    contains("YESTERDAY") | contains("Yesterday"),
    ~ lag(., 1)
  ))

# --- Step 5: Replace negative values with NA ---
# The raw data uses -9999 as a dummy for missing/unavailable periods.
# Any negative value in a health metric is clinically implausible, so
# we replace all negatives with NA.

modeling_data_full <- modeling_data_full %>%
  mutate(across(where(is.numeric), ~ if_else(. < 0, NA_real_, .)))

# --- Step 6: Drop high-missing columns ---
# We compute missingness on the DEVELOPMENT period only (up to Sep 30 2025)
# and apply the same column filter to the full dataset. This ensures the
# columns match what the model was trained on.

dev_rows <- modeling_data_full %>% filter(date <= as.Date("2025-09-30"))

missing_pct <- purrr::map_dbl(
  dev_rows %>% select(-date),
  ~ mean(is.na(.)) * 100
)

cols_low_missing <- names(missing_pct)[missing_pct <= 50]
# Always keep the outcome column even if it has missing values
cols_low_missing <- unique(c(cols_low_missing, outcome_col))

modeling_data_full <- modeling_data_full %>%
  select(date, all_of(base::intersect(cols_low_missing, names(.))))

message("  After dropping >50% missing: ", ncol(modeling_data_full), " cols")

# =============================================================================
# SECTION 4: BUILD FEATURES FOR ASSESSMENT PERIOD AND IMPUTE
# =============================================================================
# Run build_features() on the FULL dataset so lag features for the first
# assessment rows can reference development-period values.
#
# Then split into development and assessment periods and impute assessment
# features using the development medians from Section 2.
#
# Why not recompute medians from the assessment data?
#   The models were trained assuming development-period distributions.
#   Assessment-period medians could differ (e.g. winter vs summer patterns)
#   and using them would create a mismatch between what the model learned
#   and what it receives as input. This is the same principle as using
#   training-fold medians for validation data in CV.
# =============================================================================

message("\n=== Section 4: Building features and imputing ===")

# Build features on the full dataset
# build_features() returns list(data = tibble). No imputation inside it.
result_full  <- build_features(modeling_data_full, core_predictors = core_predictors)
df_feat_full <- result_full$data

message("  Full feature matrix: ", nrow(df_feat_full), " rows, ",
        ncol(df_feat_full), " cols")

# Define assessment period boundaries
assess_start <- as.Date("2025-10-01")
assess_end <- as.Date("2026-02-17")  # was 2026-03-31

# The last development date is Sep 30 — we need it as the first forecast origin
dev_end <- as.Date("2025-09-30")

# Split: keep development rows for reference, extract assessment rows
df_assess <- df_feat_full %>% filter(date >= dev_end)
# Note: we include dev_end (Sep 30) because it's the origin for the first
# forecast period (predicting Oct 1-10).

message("  Assessment feature rows (from Sep 30): ", nrow(df_assess))

# Impute assessment features using development medians
# Only impute columns that exist in both the assessment data and medians vector

impute_cols <- base::intersect(feat_cols, names(df_assess))

for (col in impute_cols) {
  med <- dev_medians[col]
  if (!is.null(med) && !is.na(med)) {
    df_assess[[col]] <- ifelse(is.na(df_assess[[col]]), med, df_assess[[col]])
  }
}

# Check for remaining NAs
na_check <- sum(sapply(df_assess[base::intersect(feat_cols, names(df_assess))],
                       function(x) sum(is.na(x))))
if (na_check > 0) {
  warning(na_check, " NAs remain in assessment features after imputation.")
} else {
  message("  Assessment imputation complete. No NAs in feature columns.")
}

# Columns that exist in feat_cols but not in assessment data
missing_cols <- base::setdiff(feat_cols, names(df_assess))
if (length(missing_cols) > 0) {
  warning(length(missing_cols), " feat_cols not found in assessment data: ",
          paste(head(missing_cols, 5), collapse = ", "))
}

# =============================================================================
# SECTION 5: GENERATE ROLLING 10-DAY FORECASTS
# =============================================================================
# The competition requires 173 rolling forecast periods over the assessment
# window (Oct 1 - Mar 31 = 182 days, minus 9 for the last incomplete window).
#
# For each period p = 1:173:
#   - Forecast origin D is the day BEFORE the 10-day window
#   - Features for date D are extracted from df_assess
#   - The 10 pre-fitted RF models predict D+1 through D+10
#
# Period 1:   origin = Sep 30, predicts Oct 1-10
# Period 2:   origin = Oct 1,  predicts Oct 2-11
# ...
# Period 173: origin = Mar 21, predicts Mar 22-31
#
# The submission format matches the example: one row per period, columns
# forecast_id and day_1 through day_10.
# =============================================================================

message("\n=== Section 5: Generating rolling forecasts ===")

# Define the 173 origin dates
# First origin: Sep 30 (predicts Oct 1-10)
# Last origin:  Mar 21 (predicts Mar 22-31)
first_origin <- as.Date("2025-09-30")
last_origin <- as.Date("2026-02-07")  # was 2026-03-21
origin_dates <- seq(first_origin, last_origin, by = "day")

n_periods <- length(origin_dates)
message("  Number of forecast periods: ", n_periods)

if (n_periods != 131) { # was 173
  warning("Expected 131 periods but got ", n_periods, ". Check origin date range.")
}

# Prepare feature matrix for prediction
# Only use columns in feat_cols to match what models were trained on
avail_feat_cols <- base::intersect(feat_cols, names(df_assess))

# Prediction storage: 173 rows x 10 columns
pred_matrix <- matrix(NA_real_, nrow = n_periods, ncol = 10)

for (p in seq_len(n_periods)) {
  
  origin <- origin_dates[p]
  
  # Extract the feature row for this origin date
  row_idx <- which(df_assess$date == origin)
  
  if (length(row_idx) == 0) {
    warning("No feature row found for origin date ", origin, " — skipping period ", p)
    next
  }
  
  if (length(row_idx) > 1) {
    warning("Multiple rows for ", origin, " — using first.")
    row_idx <- row_idx[1]
  }
  
  # Build feature vector as a data.frame (ranger expects data.frame for predict)
  X_origin <- as.data.frame(
    as.matrix(df_assess[row_idx, avail_feat_cols, drop = FALSE])
  )
  
  # Predict each horizon using the corresponding RF model
  for (h in 1:10) {
    pred_matrix[p, h] <- predict(final_rf[[h]], data = X_origin)$predictions
  }
}

# Clip negative predictions to 0
# Avoidable deaths cannot be negative. pmax() is vectorised: for each element,
# it returns the larger of the prediction or 0.
pred_matrix <- pmax(pred_matrix, 0)

#!!!addition!!!

pred_matrix_raw <- pred_matrix 

#!!!

message("  Predictions generated for ", sum(!is.na(pred_matrix[, 1])),
        " of ", n_periods, " periods.")

# =============================================================================
# BIAS CORRECTION PART 2
# =============================================================================
# Loads both correction vectors and applies them to pred_matrix.
#
# pred_matrix layout (as built in Section 5):
#   - 173 rows, one per forecast origin date
#   - 10 columns: day_1 through day_10
#   - origin_dates: Date vector of length 173, aligned row-for-row
#
# APPLICATION ORDER
# -----------------
#   Step 1: DOW correction — per-cell, based on target date of each horizon
#     For origin_dates[p], horizon h lands on origin_dates[p] + h days.
#     Look up wday() of that target date → apply dow_correction_vec[dow].
#
#   Step 2: Monthly correction — per-row, based on origin date
#     Look up month() of origin_dates[p] → add monthly_correction_vec[month]
#     to all 10 cells in that row.
#
#   Step 3: Clip to 0 — corrections can push predictions below 0.
# =============================================================================

# Load correction vectors computed in 05_residual_analysis.R
dow_correction_vec     <- readRDS("results/dow_bias_corrections.rds")
monthly_correction_vec <- readRDS("results/monthly_bias_corrections.rds")

message("\n=== Applying bias corrections ===")
message("  DOW corrections available for days:    ",
        paste(names(dow_correction_vec), collapse = ", "))
message("  Monthly corrections available for months: ",
        paste(names(monthly_correction_vec), collapse = ", "))

pred_matrix_corrected <- pred_matrix   # work on a copy

# --------------------------------------------------------------------------
# Step 1: DOW correction — applied at target date, per horizon
# --------------------------------------------------------------------------
# For each origin p and each horizon h (1:10), the target date is
# origin_dates[p] + h. We look up the DOW of that date and apply the
# corresponding correction to pred_matrix[p, h].
# --------------------------------------------------------------------------

for (p in seq_len(nrow(pred_matrix_corrected))) {
  for (h in 1:10) {
    
    target_date <- origin_dates[p] + h
    target_dow  <- as.character(wday(target_date, week_start = 1))  # "1"–"7"
    dow_corr    <- dow_correction_vec[target_dow]
    
    if (!is.na(dow_corr)) {
      pred_matrix_corrected[p, h] <- pred_matrix_corrected[p, h] + dow_corr
    }
    # If DOW not found (should not happen), leave prediction unchanged
  }
}

message("  DOW correction applied at target date for each horizon.")

# --------------------------------------------------------------------------
# Step 2: Monthly correction — applied at origin date, across all horizons
# --------------------------------------------------------------------------

for (p in seq_len(nrow(pred_matrix_corrected))) {
  
  origin_month   <- as.character(month(origin_dates[p]))
  monthly_corr   <- monthly_correction_vec[origin_month]
  
  if (!is.na(monthly_corr)) {
    pred_matrix_corrected[p, ] <- pred_matrix_corrected[p, ] + monthly_corr
  }
  # If month not observed in CV, leave predictions unchanged (conservative fallback)
}

message("  Monthly correction applied at origin date across all horizons.")

# --------------------------------------------------------------------------
# Step 3: Clip to 0 — both corrections combined could push below 0
# --------------------------------------------------------------------------

pred_matrix_corrected <- pmax(pred_matrix_corrected, 0)

# --------------------------------------------------------------------------
# Diagnostics: inspect the total shift from corrections
# --------------------------------------------------------------------------

mean_before <- mean(pred_matrix,           na.rm = TRUE)
mean_after  <- mean(pred_matrix_corrected, na.rm = TRUE)

message("  Mean prediction before correction: ", round(mean_before, 4))
message("  Mean prediction after correction:  ", round(mean_after,  4))
message("  Mean shift (total): ",                round(mean_after - mean_before, 4))

# Per-DOW mean shift — sanity check that Saturday moved more than Tuesday
dow_shift_check <- sapply(1:10, function(h) {
  mean(pred_matrix_corrected[, h] - pred_matrix[, h], na.rm = TRUE)
})
names(dow_shift_check) <- paste0("h", 1:10)
message("  Mean shift by horizon (h1–h10):")
print(round(dow_shift_check, 4))

# --------------------------------------------------------------------------
# Diagnostic plot: mean correction by day of week across all 173 periods
# --------------------------------------------------------------------------
# For each of the 10 horizons, compute the mean shift (corrected - raw).
# Then label each horizon with the day of week it most commonly lands on.
# This lets you visually confirm that Saturday-landing horizons shift more
# than midweek horizons — the key claim of the DOW correction.
#
# Note: a given horizon (e.g. h1) lands on different days of the week
# across the 173 periods, so we take the mean shift per horizon. To show
# the DOW pattern more directly, we also plot mean shift grouped by the
# actual target DOW across all 173 × 10 = 1730 cells.
# --------------------------------------------------------------------------


#!!!changes made to line 666: pred_matrix_raw instead of pred_matrix
# Build a long-format data frame of all corrections applied
correction_long <- expand.grid(period = 1:nrow(pred_matrix), horizon = 1:10) %>%
  as_tibble() %>%
  mutate(
    origin_date = origin_dates[period],
    target_date = origin_date + horizon,
    target_dow  = wday(target_date, week_start = 1),
    dow_label   = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")[target_dow],
    raw_pred    = as.vector(pred_matrix_raw),        # <-- use saved raw
    corr_pred   = as.vector(pred_matrix_corrected),
    shift       = corr_pred - raw_pred
  )

# Plot 1: mean shift by target day of week
# This is the cleanest way to show the DOW correction is working —
# each bar represents the average correction applied to predictions
# landing on that day, aggregated across all periods and horizons.
correction_long %>%
  mutate(dow_label = factor(dow_label,
                            levels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun"))) %>%
  group_by(dow_label) %>%
  summarise(mean_shift = mean(shift), .groups = "drop") %>%
  ggplot(aes(x = dow_label, y = mean_shift, fill = mean_shift)) +
  geom_col() +
  scale_fill_gradient(low = "steelblue", high = "tomato") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title    = "Mean Correction Applied by Target Day of Week",
    subtitle = "Averaged across all 131 forecast periods | DOW + monthly corrections combined",
    x        = NULL,
    y        = "Mean correction (corrected − raw)",
    fill     = "Correction"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# Plot 2: mean raw vs corrected prediction by target DOW
# Shows not just the shift but what the predictions actually look like
# before and after — useful for the report to show the correction
# brings predictions closer to observed levels.
correction_long %>%
  mutate(dow_label = factor(dow_label,
                            levels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun"))) %>%
  group_by(dow_label) %>%
  summarise(
    mean_raw  = mean(raw_pred),
    mean_corr = mean(corr_pred),
    .groups   = "drop"
  ) %>%
  pivot_longer(cols = c(mean_raw, mean_corr),
               names_to  = "version",
               values_to = "mean_pred") %>%
  mutate(version = if_else(version == "mean_raw", "Raw", "Corrected")) %>%
  ggplot(aes(x = dow_label, y = mean_pred, fill = version)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c("Raw" = "grey60", "Corrected" = "steelblue")) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title    = "Raw vs Corrected Predictions by Target Day of Week",
    subtitle = "Mean across all 131 forecast periods",
    x        = NULL,
    y        = "Mean prediction",
    fill     = NULL
  ) +
  theme_minimal()

# Use the corrected matrix going forward into Section 6
pred_matrix <- pred_matrix_corrected

# =============================================================================
# SECTION 6: VALIDATE AND SAVE
# =============================================================================
# Sanity checks on predictions, then write submission files.
#
# Checks:
#   1. No NA predictions
#   2. All predictions >= 0 (post-clipping)
#   3. Correct number of rows (173)
#   4. Values in a plausible range (based on development outcome distribution)
#
# Output:
#   submission/pred_matrix.csv — competition submission format
#   submission/pred_matrix.rds — R object for reproducibility
# =============================================================================

message("\n=== Section 6: Validation and saving ===")

# --- Check 1: No NAs ---
n_na <- sum(is.na(pred_matrix))
if (n_na > 0) {
  warning("ALERT: ", n_na, " NA predictions found!")
} else {
  message("  Check 1 PASSED: No NA predictions.")
}

# --- Check 2: All non-negative ---
n_neg <- sum(pred_matrix < 0, na.rm = TRUE)
if (n_neg > 0) {
  warning("ALERT: ", n_neg, " negative predictions found after clipping!")
} else {
  message("  Check 2 PASSED: All predictions >= 0.")
}

# --- Check 3: Correct dimensions ---
if (nrow(pred_matrix) == 131 && ncol(pred_matrix) == 10) {  # was 173
  message("  Check 3 PASSED: Dimensions are 131 x 10.")
} else {
  warning("ALERT: Unexpected dimensions — ",
          nrow(pred_matrix), " x ", ncol(pred_matrix))
}

# --- Check 4: Plausible range ---
pred_min  <- min(pred_matrix,  na.rm = TRUE)
pred_max  <- max(pred_matrix,  na.rm = TRUE)
pred_mean <- mean(pred_matrix, na.rm = TRUE)

# Development outcome summary for comparison
dev_outcome <- df_feat_raw[[outcome_col]]
dev_mean    <- mean(dev_outcome, na.rm = TRUE)
dev_max     <- max(dev_outcome,  na.rm = TRUE)

message("  Prediction range: [", round(pred_min, 4), ", ",
        round(pred_max, 4), "]")
message("  Prediction mean:  ", round(pred_mean, 4))
message("  Development mean: ", round(dev_mean, 4),
        " | max: ", round(dev_max, 4))

if (pred_max > dev_max * 2) {
  warning("Some predictions exceed 2x the development maximum — inspect for outliers.")
}

# --- Build submission data frame ---
# Format: forecast_id (1:173), day_1 through day_10
pred_out <- as.data.frame(pred_matrix)
colnames(pred_out) <- paste0("day_", 1:10)
pred_out$forecast_id <- 1:n_periods

# Reorder: forecast_id first
pred_out <- pred_out[, c("forecast_id", paste0("day_", 1:10))]

# --- Save ---
write.csv(pred_out, "submission/pred_matrix.csv", row.names = FALSE)
saveRDS(pred_out,   "submission/pred_matrix.rds")

message("\n=== SUBMISSION FILES SAVED ===")
message("  submission/pred_matrix.csv  (", nrow(pred_out), " rows)")
message("  submission/pred_matrix.rds")

# --- Final summary ---
message("\n=== FORECAST SUMMARY ===")
message("  Forecast periods:  ", n_periods)
message("  Horizon range:     days 1-10")
message("  Model:             Random Forest (all horizons)")
message("  Prediction mean:   ", round(pred_mean, 4))
message("  Prediction min:    ", round(pred_min, 4))
message("  Prediction max:    ", round(pred_max, 4))
message("  Origin date range: ", first_origin, " to ", last_origin)
message("\nDone.")

#MSE calculations:

# Extract observed values from assessment data
observed <- modeling_data_full %>%
  filter(date >= as.Date("2025-10-01"), date <= as.Date("2026-02-17")) %>%
  select(date, observed = `estimated_avoidable_deaths -- BNSSG`) %>%
  arrange(date)

# Build observed matrix aligned to pred_matrix
obs_matrix <- matrix(NA_real_, nrow = n_periods, ncol = 10)

for (p in seq_len(n_periods)) {
  for (h in 1:10) {
    target_date <- origin_dates[p] + h
    obs_row <- observed$observed[observed$date == target_date]
    if (length(obs_row) == 1) {
      obs_matrix[p, h] <- obs_row
    }
  }
}

# MSE days 1-5
mse_1to5 <- mean((obs_matrix[, 1:5] - pred_matrix[, 1:5])^2, na.rm = TRUE)

# MSE days 6-10
mse_6to10 <- mean((obs_matrix[, 6:10] - pred_matrix[, 6:10])^2, na.rm = TRUE)

cat("MSE days 1-5:  ", round(mse_1to5,  4), "\n")
cat("MSE days 6-10: ", round(mse_6to10, 4), "\n")


# =============================================================================
# MSE COMPARISON: RAW vs BIAS-CORRECTED (assessment period)
# =============================================================================
# pred_matrix_raw was saved before corrections were applied (line ~510)
# obs_matrix is already built above
# Both are n_periods x 10 matrices aligned to origin_dates

# --- Raw (uncorrected) MSE ---
mse_raw_1to5  <- mean((obs_matrix[, 1:5]  - pred_matrix_raw[, 1:5])^2,  na.rm = TRUE)
mse_raw_6to10 <- mean((obs_matrix[, 6:10] - pred_matrix_raw[, 6:10])^2, na.rm = TRUE)

# --- Corrected MSE (already computed above, just aliased for clarity) ---
mse_corr_1to5  <- mse_1to5
mse_corr_6to10 <- mse_6to10

# --- Comparison table ---
mse_comparison_assess <- tibble::tibble(
  horizon_band    = c("Days 1-5", "Days 6-10"),
  mse_uncorrected = c(mse_raw_1to5,  mse_raw_6to10),
  mse_corrected   = c(mse_corr_1to5, mse_corr_6to10)
) %>%
  mutate(
    improvement     = mse_uncorrected - mse_corrected,
    improvement_pct = round(100 * improvement / mse_uncorrected, 1)
  )

message("\n=== Bias correction MSE comparison (assessment period) ===")
print(mse_comparison_assess)



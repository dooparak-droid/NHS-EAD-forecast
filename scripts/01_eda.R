# ============================================================
# SETUP
# ============================================================
rm(list = ls()) # clear your working environment
getwd() # make sure you're in the right project directory

# Import relevant libraries
library(data.table)
library(tidyverse)
library(readxl)

# ============================================================
# READ DATA
# ============================================================
dev_data <- fread("data/turingAI_forecasting_challenge_dataset.csv")
metadata <- fread("data/metric_metadata.csv")
metrics  <- read_xlsx("data/metric_details.xlsx")

glimpse(dev_data)
unique(dev_data$variable_type)

# ============================================================
# AGGREGATE TO DAILY
# ============================================================
dev_data[, date := as.Date(dt)]
source("scripts/aggregation_map.R")

dev_data <- dev_data %>%
  left_join(aggregation_map, by = "metric_name")

daily_data <- dev_data %>%
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

# ============================================================
# PIVOT WIDE
# ============================================================
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

modeling_data <- features_wide %>%
  left_join(outcome_wide, by = "date") %>%
  arrange(date)

# ============================================================
# IDENTIFY OUTCOME COLUMN
# ============================================================
outcome_cols <- names(modeling_data)[
  names(modeling_data) %in% names(outcome_wide) &
    names(modeling_data) != "date"
]
print(outcome_cols)

# ============================================================
# HANDLE YESTERDAY LAGS
# ============================================================
modeling_data <- modeling_data %>%
  arrange(date) %>%
  mutate(across(
    contains("YESTERDAY") | contains("Yesterday"),
    ~ lag(., 1)
  ))

# ============================================================
# STEP 1: CLEAN DUMMY VALUES & TRUNCATE
# ============================================================
modeling_data <- modeling_data %>%
  mutate(across(where(is.numeric), ~if_else(. < 0, NA_real_, .)))

sum(modeling_data == -9999, na.rm = TRUE)  # verify: should be 0

# Filter to Development Period only
modeling_data <- modeling_data %>%
  filter(date <= as.Date("2025-09-30"))

# ============================================================
# STEP 2: SANITY CHECKS
# ============================================================
dim(modeling_data)
names(modeling_data)[1:20]
range(modeling_data$date)

# ============================================================
# STEP 3: MISSING DATA AUDIT
# ============================================================
missing_summary <- data.frame(
  column   = names(modeling_data),
  n_miss   = map_int(modeling_data, ~sum(is.na(.))),
  pct_miss = map_dbl(modeling_data, ~mean(is.na(.)) * 100)
) %>%
  as_tibble() %>%
  arrange(desc(pct_miss)) %>%
  filter(column != "date")

cat("Total columns:", ncol(modeling_data) - 1, "\n")
cat("Columns with 0% missing:", sum(missing_summary$pct_miss == 0), "\n")
cat("Columns with >50% missing:", sum(missing_summary$pct_miss > 50), "\n")
cat("Columns with >90% missing:", sum(missing_summary$pct_miss > 90), "\n")

ggplot(missing_summary, aes(x = pct_miss)) +
  geom_histogram(binwidth = 5, fill = "steelblue", colour = "white") +
  labs(
    title = "Distribution of Missingness Across Features",
    x     = "% Missing",
    y     = "Number of columns"
  ) +
  theme_minimal()

high_missing <- missing_summary %>% filter(pct_miss > 50)
head(high_missing, 50)
nrow(high_missing)

# ============================================================
# STEP 4: DROP HIGH-MISSING COLUMNS
# ============================================================
cols_to_keep <- missing_summary %>%
  filter(pct_miss <= 50 | column == outcome_cols) %>%
  pull(column)

cols_to_keep <- unique(c("date", outcome_cols, cols_to_keep))

modeling_data_clean <- modeling_data %>%
  select(all_of(cols_to_keep))

cat("Columns before dropping:", ncol(modeling_data), "\n")
cat("Columns after dropping:", ncol(modeling_data_clean), "\n")

# ============================================================
# STEP 5: OUTCOME INSPECTION
# ============================================================
modeling_data_clean %>%
  filter(date <= as.Date("2025-09-30")) %>%
  ggplot(aes(x = date, y = .data[[outcome_cols]])) +
  geom_line(colour = "steelblue") +
  geom_smooth(method = "loess", colour = "red", se = FALSE) +
  labs(
    title    = "Daily Estimated Avoidable Deaths",
    subtitle = "Bristol NHS, March 2023 – September 2025",
    x        = NULL,
    y        = "Estimated avoidable deaths"
  ) +
  theme_minimal()

# ============================================================
# STEP 6: DAY OF WEEK EFFECT
# ============================================================
modeling_data_clean %>%
  filter(date <= as.Date("2025-09-30")) %>%
  mutate(dow = factor(weekdays(date),
                      levels = c("Monday","Tuesday","Wednesday",
                                 "Thursday","Friday","Saturday","Sunday"))) %>%
  group_by(dow) %>%
  summarise(
    mean_deaths   = mean(.data[[outcome_cols]], na.rm = TRUE),
    sd_deaths     = sd(.data[[outcome_cols]], na.rm = TRUE),
    median_deaths = median(.data[[outcome_cols]], na.rm = TRUE)
  ) %>%
  ggplot(aes(x = dow, y = mean_deaths)) +
  geom_col(fill = "steelblue") +
  geom_errorbar(aes(ymin = mean_deaths - sd_deaths,
                    ymax = mean_deaths + sd_deaths),
                width = 0.3) +
  labs(
    title    = "Mean Avoidable Deaths by Day of Week",
    subtitle = paste("Outcome analyzed:", outcome_cols),
    x        = NULL,
    y        = "Mean estimated avoidable deaths"
  ) +
  theme_minimal()

# ============================================================
# STEP 7: SEASONALITY
# ============================================================
modeling_data_clean %>%
  filter(date <= as.Date("2025-09-30")) %>%
  mutate(month = factor(format(date, "%b"), levels = month.abb)) %>%
  group_by(month) %>%
  summarise(mean_deaths = mean(.data[[outcome_cols]], na.rm = TRUE)) %>%
  ggplot(aes(x = month, y = mean_deaths)) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Mean Avoidable Deaths by Month",
    x     = NULL,
    y     = "Mean estimated avoidable deaths"
  ) +
  theme_minimal()

# ============================================================
# STEP 8: OUTCOME AUTOCORRELATION
# ============================================================
library(forecast)

outcome_series <- modeling_data_clean %>%
  filter(date <= as.Date("2025-09-30")) %>%
  arrange(date) %>%
  pull(.data[[outcome_cols]])

# Autocorrelation plot
acf(outcome_series, lag.max = 14, na.action = na.pass,
    main = "Autocorrelation of Daily Avoidable Deaths")

# Partial autocorrelation
pacf(outcome_series, lag.max = 14, na.action = na.pass,
     main = "Partial Autocorrelation of Daily Avoidable Deaths")

# ============================================================
# STEP 9: FEATURE OUTCOME AUOCORRELATION
# ============================================================
# Calculate correlation of each feature with the outcome
# Use development period only
dev_period <- modeling_data_clean %>%
  filter(date <= as.Date("2025-09-30")) %>%
  arrange(date)

outcome_vec <- dev_period %>% pull(.data[[outcome_cols]])

# Correlate all numeric features with outcome
feature_cols <- names(dev_period)[
  !names(dev_period) %in% c("date", outcome_cols)
]

cor_with_outcome <- map_dbl(feature_cols, function(col) {
  cor(dev_period[[col]], outcome_vec,
      use = "pairwise.complete.obs",
      method = "spearman")  # Spearman handles skewed data better
})

cor_summary <- data.frame(
  feature     = feature_cols,
  correlation = cor_with_outcome
) %>%
  arrange(desc(abs(correlation)))

# Top 20 most correlated features
head(cor_summary, 20)

# Plot top 20
cor_summary %>%
  slice_head(n = 20) %>%
  mutate(feature = str_trunc(feature, 50)) %>%  # truncate long names
  ggplot(aes(x = reorder(feature, abs(correlation)),
             y = correlation,
             fill = correlation > 0)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("tomato", "steelblue"),
                    labels = c("Negative", "Positive"),
                    name   = "Direction") +
  labs(
    title = "Top 20 Features Correlated with Avoidable Deaths",
    x     = NULL,
    y     = "Spearman correlation"
  ) +
  theme_minimal()

# ============================================================
# STEP 10: PLOT FEATURES AGANST OUTCOME OVER TIME
# ============================================================
# Take top 5 features by absolute correlation
top5_features <- cor_summary %>%
  slice_head(n = 5) %>%
  pull(feature)

# Plot each against outcome
dev_period %>%
  select(date, all_of(outcome_cols), all_of(top5_features)) %>%
  pivot_longer(-date, names_to = "variable", values_to = "value") %>%
  mutate(is_outcome = variable == outcome_cols) %>%
  ggplot(aes(x = date, y = value, colour = is_outcome)) +
  geom_line(alpha = 0.6) +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  scale_colour_manual(values = c("steelblue", "red"), guide = "none") +
  labs(
    title = "Top 5 Features vs Outcome Over Time",
    x     = NULL,
    y     = NULL
  ) +
  theme_minimal()

# ============================================================
# STEP 8: SAVE
# ============================================================
saveRDS(modeling_data_clean, "data/modeling_data_clean.rds")

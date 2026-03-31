# ------------------------------------------------------------------------------
# Purpose
# - Central path constants for raw data, processed splits, assessment artifacts,
#   and output files. Sourced after `load_packages.R` in each pipeline script.
# ------------------------------------------------------------------------------

# Raw inputs (semicolon-delimited wine data).
raw_train_path <- "data/raw/train.csv"
raw_test_path <- "data/raw/test.csv"

# Train/test splits produced by `scripts/split_train_test.R`.
train_split_path <- "data/processed/train_split.csv"
test_split_path <- "data/processed/test_split.csv"

# Cross-validation outputs from `scripts/model-assessment.R`.
model_assessment_metrics_path <- "data/processed/model_assessment_cv_metrics.csv"
lasso_penalty_by_fold_path <- "data/processed/lasso_nested_penalty_by_fold.csv"
xgboost_best_params_path <- "data/processed/xgboost_best_params.csv"

# Test evaluation and plots from `scripts/model-test.R`.
model_test_metrics_path <- "output/model_test_metrics.csv"
model_test_plot_path <- "output/model_test_yardsticks_bar.png"

# Final predictions from `scripts/final_model.R`.
final_model_predictions_path <- "output/final_model_predictions.csv"

# Prefer processed splits for training when present; test prefers raw Kaggle-style test if present.
default_train_path <- if (file.exists(train_split_path)) train_split_path else raw_train_path
default_test_path <- if (file.exists(raw_test_path)) raw_test_path else test_split_path

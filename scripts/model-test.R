#!/usr/bin/env Rscript

# ------------------------------------------------------------------------------
# Purpose
# - Fit candidate models on the train split and evaluate on the held-out test split.
# - Writes test metrics and an accuracy bar chart for reporting.
#
# Inputs
# - `train_split_path`, `test_split_path` (from `src/paths.R`; created by split script)
# - Optional: `lasso_penalty_by_fold_path`, `xgboost_best_params_path` from assessment
#
# Outputs
# - `model_test_metrics_path`: `output/model_test_metrics.csv`
# - `model_test_plot_path`:    `output/model_test_yardsticks_bar.png`
#
# How to run
# - `make test`
# - `Rscript scripts/model-test.R` (after `make split` and ideally `make assess`)
#
# Reproducibility
# - Fixed seed for any tuning/resampling in this script.
# ------------------------------------------------------------------------------

# Load packages, paths, metrics (`accuracy` + ord_mae helper), and `quality` coercion.
suppressPackageStartupMessages({
  source("src/load_packages.R")
  source("src/paths.R")
  source("src/metrics.R")
  source("src/quality_helpers.R")
})

# Fixed seed for reproducible resampling/tuning used before test evaluation.
set.seed(20260323)

# Require processed split files from `scripts/split_train_test.R`.
if (!file.exists(train_split_path) || !file.exists(test_split_path)) {
  stop(
    paste(
      "Missing split files. Expected:",
      train_split_path,
      "and",
      test_split_path,
      "\nRun scripts/split_train_test.R first."
    ),
    call. = FALSE
  )
}

# Load train and test CSVs (semicolon-delimited).
train_df <- readr::read_delim(train_split_path, delim = ";", show_col_types = FALSE)
test_df <- readr::read_delim(test_split_path, delim = ";", show_col_types = FALSE)

# Both splits must include the outcome column for evaluation.
if (!"quality" %in% names(train_df) || !"quality" %in% names(test_df)) {
  stop("Expected a 'quality' outcome column in both split datasets.", call. = FALSE)
}

# Align `quality` as ordered factor using training-set levels (same order for test).
quality_levels <- sort(unique(train_df$quality))
train_df <- dplyr::mutate(train_df, quality = as_ordered_quality(quality, levels = quality_levels))
test_df <- dplyr::mutate(test_df, quality = as_ordered_quality(quality, levels = quality_levels))

# Preprocessing for models that use all predictors; intercept recipe has no predictors.
base_recipe <- recipes::recipe(quality ~ ., data = train_df) %>%
  recipes::step_zv(recipes::all_predictors()) %>%
  recipes::step_normalize(recipes::all_numeric_predictors())
intercept_recipe <- recipes::recipe(quality ~ 1, data = train_df)

# Lasso strength: prefer median penalty from nested-CV file if assessment was run.
lasso_penalty <- NA_real_

if (file.exists(lasso_penalty_by_fold_path)) {
  penalty_df <- readr::read_csv(lasso_penalty_by_fold_path, show_col_types = FALSE)
  if ("penalty" %in% names(penalty_df) && nrow(penalty_df) > 0) {
    # Median penalty from nested-CV folds is robust to outlier folds.
    lasso_penalty <- stats::median(penalty_df$penalty, na.rm = TRUE)
  }
}

# If no saved penalty, tune lasso on train-only CV and pick best by accuracy.
if (is.na(lasso_penalty) || !is.finite(lasso_penalty)) {
  lasso_tune_spec <- parsnip::multinom_reg(penalty = tune::tune(), mixture = 1) %>%
    parsnip::set_engine("glmnet")

  lasso_tune_wf <- workflows::workflow() %>%
    workflows::add_recipe(base_recipe) %>%
    workflows::add_model(lasso_tune_spec)

  tune_folds <- rsample::vfold_cv(train_df, v = 5, strata = quality)
  tune_grid <- dials::grid_regular(dials::penalty(), levels = 30)

  tuned_lasso <- suppressWarnings(
    suppressMessages(
      tune::tune_grid(
        object = lasso_tune_wf,
        resamples = tune_folds,
        grid = tune_grid,
        metrics = metrics_used,
        control = tune::control_grid(save_pred = FALSE)
      )
    )
  )

  lasso_penalty <- tune::select_best(tuned_lasso, metric = "accuracy")$penalty
}

# Final specs: lasso (glmnet), full multinomial OLS (nnet), intercept-only (nnet).
lasso_spec <- parsnip::multinom_reg(penalty = lasso_penalty, mixture = 1) %>%
  parsnip::set_engine("glmnet")
ols_spec <- parsnip::multinom_reg(penalty = 0) %>% parsnip::set_engine("nnet", trace = FALSE)
intercept_spec <- parsnip::multinom_reg(penalty = 0) %>% parsnip::set_engine("nnet", trace = FALSE)

# xgboost hyperparameters: load from assessment if present, else tune below.
xgb_params <- NULL
if (file.exists(xgboost_best_params_path)) {
  xgb_params <- readr::read_csv(xgboost_best_params_path, show_col_types = FALSE)
}

# Boosted trees with tunable hyperparameters (filled from file or `tune_grid`).
xgb_spec <- parsnip::boost_tree(
  trees = tune::tune(),
  tree_depth = tune::tune(),
  learn_rate = tune::tune(),
  loss_reduction = tune::tune(),
  mtry = tune::tune(),
  min_n = tune::tune()
) %>%
  parsnip::set_mode("classification") %>%
  parsnip::set_engine("xgboost")

# One workflow per model variant.
lasso_wf <- workflows::workflow() %>%
  workflows::add_recipe(base_recipe) %>%
  workflows::add_model(lasso_spec)

ols_wf <- workflows::workflow() %>%
  workflows::add_recipe(base_recipe) %>%
  workflows::add_model(ols_spec)

intercept_wf <- workflows::workflow() %>%
  workflows::add_recipe(intercept_recipe) %>%
  workflows::add_model(intercept_spec)

xgb_wf <- workflows::workflow() %>%
  workflows::add_recipe(base_recipe) %>%
  workflows::add_model(xgb_spec)

# When no saved xgboost params, run a small tuning grid on train CV; best by ord_mae.
if (is.null(xgb_params) || nrow(xgb_params) == 0) {
  xgb_param_set <- dials::parameters(
    dials::trees(),
    dials::tree_depth(),
    dials::learn_rate(),
    dials::loss_reduction(),
    dials::mtry(range = c(1L, max(1L, ncol(train_df) - 1L))),
    dials::min_n()
  )
  xgb_grid <- dials::grid_space_filling(xgb_param_set, size = 25)
  tuned_xgb <- suppressWarnings(
    suppressMessages(
      tune::tune_grid(
        object = xgb_wf,
        resamples = rsample::vfold_cv(train_df, v = 5),
        grid = xgb_grid,
        metrics = metrics_used,
        control = tune::control_grid(save_pred = FALSE)
      )
    )
  )
  xgb_params <- tune::select_best(tuned_xgb, metric = "ord_mae")
}

# Plug chosen xgboost settings into the workflow spec.
xgb_final_wf <- tune::finalize_workflow(xgb_wf, xgb_params)

# Fit each model on the full training split (no further CV here).
fitted_lasso <- suppressWarnings(parsnip::fit(lasso_wf, data = train_df))
fitted_ols <- suppressWarnings(parsnip::fit(ols_wf, data = train_df))
fitted_intercept <- suppressWarnings(parsnip::fit(intercept_wf, data = train_df))
fitted_xgb <- suppressWarnings(parsnip::fit(xgb_final_wf, data = train_df))

# Predicted class labels on test; keep truth and a model name for stacking metrics.
pred_lasso <- predict(fitted_lasso, new_data = test_df, type = "class") %>%
  dplyr::bind_cols(test_df %>% dplyr::select(quality)) %>%
  dplyr::mutate(model = "lasso")

pred_ols <- predict(fitted_ols, new_data = test_df, type = "class") %>%
  dplyr::bind_cols(test_df %>% dplyr::select(quality)) %>%
  dplyr::mutate(model = "ols_full")

pred_intercept <- predict(fitted_intercept, new_data = test_df, type = "class") %>%
  dplyr::bind_cols(test_df %>% dplyr::select(quality)) %>%
  dplyr::mutate(model = "intercept_only")

pred_xgb <- predict(fitted_xgb, new_data = test_df, type = "class") %>%
  dplyr::bind_cols(test_df %>% dplyr::select(quality)) %>%
  dplyr::mutate(model = "xgboost")

# Long-format table of all models' predictions on the test set.
all_preds <- dplyr::bind_rows(pred_lasso, pred_ols, pred_intercept, pred_xgb)

# Per model: yardstick accuracy plus custom ord_mae (mean absolute error on ordered levels).
test_metrics <- all_preds %>%
  dplyr::group_by(model) %>%
  metrics_used(truth = quality, estimate = .pred_class) %>%
  dplyr::bind_rows(
    all_preds %>%
      dplyr::group_by(model) %>%
      dplyr::summarise(
        .metric = "ord_mae",
        .estimator = "multiclass",
        .estimate = ord_mae_vec(quality, .pred_class),
        .groups = "drop"
      )
  ) %>%
  dplyr::ungroup()

# Round for readable console output and print the full metric table.
errors_for_terminal <- test_metrics %>%
  dplyr::arrange(.metric, model) %>%
  dplyr::mutate(.estimate = round(.estimate, 4))

print(errors_for_terminal)

# Ensure output directory exists, then save metrics CSV.
if (!dir.exists("output")) dir.create("output", recursive = TRUE)

readr::write_csv(test_metrics, model_test_metrics_path)

# Bar chart: test accuracy only, fixed order of model factor for the plot.
plot_data <- test_metrics %>%
  dplyr::filter(.metric == "accuracy") %>%
  dplyr::mutate(
    model = factor(model, levels = c("xgboost", "lasso", "ols_full", "intercept_only"))
  )

# ggplot2 column chart of accuracy by model, y-axis clamped to [0, 1].
yardstick_plot <- ggplot2::ggplot(plot_data, ggplot2::aes(x = model, y = .estimate, fill = model)) +
  ggplot2::geom_col(width = 0.7, show.legend = FALSE) +
  ggplot2::labs(
    title = "Test-Set Accuracy by Model",
    x = "Model",
    y = "Accuracy"
  ) +
  ggplot2::ylim(0, 1) +
  ggplot2::theme_minimal(base_size = 12)

# Save PNG next to the metrics CSV for reports or slides.
ggplot2::ggsave(
  filename = model_test_plot_path,
  plot = yardstick_plot,
  width = 10,
  height = 4.5,
  dpi = 300
)


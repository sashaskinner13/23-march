#!/usr/bin/env Rscript

# ------------------------------------------------------------------------------
# Purpose
# - Run cross-validation model assessment to compare candidate classifiers for
#   ordered wine `quality`.
# - Produces comparable metrics across models and persists the key tuning
#   artifacts needed downstream (lasso penalty by fold; best xgboost parameters).
#
# Inputs
# - `default_train_path` (from `src/paths.R`):
#   - `data/processed/train_split.csv` if it exists, else `data/raw/train.csv`
# - Required columns: `quality` (outcome), plus predictor columns
#
# Outputs
# - `model_assessment_metrics_path`: `data/processed/model_assessment_cv_metrics.csv`
# - `lasso_penalty_by_fold_path`:    `data/processed/lasso_nested_penalty_by_fold.csv`
# - `xgboost_best_params_path`:      `data/processed/xgboost_best_params.csv`
#
# How to run
# - `make assess`
# - `Rscript scripts/model-assessment.R`
#
# Reproducibility
# - Uses a fixed seed for folds, tuning, and any randomness in resampling.
# ------------------------------------------------------------------------------

# Load packages, path constants, evaluation metrics, and `quality` factor helpers.
suppressPackageStartupMessages({
  source("src/load_packages.R")
  source("src/paths.R")
  source("src/metrics.R")
  source("src/quality_helpers.R")
})

# Fixed seed for reproducible folds, tuning, and predictions.
set.seed(20260323)

# ---- Load and validate modeling data ------------------------------------------
# Read training data from `default_train_path` (split file if present, else raw).
df <- readr::read_delim(
  file = default_train_path,
  delim = ";",
  show_col_types = FALSE
)

# Fail fast if the outcome column is missing.
if (!"quality" %in% names(df)) {
  stop("Expected a 'quality' column in modeling data.", call. = FALSE)
}
# Coerce `quality` to an ordered factor with stable level order across the pipeline.
quality_levels <- sort(unique(df$quality))
df <- dplyr::mutate(df, quality = as_ordered_quality(quality, levels = quality_levels))

# Shared outer 5-fold CV used by all models.
outer_folds <- rsample::vfold_cv(df, v = 5, strata = quality)

# ---- Shared preprocessing recipes ---------------------------------------------
# Full model: all predictors; drop zero-variance columns; normalize numeric inputs.
base_recipe <- recipes::recipe(quality ~ ., data = df) %>%
  recipes::step_zv(recipes::all_predictors()) %>%
  recipes::step_normalize(recipes::all_numeric_predictors())
# Intercept-only recipe: outcome ~ 1 (no predictors), for the baseline model.
intercept_recipe <- recipes::recipe(quality ~ 1, data = df)

# ---- Model specs and tuning grids ---------------------------------------------
# Lasso multinomial logistic (glmnet); penalty is tuned in nested CV below.
lasso_spec <- parsnip::multinom_reg(penalty = tune::tune(), mixture = 1) %>%
  parsnip::set_engine("glmnet")

# Full multinomial logistic with no regularization (nnet engine).
full_spec <- parsnip::multinom_reg(penalty = 0) %>%
  parsnip::set_engine("nnet", trace = FALSE)

# Intercept-only multinomial (predicts the majority class distribution).
intercept_spec <- parsnip::multinom_reg(penalty = 0) %>%
  parsnip::set_engine("nnet", trace = FALSE)

# Workflows pair each recipe with its model spec.
lasso_wf <- workflows::workflow() %>%
  workflows::add_recipe(base_recipe) %>%
  workflows::add_model(lasso_spec)

full_wf <- workflows::workflow() %>%
  workflows::add_recipe(base_recipe) %>%
  workflows::add_model(full_spec)

intercept_wf <- workflows::workflow() %>%
  workflows::add_recipe(intercept_recipe) %>%
  workflows::add_model(intercept_spec)

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

xgb_wf <- workflows::workflow() %>%
  workflows::add_recipe(base_recipe) %>%
  workflows::add_model(xgb_spec)

# Regular grid of `penalty` values for lasso tuning inside each outer fold.
lasso_grid <- dials::grid_regular(dials::penalty(), levels = 30)

# Hyperparameter ranges and a space-filling grid for xgboost tuning on outer folds.
xgb_param_set <- dials::parameters(
  dials::trees(),
  dials::tree_depth(),
  dials::learn_rate(),
  dials::loss_reduction(),
  dials::mtry(range = c(1L, max(1L, ncol(df) - 1L))),
  dials::min_n()
)
xgb_grid <- dials::grid_space_filling(xgb_param_set, size = 25)

# ---- Lasso: nested CV for robust penalty selection ----------------------------
# For each outer fold: tune lasso on inner CV, fit on analysis, predict on assessment.
lasso_nested_results <- purrr::map2_dfr(
  outer_folds$splits,
  outer_folds$id,
  function(split, fold_id) {
    # Training and holdout rows for this outer fold.
    analysis_data <- rsample::analysis(split)
    assessment_data <- rsample::assessment(split)

    # Inner 5-fold CV on the outer-fold training portion only.
    inner_folds <- rsample::vfold_cv(analysis_data, v = 5, strata = quality)

    # Grid search for the best penalty using `metrics_used` (includes accuracy).
    tuned_lasso <- suppressWarnings(
      suppressMessages(
        tune::tune_grid(
          object = lasso_wf,
          resamples = inner_folds,
          grid = lasso_grid,
          metrics = metrics_used,
          control = tune::control_grid(save_pred = FALSE)
        )
      )
    )

    # Pick best penalty by accuracy, refit on full outer-fold training data.
    best_params <- tune::select_best(tuned_lasso, metric = "accuracy")
    final_lasso <- tune::finalize_workflow(lasso_wf, best_params)
    fitted_lasso <- suppressWarnings(parsnip::fit(final_lasso, data = analysis_data))

    # Class predictions on the outer-fold assessment set.
    preds <- predict(fitted_lasso, new_data = assessment_data, type = "class") %>%
      dplyr::bind_cols(assessment_data %>% dplyr::select(quality))

    # Accuracy row plus custom ordered MAE for this outer fold.
    fold_accuracy <- yardstick::accuracy(preds, truth = quality, estimate = .pred_class) %>%
      dplyr::mutate(model = "lasso", fold = fold_id, penalty = best_params$penalty)

    fold_ord_mae <- tibble::tibble(
      model = "lasso",
      fold = fold_id,
      penalty = best_params$penalty,
      .metric = "ord_mae",
      .estimator = "multiclass",
      .estimate = ord_mae_vec(preds$quality, preds$.pred_class)
    )

    fold_metrics <- dplyr::bind_rows(fold_accuracy, fold_ord_mae)

    fold_metrics
  }
)

# Mean / SE of metrics across outer folds for the lasso model.
lasso_summary <- lasso_nested_results %>%
  dplyr::group_by(model, .metric) %>%
  dplyr::summarise(
    mean = mean(.estimate),
    std_err = stats::sd(.estimate) / sqrt(dplyr::n()),
    n = dplyr::n(),
    .groups = "drop"
  )

# Keep fold-level predictions when refitting models for ord_mae aggregation.
resample_control <- tune::control_resamples(save_pred = TRUE)

# ---- XGBoost: tune on outer folds, select by ordered MAE ----------------------
# Tune xgboost hyperparameters across the same outer folds; store predictions.
xgb_tuned <- suppressWarnings(
  suppressMessages(
    tune::tune_grid(
      object = xgb_wf,
      resamples = outer_folds,
      grid = xgb_grid,
      metrics = metrics_used,
      control = tune::control_grid(save_pred = TRUE)
    )
  )
)

# For each tuned config, compute mean ordered MAE across folds to rank models.
xgb_tuned_preds <- tune::collect_predictions(xgb_tuned)
xgb_ord_mae_by_config <- xgb_tuned_preds %>%
  dplyr::group_by(.config, id) %>%
  dplyr::summarise(.estimate = ord_mae_vec(quality, .pred_class), .groups = "drop") %>%
  dplyr::group_by(.config) %>%
  dplyr::summarise(mean_ord_mae = mean(.estimate), .groups = "drop") %>%
  dplyr::arrange(mean_ord_mae)

# All candidate configs with accuracy-based ranking from tune.
xgb_params_by_config <- tune::show_best(xgb_tuned, metric = "accuracy", n = Inf) %>%
  dplyr::select(.config, dplyr::everything())

# Join accuracy-ranked configs with ord_mae; take the row with lowest mean ord_mae.
xgb_best <- xgb_params_by_config %>%
  dplyr::inner_join(xgb_ord_mae_by_config, by = ".config") %>%
  dplyr::arrange(mean_ord_mae) %>%
  dplyr::slice(1) %>%
  dplyr::select(-mean_ord_mae, -dplyr::starts_with(".metric"))

# Refit the chosen xgboost settings on each outer fold for tidy metrics + preds.
xgb_resampled <- suppressWarnings(
  suppressMessages(
    tune::fit_resamples(
      object = tune::finalize_workflow(xgb_wf, xgb_best),
      resamples = outer_folds,
      metrics = metrics_used,
      control = resample_control
    )
  )
)

# Attach model label for binding with other models' prediction tables.
xgb_preds <- tune::collect_predictions(xgb_resampled) %>%
  dplyr::mutate(model = "xgboost")

xgb_metrics_accuracy <- tune::collect_metrics(xgb_resampled) %>%
  dplyr::mutate(model = "xgboost") %>%
  dplyr::select(model, .metric, mean, std_err, n)

# Derive fold-level ord_mae from predictions, then summarize mean/SE like yardstick.
xgb_metrics_ord_mae <- xgb_preds %>%
  dplyr::group_by(model, id) %>%
  dplyr::summarise(
    .estimate = ord_mae_vec(quality, .pred_class),
    .groups = "drop"
  ) %>%
  dplyr::summarise(
    model = "xgboost",
    .metric = "ord_mae",
    mean = mean(.estimate),
    std_err = stats::sd(.estimate) / sqrt(dplyr::n()),
    n = dplyr::n(),
    .groups = "drop"
  )

# Single row set per metric for xgboost (accuracy from tune; ord_mae custom).
xgb_metrics <- dplyr::bind_rows(xgb_metrics_accuracy, xgb_metrics_ord_mae)

# ---- Baselines: full multinom and intercept-only ------------------------------
# Multinomial with all predictors: resample on outer folds with saved predictions.
full_resampled <- suppressWarnings(
  suppressMessages(
    tune::fit_resamples(
      object = full_wf,
      resamples = outer_folds,
      metrics = metrics_used,
      control = resample_control
    )
  )
)

# Intercept-only baseline: same outer folds and metric set.
intercept_resampled <- suppressWarnings(
  suppressMessages(
    tune::fit_resamples(
      object = intercept_wf,
      resamples = outer_folds,
      metrics = metrics_used,
      control = resample_control
    )
  )
)

# Collect default metrics (accuracy) from tune for full and intercept models.
full_summary <- tune::collect_metrics(full_resampled) %>%
  dplyr::mutate(model = "ols_full") %>%
  dplyr::select(model, .metric, mean, std_err, n)

intercept_summary <- tune::collect_metrics(intercept_resampled) %>%
  dplyr::mutate(model = "intercept_only") %>%
  dplyr::select(model, .metric, mean, std_err, n)

# Predictions used to compute ord_mae for ols_full and intercept_only.
full_preds <- tune::collect_predictions(full_resampled) %>%
  dplyr::mutate(model = "ols_full")

intercept_preds <- tune::collect_predictions(intercept_resampled) %>%
  dplyr::mutate(model = "intercept_only")

# Per-fold ord_mae, then averaged per model for summary rows.
ord_mae_summary <- dplyr::bind_rows(full_preds, intercept_preds) %>%
  dplyr::group_by(model, id) %>%
  dplyr::summarise(.estimate = ord_mae_vec(quality, .pred_class), .groups = "drop") %>%
  dplyr::group_by(model) %>%
  dplyr::summarise(
    .metric = "ord_mae",
    mean = mean(.estimate),
    std_err = stats::sd(.estimate) / sqrt(dplyr::n()),
    n = dplyr::n(),
    .groups = "drop"
  )

# Append ord_mae summaries to the accuracy-based summaries for each baseline.
full_summary <- dplyr::bind_rows(full_summary, dplyr::filter(ord_mae_summary, model == "ols_full"))
intercept_summary <- dplyr::bind_rows(intercept_summary, dplyr::filter(ord_mae_summary, model == "intercept_only"))

# One table: all models, sorted by metric name then model name.
all_metrics <- dplyr::bind_rows(lasso_summary, xgb_metrics, full_summary, intercept_summary) %>%
  dplyr::arrange(.metric, model)

# ---- Persist outputs for downstream scripts -----------------------------------
# One row per outer fold with the lasso penalty chosen in nested CV (for median fallback).
penalty_by_fold <- lasso_nested_results %>%
  dplyr::filter(.metric == "accuracy") %>%
  dplyr::select(fold, penalty)

# Write CV metric table, per-fold lasso penalties, and the selected xgboost hyperparameters.
readr::write_csv(all_metrics, model_assessment_metrics_path)
readr::write_csv(penalty_by_fold, lasso_penalty_by_fold_path)
readr::write_csv(xgb_best, xgboost_best_params_path)


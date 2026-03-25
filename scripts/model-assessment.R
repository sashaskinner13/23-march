#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  source("src/load_packages.R")
  source("src/paths.R")
  source("src/metrics.R")
  source("src/quality_helpers.R")
})

# Fixed seed for reproducible folds, tuning, and predictions.
set.seed(20260323)

df <- readr::read_delim(
  file = default_train_path,
  delim = ";",
  show_col_types = FALSE
)

if (!"quality" %in% names(df)) {
  stop("Expected a 'quality' column in modeling data.", call. = FALSE)
}
quality_levels <- sort(unique(df$quality))
df <- dplyr::mutate(df, quality = as_ordered_quality(quality, levels = quality_levels))

# Shared outer 5-fold CV used by all models.
outer_folds <- rsample::vfold_cv(df, v = 5, strata = quality)

base_recipe <- recipes::recipe(quality ~ ., data = df) %>%
  recipes::step_zv(recipes::all_predictors()) %>%
  recipes::step_normalize(recipes::all_numeric_predictors())
intercept_recipe <- recipes::recipe(quality ~ 1, data = df)

lasso_spec <- parsnip::multinom_reg(penalty = tune::tune(), mixture = 1) %>%
  parsnip::set_engine("glmnet")

full_spec <- parsnip::multinom_reg(penalty = 0) %>%
  parsnip::set_engine("nnet", trace = FALSE)

intercept_spec <- parsnip::multinom_reg(penalty = 0) %>%
  parsnip::set_engine("nnet", trace = FALSE)

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

lasso_grid <- dials::grid_regular(dials::penalty(), levels = 30)

xgb_param_set <- dials::parameters(
  dials::trees(),
  dials::tree_depth(),
  dials::learn_rate(),
  dials::loss_reduction(),
  dials::mtry(range = c(1L, max(1L, ncol(df) - 1L))),
  dials::min_n()
)
xgb_grid <- dials::grid_space_filling(xgb_param_set, size = 25)

lasso_nested_results <- purrr::map2_dfr(
  outer_folds$splits,
  outer_folds$id,
  function(split, fold_id) {
    analysis_data <- rsample::analysis(split)
    assessment_data <- rsample::assessment(split)

    inner_folds <- rsample::vfold_cv(analysis_data, v = 5, strata = quality)

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

    best_params <- tune::select_best(tuned_lasso, metric = "accuracy")
    final_lasso <- tune::finalize_workflow(lasso_wf, best_params)
    fitted_lasso <- suppressWarnings(parsnip::fit(final_lasso, data = analysis_data))

    preds <- predict(fitted_lasso, new_data = assessment_data, type = "class") %>%
      dplyr::bind_cols(assessment_data %>% dplyr::select(quality))

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

lasso_summary <- lasso_nested_results %>%
  dplyr::group_by(model, .metric) %>%
  dplyr::summarise(
    mean = mean(.estimate),
    std_err = stats::sd(.estimate) / sqrt(dplyr::n()),
    n = dplyr::n(),
    .groups = "drop"
  )

resample_control <- tune::control_resamples(save_pred = TRUE)

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

xgb_tuned_preds <- tune::collect_predictions(xgb_tuned)
xgb_ord_mae_by_config <- xgb_tuned_preds %>%
  dplyr::group_by(.config, id) %>%
  dplyr::summarise(.estimate = ord_mae_vec(quality, .pred_class), .groups = "drop") %>%
  dplyr::group_by(.config) %>%
  dplyr::summarise(mean_ord_mae = mean(.estimate), .groups = "drop") %>%
  dplyr::arrange(mean_ord_mae)

xgb_params_by_config <- tune::show_best(xgb_tuned, metric = "accuracy", n = Inf) %>%
  dplyr::select(.config, dplyr::everything())

xgb_best <- xgb_params_by_config %>%
  dplyr::inner_join(xgb_ord_mae_by_config, by = ".config") %>%
  dplyr::arrange(mean_ord_mae) %>%
  dplyr::slice(1) %>%
  dplyr::select(-mean_ord_mae, -dplyr::starts_with(".metric"))

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

xgb_preds <- tune::collect_predictions(xgb_resampled) %>%
  dplyr::mutate(model = "xgboost")

xgb_metrics_accuracy <- tune::collect_metrics(xgb_resampled) %>%
  dplyr::mutate(model = "xgboost") %>%
  dplyr::select(model, .metric, mean, std_err, n)

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

xgb_metrics <- dplyr::bind_rows(xgb_metrics_accuracy, xgb_metrics_ord_mae)

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

full_summary <- tune::collect_metrics(full_resampled) %>%
  dplyr::mutate(model = "ols_full") %>%
  dplyr::select(model, .metric, mean, std_err, n)

intercept_summary <- tune::collect_metrics(intercept_resampled) %>%
  dplyr::mutate(model = "intercept_only") %>%
  dplyr::select(model, .metric, mean, std_err, n)

full_preds <- tune::collect_predictions(full_resampled) %>%
  dplyr::mutate(model = "ols_full")

intercept_preds <- tune::collect_predictions(intercept_resampled) %>%
  dplyr::mutate(model = "intercept_only")

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

full_summary <- dplyr::bind_rows(full_summary, dplyr::filter(ord_mae_summary, model == "ols_full"))
intercept_summary <- dplyr::bind_rows(intercept_summary, dplyr::filter(ord_mae_summary, model == "intercept_only"))

all_metrics <- dplyr::bind_rows(lasso_summary, xgb_metrics, full_summary, intercept_summary) %>%
  dplyr::arrange(.metric, model)

penalty_by_fold <- lasso_nested_results %>%
  dplyr::filter(.metric == "accuracy") %>%
  dplyr::select(fold, penalty)

readr::write_csv(all_metrics, model_assessment_metrics_path)
readr::write_csv(penalty_by_fold, lasso_penalty_by_fold_path)
readr::write_csv(xgb_best, xgboost_best_params_path)


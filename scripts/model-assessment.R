#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  source("src/load_packages.R")
})

# Fixed seed for reproducible folds, tuning, and predictions.
set.seed(20260323)

metrics_used <- yardstick::metric_set(yardstick::accuracy)

data_path <- if (file.exists("data/processed/train_split.csv")) {
  "data/processed/train_split.csv"
} else {
  "data/processed/train.csv"
}

df <- readr::read_delim(
  file = data_path,
  delim = ";",
  show_col_types = FALSE
)

if (!"quality" %in% names(df)) {
  stop("Expected a 'quality' column in modeling data.", call. = FALSE)
}
df <- dplyr::mutate(df, quality = factor(quality))

# Shared outer 5-fold CV used by all models.
outer_folds <- rsample::vfold_cv(df, v = 5)

base_recipe <- recipes::recipe(quality ~ ., data = df)
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

lasso_grid <- dials::grid_regular(dials::penalty(), levels = 30)

lasso_nested_results <- purrr::map2_dfr(
  outer_folds$splits,
  outer_folds$id,
  function(split, fold_id) {
    analysis_data <- rsample::analysis(split)
    assessment_data <- rsample::assessment(split)

    inner_folds <- rsample::vfold_cv(analysis_data, v = 5)

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

    best_accuracy <- tune::select_best(tuned_lasso, metric = "accuracy")
    final_lasso <- tune::finalize_workflow(lasso_wf, best_accuracy)
    fitted_lasso <- suppressWarnings(parsnip::fit(final_lasso, data = analysis_data))

    preds <- predict(fitted_lasso, new_data = assessment_data, type = "class") %>%
      dplyr::bind_cols(assessment_data %>% dplyr::select(quality))

    fold_metrics <- metrics_used(preds, truth = quality, estimate = .pred_class) %>%
      dplyr::mutate(
        model = "lasso",
        fold = fold_id,
        penalty = best_accuracy$penalty
      )

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

all_metrics <- dplyr::bind_rows(lasso_summary, full_summary, intercept_summary) %>%
  dplyr::arrange(.metric, model)

penalty_by_fold <- lasso_nested_results %>%
  dplyr::filter(.metric == "accuracy") %>%
  dplyr::select(fold, penalty)

metrics_output <- "data/processed/model_assessment_cv_metrics.csv"
penalty_output <- "data/processed/lasso_nested_penalty_by_fold.csv"

readr::write_csv(all_metrics, metrics_output)
readr::write_csv(penalty_by_fold, penalty_output)


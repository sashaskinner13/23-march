#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  source("src/load_packages.R")
})

# Fixed seed for reproducible resampling/tuning used before test evaluation.
set.seed(20260323)

train_path <- "data/processed/train_split.csv"
test_path <- "data/processed/test_split.csv"
penalty_path <- "data/processed/lasso_nested_penalty_by_fold.csv"

if (!file.exists(train_path) || !file.exists(test_path)) {
  stop(
    paste(
      "Missing split files. Expected:",
      train_path,
      "and",
      test_path,
      "\nRun scripts/split_train_test.R first."
    ),
    call. = FALSE
  )
}

train_df <- readr::read_delim(train_path, delim = ";", show_col_types = FALSE)
test_df <- readr::read_delim(test_path, delim = ";", show_col_types = FALSE)

if (!"quality" %in% names(train_df) || !"quality" %in% names(test_df)) {
  stop("Expected a 'quality' outcome column in both split datasets.", call. = FALSE)
}

metrics_used <- yardstick::metric_set(yardstick::rmse, yardstick::mae, yardstick::rsq)

base_recipe <- recipes::recipe(quality ~ ., data = train_df)
intercept_recipe <- recipes::recipe(quality ~ 1, data = train_df)

lasso_penalty <- NA_real_

if (file.exists(penalty_path)) {
  penalty_df <- readr::read_csv(penalty_path, show_col_types = FALSE)
  if ("penalty" %in% names(penalty_df) && nrow(penalty_df) > 0) {
    # Median penalty from nested-CV folds is robust to outlier folds.
    lasso_penalty <- stats::median(penalty_df$penalty, na.rm = TRUE)
  }
}

if (is.na(lasso_penalty) || !is.finite(lasso_penalty)) {
  cat("No valid nested-CV penalty found. Tuning lasso penalty on train split with 5-fold CV...\n")
  lasso_tune_spec <- parsnip::linear_reg(penalty = tune::tune(), mixture = 1) %>%
    parsnip::set_engine("glmnet")

  lasso_tune_wf <- workflows::workflow() %>%
    workflows::add_recipe(base_recipe) %>%
    workflows::add_model(lasso_tune_spec)

  tune_folds <- rsample::vfold_cv(train_df, v = 5)
  tune_grid <- dials::grid_regular(dials::penalty(), levels = 30)

  tuned_lasso <- tune::tune_grid(
    object = lasso_tune_wf,
    resamples = tune_folds,
    grid = tune_grid,
    metrics = metrics_used,
    control = tune::control_grid(save_pred = FALSE)
  )

  lasso_penalty <- tune::select_best(tuned_lasso, metric = "rmse")$penalty
}

lasso_spec <- parsnip::linear_reg(penalty = lasso_penalty, mixture = 1) %>%
  parsnip::set_engine("glmnet")
ols_spec <- parsnip::linear_reg() %>% parsnip::set_engine("lm")
intercept_spec <- parsnip::linear_reg() %>% parsnip::set_engine("lm")

lasso_wf <- workflows::workflow() %>%
  workflows::add_recipe(base_recipe) %>%
  workflows::add_model(lasso_spec)

ols_wf <- workflows::workflow() %>%
  workflows::add_recipe(base_recipe) %>%
  workflows::add_model(ols_spec)

intercept_wf <- workflows::workflow() %>%
  workflows::add_recipe(intercept_recipe) %>%
  workflows::add_model(intercept_spec)

fitted_lasso <- parsnip::fit(lasso_wf, data = train_df)
fitted_ols <- parsnip::fit(ols_wf, data = train_df)
fitted_intercept <- parsnip::fit(intercept_wf, data = train_df)

pred_lasso <- predict(fitted_lasso, new_data = test_df) %>%
  dplyr::bind_cols(test_df %>% dplyr::select(quality)) %>%
  dplyr::mutate(model = "lasso")

pred_ols <- predict(fitted_ols, new_data = test_df) %>%
  dplyr::bind_cols(test_df %>% dplyr::select(quality)) %>%
  dplyr::mutate(model = "ols_full")

pred_intercept <- predict(fitted_intercept, new_data = test_df) %>%
  dplyr::bind_cols(test_df %>% dplyr::select(quality)) %>%
  dplyr::mutate(model = "intercept_only")

all_preds <- dplyr::bind_rows(pred_lasso, pred_ols, pred_intercept)

test_metrics <- all_preds %>%
  dplyr::group_by(model) %>%
  metrics_used(truth = quality, estimate = .pred) %>%
  dplyr::ungroup()

errors_for_terminal <- test_metrics %>%
  dplyr::filter(.metric %in% c("rmse", "mae")) %>%
  dplyr::arrange(model, .metric) %>%
  dplyr::mutate(.estimate = round(.estimate, 4))

cat("Test-set error metrics (quality prediction):\n")
print(errors_for_terminal)

cat("\nLasso penalty used:", signif(lasso_penalty, 4), "\n")

metrics_output <- "output/model_test_metrics.csv"
plot_output <- "output/model_test_yardsticks_bar.png"

readr::write_csv(test_metrics, metrics_output)

plot_data <- test_metrics %>%
  dplyr::mutate(
    model = factor(model, levels = c("lasso", "ols_full", "intercept_only")),
    .metric = factor(.metric, levels = c("rmse", "mae", "rsq"))
  )

yardstick_plot <- ggplot2::ggplot(plot_data, ggplot2::aes(x = model, y = .estimate, fill = model)) +
  ggplot2::geom_col(width = 0.7, show.legend = FALSE) +
  ggplot2::facet_wrap(~.metric, scales = "free_y") +
  ggplot2::labs(
    title = "Test-Set Yardstick Metrics by Model",
    x = "Model",
    y = "Metric Value"
  ) +
  ggplot2::theme_minimal(base_size = 12)

ggplot2::ggsave(
  filename = plot_output,
  plot = yardstick_plot,
  width = 10,
  height = 4.5,
  dpi = 300
)

cat("\nSaved outputs:\n")
cat(paste0("- ", metrics_output, "\n"))
cat(paste0("- ", plot_output, "\n"))


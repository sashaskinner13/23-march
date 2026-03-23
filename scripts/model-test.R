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

train_df <- dplyr::mutate(train_df, quality = factor(quality))
test_df <- dplyr::mutate(test_df, quality = factor(quality, levels = levels(train_df$quality)))

metrics_used <- yardstick::metric_set(yardstick::accuracy)

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
  lasso_tune_spec <- parsnip::linear_reg(penalty = tune::tune(), mixture = 1) %>%
    parsnip::set_engine("glmnet")

  lasso_tune_wf <- workflows::workflow() %>%
    workflows::add_recipe(base_recipe) %>%
    workflows::add_model(lasso_tune_spec)

  tune_folds <- rsample::vfold_cv(train_df, v = 5)
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

lasso_spec <- parsnip::multinom_reg(penalty = lasso_penalty, mixture = 1) %>%
  parsnip::set_engine("glmnet")
ols_spec <- parsnip::multinom_reg(penalty = 0) %>% parsnip::set_engine("nnet", trace = FALSE)
intercept_spec <- parsnip::multinom_reg(penalty = 0) %>% parsnip::set_engine("nnet", trace = FALSE)

lasso_wf <- workflows::workflow() %>%
  workflows::add_recipe(base_recipe) %>%
  workflows::add_model(lasso_spec)

ols_wf <- workflows::workflow() %>%
  workflows::add_recipe(base_recipe) %>%
  workflows::add_model(ols_spec)

intercept_wf <- workflows::workflow() %>%
  workflows::add_recipe(intercept_recipe) %>%
  workflows::add_model(intercept_spec)

fitted_lasso <- suppressWarnings(parsnip::fit(lasso_wf, data = train_df))
fitted_ols <- suppressWarnings(parsnip::fit(ols_wf, data = train_df))
fitted_intercept <- suppressWarnings(parsnip::fit(intercept_wf, data = train_df))

pred_lasso <- predict(fitted_lasso, new_data = test_df, type = "class") %>%
  dplyr::bind_cols(test_df %>% dplyr::select(quality)) %>%
  dplyr::mutate(model = "lasso")

pred_ols <- predict(fitted_ols, new_data = test_df, type = "class") %>%
  dplyr::bind_cols(test_df %>% dplyr::select(quality)) %>%
  dplyr::mutate(model = "ols_full")

pred_intercept <- predict(fitted_intercept, new_data = test_df, type = "class") %>%
  dplyr::bind_cols(test_df %>% dplyr::select(quality)) %>%
  dplyr::mutate(model = "intercept_only")

all_preds <- dplyr::bind_rows(pred_lasso, pred_ols, pred_intercept)

test_metrics <- all_preds %>%
  dplyr::group_by(model) %>%
  metrics_used(truth = quality, estimate = .pred_class) %>%
  dplyr::ungroup()

errors_for_terminal <- test_metrics %>%
  dplyr::arrange(model) %>%
  dplyr::mutate(.estimate = round(.estimate, 4))

print(errors_for_terminal)

metrics_output <- "output/model_test_metrics.csv"
plot_output <- "output/model_test_yardsticks_bar.png"
if (!dir.exists("output")) dir.create("output", recursive = TRUE)

readr::write_csv(test_metrics, metrics_output)

plot_data <- test_metrics %>%
  dplyr::mutate(
    model = factor(model, levels = c("lasso", "ols_full", "intercept_only"))
  )

yardstick_plot <- ggplot2::ggplot(plot_data, ggplot2::aes(x = model, y = .estimate, fill = model)) +
  ggplot2::geom_col(width = 0.7, show.legend = FALSE) +
  ggplot2::labs(
    title = "Test-Set Accuracy by Model",
    x = "Model",
    y = "Accuracy"
  ) +
  ggplot2::ylim(0, 1) +
  ggplot2::theme_minimal(base_size = 12)

ggplot2::ggsave(
  filename = plot_output,
  plot = yardstick_plot,
  width = 10,
  height = 4.5,
  dpi = 300
)


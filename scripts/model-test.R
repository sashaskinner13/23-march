#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  source("src/load_packages.R")
})

# Fixed seed for reproducible resampling/tuning used before test evaluation.
set.seed(20260323)

# #region agent log
debug_log_path <- ".cursor/debug-b139f0.log"
debug_run_id <- paste0("pre-fix-", format(Sys.time(), "%Y%m%d%H%M%S"))
debug_log <- function(hypothesis_id, location, message, data = list()) {
  entry <- list(
    sessionId = "b139f0",
    runId = debug_run_id,
    hypothesisId = hypothesis_id,
    location = location,
    message = message,
    data = data,
    timestamp = as.integer(as.numeric(Sys.time()) * 1000)
  )
  json_line <- if (requireNamespace("jsonlite", quietly = TRUE)) {
    jsonlite::toJSON(entry, auto_unbox = TRUE, null = "null")
  } else {
    paste0(
      "{\"sessionId\":\"b139f0\",\"runId\":\"", debug_run_id,
      "\",\"hypothesisId\":\"", hypothesis_id,
      "\",\"location\":\"", location,
      "\",\"message\":\"", message,
      "\",\"timestamp\":", as.integer(as.numeric(Sys.time()) * 1000), "}"
    )
  }
  cat(as.character(json_line), "\n", file = debug_log_path, append = TRUE)
}
debug_log("H3", "scripts/model-test.R:37", "script_started", list())
# #endregion

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

# #region agent log
debug_log(
  "H2",
  "scripts/model-test.R:136",
  "all_preds_built",
  list(
    nrow = nrow(all_preds),
    cols = paste(names(all_preds), collapse = ","),
    has_quality = "quality" %in% names(all_preds),
    has_pred = ".pred" %in% names(all_preds)
  )
)
# #endregion

test_metrics <- tryCatch(
  {
    # #region agent log
    debug_log(
      "H1",
      "scripts/model-test.R:152",
      "before_test_metrics_compute",
      list(
        quality_class = paste(class(all_preds$quality), collapse = ","),
        pred_class = paste(class(all_preds$.pred), collapse = ","),
        quality_na = sum(is.na(all_preds$quality)),
        pred_na = sum(is.na(all_preds$.pred))
      )
    )
    # #endregion

    metrics <- all_preds %>%
      dplyr::group_by(model) %>%
      metrics_used(truth = quality, estimate = .pred) %>%
      dplyr::ungroup()

    # #region agent log
    debug_log(
      "H4",
      "scripts/model-test.R:170",
      "test_metrics_computed",
      list(nrow = nrow(metrics), cols = paste(names(metrics), collapse = ","))
    )
    # #endregion
    metrics
  },
  error = function(e) {
    # #region agent log
    debug_log(
      "H1",
      "scripts/model-test.R:180",
      "test_metrics_compute_error",
      list(error = conditionMessage(e), metrics_used_exists = exists("metrics_used"))
    )
    # #endregion
    stop(e)
  }
)

errors_for_terminal <- test_metrics %>%
  dplyr::filter(.metric %in% c("rmse", "mae")) %>%
  dplyr::arrange(model, .metric) %>%
  dplyr::mutate(.estimate = round(.estimate, 4))

cat("Test-set error metrics (quality prediction):\n")
print(errors_for_terminal)

cat("\nLasso penalty used:", signif(lasso_penalty, 4), "\n")

metrics_output <- "output/model_test_metrics.csv"
plot_output <- "output/model_test_yardsticks_bar.png"
if (!dir.exists("output")) dir.create("output", recursive = TRUE)

# #region agent log
debug_log(
  "H5",
  "scripts/model-test.R:200",
  "before_write_outputs",
  list(test_metrics_exists = exists("test_metrics"), test_metrics_nrow = nrow(test_metrics))
)
# #endregion

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


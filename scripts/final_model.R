#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  source("src/load_packages.R")
})

set.seed(20260323)

train_path <- if (file.exists("data/processed/train_split.csv")) {
  "data/processed/train_split.csv"
} else {
  "data/raw/train.csv"
}

metrics_path <- "data/processed/model_assessment_cv_metrics.csv"
penalty_path <- "data/processed/lasso_nested_penalty_by_fold.csv"
predictions_output <- "output/final_model_predictions.csv"

if (!file.exists(metrics_path)) {
  stop(
    paste(
      "Missing model test metrics at",
      metrics_path,
      "\nRun scripts/model-test.R first."
    ),
    call. = FALSE
  )
}

test_path <- if (file.exists("data/raw/test.csv")) {
  "data/raw/test.csv"
} else if (file.exists("data/processed/test_split.csv")) {
  "data/processed/test_split.csv"
} else {
  stop(
    "Missing test data. Expected data/raw/test.csv or data/processed/test_split.csv.",
    call. = FALSE
  )
}

train_df <- readr::read_delim(train_path, delim = ";", show_col_types = FALSE)
test_df <- readr::read_delim(test_path, delim = ";", show_col_types = FALSE)
metrics_df <- readr::read_csv(metrics_path, show_col_types = FALSE)

if (!"quality" %in% names(train_df)) {
  stop("Expected a 'quality' column in training data.", call. = FALSE)
}
train_df <- dplyr::mutate(train_df, quality = factor(quality))

best_model <- metrics_df %>%
  dplyr::filter(.metric == "accuracy", !is.na(mean)) %>%
  dplyr::arrange(dplyr::desc(mean)) %>%
  dplyr::slice(1) %>%
  dplyr::pull(model)

if (length(best_model) == 0 || is.na(best_model)) {
  stop("No valid best model found from accuracy in model_assessment_cv_metrics.csv.", call. = FALSE)
}

base_recipe <- recipes::recipe(quality ~ ., data = train_df)
intercept_recipe <- recipes::recipe(quality ~ 1, data = train_df)

if (best_model == "lasso") {
    lasso_penalty <- NA_real_
    if (file.exists(penalty_path)) {
      penalty_df <- readr::read_csv(penalty_path, show_col_types = FALSE)
      if ("penalty" %in% names(penalty_df) && nrow(penalty_df) > 0) {
        lasso_penalty <- stats::median(penalty_df$penalty, na.rm = TRUE)
      }
    }

    if (is.na(lasso_penalty) || !is.finite(lasso_penalty)) {
      lasso_tune_spec <- parsnip::multinom_reg(penalty = tune::tune(), mixture = 1) %>%
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
            metrics = yardstick::metric_set(yardstick::accuracy),
            control = tune::control_grid(save_pred = FALSE)
          )
        )
      )
      lasso_penalty <- tune::select_best(tuned_lasso, metric = "accuracy")$penalty
    }

    lasso_spec <- parsnip::multinom_reg(penalty = lasso_penalty, mixture = 1) %>%
      parsnip::set_engine("glmnet")
    model_wf <- workflows::workflow() %>%
      workflows::add_recipe(base_recipe) %>%
      workflows::add_model(lasso_spec)
} else if (best_model == "ols_full") {
    ols_spec <- parsnip::multinom_reg(penalty = 0) %>%
      parsnip::set_engine("nnet", trace = FALSE)
    model_wf <- workflows::workflow() %>%
      workflows::add_recipe(base_recipe) %>%
      workflows::add_model(ols_spec)
} else if (best_model == "intercept_only") {
    intercept_spec <- parsnip::multinom_reg(penalty = 0) %>%
      parsnip::set_engine("nnet", trace = FALSE)
    model_wf <- workflows::workflow() %>%
      workflows::add_recipe(intercept_recipe) %>%
      workflows::add_model(intercept_spec)
} else {
  stop(paste("Unsupported best model:", best_model), call. = FALSE)
}

fitted_model <- suppressWarnings(parsnip::fit(model_wf, data = train_df))
preds <- predict(fitted_model, new_data = test_df, type = "class")

if (!dir.exists("output")) dir.create("output", recursive = TRUE)

predictions_df <- dplyr::bind_cols(test_df, preds) %>%
  dplyr::mutate(
    selected_model = best_model,
    predicted_quality = as.character(.pred_class)
  )

readr::write_csv(predictions_df, predictions_output)

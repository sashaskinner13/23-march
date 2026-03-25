#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  source("src/load_packages.R")
  source("src/paths.R")
  source("src/metrics.R")
  source("src/quality_helpers.R")
})

set.seed(20260323)

if (!file.exists(model_assessment_metrics_path)) {
  stop(
    paste(
      "Missing model test metrics at",
      model_assessment_metrics_path,
      "\nRun scripts/model-test.R first."
    ),
    call. = FALSE
  )
}

if (!file.exists(default_test_path)) {
  stop(
    "Missing test data. Expected data/raw/test.csv or data/processed/test_split.csv.",
    call. = FALSE
  )
}

train_df <- readr::read_delim(default_train_path, delim = ";", show_col_types = FALSE)
test_df <- readr::read_delim(default_test_path, delim = ";", show_col_types = FALSE)
metrics_df <- readr::read_csv(model_assessment_metrics_path, show_col_types = FALSE)

if (!"quality" %in% names(train_df)) {
  stop("Expected a 'quality' column in training data.", call. = FALSE)
}
quality_levels <- sort(unique(train_df$quality))
train_df <- dplyr::mutate(train_df, quality = as_ordered_quality(quality, levels = quality_levels))

best_model <- metrics_df %>%
  dplyr::filter(.metric == "ord_mae", !is.na(mean)) %>%
  dplyr::arrange(mean) %>%
  dplyr::slice(1) %>%
  dplyr::pull(model)

if (length(best_model) == 0 || is.na(best_model)) {
  stop("No valid best model found from accuracy in model_assessment_cv_metrics.csv.", call. = FALSE)
}

base_recipe <- recipes::recipe(quality ~ ., data = train_df) %>%
  recipes::step_zv(recipes::all_predictors()) %>%
  recipes::step_normalize(recipes::all_numeric_predictors())
intercept_recipe <- recipes::recipe(quality ~ 1, data = train_df)

if (best_model == "lasso") {
    lasso_penalty <- NA_real_
    if (file.exists(lasso_penalty_by_fold_path)) {
      penalty_df <- readr::read_csv(lasso_penalty_by_fold_path, show_col_types = FALSE)
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
} else if (best_model == "xgboost") {
    xgb_params <- NULL
    if (file.exists(xgboost_best_params_path)) {
      xgb_params <- readr::read_csv(xgboost_best_params_path, show_col_types = FALSE)
    }

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

    model_wf <- tune::finalize_workflow(xgb_wf, xgb_params)
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

readr::write_csv(predictions_df, final_model_predictions_path)

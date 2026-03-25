raw_train_path <- "data/raw/train.csv"
raw_test_path <- "data/raw/test.csv"

train_split_path <- "data/processed/train_split.csv"
test_split_path <- "data/processed/test_split.csv"

model_assessment_metrics_path <- "data/processed/model_assessment_cv_metrics.csv"
lasso_penalty_by_fold_path <- "data/processed/lasso_nested_penalty_by_fold.csv"
xgboost_best_params_path <- "data/processed/xgboost_best_params.csv"

model_test_metrics_path <- "output/model_test_metrics.csv"
model_test_plot_path <- "output/model_test_yardsticks_bar.png"

final_model_predictions_path <- "output/final_model_predictions.csv"

default_train_path <- if (file.exists(train_split_path)) train_split_path else raw_train_path
default_test_path <- if (file.exists(raw_test_path)) raw_test_path else test_split_path

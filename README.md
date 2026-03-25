# Wine Quality Classification Pipeline

This project builds and evaluates wine `quality` classification models in R with tidymodels, including a tuned boosted-tree model (`xgboost`).

## What it does
- Splits raw data into train/test sets (stratified by `quality`).
- Trains and compares models:
  - multinomial logistic regression (unpenalized; `nnet`)
  - lasso-regularized multinomial logistic regression (`glmnet`)
  - boosted trees (`xgboost`)
  - intercept-only baseline (`nnet`)
- Evaluates with:
  - **accuracy** (higher is better)
  - **ord_mae** (mean absolute error on ordered `quality` levels; lower is better)
- Selects the final model by the best cross-validation **ord_mae** and generates predictions for the test data.
- Prints the final model accuracy in the terminal when `quality` is present in the test data.

## Run the pipeline
- `make split` - create train/test split files.
- `make assess` - run cross-validation model assessment (accuracy + ord_mae).
- `make test` - evaluate models on the test split (prints metrics table).
- `make final` - fit the selected final model and generate predictions (prints final accuracy if available).
- `make all` - run the full pipeline in sequence.

## Project helpers
- `src/paths.R`: shared file paths for inputs/outputs used by scripts.
- `src/quality_helpers.R`: consistent ordered-factor handling for `quality`.
- `src/metrics.R`: shared metric definitions (accuracy + `ord_mae_vec()` helper).

## Notes on packages
- Scripts will install missing R packages automatically.
- To avoid system-library permission issues, packages are installed into a project-local library at `renv/library`.

## Main outputs
- `data/processed/train_split.csv`
- `data/processed/test_split.csv`
- `data/processed/model_assessment_cv_metrics.csv`
- `data/processed/lasso_nested_penalty_by_fold.csv`
- `data/processed/xgboost_best_params.csv`
- `output/model_test_metrics.csv`
- `output/model_test_yardsticks_bar.png`
- `output/final_model_predictions.csv`

# Wine Quality Classification Pipeline

This project builds and evaluates multinomial logistic classification models for wine `quality` in R with tidymodels.

## What it does
- Splits raw data into train and test sets.
- Trains and compares three models: lasso multinomial, full multinomial, and intercept-only multinomial.
- Uses **accuracy** (percent correctly classified) as the predictive metric for cross-validation and test evaluation.
- Selects the final model by highest cross-validation accuracy and generates class predictions for test data.

## Run the pipeline
- `make split` - create train/test split files.
- `make assess` - run cross-validation model assessment.
- `make test` - evaluate models on the test split (prints accuracy table).
- `make final` - fit the selected final model and generate predictions.
- `make all` - run the full pipeline in sequence.

## Main outputs
- `data/processed/train_split.csv`
- `data/processed/test_split.csv`
- `data/processed/model_assessment_cv_metrics.csv`
- `data/processed/lasso_nested_penalty_by_fold.csv`
- `output/model_test_metrics.csv`
- `output/model_test_yardsticks_bar.png`
- `output/final_model_predictions.csv`

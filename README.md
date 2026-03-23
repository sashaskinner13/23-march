This is a project for our ECN372 class. It builds and evaluates wine-quality prediction models in R using the tidymodels ecosystem.
It splits raw data into train/test sets, compares lasso, full OLS, and intercept-only baselines with 5-fold cross-validation and test-set metrics, then selects the best model by RMSE to generate final predictions.
Automation is provided through a Makefile (`make split`, `make assess`, `make test`, `make final`, `make all`), and key outputs are written to `data/processed/` and `output/`.

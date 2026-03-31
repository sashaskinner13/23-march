# Makefile for the wine-quality classification pipeline.
# Run from the project root. Each target invokes the matching R script via Rscript.

.DEFAULT_GOAL := help
.PHONY: help split assess test final all

# Print available targets and short descriptions.
help:
	@echo "Available targets:"
	@echo "  make split   - Create train/test split files"
	@echo "  make assess  - Run multinomial CV assessment (accuracy)"
	@echo "  make test    - Run multinomial test evaluation (accuracy)"
	@echo "  make final   - Fit best model by CV accuracy and predict test data"
	@echo "  make all     - Run split + assess + test + final"

# Stratified train/test split from raw training data.
split:
	Rscript scripts/split_train_test.R

# Cross-validated model comparison and hyperparameter artifacts.
assess:
	Rscript scripts/model-assessment.R

# Fit models on train split and evaluate on test split; write metrics + plot.
test:
	Rscript scripts/model-test.R

# Select best CV model by ord_mae, refit, predict test, write predictions CSV.
final:
	Rscript scripts/final_model.R

# Full pipeline in order: split → assess → test → final.
all: split assess test final

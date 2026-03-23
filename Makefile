.DEFAULT_GOAL := help
.PHONY: help split assess test final all

help:
	@echo "Available targets:"
	@echo "  make split   - Create train/test split files"
	@echo "  make assess  - Run multinomial CV assessment (accuracy)"
	@echo "  make test    - Run multinomial test evaluation (accuracy)"
	@echo "  make final   - Fit best model by CV accuracy and predict test data"
	@echo "  make all     - Run split + assess + test + final"

split:
	Rscript scripts/split_train_test.R

assess:
	Rscript scripts/model-assessment.R

test:
	Rscript scripts/model-test.R

final:
	Rscript scripts/final_model.R

all: split assess test final

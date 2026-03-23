.PHONY: split assess test final all

split:
	Rscript scripts/split_train_test.R

assess:
	Rscript scripts/model-assessment.R

test:
	Rscript scripts/model-test.R

final:
	Rscript scripts/final_model.R

all: split assess test final

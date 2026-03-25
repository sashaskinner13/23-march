#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  source("src/load_packages.R")
  source("src/paths.R")
})

set.seed(123)

df <- readr::read_delim(
  file = raw_train_path,
  delim = ";",
  show_col_types = FALSE
)

split_obj <- rsample::initial_split(df, prop = 0.8, strata = quality)
train_split <- rsample::training(split_obj)
test_split <- rsample::testing(split_obj)

readr::write_delim(train_split, train_split_path, delim = ";")
readr::write_delim(test_split, test_split_path, delim = ";")


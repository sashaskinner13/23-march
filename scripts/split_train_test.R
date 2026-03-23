#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  source("src/load_packages.R")
})

set.seed(123)

input_path <- "data/raw/train.csv"
train_output_path <- "data/processed/train_split.csv"
test_output_path <- "data/processed/test_split.csv"

df <- readr::read_delim(
  file = input_path,
  delim = ";",
  show_col_types = FALSE
)

split_obj <- rsample::initial_split(df, prop = 0.8)
train_split <- rsample::training(split_obj)
test_split <- rsample::testing(split_obj)

readr::write_delim(train_split, train_output_path, delim = ";")
readr::write_delim(test_split, test_output_path, delim = ";")


#!/usr/bin/env Rscript

# ------------------------------------------------------------------------------
# Purpose
# - Create a reproducible train/test split from the raw training dataset.
#
# Inputs
# - `raw_train_path` (from `src/paths.R`): `data/raw/train.csv` (semicolon-delimited)
# - Required columns: `quality` (used for stratified splitting)
#
# Outputs
# - `train_split_path`: `data/processed/train_split.csv` (semicolon-delimited)
# - `test_split_path`:  `data/processed/test_split.csv`  (semicolon-delimited)
#
# How to run
# - `make split`
# - `Rscript scripts/split_train_test.R`
#
# Reproducibility
# - Uses a fixed seed for a stable split across runs.
# ------------------------------------------------------------------------------

# Load shared helpers: install/load packages and define path constants used below.
suppressPackageStartupMessages({
  source("src/load_packages.R")
  source("src/paths.R")
})

# ---- Load data ----------------------------------------------------------------
# Fix RNG so the same split is reproduced on every run.
set.seed(123)

# Read the raw training CSV (semicolon-separated) into a data frame.
df <- readr::read_delim(
  file = raw_train_path,
  delim = ";",
  show_col_types = FALSE
)

# ---- Stratified split and write artifacts -------------------------------------
# 80% train / 20% test, stratified on `quality` so class proportions match.
split_obj <- rsample::initial_split(df, prop = 0.8, strata = quality)
train_split <- rsample::training(split_obj)
test_split <- rsample::testing(split_obj)

# Persist both splits as semicolon-delimited CSVs for downstream modeling scripts.
readr::write_delim(train_split, train_split_path, delim = ";")
readr::write_delim(test_split, test_split_path, delim = ";")


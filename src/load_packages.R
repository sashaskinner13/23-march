# ------------------------------------------------------------------------------
# Purpose
# - Shared entrypoint sourced by pipeline scripts: configure CRAN, use a
#   project-local library, install missing packages if needed, then attach them.
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  # Ensure `install.packages()` has a valid CRAN mirror when none is configured.
  repos <- getOption("repos")
  if (is.null(repos) || length(repos) == 0 || isTRUE(all(repos == "@CRAN@")) || isTRUE(repos[["CRAN"]] == "@CRAN@")) {
    options(repos = c(CRAN = "https://cloud.r-project.org"))
  }

  # Prefer packages under `renv/library` so installs do not require system library writes.
  project_lib <- normalizePath("renv/library", winslash = "/", mustWork = FALSE)
  if (!dir.exists(project_lib)) {
    dir.create(project_lib, recursive = TRUE, showWarnings = FALSE)
  }
  .libPaths(c(project_lib, .libPaths()))

  # Packages required by tidymodels workflows, I/O, and xgboost engine.
  required_packages <- c(
    "tidyverse",
    "tidymodels",
    "janitor",
    "skimr",
    "xgboost"
  )

  # Install only what is not already available in the search path.
  missing_packages <- required_packages[!vapply(
    required_packages,
    requireNamespace,
    FUN.VALUE = logical(1),
    quietly = TRUE
  )]

  if (length(missing_packages) > 0) {
    message(
      "Installing missing required package(s): ",
      paste(missing_packages, collapse = ", ")
    )
    install.packages(missing_packages, dependencies = TRUE, lib = project_lib)
  }

  # Load into the global environment for subsequent `source()`d scripts.
  invisible(lapply(required_packages, library, character.only = TRUE))
})



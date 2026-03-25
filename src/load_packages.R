suppressPackageStartupMessages({
  repos <- getOption("repos")
  if (is.null(repos) || length(repos) == 0 || isTRUE(all(repos == "@CRAN@")) || isTRUE(repos[["CRAN"]] == "@CRAN@")) {
    options(repos = c(CRAN = "https://cloud.r-project.org"))
  }

  project_lib <- normalizePath("renv/library", winslash = "/", mustWork = FALSE)
  if (!dir.exists(project_lib)) {
    dir.create(project_lib, recursive = TRUE, showWarnings = FALSE)
  }
  .libPaths(c(project_lib, .libPaths()))

  required_packages <- c(
    "tidyverse",
    "tidymodels",
    "janitor",
    "skimr",
    "xgboost"
  )

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

  invisible(lapply(required_packages, library, character.only = TRUE))
})



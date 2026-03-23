suppressPackageStartupMessages({
  required_packages <- c(
    "tidyverse",
    "tidymodels",
    "janitor",
    "skimr"
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
    install.packages(missing_packages, dependencies = TRUE)
  }

  invisible(lapply(required_packages, library, character.only = TRUE))
})



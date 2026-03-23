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
    stop(
      paste(
        "Missing required package(s):",
        paste(missing_packages, collapse = ", "),
        "\nInstall them with install.packages(...) before running modeling scripts."
      ),
      call. = FALSE
    )
  }

  invisible(lapply(required_packages, library, character.only = TRUE))
})


as_ordered_quality <- function(x, levels = NULL) {
  if (is.null(levels)) {
    levels <- sort(unique(x))
  }

  ordered(as.character(x), levels = as.character(levels))
}


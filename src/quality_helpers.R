# ------------------------------------------------------------------------------
# Purpose
# - Ensure `quality` is modeled as an ordered factor with consistent levels
#   across train, test, and resampling folds.
# ------------------------------------------------------------------------------

# Coerce to ordered factor; if `levels` is NULL, infer sorted unique values from `x`.
as_ordered_quality <- function(x, levels = NULL) {
  if (is.null(levels)) {
    levels <- sort(unique(x))
  }

  ordered(as.character(x), levels = as.character(levels))
}


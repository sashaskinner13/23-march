# ------------------------------------------------------------------------------
# Purpose
# - Custom ordinal error metric for ordered `quality` levels and a shared
#   yardstick metric set used in tuning (`accuracy` only in `metrics_used`).
# ------------------------------------------------------------------------------

# Mean absolute difference between integer outcome levels (ordinal MAE).
ord_mae_vec <- function(truth, estimate, ...) {
  truth <- as.integer(truth)
  estimate <- as.integer(estimate)
  mean(abs(truth - estimate), na.rm = TRUE)
}

# Default metrics for `tune` and `fit_resamples` (add `ord_mae` separately where needed).
metrics_used <- yardstick::metric_set(yardstick::accuracy)

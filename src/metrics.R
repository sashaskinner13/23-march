ord_mae_vec <- function(truth, estimate, ...) {
  truth <- as.integer(truth)
  estimate <- as.integer(estimate)
  mean(abs(truth - estimate), na.rm = TRUE)
}

metrics_used <- yardstick::metric_set(yardstick::accuracy)

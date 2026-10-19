# validate_R_bvar.R
# Estimate BVAR using the R BVAR package on the same data and hyperparameters
# as our Python implementation, and save the posterior mean for comparison.

library(BVAR)

# ---- 1. Load data ----
df <- read.csv("/home/andres/work/gpm_ideas/brazil_wp/python/data/estimation_data.csv")

# Select 2004Q1 to 2022Q4 and same 5 variables
# dates in csv are like "2004Q1"
df <- df[df$date >= "2004Q1" & df$date <= "2022Q4", ]
var_names <- c("DA_LGDP", "DA_LCPI", "DA_LCPIX", "RS", "DA_LS")
data <- df[, var_names]
data <- na.omit(data)
data <- as.matrix(data)

cat(sprintf("Data: %d obs x %d variables\n", nrow(data), ncol(data)))
cat(sprintf("Variables: %s\n", paste(var_names, collapse=", ")))

# ---- 2. Compute psi (AR residual std) ----
# Same as Python: univariate AR(4) for each variable
p <- 4
M <- ncol(data)
T_raw <- nrow(data)
psi <- numeric(M)

for (j in 1:M) {
  y <- data[(p+1):T_raw, j]
  X_ar <- cbind(1, sapply(1:p, function(lag) data[(p+1-lag):(T_raw-lag), j]))
  b <- solve(t(X_ar) %*% X_ar, t(X_ar) %*% y)
  resid <- y - X_ar %*% b
  psi[j] <- sd(resid) * sqrt((length(resid)-1)/length(resid))  # match ddof=1
  # Actually numpy std with ddof=1 divides by N-1
  psi[j] <- sd(resid)
}
cat(sprintf("Psi: %s\n", paste(round(psi, 3), collapse=", ")))

# ---- 3. Build design matrices ----
Y <- data[(p+1):T_raw, ]
T_eff <- nrow(Y)
X_lags <- do.call(cbind, lapply(1:p, function(lag) {
  data[(p+1-lag):(T_raw-lag), ]
}))
X <- cbind(1, X_lags)

cat(sprintf("Y: %d x %d, X: %d x %d\n", nrow(Y), ncol(Y), nrow(X), ncol(X)))

# ---- 4. R bvar analytical posterior (Minnesota prior, diagonal Omega) ----
# Hyperparameters matching Python
lambda1 <- 0.2   # overall tightness
alpha   <- 2.0   # lag decay (lambda3)
delta   <- c(0, 0, 0, 1, 0)  # prior mean on first own lag

# Build diagonal Omega (prior variances for each coefficient)
# Omega structure: [intercept, var1_lag1, var2_lag1, ..., varM_lag1, var1_lag2, ...]
K_total <- 1 + M * p  # intercept + M*p lag coefficients
omega <- numeric(K_total)
omega[1] <- 1e7  # diffuse intercept (matching R bvar default)

for (lag in 1:p) {
  start_idx <- 1 + M * (lag - 1) + 1  # +1 for R indexing
  end_idx <- 1 + M * lag
  omega[start_idx:end_idx] <- lambda1^2 / (lag^alpha) / psi
}

cat(sprintf("Omega (first 10): %s\n", paste(round(omega[1:10], 6), collapse=", ")))

# Prior mean: zeros except delta_j on own first lag
B_prior <- matrix(0, nrow=K_total, ncol=M)
for (j in 1:M) {
  B_prior[1 + j, j] <- delta[j]  # own lag-1
}

# ---- 5. Compute analytical posterior ----
# B_post = (X'X + Omega_inv)^{-1} (X'Y + Omega_inv * B_prior)
Omega_inv <- diag(1.0 / omega)

XtX <- t(X) %*% X
XtY <- t(X) %*% Y

V_post_inv <- XtX + Omega_inv
V_post <- solve(V_post_inv)
B_post <- V_post %*% (XtY + Omega_inv %*% B_prior)

# ---- 6. Print results ----
cat("\n")
cat("======================================================================\n")
cat("R BVAR POSTERIOR MEAN (analytical, diagonal Omega, no MCMC)\n")
cat("======================================================================\n")

cat("\nOwn lag-1 coefficients B_post[1+j, j]:\n")
for (j in 1:M) {
  cat(sprintf("  %-12s: B[%d,%d] = %8.4f  (delta=%d)\n",
              var_names[j], 1+j, j, B_post[1+j, j], delta[j]))
}

# ---- 7. Full coefficient matrix (first lag block) ----
cat("\n")
cat("======================================================================\n")
cat("FULL FIRST-LAG COEFFICIENT BLOCK B_post[2:(M+1), 1:M]\n")
cat("======================================================================\n")
cat("(rows = lag-1 regressors, cols = equations)\n\n")

# Header
cat(sprintf("%-12s", ""))
for (j in 1:M) cat(sprintf("  %12s", var_names[j]))
cat("\n")
for (j in 1:M) {
  cat(sprintf("%-12s", var_names[j]))
  for (i in 1:M) {
    cat(sprintf("  %12.6f", B_post[1+j, i]))
  }
  cat("\n")
}

# ---- 8. 1-step forecast ----
cat("\n")
cat("======================================================================\n")
cat("1-STEP AHEAD FORECAST from last observation\n")
cat("======================================================================\n")

last_obs <- data[T_raw:(T_raw-p+1), ]  # most recent p obs, reversed
x_last <- c(1, as.vector(t(last_obs)))
fcst <- as.vector(x_last %*% B_post)

cat(sprintf("\n%-12s  %8s  %8s\n", "Variable", "Forecast", "Last obs"))
cat(paste(rep("-", 40), collapse=""), "\n")
for (j in 1:M) {
  cat(sprintf("%-12s  %8.2f  %8.2f\n", var_names[j], fcst[j], data[T_raw, j]))
}

# ---- 9. Save for Python comparison ----
results <- list(
  B_post = B_post,
  V_post = V_post,
  psi = psi,
  omega = omega,
  forecast = fcst
)

saveRDS(results, "/home/andres/work/gpm_ideas/brazil_wp/python/results/evaluation/r_bvar_results.rds")

# Also save as CSV for easy Python reading
write.csv(B_post, "/home/andres/work/gpm_ideas/brazil_wp/python/results/evaluation/r_bvar_B_post.csv",
          row.names=FALSE)
write.csv(data.frame(variable=var_names, psi=psi, omega_lag1=omega[2:(M+1)]),
          "/home/andres/work/gpm_ideas/brazil_wp/python/results/evaluation/r_bvar_psi.csv",
          row.names=FALSE)

cat("\nSaved results to results/evaluation/r_bvar_*.csv\n")

# ---- 10. Now run with the R BVAR package to verify our analytical calc ----
cat("\n")
cat("======================================================================\n")
cat("VERIFICATION: R BVAR package (MCMC-based)\n")
cat("======================================================================\n")

# The BVAR package uses MCMC with hierarchical priors on lambda and alpha.
# To fix lambda and alpha at specific values, set min=max=mode.
# The package also draws psi from a Gamma prior by default.
# To fix psi at our AR-based estimates, set min=max=mode.
mn_prior <- bv_mn(
  lambda = bv_lambda(mode = lambda1, sd = 0.001, min = lambda1 - 1e-6, max = lambda1 + 1e-6),
  alpha  = bv_alpha(mode = alpha,   sd = 0.001, min = alpha - 1e-6,   max = alpha + 1e-6),
  psi    = bv_psi(mode = psi, min = psi - 1e-6, max = psi + 1e-6),
  var    = 1e7,
  b      = delta
)
my_priors <- bv_priors(mn = mn_prior)

# Run BVAR with MCMC (large number of draws to reduce MC error)
set.seed(42)
bvar_fit <- bvar(data, lags=p, priors=my_priors,
                 n_draw=50000, n_burn=10000, verbose=FALSE)

# Extract posterior mean of coefficients
B_mcmc <- coef(bvar_fit, type="mean")
cat("\nMCMC posterior mean — own lag-1 coefficients:\n")
for (j in 1:M) {
  # coef returns (Kp+1, M) with intercept in first row
  cat(sprintf("  %-12s: B[%d,%d] = %8.4f  (analytical: %8.4f)\n",
              var_names[j], 1+j, j, B_mcmc[1+j, j], B_post[1+j, j]))
}

cat("\n")
cat("======================================================================\n")
cat("FULL FIRST-LAG COEFFICIENT BLOCK — MCMC\n")
cat("======================================================================\n")
cat(sprintf("%-12s", ""))
for (j in 1:M) cat(sprintf("  %12s", var_names[j]))
cat("\n")
for (j in 1:M) {
  cat(sprintf("%-12s", var_names[j]))
  for (i in 1:M) {
    cat(sprintf("  %12.6f", B_mcmc[1+j, i]))
  }
  cat("\n")
}

# Print max absolute difference between analytical and MCMC
max_diff <- max(abs(B_post - B_mcmc))
cat(sprintf("\nMax |analytical - MCMC|: %.6f\n", max_diff))

# Save MCMC coefficient matrix too
write.csv(B_mcmc, "/home/andres/work/gpm_ideas/brazil_wp/python/results/evaluation/r_bvar_B_mcmc.csv",
          row.names=FALSE)
cat("\nSaved MCMC results to r_bvar_B_mcmc.csv\n")

"""Empirical-Bayes hyperparameter optimisation for the BVAR.

Selects Minnesota-prior hyperparameters (lambda1, lambda2, ...) by
maximising the analytical log-marginal likelihood under Normal-Inverse-
Wishart conjugacy.

References:
    Giannone, Lenza & Primiceri (2015) "Prior Selection for Vector
    Autoregressions", *Review of Economics and Statistics* 97(2).
    Koop & Korobilis (2010), Eq. 39.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .bayesian import (
    _auto_psi,
    _create_dummy_data,
    _posterior_params,
    _sample_niw,
    estimate_bvar,
)
from .core import build_design_matrices
from .types import (
    BVARDraws,
    BVARPosteriorParams,
    BVARPriorOptions,
    EBResult,
)


# ======================================================================== #
#  Log-marginal likelihood
# ======================================================================== #

def _multivariate_log_gamma(x: float, K: int) -> float:
    """Log of the multivariate gamma function Gamma_K(x).

    Gamma_K(x) = pi^{K(K-1)/4} prod_{i=1}^{K} Gamma(x + (1-i)/2).
    """
    val = K * (K - 1) / 4.0 * np.log(np.pi)
    for i in range(1, K + 1):
        val += gammaln(x + (1 - i) / 2.0)
    return val


def _log_det_pd(M: np.ndarray) -> float:
    """Log-determinant of a positive-definite matrix via Cholesky."""
    L = np.linalg.cholesky(M)
    return 2.0 * np.sum(np.log(np.diag(L)))


def _niw_log_marginal_likelihood(
    Y: np.ndarray,
    X: np.ndarray,
    Y_d: np.ndarray,
    X_d: np.ndarray,
    K: int,
) -> float:
    """Analytical log p(Y | lambda) under NIW conjugacy.

    Computes the integrated likelihood (marginalising out B and Sigma)
    from the prior encoded by (Y_d, X_d) and the actual data (Y, X).

    Formula (Koop & Korobilis 2010, Eq. 39)::

        log p(Y|lam) = -(T*K/2) log(pi)
                      + log Gamma_K(nu_post/2) - log Gamma_K(nu_prior/2)
                      + (nu_prior/2) log|S_prior| - (nu_post/2) log|S_post|
                      + (K/2)(log|Omega_prior| - log|Omega_post|)

    where Omega = (X'X)^{-1} is the coefficient-covariance factor
    (prior or posterior depending on which X is used).

    Args:
        Y, X: Actual data matrices.
        Y_d, X_d: Dummy (prior) data matrices.
        K: Number of endogenous variables.

    Returns:
        log_ml: Scalar log-marginal likelihood.
    """
    T_data = Y.shape[0]

    # Prior parameters (from dummies alone)
    XdtXd = X_d.T @ X_d
    XdtYd = X_d.T @ Y_d
    try:
        Omega_prior = np.linalg.inv(XdtXd)
    except np.linalg.LinAlgError:
        return -1e20
    B_prior = Omega_prior @ XdtYd
    E_prior = Y_d - X_d @ B_prior
    S_prior = E_prior.T @ E_prior
    nu_prior = Y_d.shape[0] - X_d.shape[1]

    # Posterior parameters (augmented data)
    Y_star = np.vstack([Y, Y_d])
    X_star = np.vstack([X, X_d])
    XstXs = X_star.T @ X_star
    XstYs = X_star.T @ Y_star
    try:
        Omega_post = np.linalg.inv(XstXs)
    except np.linalg.LinAlgError:
        return -1e20
    B_post = Omega_post @ XstYs
    E_post = Y_star - X_star @ B_post
    S_post = E_post.T @ E_post
    nu_post = Y_star.shape[0] - X_star.shape[1]

    # Guard: degrees of freedom must be positive
    if nu_prior < K or nu_post < K:
        return -1e20

    # Log-marginal likelihood
    try:
        log_det_S_prior = _log_det_pd(S_prior)
        log_det_S_post = _log_det_pd(S_post)
        log_det_Omega_prior = _log_det_pd(Omega_prior)
        log_det_Omega_post = _log_det_pd(Omega_post)
    except np.linalg.LinAlgError:
        return -1e20

    lml = (
        -(T_data * K / 2.0) * np.log(np.pi)
        + _multivariate_log_gamma(nu_post / 2.0, K)
        - _multivariate_log_gamma(nu_prior / 2.0, K)
        + (nu_prior / 2.0) * log_det_S_prior
        - (nu_post / 2.0) * log_det_S_post
        + (K / 2.0) * (log_det_Omega_prior - log_det_Omega_post)
    )
    return lml


# ======================================================================== #
#  EB objective
# ======================================================================== #

def _eb_objective(
    log_gamma: np.ndarray,
    data: np.ndarray,
    p: int,
    param_names: List[str],
    base_options: BVARPriorOptions,
    K: int,
    Y: np.ndarray,
    X: np.ndarray,
    n_exog: int = 0,
    exog_prior_scales: Optional[np.ndarray] = None,
) -> float:
    """Negative log-marginal likelihood as a function of log-hyperparameters.

    Optimisation is performed in log-space to enforce positivity.
    """
    gamma = np.exp(log_gamma)

    # Update options with current hyperparameters
    opts = BVARPriorOptions(
        lambda1=base_options.lambda1,
        lambda2=base_options.lambda2,
        lambda3=base_options.lambda3,
        lambda4=base_options.lambda4,
        delta=base_options.delta,
        add_sims=base_options.add_sims,
        lambda_soc=base_options.lambda_soc,
        lambda_dio=base_options.lambda_dio,
        bvar_compat=base_options.bvar_compat,
    )
    for i, name in enumerate(param_names):
        setattr(opts, name, gamma[i])

    try:
        Y_d, X_d, _ = _create_dummy_data(opts, K, p, data, n_exog=n_exog,
                                          exog_prior_scales=exog_prior_scales)
        lml = _niw_log_marginal_likelihood(Y, X, Y_d, X_d, K)
    except Exception:
        lml = -1e20

    if not np.isfinite(lml):
        return 1e20
    return -lml


# ======================================================================== #
#  Public API
# ======================================================================== #

@dataclass
class EBConfig:
    """Per-hyperparameter configuration for Empirical Bayes.

    For each hyperparameter, specify ``'optimize'`` or ``'fixed'``.
    If ``'optimize'``, provide an ``initial`` value as the starting point.
    If ``'fixed'``, the value from the base ``BVARPriorOptions`` is used.
    """
    lambda1: str = "optimize"
    lambda1_init: float = 0.2
    lambda2: str = "fixed"
    lambda3: str = "fixed"
    lambda4: str = "fixed"
    lambda_soc: str = "fixed"
    lambda_dio: str = "fixed"


def estimate_eb_bvar(
    data: np.ndarray,
    p: int,
    base_options: Optional[BVARPriorOptions] = None,
    eb_config: Optional[EBConfig] = None,
    n_draws: int = 2000,
    constant: bool = True,
    filter_unstable: bool = True,
    var_names: Optional[List[str]] = None,
    seed: Optional[int] = None,
    exog: Optional[np.ndarray] = None,
    exog_prior_scales: Optional[np.ndarray] = None,
) -> EBResult:
    """Empirical-Bayes BVAR: optimise hyperparameters then sample.

    Maximises the log-marginal likelihood over the subset of
    hyperparameters marked as ``'optimize'`` in *eb_config*, then
    calls :func:`bayesian.estimate_bvar` with the optimised values.

    Args:
        data: (T_raw, K) array or DataFrame.
        p: Lag order.
        base_options: Base prior options (fixed values + starting points).
        eb_config: Which hyperparameters to optimise.
        n_draws: Posterior draws after optimisation.
        constant: Include intercept.
        filter_unstable: Discard explosive draws.
        var_names: Variable names.
        seed: Random seed.
        exog: (T_raw, m) exogenous regressors or None.
        exog_prior_scales: (m,) prior std-dev scales for exog coefficients.

    Returns:
        EBResult with optimised hyperparameters, draws, and diagnostics.
    """
    import pandas as pd

    if base_options is None:
        base_options = BVARPriorOptions()
    if eb_config is None:
        eb_config = EBConfig()

    if isinstance(data, pd.DataFrame):
        if var_names is None:
            var_names = list(data.columns)
        data = data.values
    data = np.asarray(data, dtype=np.float64)
    K = data.shape[1]

    if base_options.delta is None:
        base_options.delta = np.zeros(K)

    # Build design matrices once
    vd = build_design_matrices(data, p, constant=constant,
                               var_names=var_names, exog=exog)
    Y, X, info = vd

    # Identify which hyperparameters to optimise
    all_params = ["lambda1", "lambda2", "lambda3", "lambda4",
                  "lambda_soc", "lambda_dio"]
    opt_names = []
    init_vals = []
    for name in all_params:
        mode = getattr(eb_config, name, "fixed")
        if mode == "optimize":
            init_attr = f"{name}_init"
            init_val = getattr(eb_config, init_attr,
                               getattr(base_options, name))
            opt_names.append(name)
            init_vals.append(init_val)

    if len(opt_names) == 0:
        # Nothing to optimise - just estimate with base options
        draws, post, diag = estimate_bvar(
            data, p, base_options, n_draws, constant,
            filter_unstable, var_names, seed, exog=exog,
            exog_prior_scales=exog_prior_scales,
        )
        return EBResult(
            posterior_draws=draws,
            posterior_params=post,
            optimised_params={},
            log_ml=np.nan,
            info=diag,
        )

    log_init = np.log(np.array(init_vals))

    # Optimise (in log-space; bounds prevent extreme values)
    bounds = [(np.log(1e-3), np.log(10.0))] * len(log_init)
    result = minimize(
        _eb_objective,
        log_init,
        args=(data, p, opt_names, base_options, K, Y, X, info.n_exog,
              exog_prior_scales),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    # Extract optimised values
    opt_gamma = np.exp(result.x)
    opt_dict = {}
    final_options = BVARPriorOptions(
        lambda1=base_options.lambda1,
        lambda2=base_options.lambda2,
        lambda3=base_options.lambda3,
        lambda4=base_options.lambda4,
        delta=base_options.delta,
        add_sims=base_options.add_sims,
        lambda_soc=base_options.lambda_soc,
        lambda_dio=base_options.lambda_dio,
        bvar_compat=base_options.bvar_compat,
    )
    for i, name in enumerate(opt_names):
        setattr(final_options, name, float(opt_gamma[i]))
        opt_dict[name] = float(opt_gamma[i])

    log_ml = -result.fun

    print(f"  EB optimisation converged: {result.success}")
    print(f"  Log-marginal likelihood: {log_ml:.4f}")
    for name, val in opt_dict.items():
        print(f"    {name} = {val:.6f}")

    # Estimate with optimised hyperparameters
    draws, post, diag = estimate_bvar(
        data, p, final_options, n_draws, constant,
        filter_unstable, var_names, seed, exog=exog,
        exog_prior_scales=exog_prior_scales,
    )

    diag["eb_result"] = result
    diag["eb_opt_names"] = opt_names

    return EBResult(
        posterior_draws=draws,
        posterior_params=post,
        optimised_params=opt_dict,
        log_ml=log_ml,
        info=diag,
    )

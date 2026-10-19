"""Bayesian VAR estimation with Minnesota (dummy-observation) priors.

Implements the conjugate Normal-Inverse-Wishart approach with optional
Sims (1993) Sum-of-Coefficients and Dummy-Initial-Observation priors.
Posterior draws are obtained by direct simulation (no MCMC required).

References:
    Litterman (1986), Doan, Litterman & Sims (1984),
    Sims (1993), Kadiyala & Karlsson (1997),
    Giannone, Lenza & Primiceri (2015).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from .core import build_design_matrices, is_stable
from .types import BVARDraws, BVARPosteriorParams, BVARPriorOptions, VARInfo


# ======================================================================== #
#  Innovation scales
# ======================================================================== #

def _auto_psi(data: np.ndarray, p: int) -> np.ndarray:
    """Estimate per-variable innovation scales via univariate AR(p).

    For each series j, fits  y_t = a_0 + a_1 y_{t-1} + ... + a_p y_{t-p} + e_t
    by OLS and returns std(e).

    Args:
        data: (T_raw, K) raw data matrix.
        p: Lag order.

    Returns:
        psi: (K,) array of residual standard deviations.
    """
    T_raw, K = data.shape
    psi = np.ones(K)
    if T_raw <= p:
        return psi

    for j in range(K):
        y = data[p:, j]                          # (T-p,)
        X_ar = np.ones((T_raw - p, p + 1))       # intercept + p lags
        for lag in range(1, p + 1):
            X_ar[:, lag] = data[p - lag: T_raw - lag, j]
        b = np.linalg.lstsq(X_ar, y, rcond=None)[0]
        resid = y - X_ar @ b
        s = np.std(resid, ddof=1)
        psi[j] = max(s, 1e-12)                   # guard against zero
    return psi


# ======================================================================== #
#  Dummy observations
# ======================================================================== #

def _create_dummy_data(
    options: BVARPriorOptions,
    K: int,
    p: int,
    data: np.ndarray,
    n_exog: int = 0,
    exog_prior_scales: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate dummy observations encoding the Minnesota prior.

    The dummy-observation approach yields Y_d, X_d such that the
    augmented-data OLS estimate is the posterior mean of the
    Normal-Inverse-Wishart conjugate update.

    The implementation supports two formula variants:

    * ``bvar_compat=True`` (default): matches the R ``bvar`` package.
    * ``bvar_compat=False``: legacy Octave GPM-toolbox formulas.

    Optionally appends Sims (1993) SOC and DIO dummy rows when
    ``options.add_sims`` is ``True``.

    When *exog_prior_scales* is provided, Minnesota-style dummy
    observations are added for the exogenous regressors, shrinking
    coefficients toward zero with precision proportional to
    ``1 / (lambda_exog * exog_prior_scales[j])^2`` for column j.
    This implements lag-decaying priors on exogenous lags.

    Args:
        options: BVARPriorOptions instance.
        K: Number of endogenous variables.
        p: Lag order.
        data: (T_raw, K) raw data.
        n_exog: Number of exogenous regressors.
        exog_prior_scales: (n_exog,) array of per-column prior
            standard-deviation scales.  Larger values -> looser prior.
            Typical usage: scale = sigma_z / (1 + lag_index)^lambda3
            for lag-decaying commodity effects.  If ``None``, a diffuse
            prior is used (backward-compatible).

    Returns:
        (Y_d, X_d, psi):
          - Y_d : (n_d, K)  dummy response rows.
          - X_d : (n_d, k)  dummy regressor rows,  k = K*p + int(constant).
          - psi : (K,) innovation scales.
    """
    psi = _auto_psi(data, p)
    lam1 = options.lambda1
    lam2 = options.lambda2
    lam3 = options.lambda3
    lam4 = options.lambda4
    delta = options.delta if options.delta is not None else np.zeros(K)

    constant = True  # dummy data always built with intercept column
    k = K * p + 1    # number of regressor columns (with constant)

    rows_Y = []
    rows_X = []

    if options.bvar_compat:
        # ---- R BVAR-compatible Minnesota dummies ----
        #
        # Matches the R BVAR package (Kuschnig & Vashold 2021).
        # The conditional prior is  vec(B)|Sigma ~ N(b, Sigma x Omega)
        # where Omega = diag(omega) with:
        #
        #   omega[j, lag_l] = lam1^2 / (l^lam3 * psi_j)
        #   omega[intercept] = lam4    (prior variance parameter)
        #
        # Each own-lag dummy row encodes precision 1/omega via:
        #   scale_x = sqrt(l^lam3 * psi_j) / lam1
        # so that (X_d'X_d)[col,col] = scale_x^2 = l^lam3 * psi_j / lam1^2
        # and Omega[col,col] = 1/scale_x^2 = lam1^2 / (l^lam3 * psi_j).
        #
        # Cross-variable shrinkage comes entirely from Sigma x Omega;
        # the cross rows below add extra tightening controlled by lam2.
        # Setting lam2 >> 1 effectively disables them (pure R bvar).
        for lag in range(1, p + 1):
            for j in range(K):
                yd_row = np.zeros(K)
                xd_row = np.zeros(k)
                scale_x = np.sqrt(lag**lam3 * psi[j]) / lam1
                col = 1 + (lag - 1) * K + j  # position of own lag
                xd_row[col] = scale_x
                # Prior mean: delta_j on own first lag
                yd_row[j] = delta[j] * scale_x if lag == 1 else 0.0
                rows_Y.append(yd_row)
                rows_X.append(xd_row)

        # Cross-variable shrinkage (off-diagonal)
        # Extra dummy rows that tighten cross-equation coefficients
        # toward zero.  The R BVAR package does NOT include these
        # (cross-variable shrinkage comes purely from Sigma x Omega).
        # Setting lam2 >> 1 removes their influence, recovering R's
        # diagonal-Omega prior exactly.
        for lag in range(1, p + 1):
            for i in range(K):
                for j in range(K):
                    if i == j:
                        continue
                    yd_row = np.zeros(K)
                    xd_row = np.zeros(k)
                    scale_x = np.sqrt(lag**lam3 * psi[i]) / (lam1 * lam2)
                    col = 1 + (lag - 1) * K + j
                    xd_row[col] = scale_x
                    # Own-equation prior mean: delta_j on first lag
                    yd_row[j] = delta[j] * scale_x if lag == 1 else 0.0
                    rows_Y.append(yd_row)
                    rows_X.append(xd_row)

    else:
        # ---- Legacy GPM-toolbox Minnesota dummies ----
        for lag in range(1, p + 1):
            for j in range(K):
                yd_row = np.zeros(K)
                xd_row = np.zeros(k)
                scale_own = lam1 / (lag**lam3)
                col = 1 + (lag - 1) * K + j
                xd_row[col] = scale_own
                yd_row[j] = delta[j] * scale_own if lag == 1 else 0.0
                rows_Y.append(yd_row)
                rows_X.append(xd_row)

                # Cross-variable entries
                for i in range(K):
                    if i == j:
                        continue
                    yd_cross = np.zeros(K)
                    xd_cross = np.zeros(k)
                    scale_cross = (lam1 * lam2 * psi[i]) / (psi[j] * lag**lam3)
                    xd_cross[col] = scale_cross
                    # Own-equation prior mean: delta_j on first lag
                    yd_cross[j] = delta[j] * scale_cross if lag == 1 else 0.0
                    rows_Y.append(yd_cross)
                    rows_X.append(xd_cross)

    # Intercept dummy
    yd_int = np.zeros(K)
    xd_int = np.zeros(k)
    if options.bvar_compat:
        # In bvar_compat mode, lam4 is the prior variance on the intercept
        # (matching R BVAR's 'var' parameter, default 1e7).
        # Precision = 1/lam4, scale = 1/sqrt(lam4).
        xd_int[0] = 1.0 / np.sqrt(lam4)
    else:
        xd_int[0] = 1.0 / lam4
    rows_Y.append(yd_int)
    rows_X.append(xd_int)

    # ---- Sims priors (optional) ----
    if options.add_sims:
        y_mean = np.nanmean(data[:p, :], axis=0)  # mean of first p obs

        # Sum-of-Coefficients (SOC): K rows
        for j in range(K):
            yd_soc = np.zeros(K)
            xd_soc = np.zeros(k)
            yd_soc[j] = y_mean[j] / options.lambda_soc
            for lag in range(1, p + 1):
                col = 1 + (lag - 1) * K + j
                xd_soc[col] = y_mean[j] / options.lambda_soc
            # No intercept contribution
            rows_Y.append(yd_soc)
            rows_X.append(xd_soc)

        # Dummy-Initial-Observation (DIO): 1 row
        yd_dio = y_mean / options.lambda_dio
        xd_dio = np.zeros(k)
        xd_dio[0] = 1.0 / options.lambda_dio  # intercept
        for lag in range(1, p + 1):
            for j in range(K):
                col = 1 + (lag - 1) * K + j
                xd_dio[col] = y_mean[j] / options.lambda_dio
        rows_Y.append(yd_dio)
        rows_X.append(xd_dio)

    Y_d = np.array(rows_Y)  # (n_d, K)
    X_d = np.array(rows_X)  # (n_d, k)

    # Exogenous regressors: pad existing dummies with zero columns,
    # then add prior dummy observations for the exogenous coefficients.
    #
    # If exog_prior_scales is provided, the prior for exog column j is:
    #   D[:,j] ~ N(0, Sigma (x) (lambda_exog * scale_j)^{-2})
    # implemented via a dummy row with precision 1/(lambda_exog * scale_j).
    #
    # If not provided, a diffuse prior is used (eps = 1e-4 precision).
    if n_exog > 0:
        n_d = X_d.shape[0]
        k_endog = X_d.shape[1]
        X_d = np.hstack([X_d, np.zeros((n_d, n_exog))])

        lam_exog = options.lambda_exog

        for j in range(n_exog):
            x_row = np.zeros(k_endog + n_exog)
            if exog_prior_scales is not None:
                # Minnesota-style: precision = 1/(lambda_exog * scale_j)
                sigma_j = lam_exog * exog_prior_scales[j]
                precision_j = 1.0 / max(sigma_j, 1e-12)
            else:
                # Diffuse fallback
                precision_j = 1e-4
            x_row[k_endog + j] = precision_j
            X_d = np.vstack([X_d, x_row])
            Y_d = np.vstack([Y_d, np.zeros(K)])

    return Y_d, X_d, psi


# ======================================================================== #
#  Analytical posterior
# ======================================================================== #

def _posterior_params(
    Y: np.ndarray,
    X: np.ndarray,
    Y_d: np.ndarray,
    X_d: np.ndarray,
) -> BVARPosteriorParams:
    """Compute analytical Normal-Inverse-Wishart posterior parameters.

    Augments the actual data (Y, X) with dummy observations (Y_d, X_d)
    and computes the posterior analytically.

    The posterior is::

        Sigma | Y*  ~  IW(S_post, df_post)
        vec(B) | Sigma, Y*  ~  N(vec(B_post),  Sigma (x) V_post)

    Args:
        Y: (T, K) actual response matrix.
        X: (T, k) actual regressor matrix.
        Y_d: (n_d, K) dummy response rows.
        X_d: (n_d, k) dummy regressor rows.

    Returns:
        BVARPosteriorParams.
    """
    Y_star = np.vstack([Y, Y_d])
    X_star = np.vstack([X, X_d])

    T_star = X_star.shape[0]
    k = X_star.shape[1]

    XtX = X_star.T @ X_star
    XtY = X_star.T @ Y_star

    V_post = np.linalg.inv(XtX)
    B_post = V_post @ XtY

    E_star = Y_star - X_star @ B_post
    S_post = E_star.T @ E_star

    df_post = T_star - k

    return BVARPosteriorParams(
        B_post=B_post, V_post=V_post, S_post=S_post,
        df_post=df_post, T_star=T_star,
    )


# ======================================================================== #
#  Posterior sampling
# ======================================================================== #

def _sample_wishart(df: int, scale: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw from Wishart(df, scale) via Bartlett decomposition.

    Args:
        df: Degrees of freedom (>= K).
        scale: (K, K) positive-definite scale matrix.
        rng: NumPy random generator.

    Returns:
        (K, K) Wishart draw.
    """
    K = scale.shape[0]
    L = cholesky(scale, lower=True)
    # Bartlett decomposition
    A = np.zeros((K, K))
    for i in range(K):
        A[i, i] = np.sqrt(rng.chisquare(df - i))
        for j in range(i):
            A[i, j] = rng.standard_normal()
    W = L @ A @ A.T @ L.T
    return W


def _sample_niw(
    posterior_params: BVARPosteriorParams,
    n_draws: int,
    K: int,
    p: int,
    constant: bool,
    var_names: Optional[List[str]],
    rng: np.random.Generator,
) -> BVARDraws:
    """Draw from the Normal-Inverse-Wishart posterior.

    For each draw i = 1, ..., n_draws:
        1. Sigma_i  ~ IW(S_post, df_post)
        2. B_i | Sigma_i  ~ MN(B_post, V_post, Sigma_i)

    Args:
        posterior_params: Analytical posterior from ``_posterior_params``.
        n_draws: Number of draws.
        K: Number of variables.
        p: Lag order.
        constant: Whether intercept is included.
        var_names: Variable names.
        rng: NumPy random generator.

    Returns:
        BVARDraws.
    """
    B_post = posterior_params.B_post
    V_post = posterior_params.V_post
    S_post = posterior_params.S_post
    df = posterior_params.df_post

    k = B_post.shape[0]  # Kp + constant

    # Cholesky of V_post for coefficient sampling
    L_V = cholesky(V_post, lower=True)

    # Scale for Wishart (inverse of S_post)
    S_inv = np.linalg.inv(S_post)
    S_inv = (S_inv + S_inv.T) / 2  # enforce symmetry

    B_draws = []
    Sigma_draws = []

    for _ in range(n_draws):
        # Step 1: Sigma ~ IW(S_post, df) = inv(Wishart(df, S_post^{-1}))
        W = _sample_wishart(df, S_inv, rng)
        Sigma_i = np.linalg.inv(W)
        Sigma_i = (Sigma_i + Sigma_i.T) / 2  # enforce symmetry

        # Step 2: B | Sigma ~ MN(B_post, V_post, Sigma)
        L_Sigma = cholesky(Sigma_i, lower=True)
        Z = rng.standard_normal((k, K))
        B_i = B_post + L_V @ Z @ L_Sigma.T

        B_draws.append(B_i)
        Sigma_draws.append(Sigma_i)

    return BVARDraws(
        B_draws=B_draws, Sigma_draws=Sigma_draws,
        n_draws=n_draws, K=K, p=p,
        constant=constant, var_names=var_names,
    )


# ======================================================================== #
#  Public API
# ======================================================================== #

def estimate_bvar(
    data: np.ndarray,
    p: int,
    options: Optional[BVARPriorOptions] = None,
    n_draws: int = 2000,
    constant: bool = True,
    filter_unstable: bool = True,
    var_names: Optional[List[str]] = None,
    seed: Optional[int] = None,
    exog: Optional[np.ndarray] = None,
    exog_prior_scales: Optional[np.ndarray] = None,
) -> Tuple[BVARDraws, BVARPosteriorParams, Dict]:
    """Estimate a Bayesian VAR(p) with Minnesota priors.

    Pipeline:
        1. Build design matrices (Y, X) from *data*.
        2. Generate dummy observations encoding the Minnesota prior.
        3. Compute analytical NIW posterior parameters.
        4. Draw from the posterior via direct simulation.
        5. Optionally discard explosive draws (spectral radius >= 1).

    Args:
        data: (T_raw, K) array or DataFrame.
        p: Lag order.
        options: BVARPriorOptions (defaults are used if ``None``).
        n_draws: Number of posterior draws.
        constant: Include intercept.
        filter_unstable: Discard draws whose companion matrix is explosive.
        var_names: Variable names.
        seed: Random seed for reproducibility.
        exog: Optional (T_raw, m) exogenous regressors (e.g. COVID
              dummies).  Appended to X but not lagged.
        exog_prior_scales: Optional (m,) prior standard-deviation scales
              for exogenous coefficients.  Passed to ``_create_dummy_data``.
              Larger values -> looser prior.

    Returns:
        (draws, posterior_params, info):
          - draws : BVARDraws with (possibly filtered) posterior samples.
          - posterior_params : BVARPosteriorParams.
          - info : dict with diagnostic information.
    """
    import pandas as pd

    if options is None:
        options = BVARPriorOptions()

    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        if var_names is None:
            var_names = list(data.columns)
        data = data.values
    data = np.asarray(data, dtype=np.float64)

    T_raw, K = data.shape

    # Default delta
    if options.delta is None:
        options.delta = np.zeros(K)
    elif len(options.delta) != K:
        raise ValueError(
            f"options.delta has length {len(options.delta)}, expected {K}")

    # 1. Design matrices
    vd = build_design_matrices(data, p, constant=constant,
                               var_names=var_names, exog=exog)
    Y, X, info = vd

    # 2. Dummy observations (with Minnesota-style exog priors if scales given)
    Y_d, X_d, psi = _create_dummy_data(options, K, p, data,
                                        n_exog=info.n_exog,
                                        exog_prior_scales=exog_prior_scales)

    # 3. Posterior parameters
    post = _posterior_params(Y, X, Y_d, X_d)

    # 4. Sample
    rng = np.random.default_rng(seed)
    draws = _sample_niw(post, n_draws, K, p, constant, info.var_names, rng)

    # 5. Filter unstable draws
    n_total = n_draws
    offset = int(constant)
    Kp = K * p
    if filter_unstable:
        keep_B = []
        keep_S = []
        for i in range(draws.n_draws):
            B_i = draws.B_draws[i]
            A_i = B_i[offset:offset + Kp, :]  # lag block only (excludes exog)
            stable, _, _ = is_stable(A_i, K, p)
            if stable:
                keep_B.append(draws.B_draws[i])
                keep_S.append(draws.Sigma_draws[i])
        n_kept = len(keep_B)
        if n_kept == 0:
            print("WARNING: no stable draws found; returning all draws.")
        else:
            if n_kept < n_total:
                print(f"  Filtered unstable draws: kept {n_kept} of "
                      f"{n_total} ({100*n_kept/n_total:.1f}%)")
            draws = BVARDraws(
                B_draws=keep_B, Sigma_draws=keep_S,
                n_draws=n_kept, K=K, p=p,
                constant=constant, var_names=info.var_names,
                n_exog=info.n_exog,
            )

    diag = {
        "T_raw": T_raw,
        "K": K,
        "p": p,
        "T_eff": info.T,
        "n_dummy": Y_d.shape[0],
        "n_draws_requested": n_total,
        "n_draws_kept": draws.n_draws,
        "psi": psi,
        "last_lags": info.last_lags,
        "n_exog": info.n_exog,
    }

    return draws, post, diag

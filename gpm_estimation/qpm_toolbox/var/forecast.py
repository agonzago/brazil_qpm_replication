"""Unconditional and conditional forecasting for classical and Bayesian VARs.

The Waggoner-Zha (1999) conditional-forecast algorithm solves for the
minimum-norm structural shocks that satisfy linear constraints on the
forecast path.

References:
    Waggoner & Zha (1999) "Conditional Forecasts in Dynamic Multivariate
    Models", *Review of Economics and Statistics* 81(4).
"""

from typing import List, Optional

import numpy as np

from .core import cholesky_identification, companion_matrix
from .types import (
    BVARDraws,
    BVARForecastResult,
    ConditionalForecastResult,
    ForecastResult,
    VARModel,
)


# ======================================================================== #
#  Private: single-path forward simulation
# ======================================================================== #

def _simulate_path(
    A: np.ndarray,
    c: Optional[np.ndarray],
    Sigma: np.ndarray,
    last_lags: np.ndarray,
    horizon: int,
    K: int,
    p: int,
    add_shocks: bool = False,
    rng: Optional[np.random.Generator] = None,
    D: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulate one forecast path via companion-form recursion.

    Args:
        A: (Kp, K) lag coefficients.
        c: (K,) intercept or None.
        Sigma: (K, K) residual covariance (used only if add_shocks).
        last_lags: (p, K) matrix of recent history.
                   Row 0 = most recent observation (Y_T).
        horizon: Steps ahead.
        K: Number of variables.
        p: Lag order.
        add_shocks: Draw random shocks (stochastic forecast).
        rng: Random generator (required if add_shocks).
        D: (m, K) exogenous regressor coefficients or None.
        exog_future: (horizon, m) future values of exogenous regressors.
                     Required when D is not None.

    Returns:
        Yf: (horizon, K) forecast path.
    """
    if c is None:
        c = np.zeros(K)

    # Cholesky factor for shock generation
    L = None
    if add_shocks:
        L = np.linalg.cholesky(Sigma)

    # History buffer: store last p observations, most recent first
    buf = last_lags.copy()  # (p, K), row 0 = Y_T

    Yf = np.zeros((horizon, K))
    for h in range(horizon):
        # Build regressor from buffer: [Y_{t-1}; Y_{t-2}; ... ; Y_{t-p}]
        x = buf.ravel()  # (Kp,)  - row 0 is lag-1, row 1 is lag-2, etc.
        y_mean = A.T @ x + c

        # Exogenous contribution
        if D is not None and exog_future is not None:
            y_mean += D.T @ exog_future[h, :]

        if add_shocks and L is not None:
            eps = rng.standard_normal(K)
            y_h = y_mean + L @ eps
        else:
            y_h = y_mean

        Yf[h, :] = y_h

        # Shift buffer: push new observation to front
        buf = np.roll(buf, 1, axis=0)
        buf[0, :] = y_h

    return Yf


# ======================================================================== #
#  Private: Waggoner-Zha conditional-forecast solver
# ======================================================================== #

def _build_impulse_map(
    A: np.ndarray,
    C: np.ndarray,
    K: int,
    p: int,
    horizon: int,
) -> np.ndarray:
    """Build the (K*H x K*H) block-lower-triangular matrix mapping
    stacked structural shocks to the stacked forecast path.

    ``B[t_block, s_block]`` is the KxK impulse response of the VAR
    at horizon *t* to a one-time shock at horizon *s*, for s <= t.

    Args:
        A: (Kp, K) lag coefficients.
        C: (K, K) structural impact matrix (Cholesky).
        K, p: VAR dimensions.
        horizon: Forecast horizon H.

    Returns:
        B_map: (K*H, K*H) impulse-to-path mapping.
    """
    F = companion_matrix(A, K, p)
    H = horizon

    # Pre-compute F^n @ [C; 0] for n = 0 .. H-1
    # At each horizon n, the top-K rows give the KxK IRF at that lag.
    phi = np.zeros((H, K, K))
    state_cols = np.zeros((K * p, K))
    state_cols[:K, :] = C  # embed C in top K rows
    phi[0] = C.copy()
    for n in range(1, H):
        state_cols = F @ state_cols
        phi[n] = state_cols[:K, :]

    # Assemble block-lower-triangular matrix
    B_map = np.zeros((K * H, K * H))
    for t in range(H):
        for s in range(t + 1):
            lag = t - s
            B_map[t * K: (t + 1) * K, s * K: (s + 1) * K] = phi[lag]

    return B_map


def _wz_solve(
    A: np.ndarray,
    c: Optional[np.ndarray],
    Sigma: np.ndarray,
    last_lags: np.ndarray,
    horizon: int,
    K: int,
    p: int,
    R: np.ndarray,
    r: np.ndarray,
    shock_indices: Optional[np.ndarray] = None,
    alpha: float = 0.0,
    D: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
) -> tuple:
    """Solve for minimum-norm structural shocks satisfying constraints.

    Given linear constraints ``R @ vec(Y) = r``, finds the structural
    shock vector ``u`` of minimum norm such that the forecast path
    satisfies the constraints.

    Args:
        A, c, Sigma: VAR coefficient/covariance.
        last_lags: (p, K) history.
        horizon: H.
        K, p: Dimensions.
        R: (m, K*H) constraint selector.
        r: (m,) constraint targets.
        shock_indices: Columns of the (K*H) shock vector that are free.
                       Default: all.
        alpha: Tikhonov regularisation parameter.
        D: (m_exog, K) exogenous coefficients or None.
        exog_future: (H, m_exog) future exogenous values or None.

    Returns:
        (Yc, u_hat):
          - Yc: (H, K) conditional forecast.
          - u_hat: (K*H,) structural shock vector.
    """
    if c is None:
        c = np.zeros(K)

    C = cholesky_identification(Sigma)

    # Baseline (no-shock) forecast
    Y0 = _simulate_path(A, c, Sigma, last_lags, horizon, K, p,
                         add_shocks=False, D=D, exog_future=exog_future)
    y0_vec = Y0.ravel()  # (K*H,)

    # Impulse mapping
    B_map = _build_impulse_map(A, C, K, p, horizon)

    # Restrict to selected shocks
    KH = K * horizon
    if shock_indices is not None:
        BS = B_map[:, shock_indices]
    else:
        BS = B_map
        shock_indices = np.arange(KH)

    # Solve:  R @ BS @ u_sel = r - R @ y0_vec
    Aeq = R @ BS
    b_eq = r - R @ y0_vec

    if alpha > 0:
        # Tikhonov: u_sel = (Aeq'Aeq + alpha*I)^{-1} Aeq' b
        n_sel = Aeq.shape[1]
        u_sel = np.linalg.solve(
            Aeq.T @ Aeq + alpha * np.eye(n_sel), Aeq.T @ b_eq)
    else:
        # Minimum-norm least squares
        u_sel, _, _, _ = np.linalg.lstsq(Aeq, b_eq, rcond=None)

    # Full shock vector
    u_hat = np.zeros(KH)
    u_hat[shock_indices] = u_sel

    # Conditional forecast
    yc_vec = y0_vec + B_map @ u_hat
    Yc = yc_vec.reshape(horizon, K)

    return Yc, u_hat


# ======================================================================== #
#  Classical forecasts
# ======================================================================== #

def forecast_unconditional(
    model: VARModel,
    last_lags: np.ndarray,
    horizon: int,
    exog_future: Optional[np.ndarray] = None,
) -> ForecastResult:
    """Deterministic unconditional forecast from a classical VAR.

    Args:
        model: Estimated VARModel.
        last_lags: (p, K) recent history (row 0 = most recent).
        horizon: Steps ahead.
        exog_future: (horizon, m) future exogenous values or None.

    Returns:
        ForecastResult.
    """
    Yf = _simulate_path(
        model.A, model.c, model.Sigma, last_lags,
        horizon, model.K, model.p, add_shocks=False,
        D=model.D, exog_future=exog_future,
    )
    return ForecastResult(
        forecasts=Yf, var_names=model.var_names, horizon=horizon)


def forecast_conditional_wz(
    model: VARModel,
    last_lags: np.ndarray,
    horizon: int,
    R: np.ndarray,
    r: np.ndarray,
    shock_indices: Optional[np.ndarray] = None,
    alpha: float = 0.0,
    exog_future: Optional[np.ndarray] = None,
) -> ConditionalForecastResult:
    """Waggoner-Zha conditional forecast from a classical VAR.

    Args:
        model: Estimated VARModel.
        last_lags: (p, K) recent history.
        horizon: Steps ahead.
        R: (m, K*H) constraint selector matrix.
        r: (m,) target vector.
        shock_indices: Free shock indices (default: all).
        alpha: Tikhonov regularisation.
        exog_future: (horizon, m) future exogenous values or None.

    Returns:
        ConditionalForecastResult.
    """
    Yc, u_hat = _wz_solve(
        model.A, model.c, model.Sigma, last_lags,
        horizon, model.K, model.p, R, r, shock_indices, alpha,
        D=model.D, exog_future=exog_future,
    )
    return ConditionalForecastResult(
        forecasts=Yc, shocks=u_hat,
        var_names=model.var_names, horizon=horizon,
    )


# ======================================================================== #
#  Bayesian forecasts
# ======================================================================== #

def forecast_bvar(
    draws: BVARDraws,
    last_lags: np.ndarray,
    horizon: int,
    add_shocks: bool = True,
    seed: Optional[int] = None,
    exog_future: Optional[np.ndarray] = None,
) -> BVARForecastResult:
    """Stochastic unconditional forecast over BVAR posterior draws.

    For each draw, simulates one forward path (with or without random
    shocks), producing a distribution of forecasts.

    Args:
        draws: BVARDraws from :func:`bayesian.estimate_bvar`.
        last_lags: (p, K) recent history.
        horizon: Steps ahead.
        add_shocks: Include stochastic innovations.
        seed: Random seed.
        exog_future: (horizon, m) future values of exogenous regressors.

    Returns:
        BVARForecastResult with draw-level forecasts and percentile
        summaries.
    """
    K, p = draws.K, draws.p
    rng = np.random.default_rng(seed)
    offset = int(draws.constant)
    Kp = K * p

    all_fc = np.zeros((draws.n_draws, horizon, K))

    for i in range(draws.n_draws):
        B_i = draws.B_draws[i]
        Sigma_i = draws.Sigma_draws[i]
        c_i = B_i[0, :] if draws.constant else None
        A_i = B_i[offset:offset + Kp, :]
        D_i = B_i[offset + Kp:, :] if draws.n_exog > 0 else None

        all_fc[i] = _simulate_path(
            A_i, c_i, Sigma_i, last_lags, horizon, K, p,
            add_shocks=add_shocks, rng=rng,
            D=D_i, exog_future=exog_future,
        )

    median = np.percentile(all_fc, 50, axis=0)
    p16 = np.percentile(all_fc, 16, axis=0)
    p84 = np.percentile(all_fc, 84, axis=0)

    return BVARForecastResult(
        forecasts=all_fc, median=median, p16=p16, p84=p84,
        var_names=draws.var_names, horizon=horizon,
    )


def forecast_conditional_bvar(
    draws: BVARDraws,
    last_lags: np.ndarray,
    constraints: np.ndarray,
    horizon: Optional[int] = None,
    seed: Optional[int] = None,
    exog_future: Optional[np.ndarray] = None,
) -> BVARForecastResult:
    """Bayesian conditional forecast (per-draw Waggoner-Zha).

    For each posterior draw, solves for the minimum-norm structural
    shocks satisfying the constraints.

    Args:
        draws: BVARDraws.
        last_lags: (p, K) recent history.
        constraints: (H, K) array where ``np.nan`` marks unconstrained
                     entries and finite values are targets.
        horizon: Inferred from *constraints* if ``None``.
        seed: Random seed (unused here but kept for API consistency).
        exog_future: (H, m) future values of exogenous regressors.

    Returns:
        BVARForecastResult.
    """
    K, p = draws.K, draws.p
    offset = int(draws.constant)
    Kp = K * p
    constraints = np.asarray(constraints, dtype=np.float64)
    if horizon is None:
        horizon = constraints.shape[0]

    # Build R, r from constraints matrix
    rows_R = []
    vals_r = []
    for h in range(horizon):
        for j in range(K):
            if np.isfinite(constraints[h, j]):
                row = np.zeros(K * horizon)
                row[h * K + j] = 1.0
                rows_R.append(row)
                vals_r.append(constraints[h, j])

    if len(rows_R) == 0:
        # No constraints -> unconditional forecast
        return forecast_bvar(draws, last_lags, horizon,
                             add_shocks=False, seed=seed,
                             exog_future=exog_future)

    R = np.array(rows_R)
    r = np.array(vals_r)

    all_fc = np.zeros((draws.n_draws, horizon, K))
    all_sh = np.zeros((draws.n_draws, K * horizon))

    for i in range(draws.n_draws):
        B_i = draws.B_draws[i]
        Sigma_i = draws.Sigma_draws[i]
        c_i = B_i[0, :] if draws.constant else None
        A_i = B_i[offset:offset + Kp, :]
        D_i = B_i[offset + Kp:, :] if draws.n_exog > 0 else None

        Yc, u_hat = _wz_solve(
            A_i, c_i, Sigma_i, last_lags,
            horizon, K, p, R, r,
            D=D_i, exog_future=exog_future,
        )
        all_fc[i] = Yc
        all_sh[i] = u_hat

    median = np.percentile(all_fc, 50, axis=0)
    p16 = np.percentile(all_fc, 16, axis=0)
    p84 = np.percentile(all_fc, 84, axis=0)

    return BVARForecastResult(
        forecasts=all_fc, median=median, p16=p16, p84=p84,
        var_names=draws.var_names, horizon=horizon,
    )

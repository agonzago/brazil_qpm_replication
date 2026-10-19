"""Classical (frequentist) VAR estimation.

Provides OLS and ridge-regularised stable VAR(p) estimation.
"""

from typing import List, Optional

import numpy as np

from .core import build_design_matrices, is_stable
from .types import VARModel


def estimate_ols(
    data: np.ndarray,
    p: int,
    constant: bool = True,
    var_names: Optional[List[str]] = None,
    exog: Optional[np.ndarray] = None,
) -> VARModel:
    """Estimate a VAR(p) by ordinary least squares.

    The model is  Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p} + D z_t + u_t
    with u_t ~ N(0, Sigma).

    Args:
        data: (T_raw, K) array or DataFrame.
        p: Lag order.
        constant: Include intercept.
        var_names: Variable names.
        exog: (T_raw, m) exogenous regressors or None.

    Returns:
        VARModel with OLS estimates.
    """
    vd = build_design_matrices(data, p, constant=constant,
                               var_names=var_names, exog=exog)
    Y, X, info = vd

    # OLS: B = (X'X)^{-1} X'Y
    B = np.linalg.solve(X.T @ X, X.T @ Y)  # (k, K)

    E = Y - X @ B  # (T, K) residuals
    k = X.shape[1]
    Sigma = (E.T @ E) / (info.T - k)  # unbiased

    # Extract lag coefficients, intercept, and exogenous coefficients
    offset = int(constant)
    Kp = info.K * info.p
    c = B[0, :] if constant else None
    A = B[offset:offset + Kp, :]
    D = B[offset + Kp:, :] if info.n_exog > 0 else None

    return VARModel(
        B=B, A=A, c=c, Sigma=Sigma, E=E,
        K=info.K, p=info.p, T_eff=info.T,
        constant=constant, var_names=info.var_names,
        n_exog=info.n_exog, D=D,
    )


def estimate_stable(
    data: np.ndarray,
    p: int,
    constant: bool = True,
    max_iter: int = 5,
    ridge_factor: float = 1.5,
    ridge_init: float = 0.01,
    var_names: Optional[List[str]] = None,
    exog: Optional[np.ndarray] = None,
) -> VARModel:
    """Estimate a stable VAR(p) via iterative ridge regularisation.

    Starting from OLS, if the companion spectral radius >= 1 the
    procedure adds a ridge penalty ``lambda * I`` to the lag-coefficient
    block of ``X'X`` (the intercept row is **not** penalised) and
    re-estimates.  ``lambda`` is multiplied by *ridge_factor* each
    iteration.  If no stable solution is found after *max_iter* steps,
    the lag coefficients are rescaled so that the spectral radius equals
    0.99.

    Args:
        data: (T_raw, K) array or DataFrame.
        p: Lag order.
        constant: Include intercept.
        max_iter: Maximum ridge iterations.
        ridge_factor: Multiplicative increase per iteration.
        ridge_init: Starting ridge parameter.
        var_names: Variable names.
        exog: (T_raw, m) exogenous regressors or None.

    Returns:
        VARModel with stable coefficient estimates.
    """
    vd = build_design_matrices(data, p, constant=constant,
                               var_names=var_names, exog=exog)
    Y, X, info = vd
    K, k = info.K, X.shape[1]
    offset = int(constant)
    Kp = K * p

    lam = ridge_init
    for _ in range(max_iter + 1):
        # Ridge matrix: penalise lags, not intercept
        ridge_mat = lam * np.eye(k)
        if constant:
            ridge_mat[0, 0] = 0.0  # no penalty on intercept
        if lam == 0.0:
            B = np.linalg.solve(X.T @ X, X.T @ Y)
        else:
            B = np.linalg.solve(X.T @ X + ridge_mat, X.T @ Y)

        A = B[offset:offset + Kp, :]
        stable, rho, _ = is_stable(A, K, p)
        if stable:
            break
        lam *= ridge_factor
    else:
        # Fallback: rescale A so spectral radius = 0.99
        scale = min(0.99 / rho, 0.99) if rho >= 1.0 else 1.0
        A = A * scale
        B_new = B.copy()
        B_new[offset:offset + Kp, :] = A
        B = B_new

    E = Y - X @ B
    Sigma = (E.T @ E) / (info.T - k)

    c = B[0, :] if constant else None
    D = B[offset + Kp:, :] if info.n_exog > 0 else None

    return VARModel(
        B=B, A=A, c=c, Sigma=Sigma, E=E,
        K=K, p=p, T_eff=info.T,
        constant=constant, var_names=info.var_names,
        n_exog=info.n_exog, D=D,
    )

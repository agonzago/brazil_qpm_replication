"""Shared utilities for VAR estimation, identification, and forecasting.

Functions in this module are consumed by *classical*, *bayesian*,
*irf*, and *forecast* and should never import from them (no circular deps).
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import eig
from scipy.linalg import cholesky

from .types import VARData, VARInfo


# ======================================================================== #
#  Design matrices
# ======================================================================== #

def build_design_matrices(
    data: np.ndarray,
    p: int,
    constant: bool = True,
    var_names: Optional[List[str]] = None,
    exog: Optional[np.ndarray] = None,
) -> VARData:
    """Build lagged response and regressor matrices for a VAR(p).

    Args:
        data: (T_raw, K) array or pandas DataFrame of endogenous variables.
              If a DataFrame, column names are used as *var_names* (unless
              supplied explicitly) and the index is ignored.
        p: Lag order (positive integer).
        constant: If ``True``, prepend a column of ones to *X*.
        var_names: Optional variable-name list (length K).
        exog: Optional (T_raw, m) array of exogenous regressors.
              These are appended to X (not lagged).  Must have the same
              number of rows as *data*.

    Returns:
        VARData(Y, X, info):
          - Y : (T, K) response matrix, T = T_raw - p.
          - X : (T, k) regressor matrix,
                k = K*p + int(constant) + m.
                Column layout: [const | lag-1 | ... | lag-p | exog].
          - info : VARInfo metadata (includes n_exog).
    """
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        if var_names is None:
            var_names = list(data.columns)
        data = data.values

    data = np.asarray(data, dtype=np.float64)
    T_raw, K = data.shape
    if p < 1:
        raise ValueError(f"Lag order p must be >= 1, got {p}")
    if T_raw <= p:
        raise ValueError(
            f"Need T > p observations; got T={T_raw}, p={p}")

    T = T_raw - p  # effective sample size

    # Response matrix
    Y = data[p:, :]  # (T, K)

    # Regressor matrix: stack lag blocks horizontally
    X_blocks = []
    for lag in range(1, p + 1):
        X_blocks.append(data[p - lag: T_raw - lag, :])  # (T, K)
    X_lags = np.hstack(X_blocks)  # (T, K*p)

    if constant:
        X = np.hstack([np.ones((T, 1)), X_lags])  # (T, K*p+1)
    else:
        X = X_lags

    # Append exogenous regressors (if any)
    n_exog = 0
    if exog is not None:
        if isinstance(exog, pd.DataFrame):
            exog = exog.values
        exog = np.asarray(exog, dtype=np.float64)
        if exog.ndim == 1:
            exog = exog[:, np.newaxis]
        if exog.shape[0] != T_raw:
            raise ValueError(
                f"exog has {exog.shape[0]} rows, expected {T_raw} "
                f"(same as data)")
        exog_eff = exog[p:, :]  # (T, m) aligned with Y
        X = np.hstack([X, exog_eff])
        n_exog = exog_eff.shape[1]

    # Store last p rows for forecasting (row 0 = most recent)
    last_lags = data[-p:][::-1]  # (p, K), reversed so row 0 is Y_T

    info = VARInfo(
        K=K, p=p, T=T, constant=constant,
        var_names=var_names, last_lags=last_lags,
        n_exog=n_exog,
    )
    return VARData(Y=Y, X=X, info=info)


# ======================================================================== #
#  Companion matrix
# ======================================================================== #

def companion_matrix(A: np.ndarray, K: int, p: int) -> np.ndarray:
    """Build the (Kp x Kp) VAR companion matrix from lag coefficients.

    Args:
        A: (Kp, K) lag coefficient matrix.
           Rows are ordered [A_1 (K rows); A_2 (K rows); ... ; A_p (K rows)].
        K: Number of endogenous variables.
        p: Lag order.

    Returns:
        F: (Kp, Kp) companion matrix::

            F = [ A_1'   A_2'  ...  A_p' ]
                [ I_K     0    ...   0   ]
                [  0     I_K   ...   0   ]
                [ ...    ...   ...   0   ]
    """
    Kp = K * p
    F = np.zeros((Kp, Kp))
    # First K rows: transpose of A (each A_l block is KxK)
    F[:K, :] = A.T  # A is (Kp, K); A.T is (K, Kp) - exactly first K rows
    # Identity blocks on the sub-diagonal
    if p > 1:
        F[K:, : K * (p - 1)] = np.eye(K * (p - 1))
    return F


# ======================================================================== #
#  Stability check
# ======================================================================== #

def is_stable(
    A: np.ndarray, K: int, p: int, tol: float = 1e-8,
) -> Tuple[bool, float, np.ndarray]:
    """Check VAR stability via the spectral radius of the companion matrix.

    Args:
        A: (Kp, K) lag coefficients.
        K: Number of variables.
        p: Lag order.
        tol: Stability margin (spectral radius must be < 1 - tol).

    Returns:
        (stable, rho, F):
          - stable : ``True`` if max |eigenvalue| < 1 - tol.
          - rho    : spectral radius.
          - F      : companion matrix.
    """
    F = companion_matrix(A, K, p)
    eigenvalues = eig(F)[0]
    rho = np.max(np.abs(eigenvalues))
    return (rho < 1.0 - tol), float(rho), F


# ======================================================================== #
#  Structural identification
# ======================================================================== #

def cholesky_identification(Sigma: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor of the residual covariance.

    The structural impact matrix *C* satisfies ``C @ C.T == Sigma``
    and is used to map unit structural shocks to reduced-form innovations:
    ``u_t = C @ eps_t``.

    Args:
        Sigma: (K, K) positive-definite covariance matrix.

    Returns:
        C: (K, K) lower-triangular Cholesky factor.
    """
    return cholesky(Sigma, lower=True)


# ======================================================================== #
#  Waggoner-Zha constraint selector
# ======================================================================== #

def make_constraint_matrix(
    K: int,
    h: int,
    var_indices: np.ndarray,
    time_indices: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a constraint selector matrix for conditional forecasting.

    Constructs an (m x K*h) matrix *R* where the vectorised forecast
    ``vec(Y)`` is arranged as K-blocks per horizon:
    ``[Y_1^1, Y_1^2, ..., Y_1^K, Y_2^1, ...]``.

    Each row j of *R* selects ``Y_{time_indices[j]}^{var_indices[j]}``
    with weight ``weights[j]``.

    Args:
        K: Number of variables.
        h: Forecast horizon.
        var_indices: (m,) 0-based variable indices.
        time_indices: (m,) 1-based horizon indices (1 .. h).
        weights: (m,) optional weights (default: all ones).

    Returns:
        R: (m, K*h) sparse selector matrix (dense representation).
    """
    var_indices = np.asarray(var_indices, dtype=int)
    time_indices = np.asarray(time_indices, dtype=int)
    m = len(var_indices)
    if weights is None:
        weights = np.ones(m)
    weights = np.asarray(weights, dtype=np.float64)

    R = np.zeros((m, K * h))
    for j in range(m):
        col = (time_indices[j] - 1) * K + var_indices[j]
        R[j, col] = weights[j]
    return R

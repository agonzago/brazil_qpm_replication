"""Time series filters for trend-cycle decomposition.

Provides the Hamilton (2018) regression-based filter as an alternative
to the Hodrick-Prescott filter.

References
----------
Hamilton, J.D. (2018). "Why You Should Never Use the Hodrick-Prescott
    Filter", Review of Economics and Statistics, 100(5), 831-843.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def hamilton_filter(
    y: np.ndarray | pd.Series,
    h: int = 8,
    p: int = 4,
) -> Tuple[np.ndarray | pd.Series, np.ndarray | pd.Series]:
    """Hamilton (2018) regression-based trend-cycle decomposition.

    Regresses y_{t+h} on [1, y_t, y_{t-1}, ..., y_{t-p+1}] and returns
    the residuals as the cyclical component and fitted values as trend.

    Parameters
    ----------
    y : array-like or pd.Series
        Time series (level, typically in logs).  Must have at least
        h + p + 1 non-NaN observations.
    h : int, default 8
        Forecast horizon (quarters ahead).  Hamilton (2018) recommends
        h = 8 for quarterly data (2 years ahead).
    p : int, default 4
        Number of lags in the projection regression (y_t through
        y_{t-p+1}).  Hamilton (2018) recommends p = 4 for quarterly data.

    Returns
    -------
    cycle : same type as *y*
        Cyclical component (residuals from the projection).
        First h + p - 1 observations are NaN.
    trend : same type as *y*
        Trend component (fitted values from the projection).
        First h + p - 1 observations are NaN.

    Notes
    -----
    Unlike the HP filter, the Hamilton filter:

    * does not suffer from spurious cyclicality,
    * does not have an end-of-sample bias,
    * produces stationary residuals under general conditions, and
    * has a clear statistical interpretation as a forecast error.

    The filter requires the first h + p - 1 observations to be lost.
    For quarterly data with h = 8 and p = 4 this means 11 observations
    (~2.75 years).
    """
    is_series = isinstance(y, pd.Series)
    if is_series:
        index = y.index
        y_vals = y.values.astype(float)
    else:
        y_vals = np.asarray(y, dtype=float)

    T = len(y_vals)
    n_lost = h + p - 1

    if T < n_lost + 2:
        raise ValueError(
            f"Series too short: need at least {n_lost + 2} observations, "
            f"got {T}."
        )

    # ------------------------------------------------------------------ #
    #  Build OLS regression
    #    dependent:  y_{t+h}  for t = p-1, ..., T-h-1
    #    regressors: [1, y_t, y_{t-1}, ..., y_{t-p+1}]
    # ------------------------------------------------------------------ #
    n_obs = T - n_lost           # usable regression observations
    Y = y_vals[n_lost:]          # length n_obs

    X = np.ones((n_obs, p + 1))  # column 0 = constant
    for j in range(p):
        start = p - 1 - j
        X[:, j + 1] = y_vals[start:start + n_obs]

    # OLS via least-squares
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ beta
    resid = Y - fitted

    # Reconstruct full-length arrays with NaN padding
    cycle_full = np.full(T, np.nan)
    trend_full = np.full(T, np.nan)
    cycle_full[n_lost:] = resid
    trend_full[n_lost:] = fitted

    if is_series:
        return (pd.Series(cycle_full, index=index),
                pd.Series(trend_full, index=index))
    return cycle_full, trend_full

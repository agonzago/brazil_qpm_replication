"""Data transformation engine.

Applies log, DA, D4A, HP filter, Hamilton (2018) filter, and linear
trend transformations to the consolidated quarterly data.  Mirrors the
R transform_data.R logic exactly (with the Hamilton filter as an
addition).

Transforms
----------
log              : 100 * ln(x)  (note: R uses plain log; we keep 100* for QPM compat)
DA               : 400 * [ln(x_t) - ln(x_{t-1})]  — annualized QoQ growth
D4A              : 100 * [ln(x_t) - ln(x_{t-4})]  — YoY growth
log100_hp        : 100 * HP_cycle(ln(x), lambda=1600)
log100_hamilton  : 100 * Hamilton_cycle(ln(x), h=8, p=4)
log100_lt        : 100 * (ln(x) - linear_trend(ln(x)))
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from .variable_registry import REGISTRY, VariableDef

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_DIR = os.path.normpath(os.path.join(
    _THIS_DIR, "..",
    "estimation_data",
))


# ======================================================================== #
#  Transform functions
# ======================================================================== #

def transform_log(x: pd.Series) -> pd.Series:
    """Natural logarithm (as in the R pipeline: plain log, not 100*)."""
    return np.log(x)


def transform_DA(x: pd.Series) -> pd.Series:
    """Annualized QoQ growth: 400 * [ln(x_t) - ln(x_{t-1})]."""
    logx = np.log(x)
    return 400.0 * logx.diff()


def transform_D4A(x: pd.Series) -> pd.Series:
    """Year-over-year growth: 100 * [ln(x_t) - ln(x_{t-4})]."""
    logx = np.log(x)
    return 100.0 * (logx - logx.shift(4))


def transform_log100_hp(x: pd.Series, lam: float = 1600.0) -> pd.Series:
    """HP filter cycle of log(x), scaled by 100."""
    logx = np.log(x)
    valid = logx.dropna()
    if len(valid) < 10:
        return pd.Series(np.nan, index=x.index)
    cycle, trend = hpfilter(valid.values, lamb=lam)
    result = pd.Series(np.nan, index=x.index)
    result.loc[valid.index] = 100.0 * cycle
    return result


def transform_logHP(x: pd.Series, lam: float = 1600.0) -> pd.Series:
    """HP filter cycle of log(x) (unscaled)."""
    logx = np.log(x)
    valid = logx.dropna()
    if len(valid) < 10:
        return pd.Series(np.nan, index=x.index)
    cycle, trend = hpfilter(valid.values, lamb=lam)
    result = pd.Series(np.nan, index=x.index)
    result.loc[valid.index] = cycle
    return result


def _hamilton_filter(y_vals: np.ndarray, h: int = 8, p: int = 4):
    """Standalone Hamilton (2018) regression-based trend-cycle filter.

    Regresses y_{t+h} on [1, y_t, ..., y_{t-p+1}].  Returns
    (cycle, trend) arrays of the same length as *y_vals* with the
    first h + p - 1 entries set to NaN.
    """
    T = len(y_vals)
    n_lost = h + p - 1
    n_obs = T - n_lost
    Y = y_vals[n_lost:]
    X = np.ones((n_obs, p + 1))
    for j in range(p):
        start = p - 1 - j
        X[:, j + 1] = y_vals[start:start + n_obs]
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ beta
    resid = Y - fitted
    cycle = np.full(T, np.nan)
    trend = np.full(T, np.nan)
    cycle[n_lost:] = resid
    trend[n_lost:] = fitted
    return cycle, trend


def transform_log100_hamilton(x: pd.Series, h: int = 8, p: int = 4) -> pd.Series:
    """Hamilton (2018) filter cycle of log(x), scaled by 100."""
    logx = np.log(x)
    valid = logx.dropna()
    if len(valid) < h + p + 1:
        return pd.Series(np.nan, index=x.index)
    cycle, _ = _hamilton_filter(valid.values, h=h, p=p)
    result = pd.Series(np.nan, index=x.index)
    result.loc[valid.index] = 100.0 * cycle
    return result


def transform_logHamilton(x: pd.Series, h: int = 8, p: int = 4) -> pd.Series:
    """Hamilton (2018) filter cycle of log(x) (unscaled)."""
    logx = np.log(x)
    valid = logx.dropna()
    if len(valid) < h + p + 1:
        return pd.Series(np.nan, index=x.index)
    cycle, _ = _hamilton_filter(valid.values, h=h, p=p)
    result = pd.Series(np.nan, index=x.index)
    result.loc[valid.index] = cycle
    return result


def transform_log100_lt(x: pd.Series) -> pd.Series:
    """100 * (log(x) - fitted linear trend)."""
    logx = np.log(x)
    valid = logx.dropna()
    if len(valid) < 10:
        return pd.Series(np.nan, index=x.index)
    t = np.arange(len(valid), dtype=float)
    t -= t.mean()
    coeffs = np.polyfit(t, valid.values, 1)
    trend = np.polyval(coeffs, t)
    result = pd.Series(np.nan, index=x.index)
    result.loc[valid.index] = 100.0 * (valid.values - trend)
    return result


def transform_log_lt(x: pd.Series) -> pd.Series:
    """Linear detrended cycle of log(x) (unscaled)."""
    logx = np.log(x)
    valid = logx.dropna()
    if len(valid) < 10:
        return pd.Series(np.nan, index=x.index)
    t = np.arange(len(valid), dtype=float)
    t -= t.mean()
    coeffs = np.polyfit(t, valid.values, 1)
    trend = np.polyval(coeffs, t)
    result = pd.Series(np.nan, index=x.index)
    result.loc[valid.index] = valid.values - trend
    return result


_TRANSFORM_FNS = {
    "log": transform_log,
    "DA": transform_DA,
    "D4A": transform_D4A,
    "log100_hp": transform_log100_hp,
    "logHP": transform_logHP,
    "log100_hamilton": transform_log100_hamilton,
    "logHamilton": transform_logHamilton,
    "log100_lt": transform_log100_lt,
    "log_lt": transform_log_lt,
}


# ======================================================================== #
#  Rebasing
# ======================================================================== #

def _rebase_cpi_indices(df: pd.DataFrame,
                        base_q: str = "2018Q1") -> pd.DataFrame:
    """Rebase CPI-related index columns to base_q = 100.

    Skips columns that look like rates (majority of values in [-5, 5]).
    """
    cpi_cols = [c for c in df.columns
                if c.startswith(("Core_IPCA_", "CPI_"))
                and not c.startswith(("W_",))
                and "Expectation" not in c]

    base_period = pd.Period(base_q, freq="Q")

    for col in cpi_cols:
        x = df[col].dropna()
        if len(x) == 0:
            continue
        # Heuristic: skip if it's a growth rate, not an index
        share_small = (x.abs() <= 5).mean()
        if share_small > 0.6:
            continue
        if base_period in df.index:
            base_val = df.loc[base_period, col]
            if pd.notna(base_val) and base_val != 0:
                df[col] = df[col] / base_val * 100.0
    return df


# ======================================================================== #
#  Main entry point
# ======================================================================== #

def apply_transforms(
    consolidated: pd.DataFrame | None = None,
    input_path: str | None = None,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Apply all transformations defined in the variable registry.

    Parameters
    ----------
    consolidated : DataFrame or None
        If provided, use this DataFrame directly.
    input_path : str or None
        Path to consolidated_quarterly_data.csv (used if consolidated is None).
    output_dir : str or None
        Where to save transformed_data.csv.

    Returns
    -------
    pd.DataFrame
        Transformed data (original + transformed columns).
    """
    if output_dir is None:
        output_dir = _CSV_DIR

    # Load data
    if consolidated is None:
        if input_path is None:
            input_path = os.path.join(_CSV_DIR,
                                      "consolidated_quarterly_data.csv")
        print(f"  Loading: {input_path}")
        df = pd.read_csv(input_path, index_col="date", parse_dates=True)
        df.index = df.index.to_period("Q")
    else:
        df = consolidated.copy()

    print(f"  Input: {df.shape[0]} obs x {df.shape[1]} variables")

    # Rebase CPI indices to 2018Q1 = 100
    df = _rebase_cpi_indices(df)

    # Build transform map from registry
    n_created = 0
    for vdef in REGISTRY:
        if not vdef.transforms:
            continue
        if vdef.name not in df.columns:
            continue

        x = df[vdef.name]
        for trans_name in vdef.transforms:
            fn = _TRANSFORM_FNS.get(trans_name)
            if fn is None:
                print(f"    Unknown transform: {trans_name}")
                continue
            new_col = f"{trans_name}_{vdef.name}"
            try:
                df[new_col] = fn(x)
                n_created += 1
            except Exception as exc:
                print(f"    {new_col}: {exc}")

    print(f"  Created {n_created} transformed variables")
    print(f"  Output: {df.shape[0]} obs x {df.shape[1]} total columns")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "transformed_data.csv")
    df_save = df.copy()
    df_save.index = df_save.index.to_timestamp()
    df_save.index.name = "date"
    df_save.to_csv(out_path)
    print(f"  Saved: {out_path}")

    return df

"""BVAR vs QPM forecast evaluation: rolling-window RMSE comparison.

Estimates a large Bayesian VAR (14 endogenous variables matching QPM
observables, including the US neutral real rate) with Minnesota + Sims priors and empirical Bayes
hyperparameter selection.  Runs rolling-window unconditional forecast
evaluation and compares RMSE against the QPM.

Usage:
    python run_forecast_evaluation.py                  # BVAR only (QPM from saved)
    python run_forecast_evaluation.py --run-qpm        # also re-run QPM evaluation
    python run_forecast_evaluation.py --conditional     # (future) conditional BVAR
    python run_forecast_evaluation.py --standard-priors # compare tight vs standard priors
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_GPM_EST = _THIS_DIR.parents[1] / "gpm_estimation"
if str(_GPM_EST) not in sys.path:
    sys.path.insert(0, str(_GPM_EST))

from qpm_toolbox.var.bayesian import estimate_bvar
from qpm_toolbox.var.forecast import forecast_bvar, _wz_solve, _simulate_path
from qpm_toolbox.var.types import BVARPriorOptions

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_CSV = _THIS_DIR / "data" / "estimation_data.csv"
RESULTS_DIR = _THIS_DIR / "results" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# QPM Model A draws used for the BVAR-vs-QPM forecast comparison (Tables 4-5).
# v5: switched from results_v2/ (calibrated init) to results_v2_diffuse/
# (exact diffuse Kalman, D&K 2012) to match the paper baseline.
QPM_RESULTS_DIR = _THIS_DIR / "results_v2_diffuse" / "model_a"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIRST_ORIGIN = "2017Q1"
LAST_ORIGIN = "2022Q4"
HORIZONS = [1, 2, 3, 4, 8]
MAX_HORIZON = 8
P_LAGS = 4
N_DRAWS = 20000
SEED = 42

# Variables to evaluate (subset reported in the paper)
EVAL_VARS = ["DA_LGDP", "DA_LCPI", "DA_LCPIX", "RS"]

# Derived level variables: cumulated from growth-rate forecasts.
# LGDP_t+h = LGDP_t + (1/4) * sum_{s=1}^{h} DA_LGDP_{t+s}
# This allows like-for-like comparison with the QPM (which forecasts in levels).
EVAL_LEVEL_VARS = {"LGDP": "DA_LGDP"}  # level_var: growth_var

# Full BVAR endogenous variables (14)
# Ordered: domestic core, external, commodity gaps, neutral rates
ENDO_VARS = [
    "DA_LGDP",          # GDP growth (QPM observes log-level)
    "DA_LCPI",          # Headline inflation
    "DA_LCPIX",         # Core inflation
    "DA_LCPI_ENERGY",   # Energy inflation
    "DA_LCPI_US",       # US inflation
    "DA_LS",            # Exchange rate depreciation
    "LRWOIL_GAP",       # Oil price gap
    "LRWFOOD_GAP",      # Food price gap
    "GDP_GAP_US",       # US output gap
    "PREM",             # Risk premium (CDS)
    "RS",               # Policy rate (Selic)
    "RS_US",            # US policy rate
    "DA_TT",            # Terms of trade growth (QPM observes level)
    "RR_BAR_US",        # US neutral real rate (LWH estimate)
]

# delta vector: 1 for persistent level variables, 0 for stationary
DELTA = np.array([
    0,  # DA_LGDP
    0,  # DA_LCPI
    0,  # DA_LCPIX
    0,  # DA_LCPI_ENERGY
    0,  # DA_LCPI_US
    0,  # DA_LS
    0,  # LRWOIL_GAP
    0,  # LRWFOOD_GAP
    0,  # GDP_GAP_US
    1,  # PREM  (persistent level)
    1,  # RS    (persistent level)
    1,  # RS_US (persistent level)
    0,  # DA_TT
    1,  # RR_BAR_US (persistent level — slow-moving neutral rate)
], dtype=float)

# ---------------------------------------------------------------------------
# Prior configurations for sensitivity analysis
# ---------------------------------------------------------------------------
# "Tight" (baseline): strong Sims priors shrink toward unit-root in levels.
# This is the configuration used throughout the paper.
PRIOR_TIGHT = BVARPriorOptions(
    lambda1=0.2,         # overall tightness (Banbura et al. default)
    lambda2=1e6,         # → no cross-variable dummy rows (R bvar style)
    lambda3=2.0,         # lag decay
    lambda4=1e7,         # intercept prior variance (R bvar default)
    delta=DELTA,
    add_sims=True,
    lambda_soc=5.0,      # strong sum-of-coefficients
    lambda_dio=2.0,      # moderate dummy-initial-obs
    bvar_compat=True,
)

# "Standard": textbook Minnesota + mild Sims priors (Banbura et al. 2010).
# Used for prior sensitivity robustness check (Section V of the paper).
PRIOR_STANDARD = BVARPriorOptions(
    lambda1=0.2,
    lambda2=1e6,
    lambda3=2.0,
    lambda4=1e7,
    delta=DELTA,
    add_sims=True,
    lambda_soc=1.0,      # standard sum-of-coefficients
    lambda_dio=1.0,      # standard dummy-initial-obs
    bvar_compat=True,
)

# QPM observable → BVAR variable mapping (for RMSE comparison)
# GDP comparison uses the cumulated level forecast (LGDP), not the growth rate.
QPM_TO_BVAR = {
    "OBS_DA_LCPI": "DA_LCPI",
    "OBS_DA_LCPIX": "DA_LCPIX",
    "OBS_RS": "RS",
    "OBS_LGDP": "LGDP",  # QPM log-level ↔ BVAR cumulated growth → level
}


# ======================================================================== #
#  1. Data loading
# ======================================================================== #

def load_data(start: str = "2004Q1", end: str = "2025Q2") -> pd.DataFrame:
    """Load estimation_data.csv and prepare BVAR-ready DataFrame.

    Constructs DA_TT (terms-of-trade growth) from TT level.
    Returns DataFrame with PeriodIndex.
    """
    raw = pd.read_csv(DATA_CSV)
    raw["date"] = pd.PeriodIndex(raw["date"], freq="Q")
    raw = raw.set_index("date").sort_index()
    raw = raw.loc[start:end]

    # Construct terms-of-trade growth rate (annualized QoQ log-difference)
    raw["DA_TT"] = 400.0 * np.log(raw["TT"] / raw["TT"].shift(1))

    print(f"Sample: {raw.index[0]} – {raw.index[-1]}  ({len(raw)} obs)")
    return raw


def make_covid_dummies(df: pd.DataFrame) -> np.ndarray:
    """Build (T, 2) impulse dummy matrix for 2020Q1 and 2020Q2."""
    T = len(df)
    dummies = np.zeros((T, 2))
    for i, per in enumerate(df.index):
        if per == pd.Period("2020Q1", freq="Q"):
            dummies[i, 0] = 1.0
        elif per == pd.Period("2020Q2", freq="Q"):
            dummies[i, 1] = 1.0
    return dummies


def prepare_bvar_data(
    df: pd.DataFrame,
) -> tuple:
    """Extract endogenous array, exogenous array, and metadata.

    Returns:
        endo: (T, K) endogenous data array
        exog: (T, n_exog) exogenous regressors [COVID dummies + inflation target]
        var_names: list of endogenous variable names
        dates: PeriodIndex
        level_data: dict mapping level variable names to (T,) arrays.
                     Used for cumulated forecast comparisons (e.g. LGDP).
    """
    # Drop first obs (lost to TT differencing)
    df_clean = df.dropna(subset=ENDO_VARS + ["DA_LCPI_TAR"]).copy()

    endo = df_clean[ENDO_VARS].values.astype(np.float64)
    var_names = list(ENDO_VARS)

    # Exogenous: COVID dummies only
    # Note: the inflation target is NOT included as exogenous because
    # the BVAR would learn a spurious level relationship (RS ≈ 2×target)
    # that dominates the forecast, rather than using lag dynamics.
    exog = make_covid_dummies(df_clean)

    dates = df_clean.index

    # Level variables for cumulated-forecast comparisons
    level_data = {}
    for lv in EVAL_LEVEL_VARS:
        if lv in df_clean.columns:
            level_data[lv] = df_clean[lv].values.astype(np.float64)

    print(f"  BVAR data: {endo.shape[0]} obs × {endo.shape[1]} endo vars "
          f"+ {exog.shape[1]} exog")
    n_miss = np.isnan(endo).sum(axis=0)
    if n_miss.any():
        for j, nm in enumerate(var_names):
            if n_miss[j] > 0:
                print(f"    WARNING: {nm} has {n_miss[j]} NaN values")

    return endo, exog, var_names, dates, level_data


# ======================================================================== #
#  2. BVAR rolling-window forecast evaluation
# ======================================================================== #

def run_bvar_rolling(
    endo: np.ndarray,
    exog: np.ndarray,
    var_names: list,
    dates: pd.PeriodIndex,
    level_data: dict = None,
    eval_vars: list = None,
    first_origin: str = FIRST_ORIGIN,
    last_origin: str = LAST_ORIGIN,
    horizons: list = None,
    max_horizon: int = MAX_HORIZON,
    p: int = P_LAGS,
    n_draws: int = N_DRAWS,
    seed: int = SEED,
    conditional: bool = False,
    prior_opts: BVARPriorOptions = None,
) -> dict:
    """Rolling-window BVAR forecast evaluation.

    At each origin, re-estimates the BVAR via empirical Bayes and
    generates unconditional (or conditional) forecasts.

    Returns dict with keys: rmse, mae, bias, errors, forecast_paths,
    origins, horizons, variables.
    """
    if eval_vars is None:
        eval_vars = EVAL_VARS
    if horizons is None:
        horizons = HORIZONS

    K = endo.shape[1]
    T_data = endo.shape[0]

    # Resolve origin indices
    first_per = pd.Period(first_origin, freq="Q")
    last_per = pd.Period(last_origin, freq="Q")

    first_idx = dates.get_loc(first_per)
    last_idx = dates.get_loc(last_per)
    last_idx = min(last_idx, T_data - max(horizons) - 1)

    origins = list(range(first_idx, last_idx + 1))
    origin_dates = [dates[i] for i in origins]

    print(f"\n  BVAR rolling forecast: {len(origins)} origins, "
          f"{dates[first_idx]} to {dates[last_idx]}")
    print(f"  Horizons: {horizons}")
    print(f"  Eval vars: {eval_vars}")

    # Variable indices
    var_indices = {v: var_names.index(v) for v in eval_vars}

    # Use provided prior options, or fall back to tight (baseline) priors.
    if prior_opts is None:
        prior_opts = PRIOR_TIGHT

    # COVID-affected forecasts: drop any (origin, horizon) pair whose
    # origin OR target quarter falls in 2020Q1--2020Q4.  This excludes
    # both the contraction (2020Q1--Q2) and the subsequent rebound
    # quarters (2020Q3--Q4), which whipsaw any reduced-form forecast.
    covid_start = pd.Period("2020Q1", freq="Q")
    covid_end = pd.Period("2020Q4", freq="Q")

    # Storage — separate errors for full and ex-COVID samples
    errors = {v: {h: [] for h in horizons} for v in eval_vars}
    errors_excovid = {v: {h: [] for h in horizons} for v in eval_vars}
    forecast_paths = []

    # Level-variable errors (cumulated from growth forecasts)
    if level_data is None:
        level_data = {}
    level_vars = [lv for lv in EVAL_LEVEL_VARS if lv in level_data]
    errors_level = {lv: {h: [] for h in horizons} for lv in level_vars}
    errors_level_excovid = {lv: {h: [] for h in horizons} for lv in level_vars}

    rng = np.random.default_rng(seed)

    for origin_num, t in enumerate(origins):
        # Truncate data to [0..t]
        endo_trunc = endo[:t + 1]
        exog_trunc = exog[:t + 1]

        # Estimate BVAR with fixed hyperparameters
        draws, _, diag = estimate_bvar(
            data=endo_trunc,
            p=p,
            options=prior_opts,
            n_draws=n_draws,
            constant=True,
            filter_unstable=True,
            var_names=var_names,
            seed=int(rng.integers(0, 2**31)),
            exog=exog_trunc,
        )

        # Build last_lags: (p, K) most recent observations
        # Row 0 = most recent (Y_T), row p-1 = oldest (Y_{T-p+1})
        last_lags = endo_trunc[-p:][::-1]

        # Future exogenous: use actuals if available, else repeat last
        h_fwd = max(horizons)
        if t + h_fwd < T_data:
            exog_future = exog[t + 1: t + 1 + h_fwd]
        else:
            n_avail = T_data - t - 1
            exog_future = np.vstack([
                exog[t + 1: T_data],
                np.tile(exog[-1], (h_fwd - n_avail, 1)),
            ])

        # Unconditional forecast
        fcst_result = forecast_bvar(
            draws=draws,
            last_lags=last_lags,
            horizon=h_fwd,
            add_shocks=False,  # deterministic (median path)
            seed=int(rng.integers(0, 2**31)),
            exog_future=exog_future,
        )

        # Median forecast: (H, K)
        y_fcst = fcst_result.median

        # Store forecast path
        forecast_paths.append((origin_dates[origin_num], y_fcst.copy()))

        # Compute cumulated level forecasts for level variables.
        # DA_LGDP = 4 * diff(LGDP), so LGDP_{t+h} = LGDP_t + cumsum(DA_LGDP)/4.
        level_fcst = {}
        for lv, gv in EVAL_LEVEL_VARS.items():
            if lv in level_vars:
                gi = var_names.index(gv)
                level_origin = level_data[lv][t]
                level_fcst[lv] = level_origin + np.cumsum(y_fcst[:, gi]) / 4.0

        # Compute errors
        for h in horizons:
            actual_idx = t + h
            if actual_idx >= T_data:
                continue

            # Check if this forecast-actual pair overlaps COVID
            # (origin OR target in COVID window).
            actual_date = dates[actual_idx]
            origin_date = dates[t]
            is_covid = ((covid_start <= actual_date <= covid_end) or
                        (covid_start <= origin_date <= covid_end))

            for v in eval_vars:
                vi = var_indices[v]
                actual = float(endo[actual_idx, vi])
                predicted = float(y_fcst[h - 1, vi])
                err = actual - predicted
                errors[v][h].append(err)
                if not is_covid:
                    errors_excovid[v][h].append(err)

            # Level-variable errors
            for lv in level_vars:
                actual_level = float(level_data[lv][actual_idx])
                predicted_level = float(level_fcst[lv][h - 1])
                err_lv = actual_level - predicted_level
                errors_level[lv][h].append(err_lv)
                if not is_covid:
                    errors_level_excovid[lv][h].append(err_lv)

        if (origin_num + 1) % 5 == 0 or origin_num == 0:
            stable_pct = 100 * diag["n_draws_kept"] / diag["n_draws_requested"]
            print(f"    [{origin_num + 1}/{len(origins)}] "
                  f"Origin {dates[t]}  "
                  f"stable={diag['n_draws_kept']}/{diag['n_draws_requested']} "
                  f"({stable_pct:.0f}%)")

    # Aggregate metrics for both full and ex-COVID samples
    def _aggregate(error_dict, var_list):
        r, m, b = {}, {}, {}
        for v in var_list:
            r[v], m[v], b[v] = {}, {}, {}
            for h in horizons:
                err = np.array(error_dict[v][h])
                if len(err) > 0:
                    r[v][h] = float(np.sqrt(np.mean(err**2)))
                    m[v][h] = float(np.mean(np.abs(err)))
                    b[v][h] = float(np.mean(err))
                else:
                    r[v][h] = np.nan
                    m[v][h] = np.nan
                    b[v][h] = np.nan
        return r, m, b

    rmse, mae, bias = _aggregate(errors, eval_vars)
    rmse_xc, mae_xc, bias_xc = _aggregate(errors_excovid, eval_vars)

    # Aggregate level-variable metrics
    rmse_lv, mae_lv, bias_lv = _aggregate(errors_level, level_vars)
    rmse_lv_xc, mae_lv_xc, bias_lv_xc = _aggregate(
        errors_level_excovid, level_vars)

    # Merge level results into the main dicts
    all_vars = eval_vars + level_vars
    for lv in level_vars:
        rmse[lv] = rmse_lv[lv]
        mae[lv] = mae_lv[lv]
        bias[lv] = bias_lv[lv]
        rmse_xc[lv] = rmse_lv_xc[lv]
        mae_xc[lv] = mae_lv_xc[lv]
        bias_xc[lv] = bias_lv_xc[lv]

    # Print RMSE tables
    for label, r in [("BVAR RMSE (full sample)", rmse),
                     ("BVAR RMSE (excl. COVID)", rmse_xc)]:
        print(f"\n  {label}:")
        header = f"  {'Variable':<15}" + "".join(f"  h={h:<4}" for h in horizons)
        print(header)
        print("  " + "-" * len(header))
        for v in all_vars:
            row = f"  {v:<15}" + "".join(
                f"  {r[v][h]:6.2f}" for h in horizons)
            print(row)

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "rmse_excovid": rmse_xc,
        "mae_excovid": mae_xc,
        "bias_excovid": bias_xc,
        "errors": errors,
        "errors_excovid": errors_excovid,
        "forecast_paths": forecast_paths,
        "origins": origin_dates,
        "horizons": horizons,
        "variables": all_vars,
        "var_names": var_names,
    }


# ======================================================================== #
#  2b. BVAR conditional rolling-window forecast evaluation
# ======================================================================== #

# Indices of variables to condition on (foreign + volatile domestic)
COND_INDICES = [
    3,   # DA_LCPI_ENERGY  (volatile domestic)
    4,   # DA_LCPI_US      (foreign)
    6,   # LRWOIL_GAP      (foreign)
    7,   # LRWFOOD_GAP     (foreign)
    8,   # GDP_GAP_US      (foreign)
    11,  # RS_US           (foreign)
    13,  # RR_BAR_US       (foreign — US neutral real rate)
]


def run_bvar_rolling_conditional(
    endo: np.ndarray,
    exog: np.ndarray,
    var_names: list,
    dates: pd.PeriodIndex,
    level_data: dict = None,
    eval_vars: list = None,
    cond_indices: list = None,
    first_origin: str = FIRST_ORIGIN,
    last_origin: str = LAST_ORIGIN,
    horizons: list = None,
    max_horizon: int = MAX_HORIZON,
    p: int = P_LAGS,
    seed: int = SEED,
    prior_opts: BVARPriorOptions = None,
) -> dict:
    """Rolling-window BVAR conditional forecast evaluation.

    Uses the analytical posterior mean (no MCMC) and Waggoner-Zha (1999)
    conditional forecasting.  At each origin, conditions on the actual
    realised values of foreign variables and volatile domestic components.

    Returns dict with: rmse, mae, bias (full + ex-COVID), forecast_paths.
    """
    if eval_vars is None:
        eval_vars = EVAL_VARS
    if horizons is None:
        horizons = HORIZONS
    if cond_indices is None:
        cond_indices = COND_INDICES

    K = endo.shape[1]
    T_data = endo.shape[0]

    # Use provided prior options, or fall back to tight (baseline) priors.
    if prior_opts is None:
        prior_opts = PRIOR_TIGHT

    # Resolve origin indices
    first_per = pd.Period(first_origin, freq="Q")
    last_per = pd.Period(last_origin, freq="Q")
    first_idx = dates.get_loc(first_per)
    last_idx = dates.get_loc(last_per)
    last_idx = min(last_idx, T_data - max(horizons) - 1)

    origins = list(range(first_idx, last_idx + 1))
    origin_dates = [dates[i] for i in origins]

    cond_names = [var_names[j] for j in cond_indices]
    print(f"\n  Conditional BVAR rolling forecast: {len(origins)} origins, "
          f"{dates[first_idx]} to {dates[last_idx]}")
    print(f"  Horizons: {horizons}")
    print(f"  Conditioned on: {cond_names}")
    print(f"  Eval vars: {eval_vars}")

    # Variable indices
    var_indices = {v: var_names.index(v) for v in eval_vars}

    # COVID periods (drop any forecast whose origin OR target falls in
    # 2020Q1--2020Q4; both the contraction and the rebound whipsaw any
    # reduced-form forecast).
    covid_start = pd.Period("2020Q1", freq="Q")
    covid_end = pd.Period("2020Q4", freq="Q")

    # Storage
    errors = {v: {h: [] for h in horizons} for v in eval_vars}
    errors_excovid = {v: {h: [] for h in horizons} for v in eval_vars}
    forecast_paths = []

    # Level-variable errors
    if level_data is None:
        level_data = {}
    level_vars = [lv for lv in EVAL_LEVEL_VARS if lv in level_data]
    errors_level = {lv: {h: [] for h in horizons} for lv in level_vars}
    errors_level_excovid = {lv: {h: [] for h in horizons} for lv in level_vars}

    rng = np.random.default_rng(seed)
    Kp = K * p
    n_exog = exog.shape[1]

    for origin_num, t in enumerate(origins):
        endo_trunc = endo[:t + 1]
        exog_trunc = exog[:t + 1]

        # Estimate BVAR — only need posterior params, not draws
        _, post_params, diag = estimate_bvar(
            data=endo_trunc,
            p=p,
            options=prior_opts,
            n_draws=1,
            constant=True,
            filter_unstable=False,
            var_names=var_names,
            seed=int(rng.integers(0, 2**31)),
            exog=exog_trunc,
        )

        # Extract posterior mean as point estimate
        B_hat = post_params.B_post
        Sigma_hat = post_params.S_post / (post_params.df_post - K - 1)
        c_hat = B_hat[0, :]
        A_hat = B_hat[1:1 + Kp, :]
        D_hat = B_hat[1 + Kp:, :] if n_exog > 0 else None

        # Build last_lags
        last_lags = endo_trunc[-p:][::-1]

        # Future exogenous
        h_fwd = max(horizons)
        if t + h_fwd < T_data:
            exog_future = exog[t + 1: t + 1 + h_fwd]
        else:
            n_avail = T_data - t - 1
            exog_future = np.vstack([
                exog[t + 1: T_data],
                np.tile(exog[-1], (h_fwd - n_avail, 1)),
            ])

        # Build constraints: condition on actuals for cond_indices
        constraints = np.full((h_fwd, K), np.nan)
        effective_h = min(h_fwd, T_data - t - 1)
        for h in range(effective_h):
            for j in cond_indices:
                constraints[h, j] = endo[t + h + 1, j]

        # Build R, r from constraints (same logic as forecast_conditional_bvar)
        rows_R = []
        vals_r = []
        for h in range(h_fwd):
            for j in range(K):
                if np.isfinite(constraints[h, j]):
                    row = np.zeros(K * h_fwd)
                    row[h * K + j] = 1.0
                    rows_R.append(row)
                    vals_r.append(constraints[h, j])

        if len(rows_R) > 0:
            R = np.array(rows_R)
            r = np.array(vals_r)
            Yc, _ = _wz_solve(
                A_hat, c_hat, Sigma_hat, last_lags,
                h_fwd, K, p, R, r,
                D=D_hat, exog_future=exog_future,
            )
            y_fcst = Yc
        else:
            # No constraints — unconditional
            y_fcst = _simulate_path(
                A_hat, c_hat, Sigma_hat, last_lags,
                h_fwd, K, p, add_shocks=False,
                D=D_hat, exog_future=exog_future,
            )

        # Store forecast path
        forecast_paths.append((origin_dates[origin_num], y_fcst.copy()))

        # Cumulated level forecasts
        level_fcst = {}
        for lv, gv in EVAL_LEVEL_VARS.items():
            if lv in level_vars:
                gi = var_names.index(gv)
                level_origin = level_data[lv][t]
                level_fcst[lv] = level_origin + np.cumsum(y_fcst[:, gi]) / 4.0

        # Compute errors
        for h in horizons:
            actual_idx = t + h
            if actual_idx >= T_data:
                continue

            actual_date = dates[actual_idx]
            origin_date = dates[t]
            is_covid = ((covid_start <= actual_date <= covid_end) or
                        (covid_start <= origin_date <= covid_end))

            for v in eval_vars:
                vi = var_indices[v]
                actual = float(endo[actual_idx, vi])
                predicted = float(y_fcst[h - 1, vi])
                err = actual - predicted
                errors[v][h].append(err)
                if not is_covid:
                    errors_excovid[v][h].append(err)

            for lv in level_vars:
                actual_level = float(level_data[lv][actual_idx])
                predicted_level = float(level_fcst[lv][h - 1])
                err_lv = actual_level - predicted_level
                errors_level[lv][h].append(err_lv)
                if not is_covid:
                    errors_level_excovid[lv][h].append(err_lv)

        if (origin_num + 1) % 5 == 0 or origin_num == 0:
            print(f"    [{origin_num + 1}/{len(origins)}] Origin {dates[t]}")

    # Aggregate metrics
    def _aggregate(error_dict, var_list):
        r, m, b = {}, {}, {}
        for v in var_list:
            r[v], m[v], b[v] = {}, {}, {}
            for h in horizons:
                err = np.array(error_dict[v][h])
                if len(err) > 0:
                    r[v][h] = float(np.sqrt(np.mean(err**2)))
                    m[v][h] = float(np.mean(np.abs(err)))
                    b[v][h] = float(np.mean(err))
                else:
                    r[v][h] = np.nan
                    m[v][h] = np.nan
                    b[v][h] = np.nan
        return r, m, b

    rmse, mae, bias = _aggregate(errors, eval_vars)
    rmse_xc, mae_xc, bias_xc = _aggregate(errors_excovid, eval_vars)

    rmse_lv, mae_lv, bias_lv = _aggregate(errors_level, level_vars)
    rmse_lv_xc, mae_lv_xc, bias_lv_xc = _aggregate(
        errors_level_excovid, level_vars)

    all_vars = eval_vars + level_vars
    for lv in level_vars:
        rmse[lv] = rmse_lv[lv]
        mae[lv] = mae_lv[lv]
        bias[lv] = bias_lv[lv]
        rmse_xc[lv] = rmse_lv_xc[lv]
        mae_xc[lv] = mae_lv_xc[lv]
        bias_xc[lv] = bias_lv_xc[lv]

    # Print RMSE tables
    for label, r in [("Conditional BVAR RMSE (full sample)", rmse),
                     ("Conditional BVAR RMSE (excl. COVID)", rmse_xc)]:
        print(f"\n  {label}:")
        header = f"  {'Variable':<15}" + "".join(f"  h={h:<4}" for h in horizons)
        print(header)
        print("  " + "-" * len(header))
        for v in all_vars:
            row = f"  {v:<15}" + "".join(
                f"  {r[v][h]:6.2f}" for h in horizons)
            print(row)

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "rmse_excovid": rmse_xc,
        "mae_excovid": mae_xc,
        "bias_excovid": bias_xc,
        "errors": errors,
        "errors_excovid": errors_excovid,
        "forecast_paths": forecast_paths,
        "origins": origin_dates,
        "horizons": horizons,
        "variables": all_vars,
        "var_names": var_names,
    }


# ======================================================================== #
#  3. QPM forecast evaluation (re-run with current draws)
# ======================================================================== #

def run_qpm_rolling(
    eval_vars_qpm: list = None,
    first_origin: str = "2017-01-01",
    last_origin: str = "2022-10-01",
    horizons: list = None,
) -> dict:
    """Re-run QPM rolling forecast evaluation using saved posterior draws.

    Returns dict with rmse, mae, bias, forecast_paths in QPM observable units.
    """
    if eval_vars_qpm is None:
        eval_vars_qpm = ["OBS_DA_LCPI", "OBS_LGDP", "OBS_RS", "OBS_DA_LCPIX"]
    if horizons is None:
        horizons = HORIZONS

    # Lazy imports (JAX is heavy)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(var, '1')

    from qpm_toolbox import QPMModel

    MOD_FILE = str(_THIS_DIR / "model" / "brazil_qpm_mr_v2.qpm")
    DATA_CSV_QPM = str(_THIS_DIR / "data" / "estimation_data.csv")
    DRAWS_PATH = str(QPM_RESULTS_DIR
                     / "estimation" / "draws.npz")

    # Import run_brazil_wp_v2 configs
    sys.path.insert(0, str(_THIS_DIR))
    from run_brazil_wp_v2 import (
        CYCLE_MAP, TREND_SNR_MAP, MANUAL_CYCLE_PRIORS,
        FIXED_PARAMS, MEAS_NOISE_STD,
        load_brazil_data, compute_initial_conditions,
    )

    print("\n  Loading QPM model and posterior draws...")
    m = QPMModel(MOD_FILE, results_dir=str(
        QPM_RESULTS_DIR))

    # Load data using the same loader as run_brazil_wp.py
    ys, dates_ts, obs_names_qpm = load_brazil_data()
    m.set_data(ys, dates=dates_ts, obs_names=obs_names_qpm)

    m.set_priors(
        structural_priors=_get_structural_priors(),
        fixed_params=FIXED_PARAMS,
        meas_noise_std=MEAS_NOISE_STD,
        cycle_map=CYCLE_MAP,
        trend_snr_map=TREND_SNR_MAP,
        manual_cycle_priors=MANUAL_CYCLE_PRIORS,
    )

    # Initial conditions — same as run_brazil_wp.py
    ss_params = m.model.default_param_assignments
    init_config = compute_initial_conditions(ys, obs_names_qpm, ss_params)
    m.set_initial_conditions(init_state_config=init_config)
    m.load_draws(DRAWS_PATH)

    # Build a valid point estimate.  The element-wise posterior median
    # can violate Blanchard-Kahn conditions even when every individual
    # draw satisfies them.  Strategy:
    #   1. Try posterior median.
    #   2. If BK fails, try posterior mean.
    #   3. If that also fails, scan individual draws.
    pdict = m._resolve_params()
    sol = m.model.solve(pdict)
    if sol["solution_valid"]:
        print("  Using posterior median (BK-valid).")
    else:
        print("  Posterior median fails BK; trying posterior mean...")
        pdict_mean = dict(m.model.default_param_assignments)
        if m._fixed_params:
            pdict_mean.update(m._fixed_params)
        for name, samples in m._posterior_samples.items():
            pdict_mean[name] = float(np.mean(np.asarray(samples)))
        sol = m.model.solve(pdict_mean)
        if sol["solution_valid"]:
            pdict = pdict_mean
            print("  Posterior mean is BK-valid.")
        else:
            print("  Posterior mean also fails BK; scanning draws...")
            n_draws_avail = len(list(m._posterior_samples.values())[0])
            found = False
            for i in range(min(200, n_draws_avail)):
                pdict_i = dict(m.model.default_param_assignments)
                if m._fixed_params:
                    pdict_i.update(m._fixed_params)
                for name, samples in m._posterior_samples.items():
                    pdict_i[name] = float(np.asarray(samples)[i])
                sol_i = m.model.solve(pdict_i)
                if sol_i["solution_valid"]:
                    pdict = pdict_i
                    found = True
                    print(f"  Using draw {i} (BK-valid).")
                    break
            if not found:
                raise RuntimeError(
                    "No valid parameterisation found in first 200 draws.")

    print("  Running QPM forecast evaluation...")
    ev = m.forecast_evaluation(
        variables=eval_vars_qpm,
        first_origin=first_origin,
        last_origin=last_origin,
        horizons=horizons,
        max_horizon=MAX_HORIZON,
        stochastic=False,
        plot=False,
        params=pdict,
    )

    # Return evaluation result plus model metadata needed for
    # growth-rate derivation (first-differencing OBS_LGDP forecasts).
    _, ys_np, dates_qpm, obs_names = m._data
    return {
        "ev": ev,
        "obs_names": list(obs_names),
        "ys_np": ys_np,
        "dates_qpm": dates_qpm,
    }


def run_qpm_rolling_calibrated(
    eval_vars_qpm: list = None,
    first_origin: str = "2017-01-01",
    last_origin: str = "2022-10-01",
    horizons: list = None,
) -> dict:
    """Run QPM rolling forecast evaluation using prior mode (calibrated) params.

    Same as run_qpm_rolling but uses prior modes instead of posterior means,
    enabling a direct comparison of calibrated vs estimated QPM forecasts.
    """
    if eval_vars_qpm is None:
        eval_vars_qpm = ["OBS_DA_LCPI", "OBS_LGDP", "OBS_RS", "OBS_DA_LCPIX"]
    if horizons is None:
        horizons = HORIZONS

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(var, '1')

    from qpm_toolbox import QPMModel
    from qpm_toolbox.calibration import compute_prior_modes

    MOD_FILE = str(_THIS_DIR / "model" / "brazil_qpm_mr_v2.qpm")

    sys.path.insert(0, str(_THIS_DIR))
    from run_brazil_wp_v2 import (
        CYCLE_MAP, TREND_SNR_MAP, MANUAL_CYCLE_PRIORS,
        FIXED_PARAMS, MEAS_NOISE_STD,
        load_brazil_data, compute_initial_conditions,
        FIXED_POINT_CONFIG,
    )

    print("\n  Loading QPM model for calibrated forecast evaluation...")
    m = QPMModel(MOD_FILE, results_dir=str(
        QPM_RESULTS_DIR))

    ys, dates_ts, obs_names_qpm = load_brazil_data()
    m.set_data(ys, dates=dates_ts, obs_names=obs_names_qpm)

    m.set_priors(
        structural_priors=_get_structural_priors(),
        fixed_params=FIXED_PARAMS,
        meas_noise_std=MEAS_NOISE_STD,
        cycle_map=CYCLE_MAP,
        trend_snr_map=TREND_SNR_MAP,
        manual_cycle_priors=MANUAL_CYCLE_PRIORS,
    )

    ss_params = m.model.default_param_assignments
    init_config = compute_initial_conditions(ys, obs_names_qpm, ss_params)
    m.set_initial_conditions(init_state_config=init_config)

    # Calibrate shock variances (same pipeline as estimation)
    m.calibrate(method='both', fixed_point_config=FIXED_POINT_CONFIG,
                        cycle_persistence=0.85, drift_persistence=0.90)

    # Build parameter dict from prior modes + calibrated sigmas
    prior_modes = compute_prior_modes(m._priors)
    pdict = dict(m.model.default_param_assignments)
    pdict.update(FIXED_PARAMS)
    pdict.update(prior_modes)
    if m._calibrated_sigmas:
        pdict.update(m._calibrated_sigmas)

    sol = m.model.solve(pdict)
    if not sol["solution_valid"]:
        raise RuntimeError("Calibrated (prior mode) parameters fail BK.")
    print(f"  Calibrated params BK-valid (spectral radius: "
          f"{float(sol['spectral_radius']):.4f})")

    print("  Running calibrated QPM forecast evaluation (unconditional)...")
    ev = m.forecast_evaluation(
        variables=eval_vars_qpm,
        first_origin=first_origin,
        last_origin=last_origin,
        horizons=horizons,
        max_horizon=MAX_HORIZON,
        stochastic=False,
        plot=False,
        params=pdict,
    )

    # Conditional forecast evaluation (same conditioning as estimated QPM)
    CONDITION_ON = [
        ('obs', 'OBS_LRWOIL_GAP',     'SHK_LRWOIL_GAP'),
        ('obs', 'OBS_LRWFOOD_GAP',    'SHK_LRWFOOD_GAP'),
        ('obs', 'OBS_RS_US',          'SHK_RS_US'),
        ('obs', 'OBS_GDP_GAP_US',     'SHK_GAP_US'),
        ('obs', 'OBS_DA_LCPI_ENERGY', 'SHK_DA_LCPI_ENERGY'),
        ('obs', 'OBS_DA_LCPI_US',     'SHK_DA_LCPI_US'),
        ('obs', 'OBS_RR_BAR_US',      'SHK_RR_BAR_US'),
    ]
    print("  Running calibrated QPM forecast evaluation (conditional)...")
    ev_cond = m.conditional_forecast_evaluation(
        variables=eval_vars_qpm,
        condition_on=CONDITION_ON,
        first_origin=first_origin,
        last_origin=last_origin,
        horizons=horizons,
        max_horizon=MAX_HORIZON,
        plot=False,
        params=pdict,
    )

    _, ys_np, dates_qpm, obs_names = m._data
    return {
        "ev": ev,
        "ev_cond": ev_cond,
        "obs_names": list(obs_names),
        "ys_np": ys_np,
        "dates_qpm": dates_qpm,
    }


def _get_structural_priors():
    """Import structural prior specs from run_brazil_wp_v2."""
    from run_brazil_wp_v2 import STRUCTURAL_PRIORS
    return STRUCTURAL_PRIORS


def derive_qpm_growth_rmse(
    qpm_result: dict,
    df_bvar: pd.DataFrame,
    horizons: list = None,
) -> dict:
    """Derive growth-rate RMSE from QPM log-level GDP forecasts.

    The QPM forecasts OBS_LGDP in log-levels.  This function first-
    differences the level forecasts to obtain implied annualized quarterly
    growth rates:

        DA_LGDP_{t+h} = 4 * (OBS_LGDP_{t+h} - OBS_LGDP_{t+h-1})

    where OBS_LGDP_{t+0} is the actual log GDP at the forecast origin.
    The implied growth-rate forecast is compared against the actual
    DA_LGDP values.

    Args:
        qpm_result:  dict returned by run_qpm_rolling() with keys
                     'ev', 'obs_names', 'ys_np', 'dates_qpm'.
        df_bvar:     DataFrame with PeriodIndex containing 'DA_LGDP'.
        horizons:    list of forecast horizons.

    Returns:
        dict with 'rmse', 'rmse_excovid', 'mae', 'mae_excovid',
        'bias', 'bias_excovid' — each mapping horizon → value.
    """
    if horizons is None:
        horizons = HORIZONS

    ev = qpm_result["ev"]
    obs_names = qpm_result["obs_names"]
    ys_np = qpm_result["ys_np"]
    dates_qpm = qpm_result["dates_qpm"]

    lgdp_idx = obs_names.index("OBS_LGDP")

    # Convert QPM dates to PeriodIndex for matching with BVAR data
    if hasattr(dates_qpm, 'to_period'):
        dates_period = dates_qpm.to_period("Q")
    else:
        dates_period = pd.PeriodIndex(
            [pd.Period(d, freq="Q") for d in dates_qpm])

    # COVID filtering
    covid_start = pd.Period("2020Q1", freq="Q")
    covid_end = pd.Period("2020Q2", freq="Q")

    errors = {h: [] for h in horizons}
    errors_excovid = {h: [] for h in horizons}

    for origin_date, y_fcst in ev.forecast_paths:
        origin_period = pd.Period(origin_date, freq="Q")

        # Find origin in QPM data to get actual OBS_LGDP at origin
        t_qpm = dates_period.get_loc(origin_period)
        lgdp_origin = float(ys_np[t_qpm, lgdp_idx])

        # OBS_LGDP forecast path: (H_fwd,)
        lgdp_fcst = y_fcst[:, lgdp_idx]

        # First-difference to get implied annualized growth rate
        lgdp_with_origin = np.concatenate([[lgdp_origin], lgdp_fcst])
        da_lgdp_implied = 4.0 * np.diff(lgdp_with_origin)

        # Find origin in BVAR data
        if origin_period not in df_bvar.index:
            continue
        t_bvar = df_bvar.index.get_loc(origin_period)

        for h in horizons:
            actual_idx = t_bvar + h
            if actual_idx >= len(df_bvar):
                continue

            actual_date = df_bvar.index[actual_idx]
            actual_val = float(df_bvar["DA_LGDP"].iloc[actual_idx])
            predicted_val = float(da_lgdp_implied[h - 1])
            err = actual_val - predicted_val

            errors[h].append(err)
            is_covid = (covid_start <= actual_date <= covid_end)
            if not is_covid:
                errors_excovid[h].append(err)

    # Aggregate
    rmse, rmse_xc = {}, {}
    mae, mae_xc = {}, {}
    bias, bias_xc = {}, {}
    for h in horizons:
        e = np.array(errors[h])
        exc = np.array(errors_excovid[h])
        rmse[h] = float(np.sqrt(np.mean(e**2))) if len(e) > 0 else np.nan
        mae[h] = float(np.mean(np.abs(e))) if len(e) > 0 else np.nan
        bias[h] = float(np.mean(e)) if len(e) > 0 else np.nan
        rmse_xc[h] = float(np.sqrt(np.mean(exc**2))) if len(exc) > 0 else np.nan
        mae_xc[h] = float(np.mean(np.abs(exc))) if len(exc) > 0 else np.nan
        bias_xc[h] = float(np.mean(exc)) if len(exc) > 0 else np.nan

    print("\n  QPM growth-rate RMSE (DA_LGDP, derived from OBS_LGDP):")
    print(f"    {'h':>3}  {'Full':>8}  {'Ex-COVID':>8}")
    print("    " + "-" * 25)
    for h in horizons:
        print(f"    {h:3d}  {rmse[h]:8.2f}  {rmse_xc[h]:8.2f}")

    return {
        "rmse": rmse, "rmse_excovid": rmse_xc,
        "mae": mae, "mae_excovid": mae_xc,
        "bias": bias, "bias_excovid": bias_xc,
        "errors": errors, "errors_excovid": errors_excovid,
    }


def derive_qpm_excovid_rmse(
    qpm_result: dict,
    horizons: list = None,
) -> dict:
    """Compute ex-COVID RMSE for QPM observable-level forecasts.

    Iterates over forecast_paths from the evaluation result, identifies
    COVID-affected forecast targets (2020Q1–Q2), and computes separate
    full-sample and ex-COVID RMSE/MAE/bias for each observable variable.

    This ensures an apples-to-apples comparison with the BVAR, which
    also excludes COVID-affected actuals.

    Args:
        qpm_result:  dict returned by run_qpm_rolling().
        horizons:    list of forecast horizons.

    Returns:
        dict with 'rmse', 'rmse_excovid', 'mae', 'mae_excovid',
        'bias', 'bias_excovid' — each mapping variable → horizon → value.
        Also includes 'variables' and 'horizons'.
    """
    if horizons is None:
        horizons = HORIZONS

    ev = qpm_result["ev"]
    obs_names = qpm_result["obs_names"]
    ys_np = qpm_result["ys_np"]
    dates_qpm = qpm_result["dates_qpm"]

    # Variable list from evaluation
    eval_vars = list(ev.variables)
    var_indices = {v: obs_names.index(v) for v in eval_vars}

    # Convert dates to PeriodIndex
    if hasattr(dates_qpm, 'to_period'):
        dates_period = dates_qpm.to_period("Q")
    else:
        dates_period = pd.PeriodIndex(
            [pd.Period(d, freq="Q") for d in dates_qpm])

    covid_start = pd.Period("2020Q1", freq="Q")
    covid_end = pd.Period("2020Q2", freq="Q")

    errors_full = {v: {h: [] for h in horizons} for v in eval_vars}
    errors_excovid = {v: {h: [] for h in horizons} for v in eval_vars}

    for origin_date, y_fcst in ev.forecast_paths:
        origin_period = pd.Period(origin_date, freq="Q")
        t_origin = dates_period.get_loc(origin_period)

        for h in horizons:
            actual_idx = t_origin + h
            if actual_idx >= len(dates_period):
                continue

            actual_period = dates_period[actual_idx]
            is_covid = (covid_start <= actual_period <= covid_end)

            for v in eval_vars:
                vi = var_indices[v]
                actual_val = float(ys_np[actual_idx, vi])
                predicted_val = float(y_fcst[h - 1, vi])
                err = actual_val - predicted_val

                errors_full[v][h].append(err)
                if not is_covid:
                    errors_excovid[v][h].append(err)

    # Aggregate
    rmse, rmse_xc = {}, {}
    mae, mae_xc = {}, {}
    bias, bias_xc = {}, {}
    for v in eval_vars:
        rmse[v], rmse_xc[v] = {}, {}
        mae[v], mae_xc[v] = {}, {}
        bias[v], bias_xc[v] = {}, {}
        for h in horizons:
            e = np.array(errors_full[v][h])
            exc = np.array(errors_excovid[v][h])
            rmse[v][h] = float(np.sqrt(np.mean(e**2))) if len(e) > 0 else np.nan
            mae[v][h] = float(np.mean(np.abs(e))) if len(e) > 0 else np.nan
            bias[v][h] = float(np.mean(e)) if len(e) > 0 else np.nan
            rmse_xc[v][h] = float(np.sqrt(np.mean(exc**2))) if len(exc) > 0 else np.nan
            mae_xc[v][h] = float(np.mean(np.abs(exc))) if len(exc) > 0 else np.nan
            bias_xc[v][h] = float(np.mean(exc)) if len(exc) > 0 else np.nan

    print("\n  QPM RMSE — full sample vs ex-COVID:")
    print(f"    {'Variable':<15}  {'h':>3}  {'Full':>8}  {'Ex-COVID':>8}")
    print("    " + "-" * 40)
    for v in eval_vars:
        for h in horizons:
            print(f"    {v:<15}  {h:3d}  {rmse[v][h]:8.3f}  {rmse_xc[v][h]:8.3f}")

    return {
        "rmse": rmse, "rmse_excovid": rmse_xc,
        "mae": mae, "mae_excovid": mae_xc,
        "bias": bias, "bias_excovid": bias_xc,
        "variables": eval_vars,
        "horizons": horizons,
    }


# ======================================================================== #
#  4. Comparison and plotting
# ======================================================================== #

def save_rmse_csv(result: dict, filename: str, label: str = ""):
    """Save RMSE/MAE/bias to CSV (both full and ex-COVID)."""
    rows = []
    for v in result["variables"]:
        for h in result["horizons"]:
            row = {
                "model": label,
                "variable": v,
                "horizon": h,
                "rmse": result["rmse"][v][h],
                "mae": result["mae"][v][h],
                "bias": result["bias"][v][h],
            }
            # Add ex-COVID if available
            if "rmse_excovid" in result:
                row["rmse_excovid"] = result["rmse_excovid"][v][h]
                row["mae_excovid"] = result["mae_excovid"][v][h]
                row["bias_excovid"] = result["bias_excovid"][v][h]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / filename, index=False)
    print(f"  Saved: {RESULTS_DIR / filename}")
    return df


def compare_rmse(bvar_result: dict, qpm_rmse_csv: str = None) -> pd.DataFrame:
    """Build side-by-side RMSE comparison table.

    Uses the ex-COVID RMSE for both BVAR and QPM when available,
    ensuring an apples-to-apples comparison.  Also saves full-sample
    RMSE for reference.
    """
    # Load QPM RMSE (from saved CSV or current evaluation)
    if qpm_rmse_csv is not None:
        qpm_df = pd.read_csv(qpm_rmse_csv)
    else:
        qpm_df = pd.read_csv(RESULTS_DIR / "qpm_forecast_rmse.csv")

    # Map QPM variable names to BVAR names
    qpm_df["bvar_var"] = qpm_df["variable"].map(QPM_TO_BVAR)

    # Check if QPM CSV has ex-COVID column
    qpm_has_excovid = "rmse_excovid" in qpm_df.columns

    # Build comparison using ex-COVID RMSE
    rmse_key = "rmse_excovid" if "rmse_excovid" in bvar_result else "rmse"
    rows = []
    for v in bvar_result["variables"]:
        for h in bvar_result["horizons"]:
            bvar_rmse = bvar_result[rmse_key][v].get(h, np.nan)
            bvar_rmse_full = bvar_result["rmse"][v].get(h, np.nan)

            # Find matching QPM RMSE
            qpm_match = qpm_df[
                (qpm_df["bvar_var"] == v) & (qpm_df["horizon"] == h)
            ]
            # Use ex-COVID QPM RMSE for fair comparison when available
            if len(qpm_match) > 0:
                if qpm_has_excovid and not pd.isna(qpm_match["rmse_excovid"].iloc[0]):
                    qpm_rmse = float(qpm_match["rmse_excovid"].iloc[0])
                else:
                    qpm_rmse = float(qpm_match["rmse"].iloc[0])
                qpm_rmse_full = float(qpm_match["rmse"].iloc[0])
            else:
                qpm_rmse = np.nan
                qpm_rmse_full = np.nan

            ratio = bvar_rmse / qpm_rmse if qpm_rmse > 0 else np.nan

            rows.append({
                "variable": v,
                "horizon": h,
                "bvar_rmse": bvar_rmse,
                "bvar_rmse_full": bvar_rmse_full,
                "qpm_rmse": qpm_rmse,
                "qpm_rmse_full": qpm_rmse_full,
                "ratio_bvar_qpm": ratio,
            })

    comp = pd.DataFrame(rows)
    comp.to_csv(RESULTS_DIR / "rmse_comparison.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'rmse_comparison.csv'}")

    # Print comparison table
    label = "excl. COVID" if qpm_has_excovid else "QPM full-sample"
    print(f"\n  RMSE Comparison — BVAR vs QPM ({label}):")
    print(f"  {'Variable':<15}  {'h':>3}  {'BVAR':>7}  {'QPM':>7}  {'Ratio':>7}")
    print("  " + "-" * 45)
    for _, r in comp.iterrows():
        print(f"  {r['variable']:<15}  {r['horizon']:3.0f}  "
              f"{r['bvar_rmse']:7.2f}  {r['qpm_rmse']:7.2f}  "
              f"{r['ratio_bvar_qpm']:7.2f}")

    return comp


def plot_spaghetti(
    result: dict,
    actuals: np.ndarray,
    dates: pd.PeriodIndex,
    var_names: list,
    eval_vars: list,
    label: str = "BVAR",
    filename: str = "forecast_spaghetti_bvar.png",
):
    """Plot rolling forecast spaghetti for evaluated variables.

    Clips the plot to the evaluation window (2015 onward) and uses
    sensible y-axis limits to avoid COVID outliers dominating the figure.
    """
    n_vars = len(eval_vars)
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3.5 * n_vars),
                             sharex=False)
    if n_vars == 1:
        axes = [axes]

    # Pretty names
    pretty = {
        "DA_LGDP": "GDP Growth (QoQ ann.)",
        "DA_LCPI": "Headline Inflation (QoQ ann.)",
        "DA_LCPIX": "Core Inflation (QoQ ann.)",
        "RS": "Policy Rate (Selic)",
        "DA_LS": "Exchange Rate Depreciation",
    }

    # Y-axis limits: clip to avoid COVID extremes dominating the plot
    ylim_map = {
        "DA_LGDP": (-20, 15),
        "DA_LCPI": (-5, 20),
        "DA_LCPIX": (-5, 15),
        "RS": (0, 25),
    }

    # Plot window: start from 2015 for readability
    plot_start = pd.Period("2015Q1", freq="Q")
    plot_start_ts = plot_start.to_timestamp()

    for ax, v in zip(axes, eval_vars):
        vi = var_names.index(v)

        # Plot actuals
        ts = dates.to_timestamp()
        ax.plot(ts, actuals[:, vi], "k-", lw=1.5, label="Actual", zorder=3)

        # COVID shading
        ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-07-01"),
                   alpha=0.08, color="red", zorder=0)

        # Plot forecast paths
        for origin_date, y_fcst in result["forecast_paths"]:
            H = y_fcst.shape[0]
            fcst_dates = pd.period_range(
                start=origin_date + 1, periods=H, freq="Q")
            ax.plot(fcst_dates.to_timestamp(), y_fcst[:, vi],
                    color="steelblue", alpha=0.3, lw=0.8)

        ax.set_xlim(left=plot_start_ts)
        if v in ylim_map:
            ax.set_ylim(ylim_map[v])
        ax.set_title(pretty.get(v, v), fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label}: Rolling Forecast Paths", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {RESULTS_DIR / filename}")


def plot_spaghetti_qpm(
    qpm_result: dict,
    df_bvar: pd.DataFrame,
    eval_vars_bvar: list = None,
    filename: str = "forecast_spaghetti_qpm.png",
):
    """Plot rolling forecast spaghetti for QPM evaluation.

    Uses QPM forecast paths (in QPM observable space) and plots them
    alongside actuals from the BVAR data.  The BVAR-to-QPM variable
    mapping is used to find the correct columns.

    Args:
        qpm_result:  dict returned by run_qpm_rolling() with 'ev',
                     'obs_names', 'ys_np', 'dates_qpm'.
        df_bvar:     DataFrame with PeriodIndex containing actuals.
        eval_vars_bvar:  BVAR variable names to plot (default: EVAL_VARS).
        filename:    output PNG filename.
    """
    if eval_vars_bvar is None:
        eval_vars_bvar = EVAL_VARS

    ev = qpm_result["ev"]
    obs_names = qpm_result["obs_names"]
    ys_np = qpm_result["ys_np"]
    dates_qpm = qpm_result["dates_qpm"]

    # BVAR → QPM variable mapping (reverse of QPM_TO_BVAR)
    bvar_to_qpm = {v: k for k, v in QPM_TO_BVAR.items()}
    # Add direct mapping for growth-rate variables
    bvar_to_qpm["DA_LGDP"] = "OBS_LGDP"  # will need first-differencing
    bvar_to_qpm["DA_LCPI"] = "OBS_DA_LCPI"
    bvar_to_qpm["DA_LCPIX"] = "OBS_DA_LCPIX"

    n_vars = len(eval_vars_bvar)
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3.5 * n_vars),
                             sharex=False)
    if n_vars == 1:
        axes = [axes]

    pretty = {
        "DA_LGDP": "GDP Growth (QoQ ann.)",
        "DA_LCPI": "Headline Inflation (QoQ ann.)",
        "DA_LCPIX": "Core Inflation (QoQ ann.)",
        "RS": "Policy Rate (Selic)",
    }
    ylim_map = {
        "DA_LGDP": (-20, 15),
        "DA_LCPI": (-5, 20),
        "DA_LCPIX": (-5, 15),
        "RS": (0, 25),
    }

    plot_start_ts = pd.Timestamp("2015-01-01")

    for ax, v in zip(axes, eval_vars_bvar):
        # Plot actual from BVAR data
        ts = df_bvar.index.to_timestamp()
        ax.plot(ts, df_bvar[v].values, "k-", lw=1.5, label="Actual", zorder=3)

        # COVID shading
        ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-07-01"),
                   alpha=0.08, color="red", zorder=0)

        qpm_var = bvar_to_qpm.get(v)
        if qpm_var is None:
            continue
        qpm_idx = obs_names.index(qpm_var)

        # For DA_LGDP we need to first-difference OBS_LGDP forecasts
        is_growth = (v == "DA_LGDP")

        # Convert QPM dates to PeriodIndex for matching
        if hasattr(dates_qpm, 'to_period'):
            dates_period = dates_qpm.to_period("Q")
        else:
            dates_period = pd.PeriodIndex(
                [pd.Period(d, freq="Q") for d in dates_qpm])

        for origin_date, y_fcst in ev.forecast_paths:
            H = y_fcst.shape[0]
            origin_period = pd.Period(origin_date, freq="Q")
            fcst_periods = pd.period_range(
                start=origin_period + 1, periods=H, freq="Q")

            if is_growth:
                # First-difference level forecasts to get growth rates
                t_qpm = dates_period.get_loc(origin_period)
                lgdp_origin = float(ys_np[t_qpm, qpm_idx])
                lgdp_fcst = y_fcst[:, qpm_idx]
                lgdp_with_origin = np.concatenate([[lgdp_origin], lgdp_fcst])
                fcst_vals = 4.0 * np.diff(lgdp_with_origin)
            else:
                fcst_vals = y_fcst[:, qpm_idx]

            ax.plot(fcst_periods.to_timestamp(), fcst_vals,
                    color="darkorange", alpha=0.3, lw=0.8)

        ax.set_xlim(left=plot_start_ts)
        if v in ylim_map:
            ax.set_ylim(ylim_map[v])
        ax.set_title(pretty.get(v, v), fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("QPM: Rolling Forecast Paths", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {RESULTS_DIR / filename}")


def plot_rmse_comparison(comp: pd.DataFrame,
                         filename: str = "rmse_comparison_bars.png"):
    """Grouped bar chart comparing QPM vs BVAR RMSE."""
    # Only plot variables that have QPM comparisons
    comp_valid = comp.dropna(subset=["qpm_rmse"])
    variables = comp_valid["variable"].unique()
    horizons = sorted(comp_valid["horizon"].unique())
    n_h = len(horizons)

    pretty = {
        "LGDP": "GDP (log-level)",
        "DA_LGDP": "GDP Growth",
        "DA_LCPI": "Headline CPI",
        "DA_LCPIX": "Core CPI",
        "RS": "Policy Rate",
    }

    fig, axes = plt.subplots(1, len(variables),
                             figsize=(4.5 * len(variables), 4),
                             sharey=False)
    if len(variables) == 1:
        axes = [axes]

    x = np.arange(n_h)
    width = 0.35

    for ax, v in zip(axes, variables):
        sub = comp_valid[comp_valid["variable"] == v].sort_values("horizon")
        qpm_vals = sub["qpm_rmse"].values
        bvar_vals = sub["bvar_rmse"].values

        bars_qpm = ax.bar(x - width / 2, qpm_vals, width,
                          label="QPM", color="steelblue", alpha=0.8)
        bars_bvar = ax.bar(x + width / 2, bvar_vals, width,
                           label="BVAR", color="coral", alpha=0.8)

        ax.set_xlabel("Forecast Horizon (quarters)")
        ax.set_ylabel("RMSE")
        ax.set_title(pretty.get(v, v))
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("RMSE Comparison: QPM vs. Unconditional BVAR",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {RESULTS_DIR / filename}")


def _print_three_way(bvar_result, cond_result, qpm_rmse_path,
                     qpm_growth=None):
    """Print three-way RMSE comparison and save CSV.

    Args:
        bvar_result:  unconditional BVAR result dict (or None).
        cond_result:  conditional BVAR result dict.
        qpm_rmse_path: Path to QPM RMSE CSV.
        qpm_growth:   dict from derive_qpm_growth_rmse() with QPM DA_LGDP
                      RMSE (or None if QPM was not re-run).
    """
    # Load QPM RMSE
    qpm_df = None
    if qpm_rmse_path.exists():
        qpm_df = pd.read_csv(qpm_rmse_path)
        qpm_df["bvar_var"] = qpm_df["variable"].map(QPM_TO_BVAR)

    eval_vars = EVAL_VARS
    horizons = HORIZONS

    # Load saved unconditional BVAR results if not freshly computed
    bvar_saved = None
    if bvar_result is None:
        bvar_csv = RESULTS_DIR / "bvar_forecast_rmse.csv"
        if bvar_csv.exists():
            bvar_saved = pd.read_csv(bvar_csv)
            print(f"  Loaded saved unconditional BVAR from {bvar_csv}")

    rows = []
    for v in eval_vars:
        for h in horizons:
            # Unconditional BVAR RMSE
            if bvar_result is not None:
                bvar_rmse = bvar_result["rmse_excovid"][v][h]
                bvar_full = bvar_result["rmse"][v][h]
            elif bvar_saved is not None:
                match = bvar_saved[
                    (bvar_saved["variable"] == v) & (bvar_saved["horizon"] == h)]
                if len(match) > 0:
                    bvar_rmse = float(match["rmse_excovid"].iloc[0])
                    bvar_full = float(match["rmse"].iloc[0])
                else:
                    bvar_rmse, bvar_full = np.nan, np.nan
            else:
                bvar_rmse, bvar_full = np.nan, np.nan

            cond_rmse = cond_result["rmse_excovid"][v][h]
            cond_full = cond_result["rmse"][v][h]

            # QPM RMSE: for DA_LGDP use the growth-rate derivation;
            # for other variables use ex-COVID from CSV when available.
            qpm_rmse = np.nan
            qpm_full = np.nan
            qpm_has_excovid = (qpm_df is not None
                               and "rmse_excovid" in qpm_df.columns)
            if v == "DA_LGDP" and qpm_growth is not None:
                qpm_rmse = qpm_growth["rmse_excovid"].get(h, np.nan)
                qpm_full = qpm_growth["rmse"].get(h, np.nan)
            elif qpm_df is not None:
                qpm_match = qpm_df[
                    (qpm_df["bvar_var"] == v) & (qpm_df["horizon"] == h)]
                if len(qpm_match) > 0:
                    qpm_full = float(qpm_match["rmse"].iloc[0])
                    if qpm_has_excovid and not pd.isna(
                            qpm_match["rmse_excovid"].iloc[0]):
                        qpm_rmse = float(qpm_match["rmse_excovid"].iloc[0])
                    else:
                        qpm_rmse = qpm_full

            rows.append({
                "variable": v,
                "horizon": h,
                "bvar_uncond": bvar_rmse,
                "bvar_uncond_full": bvar_full,
                "bvar_cond": cond_rmse,
                "bvar_cond_full": cond_full,
                "qpm": qpm_rmse,
                "qpm_full": qpm_full,
            })

    comp3 = pd.DataFrame(rows)
    comp3.to_csv(RESULTS_DIR / "rmse_three_way.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'rmse_three_way.csv'}")

    # Print table (ex-COVID)
    print("\n  Three-Way RMSE Comparison (excl. COVID):")
    print(f"  {'Variable':<12} {'h':>3}  {'Uncond':>8}  {'Cond':>8}  "
          f"{'QPM':>8}  {'Cond/QPM':>8}")
    print("  " + "-" * 56)
    for _, r in comp3.iterrows():
        ratio = r['bvar_cond'] / r['qpm'] if r['qpm'] > 0 else np.nan
        print(f"  {r['variable']:<12} {r['horizon']:3.0f}  "
              f"{r['bvar_uncond']:8.2f}  {r['bvar_cond']:8.2f}  "
              f"{r['qpm']:8.2f}  {ratio:8.2f}")

    # Print table (full sample including COVID)
    print("\n  Three-Way RMSE Comparison (full sample, incl. COVID):")
    print(f"  {'Variable':<12} {'h':>3}  {'Uncond':>8}  {'Cond':>8}  "
          f"{'QPM':>8}  {'Cond/QPM':>8}")
    print("  " + "-" * 56)
    for _, r in comp3.iterrows():
        ratio = r['bvar_cond_full'] / r['qpm_full'] if r['qpm_full'] > 0 else np.nan
        print(f"  {r['variable']:<12} {r['horizon']:3.0f}  "
              f"{r['bvar_uncond_full']:8.2f}  {r['bvar_cond_full']:8.2f}  "
              f"{r['qpm_full']:8.2f}  {ratio:8.2f}")


# ======================================================================== #
#  5. Main
# ======================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="BVAR vs QPM forecast evaluation")
    parser.add_argument("--run-qpm", action="store_true",
                        help="Re-run QPM forecast evaluation from draws")
    parser.add_argument("--conditional", action="store_true",
                        help="Run conditional BVAR forecasts")
    parser.add_argument("--conditional-only", action="store_true",
                        help="Only run conditional BVAR (skip unconditional)")
    parser.add_argument("--standard-priors", action="store_true",
                        help="Run prior sensitivity: compare tight vs standard Sims priors")
    parser.add_argument("--run-qpm-calibrated", action="store_true",
                        help="Run QPM forecast with prior modes (calibrated params)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    print("=" * 70)
    print("  BVAR vs QPM Forecast Evaluation")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load and prepare data
    # ---------------------------------------------------------------
    print("\n--- 1. Loading data ---")
    df = load_data()
    endo, exog, var_names, dates, level_data = prepare_bvar_data(df)

    # ---------------------------------------------------------------
    # 2. BVAR unconditional rolling forecast evaluation
    # ---------------------------------------------------------------
    bvar_result = None
    if not args.conditional_only:
        print("\n--- 2. BVAR rolling-window forecast evaluation ---")
        bvar_result = run_bvar_rolling(
            endo=endo, exog=exog, var_names=var_names, dates=dates,
            level_data=level_data,
        )
        save_rmse_csv(bvar_result, "bvar_forecast_rmse.csv", "BVAR")

    # ---------------------------------------------------------------
    # 2a. Prior sensitivity: standard Minnesota vs tight Sims priors
    # ---------------------------------------------------------------
    bvar_standard = None
    if args.standard_priors and not args.conditional_only:
        print("\n--- 2a. Prior sensitivity: standard Sims priors ---")
        print("  (lambda_soc=1.0, lambda_dio=1.0 vs baseline 5.0/2.0)")
        bvar_standard = run_bvar_rolling(
            endo=endo, exog=exog, var_names=var_names, dates=dates,
            level_data=level_data,
            prior_opts=PRIOR_STANDARD,
        )
        save_rmse_csv(bvar_standard, "bvar_standard_prior_rmse.csv",
                      "BVAR_Standard")

        # Print side-by-side comparison
        if bvar_result is not None:
            print("\n  Prior Sensitivity: Tight vs Standard (RMSE excl. COVID)")
            print(f"  {'Variable':<15}  {'h':>3}  {'Tight':>8}  {'Standard':>8}  {'Ratio':>8}")
            print("  " + "-" * 50)
            for v in bvar_result["variables"]:
                for h in bvar_result["horizons"]:
                    tight_val = bvar_result["rmse_excovid"][v][h]
                    std_val = bvar_standard["rmse_excovid"][v][h]
                    ratio = std_val / tight_val if tight_val > 0 else np.nan
                    print(f"  {v:<15}  {h:3d}  {tight_val:8.2f}  "
                          f"{std_val:8.2f}  {ratio:8.2f}")

            # Save comparison CSV
            sens_rows = []
            for v in bvar_result["variables"]:
                for h in bvar_result["horizons"]:
                    sens_rows.append({
                        "variable": v,
                        "horizon": h,
                        "rmse_tight": bvar_result["rmse_excovid"][v][h],
                        "rmse_standard": bvar_standard["rmse_excovid"][v][h],
                        "rmse_tight_full": bvar_result["rmse"][v][h],
                        "rmse_standard_full": bvar_standard["rmse"][v][h],
                    })
            sens_df = pd.DataFrame(sens_rows)
            sens_df.to_csv(RESULTS_DIR / "bvar_prior_sensitivity.csv",
                          index=False)
            print(f"  Saved: {RESULTS_DIR / 'bvar_prior_sensitivity.csv'}")

    # ---------------------------------------------------------------
    # 2b. BVAR conditional rolling forecast evaluation
    # ---------------------------------------------------------------
    cond_result = None
    if args.conditional or args.conditional_only:
        print("\n--- 2b. Conditional BVAR rolling-window forecast ---")
        cond_result = run_bvar_rolling_conditional(
            endo=endo, exog=exog, var_names=var_names, dates=dates,
            level_data=level_data,
        )
        save_rmse_csv(cond_result, "cond_bvar_forecast_rmse.csv",
                      "BVAR_Conditional")

    # ---------------------------------------------------------------
    # 3. QPM forecast evaluation (optional re-run)
    # ---------------------------------------------------------------
    qpm_rmse_path = RESULTS_DIR / "qpm_forecast_rmse.csv"

    qpm_growth = None
    qpm_excovid = None
    qpm_dict = None
    if args.run_qpm:
        print("\n--- 3. QPM rolling-window forecast evaluation ---")
        qpm_dict = run_qpm_rolling()
        qpm_ev = qpm_dict["ev"]

        # Compute ex-COVID RMSE for observable variables
        print("\n--- 3a. QPM ex-COVID RMSE (observable variables) ---")
        qpm_excovid = derive_qpm_excovid_rmse(qpm_dict)

        # Save QPM RMSE with both full and ex-COVID columns
        qpm_rows = []
        for v in qpm_ev.variables:
            for h in qpm_ev.horizons:
                row = {
                    "model": "QPM",
                    "variable": v,
                    "horizon": h,
                    "rmse": qpm_ev.rmse[v][h],
                    "mae": qpm_ev.mae[v][h],
                    "bias": qpm_ev.bias[v][h],
                }
                if qpm_excovid is not None and v in qpm_excovid["rmse_excovid"]:
                    row["rmse_excovid"] = qpm_excovid["rmse_excovid"][v][h]
                    row["mae_excovid"] = qpm_excovid["mae_excovid"][v][h]
                    row["bias_excovid"] = qpm_excovid["bias_excovid"][v][h]
                qpm_rows.append(row)
        pd.DataFrame(qpm_rows).to_csv(qpm_rmse_path, index=False)
        print(f"  Saved: {qpm_rmse_path}")

        # Derive QPM growth-rate RMSE from OBS_LGDP level forecasts
        print("\n--- 3b. QPM growth-rate RMSE (DA_LGDP from OBS_LGDP) ---")
        qpm_growth = derive_qpm_growth_rmse(qpm_dict, df)

        # Save QPM growth-rate RMSE
        gr_rows = []
        for h in HORIZONS:
            gr_rows.append({
                "model": "QPM_growth",
                "variable": "DA_LGDP",
                "horizon": h,
                "rmse": qpm_growth["rmse"][h],
                "mae": qpm_growth["mae"][h],
                "bias": qpm_growth["bias"][h],
                "rmse_excovid": qpm_growth["rmse_excovid"][h],
                "mae_excovid": qpm_growth["mae_excovid"][h],
                "bias_excovid": qpm_growth["bias_excovid"][h],
            })
        gr_df = pd.DataFrame(gr_rows)
        gr_df.to_csv(RESULTS_DIR / "qpm_growth_rate_rmse.csv", index=False)
        print(f"  Saved: {RESULTS_DIR / 'qpm_growth_rate_rmse.csv'}")
    elif not qpm_rmse_path.exists():
        old_path = RESULTS_DIR / "forecast_rmse.csv"
        if old_path.exists():
            print(f"\n  Using existing QPM RMSE from {old_path}")
            old_df = pd.read_csv(old_path)
            old_df["model"] = "QPM"
            old_df.to_csv(qpm_rmse_path, index=False)

    # Try to load saved QPM growth-rate RMSE if not freshly computed
    qpm_growth_path = RESULTS_DIR / "qpm_growth_rate_rmse.csv"
    if qpm_growth is None and qpm_growth_path.exists():
        gr_df = pd.read_csv(qpm_growth_path)
        qpm_growth = {
            "rmse_excovid": {
                int(row["horizon"]): float(row["rmse_excovid"])
                for _, row in gr_df.iterrows()
            },
            "rmse": {
                int(row["horizon"]): float(row["rmse"])
                for _, row in gr_df.iterrows()
            },
        }
        print(f"\n  Loaded QPM growth-rate RMSE from {qpm_growth_path}")

    # ---------------------------------------------------------------
    # 3c. Calibrated QPM forecast evaluation (prior modes)
    # ---------------------------------------------------------------
    qpm_cal_path = RESULTS_DIR / "qpm_calibrated_forecast_rmse.csv"
    qpm_cal_growth = None
    if args.run_qpm_calibrated:
        print("\n--- 3c. Calibrated QPM rolling-window forecast evaluation ---")
        qpm_cal_dict = run_qpm_rolling_calibrated()
        qpm_cal_ev = qpm_cal_dict["ev"]

        qpm_cal_excovid = derive_qpm_excovid_rmse(qpm_cal_dict)

        cal_rows = []
        for v in qpm_cal_ev.variables:
            for h in qpm_cal_ev.horizons:
                row = {
                    "model": "QPM_Calibrated",
                    "variable": v,
                    "horizon": h,
                    "rmse": qpm_cal_ev.rmse[v][h],
                    "mae": qpm_cal_ev.mae[v][h],
                    "bias": qpm_cal_ev.bias[v][h],
                }
                if qpm_cal_excovid is not None and v in qpm_cal_excovid["rmse_excovid"]:
                    row["rmse_excovid"] = qpm_cal_excovid["rmse_excovid"][v][h]
                    row["mae_excovid"] = qpm_cal_excovid["mae_excovid"][v][h]
                    row["bias_excovid"] = qpm_cal_excovid["bias_excovid"][v][h]
                cal_rows.append(row)
        pd.DataFrame(cal_rows).to_csv(qpm_cal_path, index=False)
        print(f"  Saved: {qpm_cal_path}")

        print("\n--- 3d. Calibrated QPM growth-rate RMSE ---")
        qpm_cal_growth = derive_qpm_growth_rmse(qpm_cal_dict, df)
        cal_gr_rows = []
        for h in HORIZONS:
            cal_gr_rows.append({
                "model": "QPM_Calibrated_growth",
                "variable": "DA_LGDP",
                "horizon": h,
                "rmse": qpm_cal_growth["rmse"][h],
                "mae": qpm_cal_growth["mae"][h],
                "bias": qpm_cal_growth["bias"][h],
                "rmse_excovid": qpm_cal_growth["rmse_excovid"][h],
                "mae_excovid": qpm_cal_growth["mae_excovid"][h],
                "bias_excovid": qpm_cal_growth["bias_excovid"][h],
            })
        cal_gr_df = pd.DataFrame(cal_gr_rows)
        cal_gr_df.to_csv(RESULTS_DIR / "qpm_calibrated_growth_rate_rmse.csv",
                         index=False)
        print(f"  Saved: {RESULTS_DIR / 'qpm_calibrated_growth_rate_rmse.csv'}")

        # Save conditional calibrated QPM RMSE
        qpm_cal_cond_path = RESULTS_DIR / "qpm_calibrated_conditional_forecast_rmse.csv"
        qpm_cal_cond_ev = qpm_cal_dict["ev_cond"]
        cal_cond_rows = []
        for v in qpm_cal_cond_ev.variables:
            for h in qpm_cal_cond_ev.horizons:
                cal_cond_rows.append({
                    "model": "QPM_Calibrated_Conditional",
                    "variable": v,
                    "horizon": h,
                    "rmse": qpm_cal_cond_ev.rmse[v][h],
                    "mae": qpm_cal_cond_ev.mae[v][h],
                    "bias": qpm_cal_cond_ev.bias[v][h],
                })
        pd.DataFrame(cal_cond_rows).to_csv(qpm_cal_cond_path, index=False)
        print(f"  Saved: {qpm_cal_cond_path}")

    # ---------------------------------------------------------------
    # 4. Compare RMSE
    # ---------------------------------------------------------------
    if bvar_result is not None:
        print("\n--- 4. RMSE comparison ---")
        comp = compare_rmse(bvar_result)
    else:
        comp = None

    # ---------------------------------------------------------------
    # 4b. Three-way comparison (if conditional was run)
    # ---------------------------------------------------------------
    if cond_result is not None:
        print("\n--- 4b. Three-way RMSE comparison ---")
        _print_three_way(bvar_result, cond_result, qpm_rmse_path,
                         qpm_growth=qpm_growth)

    # ---------------------------------------------------------------
    # 5. Plots
    # ---------------------------------------------------------------
    if not args.no_plots:
        print("\n--- 5. Generating plots ---")
        if bvar_result is not None:
            plot_spaghetti(
                bvar_result, endo, dates, var_names, EVAL_VARS,
                label="BVAR (unconditional)",
                filename="forecast_spaghetti_bvar.png",
            )
        if cond_result is not None:
            plot_spaghetti(
                cond_result, endo, dates, var_names, EVAL_VARS,
                label="BVAR (conditional)",
                filename="forecast_spaghetti_cond.png",
            )
        if qpm_dict is not None:
            plot_spaghetti_qpm(
                qpm_dict, df,
                eval_vars_bvar=EVAL_VARS,
                filename="forecast_spaghetti_qpm.png",
            )
        if comp is not None:
            plot_rmse_comparison(comp)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

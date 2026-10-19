"""Brazil BVAR analysis: sign-restriction-identified impulse responses.

Implements the econometric assessment recommendations:
  - Target-demeaned inflation (IPCA − target)
  - p = 4 lags (standard quarterly)
  - COVID impulse dummies as exogenous regressors
  - Commodity prices as block-exogenous (Granger exogeneity)
  - Sub-sample comparison: pre-COVID vs full sample
  - 4-variable parsimonious and 5-variable extended specifications

Identification:
  - **Monetary policy shock**: sign restrictions (Uhlig 2005).
    Selic rate + (impact), inflation − (h=1..2), BRL/USD − (impact).

Usage:
    python run_bvar_analysis.py                  # full pipeline, both specs
    python run_bvar_analysis.py --spec 4var      # only 4-var
    python run_bvar_analysis.py --spec 5var      # only 5-var
    python run_bvar_analysis.py --skip-subsample # skip pre-COVID comparison
    python run_bvar_analysis.py --no-show        # suppress interactive plots
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Path setup: add the toolbox to sys.path
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_GPM_EST = _THIS_DIR.parents[1] / "gpm_estimation"
if str(_GPM_EST) not in sys.path:
    sys.path.insert(0, str(_GPM_EST))

from qpm_toolbox.var.bayesian import estimate_bvar, _create_dummy_data
from qpm_toolbox.var.empirical_bayes import (
    estimate_eb_bvar, EBConfig, _niw_log_marginal_likelihood,
)
from qpm_toolbox.var.irf import (
    irf_bvar_sign_restrictions, SignRestriction,
)
from qpm_toolbox.var.forecast import forecast_bvar
from qpm_toolbox.var.types import BVARPriorOptions

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_CSV = _THIS_DIR / "shared_data" / "transformed_data.csv"
FIG_DIR = _THIS_DIR / "figures" / "bvar"
RES_DIR = _THIS_DIR / "results" / "bvar"


# ======================================================================== #
#  1. Data loading and preparation
# ======================================================================== #

def load_data(
    start: str = "2004Q1",
    end: str = "2025Q2",
) -> pd.DataFrame:
    """Load transformed_data.csv and construct BVAR-ready variables.

    Returns a DataFrame indexed by quarterly PeriodIndex with all
    candidate variables.
    """
    raw = pd.read_csv(DATA_CSV)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()

    # Quarterly index
    raw.index = raw.index.to_period("Q")

    # Trim to sample
    raw = raw.loc[start:end]

    # Construct target-demeaned inflation
    raw["ipca_demeaned"] = raw["DA_CPI_Brazil"] - raw["Brazil_Inflation_Target"]

    # Construct core inflation (demeaned)
    if "DA_Core_IPCA_EX3" in raw.columns:
        raw["ipca_core"] = raw["DA_Core_IPCA_EX3"]
        raw["ipca_core_demeaned"] = (
            raw["DA_Core_IPCA_EX3"] - raw["Brazil_Inflation_Target"])
    elif "DA_Core_IPCA_Average" in raw.columns:
        raw["ipca_core"] = raw["DA_Core_IPCA_Average"]
        raw["ipca_core_demeaned"] = (
            raw["DA_Core_IPCA_Average"] - raw["Brazil_Inflation_Target"])

    # Rename for clarity
    cols = {
        "dlog_commod_brl":  raw["DA_IC_Br_BRL"],
        "gdp_gap":          raw["log100_hp_GDP_Brazil_SA"],
        "gdp_growth":       raw["DA_GDP_Brazil_SA"],
        "exp_inf":          raw["IPCA_12M_Expectation"],
        "ipca_demeaned":    raw["ipca_demeaned"],
        "ipca_raw":         raw["DA_CPI_Brazil"],
        "selic_rate":       raw["Brazil_Policy_Rate"],
        "dlog_brl_usd":    raw["DA_Exchange_Rate"],
        "cds5":             raw["CDS5_Brazil"],
    }
    if "ipca_core" in raw.columns:
        cols["ipca_core"] = raw["ipca_core"]
        cols["ipca_core_demeaned"] = raw["ipca_core_demeaned"]

    df = pd.DataFrame(cols, index=raw.index)

    # Report sample
    n_obs = len(df)
    n_miss = df.isna().sum()
    print(f"Sample: {df.index[0]} – {df.index[-1]}  ({n_obs} obs)")
    if n_miss.any():
        print("  Missing values:")
        for col in n_miss[n_miss > 0].index:
            print(f"    {col}: {n_miss[col]}")

    return df


def make_covid_dummies(df: pd.DataFrame) -> np.ndarray:
    """Build (T, 2) impulse dummy matrix for 2020Q1 and 2020Q2."""
    T = len(df)
    dummies = np.zeros((T, 2))
    idx = df.index
    for i, per in enumerate(idx):
        if per == pd.Period("2020Q1", freq="Q"):
            dummies[i, 0] = 1.0
        elif per == pd.Period("2020Q2", freq="Q"):
            dummies[i, 1] = 1.0
    return dummies


def make_exog_matrix(
    df: pd.DataFrame,
    p: int = 4,
    include_commodity_lags: bool = True,
    lambda3: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build full exogenous regressor matrix and prior scales.

    Includes COVID impulse dummies and commodity price growth
    (current + p lags), enforcing Granger exogeneity of world
    commodity prices for the Brazilian domestic block.

    Returns both the exogenous matrix and a per-column prior
    standard-deviation scale array.  The scale encodes:

    - COVID dummies: scale = 100 (very diffuse, like intercept).
    - Commodity current: scale = sigma_commod.
    - Commodity lag l: scale = sigma_commod / l^{lambda3}
      (Minnesota-style lag decay → stronger shrinkage for longer lags).

    These scales are passed to ``_create_dummy_data`` and combined
    with ``lambda_exog`` to set the effective prior precision:
    ``precision_j = 1 / (lambda_exog * scale_j)``.

    Args:
        df: Full DataFrame from load_data().
        p: Number of commodity lags to include.
        include_commodity_lags: If True, include lagged commodity values.
        lambda3: Lag-decay exponent (same as the endogenous Minnesota
                 parameter).

    Returns:
        (exog, prior_scales):
          - exog: (T, m) array.
          - prior_scales: (m,) array of prior std-dev scales per column.
    """
    covid = make_covid_dummies(df)
    commod = df["dlog_commod_brl"].values
    sigma_commod = np.nanstd(commod)

    cols = [covid]
    scales = []

    # COVID dummies: very diffuse prior (like intercept term)
    scales.extend([100.0, 100.0])

    # Current commodity value (lag 0)
    cols.append(commod[:, np.newaxis])
    scales.append(sigma_commod)

    # Lagged commodity values
    if include_commodity_lags:
        for lag in range(1, p + 1):
            lagged = np.full_like(commod, 0.0)
            lagged[lag:] = commod[:-lag]
            cols.append(lagged[:, np.newaxis])
            # Lag-decaying prior: tighter for longer lags
            scales.append(sigma_commod / lag**lambda3)

    exog = np.hstack(cols)
    prior_scales = np.array(scales)
    n_commod_cols = 1 + (p if include_commodity_lags else 0)
    print(f"  Exog matrix: {exog.shape[1]} columns "
          f"(2 COVID + {n_commod_cols} commodity)")
    print(f"  Exog prior scales: COVID={scales[0]:.1f}, "
          f"commod_t={scales[2]:.2f}, "
          f"commod_{{t-{p}}}={scales[-1]:.4f}")
    return exog, prior_scales


# ======================================================================== #
#  2. Variable specifications
# ======================================================================== #

def get_spec(
    name: str,
    gdp_measure: str = "gap",
    inf_measure: str = "demeaned",
) -> dict:
    """Return variable specification dict.

    Commodity prices are treated as block-exogenous (included in the
    exogenous regressor matrix rather than the endogenous system).

    Ordering follows the speed-of-adjustment logic (CEE 1999):
      real activity (sluggish) → expectations → inflation → policy → financial.

    Args:
        name: '4var' or '5var'.
        gdp_measure: 'gap' (HP-cycle output gap) or 'growth' (annualised
                     quarterly GDP growth, 400 x Dlog).
        inf_measure: 'demeaned' (IPCA minus target) or 'raw' (IPCA level).

    Keys: 'vars', 'delta', 'label', 'var_labels', 'inf_var'.
    """
    # GDP measure configuration
    if gdp_measure == "gap":
        gdp_var = "gdp_gap"
        gdp_label = "GDP gap (HP cycle)"
        gdp_delta = 0.0   # stationary
    elif gdp_measure == "growth":
        gdp_var = "gdp_growth"
        gdp_label = "GDP growth (ann. QoQ)"
        gdp_delta = 0.0   # stationary
    else:
        raise ValueError(f"Unknown gdp_measure: {gdp_measure}")

    # Inflation measure configuration
    if inf_measure == "demeaned":
        inf_var = "ipca_demeaned"
        inf_label = "Headline IPCA (demeaned)"
        inf_delta = 0.0   # stationary (target-demeaned)
    elif inf_measure == "raw":
        inf_var = "ipca_raw"
        inf_label = "Headline IPCA"
        inf_delta = 0.0   # stationary (QoQ annualised rate)
    elif inf_measure == "core":
        inf_var = "ipca_core"
        inf_label = "Core IPCA (EX3)"
        inf_delta = 0.0
    elif inf_measure == "core_demeaned":
        inf_var = "ipca_core_demeaned"
        inf_label = "Core IPCA (EX3, demeaned)"
        inf_delta = 0.0
    else:
        raise ValueError(f"Unknown inf_measure: {inf_measure}")

    suffix = f" [{gdp_measure}, {inf_measure}]"

    if name == "4var":
        return {
            "vars": [
                gdp_var,
                inf_var,
                "selic_rate",
                "dlog_brl_usd",
            ],
            "delta": np.array([gdp_delta, inf_delta, 0.8, 0.0]),
            "label": f"4-Variable Parsimonious{suffix}",
            "var_labels": [
                gdp_label,
                inf_label,
                "Selic rate",
                "BRL/USD depreciation",
            ],
            "inf_var": inf_var,
        }
    elif name == "5var":
        return {
            "vars": [
                gdp_var,
                "exp_inf",
                inf_var,
                "selic_rate",
                "dlog_brl_usd",
            ],
            "delta": np.array([gdp_delta, 0.8, inf_delta, 0.8, 0.0]),
            "label": f"5-Variable Extended{suffix}",
            "var_labels": [
                gdp_label,
                "Inflation expectations",
                inf_label,
                "Selic rate",
                "BRL/USD depreciation",
            ],
            "inf_var": inf_var,
        }
    else:
        raise ValueError(f"Unknown spec: {name}")


# ======================================================================== #
#  3. Sign-restriction specification for monetary policy shock
# ======================================================================== #

def get_mp_sign_restrictions(spec: dict) -> list[SignRestriction]:
    """Define sign restrictions for monetary-policy shock identification.

    Moderate (Option B) sign restrictions following Uhlig (2005):
      - Selic rate increases on impact (h = 0)
      - Inflation falls at h = 1, 2 (two quarters after impact)
      - BRL/USD depreciation rate falls on impact (h = 0): appreciation

    GDP/output is left unrestricted — the sacrifice ratio is the
    object of interest, so we avoid restricting its sign.

    The restriction horizons are kept short to avoid excessively
    constraining the posterior (Arias, Rubio-Ramírez & Waggoner 2018)
    while still separating monetary policy shocks from demand shocks.

    References:
        Uhlig (2005), *Journal of Monetary Economics* 52(2).
        Arias, Rubio-Ramírez & Waggoner (2018), *Econometrica* 86(2).
    """
    var_names = spec["vars"]
    inf_var = spec.get("inf_var", "ipca_demeaned")

    restrictions = [
        SignRestriction("selic_rate", "+", [0]),
        SignRestriction(inf_var, "-", [1, 2]),
        SignRestriction("dlog_brl_usd", "-", [0]),
    ]

    # If inflation expectations are in the system, require them to fall
    if "exp_inf" in var_names:
        restrictions.append(
            SignRestriction("exp_inf", "-", [1, 2])
        )

    return restrictions


# ======================================================================== #
#  4. Estimation
# ======================================================================== #

def run_bvar_estimation(
    df: pd.DataFrame,
    spec: dict,
    exog: np.ndarray | None = None,
    exog_prior_scales: np.ndarray | None = None,
    p: int = 4,
    n_draws: int = 5000,
    seed: int = 42,
    lambda1: float = 0.5,
) -> tuple:
    """Estimate BVAR for given specification.

    Returns (draws, posterior_params, diag).
    """
    endog = df[spec["vars"]].dropna()
    var_names = spec["vars"]

    # Align exog to the endog sample
    if exog is not None:
        # Use iloc positions matching endog within df
        mask = df.index.isin(endog.index)
        exog_aligned = exog[mask]
    else:
        exog_aligned = None

    prior = BVARPriorOptions(
        lambda1=lambda1,
        lambda2=0.5,
        lambda3=2.0,
        lambda4=100.0,
        delta=spec["delta"],
        add_sims=True,
        lambda_soc=1.0,
        lambda_dio=1.0,
        bvar_compat=True,
    )

    print(f"\n{'='*60}")
    print(f"Estimating BVAR: {spec['label']}")
    print(f"  Variables: {var_names}")
    print(f"  p = {p}, n_draws = {n_draws}, lambda1 = {lambda1}")
    print(f"  Sample: {endog.index[0]} – {endog.index[-1]} ({len(endog)} obs)")
    if exog_aligned is not None:
        print(f"  Exogenous regressors: {exog_aligned.shape[1]} columns")
    print(f"{'='*60}")

    draws, post, diag = estimate_bvar(
        endog.values, p, prior, n_draws=n_draws,
        constant=True, filter_unstable=True,
        var_names=var_names, seed=seed,
        exog=exog_aligned,
        exog_prior_scales=exog_prior_scales,
    )

    print(f"  Draws retained: {draws.n_draws} / {n_draws}")
    print(f"  Spectral radius (median): {diag.get('rho_median', 'N/A')}")

    return draws, post, diag


# ======================================================================== #
#  5. IRF computation and normalisation
# ======================================================================== #

def compute_mp_irf_sign_restricted(
    draws, spec: dict,
    horizon: int = 20,
    n_rotations: int = 500,
    seed: int = 42,
) -> "BVARIRFResult":
    """Compute sign-restriction-identified MP shock IRFs.

    The shock is normalised so that the Selic rate increases by
    1 pp (100 bp) on impact.

    Args:
        draws: BVAR posterior draws.
        spec: Variable specification dict.
        horizon: IRF horizon.
        n_rotations: Random rotations per posterior draw.
        seed: Random seed.

    Returns:
        BVARIRFResult with n_shocks = 1 (the identified MP shock),
        normalised to 100 bp at h = 0.
    """
    restrictions = get_mp_sign_restrictions(spec)

    print(f"\n  Sign-restriction identification of MP shock:")
    print(f"    Restrictions:")
    for r in restrictions:
        print(f"      {r.response} {r.sign} at h = {r.horizons}")

    mp_irf = irf_bvar_sign_restrictions(
        draws, horizon, restrictions,
        n_rotations=n_rotations,
        seed=seed,
        normalise_var="selic_rate",
        normalise_target=1.0,      # 100 bp = 1 pp
        normalise_horizon=0,
        shock_label="MP_shock",
    )
    return mp_irf


# ======================================================================== #
#  6. Sacrifice ratio and pass-through
# ======================================================================== #

def compute_sacrifice_ratio(mp_irf, spec: dict, max_h: int = 16) -> dict:
    """Cumulative sacrifice ratio from sign-restricted MP shock IRFs.

    Sacrifice ratio = cumulative GDP gap loss / cumulative inflation reduction.
    Both measured as cumulative sums of median IRF responses.

    The MP shock has shock_label = "MP_shock" (single identified shock,
    shock index 0 in the BVARIRFResult).
    """
    var_names = spec["vars"]
    # GDP variable is either gdp_gap or gdp_growth
    gdp_var = "gdp_gap" if "gdp_gap" in var_names else "gdp_growth"
    gdp_idx = var_names.index(gdp_var)
    inf_var = spec.get("inf_var", "ipca_demeaned")
    inf_idx = var_names.index(inf_var)

    si = 0  # sign-restricted result has only one shock

    horizons = np.arange(max_h + 1)
    cum_gdp = np.cumsum(mp_irf.median[gdp_idx, si, :max_h + 1])
    cum_inf = np.cumsum(mp_irf.median[inf_idx, si, :max_h + 1])

    # Sacrifice ratio: |cumulative output loss| / |cumulative inflation reduction|
    # After a contractionary shock, both cum_gdp and cum_inf are typically
    # negative.  SR = cum_gdp / cum_inf gives a positive number when both
    # are negative (output falls, inflation falls).
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(np.abs(cum_inf) > 1e-8, cum_gdp / cum_inf, np.nan)

    return {
        "horizons": horizons,
        "cum_gdp": cum_gdp,
        "cum_inf": cum_inf,
        "sacrifice_ratio": sr,
    }


# ======================================================================== #
#  7. Lag selection
# ======================================================================== #

def lag_selection(
    df: pd.DataFrame,
    spec: dict,
    exog_builder: callable,
    lags: list[int] | None = None,
    lambda1: float = 0.5,
) -> pd.DataFrame:
    """Compare log-marginal likelihoods for different lag orders.

    Uses the analytical Normal-Inverse-Wishart log-marginal likelihood
    (Koop & Korobilis 2010, Eq. 39).

    Args:
        exog_builder: Callable(df, p) -> (np.ndarray, np.ndarray) that
                      builds the exogenous matrix and prior scales for
                      a given lag order p.
    """
    from qpm_toolbox.var.core import build_design_matrices

    if lags is None:
        lags = [2, 4, 6]

    endog = df[spec["vars"]].dropna()
    K = len(spec["vars"])

    results = []
    for p_test in lags:
        print(f"  Lag selection: p = {p_test} ...")

        exog, exog_scales = exog_builder(df, p_test)
        mask = df.index.isin(endog.index)
        exog_aligned = exog[mask]

        prior = BVARPriorOptions(
            lambda1=lambda1, lambda2=0.5, lambda3=2.0, lambda4=100.0,
            delta=spec["delta"], add_sims=True,
            lambda_soc=1.0, lambda_dio=1.0, bvar_compat=True,
        )

        # Build data and dummy matrices
        vd = build_design_matrices(endog.values, p_test, constant=True,
                                   var_names=spec["vars"], exog=exog_aligned)
        Y, X, info = vd

        Y_d, X_d, _ = _create_dummy_data(prior, K, p_test, endog.values,
                                          n_exog=info.n_exog,
                                          exog_prior_scales=exog_scales)

        # Analytical log-marginal likelihood
        lml = _niw_log_marginal_likelihood(Y, X, Y_d, X_d, K)

        results.append({
            "p": p_test,
            "log_ml": lml,
            "T_eff": info.T,
        })

    return pd.DataFrame(results)


# ======================================================================== #
#  8. Empirical Bayes sensitivity
# ======================================================================== #

def run_eb_sensitivity(
    df: pd.DataFrame,
    spec: dict,
    exog: np.ndarray | None,
    exog_prior_scales: np.ndarray | None = None,
    p: int = 4,
    n_draws: int = 5000,
    seed: int = 42,
    lambda1_init: float = 0.5,
) -> dict:
    """Empirical Bayes optimisation of lambda1."""
    endog = df[spec["vars"]].dropna()

    if exog is not None:
        mask = df.index.isin(endog.index)
        exog_aligned = exog[mask]
    else:
        exog_aligned = None

    base_prior = BVARPriorOptions(
        lambda1=lambda1_init, lambda2=0.5, lambda3=2.0, lambda4=100.0,
        delta=spec["delta"], add_sims=True,
        lambda_soc=1.0, lambda_dio=1.0, bvar_compat=True,
    )

    eb_cfg = EBConfig(
        lambda1="optimize",
        lambda1_init=lambda1_init,
    )

    print(f"\n  Empirical Bayes optimisation ({spec['label']}):")
    eb_result = estimate_eb_bvar(
        endog.values, p, base_options=base_prior,
        eb_config=eb_cfg, n_draws=n_draws,
        constant=True, filter_unstable=True,
        var_names=spec["vars"], seed=seed,
        exog=exog_aligned,
        exog_prior_scales=exog_prior_scales,
    )

    return {
        "draws": eb_result.posterior_draws,
        "opt_params": eb_result.optimised_params,
        "log_ml": eb_result.log_ml,
    }


# ======================================================================== #
#  9. Sub-sample analysis
# ======================================================================== #

def run_subsample_analysis(
    df_full: pd.DataFrame,
    spec: dict,
    prefix: str,
    p: int = 4,
    n_draws: int = 5000,
    seed: int = 42,
    lambda1: float = 0.5,
    n_rotations: int = 500,
    sub_start: str = "2009Q1",
    sub_end: str = "2019Q4",
) -> dict:
    """Compare a restricted sub-sample vs the full sample.

    Full sample includes COVID dummies + commodity exog.
    Sub-sample: specified by sub_start/sub_end.

    Saves comparison figures and per-subsample IRF CSVs.

    Args:
        df_full: Full sample DataFrame.
        spec: Variable specification dict.
        prefix: File naming prefix (e.g. '4var_gap_demeaned').
        sub_start: Start period for sub-sample (e.g. '2009Q1').
        sub_end: End period for sub-sample (e.g. '2019Q4').
    """
    sub_label = f"{sub_start}_{sub_end}".replace("Q", "q")
    full_label = (f"{df_full.index[0]}_{df_full.index[-1]}"
                  .replace("Q", "q").replace(" ", ""))

    df_sub = df_full.loc[sub_start:sub_end]
    exog_full, scales_full = make_exog_matrix(df_full, p=p)
    exog_sub, scales_sub = make_exog_matrix(df_sub, p=p)

    print(f"\n--- Full sample ({df_full.index[0]}–{df_full.index[-1]}) ---")
    draws_full, _, _ = run_bvar_estimation(
        df_full, spec, exog=exog_full, exog_prior_scales=scales_full,
        p=p, n_draws=n_draws, seed=seed, lambda1=lambda1)

    print(f"\n--- Sub-sample ({sub_start}–{sub_end}) ---")
    draws_sub, _, _ = run_bvar_estimation(
        df_sub, spec, exog=exog_sub, exog_prior_scales=scales_sub,
        p=p, n_draws=n_draws, seed=seed, lambda1=lambda1)

    horizon = 20

    # Sign-restricted MP shock for both
    mp_irf_full = compute_mp_irf_sign_restricted(
        draws_full, spec, horizon=horizon,
        n_rotations=n_rotations, seed=seed)
    mp_irf_sub = compute_mp_irf_sign_restricted(
        draws_sub, spec, horizon=horizon,
        n_rotations=n_rotations, seed=seed)

    # Sacrifice ratios
    sr_full = compute_sacrifice_ratio(mp_irf_full, spec)
    sr_sub = compute_sacrifice_ratio(mp_irf_sub, spec)

    print(f"\n  Sacrifice ratio comparison (h=12):")
    print(f"    Full sample: {sr_full['sacrifice_ratio'][12]:.2f}")
    print(f"    Sub-sample:  {sr_sub['sacrifice_ratio'][12]:.2f}")

    # --- Export CSVs ---
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # Sub-sample IRFs
    export_irf_csv(mp_irf_sub, spec, "MP_shock",
                   RES_DIR / f"{prefix}_{sub_label}_irf_mp_sign.csv")
    export_sacrifice_ratio_csv(
        sr_sub, RES_DIR / f"{prefix}_{sub_label}_sacrifice_ratio.csv")

    # Full-sample IRFs
    export_irf_csv(mp_irf_full, spec, "MP_shock",
                   RES_DIR / f"{prefix}_{full_label}_irf_mp_sign.csv")
    export_sacrifice_ratio_csv(
        sr_full, RES_DIR / f"{prefix}_{full_label}_sacrifice_ratio.csv")

    return {
        "draws_full": draws_full,
        "draws_sub": draws_sub,
        "mp_irf_full": mp_irf_full,
        "mp_irf_sub": mp_irf_sub,
        "sr_full": sr_full,
        "sr_sub": sr_sub,
        "sub_label": sub_label,
    }


# ======================================================================== #
#  10. Plotting
# ======================================================================== #

_SHOCK_DISPLAY_LABELS = {
    "MP_shock": "Monetary Policy (sign-restricted)",
}


def _irf_var_label(var_name: str, spec: dict) -> str:
    """Map variable name to a display label."""
    idx = spec["vars"].index(var_name)
    return spec["var_labels"][idx]


def _shock_display_label(shock_name: str) -> str:
    """Map shock name to a display label."""
    return _SHOCK_DISPLAY_LABELS.get(shock_name, shock_name)


def plot_irf_panel(
    irf_result, spec: dict, shock_var: str,
    title_suffix: str = "", horizon: int | None = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Plot IRF panel: all variable responses to a single shock.

    Works with both Cholesky and sign-restricted IRFs.  The shock
    is identified by *shock_var* which must appear in
    ``irf_result.shock_names``.
    """
    var_names = spec["vars"]
    K = len(var_names)
    shock_names = irf_result.shock_names
    si = shock_names.index(shock_var)

    if horizon is None:
        horizon = irf_result.horizon
    hs = np.arange(horizon + 1)

    ncols = min(3, K)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    shock_label = _shock_display_label(shock_var)
    fig.suptitle(
        f"IRFs to {shock_label} shock — {spec['label']}{title_suffix}",
        fontsize=13, fontweight="bold")

    for i, vname in enumerate(var_names):
        ax = axes.flat[i]
        ri = var_names.index(vname)

        med = irf_result.median[ri, si, :horizon + 1]
        lo = irf_result.p16[ri, si, :horizon + 1]
        hi = irf_result.p84[ri, si, :horizon + 1]

        ax.fill_between(hs, lo, hi, alpha=0.25, color="steelblue")
        ax.plot(hs, med, "o-", color="steelblue", markersize=3, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(_irf_var_label(vname, spec), fontsize=10)
        ax.set_xlim(0, horizon)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Quarters")

    # Hide unused axes
    for j in range(K, nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_irf_comparison(
    irf_a, irf_b, spec: dict, shock_var: str,
    label_a: str = "Full sample", label_b: str = "Pre-COVID",
    horizon: int = 20, figsize: tuple = (14, 10),
) -> plt.Figure:
    """Overlay two IRF sets for sub-sample comparison."""
    var_names = spec["vars"]
    K = len(var_names)
    si = irf_a.shock_names.index(shock_var)
    hs = np.arange(horizon + 1)

    ncols = min(3, K)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    shock_label = _shock_display_label(shock_var)
    fig.suptitle(
        f"IRF comparison: {shock_label} — {spec['label']}",
        fontsize=13, fontweight="bold")

    for i, vname in enumerate(var_names):
        ax = axes.flat[i]
        ri = var_names.index(vname)

        med_a = irf_a.median[ri, si, :horizon + 1]
        lo_a = irf_a.p16[ri, si, :horizon + 1]
        hi_a = irf_a.p84[ri, si, :horizon + 1]

        med_b = irf_b.median[ri, si, :horizon + 1]
        lo_b = irf_b.p16[ri, si, :horizon + 1]
        hi_b = irf_b.p84[ri, si, :horizon + 1]

        ax.fill_between(hs, lo_a, hi_a, alpha=0.15, color="steelblue")
        ax.plot(hs, med_a, "-", color="steelblue", linewidth=1.5, label=label_a)

        ax.fill_between(hs, lo_b, hi_b, alpha=0.15, color="firebrick")
        ax.plot(hs, med_b, "--", color="firebrick", linewidth=1.5, label=label_b)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(_irf_var_label(vname, spec), fontsize=10)
        ax.set_xlim(0, horizon)
        if i == 0:
            ax.legend(fontsize=8)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Quarters")

    for j in range(K, nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_forecast_fan(
    draws, df: pd.DataFrame, spec: dict,
    horizon: int = 12, exog_future: np.ndarray | None = None,
    seed: int = 42, figsize: tuple = (14, 10),
) -> plt.Figure:
    """Fan chart: stochastic forecast distribution."""
    from qpm_toolbox.var.core import build_design_matrices

    endog = df[spec["vars"]].dropna()
    vd = build_design_matrices(endog.values, draws.p, constant=True,
                               var_names=spec["vars"])
    last_lags = vd.info.last_lags

    fc = forecast_bvar(
        draws, last_lags, horizon,
        add_shocks=True, seed=seed,
        exog_future=exog_future,
    )

    var_names = spec["vars"]
    K = len(var_names)
    ncols = min(3, K)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    # Historical tail for context
    n_hist = min(20, len(endog))
    hist = endog.iloc[-n_hist:]

    fig.suptitle(
        f"Forecast fan chart — {spec['label']}",
        fontsize=13, fontweight="bold")

    for i, vname in enumerate(var_names):
        ax = axes.flat[i]

        # Historical
        hist_x = np.arange(-n_hist, 0)
        ax.plot(hist_x, hist[vname].values, "k-", linewidth=1)

        # Forecast
        fc_x = np.arange(1, horizon + 1)
        p5 = np.percentile(fc.forecasts[:, :, i], 5, axis=0)
        p16 = fc.p16[:, i]
        p50 = fc.median[:, i]
        p84 = fc.p84[:, i]
        p95 = np.percentile(fc.forecasts[:, :, i], 95, axis=0)

        ax.fill_between(fc_x, p5, p95, alpha=0.1, color="steelblue")
        ax.fill_between(fc_x, p16, p84, alpha=0.25, color="steelblue")
        ax.plot(fc_x, p50, "o-", color="steelblue", markersize=3)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.set_title(_irf_var_label(vname, spec), fontsize=10)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Quarters from end of sample")

    for j in range(K, nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ======================================================================== #
#  11. CSV export
# ======================================================================== #

def export_irf_csv(
    irf_result, spec: dict, shock_var: str,
    filepath: Path, horizon: int | None = None,
):
    """Export IRF median and bands to CSV.

    Works with both Cholesky (multi-shock) and sign-restricted
    (single-shock) BVARIRFResult objects.
    """
    var_names = spec["vars"]
    si = irf_result.shock_names.index(shock_var)
    if horizon is None:
        horizon = irf_result.horizon
    hs = np.arange(horizon + 1)

    rows = []
    for h in hs:
        row = {"horizon": h}
        for ri, vname in enumerate(var_names):
            row[f"{vname}_median"] = irf_result.median[ri, si, h]
            row[f"{vname}_p16"] = irf_result.p16[ri, si, h]
            row[f"{vname}_p84"] = irf_result.p84[ri, si, h]
        rows.append(row)

    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")


def export_sacrifice_ratio_csv(sr_dict: dict, filepath: Path):
    """Export sacrifice ratio profile to CSV."""
    df = pd.DataFrame({
        "horizon": sr_dict["horizons"],
        "cum_gdp_gap": sr_dict["cum_gdp"],
        "cum_inflation": sr_dict["cum_inf"],
        "sacrifice_ratio": sr_dict["sacrifice_ratio"],
    })
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")


# ======================================================================== #
#  Main pipeline
# ======================================================================== #

def run_single_spec(
    spec_name: str,
    df: pd.DataFrame,
    p: int = 4,
    lambda1: float = 0.5,
    do_subsample: bool = True,
    show_plots: bool = True,
    n_rotations: int = 500,
    seed: int = 42,
    gdp_measure: str = "gap",
    inf_measure: str = "demeaned",
    sample_label: str = "",
    subsample_start: str = "2009Q1",
    subsample_end: str = "2019Q4",
):
    """Run the full analysis pipeline for one specification.

    Identification:
      - MP shock: sign restrictions (Uhlig 2005), normalised to 100bp.
    """
    spec = get_spec(spec_name, gdp_measure=gdp_measure,
                    inf_measure=inf_measure)
    prefix = f"{spec_name}_{gdp_measure}_{inf_measure}"
    if sample_label:
        prefix = f"{prefix}_{sample_label}"

    # Build exogenous matrix: COVID dummies + commodity (current + lags)
    exog, exog_prior_scales = make_exog_matrix(df, p=p)

    # --- Estimation ---
    draws, post, diag = run_bvar_estimation(
        df, spec, exog=exog, exog_prior_scales=exog_prior_scales,
        p=p, lambda1=lambda1, seed=seed)

    # --- MP shock: sign-restriction identification ---
    mp_irf = compute_mp_irf_sign_restricted(
        draws, spec, horizon=20,
        n_rotations=n_rotations, seed=seed)

    # --- Sacrifice ratio (from sign-restricted MP shock) ---
    sr = compute_sacrifice_ratio(mp_irf, spec)
    print(f"\n  Sacrifice ratio at h=8: {sr['sacrifice_ratio'][8]:.2f}")
    print(f"  Sacrifice ratio at h=12: {sr['sacrifice_ratio'][12]:.2f}")

    # --- Lag selection ---
    print("\n  Lag selection:")
    exog_builder = lambda df_, p_: make_exog_matrix(df_, p=p_)
    lag_df = lag_selection(df, spec, exog_builder, lambda1=lambda1)
    print(lag_df.to_string(index=False))

    # --- EB sensitivity ---
    eb = run_eb_sensitivity(df, spec, exog,
                            exog_prior_scales=exog_prior_scales,
                            lambda1_init=lambda1, seed=seed)

    # --- Figures ---
    figs = {}

    # MP shock IRF panel (sign-restricted, normalised 100bp)
    fig_mp = plot_irf_panel(
        mp_irf, spec, "MP_shock",
        title_suffix=" (100bp, sign-restricted)")
    figs[f"{prefix}_irf_mp_sign"] = fig_mp

    # Forecast fan chart
    fig_fan = plot_forecast_fan(draws, df, spec, seed=seed)
    figs[f"{prefix}_forecast_fan"] = fig_fan

    # --- Sub-sample comparison ---
    if do_subsample:
        sub = run_subsample_analysis(
            df, spec, prefix, p=p, lambda1=lambda1, seed=seed,
            n_rotations=n_rotations,
            sub_start=subsample_start, sub_end=subsample_end)

        sub_lbl = sub["sub_label"]
        label_b = f"{subsample_start}–{subsample_end}"

        # MP shock comparison (sign-restricted)
        fig_mp_comp = plot_irf_comparison(
            sub["mp_irf_full"], sub["mp_irf_sub"], spec, "MP_shock",
            label_a="Full sample", label_b=label_b)
        figs[f"{prefix}_irf_mp_subsample_{sub_lbl}"] = fig_mp_comp

    # --- Save figures ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in figs.items():
        fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"\n  Figures saved to {FIG_DIR}/")

    # --- Save CSVs ---
    RES_DIR.mkdir(parents=True, exist_ok=True)
    export_irf_csv(mp_irf, spec, "MP_shock",
                   RES_DIR / f"{prefix}_irf_mp_sign.csv")
    export_sacrifice_ratio_csv(sr, RES_DIR / f"{prefix}_sacrifice_ratio.csv")
    lag_df.to_csv(RES_DIR / f"{prefix}_lag_selection.csv", index=False)
    print(f"  CSVs saved to {RES_DIR}/")

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return {
        "draws": draws,
        "mp_irf": mp_irf,
        "sacrifice_ratio": sr,
        "lag_selection": lag_df,
        "eb": eb,
        "figs": figs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Brazil BVAR analysis (sign-restriction identification)")
    parser.add_argument("--spec", choices=["4var", "5var", "both"],
                        default="both")
    parser.add_argument("--skip-subsample", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--lambda1", type=float, default=0.5)
    parser.add_argument("--n-draws", type=int, default=5000)
    parser.add_argument("--n-rotations", type=int, default=500,
                        help="Random rotations per draw for sign restrictions")
    parser.add_argument("--gdp-measure", choices=["gap", "growth", "both"],
                        default="gap",
                        help="GDP measure: 'gap' (HP cycle), 'growth' "
                             "(ann. QoQ), or 'both' for comparison")
    parser.add_argument("--inf-measure",
                        choices=["demeaned", "raw", "core", "core_demeaned",
                                 "headline_both", "all"],
                        default="demeaned",
                        help="Inflation measure: 'demeaned' (headline IPCA "
                             "minus target), 'raw' (headline IPCA), 'core' "
                             "(Core IPCA EX3), 'core_demeaned' (Core minus "
                             "target), 'headline_both' (demeaned+raw), "
                             "'all' (all four)")
    parser.add_argument("--start", type=str, default="2004Q1",
                        help="Sample start period (e.g. '2009Q1')")
    parser.add_argument("--end", type=str, default="2025Q2",
                        help="Sample end period (e.g. '2019Q4')")
    parser.add_argument("--sub-start", type=str, default="2009Q1",
                        help="Subsample comparison start (e.g. '2009Q1')")
    parser.add_argument("--sub-end", type=str, default="2019Q4",
                        help="Subsample comparison end (e.g. '2019Q4')")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Build a sample label for file naming if non-default sample
    sample_label = ""
    if args.start != "2004Q1" or args.end != "2025Q2":
        sample_label = (f"{args.start}_{args.end}"
                        .replace("Q", "q"))

    # Load data
    df = load_data(start=args.start, end=args.end)

    specs = []
    if args.spec in ("4var", "both"):
        specs.append("4var")
    if args.spec in ("5var", "both"):
        specs.append("5var")

    gdp_measures = []
    if args.gdp_measure in ("gap", "both"):
        gdp_measures.append("gap")
    if args.gdp_measure in ("growth", "both"):
        gdp_measures.append("growth")

    inf_measures = []
    if args.inf_measure in ("demeaned", "headline_both", "all"):
        inf_measures.append("demeaned")
    if args.inf_measure in ("raw", "headline_both", "all"):
        inf_measures.append("raw")
    if args.inf_measure in ("core", "all"):
        inf_measures.append("core")
    if args.inf_measure in ("core_demeaned", "all"):
        inf_measures.append("core_demeaned")
    # Single option that doesn't match any of the above
    if not inf_measures:
        inf_measures.append(args.inf_measure)

    results = {}
    for sp in specs:
        for gm in gdp_measures:
            for im in inf_measures:
                key = f"{sp}_{gm}_{im}"
                results[key] = run_single_spec(
                    sp, df,
                    p=args.p, lambda1=args.lambda1,
                    do_subsample=not args.skip_subsample,
                    show_plots=not args.no_show,
                    n_rotations=args.n_rotations,
                    seed=args.seed,
                    gdp_measure=gm,
                    inf_measure=im,
                    sample_label=sample_label,
                    subsample_start=args.sub_start,
                    subsample_end=args.sub_end,
                )

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print(f"Figures: {FIG_DIR}/")
    print(f"Results: {RES_DIR}/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()

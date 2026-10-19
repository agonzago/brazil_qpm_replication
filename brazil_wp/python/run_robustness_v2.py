"""Prior sensitivity analysis for the Brazil QPM v2 (real-level UIP).

Robustness specifications:

  rub1 -- Wider structural priors (halve concentration on key params)
  rub3 -- Weaker Taylor rule (inflation response mode 1.00 vs. 1.50)
  rub4 -- Pure random walk PI_TREND (alpha_pi fixed at zero)
  rub5 -- Focus IPCA expectations as second PI_TREND measure (16 obs)

Each specification re-runs the full pipeline: calibration + NUTS.
Structural parameters not listed explicitly inherit the baseline values.

Uses the v2 model (brazil_qpm_mr_v2.qpm) with real-level UIP on LZ_GAP
and exact diffuse Kalman filter initialization (Durbin & Koopman 2012)
by default.

Usage:
    python run_robustness_v2.py --spec rub1
    python run_robustness_v2.py --spec rub4
    python run_robustness_v2.py --spec rub5
    python run_robustness_v2.py --spec all
    python run_robustness_v2.py --spec all --no-diffuse   # DARE init instead

Windows server: no unicode, no special characters.
"""

import os
import sys
import copy
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX_DIR = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX_DIR)

# ---------------------------------------------------------------------------
# JAX / numpyro setup -- only when running a single spec (worker process).
# When --spec all, this process is just a subprocess launcher (like
# run_all_models_v2.py) and must NOT initialise JAX.
# ---------------------------------------------------------------------------
def _is_single_spec():
    """Return True if this invocation will run a single spec in-process."""
    try:
        idx = sys.argv.index('--spec')
        val = sys.argv[idx + 1]
        # Single spec if no comma and not "all"
        return ',' not in val and val != 'all'
    except (ValueError, IndexError):
        return False

if _is_single_spec():
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(var, '1')

    # Match N_PARALLEL to --chains so XLA creates exactly the right
    # number of virtual devices.
    _chains = 8  # default, overridden by --chains below
    try:
        _ci = sys.argv.index('--chains')
        _chains = int(sys.argv[_ci + 1])
    except (ValueError, IndexError):
        pass
    N_PARALLEL = int(os.environ.get('N_PARALLEL', _chains))
    os.environ['XLA_FLAGS'] = (
        os.environ.get('XLA_FLAGS', '') +
        ' --xla_force_host_platform_device_count={}'.format(N_PARALLEL)
    ).strip()

    import jax
    jax.config.update("jax_enable_x64", True)

    import numpyro
    numpyro.set_host_device_count(N_PARALLEL)

    import numpy as np
    import pandas as pd
    from qpm_toolbox import QPMModel
else:
    # Batch parent: only needs lightweight imports for subprocess management.
    # numpy/pandas/QPMModel are not needed.
    import numpy as np
    import pandas as pd

# ======================================================================
# Paths -- v2 model files
# ======================================================================
# v5+ TT-trend variant selection (env-var driven via _v5_sweep_config).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v5_sweep_config as _V5SC

# RUB_BASE_MODEL: select the underlying IS-curve structural form for the
# RUB suite.  Default 'a' (Model A's full Koyck filters) preserves the
# legacy behaviour.  Set to 'c' to put the RUBs on top of Model C
# (drops rho_QRR, rho_QZ, rho_QZ_E, rho_QZ_F Koyck filters).
_RUB_BASE = os.environ.get('RUB_BASE_MODEL', 'a').strip().lower()
if _RUB_BASE not in ('a', 'c'):
    raise ValueError(f"RUB_BASE_MODEL must be 'a' or 'c', got '{_RUB_BASE}'")

if _V5SC.V5_TT_MODE == 'tt_trend':
    if _RUB_BASE == 'c':
        MOD_FILE = os.path.join(SCRIPT_DIR, "model",
                                "brazil_qpm_model_c_tt_trend_v2.qpm")
        MOD_FILE_FOCUS = os.path.join(SCRIPT_DIR, "model",
                                      "brazil_qpm_model_c_focus_tt_trend_v2.qpm")
    else:
        MOD_FILE = os.path.join(SCRIPT_DIR, "model",
                                "brazil_qpm_mr_tt_trend_v2.qpm")
        MOD_FILE_FOCUS = os.path.join(SCRIPT_DIR, "model",
                                      "brazil_qpm_mr_focus_tt_trend_v2.qpm")
else:
    MOD_FILE = os.path.join(SCRIPT_DIR, "model", "brazil_qpm_mr_v2.qpm")
    MOD_FILE_FOCUS = os.path.join(SCRIPT_DIR, "model",
                                  "brazil_qpm_mr_focus_v2.qpm")
_LOCAL_CSV = os.path.join(SCRIPT_DIR, "data", "estimation_data.csv")
_TOOLBOX_CSV = os.path.join(TOOLBOX_DIR, "data", "brazil", "estimation_data.csv")
DATA_CSV = _LOCAL_CSV if os.path.exists(_LOCAL_CSV) else _TOOLBOX_CSV

# ======================================================================
# NUTS settings
# ======================================================================
WARMUP = 600
SAMPLES = 1500
CHAINS = 8

# ======================================================================
# Baseline configuration (identical to run_brazil_wp_v2.py)
# ======================================================================
CYCLE_MAP = {
    # OBS_LGDP cycle SNR 0.30->0.20 (consistent across all v2 runners).
    'OBS_LGDP':           ('SHK_GDP_GAP',         0.20),
    'OBS_DA_LCPI':        ('SHK_DA_LCPI',         0.30),
    'OBS_DA_LCPIX':       ('SHK_DA_LCPIX',        0.30),
    'OBS_DA_LCPI_ENERGY': ('SHK_DA_LCPI_ENERGY',  0.50),
    'OBS_DA_LCPI_US':     ('SHK_DA_LCPI_US',      0.30),
    'OBS_DA_LS':          ('SHK_LZ_GAP',          0.25),
    'OBS_LRWOIL_GAP':     ('SHK_LRWOIL_GAP',     0.50),
    'OBS_LRWFOOD_GAP':    ('SHK_LRWFOOD_GAP',    0.50),
    'OBS_GDP_GAP_US':     ('SHK_GAP_US',          0.30),
    'OBS_PREM':           ('SHK_PREM_GAP',        0.50),
    'OBS_RS':             ('SHK_RS',              0.50),
    'OBS_RS_US':          ('SHK_RS_US',           0.50),
    'OBS_TT':             ('SHK_TT_GAP',          0.50),
}

MANUAL_CYCLE_PRIORS = [
    {"name": "sigma_SHK_DA_LCPI_FOOD",
     "prior": "invgamma",
     "args": {"concentration": 3.0, "rate": 4.0 * 2.0}},
]

TREND_SNR_MAP = {
    # LGDP_BAR/G_TREND loosened (consistent with Models A and F).
    'SHK_LGDP_BAR':   ('OBS_LGDP',      'output_idio_trend',  50,  'level'),
    'SHK_G_TREND':     ('OBS_LGDP',      'output_idio_trend', 200,  'growth'),
    'SHK_LZ_BAR':      ('OBS_DA_LS',     'reer_idio_trend',   200,  'level'),
    'SHK_DLZ_TREND':   ('OBS_DA_LS',     'reer_idio_trend',   400,  'growth'),
    'SHK_PREM_BAR':    ('OBS_PREM',      'prem_idio_trend',   150,  'level'),
    'SHK_RR_BAR':      ('OBS_RS',        'rate_idio_trend',   1200, 'level'),
    'SHK_PI_TREND':    ('OBS_DA_LCPI',   'pi_idio_trend',     2400, 'level'),
    # RR_BAR_US loosened from 200 to 50 to track the observed HLW r*
    # series closely (consistent across all v2 model runners).
    'SHK_RR_BAR_US':   ('OBS_RR_BAR_US', 'rate_idio_trend',    50,  'level'),
    # SHK_TT_BAR: v5 (2026-05-10) — SNR=100 calibration + tight α=5 prior.
    'SHK_TT_BAR':      ('OBS_TT',        'tt_idio_trend',     100, 'level'),
    'SHK_PI_TREND_US': ('OBS_DA_LCPI_US','pi_idio_trend',    24000, 'level'),
}

FIXED_POINT_CONFIG = {
    'max_iter': 50,
    'tol': 1e-4,
    'damping': 0.5,
    'sigma_bounds': {
        'sigma_SHK_G_TREND':  (1e-8, 0.40),
        'sigma_SHK_RR_BAR':  (1e-8, 0.50),
    },
}

FIXED_PARAMS = {
    'lambda1s': 0.1581,
    'lambda2s': 0.1142,
    # sigma_SHK_TT_BAR moved to TREND_SNR_MAP in v5 (2026-05-10).
}

MEAS_NOISE_STD = {
    'OBS_LGDP': 0.0, 'OBS_DA_LCPI': 0.0, 'OBS_DA_LCPIX': 0.0,
    'OBS_DA_LCPI_ENERGY': 0.0, 'OBS_DA_LCPI_US': 0.0, 'OBS_DA_LS': 0.0,
    'OBS_LRWOIL_GAP': 2.0, 'OBS_LRWFOOD_GAP': 2.0,
    'OBS_GDP_GAP_US': 0.5, 'OBS_PREM': 0.5,
    'OBS_RS': 0.0, 'OBS_RS_US': 0.0, 'OBS_TT': 0.0,
    'OBS_DA_LCPI_TAR': 0.3,
    'OBS_RR_BAR_US': 0.5,
}

OBS_TREND_MAP = {
    "OBS_LGDP": "LGDP_BAR", "OBS_DA_LCPI": "PI_TREND",
    "OBS_DA_LCPIX": "PI_TREND", "OBS_DA_LCPI_ENERGY": "PI_TREND",
    "OBS_DA_LCPI_US": "PI_TREND_US", "OBS_DA_LS": None,
    "OBS_LRWOIL_GAP": None, "OBS_LRWFOOD_GAP": None,
    "OBS_GDP_GAP_US": None, "OBS_PREM": "PREM_BAR",
    "OBS_RS": "RS_TREND", "OBS_RS_US": "RS_US_TREND",
    "OBS_TT": "TT_BAR", "OBS_DA_LCPI_TAR": "PI_TREND",
    "OBS_RR_BAR_US": "RR_BAR_US",
}

# ======================================================================
# Baseline structural priors (verbatim from run_brazil_wp_v2.py)
# ======================================================================
BASELINE_PRIORS = [
    # IS Curve
    {"name": "beta1",     "prior": "beta",  "args": {"concentration1": 7.80, "concentration2": 4.20}},
    {"name": "beta2",     "prior": "beta",  "args": {"concentration1": 1.70, "concentration2": 7.30}},
    {"name": "beta3",     "prior": "beta",  "args": {"concentration1": 2.28, "concentration2": 7.72}},
    {"name": "beta4",     "prior": "beta",  "args": {"concentration1": 2.60, "concentration2": 7.40}},
    {"name": "beta5",     "prior": "gamma", "args": {"concentration": 2.8604, "rate": 169.1271}},
    {"name": "phi_tt_is", "prior": "gamma", "args": {"concentration": 3.2553, "rate": 45.1062}},
    # Phillips Curve
    {"name": "lambda1x",  "prior": "beta",  "args": {"concentration1": 5.80, "concentration2": 4.20}},
    {"name": "lambda2x",  "prior": "gamma", "args": {"concentration": 4.3290, "rate": 20.8062}},
    {"name": "kappa_z",   "prior": "gamma", "args": {"concentration": 3.2553, "rate": 90.2124}},
    {"name": "lambda4x",  "prior": "halfnormal", "args": {"scale": 0.03}},
    {"name": "lambda5x",  "prior": "gamma", "args": {"concentration": 2.1816, "rate": 295.4066}},
    {"name": "lambda6x",  "prior": "gamma", "args": {"concentration": 2.6180, "rate": 202.2542}},
    # Component Inflations
    {"name": "lambda_E1", "prior": "beta",  "args": {"concentration1": 11.20, "concentration2": 2.80}},
    {"name": "theta_E",   "prior": "gamma", "args": {"concentration": 3.1194, "rate": 70.6476}},
    {"name": "lambda_F1", "prior": "beta",  "args": {"concentration1": 6.20, "concentration2": 3.80}},
    {"name": "theta_F",   "prior": "gamma", "args": {"concentration": 3.4914, "rate": 62.2839}},
    # Taylor Rule
    {"name": "gamma1",    "prior": "beta",  "args": {"concentration1": 8.50, "concentration2": 3.50}},
    {"name": "gamma2",    "prior": "gamma", "args": {"concentration": 4.0000, "rate": 2.0000}},
    {"name": "gamma4",    "prior": "gamma", "args": {"concentration": 3.5302, "rate": 12.5260}},
    # UIP / Exchange Rate
    {"name": "rho_ls",    "prior": "beta",  "args": {"concentration1": 3.10, "concentration2": 4.90}},
    {"name": "psi_uip",   "prior": "gamma", "args": {"concentration": 3.4914, "rate": 2.4914}},
    # Risk Premium
    {"name": "rho_prem_gap", "prior": "beta",  "args": {"concentration1": 5.80, "concentration2": 4.20}},
    {"name": "phi_tt_gap",   "prior": "gamma", "args": {"concentration": 3.4914, "rate": 24.9136}},
    # Koyck Filters
    {"name": "rho_QZ",    "prior": "beta",  "args": {"concentration1": 11.80, "concentration2": 2.20}},
    {"name": "rho_QRR",   "prior": "beta",  "args": {"concentration1": 9.50, "concentration2": 2.50}},
    {"name": "rho_QZ_E",  "prior": "beta",  "args": {"concentration1": 9.80, "concentration2": 2.20}},
    {"name": "rho_QZ_F",  "prior": "beta",  "args": {"concentration1": 9.80, "concentration2": 2.20}},
    # Persistence
    {"name": "rho_tt",           "prior": "beta", "args": {"concentration1": 7.00, "concentration2": 3.00}},
    {"name": "rho_lrwoil_gap",   "prior": "beta", "args": {"concentration1": 8.57, "concentration2": 3.43}},
    {"name": "rho_lrwfood_gap",  "prior": "beta", "args": {"concentration1": 8.54, "concentration2": 3.46}},
    {"name": "rho_resn_gdp_gap", "prior": "beta", "args": {"concentration1": 3.00, "concentration2": 3.00}},
    {"name": "rho_res_rs",       "prior": "beta", "args": {"concentration1": 3.00, "concentration2": 3.00}},
    # US Block
    {"name": "rho_US",      "prior": "beta",  "args": {"concentration1": 11.56, "concentration2": 2.44}},
    {"name": "beta1_US",    "prior": "beta",  "args": {"concentration1": 5.79, "concentration2": 4.21}},
    {"name": "beta2_US",    "prior": "gamma", "args": {"concentration": 3.8112, "rate": 19.5223}},
    {"name": "beta3_us",    "prior": "gamma", "args": {"concentration": 3.4914, "rate": 12.4568}},
    {"name": "rho_pie_us",  "prior": "beta",  "args": {"concentration1": 3.70, "concentration2": 4.30}},
    {"name": "gamma2_us",   "prior": "gamma", "args": {"concentration": 4.0000, "rate": 2.0000}},
    {"name": "gamma4_us",   "prior": "gamma", "args": {"concentration": 4.0000, "rate": 20.0000}},
    # ToT channels
    {"name": "phi_tt_pc",   "prior": "halfnormal", "args": {"scale": 0.03}},
    {"name": "rho_QTT",     "prior": "beta",       "args": {"concentration1": 3.00, "concentration2": 3.00}},
    # Trend Convergence Speeds
    {"name": "alpha_g",      "prior": "beta", "args": {"concentration1": 1.80, "concentration2": 8.20}},
    {"name": "alpha_dlz",    "prior": "beta", "args": {"concentration1": 1.80, "concentration2": 8.20}},
    {"name": "alpha_prem",   "prior": "beta", "args": {"concentration1": 1.80, "concentration2": 8.20}},
    {"name": "alpha_rr",     "prior": "beta", "args": {"concentration1": 1.56, "concentration2": 8.44}},
    {"name": "alpha_pi",     "prior": "beta", "args": {"concentration1": 2.20, "concentration2": 7.80}},
    {"name": "alpha_rr_us",  "prior": "beta", "args": {"concentration1": 1.56, "concentration2": 8.44}},
    {"name": "alpha_pi_us",  "prior": "beta", "args": {"concentration1": 1.56, "concentration2": 8.44}},
]


# When RUB_BASE_MODEL='c' the underlying mod file drops the rho_QRR,
# rho_QZ_E, rho_QZ_F Koyck-filter parameters; strip their priors so the
# sampler does not pick up parameters that do not exist in the model.
if _RUB_BASE == 'c':
    _MODEL_C_DROPPED = {'rho_QRR', 'rho_QZ_E', 'rho_QZ_F'}
    BASELINE_PRIORS = [p for p in BASELINE_PRIORS
                       if p['name'] not in _MODEL_C_DROPPED]


# ======================================================================
# Helper: replace a named prior in the list
# ======================================================================
def replace_prior(priors, name, new_prior):
    """Return a copy of priors with the entry for `name` replaced."""
    out = copy.deepcopy(priors)
    for i, p in enumerate(out):
        if p["name"] == name:
            out[i] = new_prior
            return out
    raise ValueError("Parameter '{}' not found in priors".format(name))


def replace_priors(priors, replacements):
    """Replace multiple priors. `replacements` is a list of dicts."""
    out = copy.deepcopy(priors)
    for new_p in replacements:
        out = replace_prior(out, new_p["name"], new_p)
    return out


def remove_prior(priors, name):
    """Return a copy of priors with the entry for `name` removed."""
    return [p for p in copy.deepcopy(priors) if p["name"] != name]


# ======================================================================
# Robustness specifications
# ======================================================================

def make_rub1_priors():
    """RUB1: Wider structural priors on key parameters.

    Halve the concentration (kappa) on the four parameters with the
    largest prior-to-posterior movements, keeping modes unchanged.

    Beta(a,b) with mode m and kappa:  a = m*(kappa-2)+1, b = (1-m)*(kappa-2)+1
    Gamma(a,r) with mode m and std s: a = (m/s)^2 + 1, r = (a-1)/m
    """
    return replace_priors(BASELINE_PRIORS, [
        # lambda1x: mode=0.60, kappa 10 -> 5 (much wider)
        # Beta: a = 0.60*(5-2)+1 = 2.80, b = 0.40*(5-2)+1 = 2.20
        {"name": "lambda1x", "prior": "beta",
         "args": {"concentration1": 2.80, "concentration2": 2.20}},
        # gamma2: mode=1.50, std 1.00 -> 2.00 (double variance)
        # Gamma: a = (1.50/2.00)^2 + 1 = 1.5625, r = 0.5625/1.50 = 0.375
        {"name": "gamma2", "prior": "gamma",
         "args": {"concentration": 1.5625, "rate": 0.3750}},
        # alpha_rr: mode=0.07, kappa 10 -> 5 (wider)
        # Beta: a = 0.07*(5-2)+1 = 1.21, b = 0.93*(5-2)+1 = 3.79
        {"name": "alpha_rr", "prior": "beta",
         "args": {"concentration1": 1.21, "concentration2": 3.79}},
        # lambda2x: mode=0.16, std 0.10 -> 0.20
        # Gamma: a = (0.16/0.20)^2 + 1 = 1.64, r = 0.64/0.16 = 4.0
        {"name": "lambda2x", "prior": "gamma",
         "args": {"concentration": 1.6400, "rate": 4.0000}},
    ])


def make_rub3_priors():
    """RUB3: Weaker Taylor rule prior.

    Shift gamma_2 mode from 1.50 to 1.00 and widen the prior
    (std 1.00 -> 1.50). This tests whether the aggressive posterior
    gamma_2 ~ 3.0 is robust to a prior that puts more mass on weak
    inflation responses.

    Gamma: a = (1.00/1.50)^2 + 1 = 1.4444, r = 0.4444/1.00 = 0.4444
    """
    return replace_priors(BASELINE_PRIORS, [
        {"name": "gamma2", "prior": "gamma",
         "args": {"concentration": 1.4444, "rate": 0.4444}},
    ])



def make_rub4_priors():
    """RUB4: Pure random walk PI_TREND (alpha_pi fixed at zero).

    Remove alpha_pi from estimation entirely.  PI_TREND is identified
    solely through OBS_DA_LCPI_TAR = PI_TREND (meas_noise=0.3) and
    the three inflation measurement equations.

    This tests whether the near-random-walk posterior (alpha_pi ~ 0.03)
    has material impact on the key structural parameter estimates.
    """
    return remove_prior(BASELINE_PRIORS, "alpha_pi")


def make_rub5_priors():
    """RUB5: Focus IPCA as second PI_TREND measure.

    All priors identical to baseline (alpha_pi estimated).  The extra
    information comes from the 16th observable OBS_FOCUS_IPCA = PI_TREND.
    """
    return copy.deepcopy(BASELINE_PRIORS)


# ======================================================================
# Data loading (from run_brazil_wp_v2.py)
# ======================================================================
def load_brazil_data():
    df = pd.read_csv(DATA_CSV, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp()
    df['OBS_RR_BAR_US'] = df['RR_BAR_US']
    obs_names = [
        'OBS_LGDP', 'OBS_DA_LCPI', 'OBS_DA_LCPIX', 'OBS_DA_LCPI_ENERGY',
        'OBS_DA_LCPI_US', 'OBS_DA_LS', 'OBS_LRWOIL_GAP', 'OBS_LRWFOOD_GAP',
        'OBS_GDP_GAP_US', 'OBS_PREM', 'OBS_RS', 'OBS_RS_US', 'OBS_TT',
        'OBS_DA_LCPI_TAR', 'OBS_RR_BAR_US',
    ]
    ys = df[obs_names].values.astype(np.float64)
    dates = df.index
    return ys, dates, obs_names


def load_brazil_data_focus():
    """Load data with Focus IPCA 12M as 16th observable."""
    df = pd.read_csv(DATA_CSV, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp()
    df['OBS_RR_BAR_US'] = df['RR_BAR_US']
    df['OBS_FOCUS_IPCA'] = df['FOCUS_IPCA_12M']
    obs_names = [
        'OBS_LGDP', 'OBS_DA_LCPI', 'OBS_DA_LCPIX', 'OBS_DA_LCPI_ENERGY',
        'OBS_DA_LCPI_US', 'OBS_DA_LS', 'OBS_LRWOIL_GAP', 'OBS_LRWFOOD_GAP',
        'OBS_GDP_GAP_US', 'OBS_PREM', 'OBS_RS', 'OBS_RS_US', 'OBS_TT',
        'OBS_DA_LCPI_TAR', 'OBS_RR_BAR_US', 'OBS_FOCUS_IPCA',
    ]
    ys = df[obs_names].values.astype(np.float64)
    dates = df.index
    return ys, dates, obs_names


MEAS_NOISE_STD_FOCUS = {
    **MEAS_NOISE_STD,
    'OBS_FOCUS_IPCA': 0.6,
}

OBS_TREND_MAP_FOCUS = {
    **OBS_TREND_MAP,
    "OBS_FOCUS_IPCA": "PI_TREND",
}


def compute_initial_conditions(ys, obs_names, ss_params):
    """Data-based initial state for trend variables.

    Identical to run_brazil_wp_v2.py: uses {"mean", "std"} format for
    diffuse initialization, UIP-consistent RR_BAR, and data-derived
    G_TREND.
    """
    idx = {n: i for i, n in enumerate(obs_names)}
    gdp_col = ys[:, idx['OBS_LGDP']]
    valid_gdp = gdp_col[~np.isnan(gdp_col)]
    avg_growth = float(np.mean(np.diff(valid_gdp)))

    return {
        "LGDP_BAR":    {"mean": float(ys[0, idx['OBS_LGDP']]),           "std": 2.0},
        "G_TREND":     {"mean": avg_growth,                               "std": 0.2},
        "LZ_BAR":      {"mean": 0.0,                                     "std": 5.0},
        "DLZ_TREND":   {"mean": ss_params['d_lz_ss'],                    "std": 0.5},
        "PREM_BAR":    {"mean": float(ys[0, idx['OBS_PREM']]),           "std": 1.0},
        "RR_BAR":      {"mean": float(ys[0, idx['OBS_RR_BAR_US']]) +
                                float(ys[0, idx['OBS_PREM']]) +
                                4 * ss_params['d_lz_ss'],               "std": 1.5},
        "PI_TREND":    {"mean": float(ys[0, idx['OBS_DA_LCPI_TAR']]),    "std": 0.5},
        "RR_BAR_US":   {"mean": float(ys[0, idx['OBS_RR_BAR_US']]),       "std": 0.5},
        "TT_BAR":      {"mean": float(ys[0, idx['OBS_TT']]),             "std": 2.0},
        "PI_TREND_US": {"mean": ss_params['pi_us_ss'],                   "std": 0.1},
    }


# ======================================================================
# Specification registry
# ======================================================================
SPECS = {
    'rub1': {
        'label': 'Wider structural priors',
        'priors_fn': make_rub1_priors,
        'fp_config': FIXED_POINT_CONFIG,
        'fp_prior_alpha': 5.0,
    },
    'rub3': {
        'label': 'Weaker Taylor rule prior (mode=1.00)',
        'priors_fn': make_rub3_priors,
        'fp_config': FIXED_POINT_CONFIG,
        'fp_prior_alpha': 5.0,
    },
    'rub4': {
        'label': 'Pure random walk PI_TREND (alpha_pi=0)',
        'priors_fn': make_rub4_priors,
        'fixed_params': {**FIXED_PARAMS, 'alpha_pi': 0.0},
        'fp_config': FIXED_POINT_CONFIG,
        'fp_prior_alpha': 5.0,
    },
    'rub5': {
        'label': 'Focus IPCA as second PI_TREND measure (16 obs)',
        'priors_fn': make_rub5_priors,
        'fp_config': FIXED_POINT_CONFIG,
        'fp_prior_alpha': 5.0,
        'mod_file': MOD_FILE_FOCUS,
        'load_data_fn': load_brazil_data_focus,
        'meas_noise_std': MEAS_NOISE_STD_FOCUS,
        'obs_trend_map': OBS_TREND_MAP_FOCUS,
    },
}


# ======================================================================
# Key parameters to report
# ======================================================================
KEY_PARAMS = [
    'lambda1x', 'lambda2x', 'gamma1', 'gamma2', 'gamma4',
    'beta1', 'beta3', 'alpha_rr', 'alpha_pi', 'alpha_dlz',
    'rho_ls', 'kappa_z',
]


def summarize_posterior(m, spec_name, label):
    """Print a summary table of key posterior parameters."""
    samples = m._posterior_samples
    if samples is None:
        print("  [WARNING] No posterior samples for {}".format(spec_name))
        return None

    print("\n" + "=" * 70)
    print("  POSTERIOR SUMMARY: {} ({})".format(spec_name.upper(), label))
    print("=" * 70)
    print("{:<16s}  {:>8s}  {:>8s}  {:>10s}  {:>10s}".format(
        "Parameter", "Mean", "Std", "HDI_lo", "HDI_hi"))
    print("-" * 58)

    results = {}
    for name in KEY_PARAMS:
        if name not in samples:
            continue
        draws = np.array(samples[name])
        mn = float(np.mean(draws))
        sd = float(np.std(draws))
        lo = float(np.percentile(draws, 16))
        hi = float(np.percentile(draws, 84))
        print("{:<16s}  {:8.4f}  {:8.4f}  {:10.4f}  {:10.4f}".format(
            name, mn, sd, lo, hi))
        results[name] = {'mean': mn, 'std': sd, 'hdi_lo': lo, 'hdi_hi': hi}

    # Sacrifice ratio (if available)
    try:
        sr_result = m.sacrifice_ratio(
            gap_variable='GDP_GAP', inflation_variable='DA_LCPIX',
            target_variable='PI_TREND',
            reduction_pp=1.0, n_draws=200,
            horizon_years=30, plot=False)
        if sr_result is not None and hasattr(sr_result, 'sr_draws') and sr_result.sr_draws is not None:
            sr_arr = np.array(sr_result.sr_draws)
            print("\n  Sacrifice ratio: median={:.3f}, 68% HDI=[{:.3f}, {:.3f}]".format(
                float(np.median(sr_arr)),
                float(np.percentile(sr_arr, 16)),
                float(np.percentile(sr_arr, 84))))
    except Exception as e:
        print("\n  [Sacrifice ratio computation skipped: {}]".format(e))

    return results


def reanalyze_spec(spec_name, spec_config, use_diffuse=True):
    """Reload saved draws and re-run posterior summary + sacrifice ratio.

    No MCMC, no calibration — just model setup, load_draws(), and analysis.
    """
    label = spec_config['label']
    mod_file = spec_config.get('mod_file', MOD_FILE)
    load_fn = spec_config.get('load_data_fn', load_brazil_data)
    meas_noise = spec_config.get('meas_noise_std', MEAS_NOISE_STD)

    results_dir = os.path.join(SCRIPT_DIR, _results_base(use_diffuse), spec_name)

    draws_path = os.path.join(results_dir, "estimation", "draws.npz")
    if not os.path.isfile(draws_path):
        print("  [SKIP] No draws found for {} at {}".format(spec_name, draws_path))
        return None

    filter_label = "DIFFUSE" if use_diffuse else "DARE"
    print("\n" + "#" * 70)
    print("#  REANALYZE: {} -- {} [{}]".format(
        spec_name.upper(), label, filter_label))
    print("#" * 70)

    # Minimal model setup (needed for sacrifice_ratio / pass_through)
    ys, dates, obs_names = load_fn()
    m = QPMModel(mod_file, results_dir=results_dir)

    ss_params = m.model.default_param_assignments
    init_config = compute_initial_conditions(ys, obs_names, ss_params)

    m.set_data(ys, dates=dates, obs_names=obs_names)
    # v5+ TT-trend wiring (per-spec, idempotent on shared CYCLE_MAP/TREND_SNR_MAP).
    import copy as _copy
    import _tt_trend_config as _tt
    _priors = spec_config['priors_fn']()
    _fixed_params = dict(spec_config.get('fixed_params', FIXED_PARAMS))
    fp_config = _copy.deepcopy(spec_config['fp_config'])
    _tt.configure_tt_trend(
        mod_file=mod_file,
        structural_priors=_priors,
        cycle_map=CYCLE_MAP,
        trend_snr_map=TREND_SNR_MAP,
        fixed_point_config=fp_config,
        fixed_params=_fixed_params,
        apply_opt1=(_V5SC.V5_TT_MODE == 'tt_trend'),
    )
    m.set_priors(
        structural_priors=_priors,
        fixed_params=_fixed_params,
        meas_noise_std=meas_noise,
        cycle_map=CYCLE_MAP,
        trend_snr_map=TREND_SNR_MAP,
        manual_cycle_priors=MANUAL_CYCLE_PRIORS,
    )
    m.set_initial_conditions(init_state_config=init_config)

    # Calibrate (needed for sacrifice_ratio — sigma priors must be resolved)
    fp_prior_alpha = spec_config.get('fp_prior_alpha', 5.0)
    m.calibrate(method='both', fixed_point_config=fp_config,
                fp_prior_alpha=fp_prior_alpha,
                cycle_persistence=0.85, drift_persistence=0.90)
    _V5SC.apply_post_calibration_pins(m)
    if _V5SC.V5_TT_MODE == 'tt_trend':
        for _p in m._priors:
            if _p['name'] == 'sigma_SHK_TT_GAP':
                _a, _r = _p['args']['concentration'], _p['args']['rate']
                _mode = _r / (_a + 1)
                _p['args']['concentration'] = 20.0
                _p['args']['rate'] = 21.0 * _mode
                print(f"  v5+ opt1: sigma_SHK_TT_GAP α {_a:.1f} → 20.0, "
                      f"mode preserved at {_mode:.4f}")
                break

    # Load existing draws (no MCMC)
    m.load_draws(draws_path)
    print("  Loaded {} posterior draws from {}".format(
        list(m._posterior_samples.values())[0].shape[0], draws_path))

    # Summary + sacrifice ratio
    results = summarize_posterior(m, spec_name, label)
    return results


def save_results(all_results, outdir):
    """Save a per-spec CSV of key parameters (safe for subprocess use)."""
    if not all_results:
        return
    for spec_name, res in all_results.items():
        if res is None:
            continue
        rows = []
        for pname, vals in res.items():
            rows.append({
                'spec': spec_name,
                'parameter': pname,
                'mean': vals['mean'],
                'std': vals['std'],
                'hdi_lo': vals['hdi_lo'],
                'hdi_hi': vals['hdi_hi'],
            })
        if rows:
            df = pd.DataFrame(rows)
            # Per-spec file (subprocess-safe, no overwrite race)
            spec_dir = os.path.join(outdir, spec_name)
            os.makedirs(spec_dir, exist_ok=True)
            fpath = os.path.join(spec_dir, "posterior_summary.csv")
            df.to_csv(fpath, index=False)
            print("\n  Spec summary saved to: {}".format(fpath))


def merge_comparison_csv(specs, outdir):
    """Merge per-spec posterior_summary.csv files into one comparison table."""
    frames = []
    for spec_name in specs:
        fpath = os.path.join(outdir, spec_name, "posterior_summary.csv")
        if os.path.exists(fpath):
            frames.append(pd.read_csv(fpath))
    if frames:
        df = pd.concat(frames, ignore_index=True)
        fpath = os.path.join(outdir, "robustness_comparison.csv")
        df.to_csv(fpath, index=False)
        print("\n  Comparison table saved to: {}".format(fpath))


# ======================================================================
# Results directory logic
# ======================================================================
def _results_base(use_diffuse):
    """Return 'results_v2_diffuse' or 'results_v2' depending on filter."""
    return "results_v2_diffuse" if use_diffuse else "results_v2"


# ======================================================================
# Main
# ======================================================================
def run_spec(spec_name, spec_config, calibrate_only=False,
             warmup=WARMUP, samples=SAMPLES, chains=CHAINS,
             target_accept=0.90, use_diffuse=True):
    """Run a single robustness specification."""
    import copy as _copy
    import _tt_trend_config as _tt
    label = spec_config['label']
    priors = spec_config['priors_fn']()
    fp_config = _copy.deepcopy(spec_config['fp_config'])
    fp_prior_alpha = spec_config.get('fp_prior_alpha', 5.0)

    # Per-spec overrides (defaults = baseline 15-obs configuration)
    mod_file = spec_config.get('mod_file', MOD_FILE)
    load_fn = spec_config.get('load_data_fn', load_brazil_data)
    meas_noise = spec_config.get('meas_noise_std', MEAS_NOISE_STD)
    obs_trend = spec_config.get('obs_trend_map', OBS_TREND_MAP)
    fixed_params = dict(spec_config.get('fixed_params', FIXED_PARAMS))

    # v5+ TT-trend wiring (per-spec; helper is idempotent on shared maps).
    _tt.configure_tt_trend(
        mod_file=mod_file,
        structural_priors=priors,
        cycle_map=CYCLE_MAP,
        trend_snr_map=TREND_SNR_MAP,
        fixed_point_config=fp_config,
        fixed_params=fixed_params,
        apply_opt1=(_V5SC.V5_TT_MODE == 'tt_trend'),
    )

    # v5+ path override (set by run_all_models_v5plus.sh per spec) routes
    # results to .../<spec>_v5plus/ without conflicting with the baseline tree.
    _override = os.environ.get("V5PLUS_RESULTS_OVERRIDE", "").strip()
    if _override:
        results_dir = (_override if os.path.isabs(_override)
                       else os.path.join(SCRIPT_DIR, _override))
    else:
        results_dir = os.path.join(SCRIPT_DIR, _results_base(use_diffuse), spec_name)
    os.makedirs(results_dir, exist_ok=True)

    filter_label = "DIFFUSE" if use_diffuse else "DARE"
    print("\n" + "#" * 70)
    print("#  ROBUSTNESS: {} -- {} [{}]".format(
        spec_name.upper(), label, filter_label))
    print("#" * 70)

    # Load data
    ys, dates, obs_names = load_fn()
    print("\n  Data: {} quarters, {} observables".format(
        ys.shape[0], ys.shape[1]))

    # Create model
    m = QPMModel(mod_file, results_dir=results_dir)

    # Initial conditions
    ss_params = m.model.default_param_assignments
    init_config = compute_initial_conditions(ys, obs_names, ss_params)

    # Configure
    m.set_data(ys, dates=dates, obs_names=obs_names)
    m.set_priors(
        structural_priors=priors,
        fixed_params=fixed_params,
        meas_noise_std=meas_noise,
        cycle_map=CYCLE_MAP,
        trend_snr_map=TREND_SNR_MAP,
        manual_cycle_priors=MANUAL_CYCLE_PRIORS,
    )
    m.set_initial_conditions(init_state_config=init_config)

    # Check solution
    m.solution_check()

    # Calibrate (SNR + fixed-point)
    m.calibrate(method='both', fixed_point_config=fp_config,
                fp_prior_alpha=fp_prior_alpha,
                cycle_persistence=0.85, drift_persistence=0.90)

    # v5 sweep: post-calibration α=20 prior pins on TT/PREM shocks.
    _V5SC.apply_post_calibration_pins(m)

    # v5+ opt1: α=20 pin on sigma_SHK_TT_GAP (independent of V5_TT_PIN).
    if _V5SC.V5_TT_MODE == 'tt_trend':
        for _p in m._priors:
            if _p['name'] == 'sigma_SHK_TT_GAP':
                _a, _r = _p['args']['concentration'], _p['args']['rate']
                _mode = _r / (_a + 1)
                _p['args']['concentration'] = 20.0
                _p['args']['rate'] = 21.0 * _mode
                print(f"  v5+ opt1: sigma_SHK_TT_GAP α {_a:.1f} → 20.0, "
                      f"mode preserved at {_mode:.4f}")
                break

    # Filter diagnostic
    m.filter_diagnostic(obs_trend_map=obs_trend)

    if calibrate_only:
        print("\n  --calibrate-only: stopping before NUTS.")
        return None

    # Diffuse filter kwargs
    diffuse_kwargs = {}
    if use_diffuse:
        diffuse_kwargs = dict(use_diffuse=True, lyap_iters=8, n_diffuse=10)
        print("  [Exact diffuse initialization enabled]")

    # Estimate
    print("\n  NUTS: warmup={}, samples={}, chains={}, target_accept={}".format(
        warmup, samples, chains, target_accept))
    m.estimate(
        warmup=warmup,
        samples=samples,
        chains=chains,
        chain_method='parallel',
        target_accept=target_accept,
        seed=42,
        trace_groups={
            'is_curve': ['beta1', 'beta2', 'beta3', 'beta4'],
            'phillips': ['lambda1x', 'lambda2x', 'kappa_z'],
            'taylor':   ['gamma1', 'gamma2', 'gamma4'],
            'fx':       ['rho_ls', 'psi_uip', 'rho_prem_gap'],
        },
        **diffuse_kwargs,
    )

    # Post-estimation filter diagnostic
    m.filter_diagnostic(obs_trend_map=obs_trend)

    # Summary
    results = summarize_posterior(m, spec_name, label)
    return results


def _ensure_arviz_cache():
    """Pre-create ArviZ daily-warning stamp to avoid race on parallel import."""
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "arviz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = cache_dir / "daily_warning"
    if not stamp.exists():
        from datetime import date
        stamp.write_text(date.today().isoformat())


def _run_subprocess(spec_name, python_exe, extra_args=None):
    """Launch a single spec as a subprocess. Returns (spec, returncode, elapsed)."""
    import time as _time
    if extra_args is None:
        extra_args = []
    # Determine results base from extra_args
    use_diffuse = '--no-diffuse' not in extra_args
    log_dir = os.path.join(SCRIPT_DIR, _results_base(use_diffuse), spec_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")

    cmd = [python_exe, os.path.abspath(__file__), "--spec", spec_name] + extra_args
    t0 = _time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=SCRIPT_DIR)
    elapsed = _time.time() - t0
    return spec_name, proc.returncode, elapsed


def main():
    import time as _time

    parser = argparse.ArgumentParser(
        description="Prior sensitivity analysis for Brazil QPM v2 "
                    "(real-level UIP, exact diffuse filter)")
    parser.add_argument(
        "--spec", type=str, default="all",
        help="Which spec(s) to run: a single name, comma-separated list "
             "(e.g. rub4,rub5), or 'all' (default: all)")
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Max parallel subprocesses when --spec all (default: 4)")
    parser.add_argument(
        "--warmup", type=int, default=WARMUP,
        help="NUTS warmup iterations (default: {})".format(WARMUP))
    parser.add_argument(
        "--samples", type=int, default=SAMPLES,
        help="NUTS posterior samples per chain (default: {})".format(SAMPLES))
    parser.add_argument(
        "--chains", type=int, default=CHAINS,
        help="Number of MCMC chains (default: {})".format(CHAINS))
    parser.add_argument(
        "--target-accept", type=float, default=0.90,
        help="NUTS target acceptance probability (default: 0.90)")
    parser.add_argument(
        "--calibrate-only", action="store_true",
        help="Run calibration + filter diagnostic only, skip NUTS")
    parser.add_argument(
        "--no-diffuse", action="store_true",
        help="Use DARE initialization instead of exact diffuse "
             "(results go to results_v2/ instead of results_v2_diffuse/)")
    parser.add_argument(
        "--reanalyze", action="store_true",
        help="Skip MCMC: reload saved draws and re-run posterior summary, "
             "sacrifice ratio, and comparison CSV only")
    parser.add_argument(
        "--python", type=str, default=sys.executable,
        help="Python interpreter to use (default: current)")
    args = parser.parse_args()

    use_diffuse = not args.no_diffuse

    # Parse spec argument: single name, comma-separated list, or "all"
    all_spec_names = list(SPECS.keys())
    if args.spec == "all":
        specs_to_run = all_spec_names
    elif ',' in args.spec:
        specs_to_run = [s.strip() for s in args.spec.split(',')]
        for s in specs_to_run:
            if s not in SPECS:
                parser.error("Unknown spec '{}'. Choose from: {}".format(
                    s, ', '.join(all_spec_names)))
    else:
        # Single spec: run in-process (this is also the subprocess entry point)
        if args.spec not in SPECS:
            parser.error("Unknown spec '{}'. Choose from: {}".format(
                args.spec, ', '.join(all_spec_names)))
        spec_config = SPECS[args.spec]
        if args.reanalyze:
            results = reanalyze_spec(args.spec, spec_config,
                                     use_diffuse=use_diffuse)
        else:
            results = run_spec(args.spec, spec_config,
                               calibrate_only=args.calibrate_only,
                               warmup=args.warmup, samples=args.samples,
                               chains=args.chains,
                               target_accept=args.target_accept,
                               use_diffuse=use_diffuse)
        if results:
            outdir = os.path.join(SCRIPT_DIR, _results_base(use_diffuse))
            save_results({args.spec: results}, outdir)
        return

    # Multiple specs: launch subprocesses sequentially (one after the other)
    _ensure_arviz_cache()

    extra_args = []
    if args.reanalyze:
        extra_args.append("--reanalyze")
    if args.calibrate_only:
        extra_args.append("--calibrate-only")
    if args.no_diffuse:
        extra_args.append("--no-diffuse")
    extra_args.extend(["--warmup", str(args.warmup),
                       "--samples", str(args.samples),
                       "--chains", str(args.chains),
                       "--target-accept", str(args.target_accept)])

    n_workers = args.max_workers or len(specs_to_run)
    rbase = _results_base(use_diffuse)
    filter_label = "DARE" if args.no_diffuse else "DIFFUSE (D&K 2012)"

    print("=" * 70)
    print("  ROBUSTNESS BATCH RUNNER (v2)")
    print("=" * 70)
    print("  Filter:      {}".format(filter_label))
    print("  Results:     {}/".format(rbase))
    print("  Specs:       {}".format(", ".join(specs_to_run)))
    print("  Workers:     {}".format(n_workers))
    print("  Python:      {}".format(args.python))
    print("  NUTS:        warmup={}, samples={}, chains={}".format(
        args.warmup, args.samples, args.chains))
    print("=" * 70)

    t_start = _time.time()
    results = {}

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_subprocess, spec, args.python, extra_args): spec
            for spec in specs_to_run
        }
        for spec in specs_to_run:
            log_path = os.path.join(SCRIPT_DIR, rbase, spec, "run.log")
            print("  [STARTED]  {}  (log: {})".format(spec.upper(), log_path))

        for future in as_completed(futures):
            spec_name, rc, elapsed = future.result()
            status = "OK" if rc == 0 else "FAILED (exit {})".format(rc)
            mins = elapsed / 60
            print("  [{:>8s}]  {}  ({:.1f} min)".format(status, spec_name.upper(), mins))
            results[spec_name] = (rc, elapsed)

    total = _time.time() - t_start
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    for spec in sorted(results):
        rc, elapsed = results[spec]
        status = "OK" if rc == 0 else "FAIL({})".format(rc)
        print("    {}: {:>8s}  ({:.1f} min)".format(spec.upper(), status, elapsed / 60))
    print("\n  Total wall-clock time: {:.1f} min".format(total / 60))

    # Merge per-spec CSVs into one comparison table
    succeeded = [s for s, (rc, _) in results.items() if rc == 0]
    if succeeded:
        outdir = os.path.join(SCRIPT_DIR, rbase)
        merge_comparison_csv(succeeded, outdir)

    n_failed = sum(1 for rc, _ in results.values() if rc != 0)
    if n_failed:
        print("\n  WARNING: {} spec(s) failed. Check logs above.".format(n_failed))
        sys.exit(1)
    else:
        print("\n  All robustness specs completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

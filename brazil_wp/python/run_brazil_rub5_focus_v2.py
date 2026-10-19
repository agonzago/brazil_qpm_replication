"""Brazil QPM -- RUB5: Focus IPCA expectations as second PI_TREND measure.

Robustness specification that adds Focus 12-month IPCA expectations as a
second observable informing PI_TREND identification.  The baseline model
identifies PI_TREND through the inflation target alone (OBS_DA_LCPI_TAR).
This specification adds OBS_FOCUS_IPCA = PI_TREND with larger measurement
noise (0.6 pp vs 0.3 pp for the target), reflecting the noisier nature
of survey-based expectations relative to the official target.

Differences from the baseline (run_brazil_model_a.py):
  - QPM file: brazil_qpm_mr_focus.qpm (16 observables, extra measurement eq)
  - 16th observable: OBS_FOCUS_IPCA (Focus 12M IPCA expectations)
  - MEAS_NOISE_STD['OBS_FOCUS_IPCA'] = 0.6
  - OBS_TREND_MAP['OBS_FOCUS_IPCA'] = 'PI_TREND'
  - All structural priors identical to baseline (alpha_pi estimated)

Usage:
    python run_brazil_rub5_focus_v2.py                  # full pipeline
    python run_brazil_rub5_focus_v2.py --skip-estimation  # use saved draws
    python run_brazil_rub5_focus_v2.py --calibrate-only   # calibration only
"""

import os
import sys
import argparse

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

# Disable BLAS threading to prevent thread pool competition.
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')

N_PARALLEL = int(os.environ.get('N_PARALLEL', 8))
os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '') +
    f' --xla_force_host_platform_device_count={N_PARALLEL}'
).strip()

import numpyro
numpyro.set_host_device_count(N_PARALLEL)

import numpy as np
import pandas as pd

# Path setup: qpm_toolbox lives in gpm_estimation/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX_DIR = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX_DIR)

from qpm_toolbox import QPMModel, ForecastPlan

# ======================================================================
# Paths
# ======================================================================
# v5 sweep config — needed for V5_TT_MODE='tt_trend' branch.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v5_sweep_config as _V5SC
_V5SC.banner()
if _V5SC.V5_TT_MODE == 'tt_trend':
    MOD_FILE = os.path.join(SCRIPT_DIR, "model",
                            "brazil_qpm_mr_focus_tt_trend_v2.qpm")
else:
    MOD_FILE = os.path.join(SCRIPT_DIR, "model", "brazil_qpm_mr_focus_v2.qpm")
_LOCAL_CSV = os.path.join(SCRIPT_DIR, "data", "estimation_data.csv")
_TOOLBOX_CSV = os.path.join(TOOLBOX_DIR, "data", "brazil", "estimation_data.csv")
DATA_CSV = _LOCAL_CSV if os.path.exists(_LOCAL_CSV) else _TOOLBOX_CSV
_V5PLUS_OVERRIDE = os.environ.get("V5PLUS_RESULTS_OVERRIDE", "").strip()
if _V5PLUS_OVERRIDE:
    RESULTS_DIR = (_V5PLUS_OVERRIDE if os.path.isabs(_V5PLUS_OVERRIDE)
                   else os.path.join(SCRIPT_DIR, _V5PLUS_OVERRIDE))
else:
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_v2", "rub5")

# ======================================================================
# SNR calibration maps
# ======================================================================
CYCLE_MAP = {
    # OBS_LGDP cycle SNR 0.30->0.20 (consistent across all v2 runners).
    'OBS_LGDP':           ('SHK_GDP_GAP',         0.20),
    'OBS_DA_LCPI':        ('SHK_DA_LCPI',         0.30),
    'OBS_DA_LCPIX':       ('SHK_DA_LCPIX',        0.30),
    'OBS_DA_LCPI_ENERGY': ('SHK_DA_LCPI_ENERGY',  0.50),
    'OBS_DA_LCPI_US':     ('SHK_DA_LCPI_US',      0.30),
    'OBS_DA_LS':          ('SHK_LZ_GAP',              0.25),
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

# ======================================================================
# Structural priors — identical to baseline (alpha_pi estimated)
# ======================================================================
STRUCTURAL_PRIORS = [
    # ---- IS Curve ----
    {"name": "beta1",     "prior": "beta",  "args": {"concentration1": 7.80, "concentration2": 4.20}},
    {"name": "beta2",     "prior": "beta",  "args": {"concentration1": 1.70, "concentration2": 7.30}},
    {"name": "beta3",     "prior": "beta",  "args": {"concentration1": 2.28, "concentration2": 7.72}},
    {"name": "beta4",     "prior": "beta",  "args": {"concentration1": 2.60, "concentration2": 7.40}},
    {"name": "beta5",     "prior": "gamma", "args": {"concentration": 2.8604, "rate": 169.1271}},
    {"name": "phi_tt_is", "prior": "gamma", "args": {"concentration": 3.2553, "rate": 45.1062}},

    # ---- Phillips Curve (Core) ----
    {"name": "lambda1x",  "prior": "beta",  "args": {"concentration1": 5.80, "concentration2": 4.20}},
    {"name": "lambda2x",  "prior": "gamma", "args": {"concentration": 4.3290, "rate": 20.8062}},
    {"name": "kappa_z",   "prior": "gamma", "args": {"concentration": 3.2553, "rate": 90.2124}},
    {"name": "lambda4x",  "prior": "halfnormal", "args": {"scale": 0.03}},
    {"name": "lambda5x",  "prior": "gamma", "args": {"concentration": 2.1816, "rate": 295.4066}},
    {"name": "lambda6x",  "prior": "gamma", "args": {"concentration": 2.6180, "rate": 202.2542}},

    # ---- Component Inflations ----
    {"name": "lambda_E1", "prior": "beta",  "args": {"concentration1": 11.20, "concentration2": 2.80}},
    {"name": "theta_E",   "prior": "gamma", "args": {"concentration": 3.1194, "rate": 70.6476}},
    {"name": "lambda_F1", "prior": "beta",  "args": {"concentration1": 6.20, "concentration2": 3.80}},
    {"name": "theta_F",   "prior": "gamma", "args": {"concentration": 3.4914, "rate": 62.2839}},

    # ---- Taylor Rule ----
    {"name": "gamma1",    "prior": "beta",  "args": {"concentration1": 8.50, "concentration2": 3.50}},
    {"name": "gamma2",    "prior": "gamma", "args": {"concentration": 4.0000, "rate": 2.0000}},
    {"name": "gamma4",    "prior": "gamma", "args": {"concentration": 3.5302, "rate": 12.5260}},

    # ---- UIP / Exchange Rate ----
    {"name": "rho_ls",    "prior": "beta",  "args": {"concentration1": 3.10, "concentration2": 4.90}},
    {"name": "psi_uip",   "prior": "gamma", "args": {"concentration": 3.4914, "rate": 2.4914}},

    # ---- Risk Premium ----
    {"name": "rho_prem_gap", "prior": "beta",  "args": {"concentration1": 5.80, "concentration2": 4.20}},
    {"name": "phi_tt_gap",   "prior": "gamma", "args": {"concentration": 3.4914, "rate": 24.9136}},

    # ---- Koyck Filters ----
    {"name": "rho_QZ",    "prior": "beta",  "args": {"concentration1": 11.80, "concentration2": 2.20}},
    {"name": "rho_QRR",   "prior": "beta",  "args": {"concentration1": 9.50, "concentration2": 2.50}},
    {"name": "rho_QZ_E",  "prior": "beta",  "args": {"concentration1": 9.80, "concentration2": 2.20}},
    {"name": "rho_QZ_F",  "prior": "beta",  "args": {"concentration1": 9.80, "concentration2": 2.20}},

    # ---- Persistence ----
    {"name": "rho_tt",           "prior": "beta", "args": {"concentration1": 7.00, "concentration2": 3.00}},
    {"name": "rho_lrwoil_gap",   "prior": "beta", "args": {"concentration1": 8.57, "concentration2": 3.43}},
    {"name": "rho_lrwfood_gap",  "prior": "beta", "args": {"concentration1": 8.54, "concentration2": 3.46}},
    {"name": "rho_resn_gdp_gap", "prior": "beta", "args": {"concentration1": 3.00, "concentration2": 3.00}},
    {"name": "rho_res_rs",       "prior": "beta", "args": {"concentration1": 3.00, "concentration2": 3.00}},

    # ---- US Block ----
    {"name": "rho_US",      "prior": "beta",  "args": {"concentration1": 11.56, "concentration2": 2.44}},
    {"name": "beta1_US",    "prior": "beta",  "args": {"concentration1": 5.79, "concentration2": 4.21}},
    {"name": "beta2_US",    "prior": "gamma", "args": {"concentration": 3.8112, "rate": 19.5223}},
    {"name": "beta3_us",    "prior": "gamma", "args": {"concentration": 3.4914, "rate": 12.4568}},
    {"name": "rho_pie_us",  "prior": "beta",  "args": {"concentration1": 3.70, "concentration2": 4.30}},
    {"name": "gamma2_us",   "prior": "gamma", "args": {"concentration": 4.0000, "rate": 2.0000}},
    {"name": "gamma4_us",   "prior": "gamma", "args": {"concentration": 4.0000, "rate": 20.0000}},

    # ---- Phillips Curve: ToT channel ----
    {"name": "phi_tt_pc",   "prior": "halfnormal", "args": {"scale": 0.03}},

    # ---- Koyck Filter: ToT ----
    {"name": "rho_QTT",     "prior": "beta",       "args": {"concentration1": 3.00, "concentration2": 3.00}},

    # ---- Trend Convergence Speeds ----
    {"name": "alpha_g",      "prior": "beta", "args": {"concentration1": 1.80, "concentration2": 8.20}},
    {"name": "alpha_dlz",    "prior": "beta", "args": {"concentration1": 1.80, "concentration2": 8.20}},
    {"name": "alpha_prem",   "prior": "beta", "args": {"concentration1": 1.80, "concentration2": 8.20}},
    {"name": "alpha_rr",     "prior": "beta", "args": {"concentration1": 1.80, "concentration2": 8.20}},
    # alpha_pi: inflation trend -> pi_tar_ss. Estimated (unlike rub4).
    {"name": "alpha_pi",     "prior": "beta", "args": {"concentration1": 2.20, "concentration2": 7.80}},
    {"name": "alpha_rr_us",  "prior": "beta", "args": {"concentration1": 1.56, "concentration2": 8.44}},
    {"name": "alpha_pi_us",  "prior": "beta", "args": {"concentration1": 1.56, "concentration2": 8.44}},
]

# CPI basket weights (data-given, not estimated)
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
    # Focus expectations: noisier than official target
    'OBS_FOCUS_IPCA': 0.6,
}

# v5+ TT-trend variant wiring (delegated to _tt_trend_config).
import _tt_trend_config as _tt
_TT_TREND_VARIANT = _tt.configure_tt_trend(
    mod_file=MOD_FILE,
    structural_priors=STRUCTURAL_PRIORS,
    cycle_map=CYCLE_MAP,
    trend_snr_map=TREND_SNR_MAP,
    fixed_point_config=FIXED_POINT_CONFIG,
    fixed_params=FIXED_PARAMS,
    apply_opt1=(_V5SC.V5_TT_MODE == 'tt_trend'),
)
_ALT_TT_GAP_PIN = os.environ.get('ALT_TT_GAP_PIN', '').strip().lower()
if _V5SC.V5_TT_MODE == 'tt_trend' and not _ALT_TT_GAP_PIN:
    _ALT_TT_GAP_PIN = 'tight'

OBS_TREND_MAP = {
    "OBS_LGDP": "LGDP_BAR", "OBS_DA_LCPI": "PI_TREND",
    "OBS_DA_LCPIX": "PI_TREND", "OBS_DA_LCPI_ENERGY": "PI_TREND",
    "OBS_DA_LCPI_US": "PI_TREND_US", "OBS_DA_LS": None,
    "OBS_LRWOIL_GAP": None, "OBS_LRWFOOD_GAP": None,
    "OBS_GDP_GAP_US": None, "OBS_PREM": "PREM_BAR",
    "OBS_RS": "RS_TREND", "OBS_RS_US": "RS_US_TREND",
    "OBS_TT": "TT_BAR", "OBS_DA_LCPI_TAR": "PI_TREND",
    "OBS_RR_BAR_US": "RR_BAR_US",
    "OBS_FOCUS_IPCA": "PI_TREND",
}

SHOCK_GROUPS = {
    'Demand':         ['SHK_GDP_GAP'],
    'Cost-push':      ['SHK_DA_LCPIX', 'SHK_DA_LCPI_ENERGY',
                       'SHK_DA_LCPI_FOOD', 'SHK_DA_LCPI'],
    'Monetary':       ['SHK_RS'],
    'FX & premium':   ['SHK_LZ_GAP', 'SHK_PREM_GAP'],
    'Terms of trade': ['SHK_TT_GAP'],
    'Foreign':        ['SHK_GAP_US', 'SHK_DA_LCPI_US', 'SHK_RS_US'],
    'Commodities':    ['SHK_LRWOIL_GAP', 'SHK_LRWFOOD_GAP'],
    'Trends':         ['SHK_LGDP_BAR', 'SHK_G_TREND', 'SHK_LZ_BAR',
                       'SHK_DLZ_TREND', 'SHK_PREM_BAR', 'SHK_PI_TREND',
                       'SHK_RR_BAR', 'SHK_RR_BAR_US', 'SHK_TT_BAR',
                       'SHK_PI_TREND_US'],
}


# ======================================================================
# Data loading
# ======================================================================
def load_brazil_data():
    """Load and prepare the Brazil estimation dataset with Focus IPCA."""
    df = pd.read_csv(DATA_CSV, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp()

    # Map raw columns to OBS_ names
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


def compute_initial_conditions(ys, obs_names, ss_params):
    """Data-based initial state for trend variables."""
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
# Main pipeline
# ======================================================================
def main():
    import jax
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(
        description="Brazil QPM -- RUB5: Focus IPCA as second PI_TREND measure")
    parser.add_argument("--skip-estimation", action="store_true",
                        help="Skip MCMC, load saved draws")
    parser.add_argument("--only-policy", action="store_true",
                        help="Only run policy analysis (SR, PT, RMSE)")
    parser.add_argument("--skip-forecast", action="store_true",
                        help="Skip unconditional forecast and forecast evaluation")
    parser.add_argument("--skip-conditional", action="store_true",
                        help="Skip conditional forecast scenarios")
    parser.add_argument("--warmup", type=int, default=400,
                        help="NUTS warmup iterations (default: 400)")
    parser.add_argument("--samples", type=int, default=500,
                        help="NUTS posterior samples per chain (default: 500)")
    parser.add_argument("--chains", type=int, default=8,
                        help="Number of MCMC chains (default: 8)")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Run calibration + filter diagnostic + decomposition only, skip NUTS")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data and create model
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  BRAZIL QPM -- RUB5: FOCUS IPCA + TARGET (two PI_TREND measures)")
    print("=" * 70)

    ys, dates, obs_names = load_brazil_data()
    print(f"\n  Data: {ys.shape[0]} quarters, {ys.shape[1]} observables")
    q0 = pd.Timestamp(dates[0]).to_period('Q')
    qT = pd.Timestamp(dates[-1]).to_period('Q')
    print(f"  Sample: {q0} to {qT}")

    m = QPMModel(MOD_FILE, results_dir=RESULTS_DIR)

    # 2. Initial conditions
    ss_params = m.model.default_param_assignments
    init_config = compute_initial_conditions(ys, obs_names, ss_params)

    # 3. Configure
    m.set_data(ys, dates=dates, obs_names=obs_names)
    m.set_priors(
        structural_priors=STRUCTURAL_PRIORS,
        fixed_params=FIXED_PARAMS,
        meas_noise_std=MEAS_NOISE_STD,
        cycle_map=CYCLE_MAP,
        trend_snr_map=TREND_SNR_MAP,
        manual_cycle_priors=MANUAL_CYCLE_PRIORS,
    )
    m.set_initial_conditions(init_state_config=init_config)

    # 4. Check solution at prior modes
    m.solution_check()

    if not args.only_policy:
        # ----------------------------------------------------------
        # 5. Calibrate (SNR -> fixed-point)
        # ----------------------------------------------------------
        if not args.skip_estimation or args.calibrate_only:
            m.calibrate(method='both', fixed_point_config=FIXED_POINT_CONFIG,
                        cycle_persistence=0.85, drift_persistence=0.90)

        # v5 sweep: apply α=20 prior pins on TT and/or PREM shocks based
        # on V5_TT_PIN / V5_PREM_PIN env vars.  See _v5_sweep_config.py.
        _V5SC.apply_post_calibration_pins(m)

        # v5+ TT-trend opt1 auto-pin (independent of V5_TT_PIN).
        if _ALT_TT_GAP_PIN == 'tight':
            for prior in m._priors:
                if prior['name'] == 'sigma_SHK_TT_GAP':
                    a_old = prior['args']['concentration']
                    r_old = prior['args']['rate']
                    mode = r_old / (a_old + 1)
                    prior['args']['concentration'] = 20.0
                    prior['args']['rate'] = 21.0 * mode
                    print(f"  ALT_TT_GAP_PIN=tight: sigma_SHK_TT_GAP "
                          f"α {a_old:.1f} → 20.0, mode preserved at {mode:.4f}")
                    break

        # ----------------------------------------------------------
        # 6. Filter diagnostic (pre-estimation)
        # ----------------------------------------------------------
        m.filter_diagnostic(obs_trend_map=OBS_TREND_MAP)

        # --calibrate-only: produce decomposition at calibrated params, then stop
        if args.calibrate_only:
            print("\n" + "=" * 70)
            print("  HISTORICAL SHOCK DECOMPOSITION (calibrated parameters)")
            print("=" * 70)
            m.decomposition(
                variables=['GDP_GAP', 'DA_LCPIX', 'DA_LCPI', 'RS',
                            'LZ_GAP', 'PREM_GAP', 'DA_LGDP'],
                shock_groups=SHOCK_GROUPS,
            )
            sys.exit(0)

        # ----------------------------------------------------------
        # 7. Bayesian estimation
        # ----------------------------------------------------------
        if not args.skip_estimation:
            print("\n" + "=" * 70)
            print("  BAYESIAN ESTIMATION (NUTS)")
            print("=" * 70)

            m.estimate(
                warmup=args.warmup,
                samples=args.samples,
                chains=args.chains,
                chain_method='parallel',
                target_accept=0.90,
                seed=42,
                trace_groups={
                    'is_curve': ['beta1', 'beta2', 'beta3', 'beta4'],
                    'phillips': ['lambda1x', 'lambda2x', 'kappa_z'],
                    'taylor':   ['gamma1', 'gamma2', 'gamma4'],
                    'fx':       ['rho_ls', 'psi_uip', 'rho_prem_gap'],
                },
            )
        else:
            print("\n  Skipping estimation, loading saved draws...")

        # ----------------------------------------------------------
        # 8. Post-estimation diagnostics
        # ----------------------------------------------------------
        m.filter_diagnostic(obs_trend_map=OBS_TREND_MAP)

        # ----------------------------------------------------------
        # 9. IRFs
        # ----------------------------------------------------------
        print("\n" + "=" * 70)
        print("  IMPULSE RESPONSE FUNCTIONS")
        print("=" * 70)

        m.irf(shocks=['SHK_RS'],
              variables=['GDP_GAP', 'DA_LCPIX', 'DA_LCPI', 'RS',
                         'RR_GAP', 'LZ_GAP', 'DA_LS'],
              horizon=40, n_draws=200)

        m.irf(shocks=['SHK_GDP_GAP'],
              variables=['GDP_GAP', 'DA_LCPIX', 'DA_LCPI', 'RS',
                         'RR_GAP'],
              horizon=40, n_draws=200)

        m.irf(shocks=['SHK_LZ_GAP'],
              variables=['DA_LS', 'LZ_GAP', 'DA_LCPIX', 'DA_LCPI',
                         'GDP_GAP', 'RS'],
              horizon=40, n_draws=200)

        m.irf(shocks=['SHK_PREM_GAP'],
              variables=['PREM_GAP', 'DA_LS', 'LZ_GAP', 'DA_LCPI',
                         'GDP_GAP', 'RS'],
              horizon=40, n_draws=200)

        m.irf(shocks=['SHK_DA_LCPIX'],
              variables=['DA_LCPIX', 'DA_LCPI', 'GDP_GAP', 'RS',
                         'RR_GAP', 'LZ_GAP'],
              horizon=40, n_draws=200)

        m.irf(shocks=['SHK_RS_US'],
              variables=['RS_US', 'GDP_GAP_US', 'DA_LS', 'LZ_GAP',
                         'GDP_GAP', 'DA_LCPIX', 'RS'],
              horizon=40, n_draws=200)

        m.irf(shocks=['SHK_TT_GAP'],
              variables=['TT_GAP', 'GDP_GAP', 'DA_LCPIX', 'DA_LCPI',
                         'PREM_GAP', 'RS', 'LZ_GAP'],
              horizon=40, n_draws=200)

        if not args.skip_forecast:
            # ----------------------------------------------------------
            # 10. Forecast
            # ----------------------------------------------------------
            print("\n" + "=" * 70)
            print("  UNCONDITIONAL FORECAST")
            print("=" * 70)

            m.forecast(
                horizon=20,
                stochastic=True,
                n_draws=200,
                use_dk=False,
                variables=['OBS_LGDP', 'OBS_DA_LCPI', 'OBS_DA_LCPIX',
                            'OBS_RS', 'OBS_DA_LS', 'OBS_PREM'],
            )

        # ----------------------------------------------------------
        # 11. Historical shock decomposition
        # ----------------------------------------------------------
        print("\n" + "=" * 70)
        print("  HISTORICAL SHOCK DECOMPOSITION")
        print("=" * 70)

        m.decomposition(
            variables=['GDP_GAP', 'DA_LCPIX', 'DA_LCPI', 'RS',
                        'LZ_GAP', 'PREM_GAP', 'DA_LGDP'],
            shock_groups=SHOCK_GROUPS,
        )

    # ==================================================================
    # POLICY ANALYSIS
    # ==================================================================
    print("\n" + "=" * 70)
    print("  POLICY ANALYSIS")
    print("=" * 70)

    print("\n" + "-" * 50)
    print("  12. Sacrifice Ratio (posterior)")
    print("-" * 50)

    sr = m.sacrifice_ratio(
        gap_variable='GDP_GAP',
        inflation_variable='DA_LCPIX',
        target_variable='PI_TREND',
        reduction_pp=1.0,
        horizon_years=30,
        n_draws=200,
    )

    print("\n" + "-" * 50)
    print("  13. Exchange Rate Pass-Through (posterior)")
    print("-" * 50)

    pt = m.pass_through(
        shocks=['SHK_RS', 'SHK_LZ_GAP', 'SHK_RS_US'],
        inflation_variables=['DA_LCPI', 'DA_LCPIX',
                             'DA_LCPI_ENERGY', 'DA_LCPI_FOOD'],
        er_variable='DA_LS',
        horizon=12,
        normalize=True,
        n_draws=200,
    )

    if not args.skip_forecast:
        print("\n" + "-" * 50)
        print("  14. Out-of-Sample Forecast Evaluation")
        print("-" * 50)

        ev = m.forecast_evaluation(
            variables=['OBS_DA_LCPI', 'OBS_LGDP', 'OBS_RS', 'OBS_DA_LCPIX'],
            first_origin='2017-01-01',
            last_origin='2022-10-01',
            horizons=[1, 2, 3, 4, 8],
            max_horizon=8,
            stochastic=False,
        )

        print("\n" + "-" * 50)
        print("  14b. Conditional Out-of-Sample Forecast Evaluation")
        print("-" * 50)

        CONDITION_ON = [
            ('obs', 'OBS_LRWOIL_GAP',     'SHK_LRWOIL_GAP'),
            ('obs', 'OBS_LRWFOOD_GAP',    'SHK_LRWFOOD_GAP'),
            ('obs', 'OBS_RS_US',          'SHK_RS_US'),
            ('obs', 'OBS_GDP_GAP_US',     'SHK_GAP_US'),
            ('obs', 'OBS_DA_LCPI_ENERGY', 'SHK_DA_LCPI_ENERGY'),
            ('obs', 'OBS_DA_LCPI_US',     'SHK_DA_LCPI_US'),
        ]

        ev_cond = m.conditional_forecast_evaluation(
            variables=['OBS_DA_LCPI', 'OBS_LGDP', 'OBS_RS', 'OBS_DA_LCPIX'],
            condition_on=CONDITION_ON,
            first_origin='2017-01-01',
            last_origin='2022-10-01',
            horizons=[1, 2, 3, 4, 8],
            max_horizon=8,
        )

    if not args.only_policy and not args.skip_conditional:
        # ----------------------------------------------------------
        # 15. Conditional forecast scenarios
        # ----------------------------------------------------------
        print("\n" + "=" * 70)
        print("  CONDITIONAL FORECAST SCENARIOS")
        print("=" * 70)

        last_rs = float(ys[-1, obs_names.index('OBS_RS')])

        plan1 = ForecastPlan(horizon=16)
        plan1.add_condition('obs', 'OBS_RS', 'SHK_RS',
                            periods=range(4), values=[last_rs] * 4)
        m.conditional_forecast(plan1, label='rate_hold',
                               variables=['OBS_RS', 'OBS_DA_LCPI',
                                          'OBS_DA_LCPIX', 'OBS_LGDP'])

        pi_target = ss_params['pi_tar_ss']
        rate_path = [last_rs, last_rs, last_rs - 0.75, last_rs - 1.50]
        plan2 = ForecastPlan(horizon=16)
        plan2.add_condition('obs', 'OBS_RS', 'SHK_RS',
                            periods=range(4), values=rate_path)
        plan2.add_condition('obs', 'OBS_DA_LCPI', 'SHK_DA_LCPIX',
                            periods=range(4, 12),
                            values=[pi_target] * 8)
        m.conditional_forecast(plan2, label='easing_target',
                               variables=['OBS_RS', 'OBS_DA_LCPI',
                                          'OBS_DA_LCPIX', 'OBS_DA_LS'])

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if hasattr(sr, 'sr_draws') and sr.sr_draws is not None:
        sr_arr = np.array(sr.sr_draws)
        print(f"  Sacrifice ratio (posterior):")
        print(f"    Median: {np.median(sr_arr):.4f}  "
              f"Mean: {np.mean(sr_arr):.4f}  "
              f"Std: {np.std(sr_arr):.4f}")
        print(f"    HDI 68%: [{np.percentile(sr_arr, 16):.4f}, "
              f"{np.percentile(sr_arr, 84):.4f}]")
        print(f"    HDI 90%: [{np.percentile(sr_arr, 5):.4f}, "
              f"{np.percentile(sr_arr, 95):.4f}]")
    else:
        print(f"  Sacrifice ratio:   {sr.sacrifice_ratio:.4f}")
        print(f"  Cumulative gap:    {sr.cum_output_gap:.2f} pp-quarters")

    if hasattr(pt, 'pt_draws') and pt.pt_draws is not None:
        print(f"\n  Pass-through (posterior, cumulative 12Q):")
        for shk_name in ['SHK_RS', 'SHK_LZ_GAP']:
            draws = np.array(pt.pt_draws[shk_name]['DA_LCPI']['cumulative'])
            print(f"    {shk_name:15s} -> DA_LCPI:  median={np.median(draws):.3f}  "
                  f"HDI68=[{np.percentile(draws, 16):.3f}, "
                  f"{np.percentile(draws, 84):.3f}]")
    else:
        print(f"\n  Pass-through (domestic MP, headline, 12Q cumulative): "
              f"{pt.table['SHK_RS']['DA_LCPI']['cumulative']:.3f}")
        print(f"  Pass-through (ER shock, headline, 12Q cumulative):    "
              f"{pt.table['SHK_LZ_GAP']['DA_LCPI']['cumulative']:.3f}")

    print(f"\n  Results saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

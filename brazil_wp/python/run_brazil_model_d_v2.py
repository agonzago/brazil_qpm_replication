"""Model D: No QRR/QZ_E/QZ_F filters + simplified Taylor rule (1Q-ahead headline inflation).

This is a variant of the baseline Brazil QPM. Model D removes the three
auxiliary Koyck filters (QRR, QZ_E, QZ_F) and simplifies the Taylor rule
to respond to 1Q-ahead headline inflation rather than the 4Q-ahead
filtered measure. The estimated parameter set is the same as Model C
(the Taylor rule simplification changes how gamma2 enters the equation
but does not remove any estimated parameters).

This script produces all quantitative results for Model D:
  - Bayesian estimation (NUTS via NumPyro)
  - IRFs (monetary policy, demand, cost-push)
  - Sacrifice ratio (cost of disinflation)
  - Exchange rate pass-through (by shock source)
  - Historical shock decomposition
  - Out-of-sample forecast evaluation (rolling-window RMSE)
  - Conditional forecast scenarios

Usage:
    python run_brazil_model_d_v2.py                  # full pipeline
    python run_brazil_model_d_v2.py --skip-estimation  # use saved draws
    python run_brazil_model_d_v2.py --skip-estimation --only-policy  # policy analysis only
"""

import os
import sys
import argparse

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

# Disable BLAS threading to prevent thread pool competition.
# Without this, N chains x N_cores OpenBLAS threads = massive thrashing.
# Let XLA handle parallelism via virtual devices instead.
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')

# Number of parallel chains (= XLA virtual CPU devices).
# Match to chain count; 4-8 is the sweet spot on most machines.
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

# v5 sweep config — env-var driven
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v5_sweep_config as _V5SC
_V5SC.banner()

# ======================================================================
# Paths
# ======================================================================
if _V5SC.V5_TT_MODE == 'tworegime':
    MOD_FILE = os.path.join(SCRIPT_DIR, "model", "brazil_qpm_model_d_tt_anchor2_v2.qpm")
elif _V5SC.V5_TT_MODE == 'tt_trend':
    MOD_FILE = os.path.join(SCRIPT_DIR, "model", "brazil_qpm_model_d_tt_trend_v2.qpm")
else:
    MOD_FILE = os.path.join(SCRIPT_DIR, "model", "brazil_qpm_model_d_v2.qpm")
# Prefer local copy (produced by prepare_data.py); fall back to toolbox copy
_LOCAL_CSV = os.path.join(SCRIPT_DIR, "data", "estimation_data.csv")
_TOOLBOX_CSV = os.path.join(TOOLBOX_DIR, "data", "brazil", "estimation_data.csv")
DATA_CSV = _LOCAL_CSV if os.path.exists(_LOCAL_CSV) else _TOOLBOX_CSV
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_v2", "model_d")

# ======================================================================
# SNR calibration maps
# ======================================================================
# CYCLE_MAP: {observable: (cycle_shock, SNR_target)}
# SNR = sigma_cycle / sigma_obs ~ fraction of observable variance
# explained by the cyclical shock.
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

# TREND_SNR_MAP: {trend_shock: (obs_anchor, idiosyncratic_name, SNR, type)}
TREND_SNR_MAP = {
    # LGDP_BAR/G_TREND loosened (consistent with Models A and F).
    'SHK_LGDP_BAR':   ('OBS_LGDP',      'output_idio_trend',  50,  'level'),
    'SHK_G_TREND':     ('OBS_LGDP',      'output_idio_trend', 200,  'growth'),
    'SHK_LZ_BAR':      ('OBS_DA_LS',     'reer_idio_trend',   200,  'level'),
    'SHK_DLZ_TREND':   ('OBS_DA_LS',     'reer_idio_trend',   400,  'growth'),
    # v5 sweep: PREM SNR env-driven (default 30).
    'SHK_PREM_BAR':    ('OBS_PREM',      'prem_idio_trend', _V5SC.V5_PREM_SNR, 'level'),
    'SHK_PI_TREND':    ('OBS_DA_LCPI',   'pi_idio_trend',     2400, 'level'),
    'SHK_RR_BAR':      ('OBS_RS',        'rate_idio_trend',   1200, 'level'),
    # RR_BAR_US loosened from 200 to 50 to track the observed HLW r*
    # series closely (consistent across all v2 model runners).
    'SHK_RR_BAR_US':   ('OBS_RR_BAR_US', 'rate_idio_trend',    50,  'level'),
    # v5 sweep: TT_SNR env-driven (50 tight / 100 loose).
    'SHK_TT_BAR':      ('OBS_TT',        'tt_idio_trend',     _V5SC.TT_SNR, 'level'),
    'SHK_PI_TREND_US': ('OBS_DA_LCPI_US','pi_idio_trend',    24000, 'level'),
}

FIXED_POINT_CONFIG = {
    'max_iter': 50,
    'tol': 1e-4,
    'damping': 0.5,
    'sigma_bounds': dict(
        [('sigma_SHK_G_TREND', (1e-8, 0.40)),
         ('sigma_SHK_RR_BAR',  (1e-8, 0.50))]
        + ([('sigma_SHK_TT_GAP', _V5SC.TT_CAP)] if _V5SC.TT_CAP else [])
    ),
}

# ======================================================================
# Structural priors (Model D: same parameter set as Model C)
#
# Removed relative to baseline:
#   - rho_QRR  (QRR Koyck filter removed)
#   - rho_QZ_E (QZ_E Koyck filter removed)
#   - rho_QZ_F (QZ_F Koyck filter removed)
#
# The Taylor rule simplification (1Q-ahead headline inflation) does not
# remove any estimated parameters -- it changes how gamma2 enters the
# equation but gamma2 itself is still estimated.
#
# Prior motivation: all prior modes are set to the calibrated parameter
# values from the paper (Section 3.2, Table tab:full_params).
#
# Beta(a,b) priors: mode = (a-1)/(a+b-2), centered on calibration.
#   Concentration kappa = a+b controls tightness:
#     kappa ~ 6-8:   loose (std ~ 0.16-0.19), for weakly identified params
#     kappa ~ 10-12: moderate (std ~ 0.12-0.15), for literature-based params
#     kappa ~ 14-20: tight (std ~ 0.08-0.10), for well-established params
#
# HalfNormal(sigma) priors: mode = 0, sigma set so calibration value
#   falls at a sensible quantile (typically 40th-60th percentile).
#
# Key cross-validation targets (NOT used to set priors, only to verify
# that the prior predictive distribution is plausible):
#   - Sacrifice ratio:  BVAR estimates 0.91-1.17 at 12Q (Section 4.2)
#   - ERPT peak lag:    3-7 quarters (Belaisch 2003, Section 4.3)
#   - Taylor principle: gamma2 > 1 in most prior draws
#
# References:
#   [1] Minella et al. (2003) -- IT-era expectations anchoring in Brazil
#   [2] Gordon & Pessoa (2012) -- post-IT sacrifice ratio decline
#   [3] Belaisch (2003) -- incomplete exchange rate pass-through
#   [4] Arquete & Jayme (2015) -- meta-analysis of BCB Taylor rules
#   [5] Cespedes et al. (2005) -- cross-country SR benchmarks
# ======================================================================
STRUCTURAL_PRIORS = [
    # ---- IS Curve ----
    # beta1: output gap persistence; "significant inertia in aggregate
    #   demand" (Section 3.2). Calibration = 0.68.
    {"name": "beta1",     "prior": "beta",  "args": {"concentration1": 7.80, "concentration2": 4.20}},
    # beta2: forward-looking demand expectations. Calibration = 0.10.
    {"name": "beta2",     "prior": "beta",  "args": {"concentration1": 1.70, "concentration2": 7.30}},
    # beta3: real interest rate gap semi-elasticity via Koyck filter QRR;
    #   "moderately effective interest rate channel" (Section 3.2).
    #   Calibration = 0.16.
    {"name": "beta3",     "prior": "beta",  "args": {"concentration1": 2.28, "concentration2": 7.72}},
    # beta4: REER gap effect on demand; open-economy IS specification.
    #   Calibration = 0.20.
    {"name": "beta4",     "prior": "beta",  "args": {"concentration1": 2.60, "concentration2": 7.40}},
    # beta5: US output gap spillover; small foreign demand linkage.
    #   Calibration = 0.011. Gamma: mode=0.011, std=0.010.
    {"name": "beta5",     "prior": "gamma", "args": {"concentration": 2.8604, "rate": 169.1271}},
    # phi_tt_is: ToT demand channel; motivated by r=0.71 GDP-commodity
    #   correlation (Section 2.1, Fact 1). Calibration = 0.05.
    #   Gamma: mode=0.05, std=0.04.
    {"name": "phi_tt_is", "prior": "gamma", "args": {"concentration": 3.2553, "rate": 45.1062}},

    # ---- Phillips Curve (Core) ----
    # lambda1x: forward-looking weight in PC; "increased importance of
    #   expectations in IT era" -- Minella et al. (2003) [1].
    #   Calibration = 0.60. Key determinant of sacrifice ratio.
    {"name": "lambda1x",  "prior": "beta",  "args": {"concentration1": 5.80, "concentration2": 4.20}},
    # lambda2x: output gap slope; jointly calibrated for SR < 1.0
    #   -- Gordon & Pessoa (2012) [2]. Calibration = 0.16.
    #   Gamma: mode=0.16, std=0.10; allows SR in [0.3, 1.5].
    {"name": "lambda2x",  "prior": "gamma", "args": {"concentration": 4.3290, "rate": 20.8062}},
    # kappa_z: filtered REER pass-through to core inflation;
    #   "gradual and incomplete ER pass-through" -- Belaisch (2003) [3].
    #   Calibration = 0.025. Gamma: mode=0.025, std=0.020.
    {"name": "kappa_z",   "prior": "gamma", "args": {"concentration": 3.2553, "rate": 90.2124}},
    # lambda4x: REER gap level effect; zero in calibration, allow data
    #   to detect if nonzero. HN(0.03) -- mode at zero appropriate here.
    {"name": "lambda4x",  "prior": "halfnormal", "args": {"scale": 0.03}},
    # lambda5x: oil cost pass-through to core. Calibration = 0.004.
    #   Gamma: mode=0.004, std=0.005.
    {"name": "lambda5x",  "prior": "gamma", "args": {"concentration": 2.1816, "rate": 295.4066}},
    # lambda6x: food cost pass-through to core. Calibration = 0.008.
    #   Gamma: mode=0.008, std=0.008.
    {"name": "lambda6x",  "prior": "gamma", "args": {"concentration": 2.6180, "rate": 202.2542}},

    # ---- Component Inflations ----
    # lambda_E1: energy inflation persistence; high due to administered
    #   pricing regime (Petrobras fuel, ANEEL tariffs). Calibration = 0.85.
    {"name": "lambda_E1", "prior": "beta",  "args": {"concentration1": 11.20, "concentration2": 2.80}},
    # theta_E: filtered ER pass-through to energy; delayed by
    #   administered pricing. Calibration = 0.03.
    #   Gamma: mode=0.03, std=0.025.
    {"name": "theta_E",   "prior": "gamma", "args": {"concentration": 3.1194, "rate": 70.6476}},
    # lambda_F1: food inflation persistence; more market-determined
    #   than energy. Calibration = 0.65.
    {"name": "lambda_F1", "prior": "beta",  "args": {"concentration1": 6.20, "concentration2": 3.80}},
    # theta_F: filtered ER pass-through to food; faster than energy.
    #   Calibration = 0.04. Gamma: mode=0.04, std=0.03.
    {"name": "theta_F",   "prior": "gamma", "args": {"concentration": 3.4914, "rate": 62.2839}},

    # ---- Taylor Rule ----
    # gamma1: interest rate smoothing; "high inertia" consistent with
    #   Arquete & Jayme (2015) [4] meta-analysis of BCB behavior.
    #   Calibration = 0.75.
    {"name": "gamma1",    "prior": "beta",  "args": {"concentration1": 8.50, "concentration2": 3.50}},
    # gamma2: inflation gap response; satisfies Taylor principle (>1)
    #   -- Minella et al. (2003) [1]. Calibration = 1.50.
    #   Gamma: mode=1.50, std=1.00. P(gamma2>1) ~ 72%.
    {"name": "gamma2",    "prior": "gamma", "args": {"concentration": 4.0000, "rate": 2.0000}},
    # gamma4: output gap stabilization. Calibration = 0.202.
    #   Gamma: mode=0.202, std=0.15.
    {"name": "gamma4",    "prior": "gamma", "args": {"concentration": 3.5302, "rate": 12.5260}},

    # ---- UIP / Exchange Rate ----
    # rho_ls: FX partial adjustment speed; Vavra-style UIP with gradual
    #   ER reaction. Calibration = 0.35. Loose prior (kappa=8).
    {"name": "rho_ls",    "prior": "beta",  "args": {"concentration1": 3.10, "concentration2": 4.90}},
    # psi_uip: UIP semi-elasticity; sensitivity to interest differential.
    #   Calibration = 1.00. Gamma: mode=1.00, std=0.75.
    {"name": "psi_uip",   "prior": "gamma", "args": {"concentration": 3.4914, "rate": 2.4914}},

    # ---- Risk Premium ----
    # rho_prem_gap: premium gap persistence; CDS half-life dynamics
    #   documented in Section 2.2. Calibration = 0.60.
    {"name": "rho_prem_gap", "prior": "beta",  "args": {"concentration1": 5.80, "concentration2": 4.20}},
    # phi_tt_gap: premium sensitivity to ToT gap; commodity exporter
    #   risk channel. Calibration = 0.10. Gamma: mode=0.10, std=0.075.
    {"name": "phi_tt_gap",   "prior": "gamma", "args": {"concentration": 3.4914, "rate": 24.9136}},

    # ---- Koyck Filters ----
    # rho_QZ: ER-to-core PC smoothing; "gradual transmission" consistent
    #   with Belaisch (2003) [3] ERPT peak at 3-7Q. Calibration = 0.90.
    {"name": "rho_QZ",    "prior": "beta",  "args": {"concentration1": 11.80, "concentration2": 2.20}},
    # NOTE: rho_QRR, rho_QZ_E, rho_QZ_F removed in Model D
    #   (auxiliary Koyck filters eliminated)

    # ---- Persistence ----
    # rho_tt: ToT gap persistence; commodity super-cycles.
    #   Calibration = 0.75.
    {"name": "rho_tt",           "prior": "beta", "args": {"concentration1": 7.00, "concentration2": 3.00}},
    # rho_lrwoil_gap: world oil price gap persistence.
    #   Calibration = 0.757.
    {"name": "rho_lrwoil_gap",   "prior": "beta", "args": {"concentration1": 8.57, "concentration2": 3.43}},
    # rho_lrwfood_gap: world food price gap persistence.
    #   Calibration = 0.754.
    {"name": "rho_lrwfood_gap",  "prior": "beta", "args": {"concentration1": 8.54, "concentration2": 3.46}},
    # rho_resn_gdp_gap: persistent demand shock state; weakly informative
    #   symmetric prior. Calibration = 0.50.
    {"name": "rho_resn_gdp_gap", "prior": "beta", "args": {"concentration1": 3.00, "concentration2": 3.00}},
    # rho_res_rs: persistent monetary policy shock state; weakly
    #   informative symmetric prior. Calibration = 0.50.
    {"name": "rho_res_rs",       "prior": "beta", "args": {"concentration1": 3.00, "concentration2": 3.00}},

    # ---- US Block ----
    # rho_US: US Taylor rule smoothing; well-established Fed behavior.
    #   Calibration = 0.88.
    {"name": "rho_US",      "prior": "beta",  "args": {"concentration1": 11.56, "concentration2": 2.44}},
    # beta1_US: US IS curve lag persistence. Calibration = 0.60.
    {"name": "beta1_US",    "prior": "beta",  "args": {"concentration1": 5.79, "concentration2": 4.21}},
    # beta2_US: US IS real rate effect. Calibration = 0.144.
    #   Gamma: mode=0.144, std=0.10.
    {"name": "beta2_US",    "prior": "gamma", "args": {"concentration": 3.8112, "rate": 19.5223}},
    # beta3_us: US PC output gap slope. Calibration = 0.20.
    #   Gamma: mode=0.20, std=0.15.
    {"name": "beta3_us",    "prior": "gamma", "args": {"concentration": 3.4914, "rate": 12.4568}},
    # rho_pie_us: US inflation persistence; moderate anchoring under
    #   Fed IT framework. Calibration = 0.45.
    {"name": "rho_pie_us",  "prior": "beta",  "args": {"concentration1": 3.70, "concentration2": 4.30}},
    # gamma2_us: US Taylor inflation response; Taylor principle.
    #   Calibration = 1.50. Gamma: mode=1.50, std=1.00.
    {"name": "gamma2_us",   "prior": "gamma", "args": {"concentration": 4.0000, "rate": 2.0000}},
    # gamma4_us: US Taylor output gap response; dual mandate.
    #   Calibration = 0.15. Gamma: mode=0.15, std=0.10.
    {"name": "gamma4_us",   "prior": "gamma", "args": {"concentration": 4.0000, "rate": 20.0000}},

    # ---- Phillips Curve: ToT channel (previously fixed at zero) ----
    # phi_tt_pc: ToT in Phillips curve; commodity exporter cost channel.
    #   Calibration = 0.00 (restricted). HN: mode at zero tests whether
    #   this channel exists. Data determines if nonzero.
    {"name": "phi_tt_pc",   "prior": "halfnormal", "args": {"scale": 0.03}},

    # ---- Koyck Filter: ToT (previously fixed at zero) ----
    # rho_QTT: ToT Koyck filter smoothing; weakly informative symmetric
    #   prior lets data determine degree of ToT transmission lag.
    #   Calibration = 0.00.
    {"name": "rho_QTT",     "prior": "beta",       "args": {"concentration1": 3.00, "concentration2": 3.00}},

    # ---- Trend Convergence Speeds (alpha parameters) ----
    # alpha = 0 -> pure random walk (nested specification)
    # alpha = 1 -> immediate reversion to steady state
    # Half-life (quarters) = log(2)/log(1/(1-alpha))
    #
    # alpha_g: potential growth -> g_ss. Calibration = 0.10 (~7Q half-life).
    {"name": "alpha_g",      "prior": "beta", "args": {"concentration1": 1.80, "concentration2": 8.20}},
    # alpha_dlz: REER trend depreciation -> d_lz_ss. Calibration = 0.10.
    {"name": "alpha_dlz",    "prior": "beta", "args": {"concentration1": 1.80, "concentration2": 8.20}},
    # alpha_prem: risk premium trend -> prem_ss. Calibration = 0.10.
    {"name": "alpha_prem",   "prior": "beta", "args": {"concentration1": 1.80, "concentration2": 8.20}},
    # alpha_rr: domestic neutral rate -> UIP equilibrium. Calibration = 0.10.
    #   Error-correction speed to (RR_BAR_US + PREM_BAR + 4*DLZ_TREND).
    {"name": "alpha_rr",     "prior": "beta", "args": {"concentration1": 1.80, "concentration2": 8.20}},
    # alpha_pi: inflation trend -> pi_tar_ss. Calibration = 0.15
    #   (~4.5Q half-life). Faster convergence = higher CB credibility.
    {"name": "alpha_pi",     "prior": "beta", "args": {"concentration1": 2.20, "concentration2": 7.80}},
    # alpha_rr_us: US neutral rate -> rr_us_ss. Calibration = 0.07.
    {"name": "alpha_rr_us",  "prior": "beta", "args": {"concentration1": 1.56, "concentration2": 8.44}},
    # alpha_pi_us: US inflation trend -> pi_us_ss. Calibration = 0.07.
    {"name": "alpha_pi_us",  "prior": "beta", "args": {"concentration1": 1.56, "concentration2": 8.44}},
]

# CPI basket weights (data-given, not estimated)
FIXED_PARAMS = {
    'lambda1s': 0.1581,    # energy weight in headline CPI
    'lambda2s': 0.1142,    # food weight in headline CPI
    # sigma_SHK_TT_BAR moved to TREND_SNR_MAP in v5 (2026-05-10).
}
# Model D-specific deviation from the v5 anchor convention:
# Models A/B/C/E pin rho_QTT=0 under V5_TT_MODE=tworegime because TT_BAR is
# data-anchored by the two auxiliary observables, leaving the Koyck filter
# weakly identified.  Model D however drops three Koyck filters (QRR, QZ_E,
# QZ_F), so QTT is its only remaining smoothing mechanism.  Pinning rho_QTT
# to 0 *and* anchoring TT_BAR pushes Model D's stationary block to the edge
# of the unit circle (76.8% of posterior draws had spectral radius > 0.99
# in the 2026-05-12 run, with chains 2 and 3 stuck at non-mixing modes).
# Under V5_TT_MODE=tworegime we therefore leave rho_QTT FREE for Model D
# to preserve the Koyck damping that the rest of the model lacks.

MEAS_NOISE_STD = {
    'OBS_LGDP': 0.0, 'OBS_DA_LCPI': 0.0, 'OBS_DA_LCPIX': 0.0,
    'OBS_DA_LCPI_ENERGY': 0.0, 'OBS_DA_LCPI_US': 0.0, 'OBS_DA_LS': 0.0,
    'OBS_LRWOIL_GAP': 2.0, 'OBS_LRWFOOD_GAP': 2.0,
    'OBS_GDP_GAP_US': 0.5, 'OBS_PREM': 0.5,
    'OBS_RS': 0.0, 'OBS_RS_US': 0.0, 'OBS_TT': 0.0,
    'OBS_DA_LCPI_TAR': 0.3,
    'OBS_RR_BAR_US': 0.5,
}
if _V5SC.V5_TT_MODE == 'tworegime':
    MEAS_NOISE_STD['OBS_TT_BAR']      = _V5SC.V5_TT_BAR_PRE_STD
    MEAS_NOISE_STD['OBS_TT_BAR_POST'] = _V5SC.V5_TT_BAR_POST_STD

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

# Maps observables to their trend state for diagnostic plots
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

# Shock groupings for historical decomposition
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
    """Load and prepare the Brazil estimation dataset.

    The data is produced by prepare_data.py, which maps the R pipeline
    output to QPM observable names (OBS_* columns).
    """
    df = pd.read_csv(DATA_CSV, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp()

    # Map raw RR_BAR_US (LWH r* estimate) to OBS_RR_BAR_US
    df['OBS_RR_BAR_US'] = df['RR_BAR_US']

    obs_names = [
        'OBS_LGDP', 'OBS_DA_LCPI', 'OBS_DA_LCPIX', 'OBS_DA_LCPI_ENERGY',
        'OBS_DA_LCPI_US', 'OBS_DA_LS', 'OBS_LRWOIL_GAP', 'OBS_LRWFOOD_GAP',
        'OBS_GDP_GAP_US', 'OBS_PREM', 'OBS_RS', 'OBS_RS_US', 'OBS_TT',
        'OBS_DA_LCPI_TAR', 'OBS_RR_BAR_US',
    ]
    if _V5SC.V5_TT_MODE == 'tworegime':
        missing = [c for c in ('TT_BAR_PRE_HP', 'TT_BAR_POST_EXTRAP')
                   if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"V5_TT_MODE=tworegime requires columns {missing}")
        df['OBS_TT_BAR']      = df['TT_BAR_PRE_HP']
        df['OBS_TT_BAR_POST'] = df['TT_BAR_POST_EXTRAP']
        obs_names = obs_names + ['OBS_TT_BAR', 'OBS_TT_BAR_POST']
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
        "PI_TREND":    {"mean": float(ys[0, idx['OBS_DA_LCPI_TAR']]),    "std": 0.5},
        "RR_BAR":      {"mean": float(ys[0, idx['OBS_RR_BAR_US']]) +
                                float(ys[0, idx['OBS_PREM']]) +
                                4 * ss_params['d_lz_ss'],               "std": 1.5},
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

    parser = argparse.ArgumentParser(description="Brazil QPM Model D")
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
    parser.add_argument("--use-diffuse", action="store_true",
                        help="Use exact diffuse Kalman filter (D&K 2012). "
                             "Results go to results_v2_diffuse/ instead of results_v2/.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override output directory. Relative paths are resolved "
                             "against the script directory. Takes precedence over "
                             "--use-diffuse's default redirect.")
    args = parser.parse_args()

    # Resolve results directory: explicit override > diffuse default > standard.
    if args.results_dir:
        results_dir = (args.results_dir if os.path.isabs(args.results_dir)
                       else os.path.join(SCRIPT_DIR, args.results_dir))
    elif args.use_diffuse:
        results_dir = os.path.join(SCRIPT_DIR, "results_v2_diffuse", "model_d")
    else:
        results_dir = RESULTS_DIR

    # ------------------------------------------------------------------
    # 1. Load data and create model
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  BRAZIL QPM -- MODEL D (Simplified, No Auxiliary Filters)")
    print("=" * 70)

    ys, dates, obs_names = load_brazil_data()
    print(f"\n  Data: {ys.shape[0]} quarters, {ys.shape[1]} observables")
    q0 = pd.Timestamp(dates[0]).to_period('Q')
    qT = pd.Timestamp(dates[-1]).to_period('Q')
    print(f"  Sample: {q0} to {qT}")

    m = QPMModel(MOD_FILE, results_dir=results_dir)

    # 2. Initial conditions
    ss_params = m.model.default_param_assignments
    init_config = compute_initial_conditions(ys, obs_names, ss_params)

    # 3. Configure
    m.set_data(ys, dates=dates, obs_names=obs_names)
    # Drop priors whose param is in FIXED_PARAMS (no-op when V5_TT_MODE!=tworegime)
    _structural_priors = [p for p in STRUCTURAL_PRIORS
                          if p['name'] not in FIXED_PARAMS]
    _dropped = sorted({p['name'] for p in STRUCTURAL_PRIORS}
                      - {p['name'] for p in _structural_priors})
    if _dropped:
        print(f"  Dropping priors (pinned via FIXED_PARAMS): {_dropped}")
    m.set_priors(
        structural_priors=_structural_priors,
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
        # Calibration always runs (sets m._priors, consumed below and
        # needed under --skip-estimation to regenerate the forecast tables).
        m.calibrate(method='both', fixed_point_config=FIXED_POINT_CONFIG,
                    cycle_persistence=0.85, drift_persistence=0.90)

        # v5 sweep prior pins (TT and/or PREM)
        _V5SC.apply_post_calibration_pins(m)

        # v5+ opt1 auto-pin (independent of V5_TT_PIN).
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

            diffuse_kwargs = {}
            if args.use_diffuse:
                diffuse_kwargs = dict(use_diffuse=True, lyap_iters=8, n_diffuse=10)
                print("  [Exact diffuse initialization enabled]")

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
                **diffuse_kwargs,
            )
        else:
            print("\n  Skipping estimation, loading saved draws...")

        # ----------------------------------------------------------
        # 8. Post-estimation diagnostics
        # ----------------------------------------------------------
        m.filter_diagnostic(obs_trend_map=OBS_TREND_MAP)

        # ----------------------------------------------------------
        # 9. IRFs: key structural shocks
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
            # 10. Forecast (stochastic, posterior fan charts)
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
    # POLICY ANALYSIS (posterior: loops over draws for HDI)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  POLICY ANALYSIS")
    print("=" * 70)

    # ----------------------------------------------------------
    # 12. Sacrifice ratio (posterior distribution)
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # 13. Exchange rate pass-through (posterior distribution)
    # ----------------------------------------------------------
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
        # ----------------------------------------------------------
        # 14. Forecast evaluation (rolling-window RMSE)
        # ----------------------------------------------------------
        print("\n" + "-" * 50)
        print("  14. Out-of-Sample Forecast Evaluation")
        print("-" * 50)

        # Realised annualised GDP growth (DA_LGDP_ANN) for scoring the
        # model-transformed reporting variable: 4*diff(observed log GDP).
        import numpy as _np_fe
        _, _ys_fe, _, _obsn_fe = m._data
        _lgdp_fe = _ys_fe[:, _obsn_fe.index('OBS_LGDP')]
        _da_lgdp_act = _np_fe.full(_lgdp_fe.shape[0], _np_fe.nan, dtype=float)
        _da_lgdp_act[1:] = 4.0 * _np_fe.diff(_lgdp_fe)
        _REALISED_DA_LGDP = {'DA_LGDP_ANN': _da_lgdp_act}
        _COVID_WIN = ('2020Q1', '2020Q4')

        ev = m.forecast_evaluation(
            variables=['OBS_DA_LCPI', 'OBS_DA_LCPIX', 'OBS_LGDP',
                       'DA_LGDP_ANN', 'OBS_RS'],
            first_origin='2017-01-01',
            last_origin='2022-10-01',
            horizons=[1, 2, 3, 4, 8],
            max_horizon=8,
            stochastic=False,
            realised_reporting=_REALISED_DA_LGDP,
            covid_window=_COVID_WIN,
        )

        # ----------------------------------------------------------
        # 14b. Conditional forecast evaluation (rolling-window RMSE)
        # ----------------------------------------------------------
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
            variables=['OBS_DA_LCPI', 'OBS_DA_LCPIX', 'OBS_LGDP',
                       'DA_LGDP_ANN', 'OBS_RS'],
            condition_on=CONDITION_ON,
            first_origin='2017-01-01',
            last_origin='2022-10-01',
            horizons=[1, 2, 3, 4, 8],
            max_horizon=8,
            realised_reporting=_REALISED_DA_LGDP,
            covid_window=_COVID_WIN,
        )

    if not args.only_policy and not args.skip_conditional:
        # ----------------------------------------------------------
        # 15. Conditional forecast scenarios
        # ----------------------------------------------------------
        print("\n" + "=" * 70)
        print("  CONDITIONAL FORECAST SCENARIOS")
        print("=" * 70)

        last_rs = float(ys[-1, obs_names.index('OBS_RS')])

        # Scenario 1: Policy rate hold for 4 quarters
        plan1 = ForecastPlan(horizon=16)
        plan1.add_condition('obs', 'OBS_RS', 'SHK_RS',
                            periods=range(4), values=[last_rs] * 4)
        m.conditional_forecast(plan1, label='rate_hold',
                               variables=['OBS_RS', 'OBS_DA_LCPI',
                                          'OBS_DA_LCPIX', 'OBS_LGDP'],
                               n_draws=200)

        # Scenario 2: Gradual easing + inflation returning to target
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
                                          'OBS_DA_LCPIX', 'OBS_DA_LS'],
                               n_draws=200)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Sacrifice ratio summary (handles both point and posterior)
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

    # Pass-through summary (handles both point and posterior)
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

    print(f"\n  Results saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

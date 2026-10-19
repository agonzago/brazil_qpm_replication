"""Brazil QPM Working Paper -- Model G: three-way ToT decomposition.

Extends the Model A / Model F dataset and structure with a third terms-of-
trade component: a near-unit-root super-cycle TT_SC alongside TT_BAR
(secular trend) and TT_GAP (business-cycle gap).  The super-cycle drives
aggregate demand (phi_tt_sc) and the country risk premium (phi_prem_sc).
See ``model/brazil_qpm_model_g_v2.qpm`` for the structural derivation.

As in Model F, the three Taylor-rule reaction coefficients are pinned in
``FIXED_PARAMS`` rather than estimated:

    gamma1 = 0.75       interest-rate smoothing
    gamma2 = 1.50       year-ahead core inflation response (Taylor principle)
    gamma4 = 0.2023     output-gap response

This isolates the question this specification is designed to answer:
**given a fixed policy reaction function, does explicitly identifying a
commodity super-cycle component generate larger ToT-driven dynamics in
the historical decomposition than the two-component ToT in Model F?**

Outputs go to ``results_v2/model_g/`` (or ``results_v2_diffuse/model_g/``
with ``--use-diffuse``).

Usage:
    python run_brazil_model_g_v2.py                       # full pipeline
    python run_brazil_model_g_v2.py --calibrate-only      # calibration + decomposition only
    python run_brazil_model_g_v2.py --skip-estimation     # use saved draws
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

# ======================================================================
# Paths
# ======================================================================
MOD_FILE = os.path.join(SCRIPT_DIR, "model", "brazil_qpm_model_g_v2.qpm")
# Prefer local copy (produced by prepare_data.py); fall back to toolbox copy
_LOCAL_CSV = os.path.join(SCRIPT_DIR, "data", "estimation_data.csv")
_TOOLBOX_CSV = os.path.join(TOOLBOX_DIR, "data", "brazil", "estimation_data.csv")
DATA_CSV = _LOCAL_CSV if os.path.exists(_LOCAL_CSV) else _TOOLBOX_CSV
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_v2", "model_g")

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
    # SHK_TT_SC: super-cycle innovation (Model G).  Not pinned to an
    # observable in CYCLE_MAP because OBS_TT is already mapped to
    # SHK_TT_GAP for the high-frequency component; SHK_TT_SC is the
    # *low-frequency* shock that the cycle SNR machinery cannot infer
    # from a single observable.  Mode = 1.0 (.qpm calibration); the
    # InvGamma(3, 4) prior gives mean ~ 2 with a long right tail to
    # let the data inflate sigma if the medium-frequency commodity
    # variation is large.
    {"name": "sigma_SHK_TT_SC",
     "prior": "invgamma",
     "args": {"concentration": 3.0, "rate": 4.0 * 1.0}},
]

# TREND_SNR_MAP: {trend_shock: (obs_anchor, idiosyncratic_name, SNR, type)}
TREND_SNR_MAP = {
    # LGDP_BAR loosened from 100 to 50: prior diagnostic showed an almost
    # straight LGDP_BAR with the entire 2015-16 recession and post-2020
    # dynamics absorbed by the cycle (GDP_GAP). A smaller SNR ⇒ larger
    # sigma_SHK_LGDP_BAR ⇒ more curvature in potential output.
    # LGDP_BAR + G_TREND kept at canonical priors (different from Model A).
    # Even though Model F shares A's QRR Koyck filter, F's Taylor
    # coefficients are FIXED at calibrated values (gamma2 = 1.50 vs A's
    # estimated ~3.14).  With less aggressive policy, the trend SHOCKS
    # propagate through the IS curve's QRR channel into the output gap;
    # loosening trend priors (as we did for A) BACKFIRES in F because the
    # bigger trend shocks generate a bigger gap.  Tighter canonical priors
    # constrain trend shocks and keep the gap bounded.
    'SHK_LGDP_BAR':   ('OBS_LGDP',      'output_idio_trend',  50,  'level'),
    'SHK_G_TREND':     ('OBS_LGDP',      'output_idio_trend', 200,  'growth'),
    'SHK_LZ_BAR':      ('OBS_DA_LS',     'reer_idio_trend',   200,  'level'),
    'SHK_DLZ_TREND':   ('OBS_DA_LS',     'reer_idio_trend',   400,  'growth'),
    # SHK_PREM_BAR loosened (150 -> 50) for Model F: country-premium trend
    # was too smooth, leaving residual PREM variation in PREM_GAP which
    # propagated through UIP into real-rate / REER gaps and ultimately
    # the output gap (Taylor-fixed F can't compensate via policy).
    # Loosening lets PREM_BAR track medium-frequency premium shifts as
    # trend rather than gap.
    'SHK_PREM_BAR':    ('OBS_PREM',      'prem_idio_trend',    50,  'level'),
    # SHK_RR_BAR kept at canonical SNR=300 (different from A): same
    # rationale as LGDP_BAR/G_TREND -- with Taylor fixed at gamma2=1.5,
    # loosening RR_BAR amplifies trend-shock propagation into the output
    # gap via QRR, making the cycle worse rather than better.
    'SHK_RR_BAR':      ('OBS_RS',        'rate_idio_trend',   300,  'level'),
    'SHK_PI_TREND':    ('OBS_DA_LCPI',   'pi_idio_trend',     600,  'level'),
    # RR_BAR_US loosened from 200 to 50: OBS_RR_BAR_US is the HLW r* point
    # estimate (already a smoothed series); we want the model trend to
    # track it closely rather than drift via shock variance.  Smaller SNR
    # ⇒ larger sigma_SHK_RR_BAR_US ⇒ tighter fit to observed r*.
    'SHK_RR_BAR_US':   ('OBS_RR_BAR_US', 'rate_idio_trend',    50,  'level'),
    # SHK_TT_BAR excluded: sigma fixed via FIXED_PARAMS, not estimated by NUTS.
    'SHK_PI_TREND_US': ('OBS_DA_LCPI_US','pi_idio_trend',    24000, 'level'),
}

FIXED_POINT_CONFIG = {
    'max_iter': 50,
    'tol': 1e-4,
    'damping': 0.5,
    'sigma_bounds': {
        'sigma_SHK_G_TREND':  (1e-8, 0.40),
        'sigma_SHK_RR_BAR':  (1e-8, 0.50),
        # sigma_SHK_TT_BAR not in TREND_SNR_MAP -- fixed-point won't calibrate it.
    },
}

# ======================================================================
# Structural priors (48 parameters estimated, 2 fixed)
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
    # phi_tt_is: ToT demand channel (high-frequency component, on QTT).
    #   Motivated by r=0.71 GDP-commodity correlation (Section 2.1, Fact 1).
    #   Calibration = 0.05.  Gamma: mode=0.05, std=0.04.
    {"name": "phi_tt_is", "prior": "gamma", "args": {"concentration": 3.2553, "rate": 45.1062}},
    # phi_tt_sc: ToT super-cycle demand channel (Model G).  Captures the
    #   medium-frequency income effect from sustained commodity-price
    #   levels: investment, capital deepening, and durable consumption
    #   respond to multi-year ToT trajectories more strongly than to
    #   high-frequency swings.  Calibration = 0.10.  Gamma: mode=0.10,
    #   std=0.08; allows the data to push toward zero (no super-cycle
    #   demand channel) or larger if commodity income matters.
    {"name": "phi_tt_sc", "prior": "gamma", "args": {"concentration": 3.2553, "rate": 22.5530}},

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
    # Model F: gamma1, gamma2, gamma4 are FIXED at their calibrated
    # values rather than estimated.  See FIXED_PARAMS below.

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
    # phi_prem_sc: premium sensitivity to ToT super-cycle (Model G).
    #   Sustained commodity booms compress sovereign spreads (Borensztein
    #   & Panizza 2009, Hilscher & Nosbusch 2010).  Calibration = 0.05;
    #   Gamma: mode=0.05, std=0.04 (same shape as phi_tt_is).
    {"name": "phi_prem_sc",  "prior": "gamma", "args": {"concentration": 3.2553, "rate": 45.1062}},

    # ---- Koyck Filters ----
    # rho_QZ: ER-to-core PC smoothing; "gradual transmission" consistent
    #   with Belaisch (2003) [3] ERPT peak at 3-7Q. Calibration = 0.90.
    {"name": "rho_QZ",    "prior": "beta",  "args": {"concentration1": 11.80, "concentration2": 2.20}},
    # rho_QRR: real-rate transmission lag; "long and variable lags"
    #   in monetary transmission (Friedman 1961). Calibration = 0.85.
    {"name": "rho_QRR",   "prior": "beta",  "args": {"concentration1": 9.50, "concentration2": 2.50}},
    # rho_QZ_E: ER-to-energy transmission lag; administered pricing
    #   further delays pass-through. Calibration = 0.88.
    {"name": "rho_QZ_E",  "prior": "beta",  "args": {"concentration1": 9.80, "concentration2": 2.20}},
    # rho_QZ_F: ER-to-food transmission lag; market-determined but
    #   still gradual. Calibration = 0.88.
    {"name": "rho_QZ_F",  "prior": "beta",  "args": {"concentration1": 9.80, "concentration2": 2.20}},

    # ---- Persistence ----
    # rho_tt: ToT business-cycle gap persistence (high frequency).
    #   Calibration = 0.75.
    {"name": "rho_tt",           "prior": "beta", "args": {"concentration1": 7.00, "concentration2": 3.00}},
    # rho_tt_sc: ToT super-cycle persistence (Model G).  Near-unit-root,
    #   tightly anchored at 0.97 (~22Q half-life).  This prior is what
    #   identifies TT_SC as the medium-frequency component, distinct from
    #   TT_BAR (true unit root, sigma fixed) and TT_GAP (rho ~ 0.75).
    #   Beta(18.46, 1.54): mode=0.97, std~0.06.
    {"name": "rho_tt_sc",        "prior": "beta", "args": {"concentration1": 18.46, "concentration2": 1.54}},
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
    # alpha_rr: domestic neutral rate -> UIP equilibrium. Calibration = 0.07
    #   (~10Q half-life). Error-correction to (RR_BAR_US + PREM_BAR + 4*DLZ_TREND).
    {"name": "alpha_rr",     "prior": "beta", "args": {"concentration1": 1.56, "concentration2": 8.44}},
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
    # TT secular-trend shock std (Model G).  In Models A/F this was
    # raised to 3.0 so that TT_BAR could follow medium-frequency
    # commodity super-cycles -- but at the cost of leaving TT_GAP small
    # and the super-cycle dynamically inert (no transmission to
    # GDP_GAP, premium, or RS).  Model G introduces a separate
    # super-cycle component TT_SC (estimated sigma) to absorb the
    # medium-frequency variation, so TT_BAR can return to representing
    # the *secular* (Prebisch-Singer) trend with a much smaller
    # innovation variance.
    'sigma_SHK_TT_BAR': 0.5,
    # ----- Taylor rule (FIXED, as in Model F) -----
    # Pinned at calibrated values so this run isolates the effect of
    # adding the super-cycle component TT_SC, holding the policy
    # reaction function constant across G and F.
    'gamma1': 0.75,        # interest-rate smoothing
    'gamma2': 1.50,        # year-ahead core inflation response (Taylor principle)
    'gamma4': 0.2023,      # output-gap response
}

# Measurement-noise standard deviations.
# OBS_DA_LCPI_TAR is the published BCB inflation target -- a policy choice,
# not a noisy realization -- so we treat it as nearly observed (small
# slack of 0.05 retained to avoid singular Cholesky when the trend
# cannot exactly match a step change in the target).
MEAS_NOISE_STD = {
    'OBS_LGDP': 0.0, 'OBS_DA_LCPI': 0.0, 'OBS_DA_LCPIX': 0.0,
    'OBS_DA_LCPI_ENERGY': 0.0, 'OBS_DA_LCPI_US': 0.0, 'OBS_DA_LS': 0.0,
    'OBS_LRWOIL_GAP': 2.0, 'OBS_LRWFOOD_GAP': 2.0,
    'OBS_GDP_GAP_US': 0.5, 'OBS_PREM': 0.5,
    'OBS_RS': 0.0, 'OBS_RS_US': 0.0, 'OBS_TT': 0.0,
    'OBS_DA_LCPI_TAR': 0.05,
    'OBS_RR_BAR_US': 0.5,
}

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
    'Terms of trade': ['SHK_TT_GAP', 'SHK_TT_SC'],
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
# Spaghetti plot with GDP growth instead of log-level
# ======================================================================
def plot_spaghetti_growth(result, ys, dates, obs_names, title, out_path,
                          color="darkorange"):
    """Spaghetti plot that converts OBS_LGDP to annualized GDP growth.

    All other variables are plotted in their native units.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    PRETTY = {
        "OBS_DA_LCPI":  "Headline Inflation (QoQ ann.)",
        "OBS_LGDP":     "GDP Growth (QoQ ann.)",
        "OBS_RS":       "Policy Rate (Selic, %)",
        "OBS_DA_LCPIX": "Core Inflation (QoQ ann.)",
    }
    YLIM = {
        "OBS_DA_LCPI":  (-5, 20),
        "OBS_LGDP":     (-45, 20),
        "OBS_RS":       (0, 25),
        "OBS_DA_LCPIX": (-5, 15),
    }

    variables = result.variables
    n_vars = len(variables)
    ncols = min(2, n_vars)
    nrows = (n_vars + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 3.5 * nrows),
                             squeeze=False)

    dates_ts = pd.DatetimeIndex(dates)
    plot_start = pd.Timestamp("2015-01-01")
    start_idx = max(0, int(np.searchsorted(dates_ts, plot_start)))

    # Pre-compute actual DA_LGDP from OBS_LGDP
    lgdp_idx = obs_names.index("OBS_LGDP")
    lgdp_actual = ys[:, lgdp_idx]
    da_lgdp_actual = np.full_like(lgdp_actual, np.nan)
    da_lgdp_actual[1:] = 4.0 * np.diff(lgdp_actual)

    for k, v in enumerate(variables):
        ax = axes[k // ncols, k % ncols]
        vi = obs_names.index(v)
        is_gdp = (v == "OBS_LGDP")

        # Actuals
        if is_gdp:
            ax.plot(dates_ts[start_idx:], da_lgdp_actual[start_idx:],
                    'k-', lw=1.5, label="Actual", zorder=3)
        else:
            ax.plot(dates_ts[start_idx:], ys[start_idx:, vi],
                    'k-', lw=1.5, label="Actual", zorder=3)

        # COVID shading
        ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-07-01"),
                   alpha=0.08, color="red", zorder=0)

        # Overlay forecast paths
        for origin_date, y_fcst in result.forecast_paths:
            H = y_fcst.shape[0]
            fcst_dates = pd.date_range(
                start=origin_date + pd.tseries.offsets.QuarterBegin(1),
                periods=H, freq='QS')

            if is_gdp:
                t_origin = np.searchsorted(dates_ts, origin_date)
                lgdp_origin = float(ys[t_origin, lgdp_idx])
                lgdp_fcst = y_fcst[:, vi]
                lgdp_with_origin = np.concatenate([[lgdp_origin], lgdp_fcst])
                fcst_vals = 4.0 * np.diff(lgdp_with_origin)
            else:
                fcst_vals = y_fcst[:, vi]

            ax.plot(fcst_dates, fcst_vals,
                    color=color, alpha=0.25, lw=0.8)

        ax.set_xlim(left=plot_start)
        if v in YLIM:
            ax.set_ylim(YLIM[v])
        ax.set_title(PRETTY.get(v, v), fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    for j in range(n_vars, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ======================================================================
# Taylor rule decomposition
# ======================================================================
def plot_taylor_decomposition(m, n_draws=200):
    """Decompose OBS_RS into structural Taylor rule components.

    Identity (exact at each t given smoothed states):
        OBS_RS = RS_TREND + gamma1*RS(-1)
               + (1-gamma1)*DA_LCPI(+1)
               + (1-gamma1)*gamma2*D4A_LCPIX(+3)
               + (1-gamma1)*gamma4*GDP_GAP
               + RESN_RS

    where RS_TREND = RR_BAR + PI_TREND (Fisher identity at trend).

    Loops over posterior draws to compute median and 68%/90% HDI bands.
    Saves the plot to results_dir/policy/taylor_decomposition.png.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import jax.numpy as jnp
    from qpm_toolbox.kalman_filter import KalmanFilter, get_dtype

    m._check_data()
    m._auto_load_draws()
    _, ys_np, dates, obs_names = m._data
    ys_jax, H_obs, H_t, mult = m._resolve_ys_H_multipliers()
    T_obs = ys_np.shape[0]

    # State variable name -> index in augmented state vector
    state_names = list(m.model.aug_state_vars)
    idx = {n: i for i, n in enumerate(state_names)}

    # Reporting equations for lead variables
    rep_eqs = m.model.reporting_eqs  # {var: [(coeff, state, shift), ...]}

    def _compute_reporting_var(x_smooth, var_name):
        """Evaluate a reporting variable from smoothed states."""
        out = np.zeros(T_obs)
        for coeff, sname, shift in rep_eqs[var_name]:
            s_idx = idx[sname]
            for t in range(T_obs):
                t_src = t + shift
                if 0 <= t_src < T_obs:
                    out[t] += coeff * x_smooth[t_src, s_idx]
        return out

    # Determine draw indices
    has_posterior = m._posterior_samples is not None
    if has_posterior and n_draws:
        total = m._posterior_samples[
            next(iter(m._posterior_samples))].shape[0]
        n_use = min(n_draws, total)
        step = max(1, total // n_use)
        indices = list(range(0, total, step))[:n_use]
    else:
        indices = [None]

    dt = get_dtype()

    # 4-group decomposition:
    #   1. Neutral rate: gamma1*RS(-1) + (1-gamma1)*(RR_BAR + PI_TREND)
    #   2. Output gap:   (1-gamma1)*gamma4*GDP_GAP
    #   3. Inflation:    (1-gamma1)*[DA_LCPI(+1) + gamma2*D4A_LCPIX(+3)]
    #   4. Policy shock: RESN_RS
    comp_labels = [
        'Neutral rate',
        'Output gap',
        'Inflation',
        'Policy shock',
    ]
    n_comp = len(comp_labels)
    all_comp = []  # will be (n_draws, T, n_comp)
    obs_rs_draws = []

    for draw_i, mcmc_idx in enumerate(indices):
        # Build parameter dict
        if mcmc_idx is not None:
            pdict = dict(m.model.default_param_assignments)
            pdict.update(m._fixed_params or {})
            if m._calibrated_sigmas:
                pdict.update(m._calibrated_sigmas)
            for name in m._estimated_names:
                pdict[name] = float(
                    m._posterior_samples[name][mcmc_idx])
        else:
            pdict = m._resolve_params()

        # Solve model
        sol = m.model.solve(pdict)
        if not sol["solution_valid"]:
            continue

        P_aug = sol["P_aug"]
        R_aug = sol["R_aug"]
        Omega = sol["Omega"]
        c_aug = sol["c_aug"]

        # Time-varying Q_t from shock multipliers
        Q_t_local = None
        if mult is not None:
            mult_jax = jnp.asarray(mult, dtype=dt)
            R_jax = jnp.asarray(R_aug, dtype=dt)
            R_scaled = R_jax[None, :, :] * mult_jax[:, None, :]
            Q_t_local = jnp.einsum('tij,tkj->tik', R_scaled, R_scaled)

        # Run Kalman smoother
        kf = KalmanFilter(T=P_aug, R=R_aug, C=Omega, H=H_obs,
                          init_x=m._init_x, init_P=m._init_P,
                          c=c_aug, dtype=dt)
        x_smooth = np.asarray(kf.smooth_states_only(
            ys=ys_jax, H_t=H_t, Q_t=Q_t_local))

        # Extract state paths
        rs       = x_smooth[:, idx['RS']]
        rr_bar   = x_smooth[:, idx['RR_BAR']]
        pi_trend = x_smooth[:, idx['PI_TREND']]
        gdp_gap  = x_smooth[:, idx['GDP_GAP']]
        resn_rs  = x_smooth[:, idx['RESN_RS']]

        # Reporting variables (materialised leads)
        eda_lcpi1    = _compute_reporting_var(x_smooth, 'EDA_LCPI1')
        ed4a_lcpix3  = _compute_reporting_var(x_smooth, 'ED4A_LCPIX3')

        # Parameters for this draw
        g1 = float(pdict['gamma1'])
        g2 = float(pdict['gamma2'])
        g4 = float(pdict['gamma4'])

        # Compute 4 grouped additive components of OBS_RS
        # OBS_RS = RS_TREND + RS, where RS is in gap space:
        #   RS = gamma1*RS(-1) + (1-g1)*[...] + RESN_RS
        # Neutral rate: trend anchor + smoothing (gap lag)
        neutral = (rr_bar + pi_trend) + g1 * np.concatenate(
            [[0.0], rs[:-1]])
        # Output gap response
        output  = (1 - g1) * g4 * gdp_gap
        # Combined inflation response (expected + core deviation)
        infl    = (1 - g1) * (eda_lcpi1 + g2 * ed4a_lcpix3)
        # Policy shock
        shock   = resn_rs

        components = np.column_stack([neutral, output, infl, shock])
        all_comp.append(components)

        # Reconstructed OBS_RS (for verification)
        obs_rs_draws.append(components.sum(axis=1))

        if (draw_i + 1) % 50 == 0:
            print(f"    Taylor decomposition: {draw_i + 1}/{len(indices)} draws")

    all_comp = np.array(all_comp)     # (n_draws, T, 4)
    obs_rs_rec = np.array(obs_rs_draws)  # (n_draws, T)

    # Posterior statistics
    med  = np.median(all_comp, axis=0)   # (T, 6)
    lo68 = np.percentile(all_comp, 16, axis=0)
    hi68 = np.percentile(all_comp, 84, axis=0)

    # Observed policy rate from data
    obs_rs_data = ys_np[:, obs_names.index('OBS_RS')]

    # Dates
    if dates is not None:
        x_ax = (dates.to_pydatetime()
                if hasattr(dates, 'to_pydatetime') else dates)
    else:
        x_ax = np.arange(T_obs)
    has_dates = dates is not None

    # ---- Plot: overlaid time series with 68% HDI bands ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle('Taylor Rule Decomposition of the Policy Rate',
                 fontsize=13, fontweight='bold')

    colors = ['#2166ac', '#4daf4a', '#ff7f00', '#984ea3']
    styles = ['-', '--', '-.', ':']

    # Each component as a line with 68% HDI band
    for k in range(n_comp):
        ax.fill_between(x_ax, lo68[:, k], hi68[:, k],
                        alpha=0.15, color=colors[k])
        ax.plot(x_ax, med[:, k], color=colors[k], lw=1.5,
                ls=styles[k], label=comp_labels[k])

    # Observed policy rate as thick black line
    ax.plot(x_ax, obs_rs_data, 'k-', lw=2.0, label='Observed $i_t$',
            zorder=5)
    ax.axhline(0, color='grey', lw=0.6, ls=':')
    ax.set_ylabel('Percentage points')
    ax.grid(True, ls=':', alpha=0.3, axis='y')
    ax.legend(fontsize='small', loc='best', ncol=2)

    if has_dates:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = os.path.join(m.results_dir, "policy")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "taylor_decomposition.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---- Verification: reconstruction error ----
    rec_med = np.median(obs_rs_rec, axis=0)
    max_err = np.max(np.abs(rec_med - obs_rs_data))
    print(f"  Max reconstruction error (median): {max_err:.4f} pp")

    # ---- Save draws to CSV for further analysis ----
    rows = []
    for k, label in enumerate(comp_labels):
        for t in range(T_obs):
            rows.append({
                'component': label,
                'date': str(dates[t].date()) if dates is not None else t,
                'median': med[t, k],
                'lo68': lo68[t, k],
                'hi68': hi68[t, k],
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "taylor_decomposition.csv"), index=False)
    print(f"  Saved: {out_dir}/taylor_decomposition.png")
    print(f"  Saved: {out_dir}/taylor_decomposition.csv")


def plot_rstar_posterior(m, n_draws=200):
    """Plot the natural real rate (RR_BAR) with posterior uncertainty bands.

    Generates diagnostic/real_rate_natural_rate.png showing:
      - RR_BAR with 68% and 90% HDI bands
      - Ex-ante real rate: OBS_RS - PI_TREND (model-consistent)
      - RR_BAR_US for comparison
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import jax.numpy as jnp
    from qpm_toolbox.kalman_filter import KalmanFilter, get_dtype

    m._check_data()
    m._auto_load_draws()
    _, ys_np, dates, obs_names = m._data
    ys_jax, H_obs, H_t, mult = m._resolve_ys_H_multipliers()
    T_obs = ys_np.shape[0]

    state_names = list(m.model.aug_state_vars)
    idx = {n: i for i, n in enumerate(state_names)}

    # Determine draw indices
    has_posterior = m._posterior_samples is not None
    if has_posterior and n_draws:
        total = m._posterior_samples[
            next(iter(m._posterior_samples))].shape[0]
        n_use = min(n_draws, total)
        step = max(1, total // n_use)
        indices = list(range(0, total, step))[:n_use]
    else:
        indices = [None]

    dt = get_dtype()

    rr_bar_draws = []
    rr_bar_us_draws = []
    pi_trend_draws = []
    prem_bar_draws = []

    for draw_i, mcmc_idx in enumerate(indices):
        if mcmc_idx is not None:
            pdict = dict(m.model.default_param_assignments)
            pdict.update(m._fixed_params or {})
            if m._calibrated_sigmas:
                pdict.update(m._calibrated_sigmas)
            for name in m._estimated_names:
                pdict[name] = float(
                    m._posterior_samples[name][mcmc_idx])
        else:
            pdict = m._resolve_params()

        sol = m.model.solve(pdict)
        if not sol["solution_valid"]:
            continue

        P_aug = sol["P_aug"]
        R_aug = sol["R_aug"]
        Omega = sol["Omega"]
        c_aug = sol["c_aug"]

        Q_t_local = None
        if mult is not None:
            mult_jax = jnp.asarray(mult, dtype=dt)
            R_jax = jnp.asarray(R_aug, dtype=dt)
            R_scaled = R_jax[None, :, :] * mult_jax[:, None, :]
            Q_t_local = jnp.einsum('tij,tkj->tik', R_scaled, R_scaled)

        kf = KalmanFilter(T=P_aug, R=R_aug, C=Omega, H=H_obs,
                          init_x=m._init_x, init_P=m._init_P,
                          c=c_aug, dtype=dt)
        x_smooth = np.asarray(kf.smooth_states_only(
            ys=ys_jax, H_t=H_t, Q_t=Q_t_local))

        rr_bar_draws.append(x_smooth[:, idx['RR_BAR']])
        rr_bar_us_draws.append(x_smooth[:, idx['RR_BAR_US']])
        pi_trend_draws.append(x_smooth[:, idx['PI_TREND']])
        prem_bar_draws.append(x_smooth[:, idx['PREM_BAR']])

        if (draw_i + 1) % 50 == 0:
            print(f"    R* posterior: {draw_i + 1}/{len(indices)} draws")

    rr_bar_arr = np.array(rr_bar_draws)      # (n_draws, T)
    rr_bar_us_arr = np.array(rr_bar_us_draws)
    pi_trend_arr = np.array(pi_trend_draws)
    prem_bar_arr = np.array(prem_bar_draws)

    # Statistics
    rr_med = np.median(rr_bar_arr, axis=0)
    rr_lo68 = np.percentile(rr_bar_arr, 16, axis=0)
    rr_hi68 = np.percentile(rr_bar_arr, 84, axis=0)
    rr_lo90 = np.percentile(rr_bar_arr, 5, axis=0)
    rr_hi90 = np.percentile(rr_bar_arr, 95, axis=0)

    rr_us_med = np.median(rr_bar_us_arr, axis=0)
    rr_us_lo68 = np.percentile(rr_bar_us_arr, 16, axis=0)
    rr_us_hi68 = np.percentile(rr_bar_us_arr, 84, axis=0)

    pi_trend_med = np.median(pi_trend_arr, axis=0)

    prem_bar_med = np.median(prem_bar_arr, axis=0)
    prem_bar_lo68 = np.percentile(prem_bar_arr, 16, axis=0)
    prem_bar_hi68 = np.percentile(prem_bar_arr, 84, axis=0)

    # Ex-ante real rate: OBS_RS - PI_TREND (model-consistent)
    obs_rs = ys_np[:, obs_names.index('OBS_RS')]
    real_rate = obs_rs - pi_trend_med

    # Dates
    if dates is not None:
        x_ax = (dates.to_pydatetime()
                if hasattr(dates, 'to_pydatetime') else dates)
    else:
        x_ax = np.arange(T_obs)
    has_dates = dates is not None

    # ---- Plot: 2-panel figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Real Interest Rate and Estimated Natural Rate',
                 fontsize=13, fontweight='bold')

    # Panel 1: RR_BAR vs ex-ante real rate
    ax1.fill_between(x_ax, rr_lo90, rr_hi90, alpha=0.12,
                     color='#2166ac', label='90% HDI')
    ax1.fill_between(x_ax, rr_lo68, rr_hi68, alpha=0.25,
                     color='#2166ac', label='68% HDI')
    ax1.plot(x_ax, rr_med, color='#2166ac', lw=2,
             label='$\\bar{r}_t$ (posterior median)')
    ax1.plot(x_ax, real_rate, 'k-', lw=0.8, alpha=0.6,
             label='Ex-ante real rate ($i_t - \\bar{\\pi}_t$)')
    ax1.set_title('Domestic Natural Real Rate', fontsize=10)
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, ls=':', alpha=0.4)
    ax1.set_ylabel('Percent (annualised)')
    if has_dates:
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 2: RR_BAR decomposition (RR_BAR_US + PREM_BAR + attractor)
    ax2.plot(x_ax, rr_med, color='#2166ac', lw=2,
             label='$\\bar{r}_t$ (Brazil)')
    ax2.fill_between(x_ax, rr_lo68, rr_hi68, alpha=0.20,
                     color='#2166ac')
    ax2.plot(x_ax, rr_us_med, color='#b2182b', lw=1.5, ls='--',
             label='$\\bar{r}_t^*$ (US)')
    ax2.fill_between(x_ax, rr_us_lo68, rr_us_hi68, alpha=0.15,
                     color='#b2182b')
    ax2.fill_between(x_ax, prem_bar_lo68, prem_bar_hi68, alpha=0.15,
                     color='#4daf4a')
    ax2.plot(x_ax, prem_bar_med, color='#4daf4a', lw=1.2, ls=':',
             label='$\\bar{\\rho}_t$ (risk premium)')
    ax2.set_title('Natural Rate Components', fontsize=10)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, ls=':', alpha=0.4)
    ax2.set_ylabel('Percent (annualised)')
    if has_dates:
        ax2.xaxis.set_major_locator(mdates.YearLocator(5))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = os.path.join(m.results_dir, "diagnostic")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "real_rate_natural_rate.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_dir}/real_rate_natural_rate.png")


# ======================================================================
# Main pipeline
# ======================================================================
def main():
    import jax
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(description="Brazil QPM Working Paper")
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
        results_dir = os.path.join(SCRIPT_DIR, "results_v2_diffuse", "model_g")
    else:
        results_dir = RESULTS_DIR

    # ------------------------------------------------------------------
    # 1. Load data and create model
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  BRAZIL QPM -- WORKING PAPER ANALYSIS")
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
                    # Model F: gamma1/gamma2/gamma4 are fixed in
                    # FIXED_PARAMS, so there's no posterior to trace
                    # under a 'taylor' group.
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

        # Super-cycle IRF: the headline diagnostic for Model G.  A near-
        # unit-root TT_SC innovation should generate a hump-shaped GDP_GAP
        # response, a sustained premium compression, and a Taylor-rule
        # response that is larger than the high-frequency SHK_TT_GAP IRF.
        m.irf(shocks=['SHK_TT_SC'],
              variables=['TT_SC', 'GDP_GAP', 'PREM_GAP', 'RS', 'DA_LCPIX',
                         'DA_LCPI', 'LZ_GAP'],
              horizon=80, n_draws=200)

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
                use_diffuse=args.use_diffuse,
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
            baseline_group='Trends',
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

    # ----------------------------------------------------------
    # 13b. Taylor rule decomposition (posterior distribution)
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("  13b. Taylor Rule Decomposition (posterior)")
    print("-" * 50)

    plot_taylor_decomposition(m, n_draws=200)

    # ----------------------------------------------------------
    # 13c. Natural rate posterior (R* with HDI bands)
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("  13c. Natural Rate Posterior (R* plot)")
    print("-" * 50)

    plot_rstar_posterior(m, n_draws=200)

    if not args.skip_forecast:
        # ----------------------------------------------------------
        # 14. Forecast evaluation (rolling-window RMSE)
        # ----------------------------------------------------------
        print("\n" + "-" * 50)
        print("  14. Out-of-Sample Forecast Evaluation")
        print("-" * 50)

        EVAL_VARS = ['OBS_DA_LCPI', 'OBS_LGDP', 'OBS_RS', 'OBS_DA_LCPIX']

        ev = m.forecast_evaluation(
            variables=EVAL_VARS,
            first_origin='2017-01-01',
            last_origin='2022-10-01',
            horizons=[1, 2, 3, 4, 8],
            max_horizon=8,
            stochastic=False,
            plot=True,
        )

        # Regenerate spaghetti with GDP growth instead of log-level
        _, ys_np, dates_arr, obs_names_arr = m._data
        eval_dir = os.path.join(m.results_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        plot_spaghetti_growth(
            ev, ys_np, dates_arr, obs_names_arr,
            title="QPM: Unconditional Rolling Forecast Paths",
            out_path=os.path.join(eval_dir, "forecast_spaghetti.png"),
            color="steelblue",
        )

        # ----------------------------------------------------------
        # 14b. Conditional forecast evaluation (rolling-window RMSE)
        #      Conditions on external variables and energy/food inflation
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
            ('obs', 'OBS_RR_BAR_US',      'SHK_RR_BAR_US'),
        ]

        ev_cond = m.conditional_forecast_evaluation(
            variables=EVAL_VARS,
            condition_on=CONDITION_ON,
            first_origin='2017-01-01',
            last_origin='2022-10-01',
            horizons=[1, 2, 3, 4, 8],
            max_horizon=8,
            plot=True,
        )

        # Regenerate spaghetti with GDP growth instead of log-level
        plot_spaghetti_growth(
            ev_cond, ys_np, dates_arr, obs_names_arr,
            title="QPM: Conditional Rolling Forecast Paths",
            out_path=os.path.join(eval_dir,
                                  "conditional_forecast_spaghetti.png"),
            color="darkorange",
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
                               n_draws=200,
                               use_diffuse=args.use_diffuse)

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
                               n_draws=200,
                               use_diffuse=args.use_diffuse)

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

"""Result containers and type definitions for the VAR toolbox.

All public result types are NamedTuples for immutability and easy unpacking.
Prior configuration uses a mutable dataclass for convenient construction.
"""

from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional

import numpy as np


# ======================================================================== #
#  Data containers
# ======================================================================== #

class VARInfo(NamedTuple):
    """Metadata produced by :func:`core.build_design_matrices`."""
    K: int                              # number of endogenous variables
    p: int                              # lag order
    T: int                              # effective sample length (T_raw - p)
    constant: bool                      # intercept included?
    var_names: Optional[List[str]]      # variable names (None if unavailable)
    last_lags: np.ndarray               # (p, K) last p observations for forecasting
    n_exog: int = 0                     # number of exogenous regressors


class VARData(NamedTuple):
    """Design matrices for a VAR(p)."""
    Y: np.ndarray       # (T-p, K) response matrix
    X: np.ndarray       # (T-p, Kp+c) regressor matrix
    info: VARInfo


# ======================================================================== #
#  Classical VAR model
# ======================================================================== #

class VARModel(NamedTuple):
    """Estimated classical VAR(p) model.

    Coefficient layout of *B* (rows x K):
        Row 0           : intercept  (if ``constant=True``)
        Rows 1 .. Kp    : lag coefficients  [lag-1 block, lag-2 block, ...]
        Rows Kp+c .. end: exogenous regressor coefficients (if any)
    """
    B: np.ndarray                       # (Kp+c+m, K) full coefficient matrix
    A: np.ndarray                       # (Kp, K)   lag coefficients only
    c: Optional[np.ndarray]             # (K,) intercept or None
    Sigma: np.ndarray                   # (K, K)  residual covariance
    E: np.ndarray                       # (T_eff, K) residuals
    K: int
    p: int
    T_eff: int                          # effective obs used in estimation
    constant: bool
    var_names: Optional[List[str]]
    n_exog: int = 0                     # number of exogenous regressors
    D: Optional[np.ndarray] = None      # (m, K) exogenous coefficients


# ======================================================================== #
#  Bayesian VAR prior options
# ======================================================================== #

@dataclass
class BVARPriorOptions:
    """Configuration for Minnesota-prior BVAR.

    Attributes:
        lambda1:     Overall tightness (shrinkage).  Typical: 0.1-0.5.
        lambda2:     Cross-variable tightness ratio.  Typical: 0.5.
        lambda3:     Lag-decay exponent.  Typical: 1-2.
        lambda4:     Intercept prior variance.  Typical: 100.
        delta:       (K,) prior mean for own first lag.
                     0 -> stationary, 1 -> unit-root, intermediate allowed.
        add_sims:    Include Sims SOC + DIO dummy observations.
        lambda_soc:  Sum-of-Coefficients tightness.
        lambda_dio:  Dummy-Initial-Observation tightness.
        lambda_exog: Exogenous-coefficient tightness.  Controls shrinkage
                     of exogenous regressor coefficients toward zero.
                     Used only when exog_prior_scales are passed to the
                     dummy-data generator.
        bvar_compat: Use R ``bvar`` package Minnesota formulas.
    """
    lambda1: float = 0.2
    lambda2: float = 0.5
    lambda3: float = 2.0
    lambda4: float = 100.0
    delta: Optional[np.ndarray] = None
    add_sims: bool = False
    lambda_soc: float = 1.0
    lambda_dio: float = 1.0
    lambda_exog: float = 1.0
    bvar_compat: bool = True


# ======================================================================== #
#  Bayesian VAR posterior
# ======================================================================== #

class BVARPosteriorParams(NamedTuple):
    """Analytical Normal-Inverse-Wishart posterior parameters.

    The posterior is:
        Sigma | Y  ~  IW(S_post, df_post)
        vec(B) | Sigma, Y  ~  N(vec(B_post), Sigma (x) V_post)
    """
    B_post: np.ndarray      # (k, K)  posterior mean of coefficients
    V_post: np.ndarray      # (k, k)  posterior precision (cov factor)
    S_post: np.ndarray      # (K, K)  posterior scale matrix
    df_post: int            # posterior degrees of freedom
    T_star: int             # augmented sample size (T_data + T_dummy)


class BVARDraws(NamedTuple):
    """Collection of posterior draws from a BVAR.

    Each element of *B_draws* / *Sigma_draws* is one Monte Carlo draw.
    When exogenous regressors are present, each B_draw has shape
    ``(Kp + c + m, K)`` where ``m = n_exog``.
    """
    B_draws: List[np.ndarray]           # list of (k, K) coefficient draws
    Sigma_draws: List[np.ndarray]       # list of (K, K) covariance draws
    n_draws: int
    K: int
    p: int
    constant: bool
    var_names: Optional[List[str]]
    n_exog: int = 0                     # number of exogenous regressors


# ======================================================================== #
#  IRF containers
# ======================================================================== #

class IRFResult(NamedTuple):
    """Impulse-response functions from a classical VAR."""
    irf: np.ndarray                     # (H+1, K_resp, K_shock)
    var_names: Optional[List[str]]
    shock_names: Optional[List[str]]
    horizon: int


class BVARIRFResult(NamedTuple):
    """Impulse-response functions from a BVAR (with credible bands).

    *irf* stores the full draw-level array; *median*, *p16*, *p84*
    are pre-computed summaries.
    """
    irf: np.ndarray                     # (N_draws, K_resp, K_shock, H+1)
    median: np.ndarray                  # (K_resp, K_shock, H+1)
    p16: np.ndarray                     # 16th percentile
    p84: np.ndarray                     # 84th percentile
    var_names: Optional[List[str]]
    shock_names: Optional[List[str]]
    horizon: int


# ======================================================================== #
#  Forecast containers
# ======================================================================== #

class ForecastResult(NamedTuple):
    """Unconditional forecast from a classical VAR."""
    forecasts: np.ndarray               # (H, K)
    var_names: Optional[List[str]]
    horizon: int


class BVARForecastResult(NamedTuple):
    """Forecast distribution from a BVAR (with credible bands)."""
    forecasts: np.ndarray               # (N_draws, H, K)
    median: np.ndarray                  # (H, K)
    p16: np.ndarray
    p84: np.ndarray
    var_names: Optional[List[str]]
    horizon: int


class ConditionalForecastResult(NamedTuple):
    """Waggoner-Zha conditional forecast."""
    forecasts: np.ndarray               # (H, K) or (N_draws, H, K)
    shocks: np.ndarray                  # (K*H,) or (N_draws, K*H)
    var_names: Optional[List[str]]
    horizon: int


# ======================================================================== #
#  Empirical Bayes
# ======================================================================== #

class EBResult(NamedTuple):
    """Result of Empirical-Bayes BVAR estimation."""
    posterior_draws: BVARDraws
    posterior_params: BVARPosteriorParams
    optimised_params: dict              # {'lambda1': ..., 'lambda2': ..., ...}
    log_ml: float                       # maximised log-marginal likelihood
    info: dict                          # additional diagnostics

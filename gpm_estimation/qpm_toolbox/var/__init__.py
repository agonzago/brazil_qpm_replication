"""VAR and Bayesian VAR estimation toolbox.

Subpackage of ``qpm_toolbox`` providing classical and Bayesian VAR(p)
estimation, Cholesky-identified impulse response functions, unconditional
and Waggoner-Zha conditional forecasting, Empirical-Bayes hyperparameter
optimisation, and publication-quality plotting.

Quick start::

    from qpm_toolbox.var import estimate_bvar, irf_bvar, BVARPriorOptions
    import numpy as np

    Y = np.random.randn(100, 3)          # T=100, K=3
    opts = BVARPriorOptions(lambda1=0.2)
    draws, post, info = estimate_bvar(Y, p=4, options=opts, seed=42)
    irfs = irf_bvar(draws, horizon=20)

References:
    Litterman (1986), Doan, Litterman & Sims (1984),
    Sims (1993), Kadiyala & Karlsson (1997),
    Giannone, Lenza & Primiceri (2015),
    Waggoner & Zha (1999).
"""

# Core utilities
from .core import (
    build_design_matrices,
    cholesky_identification,
    companion_matrix,
    is_stable,
    make_constraint_matrix,
)

# Result types
from .types import (
    BVARDraws,
    BVARForecastResult,
    BVARIRFResult,
    BVARPosteriorParams,
    BVARPriorOptions,
    ConditionalForecastResult,
    EBResult,
    ForecastResult,
    IRFResult,
    VARData,
    VARInfo,
    VARModel,
)

# Classical estimation
from .classical import estimate_ols, estimate_stable

# Bayesian estimation
from .bayesian import estimate_bvar

# Empirical Bayes
from .empirical_bayes import EBConfig, estimate_eb_bvar

# Impulse response functions
from .irf import (
    ProxyVARDiagnostics,
    SignRestriction,
    irf_bvar,
    irf_bvar_proxy,
    irf_bvar_sign_restrictions,
    irf_cholesky,
    normalise_irf,
)

# Forecasting
from .forecast import (
    forecast_bvar,
    forecast_conditional_bvar,
    forecast_conditional_wz,
    forecast_unconditional,
)

# Plotting
from .plotting import plot_fan_chart, plot_irf, plot_irf_comparison

__all__ = [
    # Core
    "build_design_matrices", "companion_matrix", "is_stable",
    "cholesky_identification", "make_constraint_matrix",
    # Types
    "VARData", "VARInfo", "VARModel",
    "BVARPriorOptions", "BVARPosteriorParams", "BVARDraws",
    "IRFResult", "BVARIRFResult",
    "ForecastResult", "BVARForecastResult", "ConditionalForecastResult",
    "EBResult", "EBConfig",
    # Classical
    "estimate_ols", "estimate_stable",
    # Bayesian
    "estimate_bvar",
    # Empirical Bayes
    "estimate_eb_bvar",
    # IRFs
    "irf_cholesky", "irf_bvar", "irf_bvar_sign_restrictions",
    "irf_bvar_proxy", "normalise_irf",
    "SignRestriction", "ProxyVARDiagnostics",
    # Forecasting
    "forecast_unconditional", "forecast_bvar",
    "forecast_conditional_wz", "forecast_conditional_bvar",
    # Plotting
    "plot_irf", "plot_fan_chart", "plot_irf_comparison",
]

"""QPM Toolbox: Bayesian estimation of linearized DSGE/QPM models.

Usage:
    from qpm_toolbox import QPMModel

    m = QPMModel("models/brazil.qpm")
    m.set_data(ys, dates=dates)
    m.set_priors(structural_priors, ...)
    m.set_initial_conditions(init_state_config=config)
    m.calibrate(method='both')
    m.estimate(warmup=500, samples=800)
    m.irf(shocks=['SHK_RS'], variables=['GDP_GAP'])
    m.forecast(horizon=20)
    m.decomposition(variables=['GDP_GAP'], shock_groups={...})
"""

# Main user-facing class
from qpm_toolbox.qpm_model import QPMModel

# Core model and filter
from qpm_toolbox.model import DynareModelWithLambdified
from qpm_toolbox.kalman_filter import (
    KalmanFilter, DiffuseKalmanFilter, build_diffuse_init,
    solve_lyapunov_diffuse_jax,
    get_dtype, prepare_observations,
)
from qpm_toolbox.dk_smoother import DKSimulationSmoother
from qpm_toolbox.estimation import numpyro_model
from qpm_toolbox.plotting import plot_model_irfs, plot_smoothed_var

# Calibration
from qpm_toolbox.calibration import (
    calibrate_snr, calibrate_fixed_point, calibrate,
    compute_prior_modes,
)

# Validation
from qpm_toolbox.validation import validate_model_file, validate_and_report

# Forecasting
from qpm_toolbox.forecast import (
    forecast_unconditional, forecast_posterior,
    ForecastResult, get_state_forecast,
    get_obs_forecast, forecast_percentiles,
)

# Conditional forecasting
from qpm_toolbox.conditional_forecast import (
    ForecastPlan, ForecastCondition, ConditionalForecastResult,
    forecast_conditional, forecast_conditional_posterior,
    get_cond_state_forecast, get_cond_obs_forecast,
    get_conditioning_shocks, get_baseline_obs, get_baseline_state,
    cond_forecast_percentiles,
    plot_conditional_forecast, plot_conditioning_shocks,
)

# Historical shock decomposition
from qpm_toolbox.decomposition import (
    shock_decomposition, shock_decomposition_posterior,
    ShockDecompositionResult, PosteriorDecompositionResult,
    get_state_decomposition, get_obs_decomposition,
    decomposition_percentiles,
    plot_shock_decomposition,
)

# Policy analysis
from qpm_toolbox.policy_analysis import (
    sacrifice_ratio, pass_through,
    forecast_evaluation, conditional_forecast_evaluation,
    SacrificeRatioResult, PassThroughResult, ForecastEvaluationResult,
    plot_sacrifice_ratio, plot_pass_through,
    plot_forecast_spaghetti, plot_rmse_bars,
)

# MCMC reporting
from qpm_toolbox.reporting import mcmc_report

# Parser utilities (for advanced use)
from qpm_toolbox.parser import (
    irf, irf_observables, construct_initial_state,
)

# Filters (trend-cycle decomposition)
from qpm_toolbox.filters import hamilton_filter
from qpm_toolbox.hp_filter import hp_filter, hp_trend

# VAR toolbox (classical + Bayesian VAR estimation, IRFs, forecasting)
# Lazy import to avoid circular import on some platforms (Windows)
import importlib as _importlib

def __getattr__(name):
    if name == "var":
        return _importlib.import_module("qpm_toolbox.var")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

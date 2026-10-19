"""QPMModel: user-facing wrapper for QPM estimation and analysis.

Provides simple commands for the full QPM workflow:
    model.calibrate()        -- SNR + fixed-point shock std calibration
    model.estimate()         -- Bayesian MCMC estimation (NUTS)
    model.irf()              -- impulse response functions
    model.forecast()         -- unconditional forecasting with fan charts
    model.conditional_forecast()  -- Waggoner & Zha conditional forecast
    model.decomposition()    -- historical shock decomposition
    model.filter_diagnostic() -- pre-estimation smoother diagnostics
    model.solution_check()   -- Blanchard-Kahn stability diagnostics

Usage:
    from qpm_toolbox import QPMModel

    m = QPMModel("models/brazil.qpm")
    m.set_data(ys, dates=dates, obs_names=obs_names)
    m.set_priors(structural_priors, fixed_params=fixed_params,
                 meas_noise_std=meas_noise_std)
    m.set_initial_conditions(init_state_config=config)
    m.calibrate(method='both')
    m.estimate(warmup=500, samples=800)
    m.irf(shocks=['SHK_RS'], variables=['GDP_GAP', 'DA_LCPIX'])
    m.forecast(horizon=20)
    m.decomposition(variables=['GDP_GAP'], shock_groups={...})
"""

import os
from typing import Dict, List, Optional, Any, Union

import numpy as np
import jax
import jax.numpy as jnp

from qpm_toolbox.model import DynareModelWithLambdified
from qpm_toolbox.kalman_filter import (
    KalmanFilter, DiffuseKalmanFilter, build_diffuse_init,
    get_dtype, prepare_observations,
)
from qpm_toolbox.dk_smoother import DKSimulationSmoother
from qpm_toolbox.parser import (
    irf as _irf_raw, irf_observables as _irf_obs_raw,
    construct_initial_state,
)
from qpm_toolbox.calibration import (
    calibrate_snr, calibrate_fixed_point, calibrate as _calibrate_pipeline,
    compute_prior_modes,
)
from qpm_toolbox.validation import validate_model_file, validate_and_report


def _ensure_datetime_index(dates):
    """Convert dates to pandas DatetimeIndex if they are strings or a list."""
    if dates is None:
        return None
    import pandas as pd
    if isinstance(dates, pd.DatetimeIndex):
        return dates
    try:
        return pd.DatetimeIndex(pd.PeriodIndex(dates, freq='Q').to_timestamp())
    except Exception:
        return pd.DatetimeIndex(pd.to_datetime(dates))
from qpm_toolbox.forecast import (
    forecast_unconditional, forecast_posterior,
    ForecastResult, get_state_forecast, get_obs_forecast, forecast_percentiles,
)
from qpm_toolbox.decomposition import (
    shock_decomposition, shock_decomposition_posterior,
    ShockDecompositionResult, PosteriorDecompositionResult,
    get_state_decomposition, get_obs_decomposition,
    plot_shock_decomposition,
)
from qpm_toolbox.conditional_forecast import (
    ForecastPlan, forecast_conditional, forecast_conditional_posterior,
    plot_conditional_forecast, plot_conditioning_shocks,
)
from qpm_toolbox.reporting import mcmc_report
from qpm_toolbox.policy_analysis import (
    sacrifice_ratio as _sacrifice_ratio,
    pass_through as _pass_through,
    forecast_evaluation as _forecast_evaluation,
    conditional_forecast_evaluation as _conditional_forecast_evaluation,
    SacrificeRatioResult, PassThroughResult, ForecastEvaluationResult,
    plot_sacrifice_ratio, plot_pass_through,
    plot_forecast_spaghetti, plot_rmse_bars,
)


def _eval_reporting_paths(state_path, state_names, reporting_vars,
                           reporting_eqs, pre_sample_state=None):
    """Compute reporting-variable values from a (T, n_aug) state path.

    Reporting variables are deterministic linear combinations of augmented
    state variables, possibly at past or future lags (set by the
    ``reporting_equations;`` block of the .qpm file).  This helper applies
    the linear combinations to a smoothed-state path or a forecast path
    and returns a (T, n_rpt) array of values.

    Args:
        state_path:        (T, n_aug) numpy/jax array of augmented state
                           values (a smoothed posterior path or a
                           forecast path).
        state_names:       list of length ``n_aug`` matching the columns of
                           ``state_path``.
        reporting_vars:    ordered list of reporting-variable names.
        reporting_eqs:     {var_name: [(coeff, state_name, shift), ...]}
                           as returned by the parser.
        pre_sample_state:  optional (n_aug,) vector used when a lagged
                           reference points before row 0 of ``state_path``
                           (typical case: the smoothed state at the
                           forecast origin, used to anchor first-
                           difference reporting variables at horizon 1).

    Returns:
        (T, n_rpt) numpy array of reporting-variable values.  Cells where
        the required source row is out of range and no fallback is
        provided are set to NaN.
    """
    import numpy as onp

    sp = onp.asarray(state_path)
    if sp.ndim != 2:
        raise ValueError(
            f"state_path must be 2-D (T, n_aug); got shape {sp.shape}.")
    T, n_aug = sp.shape
    if len(state_names) != n_aug:
        raise ValueError(
            f"state_names length ({len(state_names)}) does not match "
            f"state_path columns ({n_aug}).")
    pre = (onp.asarray(pre_sample_state).ravel()
           if pre_sample_state is not None else None)

    out = onp.full((T, len(reporting_vars)), onp.nan, dtype=sp.dtype)
    for r_idx, rvar in enumerate(reporting_vars):
        terms = reporting_eqs.get(rvar, [])
        col = onp.zeros(T, dtype=sp.dtype)
        valid = onp.ones(T, dtype=bool)
        for coeff, sname, shift in terms:
            if sname not in state_names:
                # Silently mark this reporting var unavailable.
                valid[:] = False
                break
            s_idx = state_names.index(sname)
            for t in range(T):
                t_src = t + shift
                if 0 <= t_src < T:
                    col[t] += coeff * sp[t_src, s_idx]
                elif pre is not None and t_src < 0:
                    # Use pre-sample anchor state for lagged references
                    # that fall before row 0 (e.g., first differences at
                    # the first forecast horizon).
                    col[t] += coeff * pre[s_idx]
                else:
                    valid[t] = False
        out[:, r_idx] = onp.where(valid, col, onp.nan)
    return out


def _append_reporting_vars(result, reporting_vars, reporting_eqs,
                           init_x=None):
    """Extend a decomposition result with reporting variable contributions.

    Reporting variables are linear combinations of augmented state variables
    (possibly with time lags) defined in the ``reporting_equations;`` block
    of the .qpm model file.  They are not part of the Kalman filter; their
    shock contributions are computed here as post-processing.

    The function handles both ``ShockDecompositionResult`` (single-parameter)
    and ``PosteriorDecompositionResult`` (multiple posterior draws), detected
    by whether ``state_contrib`` has 3 or 4 dimensions.

    For a state variable ``s`` with index ``k`` and a lag ``d``, the
    contribution of shock ``j`` at time ``t`` is::

        state_contrib[j, t+d, k]    (fixed-parameter path)
        state_contrib[draw, j, t+d, k]  (posterior path)

    Pre-sample lags (``t + d < 0``):

    * **Shock contributions** remain zero -- no shocks occurred before the
      sample.
    * **Baseline** uses the Kalman filter initial state ``init_x`` so that
      first-difference reporting equations (e.g. ``DA_LGDP = LGDP_BAR -
      LGDP_BAR(-1) + ...``) produce values consistent with observed growth
      rates rather than the raw level of the initial state.

    Args:
        result:         ShockDecompositionResult or PosteriorDecompositionResult.
        reporting_vars: ordered list of reporting variable names.
        reporting_eqs:  {var_name: [(coeff, state_name, shift), ...]}.
        init_x:         (n_aug,) Kalman filter initial state mean.  When
                        provided, pre-sample lagged baseline values are set
                        to the corresponding element of ``init_x`` instead
                        of zero.

    Returns:
        A new result of the same type with extended state arrays and
        ``state_names`` updated to include the reporting variables.
    """
    import numpy as onp

    state_names = list(result.state_names)
    sc = onp.asarray(result.state_contrib)   # (n_shock, T, n_state) or (J, n_shock, T, n_state)
    sb = onp.asarray(result.state_baseline)  # (T, n_state) or (J, T, n_state)

    if init_x is not None:
        init_x_np = onp.asarray(init_x).ravel()
    else:
        init_x_np = None

    posterior_mode = sc.ndim == 4  # True for PosteriorDecompositionResult

    if posterior_mode:
        J, n_shock, T, _ = sc.shape
        rpt_sc = onp.zeros((J, n_shock, T, len(reporting_vars)), dtype=sc.dtype)
        rpt_sb = onp.zeros((J, T, len(reporting_vars)), dtype=sb.dtype)
    else:
        n_shock, T, _ = sc.shape
        rpt_sc = onp.zeros((n_shock, T, len(reporting_vars)), dtype=sc.dtype)
        rpt_sb = onp.zeros((T, len(reporting_vars)), dtype=sb.dtype)

    for r_idx, rvar in enumerate(reporting_vars):
        terms = reporting_eqs.get(rvar, [])
        for coeff, sname, shift in terms:
            if sname not in state_names:
                print(f"  Warning: reporting var '{rvar}' references "
                      f"unknown state '{sname}'. Term skipped.")
                continue
            s_idx = state_names.index(sname)
            for t in range(T):
                t_src = t + shift
                if 0 <= t_src < T:
                    if posterior_mode:
                        rpt_sc[:, :, t, r_idx] += coeff * sc[:, :, t_src, s_idx]
                        rpt_sb[:, t, r_idx]    += coeff * sb[:, t_src, s_idx]
                    else:
                        rpt_sc[:, t, r_idx] += coeff * sc[:, t_src, s_idx]
                        rpt_sb[t, r_idx]    += coeff * sb[t_src, s_idx]
                elif t_src < 0 and init_x_np is not None:
                    # Pre-sample: baseline uses init_x; shock contribs stay zero.
                    if posterior_mode:
                        rpt_sb[:, t, r_idx] += coeff * init_x_np[s_idx]
                    else:
                        rpt_sb[t, r_idx] += coeff * init_x_np[s_idx]

    new_sc = onp.concatenate([sc, rpt_sc], axis=-1)
    new_sb = onp.concatenate([sb, rpt_sb], axis=-1)
    new_names = state_names + list(reporting_vars)

    # Extend smoothed_states so that plot_shock_decomposition can overlay
    # the reporting variable's total path (baseline + sum of contributions).
    # Identity: smoothed[t] = baseline[t] + sum_j contrib[j, t]
    ss = onp.asarray(result.smoothed_states)  # (T, n_state) or (J, T, n_state)
    if posterior_mode:
        rpt_ss = rpt_sb + rpt_sc.sum(axis=1)  # (J, T, n_report)
    else:
        rpt_ss = rpt_sb + rpt_sc.sum(axis=0)  # (T, n_report)
    new_ss = onp.concatenate([ss, rpt_ss], axis=-1)

    return result._replace(state_contrib=new_sc, state_baseline=new_sb,
                           state_names=new_names, smoothed_states=new_ss)


class QPMModel:
    """User-facing wrapper for QPM estimation, forecasting, and diagnostics.

    Wraps DynareModelWithLambdified and provides a simple, chainable API
    for the full QPM analysis pipeline. All outputs are saved to a results
    directory located next to the model file.

    Attributes:
        model:      The underlying DynareModelWithLambdified instance.
        results_dir: Path where all output artifacts are saved.
    """

    def __init__(self, mod_file: str, results_dir: Optional[str] = None,
                 verbose: bool = False, validate: bool = True,
                 skip_stability_check: Optional[bool] = None):
        """Parse a .qpm/.dyn model file and initialise the wrapper.

        Args:
            mod_file:     path to the .qpm or .dyn model file.
            results_dir:  output directory for plots, tables, etc.
                          Default: 'results/' next to the model file.
            verbose:      print parser output.
            validate:     run model file validation before parsing.
            skip_stability_check:
                          Controls the Blanchard-Kahn spectral-radius gate
                          inside ``solve()``.

                          - ``None`` (default): auto.  Skip on GPU (where
                            ``jnp.linalg.eigvals`` requires a CPU host
                            callback per MCMC sample and can fail on
                            ill-conditioned matrices), enforce on CPU.
                          - ``False``: always enforce.
                          - ``True``: always skip.

                          When skipped, unstable draws are still rejected
                          via NaN log-likelihoods produced downstream by
                          the Lyapunov/Kalman steps; the rejection is just
                          less crisp than the explicit gate.
        """
        self.mod_file = os.path.abspath(mod_file)

        # Default results_dir: sibling of model file
        if results_dir is None:
            model_dir = os.path.dirname(self.mod_file)
            results_dir = os.path.join(model_dir, "results")
        self.results_dir = os.path.abspath(results_dir)

        # Validate
        if validate:
            issues = validate_model_file(self.mod_file)
            if issues:
                print(f"\nWARNING: Model file has {len(issues)} issue(s):")
                for issue in issues:
                    print(f"  - {issue}")
                print()

        # Parse.  When skip_stability_check is None (default) the
        # underlying DynareModelWithLambdified resolves it based on the
        # active JAX backend (skip on GPU, enforce on CPU).
        self.model = DynareModelWithLambdified(
            self.mod_file, verbose=verbose,
            skip_stability_check=skip_stability_check,
        )
        if self.model._skip_stability_check:
            print("\n  WARNING: Blanchard-Kahn stability check is DISABLED.")
            print("  Unstable parameter draws will be rejected via NaN log-")
            print("  likelihoods downstream, but the rejection is less crisp")
            print("  than the explicit spectral-radius gate.  Use only when")
            print("  the host-callback eigvals path is a measurable bottleneck.")

        # Internal state (populated by pipeline methods)
        self._data = None           # (ys_jax, ys_np, dates, obs_names)
        self._priors = None         # list of all prior spec dicts
        self._structural_priors = None
        self._sigma_priors = None
        self._fixed_params = None   # dict of fixed parameter values
        self._meas_noise_std = None # dict obs_name -> std
        self._H_obs = None          # (n_obs, n_obs) measurement noise cov
        self._init_x = None         # (n_aug,) initial state mean
        self._init_P = None         # (n_aug, n_aug) initial state cov
        self._param_dict = None     # current parameter dict (for fixed-param ops)
        self._calibrated_sigmas = None  # dict after calibration
        self._mcmc = None           # numpyro MCMC object
        self._posterior_samples = None  # dict from MCMC
        self._estimated_names = None    # list of estimated param names
        self._cycle_map = None
        self._trend_snr_map = None
        self._manual_cycle_priors = None

        # Tunes and shock multipliers
        self._tune_spec = None                # list of tune dicts
        self._tuned_ys = None                 # (T, n_obs) jax array (NaN->0)
        self._tuned_H_t = None                # (T, n_obs, n_obs) jax array
        self._shock_multiplier_spec = None    # user dict spec
        self._shock_multiplier_array = None   # (T, n_shock) numpy array

        print(f"Model: {os.path.basename(self.mod_file)}")
        print(f"  States: {self.model.n_stat} stationary + "
              f"{self.model.n_trend} trend = {self.model.n_aug} augmented")
        print(f"  Shocks: {self.model.n_s_shock} stationary + "
              f"{self.model.n_t_shock} trend = {self.model.n_aug_shock} total")
        print(f"  Observables: {self.model.n_obs}")
        print(f"  Results dir: {self.results_dir}")

    # ==================================================================
    # Configuration methods
    # ==================================================================

    def set_data(self, ys, dates=None, obs_names=None):
        """Attach observation data.

        Args:
            ys:        (T, n_obs) numpy array, or path to CSV file.
            dates:     pandas DatetimeIndex or array of dates (optional).
            obs_names: list of observable names (for CSV column selection).
                       If None, uses model's ordered_obs_vars.

        Returns:
            self (for chaining).
        """
        if isinstance(ys, str):
            import pandas as pd
            df = pd.read_csv(ys)
            if dates is None and 'date' in df.columns:
                dates = pd.to_datetime(df['date'])
            if obs_names is None:
                obs_names = list(self.model.ordered_obs_vars)
            ys = df[obs_names].values

        dt = get_dtype()
        ys_np = np.asarray(ys, dtype=np.float64)
        ys_jax = jnp.asarray(ys_np, dtype=dt)

        if obs_names is None:
            obs_names = list(self.model.ordered_obs_vars)

        self._data = (ys_jax, ys_np, dates, obs_names)
        print(f"  Data: {ys_np.shape[0]} periods, {ys_np.shape[1]} observables")
        return self

    def set_priors(self, structural_priors, fixed_params=None,
                   meas_noise_std=None, cycle_map=None,
                   trend_snr_map=None, manual_cycle_priors=None):
        """Attach prior specifications and calibration maps.

        Args:
            structural_priors:  list of prior spec dicts for structural params.
            fixed_params:       dict of {param_name: value} for non-estimated params.
            meas_noise_std:     dict of {obs_name: std} for measurement noise.
            cycle_map:          {obs_name: (shock_name, scale)} for SNR calibration.
            trend_snr_map:      {shock_name: (ref_obs, class, snr, ref_type)}.
            manual_cycle_priors: additional cycle priors without a dedicated obs.

        Returns:
            self (for chaining).
        """
        self._structural_priors = list(structural_priors)
        self._fixed_params = dict(fixed_params) if fixed_params else {}
        self._meas_noise_std = dict(meas_noise_std) if meas_noise_std else {}
        self._cycle_map = cycle_map
        self._trend_snr_map = trend_snr_map
        self._manual_cycle_priors = manual_cycle_priors

        # Build H_obs
        dt = get_dtype()
        n_obs = self.model.n_obs
        H_diag = np.zeros(n_obs)
        for i, obs in enumerate(self.model.ordered_obs_vars):
            std = self._meas_noise_std.get(obs, 0.0)
            H_diag[i] = std ** 2
        self._H_obs = jnp.diag(jnp.array(H_diag, dtype=dt))

        print(f"  Priors: {len(structural_priors)} structural parameters")
        if cycle_map:
            print(f"  Cycle map: {len(cycle_map)} observables")
        if trend_snr_map:
            print(f"  Trend SNR map: {len(trend_snr_map)} trend shocks")
        return self

    def set_initial_conditions(self, init_state_config=None,
                               init_x=None, init_P=None,
                               default_trend_std=0.5):
        """Set Kalman filter initial conditions.

        Either provide init_state_config (dict of {trend_var: {mean, std}})
        and let the method build init_x and init_P, or provide explicit
        init_x and init_P arrays.

        Args:
            init_state_config: dict mapping trend var names to {mean, std}.
            init_x:            explicit (n_aug,) initial state mean.
            init_P:            explicit (n_aug, n_aug) initial state cov.
            default_trend_std: default std for trend states not in config.

        Returns:
            self (for chaining).
        """
        dt = get_dtype()
        n_aug = self.model.n_aug
        n_stat = self.model.n_stat

        if init_x is not None and init_P is not None:
            self._init_x = jnp.asarray(init_x, dtype=dt)
            self._init_P = jnp.asarray(init_P, dtype=dt)
        elif init_state_config is not None:
            # Build init_x and init_P from config
            x0 = np.zeros(n_aug)
            P0_diag = np.zeros(n_aug)

            # Stationary states: zero mean, small variance
            P0_diag[:n_stat] = 1.0

            # Trend states: from config
            for i, tvar in enumerate(self.model.ordered_trend_state_vars):
                idx = n_stat + i
                cfg = init_state_config.get(tvar, {})
                x0[idx] = cfg.get("mean", 0.0)
                std = cfg.get("std", default_trend_std)
                P0_diag[idx] = std ** 2

            self._init_x = jnp.array(x0, dtype=dt)
            self._init_P = jnp.diag(jnp.array(P0_diag, dtype=dt))
        else:
            # Defaults
            x0 = np.zeros(n_aug)
            P0_diag = np.ones(n_aug)
            P0_diag[:n_stat] = 1.0
            P0_diag[n_stat:] = default_trend_std ** 2
            self._init_x = jnp.array(x0, dtype=dt)
            self._init_P = jnp.diag(jnp.array(P0_diag, dtype=dt))

        print(f"  Initial conditions set (n_aug={n_aug})")
        return self

    def set_params(self, param_dict):
        """Set fixed parameter values for non-Bayesian operations.

        These are used by irf(), filter_diagnostic(), etc. when no
        posterior samples are available.

        Returns:
            self (for chaining).
        """
        self._param_dict = dict(param_dict)
        return self

    # ==================================================================
    # Tunes and shock multipliers
    # ==================================================================

    def _resolve_date_index(self, date_str, label="date"):
        """Convert a date string to an integer index into the data sample.

        Uses ``searchsorted`` on the data's date index.  Warns if the
        requested date falls outside the sample or does not match an
        existing observation date exactly (snaps to the nearest date
        that is >= the requested one).

        Args:
            date_str: date string parseable by ``pd.Timestamp``.
            label:    descriptive label for warning messages.

        Returns:
            int index into the data array.

        Raises:
            ValueError: if the date is beyond the end of the sample.
        """
        import pandas as pd

        _, ys_np, dates, _ = self._data
        T_obs = ys_np.shape[0]
        ts = pd.Timestamp(date_str)
        idx = int(dates.searchsorted(ts))

        if idx >= T_obs:
            raise ValueError(
                f"{label} '{date_str}' is beyond the data range "
                f"(last date: {dates[-1]}).")
        if idx == 0 and ts < dates[0]:
            print(f"  WARNING: {label} '{date_str}' is before the sample "
                  f"start ({dates[0]}); clipping to index 0.")
        elif dates[idx] != ts:
            print(f"  WARNING: {label} '{date_str}' not in date index; "
                  f"snapped to {dates[idx]} (index {idx}).")
        return idx

    def set_shock_multipliers(self, spec):
        """Set per-period scaling factors for structural shocks.

        Shock multipliers allow for time-varying shock volatility, e.g.
        increasing the standard deviation of growth shocks during crisis
        periods.  The multiplier array is combined with each posterior
        draw's impact matrix R inside vmap, so Q_t varies across draws.

        This triggers the time-varying Kalman filter path (full Riccati
        recursion at each t) rather than the steady-state gain path.

        Args:
            spec: dict mapping shock names to lists of
                  ``(start_date, end_date, factor)`` tuples.
                  Dates are strings parseable by ``pd.Timestamp``.
                  ``end_date=None`` means "through end of sample".
                  ``factor`` is the multiplicative scale (1.0 = no change).

                  Example::

                      {'SHK_G_TREND':   [('2020-01-01', None, 3.0)],
                       'SHK_L_GDP_GAP': [('2020-01-01', '2020-10-01', 5.0)]}

        Returns:
            self (for chaining).
        """
        self._check_data()
        _, ys_np, dates, _ = self._data
        T_obs = ys_np.shape[0]
        shock_names = list(self.model.aug_shocks)
        n_shocks = len(shock_names)

        for name in spec:
            if name not in shock_names:
                raise ValueError(
                    f"Unknown shock '{name}'. Available: {shock_names}")

        arr = np.ones((T_obs, n_shocks), dtype=np.float64)

        for shock_name, intervals in spec.items():
            j = shock_names.index(shock_name)
            for start_str, end_str, factor in intervals:
                i_start = self._resolve_date_index(
                    start_str, label=f"multiplier start ({shock_name})")
                if end_str is None:
                    i_end = T_obs
                else:
                    i_end = self._resolve_date_index(
                        end_str, label=f"multiplier end ({shock_name})")
                if i_end <= i_start:
                    print(f"  WARNING: empty range for {shock_name}: "
                          f"start={start_str} (idx {i_start}) >= "
                          f"end={end_str} (idx {i_end}); skipping.")
                    continue
                arr[i_start:i_end, j] = factor

        self._shock_multiplier_spec = spec
        self._shock_multiplier_array = arr

        n_modified = int(np.sum(arr != 1.0))
        print(f"  Shock multipliers: {len(spec)} shock(s), "
              f"{n_modified} modified entries")
        return self

    def set_tunes(self, tunes):
        """Impose soft judgment on observable variables.

        Tunes work by modifying the observation vector and measurement
        noise covariance at specific dates.  At each tune date the
        observed value is set to the user's target and the diagonal of
        H is set to ``std**2``, controlling the filter's confidence in
        the imposed value.

        Because observables are linear functions of states
        (y_t = Omega x_t), tuning an observable implicitly tunes the
        underlying state combination.  This is model-consistent by
        construction: any state the user wishes to influence must be
        expressed through its observable mapping.

        This triggers the time-varying Kalman filter path (full Riccati
        recursion) rather than the steady-state gain.

        Each tune dict supports two forms:

        **Point tune** (single date)::

            {'obs': 'L_GDP_OBS', 'date': '2019-10-01',
             'value': 1050.2,    'std': 0.5}

        **Range tune** (constant value over a date range)::

            {'obs': 'DLA_CPI_OBS', 'start': '2020-01-01',
             'end': '2021-01-01',   'value': 2.0,  'std': 0.3}

        For a range tune, ``'end'`` is exclusive (same convention as
        Python slicing).  Use ``'end': None`` for "through end of
        sample".

        Args:
            tunes: list of tune dicts.  Each must have keys ``'obs'``,
                   ``'value'``, ``'std'``, and either ``'date'`` (point)
                   or ``'start'``/``'end'`` (range).

        Returns:
            self (for chaining).
        """
        self._check_data()
        _, ys_np, dates, _ = self._data
        dt = get_dtype()
        T_obs, n_obs = ys_np.shape
        obs_vars = list(self.model.ordered_obs_vars)

        for tune in tunes:
            if tune['obs'] not in obs_vars:
                raise ValueError(
                    f"Unknown observable '{tune['obs']}'. "
                    f"Available: {obs_vars}")
            has_point = 'date' in tune
            has_range = 'start' in tune
            if not has_point and not has_range:
                raise ValueError(
                    "Each tune must have either 'date' (point) or "
                    "'start'/'end' (range).")

        # Start from original data
        ys_tuned = ys_np.copy()

        # Base H diagonal from measurement noise spec
        base_H_diag = np.zeros(n_obs)
        if self._meas_noise_std:
            for i, obs in enumerate(obs_vars):
                base_H_diag[i] = self._meas_noise_std.get(obs, 0.0) ** 2

        # Per-period H diagonal: (T, n_obs)
        H_diag_t = np.tile(base_H_diag, (T_obs, 1))

        # Apply tunes
        n_points, n_periods = 0, 0
        for tune in tunes:
            j = obs_vars.index(tune['obs'])
            obs_label = tune['obs']

            if 'date' in tune:
                # Point tune: single date
                t = self._resolve_date_index(
                    tune['date'], label=f"tune ({obs_label})")
                ys_tuned[t, j] = tune['value']
                H_diag_t[t, j] = tune['std'] ** 2
                n_points += 1
            else:
                # Range tune: start to end
                i_start = self._resolve_date_index(
                    tune['start'], label=f"tune start ({obs_label})")
                if tune.get('end') is None:
                    i_end = T_obs
                else:
                    i_end = self._resolve_date_index(
                        tune['end'], label=f"tune end ({obs_label})")
                if i_end <= i_start:
                    print(f"  WARNING: empty tune range for {obs_label}: "
                          f"start={tune['start']} (idx {i_start}) >= "
                          f"end={tune.get('end')} (idx {i_end}); skipping.")
                    continue
                ys_tuned[i_start:i_end, j] = tune['value']
                H_diag_t[i_start:i_end, j] = tune['std'] ** 2
                n_periods += (i_end - i_start)

        # Handle NaN: replace with 0, inflate H diagonal
        fill_value = 1e10
        nan_mask = np.isnan(ys_tuned)
        ys_clean = np.where(nan_mask, 0.0, ys_tuned)
        H_diag_t[nan_mask] += fill_value

        # Build full H_t: (T, n_obs, n_obs) diagonal matrices
        H_t = np.zeros((T_obs, n_obs, n_obs), dtype=np.float64)
        eye_idx = np.arange(n_obs)
        H_t[:, eye_idx, eye_idx] = H_diag_t

        self._tune_spec = list(tunes)
        self._tuned_ys = jnp.asarray(ys_clean, dtype=dt)
        self._tuned_H_t = jnp.asarray(H_t, dtype=dt)

        n_tunes = len(tunes)
        n_nan = int(np.sum(nan_mask))
        detail = []
        if n_points:
            detail.append(f"{n_points} point")
        if n_periods:
            detail.append(f"{n_periods} range periods")
        print(f"  Tunes: {n_tunes} entry(ies) ({', '.join(detail)})")
        if n_nan > 0:
            print(f"  NaN entries: {n_nan} (masked in filter)")
        return self

    def clear_tunes(self):
        """Remove all tunes, restoring original observation data.

        Returns:
            self (for chaining).
        """
        self._tune_spec = None
        self._tuned_ys = None
        self._tuned_H_t = None
        print("  Tunes cleared.")
        return self

    def clear_shock_multipliers(self):
        """Remove all shock multipliers, restoring constant volatility.

        Returns:
            self (for chaining).
        """
        self._shock_multiplier_spec = None
        self._shock_multiplier_array = None
        print("  Shock multipliers cleared.")
        return self

    def _resolve_ys_H_multipliers(self):
        """Resolve observation data and noise for tunes/multipliers.

        Returns:
            (ys, H_obs, H_t, shock_multipliers) where:

            - ys:                (T, n_obs) JAX array (may contain NaN; the
                                 diffuse smoother handles NaN natively, see
                                 ``DiffuseKalmanFilter`` univariate update).
            - H_obs:             (n_obs, n_obs) base measurement noise cov.
            - H_t:               (T, n_obs, n_obs) or None.  Only non-None
                                 when tunes were set; for plain NaN in ys
                                 the diffuse path is preferred.
            - shock_multipliers: (T, n_shock) numpy array or None.
        """
        ys_jax = self._data[0]

        ys = self._tuned_ys if self._tuned_ys is not None else ys_jax
        H_t = self._tuned_H_t   # None if no tunes
        mult = self._shock_multiplier_array  # None if no multipliers

        return ys, self._H_obs, H_t, mult

    # ==================================================================
    # Solution diagnostics
    # ==================================================================

    def solution_check(self, params=None):
        """Check Blanchard-Kahn conditions and model stability.

        Solves the model and reports:
          - Whether the solution exists and is valid
          - Eigenvalues of the state transition matrix P_aug
          - Spectral radius (max |eigenvalue|)
          - Number of stable vs unstable eigenvalues
          - Whether BK conditions are satisfied

        Args:
            params: parameter dict (default: prior modes or current params).

        Returns:
            dict with keys 'valid', 'eigenvalues', 'spectral_radius',
                 'n_stable', 'n_unstable', 'P_aug'.
        """
        pdict = self._resolve_params(params)
        sol = self.model.solve(pdict)

        result = {
            'valid': bool(sol['solution_valid']),
            'eigenvalues': None,
            'spectral_radius': None,
            'n_stable': 0,
            'n_unstable': 0,
            'P_aug': None,
        }

        if not result['valid']:
            print("  Model does NOT solve with these parameters!")
            return result

        P_aug = np.asarray(sol['P_aug'])
        eigvals = np.linalg.eigvals(P_aug)
        abs_eigvals = np.abs(eigvals)
        spectral_radius = float(np.max(abs_eigvals))

        n_stable = int(np.sum(abs_eigvals < 1.0 - 1e-10))
        n_unit = int(np.sum(np.abs(abs_eigvals - 1.0) < 1e-10))
        n_unstable = int(np.sum(abs_eigvals > 1.0 + 1e-10))

        result.update({
            'eigenvalues': eigvals,
            'spectral_radius': spectral_radius,
            'n_stable': n_stable,
            'n_unit_root': n_unit,
            'n_unstable': n_unstable,
            'P_aug': P_aug,
        })

        print(f"\n  Solution check:")
        print(f"    Solution valid: {result['valid']}")
        print(f"    State dimension: {P_aug.shape[0]}")
        print(f"    Spectral radius: {spectral_radius:.6f}")
        print(f"    Eigenvalues: {n_stable} stable, {n_unit} unit root, "
              f"{n_unstable} unstable")

        if n_unstable > 0:
            print(f"    WARNING: {n_unstable} explosive eigenvalue(s) detected!")
            explosive = eigvals[abs_eigvals > 1.0 + 1e-10]
            for ev in explosive:
                print(f"      |lambda| = {abs(ev):.6f}  "
                      f"(lambda = {ev.real:.6f} + {ev.imag:.6f}i)")

        # Check expected unit roots = n_trend (random walk trends)
        if n_unit != self.model.n_trend:
            print(f"    NOTE: Expected {self.model.n_trend} unit roots "
                  f"(trend states), found {n_unit}")

        # Stationary block eigenvalues
        P_stat = P_aug[:self.model.n_stat, :self.model.n_stat]
        stat_eigvals = np.abs(np.linalg.eigvals(P_stat))
        stat_sr = float(np.max(stat_eigvals)) if len(stat_eigvals) > 0 else 0
        print(f"    Stationary block spectral radius: {stat_sr:.6f}")
        if stat_sr >= 1.0:
            print(f"    WARNING: Stationary block is NOT stable "
                  f"(spectral radius >= 1)!")

        return result

    # ==================================================================
    # Calibration
    # ==================================================================

    def calibrate(self, method='both', fixed_point_config=None,
                  alpha=3.0, fp_prior_alpha=5.0,
                  *,
                  cycle_persistence: float,
                  drift_persistence: float,
                  pins_fixed=None,
                  pin_trends_from_snr: bool = True,
                  q_estimator: str = "mean",
                  q_estimator_kwargs=None,
                  scale_schedule=None,
                  use_diffuse: bool = True):
        """Calibrate shock standard deviations.

        Args:
            method:             'snr', 'fixed_point', or 'both'.
            fixed_point_config: dict with 'max_iter', 'tol', 'damping',
                                'sigma_bounds', 'max_log_step',
                                'bound_freeze_steps'.
            alpha:              InverseGamma alpha for SNR priors.
            fp_prior_alpha:     InverseGamma alpha for post-FP priors.
            cycle_persistence:  rho for cycle AR(1) - required.  Drives the
                                AR(1) corrections in :func:`calibrate_snr`.
            drift_persistence:  rho_g for drift AR(1) - required.
            pins_fixed:         {sigma_name: value} - pin and freeze
                                specific shock stds during fixed-point.
                                Overrides any auto-built trend pins
                                (see ``pin_trends_from_snr``).
            pin_trends_from_snr: if ``True`` (default), every trend
                                shock listed in ``trend_snr_map`` is
                                pinned at its SNR-derived mode and the
                                fixed-point EM updates only the cycle
                                shocks.  Matches the MATLAB workflow
                                used in the reference implementations
                                (gpm7_estimation/code/run_calibration.m).
                                Set ``False`` to recover the legacy
                                behaviour where the EM updates all
                                shock stds simultaneously - typically
                                produces large drifts in trend sigmas
                                because the EM has no economic anchor.
            q_estimator:        'mean' (default) | 'trimmed_mean' | 'winsorized'.
            q_estimator_kwargs: kwargs for the aggregator (e.g.
                                ``{'alpha': 0.10}`` for trimmed_mean).
            scale_schedule:     {shock_name: [(start, end, factor), ...]}
                                deterministic per-window sigma scaling.  Uses
                                the dataset's date index automatically.
            use_diffuse:        if ``True`` (default), the fixed-point EM
                                step uses the exact diffuse Kalman
                                smoother - required for unbiased trend-
                                shock recovery on unit-root models, and
                                matches the Dynare/IRIS default.  Set
                                ``False`` to fall back to the standard
                                Kalman smoother with the supplied
                                ``init_P`` (e.g. when a
                                ``scale_schedule`` is in use; the
                                diffuse path does not yet support
                                time-varying shock scales).

        Returns:
            self (for chaining).
        """
        self._check_data()
        self._check_priors()
        self._ensure_init()

        ys_jax, ys_np, dates, obs_names = self._data

        # Build base param dict (structural modes + fixed)
        base_modes = compute_prior_modes(self._structural_priors)
        base_pdict = {**self.model.default_param_assignments,
                      **base_modes, **self._fixed_params}

        self._sigma_priors, self._calibrated_sigmas = _calibrate_pipeline(
            model=self.model,
            ys=ys_np,
            obs_names=obs_names,
            cycle_map=self._cycle_map,
            trend_snr_map=self._trend_snr_map,
            base_param_dict=base_pdict,
            H_obs=self._H_obs,
            init_x=self._init_x,
            init_P=self._init_P,
            cycle_persistence=cycle_persistence,
            drift_persistence=drift_persistence,
            method=method,
            manual_cycle_priors=self._manual_cycle_priors,
            alpha=alpha,
            fixed_point_config=fixed_point_config,
            fp_prior_alpha=fp_prior_alpha,
            pins_fixed=pins_fixed,
            pin_trends_from_snr=pin_trends_from_snr,
            q_estimator=q_estimator,
            q_estimator_kwargs=q_estimator_kwargs,
            scale_schedule=scale_schedule,
            dates=dates,
            use_diffuse=use_diffuse,
        )

        # Update _param_dict with calibrated values
        self._param_dict = {**base_pdict, **self._calibrated_sigmas}

        # Build combined priors list
        self._priors = self._structural_priors + self._sigma_priors

        return self

    # ==================================================================
    # Estimation
    # ==================================================================

    def estimate(self, warmup=500, samples=800, chains=1,
                 target_accept=0.90, max_tree_depth=10,
                 chain_method='sequential', seed=42,
                 trace_groups=None, hdi_prob=0.68, presample=0,
                 shock_multipliers=None,
                 use_diffuse=False, lyap_iters=60, n_diffuse=10):
        """Run NUTS MCMC estimation with automatic reporting.

        Args:
            warmup:           number of warmup samples per chain.
            samples:          number of posterior samples per chain.
            chains:           number of MCMC chains (>= 2 recommended
                              for R-hat convergence diagnostics).
            target_accept:    NUTS target acceptance probability (0, 1).
                              Higher values (0.90-0.99) reduce divergences
                              but increase computation.
            max_tree_depth:   maximum leapfrog tree depth (default 10).
                              Increase if hitting depth limit frequently.
            chain_method:     how to run multiple chains:
                              'sequential' -- one after another (default).
                              'vectorized' -- all chains in parallel via
                                vmap (faster on GPU, needs more memory).
                              'parallel' -- separate processes per chain.
            seed:             random seed.
            trace_groups:     dict of group_name -> [param_names] for traces.
            hdi_prob:         HDI probability for reporting.
            presample:        presample periods for likelihood.  Ignored
                              when use_diffuse=True (the exact diffuse LL
                              excludes diffuse-phase observations automatically).
            shock_multipliers: optional (T, n_shock) array of per-period
                              scale factors for structural shocks.  Entry
                              [t,j] multiplies column j of the impact
                              matrix R at time t.  Use for heteroscedastic
                              shock variances (e.g. crisis periods).
                              Ignored when use_diffuse=True.
            use_diffuse:      if True, use the exact diffuse Kalman filter
                              (Durbin & Koopman 2012) instead of the
                              approximate DARE-based filter.  Correct for
                              models with unit-root trend variables.
                              ``Pinf`` is pre-computed once from the
                              starting parameter values; ``Pstar`` is
                              re-computed at every MCMC step via a
                              JAX-native Lyapunov solver.
            lyap_iters:       number of matrix-doubling iterations for the
                              in-loop Pstar Lyapunov solver.  Use
                              ``check_lyap_convergence()`` to find the
                              minimum needed (typically 8-10 for QPM models).
            n_diffuse:        static Python int - number of periods for the
                              exact diffuse phase in the two-segment fast
                              filter.  Must be >= the true diffuse phase
                              length (periods until Pinf ~= 0).  Default 10
                              is ample for QPM models.

        Returns:
            self (for chaining).
        """
        from qpm_toolbox.estimation import numpyro_model
        import numpyro
        from numpyro.infer import MCMC, NUTS, init_to_value

        self._check_data()
        if self._priors is None:
            raise RuntimeError("No priors set. Call set_priors() and "
                               "calibrate() first.")

        # Resolve ys, H_t, shock_multipliers from tunes/multipliers
        ys_jax, H_obs, H_t, stored_mult = self._resolve_ys_H_multipliers()

        # Explicit shock_multipliers arg overrides stored
        if shock_multipliers is None:
            shock_multipliers = stored_mult

        # The exact diffuse filter handles NaN natively (univariate filter
        # skips missing observations via an is_missing mask).  Only apply the
        # time-varying H_t conversion when NOT using the diffuse filter.
        if not use_diffuse and self._tuned_ys is None:
            import numpy as _np
            has_missing = bool(_np.any(_np.isnan(_np.asarray(ys_jax))))
            if has_missing:
                n_nan = int(_np.sum(_np.isnan(_np.asarray(ys_jax))))
                print(f"  Missing data detected: {n_nan} NaN entries "
                      f"-> using time-varying Kalman filter")
                ys_clean, H_t, _ = prepare_observations(ys_jax, H_obs)
                ys_jax = ys_clean

        if not use_diffuse and (H_t is not None or shock_multipliers is not None):
            print(f"  Time-varying filter: "
                  f"H_t={'yes' if H_t is not None else 'no'}, "
                  f"multipliers={'yes' if shock_multipliers is not None else 'no'}")

        # Fixed params: everything not estimated
        estimated_names = [p["name"] for p in self._priors]
        self._estimated_names = estimated_names

        fixed_for_mcmc = {}
        for name in self.model.all_param_names:
            if name not in set(estimated_names):
                val = self._fixed_params.get(
                    name, self.model.default_param_assignments.get(name, 1.0))
                fixed_for_mcmc[name] = val

        # Init values from prior modes / calibrated sigmas
        init_vals = compute_prior_modes(self._priors)
        if self._calibrated_sigmas:
            init_vals.update(self._calibrated_sigmas)

        # Pre-compute diffuse initialisation matrices (parameter-independent).
        # Pinf is the unit-root projector; it does not change across draws.
        # ur_mask identifies which states are unit roots.
        # Pstar is re-computed at every MCMC iteration via the JAX Lyapunov
        # solver inside log_likelihood_diffuse.
        #
        # IMPORTANT: use_diffuse, Pinf, ur_mask, and lyap_iters must be Python-
        # level constants inside the traced model function, not JAX-traced
        # arguments.  lax.scan(length=lyap_iters) requires a concrete Python int.
        # We achieve this by baking these static values into the model function
        # via functools.partial BEFORE JAX traces it.  The partial captures them
        # as Python closure variables, which are visible as constants at trace time.
        from functools import partial as _partial
        if use_diffuse:
            start_params = {**fixed_for_mcmc, **init_vals}
            print("  Building diffuse initialisation matrices "
                  "(spectral analysis of P_aug)...")
            Pinf_mcmc, _, ur_mask_mcmc = self.model.build_diffuse_matrices(
                start_params, H_obs)
            n_ur = int(ur_mask_mcmc.sum())
            print(f"  Exact diffuse filter: {n_ur} unit-root state(s) "
                  f"identified (threshold 0.999).")
            print(f"  Pstar solved at each MCMC step via JAX Lyapunov "
                  f"doubling ({lyap_iters} iterations).")
            print(f"  Two-segment diffuse LL: exact diffuse for {n_diffuse} "
                  f"periods + univariate steady-state for the rest.")
            # Bake static diffuse args into the model function so they are
            # captured as Python-level constants, not JAX-traced values.
            # n_diffuse is a static int (lax.scan length must be concrete).
            _model_fn = _partial(
                numpyro_model,
                use_diffuse=True,
                Pinf=Pinf_mcmc,
                ur_mask=ur_mask_mcmc,
                lyap_iters=int(lyap_iters),
                n_diffuse=int(n_diffuse))
        else:
            _model_fn = numpyro_model

        os.makedirs(self.results_dir, exist_ok=True)
        est_dir = os.path.join(self.results_dir, "estimation")
        os.makedirs(est_dir, exist_ok=True)

        key = jax.random.PRNGKey(seed)
        kernel = NUTS(
            _model_fn,
            target_accept_prob=target_accept,
            max_tree_depth=max_tree_depth,
            init_strategy=init_to_value(values=init_vals),
        )
        mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                     num_chains=chains, chain_method=chain_method,
                     progress_bar=True)

        filter_label = "exact diffuse" if use_diffuse else "DARE (approx)"
        method_label = chain_method if chains > 1 else ""
        print(f"\nRunning MCMC: {warmup} warmup + {samples} samples "
              f"x {chains} chain(s) {method_label}")
        print(f"  filter={filter_label}, target_accept={target_accept}, "
              f"max_tree_depth={max_tree_depth}")
        print(f"  JIT-compiling model (this can take 5-15 min, "
              f"progress bar will appear after)...")
        import time as _time
        _t0 = _time.time()
        mcmc.run(key, self.model, self._priors, fixed_for_mcmc,
                 ys_jax, H_obs, self._init_x, self._init_P,
                 presample=presample,
                 H_t=H_t, shock_multipliers=shock_multipliers,
                 extra_fields=('diverging', 'mean_accept_prob',
                               'num_steps'))
        _elapsed = _time.time() - _t0
        print(f"  MCMC completed in {_elapsed/60:.1f} minutes")

        self._mcmc = mcmc
        self._posterior_samples = mcmc.get_samples()

        # Reporting (includes chain diagnostics)
        mcmc_report(mcmc, self._priors, est_dir,
                    trace_groups=trace_groups, hdi_prob=hdi_prob)

        # Auto-save draws so they can be reloaded without re-running MCMC
        self.save_draws()

        return self

    # ==================================================================
    # Save / load posterior draws
    # ==================================================================

    def save_draws(self, path: Optional[str] = None):
        """Save posterior draws to disk (.npz).

        Persists everything downstream methods need so that
        ``load_draws()`` fully restores posterior-based operations
        (forecast, decomposition, conditional forecast) without
        re-running MCMC.

        Args:
            path: file path (default: ``<results_dir>/estimation/draws.npz``).

        Returns:
            The path where the file was written.
        """
        if self._posterior_samples is None:
            raise RuntimeError("No posterior draws available. "
                               "Run estimate() first.")

        if path is None:
            est_dir = os.path.join(self.results_dir, "estimation")
            os.makedirs(est_dir, exist_ok=True)
            path = os.path.join(est_dir, "draws.npz")

        # Convert JAX arrays to NumPy for serialisation
        draws = {k: np.asarray(v) for k, v in self._posterior_samples.items()}

        # Metadata needed to reconstruct the estimation state
        import json
        meta = {
            'estimated_names': self._estimated_names or [],
            'calibrated_sigmas': self._calibrated_sigmas or {},
            'tune_spec': self._tune_spec,
            'shock_multiplier_spec': self._shock_multiplier_spec,
        }
        meta_json = np.array(json.dumps(meta))

        np.savez(path, _meta_json_=meta_json, **draws)
        print(f"  Saved {len(draws)} parameter draws "
              f"({list(draws.values())[0].shape[0]} samples) to {path}")
        return path

    def load_draws(self, path: Optional[str] = None):
        """Load posterior draws from disk (.npz).

        Restores ``_posterior_samples``, ``_estimated_names``, and
        ``_calibrated_sigmas`` so that all posterior-based methods
        (forecast, decomposition, conditional forecast) work without
        re-running MCMC.

        Args:
            path: file path (default: ``<results_dir>/estimation/draws.npz``).

        Returns:
            self (for chaining).
        """
        if path is None:
            path = os.path.join(self.results_dir, "estimation", "draws.npz")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Draws file not found: {path}")

        data = np.load(path, allow_pickle=True)

        # Restore metadata
        import json
        meta = json.loads(str(data['_meta_json_']))
        self._estimated_names = meta.get('estimated_names', [])
        cal_sigmas = meta.get('calibrated_sigmas', {})
        if cal_sigmas:
            self._calibrated_sigmas = {k: float(v)
                                       for k, v in cal_sigmas.items()}

        # Restore posterior samples (skip metadata key)
        self._posterior_samples = {
            k: jnp.asarray(data[k]) for k in data.files
            if k != '_meta_json_'
        }

        n_params = len(self._posterior_samples)
        n_draws = list(self._posterior_samples.values())[0].shape[0]
        print(f"  Loaded {n_params} parameter draws "
              f"({n_draws} samples) from {path}")

        # Rebuild tunes and multipliers from saved specs (if data is set)
        tune_spec = meta.get('tune_spec')
        mult_spec = meta.get('shock_multiplier_spec')
        if self._data is not None:
            if tune_spec:
                self.set_tunes(tune_spec)
            if mult_spec:
                self.set_shock_multipliers(mult_spec)

        return self

    def _auto_load_draws(self):
        """Load saved draws if available and not already in memory."""
        if self._posterior_samples is not None:
            return
        path = os.path.join(self.results_dir, "estimation", "draws.npz")
        if os.path.isfile(path):
            self.load_draws(path)

    # ==================================================================
    # IRFs
    # ==================================================================

    def irf(self, shocks=None, variables=None, shock_size=1.0,
            horizon=40, params=None, n_draws=None, plot=True):
        """Compute and plot impulse response functions.

        When n_draws is set and posterior samples are available, loops
        over posterior draws to produce IRFs with HDI bands.

        Args:
            shocks:     shock name(s) (str or list). Default: all shocks.
            variables:  variable name(s) to plot (str or list).
                        Default: all augmented state variables.
            shock_size: impulse size (1.0 = one std dev) or dict per shock.
            horizon:    number of periods.
            params:     parameter dict. Default: prior modes / posterior
                        median / calibrated params.
            n_draws:    number of posterior draws for HDI bands.
                        None = point IRF at resolved params.
            plot:       if True, save plots to results_dir/irf/.

        Returns:
            dict of {shock_name: (irf_median, var_names)} for point IRFs, or
            dict of {shock_name: (irf_median, var_names, irf_draws)} for
            posterior IRFs (irf_draws shape: n_valid x horizon x n_vars).
        """
        import matplotlib.pyplot as plt

        if n_draws is not None:
            self._auto_load_draws()

        all_shocks = self.model.aug_shocks
        all_vars = self.model.aug_state_vars
        all_obs = self.model.ordered_obs_vars

        # Resolve shocks
        if shocks is None:
            shocks = list(all_shocks)
        elif isinstance(shocks, str):
            shocks = [shocks]

        # Resolve variables -- can be state vars or observables
        if variables is None:
            variables = list(all_vars)
        elif isinstance(variables, str):
            variables = [variables]

        # Determine which are state vars and which are observables
        var_is_obs = []
        for v in variables:
            if v in all_vars:
                var_is_obs.append(False)
            elif v in all_obs:
                var_is_obs.append(True)
            else:
                raise ValueError(
                    f"Unknown variable '{v}'. Available states: "
                    f"{all_vars}, observables: {all_obs}")

        # ------- Collect parameter dicts to loop over -------
        use_posterior = (n_draws is not None
                        and self._posterior_samples is not None)

        if use_posterior:
            total = self._posterior_samples[
                next(iter(self._posterior_samples))].shape[0]
            n_use = min(n_draws, total)
            step = max(1, total // n_use)
            indices = list(range(0, total, step))[:n_use]
            print(f"  Computing posterior IRFs ({n_use} draws)...")
        else:
            indices = None  # single point

        def _compute_irfs_for_params(pdict):
            """Solve model and compute IRF array for one parameter set."""
            sol = self.model.solve(pdict)
            if not sol["solution_valid"]:
                return None

            P_aug = sol["P_aug"]
            R_aug = sol["R_aug"]
            Omega = sol["Omega"]

            per_shock = {}
            for shk in shocks:
                shk_idx = all_shocks.index(shk)
                sz = (shock_size.get(shk, 1.0)
                      if isinstance(shock_size, dict) else shock_size)

                state_irf = np.asarray(
                    _irf_raw(P_aug, R_aug, shk_idx, horizon))
                obs_irf = np.asarray(
                    _irf_obs_raw(P_aug, R_aug, Omega, shk_idx, horizon))

                irf_vals = np.zeros((horizon, len(variables)))
                for k, (v, is_obs) in enumerate(
                        zip(variables, var_is_obs)):
                    if is_obs:
                        idx = all_obs.index(v)
                        irf_vals[:, k] = obs_irf[:, idx] * sz
                    else:
                        idx = all_vars.index(v)
                        irf_vals[:, k] = state_irf[:, idx] * sz
                per_shock[shk] = irf_vals
            return per_shock

        # ------- Run -------
        if use_posterior:
            # Collect draws: {shock: list of (horizon, n_vars) arrays}
            all_draws = {shk: [] for shk in shocks}
            n_skipped = 0
            for mcmc_idx in indices:
                pdict = {name: self._posterior_samples[name][mcmc_idx]
                         for name in self._estimated_names}
                if self._fixed_params:
                    pdict.update(self._fixed_params)
                if self._calibrated_sigmas:
                    pdict.update(self._calibrated_sigmas)
                pdict_full = dict(self.model.default_param_assignments)
                pdict_full.update(pdict)

                result_one = _compute_irfs_for_params(pdict_full)
                if result_one is None:
                    n_skipped += 1
                    continue
                for shk in shocks:
                    all_draws[shk].append(result_one[shk])

            n_valid = len(all_draws[shocks[0]])
            if n_skipped > 0:
                print(f"  Skipped {n_skipped}/{len(indices)} draws "
                      f"(invalid solutions)")
            print(f"  Valid draws: {n_valid}")

            # Stack: (n_valid, horizon, n_vars)
            draws_stacked = {shk: np.stack(all_draws[shk])
                             for shk in shocks}
        else:
            pdict = self._resolve_params(params)
            result_one = _compute_irfs_for_params(pdict)
            if result_one is None:
                print("  Model does not solve -- cannot compute IRFs.")
                return {}
            draws_stacked = None

        # ------- Build results and plot -------
        results = {}

        if plot:
            irf_dir = os.path.join(self.results_dir, "irf")
            os.makedirs(irf_dir, exist_ok=True)

        for shk in shocks:
            if use_posterior:
                arr = draws_stacked[shk]  # (J, H, V)
                irf_median = np.median(arr, axis=0)
                irf_lo68 = np.percentile(arr, 16, axis=0)
                irf_hi68 = np.percentile(arr, 84, axis=0)
                irf_lo90 = np.percentile(arr, 5, axis=0)
                irf_hi90 = np.percentile(arr, 95, axis=0)
                results[shk] = (irf_median, list(variables), arr)
            else:
                irf_median = result_one[shk]
                results[shk] = (irf_median, list(variables))

            # Plot
            if plot:
                n_vars = len(variables)
                ncols = min(3, n_vars)
                nrows = (n_vars + ncols - 1) // ncols

                fig, axes = plt.subplots(
                    nrows, ncols,
                    figsize=(4.5 * ncols, 3.0 * nrows),
                    squeeze=False)
                sz = (shock_size.get(shk, 1.0)
                      if isinstance(shock_size, dict) else shock_size)
                fig.suptitle(f"IRF to {shk} (size={sz})", fontsize=13)

                t = np.arange(horizon)
                for k in range(n_vars):
                    ax = axes[k // ncols, k % ncols]
                    ax.plot(t, irf_median[:, k], linewidth=1.4,
                            color='#2171b5')
                    if use_posterior:
                        ax.fill_between(t, irf_lo68[:, k], irf_hi68[:, k],
                                        alpha=0.35, color='#6baed6',
                                        label='68% HDI')
                        ax.fill_between(t, irf_lo90[:, k], irf_hi90[:, k],
                                        alpha=0.15, color='#bdd7e7',
                                        label='90% HDI')
                    ax.axhline(0, color="grey", linewidth=0.6,
                               linestyle=":")
                    ax.set_title(variables[k], fontsize=10)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    if k // ncols == nrows - 1:
                        ax.set_xlabel("Quarter")

                # Add legend to first subplot
                if use_posterior:
                    axes[0, 0].legend(fontsize=7, loc='upper right')

                for k in range(n_vars, nrows * ncols):
                    axes[k // ncols, k % ncols].set_visible(False)

                fig.tight_layout(rect=[0, 0, 1, 0.94])
                fname = os.path.join(irf_dir, f"irf_{shk}.png")
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved {fname}")

        return results

    # ==================================================================
    # Policy analysis
    # ==================================================================

    def sacrifice_ratio(self, gap_variable='L_GDP_GAP',
                        inflation_variable='DLA_CPI',
                        target_variable='PI_TREND',
                        reduction_pp=1.0, horizon_years=30,
                        params=None, n_draws=None, plot=True,
                        convention='symmetric'):
        """Compute the sacrifice ratio (cost of disinflation).

        Level-form models: simulates a permanent reduction in the
        inflation target and measures the cumulative output cost,
        SR = |sum(gap_t)| / (reduction * H).

        Gap-form models (the QPM case): the target shock is neutral for
        the cyclical block, so the experiment falls back to a cost-push
        shock and SR becomes a ratio of cumulative responses.  The
        ``convention`` argument selects the placement of the absolute
        value ('symmetric', 'mixed', 'net'); see
        ``policy_analysis.sacrifice_ratio`` for the definitions and for
        why none of them is the permanent-disinflation ratio of Ball
        (1994) or Cecchetti and Rich (2001).  All conventions are
        evaluated on the same IRF and written to the draws CSV.

        When n_draws is set and posterior samples are available,
        computes the posterior distribution of the sacrifice ratio.

        Args:
            gap_variable:        output gap state name.
            inflation_variable:  inflation state or observable name.
            target_variable:     inflation target state name.
            reduction_pp:        target reduction in pp (positive).
            horizon_years:       simulation horizon in years.
            params:              parameter dict (default: resolved).
            n_draws:             number of posterior draws. None = point.
            plot:                save figure to results_dir/policy/.

        Returns:
            SacrificeRatioResult (with sr_draws attribute when posterior).
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        # Draws may live on disk rather than in memory (e.g. a re-run
        # with --skip-estimation --only-policy).  Without this the
        # posterior branch is silently skipped and the method degrades
        # to a single point estimate.
        if n_draws is not None:
            self._auto_load_draws()

        use_posterior = (n_draws is not None
                        and self._posterior_samples is not None)
        if n_draws is not None and not use_posterior:
            print("  WARNING: no posterior draws available; reporting a "
                  "point estimate only.")

        if use_posterior:
            total = self._posterior_samples[
                next(iter(self._posterior_samples))].shape[0]
            n_use = min(n_draws, total)
            step = max(1, total // n_use)
            indices = list(range(0, total, step))[:n_use]
            print(f"  Computing posterior sacrifice ratio ({n_use} draws)...")

            sr_draws = []
            gap_draws = []
            variant_draws = []
            n_skipped = 0

            for mcmc_idx in indices:
                pdict = dict(self.model.default_param_assignments)
                pdict.update(self._fixed_params or {})
                if self._calibrated_sigmas:
                    pdict.update(self._calibrated_sigmas)
                for name in self._estimated_names:
                    pdict[name] = self._posterior_samples[name][mcmc_idx]

                try:
                    r = _sacrifice_ratio(
                        model=self.model, param_dict=pdict,
                        gap_variable=gap_variable,
                        inflation_variable=inflation_variable,
                        target_variable=target_variable,
                        reduction_pp=reduction_pp,
                        horizon_years=horizon_years,
                        convention=convention,
                    )
                    sr_draws.append(r.sacrifice_ratio)
                    gap_draws.append(r.cum_output_gap)
                    variant_draws.append({
                        **{f"sr_{k}": v for k, v in r.sr_variants.items()},
                        'cum_output_gap': r.cum_output_gap,
                        'cum_abs_output_gap': r.cum_abs_output_gap,
                        'cum_inflation': r.cum_inflation,
                        'cum_abs_inflation': r.cum_abs_inflation,
                    })
                except Exception:
                    n_skipped += 1

            if n_skipped > 0:
                print(f"    Skipped {n_skipped}/{len(indices)} draws")

            sr_arr = np.array(sr_draws)
            print(f"\n  Sacrifice Ratio (posterior, {len(sr_draws)} draws,"
                  f" convention='{convention}'):")
            print(f"    Median: {np.median(sr_arr):.4f}  "
                  f"Mean: {np.mean(sr_arr):.4f}  "
                  f"Std: {np.std(sr_arr):.4f}")
            print(f"    HDI 68%: [{np.percentile(sr_arr, 16):.4f}, "
                  f"{np.percentile(sr_arr, 84):.4f}]")
            print(f"    HDI 90%: [{np.percentile(sr_arr, 5):.4f}, "
                  f"{np.percentile(sr_arr, 95):.4f}]")
            if variant_draws:
                vdf = pd.DataFrame(variant_draws)
                med = {c: float(np.median(vdf[c]))
                       for c in vdf.columns if c.startswith('sr_')}
                print("    Alternative sign conventions (posterior "
                      "medians): "
                      + ", ".join(f"{k[3:]}={v:.4f}"
                                  for k, v in med.items()))

            # Also compute the median-parameter result for plotting
            pdict_med = self._resolve_params(params)
            result = _sacrifice_ratio(
                model=self.model, param_dict=pdict_med,
                gap_variable=gap_variable,
                inflation_variable=inflation_variable,
                target_variable=target_variable,
                reduction_pp=reduction_pp,
                horizon_years=horizon_years,
                convention=convention,
            )

        else:
            pdict = self._resolve_params(params)
            result = _sacrifice_ratio(
                model=self.model, param_dict=pdict,
                gap_variable=gap_variable,
                inflation_variable=inflation_variable,
                target_variable=target_variable,
                reduction_pp=reduction_pp,
                horizon_years=horizon_years,
                convention=convention,
            )
            sr_arr = None

            print(f"\n  Sacrifice Ratio Analysis:")
            print(f"    Target reduction:     {result.reduction_pp:.1f} pp")
            print(f"    Horizon:              {result.horizon_quarters} quarters "
                  f"({horizon_years} years)")
            print(f"    Cumulative gap:       {result.cum_output_gap:.2f} pp-Q")
            print(f"    Sacrifice ratio:      {result.sacrifice_ratio:.4f}")

        if plot:
            policy_dir = os.path.join(self.results_dir, "policy")
            os.makedirs(policy_dir, exist_ok=True)

            fig = plot_sacrifice_ratio(result, show=False)
            fname = os.path.join(policy_dir, "sacrifice_ratio.png")
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved {fname}")

            # Posterior histogram
            if sr_arr is not None:
                fig2, ax = plt.subplots(figsize=(5, 3.5))
                ax.hist(sr_arr, bins=30, density=True, alpha=0.7,
                        color='#2171b5', edgecolor='white')
                ax.axvline(np.median(sr_arr), color='#cb181d',
                           linewidth=1.5, label=f'Median={np.median(sr_arr):.3f}')
                ax.axvspan(np.percentile(sr_arr, 16),
                           np.percentile(sr_arr, 84),
                           alpha=0.15, color='#cb181d', label='68% HDI')
                ax.set_xlabel('Sacrifice Ratio')
                ax.set_ylabel('Density')
                ax.set_title('Posterior Distribution of Sacrifice Ratio')
                ax.legend(fontsize=8)
                ax.grid(True, linestyle='--', alpha=0.4)
                fig2.tight_layout()
                fname2 = os.path.join(policy_dir, "sacrifice_ratio_posterior.png")
                fig2.savefig(fname2, dpi=150, bbox_inches='tight')
                plt.close(fig2)
                print(f"    Saved {fname2}")

                # Save draws to CSV
                # The headline column keeps its name for backward
                # compatibility; the convention that produced it and the
                # alternatives are written alongside it so the sign
                # convention can be audited without re-running.
                draws_df = pd.DataFrame({'sacrifice_ratio': sr_arr})
                if variant_draws:
                    draws_df = pd.concat(
                        [draws_df, pd.DataFrame(variant_draws)], axis=1)
                draws_df['convention'] = convention
                draws_df.to_csv(
                    os.path.join(policy_dir, "sacrifice_ratio_draws.csv"),
                    index=False)

        # Return a simple namespace with sr_draws attached
        class _SRResult:
            pass
        out = _SRResult()
        out.sacrifice_ratio = result.sacrifice_ratio
        out.cum_output_gap = result.cum_output_gap
        out.reduction_pp = result.reduction_pp
        out.horizon_quarters = result.horizon_quarters
        out.gap_path = result.gap_path
        out.inflation_path = result.inflation_path
        out.target_path = result.target_path
        out.params_used = result.params_used
        out.sr_draws = sr_arr
        out.convention = result.convention
        out.sr_variants = result.sr_variants
        return out

    def pass_through(self, shocks, inflation_variables,
                     er_variable, horizon=12, shock_size=1.0,
                     normalize=True, params=None, n_draws=None,
                     plot=True):
        """Compute exchange rate pass-through coefficients.

        For each shock, measures how a 1 pp exchange rate depreciation
        passes through to inflation at different horizons.

        Reports: contemporaneous, peak, cumulative PT.

        When n_draws is set and posterior samples are available,
        computes the posterior distribution of PT coefficients.

        Args:
            shocks:               list of shock names that move the ER.
            inflation_variables:  list of CPI variable names.
            er_variable:          exchange rate depreciation variable.
            horizon:              quarters for cumulative PT.
            shock_size:           initial shock size (std dev).
            normalize:            rescale to 1 pp ER depreciation.
            params:               parameter dict (default: resolved).
            n_draws:              number of posterior draws. None = point.
            plot:                 save figure to results_dir/policy/.

        Returns:
            PassThroughResult (with pt_draws attribute when posterior).
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        # See the note in sacrifice_ratio: draws may be on disk only.
        if n_draws is not None:
            self._auto_load_draws()

        use_posterior = (n_draws is not None
                        and self._posterior_samples is not None)
        if n_draws is not None and not use_posterior:
            print("  WARNING: no posterior draws available; reporting a "
                  "point estimate only.")

        if use_posterior:
            total = self._posterior_samples[
                next(iter(self._posterior_samples))].shape[0]
            n_use = min(n_draws, total)
            step = max(1, total // n_use)
            indices = list(range(0, total, step))[:n_use]
            print(f"  Computing posterior pass-through ({n_use} draws)...")

            # Collect draws: {shock: {cpi_var: {metric: [values]}}}
            pt_draws = {}
            # Horizon-level cumulative PT paths per draw
            pt_horizon_draws = {}
            for shk_name in shocks:
                pt_draws[shk_name] = {}
                pt_horizon_draws[shk_name] = {}
                for cv in inflation_variables:
                    pt_draws[shk_name][cv] = {
                        'contemporaneous': [], 'peak': [],
                        'peak_quarter': [], 'cumulative': [],
                        'elasticity': [], 'elasticity_peak': [],
                        'elasticity_peak_quarter': []}
                    pt_horizon_draws[shk_name][cv] = []

            n_skipped = 0
            for mcmc_idx in indices:
                pdict = dict(self.model.default_param_assignments)
                pdict.update(self._fixed_params or {})
                if self._calibrated_sigmas:
                    pdict.update(self._calibrated_sigmas)
                for name in self._estimated_names:
                    pdict[name] = self._posterior_samples[name][mcmc_idx]

                try:
                    r = _pass_through(
                        model=self.model, param_dict=pdict,
                        shocks=shocks,
                        inflation_variables=inflation_variables,
                        er_variable=er_variable,
                        horizon=horizon, shock_size=shock_size,
                        normalize=normalize,
                    )
                    for shk_name in shocks:
                        er_path = r.irf_paths[shk_name][er_variable]
                        for cv in inflation_variables:
                            for metric in ('contemporaneous', 'peak',
                                           'peak_quarter', 'cumulative',
                                           'elasticity', 'elasticity_peak',
                                           'elasticity_peak_quarter'):
                                pt_draws[shk_name][cv][metric].append(
                                    r.table[shk_name][cv][metric])
                            # Horizon-level cumulative CPI response
                            # (per 1pp initial depreciation, since
                            # shock is normalized to er(0)=1)
                            cpi_path = r.irf_paths[shk_name][cv]
                            cum_cpi = np.cumsum(cpi_path)
                            pt_horizon_draws[shk_name][cv].append(
                                cum_cpi)
                except Exception:
                    n_skipped += 1

            if n_skipped > 0:
                print(f"    Skipped {n_skipped}/{len(indices)} draws")

            # Print posterior summary
            print(f"\n  Pass-Through Posterior (H={horizon}Q, "
                  f"{len(indices) - n_skipped} draws):")
            print(f"  {'Shock':25s} {'CPI Variable':20s} "
                  f"{'Median':>8s} {'HDI16':>8s} {'HDI84':>8s} "
                  f"{'Elast':>8s} {'E-HDI16':>8s} {'E-HDI84':>8s}")
            print("  " + "-" * 95)
            for shk_name in shocks:
                for cv in inflation_variables:
                    vals = np.array(
                        pt_draws[shk_name][cv]['cumulative'])
                    els = np.array(
                        pt_draws[shk_name][cv]['elasticity'])
                    print(f"  {shk_name:25s} {cv:20s} "
                          f"{np.median(vals):8.3f} "
                          f"{np.percentile(vals, 16):8.3f} "
                          f"{np.percentile(vals, 84):8.3f} "
                          f"{np.median(els):8.3f} "
                          f"{np.percentile(els, 16):8.3f} "
                          f"{np.percentile(els, 84):8.3f}")
            print("  Median/HDI columns: dP(h)/dS(h), the legacy ratio "
                  "of cumulative sums.  Elast columns: dP(h)/dS(0), the "
                  "impact-normalised elasticity.")
        else:
            pt_draws = None

        # Compute point estimate for table/plots
        pdict = self._resolve_params(params)
        result = _pass_through(
            model=self.model, param_dict=pdict,
            shocks=shocks,
            inflation_variables=inflation_variables,
            er_variable=er_variable,
            horizon=horizon, shock_size=shock_size,
            normalize=normalize,
        )

        if not use_posterior:
            # Print point-estimate table
            print(f"\n  Exchange Rate Pass-Through (H={horizon}Q):")
            print(f"  {'Shock':25s} {'CPI Variable':20s} "
                  f"{'Contemp':>8s} {'Peak':>8s} {'t_peak':>6s} "
                  f"{'Cumul':>8s}")
            print("  " + "-" * 75)
            for shk in result.table:
                for cpi_var, met in result.table[shk].items():
                    print(f"  {shk:25s} {cpi_var:20s} "
                          f"{met['contemporaneous']:8.3f} "
                          f"{met['peak']:8.3f} "
                          f"{met['peak_quarter']:6d} "
                          f"{met['cumulative']:8.3f}")

        if plot:
            policy_dir = os.path.join(self.results_dir, "policy")
            os.makedirs(policy_dir, exist_ok=True)

            fig = plot_pass_through(result, show=False)
            fname = os.path.join(policy_dir, "pass_through.png")
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {fname}")

            # Dynamic horizon plot: cumulative PT path with bands
            if pt_draws is not None:
                for shk_name in shocks:
                    fig2, axes = plt.subplots(
                        1, len(inflation_variables),
                        figsize=(4 * len(inflation_variables), 3.5))
                    if len(inflation_variables) == 1:
                        axes = [axes]
                    for i, cv in enumerate(inflation_variables):
                        arr = np.array(
                            pt_horizon_draws[shk_name][cv])  # (n_draws, H)
                        horizons = np.arange(1, arr.shape[1] + 1)
                        med = np.nanmedian(arr, axis=0)
                        p16 = np.nanpercentile(arr, 16, axis=0)
                        p84 = np.nanpercentile(arr, 84, axis=0)
                        p05 = np.nanpercentile(arr, 5, axis=0)
                        p95 = np.nanpercentile(arr, 95, axis=0)
                        axes[i].fill_between(
                            horizons, p05, p95,
                            alpha=0.15, color='#2171b5', label='90% CI')
                        axes[i].fill_between(
                            horizons, p16, p84,
                            alpha=0.3, color='#2171b5', label='68% CI')
                        axes[i].plot(
                            horizons, med, color='#cb181d',
                            linewidth=1.5, label=f'Median')
                        axes[i].set_title(cv, fontsize=10)
                        axes[i].set_xlabel('Horizon (quarters)')
                        if i == 0:
                            axes[i].set_ylabel(
                                'Cumul. CPI response (pp)')
                        axes[i].legend(fontsize=6, loc='lower right')
                        axes[i].grid(True, linestyle='--', alpha=0.4)
                    fig2.suptitle(
                        f'Cumulative ERPT by Horizon -- {shk_name}',
                        fontsize=12)
                    fig2.tight_layout(rect=[0, 0, 1, 0.93])
                    fname2 = os.path.join(
                        policy_dir,
                        f"pass_through_posterior_{shk_name}.png")
                    fig2.savefig(fname2, dpi=150, bbox_inches='tight')
                    plt.close(fig2)
                    print(f"  Saved {fname2}")

                # Save terminal draws to CSV (backward compat)
                rows = []
                for shk_name in shocks:
                    for cv in inflation_variables:
                        vals = pt_draws[shk_name][cv]['cumulative']
                        els = pt_draws[shk_name][cv]['elasticity']
                        elp = pt_draws[shk_name][cv]['elasticity_peak']
                        elq = pt_draws[shk_name][cv][
                            'elasticity_peak_quarter']
                        for i, v in enumerate(vals):
                            rows.append({
                                'shock': shk_name, 'cpi_var': cv,
                                'cumulative_pt': v,
                                'elasticity': els[i],
                                'elasticity_peak': elp[i],
                                'elasticity_peak_quarter': elq[i]})
                pd.DataFrame(rows).to_csv(
                    os.path.join(policy_dir,
                                 "pass_through_draws.csv"),
                    index=False)

                # Save horizon-level draws to npz (compact)
                pt_h_dict = {}
                for shk_name in shocks:
                    for cv in inflation_variables:
                        key = f"{shk_name}__{cv}"
                        pt_h_dict[key] = np.array(
                            pt_horizon_draws[shk_name][cv])
                np.savez_compressed(
                    os.path.join(policy_dir,
                                 "pass_through_horizon_draws.npz"),
                    **pt_h_dict)

        # Return with pt_draws attached
        class _PTResult:
            pass
        out = _PTResult()
        out.table = result.table
        out.irf_paths = result.irf_paths
        out.er_variable = result.er_variable
        out.horizon = result.horizon
        out.params_used = result.params_used
        out.pt_draws = pt_draws
        return out

    def forecast_evaluation(self, variables, first_origin, last_origin,
                            horizons=None, max_horizon=8,
                            stochastic=False, params=None,
                            plot=True, seed=42,
                            use_diffuse: bool = True,
                            realised_reporting=None,
                            covid_window=None):
        """Pseudo real-time rolling-window forecast evaluation.

        At each forecast origin, runs the Kalman smoother on data up to
        that date, then forecasts forward.  Computes RMSE, MAE, and bias
        against actual data.

        Uses fixed parameters (no re-estimation per window), following
        standard QPM practice (Herbst & Schorfheide 2016; Berg et al. 2006).

        Args:
            variables:          names of observables or reporting variables
                                to evaluate.  Reporting variables are
                                forecast as linear combinations of the
                                augmented state forecast.
            first_origin:       first forecast origin date (string).
            last_origin:        last forecast origin date (string).
            horizons:           list of horizons in quarters (default:
                                1..max_horizon).
            max_horizon:        maximum forecast horizon.
            stochastic:         if True, draw random future shocks.
            params:             parameter dict (default: resolved).
            plot:               save spaghetti + RMSE plots.
            seed:               RNG seed.
            realised_reporting: optional ``{var_name: array}`` mapping
                                supplying the realised series for any
                                reporting variable in ``variables``.  Used
                                only when computing RMSE/MAE/bias against
                                a reporting-variable forecast; observables
                                are scored against ``ys`` directly.

        Returns:
            ForecastEvaluationResult.
        """
        self._check_data()
        _, ys_np, dates, obs_names = self._data
        ys_jax, H_obs, H_t, mult = self._resolve_ys_H_multipliers()

        pdict = self._resolve_params(params)

        result = _forecast_evaluation(
            model=self.model,
            param_dict=pdict,
            ys=ys_np,
            dates=dates,
            obs_names=obs_names,
            H_obs=H_obs,
            init_x=self._init_x,
            init_P=self._init_P,
            variables=variables,
            first_origin=first_origin,
            last_origin=last_origin,
            horizons=horizons,
            max_horizon=max_horizon,
            stochastic=stochastic,
            seed=seed,
            H_t=H_t if H_t is not None else None,
            shock_multipliers=mult,
            use_diffuse=use_diffuse,
            realised_reporting=realised_reporting,
            covid_window=covid_window,
        )

        # Canonical forecast-RMSE file: full-sample and COVID-excluded
        # RMSE for every evaluated variable (incl. reporting variables
        # such as DA_LGDP_ANN) in a single traceable CSV.  This is the
        # source the paper's Section 6 tables are generated from.
        if result.rmse_full is not None:
            import pandas as pd
            eval_dir = os.path.join(self.results_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            xc_rows = []
            for v in result.variables:
                for h in result.horizons:
                    xc_rows.append({
                        'variable': v, 'horizon': h,
                        'rmse_full': result.rmse_full[v][h],
                        'rmse_excovid': result.rmse_excovid[v][h],
                        'n_full': result.n_full[v][h],
                        'n_excovid': result.n_excovid[v][h],
                    })
            xc_path = os.path.join(eval_dir, "forecast_rmse_full_excovid.csv")
            pd.DataFrame(xc_rows).to_csv(xc_path, index=False)
            print(f"  Saved {xc_path}")

        if plot:
            import matplotlib.pyplot as plt
            eval_dir = os.path.join(self.results_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)

            # Spaghetti plot
            fig_sp = plot_forecast_spaghetti(
                result, ys_np, dates, obs_names, show=False)
            fname_sp = os.path.join(eval_dir, "forecast_spaghetti.png")
            fig_sp.savefig(fname_sp, dpi=150, bbox_inches='tight')
            plt.close(fig_sp)
            print(f"  Saved {fname_sp}")

            # RMSE bar chart
            fig_rmse = plot_rmse_bars(result, show=False)
            fname_rmse = os.path.join(eval_dir, "rmse_bars.png")
            fig_rmse.savefig(fname_rmse, dpi=150, bbox_inches='tight')
            plt.close(fig_rmse)
            print(f"  Saved {fname_rmse}")

            # Save RMSE table to CSV
            import pandas as pd
            rows = []
            for v in result.variables:
                for h in result.horizons:
                    rows.append({
                        'variable': v,
                        'horizon': h,
                        'rmse': result.rmse[v][h],
                        'mae': result.mae[v][h],
                        'bias': result.bias[v][h],
                    })
            df = pd.DataFrame(rows)
            csv_path = os.path.join(eval_dir, "forecast_rmse.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path}")

        return result

    def conditional_forecast_evaluation(
        self, variables, condition_on, first_origin, last_origin,
        horizons=None, max_horizon=8, params=None,
        plot=True, seed=42,
        use_diffuse: bool = True,
        realised_reporting=None,
        covid_window=None,
    ):
        """Conditional rolling-window forecast evaluation.

        Like ``forecast_evaluation``, but at each origin the forecast
        conditions on actual future paths of specified variables
        (typically external/exogenous variables and component inflations).

        This mirrors the Matlab ``app_conditional_forecast_evaluation.m``.

        Args:
            variables:    observable names to evaluate (forecast targets).
            condition_on: list of (kind, variable, shock) tuples.
                          Example::

                              [('obs', 'OBS_LRWOIL_GAP', 'SHK_LRWOIL_GAP'),
                               ('obs', 'OBS_RS_US', 'SHK_RS_US'), ...]

            first_origin: first forecast origin date (string).
            last_origin:  last forecast origin date (string).
            horizons:     list of horizons in quarters.
            max_horizon:  maximum forecast horizon.
            params:       parameter dict (default: resolved).
            plot:         save spaghetti + RMSE plots.
            seed:         RNG seed.

        Returns:
            ForecastEvaluationResult.
        """
        self._check_data()
        _, ys_np, dates, obs_names = self._data
        ys_jax, H_obs, H_t, mult = self._resolve_ys_H_multipliers()

        pdict = self._resolve_params(params)

        result = _conditional_forecast_evaluation(
            model=self.model,
            param_dict=pdict,
            ys=ys_np,
            dates=dates,
            obs_names=obs_names,
            H_obs=H_obs,
            init_x=self._init_x,
            init_P=self._init_P,
            variables=variables,
            condition_on=condition_on,
            first_origin=first_origin,
            last_origin=last_origin,
            horizons=horizons,
            max_horizon=max_horizon,
            seed=seed,
            H_t=H_t if H_t is not None else None,
            shock_multipliers=mult,
            use_diffuse=use_diffuse,
            realised_reporting=realised_reporting,
            covid_window=covid_window,
        )

        # Canonical conditional forecast-RMSE file (full + COVID-excluded).
        if result.rmse_full is not None:
            import pandas as pd
            eval_dir = os.path.join(self.results_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            xc_rows = []
            for v in result.variables:
                for h in result.horizons:
                    xc_rows.append({
                        'variable': v, 'horizon': h,
                        'rmse_full': result.rmse_full[v][h],
                        'rmse_excovid': result.rmse_excovid[v][h],
                        'n_full': result.n_full[v][h],
                        'n_excovid': result.n_excovid[v][h],
                    })
            xc_path = os.path.join(
                eval_dir, "conditional_forecast_rmse_full_excovid.csv")
            pd.DataFrame(xc_rows).to_csv(xc_path, index=False)
            print(f"  Saved {xc_path}")

        if plot:
            import matplotlib.pyplot as plt
            eval_dir = os.path.join(self.results_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)

            # Spaghetti plot
            fig_sp = plot_forecast_spaghetti(
                result, ys_np, dates, obs_names, show=False)
            fig_sp.suptitle(
                "Conditional Out-of-Sample Rolling Forecast Paths",
                fontsize=13)
            fname_sp = os.path.join(
                eval_dir, "conditional_forecast_spaghetti.png")
            fig_sp.savefig(fname_sp, dpi=150, bbox_inches='tight')
            plt.close(fig_sp)
            print(f"  Saved {fname_sp}")

            # RMSE bar chart
            fig_rmse = plot_rmse_bars(result, show=False)
            fig_rmse.suptitle("Conditional Forecast RMSE", fontsize=13)
            fname_rmse = os.path.join(
                eval_dir, "conditional_rmse_bars.png")
            fig_rmse.savefig(fname_rmse, dpi=150, bbox_inches='tight')
            plt.close(fig_rmse)
            print(f"  Saved {fname_rmse}")

            # Save RMSE table to CSV
            import pandas as pd
            rows = []
            for v in result.variables:
                for h in result.horizons:
                    rows.append({
                        'variable': v,
                        'horizon': h,
                        'rmse': result.rmse[v][h],
                        'mae': result.mae[v][h],
                        'bias': result.bias[v][h],
                    })
            df = pd.DataFrame(rows)
            csv_path = os.path.join(
                eval_dir, "conditional_forecast_rmse.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path}")

        return result

    # ==================================================================
    # Forecast
    # ==================================================================

    def forecast(self, horizon=20, x0=None, stochastic=True,
                 n_draws=None, use_dk=True, variables=None,
                 plot=True, seed=None, batch_size=128, use_diffuse=True):
        """Unconditional forecast with posterior fan charts.

        Args:
            horizon:    number of forecast periods.
            x0:         initial state (default: terminal smoothed state).
            stochastic: include future shock uncertainty.
            n_draws:    number of posterior draws (None = all).
            use_dk:     use DK simulation smoother for terminal state.
            variables:  list of obs/state names to plot.
            plot:       save fan charts to results_dir/forecast/.
            seed:       random seed (default: 123).
            use_diffuse: if True (default), recover the terminal smoothed
                        state via the exact diffuse Kalman smoother.
                        Required for models with unit-root states.
                        Set to False for the standard KF (legacy path).

        Returns:
            ForecastResult.
        """
        self._check_data()
        self._auto_load_draws()
        _, ys_np, dates, obs_names = self._data
        ys_jax, H_obs, H_t, mult = self._resolve_ys_H_multipliers()

        # The diffuse smoother does not yet support time-varying H_t /
        # shock multipliers; fall back to the standard KF in that case.
        diffuse_active = use_diffuse and (H_t is None and mult is None)
        if use_diffuse and not diffuse_active:
            print("  forecast: use_diffuse=True overridden because H_t or "
                  "shock_multipliers is non-trivial; falling back to "
                  "standard KF.")

        if self._posterior_samples is not None:
            # Posterior forecast
            key = jax.random.PRNGKey(seed or 123)
            result = forecast_posterior(
                model=self.model,
                posterior_samples=self._posterior_samples,
                estimated_names=self._estimated_names,
                fixed_params=self._fixed_params or {},
                ys=ys_jax,
                H_obs=H_obs,
                init_x=self._init_x,
                init_P=self._init_P,
                horizon=horizon,
                key=key,
                n_draws=n_draws,
                use_dk=use_dk,
                stochastic=stochastic,
                batch_size=batch_size,
                H_t=H_t,
                shock_multipliers=mult,
                use_diffuse=diffuse_active,
            )
        else:
            # Single-draw forecast at current params
            pdict = self._resolve_params(None)
            sol = self.model.solve(pdict)
            if not sol["solution_valid"]:
                raise RuntimeError("Model does not solve.")

            dt = get_dtype()

            if x0 is None:
                if diffuse_active:
                    Pinf, Pstar, _ = build_diffuse_init(
                        sol["P_aug"], sol["R_aug"], dtype=dt)
                    dkf = DiffuseKalmanFilter(
                        T=sol["P_aug"], R=sol["R_aug"], C=sol["Omega"],
                        H=H_obs, Pinf=Pinf, Pstar=Pstar,
                        init_a=self._init_x, c=sol["c_aug"], dtype=dt)
                    x_smooth = dkf.smooth_states_only(ys=ys_jax)
                else:
                    kf = KalmanFilter(
                        T=sol["P_aug"], R=sol["R_aug"], C=sol["Omega"],
                        H=H_obs, init_x=self._init_x, init_P=self._init_P,
                        c=sol["c_aug"], dtype=dt)
                    x_smooth = kf.smooth_states_only(ys=ys_jax, H_t=H_t)
                x0 = x_smooth[-1]

            key = jax.random.PRNGKey(seed or 123)
            x_fcst, y_fcst = forecast_unconditional(
                sol["P_aug"], sol["R_aug"], sol["Omega"],
                x0, horizon, key, c=sol["c_aug"], stochastic=stochastic)

            result = ForecastResult(
                state_forecasts=np.asarray(x_fcst)[None, ...],
                obs_forecasts=np.asarray(y_fcst)[None, ...],
                state_names=list(self.model.aug_state_vars),
                obs_names=list(self.model.ordered_obs_vars),
                shock_names=list(self.model.aug_shocks),
                horizon=horizon,
            )

        if plot:
            self._plot_forecast(result, variables, dates, ys_np, obs_names)

        return result

    # ==================================================================
    # Conditional forecast
    # ==================================================================

    def conditional_forecast(self, plan, variables=None, params=None,
                             n_draws=None, use_dk=False, n_tv=None,
                             plot=True, label=None, seed=None,
                             batch_size=128, use_diffuse=True):
        """Conditional forecast (Waggoner & Zha 1999).

        Imposes target paths for selected observables or states and
        solves for the structural shocks that achieve those targets.
        If posterior samples are available, produces fan charts across
        draws; otherwise uses a single fixed-parameter forecast.

        Args:
            plan:       ForecastPlan with conditions.
            variables:  obs/state names to plot (None = conditioned vars + 4).
            params:     parameter dict (default: posterior median or calibrated).
            n_draws:    number of posterior draws (None = all).
            use_dk:     use DK simulation smoother for terminal state.
            n_tv:       time-varying filter steps.
            plot:       save fan chart and implied shock plots.
            label:      scenario label for file naming (e.g. 'rate_hold').
            seed:       random seed.
            use_diffuse: if True (default), recover x_T via the exact
                        diffuse Kalman smoother.  Required for unit-root
                        models.  Auto-falls back to standard KF if H_t /
                        shock_multipliers / n_tv are non-trivial.

        Returns:
            ConditionalForecastResult.
        """
        self._check_data()
        self._auto_load_draws()
        _, ys_np, dates, obs_names = self._data
        ys_jax, H_obs, H_t, mult = self._resolve_ys_H_multipliers()
        pdict = self._resolve_params(params)

        diffuse_active = use_diffuse and (
            H_t is None and mult is None and n_tv is None)
        if use_diffuse and not diffuse_active:
            print("  conditional_forecast: use_diffuse=True overridden "
                  "because H_t / shock_multipliers / n_tv is non-trivial; "
                  "falling back to standard KF.")

        if self._posterior_samples is not None:
            key = jax.random.PRNGKey(seed or 456)
            result = forecast_conditional_posterior(
                model=self.model,
                posterior_samples=self._posterior_samples,
                estimated_names=self._estimated_names,
                fixed_params=self._fixed_params or {},
                ys=ys_jax,
                H_obs=H_obs,
                init_x=self._init_x,
                init_P=self._init_P,
                plan=plan,
                key=key,
                n_draws=n_draws,
                use_dk=use_dk,
                n_tv=n_tv,
                batch_size=batch_size,
                H_t=H_t,
                shock_multipliers=mult,
                use_diffuse=diffuse_active,
            )
        else:
            result = forecast_conditional(
                model=self.model,
                param_dict=pdict,
                ys=ys_jax,
                H_obs=H_obs,
                init_x=self._init_x,
                init_P=self._init_P,
                plan=plan,
                n_tv=n_tv,
                H_t=H_t,
                shock_multipliers=mult,
                use_diffuse=diffuse_active,
            )

        # Verify condition satisfaction
        from qpm_toolbox.conditional_forecast import (
            get_cond_obs_forecast as _get_cond_obs,
            get_cond_state_forecast as _get_cond_state,
        )
        print(f"\n  Condition verification ({label or 'conditional forecast'}):")
        all_ok = True
        for cond in plan.conditions:
            for pi, period in enumerate(cond.periods):
                target = cond.values[pi]
                if cond.kind == 'obs':
                    draws = _get_cond_obs(result, cond.variable)
                else:
                    draws = _get_cond_state(result, cond.variable)
                max_err = np.max(np.abs(draws[:, period] - target))
                if max_err > 1e-8:
                    all_ok = False
                # Print first and last period only
                if pi == 0 or pi == len(cond.periods) - 1:
                    status = "OK" if max_err < 1e-8 else "FAIL"
                    print(f"    {cond.variable:20s} h={period:2d}  "
                          f"target={target:8.4f}  max|err|={max_err:.2e}  [{status}]")
        if all_ok:
            print(f"    All conditions satisfied (tol=1e-8).")

        if plot:
            self._plot_conditional_forecast(
                result, plan, variables, dates, ys_np, obs_names, label)

        return result

    def _plot_conditional_forecast(self, result, plan, variables,
                                   dates, ys_np, obs_names, label):
        """Plot conditional forecast fan charts and implied shocks."""
        import matplotlib.pyplot as plt
        import pandas as pd

        cond_dir = os.path.join(self.results_dir, "conditional_forecast")
        os.makedirs(cond_dir, exist_ok=True)

        suffix = f"_{label}" if label else ""

        # Build forecast date axis
        if dates is not None:
            dates = _ensure_datetime_index(dates)
            last_date = dates[-1]
            fcst_dates = pd.date_range(
                start=last_date + pd.tseries.offsets.QuarterEnd(1),
                periods=result.horizon, freq='QS')
            fcst_dates_arr = fcst_dates.to_pydatetime()
            hist_dates_arr = dates.to_pydatetime()
        else:
            fcst_dates_arr = None
            hist_dates_arr = None

        # Auto-select variables: conditioned variables + up to 4 others
        if variables is None:
            cond_vars = list(dict.fromkeys(
                c.variable for c in plan.conditions))
            other_obs = [v for v in result.obs_names
                         if v not in cond_vars][:4 - min(len(cond_vars), 2)]
            variables = cond_vars + other_obs

        # Determine kind per variable
        obs_set = set(result.obs_names)
        state_set = set(result.state_names)
        obs_vars = [v for v in variables if v in obs_set]
        state_vars = [v for v in variables if v in state_set and v not in obs_set]

        n_hist_show = min(16, ys_np.shape[0])

        # Plot observables
        if obs_vars:
            fig, _ = plot_conditional_forecast(
                result, obs_vars, kind='obs',
                hist_data=ys_np[-n_hist_show:],
                hist_dates=hist_dates_arr[-n_hist_show:] if hist_dates_arr is not None else None,
                fcst_dates=fcst_dates_arr,
                title=label or "Conditional Forecast",
                show=False)
            fname = os.path.join(cond_dir, f"cond_forecast{suffix}.png")
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {fname}")

        # Plot state variables if any
        if state_vars:
            fig_s, _ = plot_conditional_forecast(
                result, state_vars, kind='state',
                fcst_dates=fcst_dates_arr,
                title=f"{label or 'Conditional Forecast'} (states)",
                show=False)
            fname_s = os.path.join(cond_dir, f"cond_forecast_states{suffix}.png")
            fig_s.savefig(fname_s, dpi=150, bbox_inches='tight')
            plt.close(fig_s)
            print(f"  Saved {fname_s}")

        # Plot implied shocks
        fig_shk, _ = plot_conditioning_shocks(
            result, fcst_dates=fcst_dates_arr,
            title=f"{label or 'Scenario'}: Implied Shocks",
            show=False)
        if fig_shk is not None:
            fname_shk = os.path.join(cond_dir, f"cond_shocks{suffix}.png")
            fig_shk.savefig(fname_shk, dpi=150, bbox_inches='tight')
            plt.close(fig_shk)
            print(f"  Saved {fname_shk}")

    # ==================================================================
    # Decomposition
    # ==================================================================

    def decomposition(self, variables=None, shock_groups=None,
                      params=None, n_draws=None, plot=True,
                      batch_size=128, include_baseline=True,
                      baseline_group=None, ylim=None,
                      use_diffuse=True):
        """Historical shock decomposition.

        When posterior samples are available and ``n_draws`` is set,
        runs the decomposition across posterior draws (integrating
        parameter uncertainty).  Otherwise uses a single parameter
        set (posterior median by default).

        Args:
            variables:    list of state/obs variable names to decompose.
            shock_groups: dict of {group_name: [shock_names]} for plotting.
            params:       parameter dict (default: posterior median).
            n_draws:      if set and posterior available, use this many
                          posterior draws for the decomposition.
            plot:         save stacked bar charts.
            batch_size:   mini-batch size for the vmapped posterior path.

        Returns:
            ShockDecompositionResult or PosteriorDecompositionResult.
        """
        self._check_data()
        if n_draws is not None:
            self._auto_load_draws()
        _, ys_np, dates, obs_names = self._data
        ys_jax, H_obs, H_t, mult = self._resolve_ys_H_multipliers()

        if n_draws is not None and self._posterior_samples is not None:
            result = shock_decomposition_posterior(
                model=self.model,
                posterior_samples=self._posterior_samples,
                estimated_names=self._estimated_names,
                fixed_params=self._fixed_params or {},
                ys=ys_jax,
                H_obs=H_obs,
                init_x=self._init_x,
                init_P=self._init_P,
                n_draws=n_draws,
                batch_size=batch_size,
                H_t=H_t,
                shock_multipliers=mult,
            )
        else:
            pdict = self._resolve_params(params)
            result = shock_decomposition(
                model=self.model,
                param_dict=pdict,
                ys=ys_jax,
                H_obs=H_obs,
                init_x=self._init_x,
                init_P=self._init_P,
                H_t=H_t,
                shock_multipliers=mult,
                use_diffuse=use_diffuse,
            )

        # ------------------------------------------------------------------
        # Augment the result with reporting variable contributions.
        # Reporting variables are linear combinations of augmented states
        # (possibly with lags) defined in the reporting_equations; block
        # of the .qpm file.  They are not part of the Kalman filter.
        # ------------------------------------------------------------------
        if self.model.reporting_vars:
            result = _append_reporting_vars(result, self.model.reporting_vars,
                                            self.model.reporting_eqs,
                                            init_x=self._init_x)

        if plot and variables and shock_groups:
            import matplotlib.pyplot as plt
            decomp_dir = os.path.join(self.results_dir, "decomposition")
            os.makedirs(decomp_dir, exist_ok=True)

            dates_dt = _ensure_datetime_index(dates)
            time_axis = dates_dt.to_pydatetime() if dates_dt is not None else None

            # Determine kind for each variable (state, obs, or reporting).
            state_vars = set(self.model.aug_state_vars)
            obs_vars_set = set(self.model.ordered_obs_vars)
            reporting_set = set(self.model.reporting_vars)

            for v in variables:
                if v in state_vars or v in reporting_set:
                    kind = 'state'
                elif v in obs_vars_set:
                    kind = 'obs'
                else:
                    print(f"  Warning: '{v}' not found, skipping.")
                    continue

                # Per-variable ylim: accept dict {var: (lo, hi)} or tuple
                var_ylim = None
                if ylim is not None:
                    if isinstance(ylim, dict):
                        var_ylim = ylim.get(v)
                    else:
                        var_ylim = ylim

                fig, _ = plot_shock_decomposition(
                    result, [v], shock_groups,
                    kind=kind, time_axis=time_axis,
                    include_baseline=include_baseline,
                    baseline_group=baseline_group,
                    ylim=var_ylim,
                    show=False)
                fname = os.path.join(decomp_dir, f"decomp_{v}.png")
                fig.savefig(fname, dpi=120, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved {fname}")

        return result

    # ==================================================================
    # Filter diagnostic
    # ==================================================================

    def filter_diagnostic(self, params=None, obs_trend_map=None,
                          use_diffuse: bool = True):
        """Pre-estimation filter diagnostic.

        Runs the Kalman smoother at fixed parameters and plots:
          1. Observables vs data with trend overlays
          2. Key stationary states

        Args:
            params:        parameter dict (default: prior modes + calibrated).
            obs_trend_map: dict of {obs_name: trend_state_name} for overlays.
            use_diffuse:   if ``True`` (default), use the exact diffuse
                           Kalman smoother (Durbin & Koopman 2012) with
                           ``Pinf``/``Pstar`` derived from the model's
                           eigenstructure.  Required for unit-root
                           models so the trend states are correctly
                           identified at the sample boundaries.  Set
                           ``False`` to fall back to the standard filter
                           with the user's ``init_P`` (legacy behaviour).
                           Defaulting to True keeps this method's
                           smoothed states consistent with
                           :meth:`decomposition` / :func:`shock_decomposition`,
                           which also default to the diffuse path.

        Returns:
            dict with 'x_smooth', 'obs_smooth', 'll'.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        self._check_data()
        _, ys_np, dates, obs_names = self._data
        ys_jax, H_obs, H_t, mult = self._resolve_ys_H_multipliers()
        pdict = self._resolve_params(params)

        dt = get_dtype()
        sol = self.model.solve(pdict)
        if not sol["solution_valid"]:
            print("  Model does NOT solve with these parameters!")
            return {}

        P_aug = sol["P_aug"]
        R_aug = sol["R_aug"]
        Omega = sol["Omega"]
        c_aug = sol["c_aug"]

        # Build per-period Q_t from shock multipliers if set
        Q_t = None
        if mult is not None:
            mult_jax = jnp.asarray(mult, dtype=dt)
            R_jax = jnp.asarray(R_aug, dtype=dt)
            R_scaled = R_jax[None, :, :] * mult_jax[:, None, :]
            Q_t = jnp.einsum('tij,tkj->tik', R_scaled, R_scaled)

        # Diffuse path requires no time-varying H_t / Q_t (not yet
        # supported in DiffuseKalmanFilter).  Auto-fall back to the
        # standard filter in that case so callers don't have to.
        _use_diffuse_eff = use_diffuse and (H_t is None) and (Q_t is None)
        if _use_diffuse_eff:
            from qpm_toolbox.kalman_filter import (
                DiffuseKalmanFilter, build_diffuse_init)
            Pinf, Pstar, _ = build_diffuse_init(P_aug, R_aug, dtype=dt)
            kf = DiffuseKalmanFilter(
                T=P_aug, R=R_aug, C=Omega, H=H_obs,
                Pinf=Pinf, Pstar=Pstar,
                init_a=self._init_x, c=c_aug, dtype=dt)
            ll = float(kf.log_likelihood(ys_jax))
            print(f"  Log-likelihood (exact diffuse): {ll:.2f}")
            x_smooth = np.asarray(kf.smooth_states_only(ys=ys_jax))
        else:
            if use_diffuse and (H_t is not None or Q_t is not None):
                print("  filter_diagnostic: H_t/Q_t in use; "
                      "falling back to standard Kalman filter.")
            kf = KalmanFilter(T=P_aug, R=R_aug, C=Omega, H=H_obs,
                              init_x=self._init_x, init_P=self._init_P,
                              c=c_aug, dtype=dt)
            ll = float(kf.log_likelihood(ys_jax))
            print(f"  Log-likelihood: {ll:.2f}")
            x_smooth = np.asarray(kf.smooth_states_only(
                ys=ys_jax, H_t=H_t, Q_t=Q_t))

        if H_t is not None or Q_t is not None:
            print(f"  Time-varying filter: "
                  f"H_t={'yes' if H_t is not None else 'no'}, "
                  f"Q_t={'yes' if Q_t is not None else 'no'}")
        obs_smooth = x_smooth @ np.asarray(Omega).T

        diag_dir = os.path.join(self.results_dir, "diagnostic")
        os.makedirs(diag_dir, exist_ok=True)

        n_s = self.model.n_stat
        n_o = self.model.n_obs
        has_dates = dates is not None

        if has_dates:
            dates_dt = _ensure_datetime_index(dates)
            dates_arr = dates_dt.to_pydatetime()
        else:
            dates_arr = np.arange(ys_np.shape[0])

        # --- Figure 1: Observables + trend ---
        ncols = min(4, n_o)
        nrows = (n_o + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.2 * nrows),
                                 squeeze=False)
        fig.suptitle(f"Filter Diagnostic: Observables (LL={ll:.1f})",
                     fontsize=13)

        for i in range(n_o):
            ax = axes[i // ncols, i % ncols]
            obs_name = self.model.ordered_obs_vars[i]
            ax.plot(dates_arr, ys_np[:, i], 'k-', alpha=0.5, lw=0.8,
                    label="Data")
            ax.plot(dates_arr, obs_smooth[:, i], 'b-', lw=1.2,
                    label="Smoothed")

            if obs_trend_map and obs_name in obs_trend_map:
                tname = obs_trend_map[obs_name]
                if tname in self.model.ordered_trend_state_vars:
                    tidx = (self.model.ordered_trend_state_vars.index(tname)
                            + n_s)
                    ax.plot(dates_arr, x_smooth[:, tidx], 'r--', lw=1.2,
                            label=f"Trend ({tname})")

            ax.set_title(obs_name, fontsize=9)
            ax.grid(True, ls=':', alpha=0.4)
            ax.legend(fontsize='xx-small', loc='best')
            if has_dates:
                ax.xaxis.set_major_locator(mdates.YearLocator(5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        for j in range(n_o, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = os.path.join(diag_dir, "diag_observables.png")
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")

        # --- Figure 2: Trend states ---
        n_t = self.model.n_trend
        if n_t > 0:
            ncols_t = min(4, n_t)
            nrows_t = (n_t + ncols_t - 1) // ncols_t
            fig2, axes2 = plt.subplots(nrows_t, ncols_t,
                                       figsize=(4.5 * ncols_t, 3.2 * nrows_t),
                                       squeeze=False)
            fig2.suptitle("Filter Diagnostic: Trend States", fontsize=13)

            for i, tvar in enumerate(self.model.ordered_trend_state_vars):
                ax = axes2[i // ncols_t, i % ncols_t]
                tidx = n_s + i
                ax.plot(dates_arr, x_smooth[:, tidx], 'b-', lw=1.5)
                ax.set_title(tvar, fontsize=9)
                ax.grid(True, ls=':', alpha=0.4)
                if has_dates:
                    ax.xaxis.set_major_locator(mdates.YearLocator(5))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            for j in range(n_t, nrows_t * ncols_t):
                axes2[j // ncols_t, j % ncols_t].set_visible(False)

            fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
            fname2 = os.path.join(diag_dir, "diag_trends.png")
            fig2.savefig(fname2, dpi=120, bbox_inches='tight')
            plt.close(fig2)
            print(f"  Saved {fname2}")

        # --- Figure 3: Key stationary states ---
        fig3, axes3 = plt.subplots(
            (n_s + 3) // 4, min(4, n_s),
            figsize=(4.5 * min(4, n_s), 3.2 * ((n_s + 3) // 4)),
            squeeze=False)
        fig3.suptitle("Filter Diagnostic: Stationary States", fontsize=13)

        for i, svar in enumerate(self.model.ordered_stat_vars):
            nr = (n_s + 3) // 4
            nc = min(4, n_s)
            if i >= nr * nc:
                break
            ax = axes3[i // nc, i % nc]
            ax.plot(dates_arr, x_smooth[:, i], 'b-', lw=1.2)
            ax.axhline(0, color='grey', lw=0.5, ls=':')
            ax.set_title(svar, fontsize=9)
            ax.grid(True, ls=':', alpha=0.4)
            if has_dates:
                ax.xaxis.set_major_locator(mdates.YearLocator(5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        for j in range(min(n_s, ((n_s + 3) // 4) * min(4, n_s)),
                       ((n_s + 3) // 4) * min(4, n_s)):
            axes3[j // min(4, n_s), j % min(4, n_s)].set_visible(False)

        fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname3 = os.path.join(diag_dir, "diag_states.png")
        fig3.savefig(fname3, dpi=120, bbox_inches='tight')
        plt.close(fig3)
        print(f"  Saved {fname3}")

        return {'x_smooth': x_smooth, 'obs_smooth': obs_smooth, 'll': ll}

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _ensure_init(self):
        """Auto-initialize Kalman initial conditions if not set."""
        if self._init_x is None or self._init_P is None:
            self.set_initial_conditions()

    def _check_data(self):
        if self._data is None:
            raise RuntimeError("No data attached. Call set_data() first.")

    def _check_priors(self):
        if self._structural_priors is None:
            raise RuntimeError("No priors set. Call set_priors() first.")
        if self._cycle_map is None or self._trend_snr_map is None:
            raise RuntimeError("cycle_map and trend_snr_map required for "
                               "calibration. Pass them to set_priors().")

    def _resolve_params(self, params=None):
        """Resolve parameter dict from various sources.

        Priority: explicit params > posterior median > calibrated params
                  > prior modes > model defaults.
        """
        if params is not None:
            return params

        pdict = dict(self.model.default_param_assignments)

        # Fixed params
        if self._fixed_params:
            pdict.update(self._fixed_params)

        # Prior modes for structural
        if self._structural_priors:
            modes = compute_prior_modes(self._structural_priors)
            pdict.update(modes)

        # Calibrated sigmas
        if self._calibrated_sigmas:
            pdict.update(self._calibrated_sigmas)

        # Posterior median (overrides all above)
        if self._posterior_samples is not None:
            for name, samples in self._posterior_samples.items():
                pdict[name] = float(np.median(np.asarray(samples)))

        # Stored param_dict
        if self._param_dict and params is None:
            # Only use if nothing else overrode
            for k, v in self._param_dict.items():
                if k not in pdict or (self._posterior_samples is None
                                      and self._calibrated_sigmas is None):
                    pdict[k] = v

        return pdict

    def _plot_forecast(self, result, variables, dates, ys_np, obs_names):
        """Plot forecast fan charts."""
        import matplotlib.pyplot as plt

        fcst_dir = os.path.join(self.results_dir, "forecast")
        os.makedirs(fcst_dir, exist_ok=True)

        if variables is None:
            variables = list(result.obs_names)[:6]

        for var in variables:
            try:
                if var in result.obs_names:
                    draws = get_obs_forecast(result, var)
                elif var in result.state_names:
                    draws = get_state_forecast(result, var)
                else:
                    print(f"  '{var}' not found, skipping.")
                    continue
            except (ValueError, IndexError):
                continue

            pcts = forecast_percentiles(draws)

            fig, ax = plt.subplots(figsize=(10, 4))

            # Historical data
            if var in obs_names and dates is not None:
                hist_idx = obs_names.index(var)
                dates_dt = _ensure_datetime_index(dates)
                ax.plot(dates_dt.to_pydatetime(), ys_np[:, hist_idx], 'k-', lw=1,
                        label='Historical')

            # Forecast
            T_data = ys_np.shape[0]
            if dates is not None:
                import pandas as pd
                dates_dt = _ensure_datetime_index(dates)
                last_date = dates_dt[-1]
                fcst_dates = pd.date_range(
                    start=last_date + pd.tseries.offsets.QuarterEnd(1),
                    periods=result.horizon, freq='QE')
                x_fcst = fcst_dates.to_pydatetime()
            else:
                x_fcst = np.arange(T_data, T_data + result.horizon)

            ax.plot(x_fcst, pcts[50], 'b-', lw=1.5, label='Median')
            if 5 in pcts and 95 in pcts:
                ax.fill_between(x_fcst, pcts[5], pcts[95],
                                alpha=0.2, color='blue', label='90% CI')
            if 16 in pcts and 84 in pcts:
                ax.fill_between(x_fcst, pcts[16], pcts[84],
                                alpha=0.35, color='blue', label='68% CI')

            ax.set_title(f"Forecast: {var}", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, ls=':', alpha=0.4)
            fig.tight_layout()

            fname = os.path.join(fcst_dir, f"forecast_{var}.png")
            fig.savefig(fname, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {fname}")

"""Post-estimation policy analysis tools for QPM models.

Three analytical outputs standard in the applied central-bank QPM literature:

1. **Sacrifice ratio** (cost of disinflation): cumulative output cost per
   percentage point permanent reduction in the inflation target.

2. **Exchange rate pass-through**: response of inflation (by component) to
   a 1 pp exchange rate depreciation from different shock sources.

3. **Forecast evaluation**: pseudo real-time rolling-window RMSE, MAE, and
   bias against actual data.

References:
    Berg, A., Karam, P. and Laxton, D. (2006).  "Practical Model-Based
        Monetary Policy Analysis -- A How-To Guide."  IMF WP/06/81.
    Forbes, K., Hjortsoe, I. and Nenova, T. (2018).  "The Shocks Matter:
        Improving our Estimates of Exchange Rate Pass-Through."  Journal of
        International Economics, 114, 255-275.
    Herbst, E. and Schorfheide, F. (2016).  Bayesian Estimation of DSGE
        Models.  Princeton University Press.
"""

from typing import Dict, List, Optional, Tuple, NamedTuple, Union
import numpy as np
import numpy as onp


# ======================================================================
# Data structures
# ======================================================================

class SacrificeRatioResult(NamedTuple):
    """Container for sacrifice-ratio analysis output.

    Attributes:
        sacrifice_ratio:    SR under the requested convention (see
                            ``sacrifice_ratio`` for the definitions).
        cum_output_gap:     signed cumulative output gap in pp-quarters.
        reduction_pp:       permanent target reduction in pp.
        horizon_quarters:   number of quarters simulated.
        gap_path:           (H,) output gap path.
        inflation_path:     (H,) inflation deviation path.
        target_path:        (H,) inflation target path.
        params_used:        parameter dict used.
        convention:         name of the convention used for
                            ``sacrifice_ratio`` ('symmetric', 'mixed',
                            'net', or 'target').
        sr_variants:        dict of every convention evaluated on the
                            same IRF, so the sensitivity of the statistic
                            to the sign convention is auditable without
                            re-running.  Empty for the 'target' method.
        cum_abs_output_gap: sum of |output gap| in pp-quarters.
        cum_inflation:      signed cumulative inflation gap in pp-quarters.
        cum_abs_inflation:  sum of |inflation gap| in pp-quarters.
    """
    sacrifice_ratio: float
    cum_output_gap: float
    reduction_pp: float
    horizon_quarters: int
    gap_path: np.ndarray
    inflation_path: np.ndarray
    target_path: np.ndarray
    params_used: dict
    convention: str = 'target'
    sr_variants: dict = {}
    cum_abs_output_gap: float = float('nan')
    cum_inflation: float = float('nan')
    cum_abs_inflation: float = float('nan')


class PassThroughResult(NamedTuple):
    """Container for exchange rate pass-through analysis.

    Attributes:
        table:       dict of {shock_name: {cpi_var: metrics_dict}}.
                     metrics_dict has keys 'contemporaneous', 'peak',
                     'peak_quarter', 'cumulative'.
        irf_paths:   dict of {shock_name: {var_name: (H,) array}}.
        er_variable: name of the exchange rate variable used.
        horizon:     horizon for cumulative computation.
        params_used: parameter dict used.
    """
    table: dict
    irf_paths: dict
    er_variable: str
    horizon: int
    params_used: dict


class ForecastEvaluationResult(NamedTuple):
    """Container for rolling-window forecast evaluation.

    Attributes:
        rmse:       {var: {h: float}}.
        mae:        {var: {h: float}}.
        bias:       {var: {h: float}}.
        errors:     {var: {h: (N_origins,) array}}.
        origins:    list of forecast origin dates.
        horizons:   list of forecast horizons evaluated.
        variables:  list of variable names.
        forecast_paths: list of (origin, (H, n_var) forecast array).
        rmse_full:    {var: {h: float}} full-sample RMSE (== rmse) when a
                      ``covid_window`` is supplied; else None.
        rmse_excovid: {var: {h: float}} RMSE after dropping (origin,target)
                      pairs intersecting the COVID window; else None.
        n_full:       {var: {h: int}} count of errors in the full sample.
        n_excovid:    {var: {h: int}} count after the COVID filter.
    """
    rmse: dict
    mae: dict
    bias: dict
    errors: dict
    origins: list
    horizons: list
    variables: list
    forecast_paths: list
    rmse_full: dict = None
    rmse_excovid: dict = None
    n_full: dict = None
    n_excovid: dict = None


# ======================================================================
# 1.  Sacrifice ratio
# ======================================================================

def sacrifice_ratio(
    model,
    param_dict: Dict[str, float],
    gap_variable: str = 'L_GDP_GAP',
    inflation_variable: str = 'DLA_CPI',
    target_variable: str = 'PI_TREND',
    reduction_pp: float = 1.0,
    horizon_years: int = 30,
    method: str = 'auto',
    inflation_shock: Optional[str] = None,
    convention: str = 'symmetric',
    *,
    irf_fn=None,
    irf_obs_fn=None,
) -> SacrificeRatioResult:
    """Compute the sacrifice ratio (output cost of disinflation).

    Two methods are available, selected by ``method``:

    **'target'** - Shock the inflation target (SHK_{target_variable}).
    Appropriate for level-form models where the inflation target enters
    the cyclical transition equations.  The SR is:

        SR = |sum gap_t| / (reduction_pp * H)

    **'costpush'** - Shock the inflation gap equation (cost-push shock).
    Standard approach for gap-form QPMs (Carabenciov et al., 2008; Benes
    et al., 2017) where cyclical equations are written in deviations from
    trends and trend shocks do not enter the cyclical block.  A unit
    impulse to the inflation gap creates excess inflation above target;
    the Taylor rule tightens and opens a negative output gap.

    **'auto'** (default) - Tries 'target' first.  If the target shock
    produces zero output gap response (gap-form model), falls back to
    'costpush'.

    Sign conventions for the 'costpush' method
    ------------------------------------------
    Under an endogenous policy rule the cost-push IRF is not
    single-signed: the inflation gap overshoots on impact and then
    undershoots as the rule brings it back, so the placement of the
    absolute value relative to the summation is not innocuous.  Three
    conventions are computed on the same IRF and returned in
    ``sr_variants``; ``convention`` selects which one populates the
    ``sacrifice_ratio`` field:

    - ``'symmetric'`` (default): ``sum|gap| / sum|inflation|``.  The
      absolute value is inside both sums, so numerator and denominator
      are treated identically and the denominator is bounded away from
      zero.  Reads as cumulative absolute output variation per unit of
      cumulative absolute inflation variation.
    - ``'mixed'``: ``|sum gap| / sum|inflation|``.  Absolute value
      outside the numerator sum, inside the denominator sum.  Retained
      for backward comparability; this was the behaviour before the
      convention argument was introduced.
    - ``'net'``: ``|sum gap| / |sum inflation|``.  Both sums are net.
      Closest to a literal reading of a cumulative-response ratio, but
      the denominator nets the policy-induced undershoot against the
      initial overshoot, so it shrinks toward zero as the estimated rule
      becomes more aggressive and the ratio inherits heavy tails.

    None of the three is the permanent-disinflation sacrifice ratio of
    Ball (1994) or Cecchetti and Rich (2001), whose denominator is a
    level change in trend inflation.  That experiment is not available in
    a gap-form model whose cyclical block is written in deviations from
    the target, which is exactly why 'auto' falls through to 'costpush'.

    Args:
        model:               DynareModelWithLambdified instance.
        param_dict:          full parameter dictionary.
        gap_variable:        output gap state variable name.
        inflation_variable:  inflation gap state variable name.
        target_variable:     inflation target state name.
        reduction_pp:        inflation reduction in pp (positive).
        horizon_years:       simulation horizon in years.
        method:              'target', 'costpush', or 'auto'.
        inflation_shock:     explicit shock name for costpush method
                             (default: SHK_{inflation_variable}).
        irf_fn:              low-level state IRF function (parser.irf).
        irf_obs_fn:          low-level obs IRF function (parser.irf_observables).

    Returns:
        SacrificeRatioResult.
    """
    from qpm_toolbox.parser import irf as _irf, irf_observables as _irf_obs

    if irf_fn is None:
        irf_fn = _irf
    if irf_obs_fn is None:
        irf_obs_fn = _irf_obs

    H = horizon_years * 4  # quarters

    sol = model.solve(param_dict)
    if not sol["solution_valid"]:
        raise RuntimeError("Model does not solve at given parameters.")

    P_aug = sol["P_aug"]
    R_aug = sol["R_aug"]
    Omega = sol["Omega"]

    all_shocks = model.aug_shocks
    all_states = model.aug_state_vars
    all_obs = model.ordered_obs_vars

    def _find_shock(name_hint):
        """Find a shock by name, trying SHK_ prefix and substring match."""
        if name_hint in all_shocks:
            return name_hint
        prefixed = f"SHK_{name_hint}"
        if prefixed in all_shocks:
            return prefixed
        candidates = [s for s in all_shocks if name_hint in s]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"Cannot find shock for '{name_hint}'. "
            f"Available: {all_shocks}")

    def _extract_path(var_name, state_irf, obs_irf, scale=1.0):
        """Extract a variable path from state or obs IRFs."""
        if var_name in all_states:
            idx = all_states.index(var_name)
            return state_irf[:, idx] * scale
        elif var_name in all_obs:
            idx = all_obs.index(var_name)
            return obs_irf[:, idx] * scale
        raise ValueError(f"Variable '{var_name}' not found in states or obs.")

    # ------------------------------------------------------------------
    # Method selection
    # ------------------------------------------------------------------
    use_method = method
    if use_method == 'auto':
        # Test whether the target shock generates a gap response
        target_shock = _find_shock(target_variable)
        shk_idx = all_shocks.index(target_shock)
        state_irf = np.asarray(irf_fn(P_aug, R_aug, shk_idx, 20))
        if gap_variable in all_states:
            gap_idx = all_states.index(gap_variable)
            gap_response = state_irf[:, gap_idx]
        else:
            obs_irf = np.asarray(irf_obs_fn(P_aug, R_aug, Omega, shk_idx, 20))
            gap_response = _extract_path(gap_variable, state_irf, obs_irf)
        if np.max(np.abs(gap_response)) < 1e-10:
            use_method = 'costpush'
        else:
            use_method = 'target'

    # ------------------------------------------------------------------
    # Method: target shock
    # ------------------------------------------------------------------
    if use_method == 'target':
        target_shock = _find_shock(target_variable)
        shk_idx = all_shocks.index(target_shock)

        state_irf = np.asarray(irf_fn(P_aug, R_aug, shk_idx, H))
        obs_irf = np.asarray(irf_obs_fn(P_aug, R_aug, Omega, shk_idx, H))

        # Scale so initial target shift = reduction_pp (negative)
        target_idx = all_states.index(target_variable)
        initial_shift = state_irf[0, target_idx]
        if abs(initial_shift) < 1e-15:
            raise RuntimeError(
                f"Shock '{target_shock}' has zero impact on "
                f"'{target_variable}'.")
        scale = -reduction_pp / initial_shift

        target_path = state_irf[:, target_idx] * scale
        gap_path = _extract_path(gap_variable, state_irf, obs_irf, scale)
        inflation_path = _extract_path(
            inflation_variable, state_irf, obs_irf, scale)

        cum_gap = float(np.sum(gap_path))
        cum_abs_gap = float(np.sum(np.abs(gap_path)))
        cum_inf_net = float(np.sum(inflation_path))
        cum_inf = float(np.sum(np.abs(inflation_path)))
        sr = abs(cum_gap) / (reduction_pp * H)
        sr_variants = {}
        convention_used = 'target'

    # ------------------------------------------------------------------
    # Method: cost-push shock (standard for gap-form QPMs)
    # ------------------------------------------------------------------
    elif use_method == 'costpush':
        # Identify the inflation shock
        if inflation_shock is not None:
            cp_shock = _find_shock(inflation_shock)
        else:
            cp_shock = _find_shock(inflation_variable)
        shk_idx = all_shocks.index(cp_shock)

        state_irf = np.asarray(irf_fn(P_aug, R_aug, shk_idx, H))
        obs_irf = np.asarray(irf_obs_fn(P_aug, R_aug, Omega, shk_idx, H))

        # Scale so that the initial inflation gap = +1pp
        inf_idx = all_states.index(inflation_variable)
        initial_inf = state_irf[0, inf_idx]
        if abs(initial_inf) < 1e-15:
            raise RuntimeError(
                f"Shock '{cp_shock}' has zero impact on "
                f"'{inflation_variable}'.")
        scale = reduction_pp / initial_inf

        inflation_path = state_irf[:, inf_idx] * scale
        gap_path = _extract_path(gap_variable, state_irf, obs_irf, scale)

        # Target path is zero for the cost-push method (target unchanged)
        if target_variable in all_states:
            ti = all_states.index(target_variable)
            target_path = state_irf[:, ti] * scale
        else:
            target_path = np.zeros(H)

        # Cumulative responses under both the net and the absolute
        # summation, so every convention below is available at once.
        cum_gap = float(np.sum(gap_path))
        cum_abs_gap = float(np.sum(np.abs(gap_path)))
        cum_inf_net = float(np.sum(inflation_path))
        cum_inf = float(np.sum(np.abs(inflation_path)))

        def _ratio(num, den):
            return num / den if abs(den) > 1e-15 else float('nan')

        sr_variants = {
            'symmetric': _ratio(cum_abs_gap, cum_inf),
            'mixed': _ratio(abs(cum_gap), cum_inf),
            'net': _ratio(abs(cum_gap), abs(cum_inf_net)),
        }
        if convention not in sr_variants:
            raise ValueError(
                f"Unknown convention '{convention}'. Use "
                f"{sorted(sr_variants)}.")
        sr = sr_variants[convention]
        convention_used = convention

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'target', "
                         f"'costpush', or 'auto'.")

    return SacrificeRatioResult(
        sacrifice_ratio=sr,
        cum_output_gap=cum_gap,
        reduction_pp=reduction_pp,
        horizon_quarters=H,
        gap_path=gap_path,
        inflation_path=inflation_path,
        target_path=target_path,
        params_used=param_dict,
        convention=convention_used,
        sr_variants=sr_variants,
        cum_abs_output_gap=cum_abs_gap,
        cum_inflation=cum_inf_net,
        cum_abs_inflation=cum_inf,
    )


# ======================================================================
# 2.  Exchange rate pass-through
# ======================================================================

def pass_through(
    model,
    param_dict: Dict[str, float],
    shocks: List[str],
    inflation_variables: List[str],
    er_variable: str,
    horizon: int = 12,
    shock_size: float = 1.0,
    normalize: bool = True,
    *,
    irf_fn=None,
    irf_obs_fn=None,
) -> PassThroughResult:
    """Compute exchange rate pass-through coefficients from IRFs.

    For each shock, computes how much of the exchange rate movement
    passes through to each inflation variable:

    - Contemporaneous:  pi(0) / ds(0)
    - Peak:             max |pi(t)| and the quarter at which it occurs
    - Cumulative(H):    sum(pi, 0:H) / sum(ds, 0:H), equivalently the
                        ratio of the price-level response to the
                        exchange-rate level response at horizon H
    - Elasticity(H):    dP(H)/dS(0), the price-level response at H per
                        unit of the *impact* exchange-rate move

    ``elasticity`` is the object the pass-through literature reports
    (a one-percent depreciation raising the price level by x percent
    after h quarters) and is the one to quote for magnitudes.  The
    ``cumulative`` metric divides by the exchange-rate level response
    *at* the horizon, which for a stationary gap shock reverts toward
    zero and can change sign inside the reporting window; the ratio
    then has a vanishing denominator and heavy tails across posterior
    draws.  Both are returned so the two can be compared.

    ``elasticity_profile`` holds the whole dP(h)/dS(0) path for
    h = 0..horizon-1, and ``er_level_impact`` / ``er_level_at_h`` record
    the denominators, in percent level deviations, that the two metrics
    respectively use.

    If ``normalize=True`` (default), all IRFs are rescaled so that the
    exchange rate response at impact is exactly +1 pp.

    Args:
        model:                DynareModelWithLambdified instance.
        param_dict:           full parameter dictionary.
        shocks:               list of shock names to evaluate.
        inflation_variables:  list of inflation variable names (state or obs).
        er_variable:          exchange rate depreciation state or obs name.
        horizon:              quarters for cumulative PT.
        shock_size:           initial shock size (std dev units).
        normalize:            if True, rescale so impact ER response = 1 pp.
        irf_fn:               low-level state IRF function.
        irf_obs_fn:           low-level obs IRF function.

    Returns:
        PassThroughResult.
    """
    from qpm_toolbox.parser import irf as _irf, irf_observables as _irf_obs

    if irf_fn is None:
        irf_fn = _irf
    if irf_obs_fn is None:
        irf_obs_fn = _irf_obs

    sol = model.solve(param_dict)
    if not sol["solution_valid"]:
        raise RuntimeError("Model does not solve at given parameters.")

    P_aug = sol["P_aug"]
    R_aug = sol["R_aug"]
    Omega = sol["Omega"]

    all_shocks = model.aug_shocks
    all_states = model.aug_state_vars
    all_obs = model.ordered_obs_vars

    def _get_var_path(state_irf, obs_irf, varname, scale=1.0):
        """Extract a variable's IRF from state or obs arrays."""
        if varname in all_states:
            idx = all_states.index(varname)
            return state_irf[:horizon, idx] * scale
        elif varname in all_obs:
            idx = all_obs.index(varname)
            return obs_irf[:horizon, idx] * scale
        else:
            raise ValueError(
                f"Variable '{varname}' not found in states or observables.")

    table = {}
    irf_paths = {}

    for shk in shocks:
        if shk not in all_shocks:
            raise ValueError(f"Unknown shock '{shk}'. Available: {all_shocks}")
        shk_idx = all_shocks.index(shk)

        # Compute full IRFs (use a generous horizon buffer)
        H_long = max(horizon + 4, 40)
        state_irf = np.asarray(irf_fn(P_aug, R_aug, shk_idx, H_long))
        obs_irf = np.asarray(
            irf_obs_fn(P_aug, R_aug, Omega, shk_idx, H_long))

        # Exchange rate path
        er_path = _get_var_path(state_irf, obs_irf, er_variable, shock_size)

        # Normalization: scale so that er_path[0] = +1.0 pp
        if normalize and abs(er_path[0]) > 1e-15:
            norm_scale = 1.0 / er_path[0]
        elif normalize and abs(er_path[0]) < 1e-15:
            # ER doesn't respond at impact; use cumulative(4) instead
            cum4 = np.sum(er_path[:min(4, horizon)])
            if abs(cum4) > 1e-15:
                norm_scale = 1.0 / (cum4 / min(4, horizon))
            else:
                norm_scale = 1.0
        else:
            norm_scale = shock_size

        er_normed = er_path * norm_scale

        shk_table = {}
        shk_paths = {er_variable: er_normed.copy()}

        for cpi_var in inflation_variables:
            cpi_path = _get_var_path(
                state_irf, obs_irf, cpi_var, shock_size) * norm_scale
            shk_paths[cpi_var] = cpi_path.copy()

            # Metrics
            contemp = cpi_path[0] / er_normed[0] if abs(er_normed[0]) > 1e-15 else np.nan
            peak_val = float(cpi_path[np.argmax(np.abs(cpi_path))])
            peak_q = int(np.argmax(np.abs(cpi_path)))

            # Paths are already truncated to ``horizon`` by _get_var_path.
            cum_er = float(np.sum(er_normed))
            cum_cpi = float(np.sum(cpi_path))
            cum_pt = cum_cpi / cum_er if abs(cum_er) > 1e-15 else np.nan

            # Impact-normalised elasticity.  The rates are quarterly
            # annualised, so cumulating and dividing by four converts
            # both series to level deviations; the factor of four
            # cancels in the ratio.  dP(h)/dS(0) is the conventional
            # pass-through object: the price-level response at horizon
            # h per unit of the impact exchange-rate move.  Unlike
            # dP(h)/dS(h) it has a fixed, non-vanishing denominator,
            # which matters here because the exchange-rate level
            # response of a stationary gap shock reverts and crosses
            # zero well inside the reporting window.
            er_impact_level = float(er_normed[0]) / 4.0
            p_level = np.cumsum(cpi_path) / 4.0
            if abs(er_impact_level) > 1e-15:
                elast = p_level / er_impact_level
                elasticity_profile = [float(v) for v in elast]
                elasticity_h = float(elast[-1])
                peak_i = int(np.argmax(np.abs(elast)))
                elasticity_peak = float(elast[peak_i])
                elasticity_peak_q = peak_i
            else:
                elasticity_profile = [float('nan')] * horizon
                elasticity_h = float('nan')
                elasticity_peak = float('nan')
                elasticity_peak_q = -1

            shk_table[cpi_var] = {
                'contemporaneous': float(contemp),
                'peak': float(peak_val),
                'peak_quarter': peak_q,
                'cumulative': float(cum_pt),
                'cumulative_sum': float(cum_cpi),
                'elasticity': elasticity_h,
                'elasticity_peak': elasticity_peak,
                'elasticity_peak_quarter': elasticity_peak_q,
                'elasticity_profile': elasticity_profile,
                'er_level_impact': er_impact_level,
                'er_level_at_h': float(np.sum(er_normed) / 4.0),
            }

        table[shk] = shk_table
        irf_paths[shk] = shk_paths

    return PassThroughResult(
        table=table,
        irf_paths=irf_paths,
        er_variable=er_variable,
        horizon=horizon,
        params_used=param_dict,
    )


# ======================================================================
# 3.  Forecast evaluation (rolling window)
# ======================================================================

def forecast_evaluation(
    model,
    param_dict: Dict[str, float],
    ys: np.ndarray,
    dates,
    obs_names: List[str],
    H_obs: np.ndarray,
    init_x: np.ndarray,
    init_P: np.ndarray,
    variables: List[str],
    first_origin: str,
    last_origin: str,
    horizons: List[int] = None,
    max_horizon: int = 8,
    stochastic: bool = False,
    seed: int = 42,
    verbose: bool = True,
    c_aug: Optional[np.ndarray] = None,
    H_t: Optional[np.ndarray] = None,
    shock_multipliers: Optional[np.ndarray] = None,
    use_diffuse: bool = True,
    realised_reporting: Optional[Dict[str, np.ndarray]] = None,
    covid_window: Optional[Tuple[str, str]] = None,
) -> ForecastEvaluationResult:
    """Pseudo real-time rolling-window forecast evaluation.

    Uses fixed parameters (no re-estimation per window).  At each forecast
    origin t, the Kalman smoother runs on data [1..t] to extract the
    terminal state x_t, then produces an H-step-ahead deterministic
    forecast.  Errors are computed against actual data.

    This is the standard methodology for QPM forecast evaluation
    (Herbst & Schorfheide 2016, Ch. 7; Berg et al. 2006, IMF WP/06/81).

    Args:
        model:             DynareModelWithLambdified instance.
        param_dict:        full parameter dictionary.
        ys:                (T, n_obs) full data array (numpy).
        dates:             pandas DatetimeIndex of length T.
        obs_names:         list of observable names.
        H_obs:             (n_obs, n_obs) measurement noise covariance.
        init_x:            (n_state,) initial state mean.
        init_P:            (n_state, n_state) initial state covariance.
        variables:         observable names to evaluate.
        first_origin:      first forecast origin date (string).
        last_origin:       last forecast origin date (string).
        horizons:          list of horizons to report (default: [1..max_horizon]).
        max_horizon:       maximum forecast horizon in quarters.
        stochastic:        if True, draw random shocks (noisy); False = deterministic.
        seed:              RNG seed.
        verbose:           print progress.
        c_aug:             (n_state,) state intercept.
        H_t:               (T, n_obs, n_obs) time-varying obs noise (optional).
        shock_multipliers: (T, n_shock) per-period shock scaling (optional).

    Returns:
        ForecastEvaluationResult.
    """
    import pandas as pd
    import jax
    import jax.numpy as jnp
    from qpm_toolbox.kalman_filter import (
        KalmanFilter, DiffuseKalmanFilter, build_diffuse_init, get_dtype)
    from qpm_toolbox.forecast import forecast_unconditional

    if horizons is None:
        horizons = list(range(1, max_horizon + 1))

    dt = get_dtype()

    # Solve model once
    sol = model.solve(param_dict)
    if not sol["solution_valid"]:
        raise RuntimeError("Model does not solve at given parameters.")

    P_aug = sol["P_aug"]
    R_aug = sol["R_aug"]
    Omega = sol["Omega"]
    c = sol["c_aug"] if c_aug is None else c_aug

    T_data = ys.shape[0]

    # Resolve origins
    dates_ts = pd.DatetimeIndex(dates)
    first_ts = pd.Timestamp(first_origin)
    last_ts = pd.Timestamp(last_origin)

    first_idx = int(dates_ts.searchsorted(first_ts))
    last_idx = int(dates_ts.searchsorted(last_ts))

    if first_idx < 8:
        raise ValueError(
            f"First origin too early ({first_origin}): need >= 8 obs before.")

    # Ensure we have enough data after last_origin for max_horizon
    last_idx = min(last_idx, T_data - max(horizons) - 1)

    origins = list(range(first_idx, last_idx + 1))
    origin_dates = [dates_ts[i] for i in origins]

    if verbose:
        print(f"  Forecast evaluation: {len(origins)} origins, "
              f"{first_origin} to {dates_ts[last_idx].strftime('%Y-%m-%d')}")
        print(f"  Horizons: {horizons}")
        print(f"  Variables: {variables}")

    # Resolve reporting-variable bookkeeping.  Variables can be either
    # observables or reporting variables (linear combinations of states,
    # possibly with lags, declared in the .qpm reporting_equations block).
    # The toolbox's `forecast_unconditional` returns x_fcst (H, n_state)
    # alongside y_fcst (H, n_obs), so reporting-variable forecasts are
    # linear post-processing of x_fcst.
    reporting_vars_avail = list(getattr(model, "reporting_vars", []) or [])
    reporting_eqs_avail = dict(getattr(model, "reporting_eqs", {}) or {})
    aug_state_names = list(getattr(model, "aug_state_vars",
                                    getattr(model, "ordered_aug_state_vars",
                                            [])))
    var_indices = {}      # observable index in obs_names
    rpt_in_use = []        # reporting variables requested
    for v in variables:
        if v in obs_names:
            var_indices[v] = obs_names.index(v)
        elif v in reporting_vars_avail:
            rpt_in_use.append(v)
        else:
            raise ValueError(
                f"Variable '{v}' is neither an observable nor a "
                f"reporting variable of the model.  Known observables: "
                f"{obs_names}; known reporting variables: "
                f"{reporting_vars_avail}.")

    if rpt_in_use:
        from qpm_toolbox.qpm_model import _eval_reporting_paths
        if not aug_state_names:
            raise RuntimeError(
                "Model does not expose `aug_state_vars`; cannot resolve "
                "reporting-variable forecasts.")
        if verbose:
            print(f"  Reporting variables requested: {rpt_in_use}")
            if realised_reporting is None or not realised_reporting:
                print("  No realised_reporting series supplied; RMSE for "
                      "reporting variables will be NaN.")

    # JAX arrays
    ys_jax = jnp.asarray(ys, dtype=dt)
    H_obs_jax = jnp.asarray(H_obs, dtype=dt)
    key = jax.random.PRNGKey(seed)

    # Storage for errors: {var: {h: list_of_errors}}
    errors = {v: {h: [] for h in horizons} for v in variables}
    # Parallel flag: did this (origin, target) intersect the COVID window?
    err_iscovid = {v: {h: [] for h in horizons} for v in variables}
    _covid_lo = _covid_hi = None
    if covid_window is not None:
        import pandas as pd
        _covid_lo = pd.Period(covid_window[0], freq='Q').start_time
        _covid_hi = pd.Period(covid_window[1], freq='Q').end_time
    forecast_paths_all = []

    for origin_num, t in enumerate(origins):
        # Truncate data to [0..t]
        ys_trunc = ys_jax[:t + 1]

        # Truncate H_t if time-varying
        H_t_trunc = None
        if H_t is not None:
            H_t_trunc = jnp.asarray(H_t[:t + 1], dtype=dt)

        # Truncate Q_t from shock multipliers
        Q_t_trunc = None
        if shock_multipliers is not None:
            mult_trunc = jnp.asarray(shock_multipliers[:t + 1], dtype=dt)
            R_jax = jnp.asarray(R_aug, dtype=dt)
            R_scaled = R_jax[None, :, :] * mult_trunc[:, None, :]
            Q_t_trunc = jnp.einsum('tij,tkj->tik', R_scaled, R_scaled)

        # Kalman smooth on truncated data.  Use the diffuse smoother
        # for unit-root models (consistent with shock_decomposition);
        # auto-fall-back to the standard filter when time-varying
        # H_t / Q_t are in use (diffuse path doesn't yet support them).
        _diffuse_eff = (use_diffuse
                        and H_t_trunc is None and Q_t_trunc is None)
        if _diffuse_eff:
            Pinf, Pstar, _ = build_diffuse_init(P_aug, R_aug, dtype=dt)
            kf = DiffuseKalmanFilter(
                T=P_aug, R=R_aug, C=Omega, H=H_obs_jax,
                Pinf=Pinf, Pstar=Pstar,
                init_a=jnp.asarray(init_x, dtype=dt),
                c=c, dtype=dt)
            x_smooth = kf.smooth_states_only(ys=ys_trunc)
        else:
            kf = KalmanFilter(T=P_aug, R=R_aug, C=Omega, H=H_obs_jax,
                              init_x=jnp.asarray(init_x, dtype=dt),
                              init_P=jnp.asarray(init_P, dtype=dt),
                              c=c, dtype=dt)
            x_smooth = kf.smooth_states_only(
                ys=ys_trunc, H_t=H_t_trunc, Q_t=Q_t_trunc)
        x_T = x_smooth[-1]

        # Forecast forward
        key, subkey = jax.random.split(key)
        H_fwd = max(horizons)
        x_fcst, y_fcst = forecast_unconditional(
            P_aug, R_aug, Omega, x_T, H_fwd, subkey,
            c=c, stochastic=stochastic)

        y_fcst_np = np.asarray(y_fcst)  # (H_fwd, n_obs)

        # Augment y_fcst with reporting-variable forecasts.  Reporting
        # variables are deterministic linear combinations of augmented
        # states (possibly lagged); for first-difference reporting vars
        # (e.g. DA_LGDP) we anchor the lag-1 reference at the smoothed
        # state at the origin (x_T).
        if rpt_in_use:
            x_T_np = np.asarray(x_T)
            x_fcst_np = np.asarray(x_fcst)
            rpt_eqs_active = {v: reporting_eqs_avail[v] for v in rpt_in_use}
            rpt_fcst = _eval_reporting_paths(
                x_fcst_np, aug_state_names, rpt_in_use, rpt_eqs_active,
                pre_sample_state=x_T_np)  # shape (H_fwd, n_rpt_in_use)
            # Concatenate as extra "pseudo-observable" columns; track
            # their column indices in the combined array.
            y_fcst_full = np.concatenate([y_fcst_np, rpt_fcst], axis=1)
            for k, v in enumerate(rpt_in_use):
                var_indices[v] = len(obs_names) + k
        else:
            y_fcst_full = y_fcst_np

        # Store forecast path for spaghetti plot
        forecast_paths_all.append(
            (origin_dates[origin_num], y_fcst_full.copy()))

        # Compute errors at each horizon.  Skip horizons whose realised
        # observation is NaN - e.g. when an observable column has been
        # masked (Model E masks 2020Q1-Q2 for COVID).  Without this the
        # NaN propagates through ``np.mean`` and the resulting RMSE,
        # MAE, and bias columns are all NaN, breaking the rmse_bars plot.
        for h in horizons:
            actual_idx = t + h
            if actual_idx >= T_data:
                continue
            for v in variables:
                vi = var_indices[v]
                if v in obs_names:
                    actual = float(ys[actual_idx, vi])
                elif (realised_reporting is not None
                      and v in realised_reporting):
                    rs = np.asarray(realised_reporting[v]).ravel()
                    actual = (float(rs[actual_idx])
                              if 0 <= actual_idx < rs.shape[0]
                              else float("nan"))
                else:
                    # No realised series provided for this reporting
                    # variable; skip RMSE accumulation.  The forecast
                    # path is still recorded in forecast_paths_all.
                    continue
                if not np.isfinite(actual):
                    continue
                predicted = float(y_fcst_full[h - 1, vi])  # h-1 since 0-indexed
                if not np.isfinite(predicted):
                    continue
                errors[v][h].append(actual - predicted)
                if covid_window is not None:
                    _od = origin_dates[origin_num]
                    _td = dates[actual_idx]
                    err_iscovid[v][h].append(
                        bool((_covid_lo <= _od <= _covid_hi)
                             or (_covid_lo <= _td <= _covid_hi)))

    # Aggregate metrics
    rmse_out = {}
    mae_out = {}
    bias_out = {}
    errors_out = {}

    for v in variables:
        rmse_out[v] = {}
        mae_out[v] = {}
        bias_out[v] = {}
        errors_out[v] = {}
        for h in horizons:
            err = np.array(errors[v][h])
            errors_out[v][h] = err
            if len(err) > 0:
                rmse_out[v][h] = float(np.sqrt(np.mean(err**2)))
                mae_out[v][h] = float(np.mean(np.abs(err)))
                bias_out[v][h] = float(np.mean(err))
            else:
                rmse_out[v][h] = np.nan
                mae_out[v][h] = np.nan
                bias_out[v][h] = np.nan

    # COVID-window split: full-sample vs (origin,target)-not-in-COVID.
    rmse_full_d = rmse_xc_d = n_full_d = n_xc_d = None
    if covid_window is not None:
        rmse_full_d, rmse_xc_d, n_full_d, n_xc_d = {}, {}, {}, {}
        for v in variables:
            rmse_full_d[v], rmse_xc_d[v] = {}, {}
            n_full_d[v], n_xc_d[v] = {}, {}
            for h in horizons:
                err = np.asarray(errors[v][h], dtype=float)
                isc = np.asarray(err_iscovid[v][h], dtype=bool)
                n_full_d[v][h] = int(err.size)
                rmse_full_d[v][h] = (float(np.sqrt(np.mean(err**2)))
                                     if err.size else np.nan)
                err_xc = err[~isc] if isc.size == err.size and err.size else err
                n_xc_d[v][h] = int(err_xc.size)
                rmse_xc_d[v][h] = (float(np.sqrt(np.mean(err_xc**2)))
                                   if err_xc.size else np.nan)

    if verbose:
        _print_rmse_table(variables, horizons, rmse_out)

    return ForecastEvaluationResult(
        rmse=rmse_out,
        mae=mae_out,
        bias=bias_out,
        errors=errors_out,
        origins=origin_dates,
        horizons=horizons,
        variables=variables,
        forecast_paths=forecast_paths_all,
        rmse_full=rmse_full_d,
        rmse_excovid=rmse_xc_d,
        n_full=n_full_d,
        n_excovid=n_xc_d,
    )


def conditional_forecast_evaluation(
    model,
    param_dict: Dict[str, float],
    ys: np.ndarray,
    dates,
    obs_names: List[str],
    H_obs: np.ndarray,
    init_x: np.ndarray,
    init_P: np.ndarray,
    variables: List[str],
    condition_on: List[Tuple[str, str, str]],
    first_origin: str,
    last_origin: str,
    horizons: List[int] = None,
    max_horizon: int = 8,
    seed: int = 42,
    verbose: bool = True,
    c_aug: Optional[np.ndarray] = None,
    H_t: Optional[np.ndarray] = None,
    shock_multipliers: Optional[np.ndarray] = None,
    use_diffuse: bool = True,
    realised_reporting: Optional[Dict[str, np.ndarray]] = None,
    covid_window: Optional[Tuple[str, str]] = None,
) -> ForecastEvaluationResult:
    """Conditional rolling-window forecast evaluation.

    At each forecast origin, conditions the forecast on actual future
    paths of specified variables, mirroring the information set available
    to a practitioner using a QPM for conditional projection.

    Supports conditioning on both observables (values from ``ys``) and
    state variables (values from a full-sample Kalman smoother).

    This is the Python equivalent of the Matlab
    ``app_conditional_forecast_evaluation.m`` script.

    Args:
        model:        DynareModelWithLambdified instance.
        param_dict:   full parameter dictionary.
        ys:           (T, n_obs) full data array (numpy).
        dates:        pandas DatetimeIndex of length T.
        obs_names:    list of observable names.
        H_obs:        (n_obs, n_obs) measurement noise covariance.
        init_x:       (n_state,) initial state mean.
        init_P:       (n_state, n_state) initial state covariance.
        variables:    observable names to evaluate (forecast targets).
        condition_on: list of (kind, variable, shock) tuples.
                      kind='obs': condition on an observable, values
                      from ``ys``.  kind='state': condition on a state
                      variable, values from full-sample Kalman smoother.
        first_origin: first forecast origin date (string).
        last_origin:  last forecast origin date (string).
        horizons:     list of horizons in quarters.
        max_horizon:  maximum forecast horizon.
        seed:         RNG seed (unused, kept for API consistency).
        verbose:      print progress.
        c_aug:        (n_state,) state intercept override.
        H_t:          (T, n_obs, n_obs) time-varying obs noise.
        shock_multipliers: (T, n_shock) per-period shock scaling.

    Returns:
        ForecastEvaluationResult (same structure as unconditional).

    References:
        Waggoner, D. and Zha, T. (1999). "Conditional Forecasts in Dynamic
            Multivariate Models."  Review of Economics and Statistics, 81(4).
    """
    import pandas as pd
    import jax
    import jax.numpy as jnp
    from qpm_toolbox.kalman_filter import (
        KalmanFilter, DiffuseKalmanFilter, build_diffuse_init, get_dtype)
    from qpm_toolbox.conditional_forecast import (
        ForecastPlan, _forward_conditional, _forward_unconditional,
    )

    if horizons is None:
        horizons = list(range(1, max_horizon + 1))

    dt = get_dtype()

    # Solve model once
    sol = model.solve(param_dict)
    if not sol["solution_valid"]:
        raise RuntimeError("Model does not solve at given parameters.")

    P_aug = sol["P_aug"]
    R_aug = sol["R_aug"]
    Omega = sol["Omega"]
    c = sol["c_aug"] if c_aug is None else c_aug

    state_names = list(model.aug_state_vars)
    obs_names_model = list(model.ordered_obs_vars)
    shock_names = list(model.aug_shocks)

    T_data = ys.shape[0]

    # Resolve origins
    dates_ts = pd.DatetimeIndex(dates)
    first_ts = pd.Timestamp(first_origin)
    last_ts = pd.Timestamp(last_origin)

    first_idx = int(dates_ts.searchsorted(first_ts))
    last_idx = int(dates_ts.searchsorted(last_ts))

    if first_idx < 8:
        raise ValueError(
            f"First origin too early ({first_origin}): need >= 8 obs before.")

    # Ensure we have enough data after last_origin for max_horizon
    last_idx = min(last_idx, T_data - max(horizons) - 1)

    origins = list(range(first_idx, last_idx + 1))
    origin_dates = [dates_ts[i] for i in origins]

    if verbose:
        print(f"  Conditional forecast evaluation: {len(origins)} origins, "
              f"{first_origin} to {dates_ts[last_idx].strftime('%Y-%m-%d')}")
        print(f"  Horizons: {horizons}")
        print(f"  Evaluating: {variables}")
        cond_names = [c[1] for c in condition_on]
        print(f"  Conditioning on: {cond_names}")

    # Resolve evaluation targets (observables + reporting variables).
    reporting_vars_avail = list(getattr(model, "reporting_vars", []) or [])
    reporting_eqs_avail = dict(getattr(model, "reporting_eqs", {}) or {})
    aug_state_names_eval = list(getattr(model, "aug_state_vars",
                                          getattr(model, "ordered_aug_state_vars",
                                                  [])))
    var_indices = {}
    rpt_in_use = []
    for v in variables:
        if v in obs_names:
            var_indices[v] = obs_names.index(v)
        elif v in reporting_vars_avail:
            rpt_in_use.append(v)
        else:
            raise ValueError(
                f"Variable '{v}' is neither an observable nor a reporting "
                f"variable of the model.  Known observables: {obs_names}; "
                f"known reporting variables: {reporting_vars_avail}.")
    if rpt_in_use:
        from qpm_toolbox.qpm_model import _eval_reporting_paths
        if verbose:
            print(f"  Reporting variables requested: {rpt_in_use}")
            if realised_reporting is None or not realised_reporting:
                print("  No realised_reporting series supplied; RMSE for "
                      "reporting variables will be NaN.")

    # Conditioning variable indices
    cond_obs_indices = {}
    cond_state_indices = {}
    has_state_conditions = False
    for kind, var, shock in condition_on:
        if kind == 'obs':
            if var in obs_names:
                cond_obs_indices[var] = obs_names.index(var)
            else:
                raise ValueError(
                    f"Conditioning observable '{var}' not found. "
                    f"Available: {obs_names}")
        elif kind == 'state':
            if var in state_names:
                cond_state_indices[var] = state_names.index(var)
                has_state_conditions = True
            else:
                raise ValueError(
                    f"Conditioning state '{var}' not found. "
                    f"Available states (first 20): {state_names[:20]}")

    # JAX arrays for Kalman smoother
    ys_jax = jnp.asarray(ys, dtype=dt)
    H_obs_jax = jnp.asarray(H_obs, dtype=dt)

    # Full-sample smoother for state conditioning values
    x_smooth_full = None
    if has_state_conditions:
        if verbose:
            print("  Running full-sample Kalman smoother for state targets...")
        Q_t_full = None
        if shock_multipliers is not None:
            mult_full = jnp.asarray(shock_multipliers, dtype=dt)
            R_jax_full = jnp.asarray(R_aug, dtype=dt)
            R_sc = R_jax_full[None, :, :] * mult_full[:, None, :]
            Q_t_full = jnp.einsum('tij,tkj->tik', R_sc, R_sc)
        H_t_full = jnp.asarray(H_t, dtype=dt) if H_t is not None else None

        kf_full = KalmanFilter(
            T=P_aug, R=R_aug, C=Omega, H=H_obs_jax,
            init_x=jnp.asarray(init_x, dtype=dt),
            init_P=jnp.asarray(init_P, dtype=dt),
            c=c, dtype=dt)
        x_smooth_full = onp.asarray(
            kf_full.smooth_states_only(
                ys=ys_jax, H_t=H_t_full, Q_t=Q_t_full))
        if verbose:
            print(f"  Smoothed states shape: {x_smooth_full.shape}")

    # NumPy arrays for conditional forecast
    P_np = onp.asarray(P_aug)
    R_np = onp.asarray(R_aug)
    Omega_np = onp.asarray(Omega)
    c_np = onp.asarray(c)

    # Storage
    errors = {v: {h: [] for h in horizons} for v in variables}
    err_iscovid = {v: {h: [] for h in horizons} for v in variables}
    _covid_lo = _covid_hi = None
    if covid_window is not None:
        import pandas as pd
        _covid_lo = pd.Period(covid_window[0], freq='Q').start_time
        _covid_hi = pd.Period(covid_window[1], freq='Q').end_time
    forecast_paths_all = []

    for origin_num, t in enumerate(origins):
        if verbose and origin_num % 5 == 0:
            print(f"    Origin {origin_num + 1}/{len(origins)}: "
                  f"{dates_ts[t].strftime('%Y-%m-%d')}")

        # --- Kalman smooth on data up to origin ---
        ys_trunc = ys_jax[:t + 1]

        H_t_trunc = None
        if H_t is not None:
            H_t_trunc = jnp.asarray(H_t[:t + 1], dtype=dt)

        Q_t_trunc = None
        if shock_multipliers is not None:
            mult_trunc = jnp.asarray(shock_multipliers[:t + 1], dtype=dt)
            R_jax = jnp.asarray(R_aug, dtype=dt)
            R_scaled = R_jax[None, :, :] * mult_trunc[:, None, :]
            Q_t_trunc = jnp.einsum('tij,tkj->tik', R_scaled, R_scaled)

        # Use diffuse smoother on truncated data when no time-varying
        # H_t / Q_t (auto-fall-back to standard otherwise).
        _diffuse_eff = (use_diffuse
                        and H_t_trunc is None and Q_t_trunc is None)
        if _diffuse_eff:
            Pinf, Pstar, _ = build_diffuse_init(P_aug, R_aug, dtype=dt)
            kf = DiffuseKalmanFilter(
                T=P_aug, R=R_aug, C=Omega, H=H_obs_jax,
                Pinf=Pinf, Pstar=Pstar,
                init_a=jnp.asarray(init_x, dtype=dt),
                c=c, dtype=dt)
            x_smooth = kf.smooth_states_only(ys=ys_trunc)
        else:
            kf = KalmanFilter(T=P_aug, R=R_aug, C=Omega, H=H_obs_jax,
                              init_x=jnp.asarray(init_x, dtype=dt),
                              init_P=jnp.asarray(init_P, dtype=dt),
                              c=c, dtype=dt)
            x_smooth = kf.smooth_states_only(
                ys=ys_trunc, H_t=H_t_trunc, Q_t=Q_t_trunc)
        x_T = onp.asarray(x_smooth[-1])

        # --- Build ForecastPlan for this origin ---
        H_fwd = max(horizons)

        # Check how many future periods are available
        avail = T_data - t - 1
        H_cond = min(H_fwd, avail)

        plan = ForecastPlan(horizon=H_fwd)

        for kind, var, shock in condition_on:
            if kind == 'obs' and var in cond_obs_indices:
                vi = cond_obs_indices[var]
                # Extract actual future values from data
                future_vals = []
                periods = []
                for h in range(H_cond):
                    idx = t + 1 + h
                    if idx < T_data:
                        val = float(ys[idx, vi])
                        if onp.isfinite(val):
                            periods.append(h)
                            future_vals.append(val)
                if len(periods) > 0:
                    plan.add_condition(
                        kind='obs',
                        variable=var,
                        shock=shock,
                        periods=periods,
                        values=future_vals,
                    )
            elif kind == 'state' and var in cond_state_indices:
                si = cond_state_indices[var]
                # Extract smoothed state values from full-sample smoother
                future_vals = []
                periods = []
                for h in range(H_cond):
                    idx = t + 1 + h
                    if idx < x_smooth_full.shape[0]:
                        val = float(x_smooth_full[idx, si])
                        if onp.isfinite(val):
                            periods.append(h)
                            future_vals.append(val)
                if len(periods) > 0:
                    plan.add_condition(
                        kind='state',
                        variable=var,
                        shock=shock,
                        periods=periods,
                        values=future_vals,
                    )

        # Compile and run conditional forecast
        compiled = plan._compile(state_names, obs_names_model, shock_names)
        x_cond, y_cond, eps = _forward_conditional(
            P_np, R_np, Omega_np, c_np, x_T, compiled)

        y_fcst_np = onp.asarray(y_cond)  # (H_fwd, n_obs)

        # Augment with reporting-variable forecasts as extra pseudo-obs
        # columns when requested.  See forecast_evaluation for details.
        if rpt_in_use:
            x_cond_np = onp.asarray(x_cond)
            x_T_np = onp.asarray(x_T)
            rpt_eqs_active = {v: reporting_eqs_avail[v] for v in rpt_in_use}
            rpt_fcst = _eval_reporting_paths(
                x_cond_np, aug_state_names_eval, rpt_in_use,
                rpt_eqs_active, pre_sample_state=x_T_np)
            y_fcst_full = onp.concatenate([y_fcst_np, rpt_fcst], axis=1)
            for k, v in enumerate(rpt_in_use):
                var_indices[v] = len(obs_names) + k
        else:
            y_fcst_full = y_fcst_np

        # Store forecast path for spaghetti plot
        forecast_paths_all.append(
            (origin_dates[origin_num], y_fcst_full.copy()))

        # Compute errors at each horizon.  Skip masked / NaN actuals so
        # the COVID mask in Model E does not propagate to all RMSEs
        # (see forecast_evaluation for the corresponding fix).
        for h in horizons:
            actual_idx = t + h
            if actual_idx >= T_data:
                continue
            for v in variables:
                vi = var_indices[v]
                if v in obs_names:
                    actual = float(ys[actual_idx, vi])
                elif (realised_reporting is not None
                      and v in realised_reporting):
                    rs = onp.asarray(realised_reporting[v]).ravel()
                    actual = (float(rs[actual_idx])
                              if 0 <= actual_idx < rs.shape[0]
                              else float("nan"))
                else:
                    continue
                if not np.isfinite(actual):
                    continue
                predicted = float(y_fcst_full[h - 1, vi])
                if not np.isfinite(predicted):
                    continue
                errors[v][h].append(actual - predicted)
                if covid_window is not None:
                    _od = origin_dates[origin_num]
                    _td = dates[actual_idx]
                    err_iscovid[v][h].append(
                        bool((_covid_lo <= _od <= _covid_hi)
                             or (_covid_lo <= _td <= _covid_hi)))

    # Aggregate metrics
    rmse_out = {}
    mae_out = {}
    bias_out = {}
    errors_out = {}

    for v in variables:
        rmse_out[v] = {}
        mae_out[v] = {}
        bias_out[v] = {}
        errors_out[v] = {}
        for h in horizons:
            err = onp.array(errors[v][h])
            errors_out[v][h] = err
            if len(err) > 0:
                rmse_out[v][h] = float(onp.sqrt(onp.mean(err**2)))
                mae_out[v][h] = float(onp.mean(onp.abs(err)))
                bias_out[v][h] = float(onp.mean(err))
            else:
                rmse_out[v][h] = onp.nan
                mae_out[v][h] = onp.nan
                bias_out[v][h] = onp.nan

    # COVID-window split: full-sample vs (origin,target)-not-in-COVID.
    rmse_full_d = rmse_xc_d = n_full_d = n_xc_d = None
    if covid_window is not None:
        rmse_full_d, rmse_xc_d, n_full_d, n_xc_d = {}, {}, {}, {}
        for v in variables:
            rmse_full_d[v], rmse_xc_d[v] = {}, {}
            n_full_d[v], n_xc_d[v] = {}, {}
            for h in horizons:
                err = onp.asarray(errors[v][h], dtype=float)
                isc = onp.asarray(err_iscovid[v][h], dtype=bool)
                n_full_d[v][h] = int(err.size)
                rmse_full_d[v][h] = (float(onp.sqrt(onp.mean(err**2)))
                                     if err.size else onp.nan)
                err_xc = err[~isc] if isc.size == err.size and err.size else err
                n_xc_d[v][h] = int(err_xc.size)
                rmse_xc_d[v][h] = (float(onp.sqrt(onp.mean(err_xc**2)))
                                   if err_xc.size else onp.nan)

    if verbose:
        _print_rmse_table(variables, horizons, rmse_out)

    return ForecastEvaluationResult(
        rmse=rmse_out,
        mae=mae_out,
        bias=bias_out,
        errors=errors_out,
        origins=origin_dates,
        horizons=horizons,
        variables=variables,
        forecast_paths=forecast_paths_all,
        rmse_full=rmse_full_d,
        rmse_excovid=rmse_xc_d,
        n_full=n_full_d,
        n_excovid=n_xc_d,
    )


def _print_rmse_table(variables, horizons, rmse):
    """Print a formatted RMSE table to console."""
    header = f"  {'Variable':25s}" + "".join(f"  h={h:<3d}" for h in horizons)
    print(f"\n  {'RMSE Table':^{len(header)}}")
    print("  " + "-" * (len(header) - 2))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for v in variables:
        row = f"  {v:25s}"
        for h in horizons:
            val = rmse[v].get(h, np.nan)
            row += f"  {val:<6.2f}"
        print(row)
    print("  " + "-" * (len(header) - 2))


# ======================================================================
# Plotting helpers
# ======================================================================

def plot_sacrifice_ratio(result: SacrificeRatioResult, show=False):
    """Two-panel plot: output gap and inflation paths after disinflation."""
    import matplotlib.pyplot as plt

    H = result.horizon_quarters
    t = np.arange(H)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Cost of Disinflation (SR = {result.sacrifice_ratio:.3f})",
        fontsize=13)

    # Output gap
    ax = axes[0]
    ax.fill_between(t, result.gap_path, 0, alpha=0.3, color='C0')
    ax.plot(t, result.gap_path, 'C0-', lw=1.5)
    ax.axhline(0, color='grey', lw=0.5, ls=':')
    ax.set_title("Output Gap (pp)")
    ax.set_xlabel("Quarters")
    ax.grid(True, ls=':', alpha=0.4)

    # Inflation deviation
    ax = axes[1]
    ax.plot(t, result.inflation_path, 'C1-', lw=1.5, label="Inflation")
    ax.plot(t, result.target_path, 'C3--', lw=1.2, label="Target")
    ax.axhline(0, color='grey', lw=0.5, ls=':')
    ax.set_title("Inflation & Target (pp dev)")
    ax.set_xlabel("Quarters")
    ax.legend(fontsize='small')
    ax.grid(True, ls=':', alpha=0.4)

    # Cumulative output gap
    ax = axes[2]
    cum_gap = np.cumsum(result.gap_path)
    ax.plot(t, cum_gap, 'C0-', lw=1.5)
    ax.axhline(result.cum_output_gap, color='C0', lw=0.8, ls='--',
               label=f"Total = {result.cum_output_gap:.2f}")
    ax.set_title("Cumulative Output Gap (pp-Q)")
    ax.set_xlabel("Quarters")
    ax.legend(fontsize='small')
    ax.grid(True, ls=':', alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if show:
        plt.show()
    return fig


def plot_pass_through(result: PassThroughResult, show=False):
    """Multi-panel plot of inflation response to ER depreciation, by shock."""
    import matplotlib.pyplot as plt

    shocks = list(result.table.keys())
    n_shocks = len(shocks)

    fig, axes = plt.subplots(1, n_shocks, figsize=(5 * n_shocks, 4),
                             squeeze=False)
    fig.suptitle("Exchange Rate Pass-Through to Inflation", fontsize=13)

    for i, shk in enumerate(shocks):
        ax = axes[0, i]
        paths = result.irf_paths[shk]
        H = result.horizon

        t = np.arange(H)
        er_path = paths[result.er_variable]
        ax.plot(t, er_path, 'k--', lw=1.2, alpha=0.6,
                label=result.er_variable)

        for cpi_var in result.table[shk]:
            cpi_path = paths[cpi_var]
            metrics = result.table[shk][cpi_var]
            label = (f"{cpi_var} "
                     f"(cum={metrics['cumulative']:.2f})")
            ax.plot(t, cpi_path, lw=1.5, label=label)

        ax.axhline(0, color='grey', lw=0.5, ls=':')
        ax.set_title(f"Shock: {shk}", fontsize=10)
        ax.set_xlabel("Quarters")
        if i == 0:
            ax.set_ylabel("pp response to 1pp depreciation")
        ax.legend(fontsize='xx-small', loc='best')
        ax.grid(True, ls=':', alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if show:
        plt.show()
    return fig


def plot_forecast_spaghetti(
    result: ForecastEvaluationResult,
    ys: np.ndarray,
    dates,
    obs_names: List[str],
    n_hist_show: int = 40,
    show: bool = False,
    realised_reporting: Optional[Dict[str, np.ndarray]] = None,
):
    """Spaghetti plots: overlaid forecast paths from all origins + actuals.

    Forecast paths returned by forecast_evaluation may have more columns
    than ``obs_names``: any reporting variables included in ``variables``
    are appended after the observable columns.  The plot helper finds the
    right column for each variable by name, and uses the optional
    ``realised_reporting`` mapping to overlay the realised historical
    series for reporting variables.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    variables = result.variables
    n_vars = len(variables)
    ncols = min(3, n_vars)
    nrows = (n_vars + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle("Out-of-Sample Rolling Forecast Paths", fontsize=13)

    dates_ts = pd.DatetimeIndex(dates)
    has_dates = dates is not None

    # Build the variable -> column-index map in the order they were appended.
    # Observables occupy the first len(obs_names) columns; reporting
    # variables follow in `result.variables` order (those not in obs_names).
    rpt_order = [v for v in variables if v not in obs_names]
    col_index = {}
    for v in variables:
        if v in obs_names:
            col_index[v] = obs_names.index(v)
        else:
            col_index[v] = len(obs_names) + rpt_order.index(v)

    for k, v in enumerate(variables):
        ax = axes[k // ncols, k % ncols]
        vi = col_index[v]

        # Historical series: ys for observables, realised_reporting for
        # reporting variables (if supplied).
        if v in obs_names:
            hist = ys[:, vi]
            hist_label = "Actual"
        elif realised_reporting is not None and v in realised_reporting:
            hist = np.asarray(realised_reporting[v]).ravel()
            hist_label = "Actual"
        else:
            hist = None
            hist_label = None

        if hist is not None:
            if has_dates:
                start_idx = max(0, len(dates_ts) - n_hist_show)
                # Align to dates length (realised series should match T).
                ax.plot(dates_ts[start_idx:len(hist)],
                        hist[start_idx:len(dates_ts)],
                        'k-', lw=1.5, label=hist_label, zorder=10)
            else:
                ax.plot(hist, 'k-', lw=1.5, label=hist_label, zorder=10)

        # Overlay forecast paths
        for origin_date, y_fcst in result.forecast_paths:
            H = y_fcst.shape[0]
            if has_dates:
                fcst_dates = pd.date_range(
                    start=origin_date + pd.tseries.offsets.QuarterBegin(1),
                    periods=H, freq='QS')
                ax.plot(fcst_dates, y_fcst[:, vi],
                        color='steelblue', alpha=0.25, lw=0.8)
            else:
                ax.plot(y_fcst[:, vi],
                        color='steelblue', alpha=0.25, lw=0.8)

        ax.set_title(v, fontsize=10)
        ax.grid(True, ls=':', alpha=0.4)
        if has_dates:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    for j in range(n_vars, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if show:
        plt.show()
    return fig


def plot_rmse_bars(result: ForecastEvaluationResult, show=False):
    """Bar chart of RMSE by variable and horizon."""
    import matplotlib.pyplot as plt

    variables = result.variables
    horizons = result.horizons
    n_vars = len(variables)

    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4),
                             squeeze=False)
    fig.suptitle("Forecast RMSE by Horizon", fontsize=13)

    for k, v in enumerate(variables):
        ax = axes[0, k]
        vals = [result.rmse[v].get(h, 0) for h in horizons]
        ax.bar(range(len(horizons)), vals, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"h={h}" for h in horizons], fontsize=8)
        ax.set_title(v, fontsize=10)
        ax.set_ylabel("RMSE" if k == 0 else "")
        ax.grid(True, axis='y', ls=':', alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if show:
        plt.show()
    return fig

"""Conditional forecasting for linear Gaussian state-space models.

Implements the Waggoner & Zha (1999) conditional forecast methodology:
given target paths for selected observables or states over the forecast
horizon, solve for the structural shocks that achieve those targets
while minimising the shock norm (maximum-entropy principle).

State-space form:
    x_t = c + T x_{t-1} + R eps_t,   eps_t ~ N(0, I)
    y_t = Omega x_t     + eta_t,     eta_t ~ N(0, H)

The per-period forward inversion solves:
    C_t @ eps_active = d_t
where C_t is the constraint matrix (rows from the impact matrix for
conditioned variables, columns for freed shocks) and d_t is the
discrepancy between the target and the unconditional forecast.

Usage (fixed parameters):
    plan = ForecastPlan(horizon=16)
    plan.add_condition('obs', 'OBS_RS', 'SHK_RS',
                       periods=range(4), values=[13.75]*4)
    result = forecast_conditional(model, param_dict, ys, H_obs,
                                   init_x, init_P, plan)

Usage (across posterior):
    result = forecast_conditional_posterior(
        model, posterior_samples, estimated_names, fixed_params,
        ys, H_obs, init_x, init_P, plan, key)

References:
    Waggoner, D. and Zha, T. (1999). "Conditional Forecasts in Dynamic
        Multivariate Models." Review of Economics and Statistics, 81(4),
        639-651.
    Del Negro, M. and Schorfheide, F. (2013). "DSGE Model-Based
        Forecasting." In Handbook of Economic Forecasting, Vol. 2A.
"""

from typing import Dict, List, Optional, NamedTuple, Any, Tuple, Union
import warnings

import jax
import jax.numpy as jnp
from jax import random
import numpy as onp

from qpm_toolbox.kalman_filter import (
    KalmanFilter, DiffuseKalmanFilter, build_diffuse_init, get_dtype,
)
from qpm_toolbox.dk_smoother import DKSimulationSmoother


# ======================================================================
# Data structures
# ======================================================================

class ForecastCondition(NamedTuple):
    """A single conditioning constraint for the forecast plan.

    Attributes:
        variable: name of the conditioned variable (state or observable).
        kind:     'obs' for observable conditioning, 'state' for state.
        periods:  int array of 0-indexed forecast horizons where this
                  condition is active.
        values:   float array of target values (same length as periods).
        shock:    name of the shock freed to satisfy this condition.
    """
    variable: str
    kind: str
    periods: onp.ndarray
    values: onp.ndarray
    shock: str


class _CompiledPlan(NamedTuple):
    """Precomputed per-period index arrays for the forward solver.

    Attributes:
        horizon:           forecast horizon H.
        per_period:        list of H dicts with keys:
            'state_rows':    int array of constrained state indices.
            'obs_rows':      int array of constrained obs indices.
            'shock_cols':    int array of freed shock column indices.
            'state_targets': float array of state target values.
            'obs_targets':   float array of obs target values.
        other_shocks_mat:  (H, n_shock) exogenous shock paths.
    """
    horizon: int
    per_period: List[dict]
    other_shocks_mat: onp.ndarray


class ConditionalForecastResult(NamedTuple):
    """Container for conditional forecast output across draws.

    Attributes:
        state_forecasts:  (J, H, n_state) conditioned state paths.
        obs_forecasts:    (J, H, n_obs)   conditioned observable paths.
        shock_paths:      (J, H, n_shock) implied shock vectors used.
        baseline_states:  (J, H, n_state) unconditional state paths.
        baseline_obs:     (J, H, n_obs)   unconditional observable paths.
        state_names:      list of state variable names.
        obs_names:        list of observable variable names.
        shock_names:      list of shock variable names.
        horizon:          forecast horizon H.
    """
    state_forecasts: onp.ndarray
    obs_forecasts: onp.ndarray
    shock_paths: onp.ndarray
    baseline_states: onp.ndarray
    baseline_obs: onp.ndarray
    state_names: List[str]
    obs_names: List[str]
    shock_names: List[str]
    horizon: int


# ======================================================================
# ForecastPlan builder
# ======================================================================

class ForecastPlan:
    """Builder for conditional forecast specifications.

    Collects variable-shock pairings with target paths, validates
    consistency, and compiles them into per-period constraint matrices.

    Usage:
        plan = ForecastPlan(horizon=16)
        plan.add_condition('obs', 'OBS_RS', 'SHK_RS',
                           periods=range(4), values=[13.75]*4)
        plan.add_condition('state', 'GDP_GAP', 'SHK_GDP_GAP',
                           periods=range(8, 16), values=0.0)
        plan.set_other_shocks('SHK_LRWOIL_GAP',
                              periods=[0, 1, 2, 3],
                              values=[2.0, 1.5, 1.0, 0.5])
    """

    def __init__(self, horizon: int):
        self.horizon = horizon
        self.conditions: List[ForecastCondition] = []
        self.other_shocks: Dict[str, Tuple[onp.ndarray, onp.ndarray]] = {}

    def add_condition(
        self,
        kind: str,
        variable: str,
        shock: str,
        periods: Union[range, List[int], onp.ndarray],
        values: Union[float, List[float], onp.ndarray],
    ) -> 'ForecastPlan':
        """Add a conditioning constraint.

        Args:
            kind:     'obs' or 'state'.
            variable: conditioned variable name.
            shock:    shock freed to match this constraint.
            periods:  0-indexed forecast horizons (e.g. range(4)).
            values:   target value(s). Scalar broadcast to all periods.

        Returns:
            self (for chaining).
        """
        if kind not in ('obs', 'state'):
            raise ValueError(f"kind must be 'obs' or 'state', got '{kind}'")
        periods = onp.asarray(list(periods), dtype=int)
        if onp.any(periods < 0) or onp.any(periods >= self.horizon):
            raise ValueError(
                f"periods must be in [0, {self.horizon}), "
                f"got {periods.min()}..{periods.max()}")
        if onp.isscalar(values) or (hasattr(values, '__len__') and len(values) == 1):
            val = float(values) if onp.isscalar(values) else float(values[0])
            values = onp.full(len(periods), val)
        else:
            values = onp.asarray(values, dtype=float)
        if len(periods) != len(values):
            raise ValueError(
                f"periods ({len(periods)}) and values ({len(values)}) "
                f"must have the same length")
        self.conditions.append(
            ForecastCondition(variable, kind, periods, values, shock))
        return self

    def set_other_shocks(
        self,
        shock: str,
        periods: Union[range, List[int], onp.ndarray],
        values: Union[float, List[float], onp.ndarray],
    ) -> 'ForecastPlan':
        """Add exogenous shock paths on top of conditioning shocks.

        These are applied before the conditioning inversion, so the
        conditioning solver accounts for their effect.

        Args:
            shock:   shock name.
            periods: 0-indexed forecast horizons.
            values:  shock values at those periods.

        Returns:
            self (for chaining).
        """
        periods = onp.asarray(list(periods), dtype=int)
        if onp.any(periods < 0) or onp.any(periods >= self.horizon):
            raise ValueError(
                f"other_shocks periods must be in [0, {self.horizon})")
        if onp.isscalar(values):
            values = onp.full(len(periods), float(values))
        else:
            values = onp.asarray(values, dtype=float)
        if len(periods) != len(values):
            raise ValueError("periods and values must have the same length")
        self.other_shocks[shock] = (periods, values)
        return self

    def validate(
        self,
        state_names: List[str],
        obs_names: List[str],
        shock_names: List[str],
    ) -> None:
        """Validate the plan against model variable/shock names.

        Raises ValueError for hard errors; emits warnings for soft
        issues (over/under-determined periods).
        """
        for cond in self.conditions:
            if cond.kind == 'obs' and cond.variable not in obs_names:
                raise ValueError(
                    f"Observable '{cond.variable}' not found in model. "
                    f"Available: {obs_names}")
            if cond.kind == 'state' and cond.variable not in state_names:
                raise ValueError(
                    f"State '{cond.variable}' not found in model. "
                    f"Available: {state_names}")
            if cond.shock not in shock_names:
                raise ValueError(
                    f"Shock '{cond.shock}' not found in model. "
                    f"Available: {shock_names}")

        for sname in self.other_shocks:
            if sname not in shock_names:
                raise ValueError(
                    f"other_shocks: '{sname}' not found in model. "
                    f"Available: {shock_names}")

        # Check per-period parity
        for h in range(self.horizon):
            n_conds = sum(1 for c in self.conditions if h in c.periods)
            n_shocks = len(set(
                c.shock for c in self.conditions if h in c.periods))
            if n_conds > n_shocks:
                warnings.warn(
                    f"Period h={h}: {n_conds} conditions but {n_shocks} "
                    f"unique shocks -- over-determined (least-squares).")

        # Check for duplicate variable-shock pairs
        pairs = [(c.variable, c.shock) for c in self.conditions]
        if len(pairs) != len(set(pairs)):
            warnings.warn("Duplicate variable-shock pairs in plan.")

    def _compile(
        self,
        state_names: List[str],
        obs_names: List[str],
        shock_names: List[str],
    ) -> _CompiledPlan:
        """Compile to per-period index arrays for the solver."""
        self.validate(state_names, obs_names, shock_names)

        n_shock = len(shock_names)
        state_idx = {n: i for i, n in enumerate(state_names)}
        obs_idx = {n: i for i, n in enumerate(obs_names)}
        shock_idx = {n: i for i, n in enumerate(shock_names)}

        # Build per-period constraint info
        per_period = []
        for h in range(self.horizon):
            s_rows, s_targets, s_shock_cols = [], [], []
            o_rows, o_targets, o_shock_cols = [], [], []

            for cond in self.conditions:
                mask = (cond.periods == h)
                if not onp.any(mask):
                    continue
                idx_in_cond = onp.where(mask)[0][0]
                target_val = cond.values[idx_in_cond]
                scol = shock_idx[cond.shock]

                if cond.kind == 'state':
                    s_rows.append(state_idx[cond.variable])
                    s_targets.append(target_val)
                    s_shock_cols.append(scol)
                else:
                    o_rows.append(obs_idx[cond.variable])
                    o_targets.append(target_val)
                    o_shock_cols.append(scol)

            shock_cols = s_shock_cols + o_shock_cols
            # Remove duplicate shock columns (preserve order)
            seen = set()
            unique_cols = []
            for sc in shock_cols:
                if sc not in seen:
                    seen.add(sc)
                    unique_cols.append(sc)

            per_period.append({
                'state_rows': onp.array(s_rows, dtype=int),
                'obs_rows': onp.array(o_rows, dtype=int),
                'shock_cols': onp.array(unique_cols, dtype=int),
                'state_targets': onp.array(s_targets),
                'obs_targets': onp.array(o_targets),
            })

        # Build other_shocks matrix
        other_mat = onp.zeros((self.horizon, n_shock))
        for sname, (periods, values) in self.other_shocks.items():
            si = shock_idx[sname]
            other_mat[periods, si] = values

        return _CompiledPlan(
            horizon=self.horizon,
            per_period=per_period,
            other_shocks_mat=other_mat,
        )


# ======================================================================
# Core forward solver
# ======================================================================

def _forward_conditional(
    T: onp.ndarray,
    R: onp.ndarray,
    Omega: onp.ndarray,
    c: onp.ndarray,
    x_T: onp.ndarray,
    compiled: _CompiledPlan,
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Forward conditional forecast (NumPy loop).

    Args:
        T:        (n_state, n_state) transition matrix.
        R:        (n_state, n_shock) shock impact matrix.
        Omega:    (n_obs, n_state) observation matrix.
        c:        (n_state,) state intercept.
        x_T:      (n_state,) terminal state.
        compiled: precomputed plan from ForecastPlan._compile().

    Returns:
        x_fcst:   (H, n_state) conditioned state paths.
        y_fcst:   (H, n_obs) conditioned observable paths.
        eps_used: (H, n_shock) shock vectors applied.
    """
    H = compiled.horizon
    n_state = T.shape[0]
    n_shock = R.shape[1]
    n_obs = Omega.shape[0]

    # Precompute Omega @ R once
    G = Omega @ R   # (n_obs, n_shock)

    x_fcst = onp.zeros((H, n_state))
    y_fcst = onp.zeros((H, n_obs))
    eps_used = onp.zeros((H, n_shock))

    x_prev = onp.asarray(x_T)

    for h in range(H):
        info = compiled.per_period[h]
        eps_other = compiled.other_shocks_mat[h]

        # 1. Unconditional state + other_shocks
        x_unf = c + T @ x_prev
        x_base = x_unf + R @ eps_other

        eps_full = eps_other.copy()

        s_rows = info['state_rows']
        o_rows = info['obs_rows']
        shock_cols = info['shock_cols']
        n_conds = len(s_rows) + len(o_rows)

        if n_conds > 0 and len(shock_cols) > 0:
            # Build constraint matrix and RHS
            blocks_C = []
            blocks_d = []

            if len(s_rows) > 0:
                C_state = R[onp.ix_(s_rows, shock_cols)]
                d_state = info['state_targets'] - x_base[s_rows]
                blocks_C.append(C_state)
                blocks_d.append(d_state)

            if len(o_rows) > 0:
                y_base = Omega @ x_base
                C_obs = G[onp.ix_(o_rows, shock_cols)]
                d_obs = info['obs_targets'] - y_base[o_rows]
                blocks_C.append(C_obs)
                blocks_d.append(d_obs)

            C = onp.vstack(blocks_C)
            d = onp.concatenate(blocks_d)

            # Solve: minimal-norm (under-det) or least-squares (over-det)
            eps_active, residuals, rank, sv = onp.linalg.lstsq(
                C, d, rcond=None)

            if sv is not None and len(sv) > 0 and sv[-1] > 0:
                cond_num = sv[0] / sv[-1]
                if cond_num > 1e10:
                    warnings.warn(
                        f"Conditioning matrix at h={h} has condition "
                        f"number {cond_num:.1e} -- constraints may be "
                        f"nearly redundant.")

            eps_full[shock_cols] += eps_active

        # Apply shocks
        x_h = x_unf + R @ eps_full
        y_h = Omega @ x_h

        x_fcst[h] = x_h
        y_fcst[h] = y_h
        eps_used[h] = eps_full
        x_prev = x_h

    return x_fcst, y_fcst, eps_used


def _forward_unconditional(
    T: onp.ndarray,
    R: onp.ndarray,
    Omega: onp.ndarray,
    c: onp.ndarray,
    x_T: onp.ndarray,
    horizon: int,
) -> Tuple[onp.ndarray, onp.ndarray]:
    """Forward unconditional (zero-shock) baseline forecast."""
    n_state = T.shape[0]
    n_obs = Omega.shape[0]
    x_fcst = onp.zeros((horizon, n_state))
    y_fcst = onp.zeros((horizon, n_obs))
    x_prev = onp.asarray(x_T)
    for h in range(horizon):
        x_h = c + T @ x_prev
        y_h = Omega @ x_h
        x_fcst[h] = x_h
        y_fcst[h] = y_h
        x_prev = x_h
    return x_fcst, y_fcst


# ======================================================================
# Fixed-parameter conditional forecast
# ======================================================================

def forecast_conditional(
    model,
    param_dict: Dict[str, float],
    ys: jnp.ndarray,
    H_obs: jnp.ndarray,
    init_x: jnp.ndarray,
    init_P: jnp.ndarray,
    plan: ForecastPlan,
    n_tv: Optional[int] = None,
    H_t: Optional[jnp.ndarray] = None,
    shock_multipliers: Optional[jnp.ndarray] = None,
    use_diffuse: bool = False,
) -> ConditionalForecastResult:
    """Conditional forecast at fixed parameters.

    Solves the model, runs the Kalman smoother to obtain x_T, then
    runs the forward conditional inversion and also produces the
    unconditional baseline for comparison.

    Args:
        model:      DynareModelWithLambdified instance.
        param_dict: parameter name -> value.
        ys:         (T_data, n_obs) observed data.
        H_obs:      (n_obs, n_obs) measurement noise covariance.
        init_x:     (n_aug,) initial state mean.
        init_P:     (n_aug, n_aug) initial state covariance.
        plan:       ForecastPlan with conditions.
        n_tv:       time-varying filter steps.
        use_diffuse: if True, recover x_T via the exact diffuse Kalman
                    smoother (D&K 2012).  Required for models with
                    unit-root states.  Incompatible with
                    H_t / shock_multipliers / n_tv.

    Returns:
        ConditionalForecastResult (J=1 draw dimension).
    """
    if use_diffuse and (H_t is not None or shock_multipliers is not None
                        or n_tv is not None):
        raise NotImplementedError(
            "forecast_conditional(use_diffuse=True) is incompatible with "
            "H_t / shock_multipliers / n_tv (the diffuse smoother does "
            "not thread time-varying matrices yet).")

    dt = get_dtype()
    sol = model.solve(param_dict)
    if not sol["solution_valid"]:
        raise RuntimeError("Model does not solve at given parameters.")

    P_aug = onp.asarray(sol["P_aug"])
    R_aug = onp.asarray(sol["R_aug"])
    Omega = onp.asarray(sol["Omega"])
    c_aug = onp.asarray(sol["c_aug"])

    state_names = list(model.aug_state_vars)
    obs_names = list(model.ordered_obs_vars)
    shock_names = list(model.aug_shocks)

    compiled = plan._compile(state_names, obs_names, shock_names)

    if use_diffuse:
        Pinf, Pstar, _ = build_diffuse_init(
            sol["P_aug"], sol["R_aug"], dtype=dt)
        dkf = DiffuseKalmanFilter(
            T=sol["P_aug"], R=sol["R_aug"], C=sol["Omega"], H=H_obs,
            Pinf=Pinf, Pstar=Pstar, init_a=init_x, c=sol["c_aug"], dtype=dt)
        x_smooth = dkf.smooth_states_only(ys=ys)
    else:
        # Build Q_t from shock multipliers if provided
        Q_t = None
        if shock_multipliers is not None:
            mult = jnp.asarray(shock_multipliers, dtype=dt)
            R_sc = sol["R_aug"][None, :, :] * mult[:, None, :]
            Q_t = jnp.einsum('tij,tkj->tik', R_sc, R_sc)

        kf = KalmanFilter(T=sol["P_aug"], R=sol["R_aug"], C=sol["Omega"],
                          H=H_obs, init_x=init_x, init_P=init_P,
                          c=sol["c_aug"], dtype=dt)
        x_smooth = kf.smooth_states_only(ys=ys, n_tv=n_tv, H_t=H_t, Q_t=Q_t)
    x_T = onp.asarray(x_smooth[-1])

    # Conditional forecast
    x_cond, y_cond, eps = _forward_conditional(
        P_aug, R_aug, Omega, c_aug, x_T, compiled)

    # Unconditional baseline
    x_base, y_base = _forward_unconditional(
        P_aug, R_aug, Omega, c_aug, x_T, plan.horizon)

    return ConditionalForecastResult(
        state_forecasts=x_cond[None, ...],      # (1, H, n_state)
        obs_forecasts=y_cond[None, ...],         # (1, H, n_obs)
        shock_paths=eps[None, ...],              # (1, H, n_shock)
        baseline_states=x_base[None, ...],       # (1, H, n_state)
        baseline_obs=y_base[None, ...],          # (1, H, n_obs)
        state_names=state_names,
        obs_names=obs_names,
        shock_names=shock_names,
        horizon=plan.horizon,
    )


# ======================================================================
# Posterior conditional forecast
# ======================================================================

def forecast_conditional_posterior(
    model,
    posterior_samples: Dict[str, Any],
    estimated_names: List[str],
    fixed_params: Dict[str, float],
    ys: jnp.ndarray,
    H_obs: jnp.ndarray,
    init_x: jnp.ndarray,
    init_P: jnp.ndarray,
    plan: ForecastPlan,
    key: jax.Array,
    n_draws: Optional[int] = None,
    use_dk: bool = True,
    n_tv: Optional[int] = None,
    verbose: bool = True,
    batch_size: Optional[int] = 128,
    H_t: Optional[jnp.ndarray] = None,
    shock_multipliers: Optional[jnp.ndarray] = None,
    use_diffuse: bool = False,
) -> ConditionalForecastResult:
    """Conditional forecast across posterior parameter draws.

    For each posterior draw theta^(j):
      1. Solve the model.
      2. Kalman smoother (+ optional DK) -> terminal state x_T.
      3. Run forward conditional inversion.
      4. Run forward unconditional baseline.

    The Kalman smoother (step 2) is the dominant cost and is vectorised
    via ``jax.vmap`` in mini-batches when ``batch_size`` is set.  The
    forward conditional inversion (step 3) uses ``numpy.linalg.lstsq``
    and is kept sequential (it is O(H n^2), negligible relative to the
    filter).

    Args:
        model:              DynareModelWithLambdified instance.
        posterior_samples:   {param_name: array(total_draws,)} from MCMC.
        estimated_names:     names of estimated parameters.
        fixed_params:        dict of fixed parameter values.
        ys:                  (T_data, n_obs) observed data.
        H_obs:               (n_obs, n_obs) measurement noise covariance.
        init_x:              (n_aug,) initial state mean.
        init_P:              (n_aug, n_aug) initial state covariance.
        plan:                ForecastPlan with conditions.
        key:                 JAX PRNG key.
        n_draws:             posterior draws to use (None = all).
        use_dk:              use DK simulation smoother for terminal state.
        n_tv:                time-varying filter steps.
        verbose:             print progress.
        batch_size:          number of draws to process simultaneously via
                             jax.vmap for the Kalman smoother.  Set to None
                             to fall back on the fully sequential loop.

    Returns:
        ConditionalForecastResult with J draws.
    """
    if use_diffuse and (H_t is not None or shock_multipliers is not None
                        or n_tv is not None):
        raise NotImplementedError(
            "forecast_conditional_posterior(use_diffuse=True) is "
            "incompatible with H_t / shock_multipliers / n_tv (the "
            "diffuse smoother does not thread time-varying matrices).")

    dt = get_dtype()

    total_draws = posterior_samples[next(iter(posterior_samples))].shape[0]
    if n_draws is None:
        n_draws = total_draws
    n_use = min(n_draws, total_draws)
    step_size = max(1, total_draws // n_use)
    indices = onp.arange(0, total_draws, step_size)[:n_use]

    key_dk, _ = random.split(key)
    keys_dk = random.split(key_dk, len(indices))

    state_names = list(model.aug_state_vars)
    obs_names = list(model.ordered_obs_vars)
    shock_names = list(model.aug_shocks)

    compiled = plan._compile(state_names, obs_names, shock_names)

    # ------------------------------------------------------------------
    # Phase 1: Solve the model at each posterior draw (sequential).
    # ------------------------------------------------------------------
    P_list, R_list, Om_list, c_list = [], [], [], []
    Pinf_list, Pstar_list = [], []
    valid_dk_keys = []
    n_skipped = 0

    for i, mcmc_idx in enumerate(indices):
        pdict = {name: posterior_samples[name][mcmc_idx]
                 for name in estimated_names}
        pdict.update(fixed_params)

        sol = model.solve(pdict)
        if not sol["solution_valid"]:
            n_skipped += 1
            continue

        P_list.append(sol["P_aug"])
        R_list.append(sol["R_aug"])
        Om_list.append(sol["Omega"])
        c_list.append(sol["c_aug"])
        if use_diffuse:
            # build_diffuse_init is numpy/scipy.linalg, not vmappable;
            # precompute Pinf/Pstar here for the Phase 2a vmap below.
            Pinf_j, Pstar_j, _ = build_diffuse_init(
                sol["P_aug"], sol["R_aug"], dtype=dt)
            Pinf_list.append(Pinf_j)
            Pstar_list.append(Pstar_j)
        valid_dk_keys.append(keys_dk[i])

    if not P_list:
        raise RuntimeError(
            f"No valid conditional forecast draws "
            f"({n_skipped} model solves failed out of {len(indices)}).")

    if verbose and n_skipped > 0:
        print(f"  forecast_conditional_posterior: {n_skipped}/{len(indices)} "
              f"draws skipped (invalid solutions)")

    J_valid = len(P_list)

    # ------------------------------------------------------------------
    # Phase 2a: Batched Kalman smooth + DK -> terminal states x_T.
    # ------------------------------------------------------------------
    if batch_size is not None:
        P_b = jnp.stack(P_list)
        R_b = jnp.stack(R_list)
        Om_b = jnp.stack(Om_list)
        c_b = jnp.stack(c_list)
        dk_k = jnp.stack(valid_dk_keys)
        if use_diffuse:
            Pinf_b = jnp.asarray(onp.stack(Pinf_list), dtype=dt)
            Pstar_b = jnp.asarray(onp.stack(Pstar_list), dtype=dt)
        else:
            Pinf_b = jnp.zeros((J_valid, 1, 1), dtype=dt)
            Pstar_b = jnp.zeros((J_valid, 1, 1), dtype=dt)

        # Pre-convert multipliers to JAX if provided (shared across draws)
        mult_jax = jnp.asarray(shock_multipliers, dtype=dt) \
            if shock_multipliers is not None else None

        def _smooth_single(P, R, Om, c, Pinf, Pstar, dk_key):
            # Build per-draw Q_t from this draw's R and shared multipliers
            Q_t_draw = None
            if mult_jax is not None:
                R_scaled = R[None, :, :] * mult_jax[:, None, :]
                Q_t_draw = jnp.einsum('tij,tkj->tik', R_scaled, R_scaled)

            if use_diffuse:
                dkf = DiffuseKalmanFilter(
                    T=P, R=R, C=Om, H=H_obs,
                    Pinf=Pinf, Pstar=Pstar, init_a=init_x, c=c, dtype=dt)
                x_smooth = dkf.smooth_states_only(ys=ys)
            else:
                kf = KalmanFilter(T=P, R=R, C=Om, H=H_obs,
                                  init_x=init_x, init_P=init_P,
                                  c=c, dtype=dt)
                x_smooth = kf.smooth_states_only(
                    ys=ys, n_tv=n_tv, H_t=H_t, Q_t=Q_t_draw)
            if use_dk:
                dk_obj = DKSimulationSmoother(
                    H=H_obs, init_x=init_x, init_P=init_P, dtype=dt)
                x_draw = dk_obj.draw(P, R, Om, ys, dk_key, x_smooth, c=c)
                return x_draw[-1]
            else:
                return x_smooth[-1]

        _batched_smooth = jax.vmap(_smooth_single)

        xT_chunks = []
        for start in range(0, J_valid, batch_size):
            end = min(start + batch_size, J_valid)
            xT_batch = _batched_smooth(
                P_b[start:end], R_b[start:end], Om_b[start:end],
                c_b[start:end],
                Pinf_b[start:end], Pstar_b[start:end],
                dk_k[start:end])
            xT_chunks.append(onp.asarray(xT_batch))
            if verbose:
                print(f"  forecast_conditional_posterior: "
                      f"smooth batch {end}/{J_valid}")

        x_T_all = onp.concatenate(xT_chunks, axis=0)   # (J, n_aug)

    else:
        # Sequential terminal-state computation
        dk = DKSimulationSmoother(
            H=H_obs, init_x=init_x, init_P=init_P,
            dtype=dt) if use_dk else None

        x_T_all = onp.zeros((J_valid, P_list[0].shape[0]))
        for j in range(J_valid):
            # Build per-draw Q_t from this draw's R and shared multipliers
            Q_t_j = None
            if shock_multipliers is not None:
                mult = jnp.asarray(shock_multipliers, dtype=dt)
                R_j = R_list[j]
                R_scaled = R_j[None, :, :] * mult[:, None, :]
                Q_t_j = jnp.einsum('tij,tkj->tik', R_scaled, R_scaled)

            if use_diffuse:
                # Pinf_list / Pstar_list were populated in Phase 1.
                dkf = DiffuseKalmanFilter(
                    T=P_list[j], R=R_list[j], C=Om_list[j], H=H_obs,
                    Pinf=Pinf_list[j], Pstar=Pstar_list[j],
                    init_a=init_x, c=c_list[j], dtype=dt)
                x_smooth = dkf.smooth_states_only(ys=ys)
            else:
                kf = KalmanFilter(T=P_list[j], R=R_list[j], C=Om_list[j],
                                  H=H_obs, init_x=init_x, init_P=init_P,
                                  c=c_list[j], dtype=dt)
                x_smooth = kf.smooth_states_only(
                    ys=ys, n_tv=n_tv, H_t=H_t, Q_t=Q_t_j)
            if use_dk:
                x_draw = dk.draw(P_list[j], R_list[j], Om_list[j],
                                 ys, valid_dk_keys[j], x_smooth,
                                 c=c_list[j])
                x_T_all[j] = onp.asarray(x_draw[-1])
            else:
                x_T_all[j] = onp.asarray(x_smooth[-1])

    # ------------------------------------------------------------------
    # Phase 2b: Forward conditional + unconditional baseline (sequential,
    # _forward_conditional uses onp.linalg.lstsq, not JAX-traceable).
    # ------------------------------------------------------------------
    all_x_cond, all_y_cond, all_eps = [], [], []
    all_x_base, all_y_base = [], []

    for j in range(J_valid):
        P_np = onp.asarray(P_list[j])
        R_np = onp.asarray(R_list[j])
        Om_np = onp.asarray(Om_list[j])
        c_np = onp.asarray(c_list[j])
        x_T = x_T_all[j]

        x_cond, y_cond, eps = _forward_conditional(
            P_np, R_np, Om_np, c_np, x_T, compiled)
        x_base, y_base = _forward_unconditional(
            P_np, R_np, Om_np, c_np, x_T, plan.horizon)

        all_x_cond.append(x_cond)
        all_y_cond.append(y_cond)
        all_eps.append(eps)
        all_x_base.append(x_base)
        all_y_base.append(y_base)

    if verbose:
        print(f"  forecast_conditional_posterior: {J_valid} draws, "
              f"horizon={plan.horizon}")

    return ConditionalForecastResult(
        state_forecasts=onp.stack(all_x_cond, axis=0),
        obs_forecasts=onp.stack(all_y_cond, axis=0),
        shock_paths=onp.stack(all_eps, axis=0),
        baseline_states=onp.stack(all_x_base, axis=0),
        baseline_obs=onp.stack(all_y_base, axis=0),
        state_names=state_names,
        obs_names=obs_names,
        shock_names=shock_names,
        horizon=plan.horizon,
    )


# ======================================================================
# Accessor helpers
# ======================================================================

def get_cond_state_forecast(
    result: ConditionalForecastResult, var_name: str,
) -> onp.ndarray:
    """Extract (J, H) conditioned state forecast draws."""
    idx = result.state_names.index(var_name)
    return result.state_forecasts[:, :, idx]


def get_cond_obs_forecast(
    result: ConditionalForecastResult, var_name: str,
) -> onp.ndarray:
    """Extract (J, H) conditioned observable forecast draws."""
    idx = result.obs_names.index(var_name)
    return result.obs_forecasts[:, :, idx]


def get_conditioning_shocks(
    result: ConditionalForecastResult, shock_name: str,
) -> onp.ndarray:
    """Extract (J, H) implied shock paths for a given shock."""
    idx = result.shock_names.index(shock_name)
    return result.shock_paths[:, :, idx]


def get_baseline_obs(
    result: ConditionalForecastResult, var_name: str,
) -> onp.ndarray:
    """Extract (J, H) unconditional baseline observable paths."""
    idx = result.obs_names.index(var_name)
    return result.baseline_obs[:, :, idx]


def get_baseline_state(
    result: ConditionalForecastResult, var_name: str,
) -> onp.ndarray:
    """Extract (J, H) unconditional baseline state paths."""
    idx = result.state_names.index(var_name)
    return result.baseline_states[:, :, idx]


def cond_forecast_percentiles(
    draws: onp.ndarray,
    percentiles: tuple = (5, 16, 50, 84, 95),
) -> Dict[int, onp.ndarray]:
    """Compute percentile bands from (J, H) draws.

    Returns:
        {percentile: (H,) array}.
    """
    return {p: onp.percentile(draws, p, axis=0) for p in percentiles}


# ======================================================================
# Plotting
# ======================================================================

def plot_conditional_forecast(
    result: ConditionalForecastResult,
    variables: List[str],
    kind: str = 'obs',
    hist_data: Optional[onp.ndarray] = None,
    hist_dates: Optional[onp.ndarray] = None,
    fcst_dates: Optional[onp.ndarray] = None,
    show_baseline: bool = True,
    percentiles: tuple = (5, 16, 50, 84, 95),
    ncols: int = 2,
    figsize_per_ax: Tuple[float, float] = (7.0, 3.5),
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot conditional forecast fan charts.

    For each variable: filled bands for conditioned path, dashed grey
    for unconditional baseline, solid black for historical data.

    Args:
        result:       ConditionalForecastResult.
        variables:    variable names to plot.
        kind:         'obs' or 'state'.
        hist_data:    (T_hist, n_var) historical data matching variables.
        hist_dates:   date array for historical data.
        fcst_dates:   date array for forecast periods.
        show_baseline: overlay unconditional baseline median.
        percentiles:  percentile levels for fan chart.
        ncols:        columns in subplot grid.
        figsize_per_ax: (width, height) per subplot.
        title:        figure title.
        show:         call plt.show().

    Returns:
        (fig, axes).
    """
    import matplotlib.pyplot as plt

    n_vars = len(variables)
    nrows = max(1, (n_vars + ncols - 1) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False)
    if title:
        fig.suptitle(title, fontsize=13)

    H = result.horizon

    for vi, var_name in enumerate(variables):
        ax = axes[vi // ncols, vi % ncols]

        if kind == 'obs':
            draws = get_cond_obs_forecast(result, var_name)
            base_draws = get_baseline_obs(result, var_name)
        else:
            draws = get_cond_state_forecast(result, var_name)
            base_draws = get_baseline_state(result, var_name)

        bands = cond_forecast_percentiles(draws, percentiles)
        x_ax = fcst_dates if fcst_dates is not None else onp.arange(H)

        # Historical data
        if hist_data is not None and hist_dates is not None:
            if kind == 'obs':
                hi = result.obs_names.index(var_name)
            else:
                hi = result.state_names.index(var_name)
            if hi < hist_data.shape[1]:
                ax.plot(hist_dates, hist_data[:, hi], 'k-', lw=1.2,
                        label='Historical')

        # Fan chart
        pcts = sorted(percentiles)
        alphas = [0.15, 0.30]
        for ai, (lo, hi) in enumerate(
                zip(pcts[:len(pcts)//2], pcts[len(pcts)//2+1:][::-1])):
            ax.fill_between(x_ax, bands[lo], bands[hi],
                            alpha=alphas[min(ai, len(alphas)-1)],
                            color='steelblue')
        # Median
        if 50 in bands:
            ax.plot(x_ax, bands[50], '-', color='steelblue', lw=1.5,
                    label='Conditional (median)')

        # Baseline
        if show_baseline:
            base_med = onp.median(base_draws, axis=0)
            ax.plot(x_ax, base_med, '--', color='grey', lw=1.2,
                    label='Unconditional')

        ax.set_title(var_name, fontsize=11, fontweight='bold')
        ax.grid(True, ls=':', alpha=0.3)
        ax.axhline(0, color='grey', lw=0.5, ls=':')
        if vi == 0:
            ax.legend(fontsize='x-small', loc='best')

    for j in range(n_vars, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)
    if show:
        plt.show()
    return fig, axes


def plot_conditioning_shocks(
    result: ConditionalForecastResult,
    shocks: Optional[List[str]] = None,
    fcst_dates: Optional[onp.ndarray] = None,
    percentiles: tuple = (5, 16, 50, 84, 95),
    ncols: int = 3,
    figsize_per_ax: Tuple[float, float] = (5.0, 3.0),
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot implied shock paths from the conditioning exercise.

    Shows fan charts for each active shock, allowing the user to assess
    whether implied shocks are economically plausible.

    Args:
        result:      ConditionalForecastResult.
        shocks:      shock names to plot (None = all non-zero).
        fcst_dates:  date array for forecast periods.
        percentiles: percentile levels.
        ncols:       columns in subplot grid.
        figsize_per_ax: (width, height) per subplot.
        title:       figure title.
        show:        call plt.show().

    Returns:
        (fig, axes).
    """
    import matplotlib.pyplot as plt

    if shocks is None:
        # Auto-detect: shocks with non-zero median at any period
        shocks = []
        for si, sname in enumerate(result.shock_names):
            med = onp.median(result.shock_paths[:, :, si], axis=0)
            if onp.any(onp.abs(med) > 1e-10):
                shocks.append(sname)
    if not shocks:
        print("  No active conditioning shocks to plot.")
        return None, None

    H = result.horizon
    n_shocks = len(shocks)
    nrows = max(1, (n_shocks + ncols - 1) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False)
    if title:
        fig.suptitle(title, fontsize=13)

    x_ax = fcst_dates if fcst_dates is not None else onp.arange(H)

    for si, sname in enumerate(shocks):
        ax = axes[si // ncols, si % ncols]
        draws = get_conditioning_shocks(result, sname)
        bands = cond_forecast_percentiles(draws, percentiles)

        pcts = sorted(percentiles)
        alphas = [0.15, 0.30]
        for ai, (lo, hi) in enumerate(
                zip(pcts[:len(pcts)//2], pcts[len(pcts)//2+1:][::-1])):
            ax.fill_between(x_ax, bands[lo], bands[hi],
                            alpha=alphas[min(ai, len(alphas)-1)],
                            color='indianred')
        if 50 in bands:
            ax.plot(x_ax, bands[50], '-', color='indianred', lw=1.5)

        ax.axhline(0, color='grey', lw=0.5, ls=':')
        ax.set_title(sname, fontsize=10, fontweight='bold')
        ax.grid(True, ls=':', alpha=0.3)

    for j in range(n_shocks, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)
    if show:
        plt.show()
    return fig, axes

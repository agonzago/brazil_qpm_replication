"""Historical shock decomposition for linear Gaussian state-space models.

Given smoothed structural shocks from the Koopman (1993) disturbance
smoother, decomposes each state variable (or observable) into the
cumulative contribution of each structural shock, plus a deterministic
baseline (initial conditions + mean-reversion drift).

State-space form:
    x_t = c + T x_{t-1} + R eps_t,   eps_t ~ N(0, I)
    y_t = Omega x_t      + eta_t,    eta_t ~ N(0, H)

The decomposition at time t:
    x_t = T^{t+1} x_0 + sum_{s=0}^{t} T^{t-s} c
          + sum_j sum_{s=0}^{t} T^{t-s} R_j * eps_hat_{j,s}

For observables:  y_t^{(j)} = Omega @ x_t^{(j)}.

The "deterministic baseline" absorbs the propagated (smoothed) initial
state and the intercept drift, ensuring an exact decomposition:
    x_smooth[t] = baseline[t] + sum_j contribution_j[t]

Usage (fixed parameters):
    result = shock_decomposition(model, param_dict, ys, H_obs,
                                  init_x, init_P, n_tv=10)

Usage (across posterior):
    result = shock_decomposition_posterior(
        model, posterior_samples, estimated_names, fixed_params,
        ys, H_obs, init_x, init_P, n_draws=100, n_tv=10)

References:
    Koopman, S.J. (1993). "Disturbance Smoother for State Space Models."
        Biometrika, 80(1), 117-126.
"""

from typing import Dict, List, Optional, NamedTuple, Any, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as onp

from qpm_toolbox.kalman_filter import (
    KalmanFilter, DiffuseKalmanFilter, build_diffuse_init, get_dtype,
)


# ======================================================================
# Result containers
# ======================================================================

class ShockDecompositionResult(NamedTuple):
    """Container for historical shock decomposition at fixed parameters.

    The decomposition satisfies (exactly):
        smoothed_states[t] = state_baseline[t] + sum_j state_contrib[j, t]
        smoothed_obs[t]    = obs_baseline[t]   + sum_j obs_contrib[j, t]

    where the baseline absorbs the propagated initial state, the
    deterministic drift from mean-reversion intercepts, and the
    Kalman smoother's update of the initial conditions.

    Attributes:
        state_contrib:   (n_shock, T, n_state)  contribution of each shock.
        state_baseline:  (T, n_state)  deterministic + initial conditions.
        obs_contrib:     (n_shock, T, n_obs)  contribution to observables.
        obs_baseline:    (T, n_obs)  deterministic + initial conditions.
        smoothed_states: (T, n_state)  smoothed state path.
        smoothed_obs:    (T, n_obs)  smoothed observable path.
        eps_hat:         (T, n_shock)  smoothed structural shocks.
        state_names:     list of state variable names.
        obs_names:       list of observable variable names.
        shock_names:     list of shock variable names.
    """
    state_contrib: onp.ndarray
    state_baseline: onp.ndarray
    obs_contrib: onp.ndarray
    obs_baseline: onp.ndarray
    smoothed_states: onp.ndarray
    smoothed_obs: onp.ndarray
    eps_hat: onp.ndarray
    state_names: List[str]
    obs_names: List[str]
    shock_names: List[str]


class PosteriorDecompositionResult(NamedTuple):
    """Container for shock decomposition across posterior draws.

    Each array has a leading draw dimension (J = number of valid draws).

    Attributes:
        state_contrib:   (J, n_shock, T, n_state)  per-draw contributions.
        state_baseline:  (J, T, n_state)  per-draw baselines.
        obs_contrib:     (J, n_shock, T, n_obs)  per-draw observable contribs.
        obs_baseline:    (J, T, n_obs)  per-draw observable baselines.
        smoothed_states: (J, T, n_state)  per-draw smoothed states.
        smoothed_obs:    (J, T, n_obs)  per-draw smoothed observables.
        state_names:     list of state variable names.
        obs_names:       list of observable variable names.
        shock_names:     list of shock variable names.
    """
    state_contrib: onp.ndarray
    state_baseline: onp.ndarray
    obs_contrib: onp.ndarray
    obs_baseline: onp.ndarray
    smoothed_states: onp.ndarray
    smoothed_obs: onp.ndarray
    state_names: List[str]
    obs_names: List[str]
    shock_names: List[str]


# ======================================================================
# Core forward recursion (JAX-traceable)
# ======================================================================

def _forward_decomposition(T_mat, R, c, eps_hat, x_init):
    """Forward recursion computing each shock's cumulative contribution.

    For shock j the recursive formula is:
        contrib_j[-1] = 0
        contrib_j[t]  = T @ contrib_j[t-1] + R[:, j] * eps_hat[j, t]

    All shocks are propagated simultaneously by vectorising across the
    shock dimension (rows of a (n_shock, n_state) matrix).

    Also computes initial-conditions propagation (T^{t+1} x_init) and
    deterministic drift (sum of T^s c) for later adjustment.

    Args:
        T_mat:   (n_state, n_state) transition matrix.
        R:       (n_state, n_shock) shock impact matrix.
        c:       (n_state,) state intercept.
        eps_hat: (T_steps, n_shock) smoothed structural shocks.
        x_init:  (n_state,) initial state mean.

    Returns:
        contrib: (T_steps, n_shock, n_state) shock contributions.
        ic:      (T_steps, n_state)  initial-conditions component.
        drift:   (T_steps, n_state)  deterministic-drift component.
    """
    n_state = T_mat.shape[0]
    dt = T_mat.dtype

    def step(carry, eps_t):
        contrib_prev, ic_prev, drift_prev = carry
        # Shock contributions: row j -> T @ row_j + R[:,j] * eps_j
        contrib_next = contrib_prev @ T_mat.T + eps_t[:, None] * R.T
        ic_next = T_mat @ ic_prev
        drift_next = T_mat @ drift_prev + c
        return (contrib_next, ic_next, drift_next), \
               (contrib_next, ic_next, drift_next)

    n_shock = R.shape[1]
    init = (
        jnp.zeros((n_shock, n_state), dtype=dt),
        jnp.asarray(x_init, dtype=dt),
        jnp.zeros(n_state, dtype=dt),
    )
    _, (contrib, ic, drift) = lax.scan(step, init, eps_hat)
    return contrib, ic, drift          # each: (T, ...)


# ======================================================================
# Fixed-parameter decomposition
# ======================================================================

def shock_decomposition(
    model,
    param_dict: Dict[str, float],
    ys: jnp.ndarray,
    H_obs: jnp.ndarray,
    init_x: jnp.ndarray,
    init_P: jnp.ndarray,
    n_tv: Optional[int] = None,
    H_t: Optional[jnp.ndarray] = None,
    shock_multipliers: Optional[jnp.ndarray] = None,
    use_diffuse: bool = False,
) -> ShockDecompositionResult:
    """Historical shock decomposition at fixed parameters.

    Steps:
      1. Solve the model.
      2. Run the Kalman smoother (standard RTS or exact diffuse).
      3. Forward-iterate the decomposition from smoothed shocks.
      4. Adjust the baseline so the decomposition is exact.

    Args:
        model:      DynareModelWithLambdified instance.
        param_dict: parameter name -> value.
        ys:         (T, n_obs) observed data.
        H_obs:      (n_obs, n_obs) measurement noise covariance.
        init_x:     (n_aug,) initial state mean.
        init_P:     (n_aug, n_aug) initial state covariance.
        n_tv:       time-varying filter steps (passed to filter_full).
        use_diffuse: if True, use the exact diffuse Kalman smoother
                     (required for models with unit-root states).

    Returns:
        ShockDecompositionResult.
    """
    dt = get_dtype()

    sol = model.solve(param_dict)
    if not sol["solution_valid"]:
        raise RuntimeError("Model does not solve at given parameters.")

    P_aug = sol["P_aug"]
    R_aug = sol["R_aug"]
    Omega = sol["Omega"]
    c_aug = sol["c_aug"]

    # Build Q_t from shock multipliers if provided
    Q_t = None
    if shock_multipliers is not None:
        mult = jnp.asarray(shock_multipliers, dtype=dt)
        R_scaled = R_aug[None, :, :] * mult[:, None, :]
        Q_t = jnp.einsum('tij,tkj->tik', R_scaled, R_scaled)

    return _decompose_from_matrices(
        P_aug, R_aug, Omega, c_aug, H_obs, init_x, init_P, ys,
        model.aug_state_vars, model.ordered_obs_vars, model.aug_shocks,
        n_tv=n_tv, H_t=H_t, Q_t=Q_t, dt=dt, use_diffuse=use_diffuse)


def _decompose_from_matrices(
    P_aug, R_aug, Omega, c_aug, H_obs, init_x, init_P, ys,
    state_names, obs_names, shock_names,
    n_tv=None, H_t=None, Q_t=None, dt=None,
    use_diffuse=False,
):
    """Decomposition given pre-solved system matrices (internal).

    When ``use_diffuse=True``, the exact diffuse Kalman smoother
    (Durbin & Koopman 2012) is used instead of the standard RTS smoother.
    This is required for models with unit-root states.
    """
    if dt is None:
        dt = get_dtype()

    if use_diffuse:
        Pinf, Pstar, _ = build_diffuse_init(P_aug, R_aug, dtype=dt)
        dkf = DiffuseKalmanFilter(
            T=P_aug, R=R_aug, C=Omega, H=H_obs,
            Pinf=Pinf, Pstar=Pstar, init_a=init_x, c=c_aug, dtype=dt)
        sm = dkf.smooth(ys)
        eps_hat_jax = sm['etahat']
        x_smooth = onp.asarray(sm['alphahat'])
    else:
        kf = KalmanFilter(T=P_aug, R=R_aug, C=Omega, H=H_obs,
                           init_x=init_x, init_P=init_P, c=c_aug, dtype=dt)

        fr = kf.filter_full(ys, n_tv=n_tv, H_t=H_t, Q_t=Q_t)
        eps_hat_jax, _ = kf.smooth_disturbances(filter_results=fr)
        x_smooth = onp.asarray(kf.smooth_states_only(filter_results=fr))

    # Forward decomposition (in JAX)
    contrib, ic, drift = _forward_decomposition(
        P_aug, R_aug, c_aug, eps_hat_jax, init_x)

    contrib = onp.asarray(contrib)      # (T, n_shock, n_state)
    ic = onp.asarray(ic)                # (T, n_state)
    drift = onp.asarray(drift)          # (T, n_state)
    eps_hat = onp.asarray(eps_hat_jax)  # (T, n_shock)

    # Exact baseline: absorb any smoother-initial-state discrepancy
    total_shocks = contrib.sum(axis=1)              # (T, n_state)
    state_baseline = x_smooth - total_shocks        # (T, n_state)

    # Observable decomposition via Omega
    Omega_np = onp.asarray(Omega)
    # obs_contrib[t, j, k] = sum_i Omega[k, i] * contrib[t, j, i]
    obs_contrib = onp.einsum('ki,tji->tjk', Omega_np, contrib)
    smoothed_obs = (Omega_np @ x_smooth.T).T
    obs_baseline = smoothed_obs - obs_contrib.sum(axis=1)

    return ShockDecompositionResult(
        state_contrib=contrib.transpose(1, 0, 2),       # (n_shock, T, n_state)
        state_baseline=state_baseline,
        obs_contrib=obs_contrib.transpose(1, 0, 2),      # (n_shock, T, n_obs)
        obs_baseline=obs_baseline,
        smoothed_states=x_smooth,
        smoothed_obs=smoothed_obs,
        eps_hat=eps_hat,
        state_names=list(state_names),
        obs_names=list(obs_names),
        shock_names=list(shock_names),
    )


# ======================================================================
# Posterior decomposition
# ======================================================================

def shock_decomposition_posterior(
    model,
    posterior_samples: Dict[str, Any],
    estimated_names: List[str],
    fixed_params: Dict[str, float],
    ys: jnp.ndarray,
    H_obs: jnp.ndarray,
    init_x: jnp.ndarray,
    init_P: jnp.ndarray,
    n_draws: Optional[int] = None,
    n_tv: Optional[int] = None,
    verbose: bool = True,
    batch_size: Optional[int] = 128,
    H_t: Optional[jnp.ndarray] = None,
    shock_multipliers: Optional[jnp.ndarray] = None,
) -> PosteriorDecompositionResult:
    """Historical shock decomposition across posterior parameter draws.

    For each posterior draw theta^(j):
      1. Solve the model for theta^(j).
      2. Run filter + disturbance smoother + RTS smoother.
      3. Compute the forward decomposition.

    Args:
        model:              DynareModelWithLambdified instance.
        posterior_samples:   {param_name: array(total_draws,)} from MCMC.
        estimated_names:     names of estimated parameters.
        fixed_params:        dict of fixed (non-estimated) parameter values.
        ys:                  (T_data, n_obs) observed data.
        H_obs:               (n_obs, n_obs) measurement noise covariance.
        init_x:              (n_aug,) initial state mean.
        init_P:              (n_aug, n_aug) initial state covariance.
        n_draws:             how many posterior draws to use (None = all).
        n_tv:                time-varying filter steps (passed to filter_full).
        verbose:             print progress information.
        batch_size:          number of draws to process simultaneously via
                             jax.vmap.  Controls peak device memory.  Set to
                             None to fall back on the sequential loop.

    Returns:
        PosteriorDecompositionResult.
    """
    dt = get_dtype()

    total_draws = posterior_samples[next(iter(posterior_samples))].shape[0]
    if n_draws is None:
        n_draws = total_draws
    n_use = min(n_draws, total_draws)
    step_size = max(1, total_draws // n_use)
    indices = onp.arange(0, total_draws, step_size)[:n_use]

    # ------------------------------------------------------------------
    # Phase 1: Solve the model at each posterior draw (sequential).
    # ------------------------------------------------------------------
    P_list, R_list, Om_list, c_list = [], [], [], []
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

    if not P_list:
        raise RuntimeError(
            f"No valid decomposition draws ({n_skipped} model solves failed "
            f"out of {len(indices)}).")

    if verbose and n_skipped > 0:
        print(f"  shock_decomposition_posterior: {n_skipped}/{len(indices)} "
              f"draws skipped (invalid model solutions)")

    J_valid = len(P_list)

    # ------------------------------------------------------------------
    # Phase 2: Filter + disturbance smoother + forward decomposition.
    # ------------------------------------------------------------------
    if batch_size is not None:
        # ---- Batched path (jax.vmap in mini-batches) ----
        P_b = jnp.stack(P_list)
        R_b = jnp.stack(R_list)
        Om_b = jnp.stack(Om_list)
        c_b = jnp.stack(c_list)

        # Pre-convert multipliers to JAX if provided (shared across draws)
        mult_jax = jnp.asarray(shock_multipliers, dtype=dt) \
            if shock_multipliers is not None else None

        def _single(P, R, Om, c):
            # Build per-draw Q_t from this draw's R and shared multipliers
            Q_t_draw = None
            if mult_jax is not None:
                R_scaled = R[None, :, :] * mult_jax[:, None, :]
                Q_t_draw = jnp.einsum('tij,tkj->tik', R_scaled, R_scaled)

            kf = KalmanFilter(T=P, R=R, C=Om, H=H_obs,
                              init_x=init_x, init_P=init_P,
                              c=c, dtype=dt)
            fr = kf.filter_full(ys, n_tv=n_tv, H_t=H_t, Q_t=Q_t_draw)
            eps_hat, _ = kf.smooth_disturbances(filter_results=fr)
            x_smooth = kf.smooth_states_only(filter_results=fr)

            contrib, _ic, _drift = _forward_decomposition(
                P, R, c, eps_hat, init_x)

            # Stay in JAX (no onp.asarray mid-computation)
            total_shocks = contrib.sum(axis=1)
            state_baseline = x_smooth - total_shocks
            obs_contrib = jnp.einsum('ki,tji->tjk', Om, contrib)
            smoothed_obs = (Om @ x_smooth.T).T
            obs_baseline = smoothed_obs - obs_contrib.sum(axis=1)

            return (contrib.transpose(1, 0, 2), state_baseline,
                    obs_contrib.transpose(1, 0, 2), obs_baseline,
                    x_smooth, smoothed_obs)

        _batched = jax.vmap(_single)

        chunks_sc, chunks_sb = [], []
        chunks_oc, chunks_ob = [], []
        chunks_xs, chunks_ys = [], []
        for start in range(0, J_valid, batch_size):
            end = min(start + batch_size, J_valid)
            sc, sb, oc, ob, xs, ys_sm = _batched(
                P_b[start:end], R_b[start:end],
                Om_b[start:end], c_b[start:end])
            chunks_sc.append(onp.asarray(sc))
            chunks_sb.append(onp.asarray(sb))
            chunks_oc.append(onp.asarray(oc))
            chunks_ob.append(onp.asarray(ob))
            chunks_xs.append(onp.asarray(xs))
            chunks_ys.append(onp.asarray(ys_sm))
            if verbose:
                print(f"  shock_decomposition_posterior: batch {end}/{J_valid}")

        all_sc = onp.concatenate(chunks_sc, axis=0)
        all_sb = onp.concatenate(chunks_sb, axis=0)
        all_oc = onp.concatenate(chunks_oc, axis=0)
        all_ob = onp.concatenate(chunks_ob, axis=0)
        all_xs = onp.concatenate(chunks_xs, axis=0)
        all_ys = onp.concatenate(chunks_ys, axis=0)

    else:
        # ---- Sequential fallback ----
        all_sc, all_sb = [], []
        all_oc, all_ob = [], []
        all_xs, all_ys = [], []

        for j in range(J_valid):
            # Build per-draw Q_t from this draw's R and shared multipliers
            Q_t_j = None
            if shock_multipliers is not None:
                mult = jnp.asarray(shock_multipliers, dtype=dt)
                R_j = R_list[j]
                R_scaled = R_j[None, :, :] * mult[:, None, :]
                Q_t_j = jnp.einsum('tij,tkj->tik', R_scaled, R_scaled)

            res = _decompose_from_matrices(
                P_list[j], R_list[j], Om_list[j], c_list[j],
                H_obs, init_x, init_P, ys,
                model.aug_state_vars, model.ordered_obs_vars,
                model.aug_shocks, n_tv=n_tv, H_t=H_t, Q_t=Q_t_j, dt=dt)

            all_sc.append(res.state_contrib)
            all_sb.append(res.state_baseline)
            all_oc.append(res.obs_contrib)
            all_ob.append(res.obs_baseline)
            all_xs.append(res.smoothed_states)
            all_ys.append(res.smoothed_obs)

            if verbose and (j + 1) % 50 == 0:
                print(f"  shock_decomposition_posterior: "
                      f"{j+1}/{J_valid} draws processed")

        all_sc = onp.stack(all_sc, axis=0)
        all_sb = onp.stack(all_sb, axis=0)
        all_oc = onp.stack(all_oc, axis=0)
        all_ob = onp.stack(all_ob, axis=0)
        all_xs = onp.stack(all_xs, axis=0)
        all_ys = onp.stack(all_ys, axis=0)

    if verbose:
        print(f"  shock_decomposition_posterior: {J_valid} valid draws")

    return PosteriorDecompositionResult(
        state_contrib=all_sc,
        state_baseline=all_sb,
        obs_contrib=all_oc,
        obs_baseline=all_ob,
        smoothed_states=all_xs,
        smoothed_obs=all_ys,
        state_names=list(model.aug_state_vars),
        obs_names=list(model.ordered_obs_vars),
        shock_names=list(model.aug_shocks),
    )


# ======================================================================
# Accessor helpers
# ======================================================================

def get_state_decomposition(
    result: ShockDecompositionResult,
    var_name: str,
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Extract the decomposition for a single state variable.

    Args:
        result:   ShockDecompositionResult (fixed-parameter).
        var_name: name of the state variable (must be in state_names).

    Returns:
        contributions: (n_shock, T)  each shock's contribution.
        baseline:      (T,)  deterministic baseline.
        total:         (T,)  smoothed state (= baseline + sum of contribs).
    """
    idx = result.state_names.index(var_name)
    return (
        result.state_contrib[:, :, idx],
        result.state_baseline[:, idx],
        result.smoothed_states[:, idx],
    )


def get_obs_decomposition(
    result: ShockDecompositionResult,
    var_name: str,
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Extract the decomposition for a single observable.

    Args:
        result:   ShockDecompositionResult (fixed-parameter).
        var_name: name of the observable (must be in obs_names).

    Returns:
        contributions: (n_shock, T)  each shock's contribution.
        baseline:      (T,)  deterministic baseline.
        total:         (T,)  smoothed observable (= baseline + sum of contribs).
    """
    idx = result.obs_names.index(var_name)
    return (
        result.obs_contrib[:, :, idx],
        result.obs_baseline[:, idx],
        result.smoothed_obs[:, idx],
    )


def decomposition_percentiles(
    posterior_result: PosteriorDecompositionResult,
    var_name: str,
    kind: str = 'state',
    percentiles: tuple = (5, 16, 50, 84, 95),
) -> Dict[str, Dict[int, onp.ndarray]]:
    """Compute percentile bands from the posterior decomposition.

    Args:
        posterior_result: PosteriorDecompositionResult.
        var_name:    variable name.
        kind:        'state' or 'obs'.
        percentiles: tuple of percentile values.

    Returns:
        dict with keys:
            'total':     {pct: (T,) array}
            'baseline':  {pct: (T,) array}
            shock_name:  {pct: (T,) array}  for each shock.
    """
    if kind == 'state':
        names = posterior_result.state_names
        contrib = posterior_result.state_contrib      # (J, n_shock, T, n_state)
        baseline = posterior_result.state_baseline    # (J, T, n_state)
        total_arr = posterior_result.smoothed_states  # (J, T, n_state)
    else:
        names = posterior_result.obs_names
        contrib = posterior_result.obs_contrib
        baseline = posterior_result.obs_baseline
        total_arr = posterior_result.smoothed_obs

    idx = names.index(var_name)
    shock_names = posterior_result.shock_names

    out = {}
    out['total'] = {p: onp.percentile(total_arr[:, :, idx], p, axis=0)
                    for p in percentiles}
    out['baseline'] = {p: onp.percentile(baseline[:, :, idx], p, axis=0)
                       for p in percentiles}
    for si, sname in enumerate(shock_names):
        out[sname] = {p: onp.percentile(contrib[:, si, :, idx], p, axis=0)
                      for p in percentiles}
    return out


# ======================================================================
# Plotting
# ======================================================================

def plot_shock_decomposition(
    result: ShockDecompositionResult,
    variables: List[str],
    shock_groups: Dict[str, List[str]],
    kind: str = 'state',
    time_axis: Optional[onp.ndarray] = None,
    ncols: int = 2,
    figsize_per_ax: Tuple[float, float] = (7.0, 3.5),
    colors: Optional[Dict[str, Any]] = None,
    bar_width: float = 0.8,
    title: Optional[str] = None,
    include_baseline: bool = False,
    baseline_group: Optional[str] = None,
    show: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
):
    """Plot historical shock decomposition as stacked bar charts.

    Positive contributions are stacked above zero, negative below.
    The smoothed variable is overlaid as a solid black line.

    Args:
        result:       ShockDecompositionResult (fixed-parameter).
        variables:    list of variable names to plot.
        shock_groups: OrderedDict mapping group_name -> list of shock names.
                      Contributions within each group are summed.
        kind:         'state' for state variables, 'obs' for observables.
        time_axis:    array-like for the x-axis (dates or indices).
        ncols:        columns in the subplot grid.
        figsize_per_ax: (width, height) per subplot.
        colors:       dict mapping group_name -> colour.
        bar_width:    width of each bar (default 0.8).
        title:        overall figure title.
        include_baseline: if True, add the baseline as a separate group.
        baseline_group: if set, merge baseline into this named group
                      instead of showing it separately (implies
                      include_baseline=True).
        show:         if True, call plt.show().

    Returns:
        (fig, axes)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if kind == 'state':
        all_names = result.state_names
        contrib = result.state_contrib       # (n_shock, T, n_state)
        baseline = result.state_baseline     # (T, n_state)
        total = result.smoothed_states       # (T, n_state)
    else:
        all_names = result.obs_names
        contrib = result.obs_contrib
        baseline = result.obs_baseline
        total = result.smoothed_obs

    T_steps = total.shape[0]
    if time_axis is None:
        time_axis = onp.arange(T_steps)

    shock_idx = {s: i for i, s in enumerate(result.shock_names)}
    group_names = list(shock_groups.keys())

    # Resolve baseline handling
    if baseline_group is not None:
        include_baseline = True   # need the data
    show_baseline_separate = include_baseline and (baseline_group is None)
    n_groups = len(group_names) + (1 if show_baseline_separate else 0)

    # Colours
    if colors is None:
        cmap = plt.cm.get_cmap('tab10', n_groups)
        colors = {g: cmap(i) for i, g in enumerate(group_names)}
        if show_baseline_separate:
            colors['Baseline'] = 'lightgrey'

    # Auto-detect bar width for datetime axes
    use_dates = hasattr(time_axis[0], 'strftime') if len(time_axis) > 0 else False
    if use_dates and bar_width < 2.0:
        # Infer spacing from first two dates (in days)
        from datetime import timedelta
        if len(time_axis) >= 2:
            dt_delta = (time_axis[1] - time_axis[0])
            if hasattr(dt_delta, 'days'):
                bar_width = dt_delta.days * 0.85
            else:
                bar_width = 80  # ~quarterly default
        else:
            bar_width = 80

    n_vars = len(variables)
    nrows = max(1, (n_vars + ncols - 1) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False)
    if title:
        fig.suptitle(title, fontsize=13)

    for vi, var_name in enumerate(variables):
        ax = axes[vi // ncols, vi % ncols]
        var_idx = all_names.index(var_name)

        # Compute grouped contributions
        grouped = {}
        for gname, shk_list in shock_groups.items():
            g = onp.zeros(T_steps)
            for shk in shk_list:
                if shk in shock_idx:
                    g += contrib[shock_idx[shk], :, var_idx]
            grouped[gname] = g
        if baseline_group is not None and baseline_group in grouped:
            grouped[baseline_group] = grouped[baseline_group] + baseline[:, var_idx]
        elif show_baseline_separate:
            grouped['Baseline'] = baseline[:, var_idx]

        all_group_names = list(grouped.keys())

        # Stacked bar: positive above zero, negative below
        pos_bottom = onp.zeros(T_steps)
        neg_bottom = onp.zeros(T_steps)

        for gi, gname in enumerate(all_group_names):
            vals = grouped[gname]
            pos_vals = onp.maximum(vals, 0.0)
            neg_vals = onp.minimum(vals, 0.0)
            c = colors.get(gname, f'C{gi}')

            ax.bar(time_axis, pos_vals, bottom=pos_bottom,
                   width=bar_width, color=c, label=gname,
                   edgecolor='none', alpha=0.70, zorder=2)
            ax.bar(time_axis, neg_vals, bottom=neg_bottom,
                   width=bar_width, color=c,
                   edgecolor='none', alpha=0.70, zorder=2)

            pos_bottom = pos_bottom + pos_vals
            neg_bottom = neg_bottom + neg_vals

        # Total (smoothed variable) as solid line - drawn last, high zorder
        ax.plot(time_axis, total[:, var_idx], 'k-', lw=2.5,
                label='Smoothed', zorder=10, alpha=1.0)
        ax.axhline(0, color='grey', lw=0.6, ls=':')

        ax.set_title(var_name, fontsize=11, fontweight='bold')
        ax.grid(True, ls=':', alpha=0.3, axis='y')

        if use_dates:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        if ylim is not None:
            ax.set_ylim(ylim)

        if vi == 0:
            ax.legend(fontsize='xx-small', loc='best', ncol=2)

    for j in range(n_vars, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)
    if show:
        plt.show()
    return fig, axes

"""Unconditional forecasting for linear Gaussian state-space models.

Implements the Del Negro & Schorfheide (2013) unconditional forecast:
given the terminal smoothed state x_T and posterior parameter draws,
simulate forward using the state transition with random future shocks.

State-space form:
    x_t = c + T x_{t-1} + R eps_t,   eps_t ~ N(0, I)
    y_t = Omega x_t     + eta_t,     eta_t ~ N(0, H)

Observable forecasts report E[y_{T+h} | x_{T+h}] = Omega x_{T+h} without
measurement noise, following the standard convention.  The measurement
noise H represents observational error, not an economically meaningful
source of variation.

Usage (single draw):
    x_fcst, y_fcst = forecast_unconditional(T, R, Omega, x_T, horizon, key)

Usage (across posterior):
    result = forecast_posterior(model, posterior_samples, estimated_names,
                                fixed_params, ys, H_obs, init_x, init_P,
                                horizon, key)
    gdp_draws = get_obs_forecast(result, "OBS_LGDP")
    bands = forecast_percentiles(gdp_draws)

Reference:
    Del Negro, M. and Schorfheide, F. (2013). "DSGE Model-Based
    Forecasting." In Handbook of Economic Forecasting, Vol. 2A,
    ch. 2, pp. 57-140.
"""

from typing import Dict, List, Optional, NamedTuple, Any

import jax
import jax.numpy as jnp
from jax import lax, random
import numpy as onp

from qpm_toolbox.kalman_filter import (
    KalmanFilter, DiffuseKalmanFilter, build_diffuse_init, get_dtype,
)
from qpm_toolbox.dk_smoother import DKSimulationSmoother


# ======================================================================
# Single-draw unconditional forecast
# ======================================================================

def forecast_unconditional(
    T: jnp.ndarray,
    R: jnp.ndarray,
    Omega: jnp.ndarray,
    x_T: jnp.ndarray,
    horizon: int,
    key: jax.Array,
    c: Optional[jnp.ndarray] = None,
    stochastic: bool = True,
) -> tuple:
    """Simulate the state-space system forward from x_T.

    When stochastic=True (default), draws random future shocks
    eps_{T+1}, ..., eps_{T+H} and propagates the state equation forward.
    When stochastic=False, sets all future shocks to zero, producing a
    smooth deterministic mean-path forecast x_{T+h} = c + T x_{T+h-1}.

    Observable forecasts are y_{T+h} = Omega x_{T+h} (no measurement noise).

    Args:
        T:           (n_state, n_state) state transition matrix.
        R:           (n_state, n_shock) shock impact matrix.
        Omega:       (n_obs, n_state)   observation matrix.
        x_T:         (n_state,)         terminal state (initial condition).
        horizon:     number of forecast periods H.
        key:         JAX PRNG key for shock draws (unused if stochastic=False).
        c:           (n_state,) state intercept, default zeros.
        stochastic:  if True, draw random future shocks; if False, shocks = 0.

    Returns:
        x_forecast: (H, n_state) forecast state paths.
        y_forecast: (H, n_obs)   forecast observable paths.
    """
    dt = T.dtype
    n_state = T.shape[0]
    n_shock = R.shape[1]

    c_vec = jnp.zeros(n_state, dtype=dt) if c is None \
        else jnp.asarray(c, dtype=dt)

    if stochastic:
        eps_all = random.normal(key, (horizon, n_shock), dtype=dt)
    else:
        eps_all = jnp.zeros((horizon, n_shock), dtype=dt)

    def step(x_prev, eps_t):
        x_next = c_vec + T @ x_prev + R @ eps_t
        y_next = Omega @ x_next
        return x_next, (x_next, y_next)

    _, (x_forecast, y_forecast) = lax.scan(step, x_T, eps_all)

    return x_forecast, y_forecast


# ======================================================================
# Result container
# ======================================================================

class ForecastResult(NamedTuple):
    """Container for forecast output across posterior draws.

    Attributes:
        state_forecasts: (J, H, n_state) state forecast paths.
        obs_forecasts:   (J, H, n_obs)   observable forecast paths.
        state_names:     list of n_state state variable names.
        obs_names:       list of n_obs observable variable names.
        shock_names:     list of n_shock shock variable names.
        horizon:         forecast horizon H.
    """
    state_forecasts: onp.ndarray
    obs_forecasts: onp.ndarray
    state_names: List[str]
    obs_names: List[str]
    shock_names: List[str]
    horizon: int


# ======================================================================
# Posterior forecast aggregation
# ======================================================================

def forecast_posterior(
    model,
    posterior_samples: Dict[str, Any],
    estimated_names: List[str],
    fixed_params: Dict[str, float],
    ys: jnp.ndarray,
    H_obs: jnp.ndarray,
    init_x: jnp.ndarray,
    init_P: jnp.ndarray,
    horizon: int,
    key: jax.Array,
    n_draws: Optional[int] = None,
    use_dk: bool = True,
    n_tv: Optional[int] = None,
    stochastic: bool = True,
    verbose: bool = True,
    batch_size: Optional[int] = 128,
    H_t: Optional[jnp.ndarray] = None,
    shock_multipliers: Optional[jnp.ndarray] = None,
    use_diffuse: bool = False,
) -> ForecastResult:
    """Produce unconditional forecasts across posterior parameter draws.

    For each posterior draw theta^(j):
      1. Solve the model for theta^(j).
      2. Run the Kalman smoother (and optionally the Durbin-Koopman
         simulation smoother) to obtain the terminal state x_T^(j).
      3. Simulate forward H periods (with random future shocks if
         stochastic=True, or deterministically if stochastic=False).

    When stochastic=True (default), the fan charts integrate both
    parameter uncertainty and future shock uncertainty.  When
    stochastic=False, the fan charts reflect only parameter and
    terminal-state uncertainty, producing smooth paths useful for
    verifying gap convergence dynamics.

    Args:
        model:              DynareModelWithLambdified instance.
        posterior_samples:  {param_name: array(total_draws,)} from MCMC.
        estimated_names:    names of estimated parameters.
        fixed_params:       dict of fixed (non-estimated) parameter values.
        ys:                 (T_data, n_obs) observed data.
        H_obs:              (n_obs, n_obs) measurement noise covariance.
        init_x:             (n_aug,) initial state mean.
        init_P:             (n_aug, n_aug) initial state covariance.
        horizon:            number of forecast periods H.
        key:                JAX PRNG key.
        n_draws:            how many posterior draws to use (None = all).
        use_dk:             if True, use Durbin-Koopman simulation smoother
                            for the terminal state; otherwise use RTS mean.
        n_tv:               time-varying filter steps (passed to filter_full).
        stochastic:         if True, draw random future shocks; if False,
                            set future shocks to zero (deterministic paths).
        verbose:            print progress information.
        batch_size:         number of draws to process simultaneously via
                            jax.vmap.  Controls peak device memory.  Set to
                            None to fall back on the sequential loop.
        H_t:                (T, n_obs, n_obs) per-period observation noise
                            covariance (e.g. from tunes with missing data).
                            If None, uses constant H_obs.
        shock_multipliers:  (T, n_shock) per-period scaling of shock std
                            deviations.  Converted to Q_t per draw using
                            each draw's R_aug.
        use_diffuse:        if True, use the exact diffuse Kalman smoother
                            (Durbin & Koopman 2012) to obtain x_T.  Required
                            for models with unit-root states.  Incompatible
                            with H_t / shock_multipliers / n_tv (the diffuse
                            smoother does not yet thread time-varying
                            matrices); pass use_diffuse=False if any are set.

    Returns:
        ForecastResult with shapes (J, H, n_state) and (J, H, n_obs).
    """
    if use_diffuse and (H_t is not None or shock_multipliers is not None
                        or n_tv is not None):
        raise NotImplementedError(
            "forecast_posterior(use_diffuse=True) is incompatible with "
            "H_t / shock_multipliers / n_tv (the diffuse smoother does "
            "not thread time-varying matrices yet).  Pass "
            "use_diffuse=False or remove the time-varying inputs.")
    dt = get_dtype()

    total_draws = posterior_samples[next(iter(posterior_samples))].shape[0]
    if n_draws is None:
        n_draws = total_draws
    n_use = min(n_draws, total_draws)
    step_size = max(1, total_draws // n_use)
    indices = onp.arange(0, total_draws, step_size)[:n_use]

    key_dk, key_fcst = random.split(key)
    keys_dk = random.split(key_dk, len(indices))
    keys_fcst = random.split(key_fcst, len(indices))

    # ------------------------------------------------------------------
    # Phase 1: Solve the model at each posterior draw (sequential).
    # The lambdified symbolic functions are fast but not vmappable, so
    # we loop here and stack only the valid system-matrix solutions.
    # ------------------------------------------------------------------
    P_list, R_list, Om_list, c_list = [], [], [], []
    Pinf_list, Pstar_list = [], []
    valid_dk_keys, valid_fcst_keys = [], []
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
            # build_diffuse_init uses numpy/scipy.linalg and is NOT
            # JAX-traceable; precompute Pinf/Pstar here so we can stack
            # them and pass into the vmapped Phase 2 below.
            Pinf_j, Pstar_j, _ = build_diffuse_init(
                sol["P_aug"], sol["R_aug"], dtype=dt)
            Pinf_list.append(Pinf_j)
            Pstar_list.append(Pstar_j)
        valid_dk_keys.append(keys_dk[i])
        valid_fcst_keys.append(keys_fcst[i])

    if not P_list:
        raise RuntimeError(
            "No valid forecast draws produced "
            f"({n_skipped} model solves failed out of {len(indices)}).")

    if verbose and n_skipped > 0:
        print(f"  forecast_posterior: {n_skipped}/{len(indices)} draws "
              f"skipped (invalid model solutions)")

    J_valid = len(P_list)

    # ------------------------------------------------------------------
    # Phase 2: Kalman smooth + (optional DK) + forecast.
    # ------------------------------------------------------------------
    if batch_size is not None:
        # ---- Batched path (jax.vmap in mini-batches) ----
        P_b = jnp.stack(P_list)
        R_b = jnp.stack(R_list)
        Om_b = jnp.stack(Om_list)
        c_b = jnp.stack(c_list)
        dk_k = jnp.stack(valid_dk_keys)
        fc_k = jnp.stack(valid_fcst_keys)
        if use_diffuse:
            Pinf_b = jnp.asarray(onp.stack(Pinf_list), dtype=dt)
            Pstar_b = jnp.asarray(onp.stack(Pstar_list), dtype=dt)
        else:
            # Stand-in arrays for the vmap signature; never read on the
            # standard-KF branch.
            Pinf_b = jnp.zeros((J_valid, 1, 1), dtype=dt)
            Pstar_b = jnp.zeros((J_valid, 1, 1), dtype=dt)

        # Pre-convert multipliers to JAX if provided (shared across draws)
        mult_jax = jnp.asarray(shock_multipliers, dtype=dt) \
            if shock_multipliers is not None else None

        def _single(P, R, Om, c, Pinf, Pstar, dk_key, fcst_key):
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
                x_T = x_draw[-1]
            else:
                x_T = x_smooth[-1]
            return forecast_unconditional(
                P, R, Om, x_T, horizon, fcst_key, c=c,
                stochastic=stochastic)

        _batched = jax.vmap(_single)

        all_x, all_y = [], []
        for start in range(0, J_valid, batch_size):
            end = min(start + batch_size, J_valid)
            xf, yf = _batched(
                P_b[start:end], R_b[start:end], Om_b[start:end],
                c_b[start:end],
                Pinf_b[start:end], Pstar_b[start:end],
                dk_k[start:end], fc_k[start:end])
            all_x.append(onp.asarray(xf))
            all_y.append(onp.asarray(yf))
            if verbose:
                print(f"  forecast_posterior: batch {end}/{J_valid}")

        x_all = onp.concatenate(all_x, axis=0)
        y_all = onp.concatenate(all_y, axis=0)

    else:
        # ---- Sequential fallback ----
        dk = DKSimulationSmoother(
            H=H_obs, init_x=init_x, init_P=init_P,
            dtype=dt) if use_dk else None

        all_x_list, all_y_list = [], []
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
                x_draw = dk.draw(P_list[j], R_list[j], Om_list[j], ys,
                                 valid_dk_keys[j], x_smooth, c=c_list[j])
                x_T = x_draw[-1]
            else:
                x_T = x_smooth[-1]

            x_fcst, y_fcst = forecast_unconditional(
                P_list[j], R_list[j], Om_list[j], x_T, horizon,
                valid_fcst_keys[j], c=c_list[j], stochastic=stochastic)
            all_x_list.append(onp.asarray(x_fcst))
            all_y_list.append(onp.asarray(y_fcst))

        x_all = onp.stack(all_x_list, axis=0)
        y_all = onp.stack(all_y_list, axis=0)

    if verbose:
        print(f"  forecast_posterior: {J_valid} draws, horizon={horizon}")

    return ForecastResult(
        state_forecasts=x_all,
        obs_forecasts=y_all,
        state_names=list(model.aug_state_vars),
        obs_names=list(model.ordered_obs_vars),
        shock_names=list(model.aug_shocks),
        horizon=horizon,
    )


# ======================================================================
# Helper functions for extracting and summarising forecasts
# ======================================================================

def get_state_forecast(result: ForecastResult, var_name: str) -> onp.ndarray:
    """Extract forecast draws for a single state variable.

    Args:
        result:   ForecastResult from forecast_posterior.
        var_name: name of the state variable (must be in state_names).

    Returns:
        (J, H) array of forecast draws.
    """
    idx = result.state_names.index(var_name)
    return result.state_forecasts[:, :, idx]


def get_obs_forecast(result: ForecastResult, var_name: str) -> onp.ndarray:
    """Extract forecast draws for a single observable.

    Args:
        result:   ForecastResult from forecast_posterior.
        var_name: name of the observable (must be in obs_names).

    Returns:
        (J, H) array of forecast draws.
    """
    idx = result.obs_names.index(var_name)
    return result.obs_forecasts[:, :, idx]


def forecast_percentiles(
    draws: onp.ndarray,
    percentiles: tuple = (5, 16, 50, 84, 95),
) -> Dict[int, onp.ndarray]:
    """Compute percentile bands from forecast draws.

    Args:
        draws:       (J, H) array of draws for one variable.
        percentiles: tuple of percentile values.

    Returns:
        dict mapping percentile -> (H,) array.
    """
    return {p: onp.percentile(draws, p, axis=0) for p in percentiles}

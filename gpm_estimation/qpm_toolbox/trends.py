"""Deterministic trend handling for the calibration filter.

Computes the unconditional deterministic path implied by the state-space
solution (P_aug, c_aug, init_x).  Subtracting this path from the
observables before the calibration's steady-state Kalman filter renders
the filter's view of the data zero-mean around the implied trend, so
the disturbance smoother used by ``calibrate_fixed_point`` operates on
a system whose unit-root states no longer carry a drifting deterministic
mean.

Mathematically this is exact: the Koopman (1993) disturbance smoother
returns the same ``eps_hat`` and ``V`` whether it is run on the level
data with ``c=c_aug, init_x=init_x`` or on the deviation data with
``c=0, init_x=0``, because both forms produce identical innovations
``v_t``.  Computationally, however, the deviation form keeps the
filter's state-prediction trajectory bounded and avoids the EM update
divergence we observe on unit-root systems when the level form is
combined with a small ``init_P`` for trend states.

The filter's exact-diffuse counterpart (``DiffuseKalmanFilter``,
App.~A.1.3 of the WP) is *not* affected by this and continues to operate
on level data with ``c=c_aug``.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


def compute_deterministic_path(
    P_aug, c_aug, init_x, Omega, T_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic forecast of the state and observable trajectories.

    Implements the recursion

        x_det(0) = init_x
        x_det(t) = P_aug @ x_det(t-1) + c_aug,   t = 1, ..., T-1
        y_det(t) = Omega @ x_det(t)

    over ``T_steps`` periods.  This is exactly the path the Kalman filter
    would predict in the absence of innovations and measurement noise.

    Args:
        P_aug:     (n_aug, n_aug) augmented transition matrix.
        c_aug:     (n_aug,) state intercept (``[0_stat ; c_trend]`` for
                   our parser convention).  ``None`` is treated as zero.
        init_x:    (n_aug,) initial state mean.
        Omega:     (n_obs, n_aug) observation loading matrix.
        T_steps:   number of periods to roll forward (length of ys).

    Returns:
        Tuple ``(x_det, y_det)`` of shapes ``(T_steps, n_aug)`` and
        ``(T_steps, n_obs)``.  ``x_det[0] == init_x`` and
        ``y_det[0] == Omega @ init_x``.
    """
    P_aug = jnp.asarray(P_aug)
    Omega = jnp.asarray(Omega)
    init_x = jnp.asarray(init_x)
    n_aug = init_x.shape[0]

    if c_aug is None:
        c_aug = jnp.zeros(n_aug, dtype=P_aug.dtype)
    else:
        c_aug = jnp.asarray(c_aug)

    def step(x_prev, _):
        x_t = P_aug @ x_prev + c_aug
        return x_t, x_t

    # Roll forward T_steps - 1 transitions starting from init_x.
    _, x_rest = jax.lax.scan(step, init_x, None, length=T_steps - 1)
    # Stack init_x as period 0, then the (T_steps - 1) propagated states.
    x_det = jnp.concatenate([init_x[None, :], x_rest], axis=0)
    y_det = x_det @ Omega.T
    return x_det, y_det


def detrend_observations(
    ys, P_aug, c_aug, init_x, Omega,
):
    """Subtract the deterministic mean from a level-form observation
    matrix to obtain its deviation form.

    Convenience wrapper that calls :func:`compute_deterministic_path`
    with ``T_steps = ys.shape[0]`` and returns
    ``(ys_dev, x_det, y_det)`` so callers can re-attach the trend to
    smoothed states if desired.

    Args:
        ys: ``(T, n_obs)`` observations in level form (jax/numpy).
        P_aug, c_aug, init_x, Omega: as in
            :func:`compute_deterministic_path`.

    Returns:
        ``(ys_dev, x_det, y_det)``.  ``ys_dev = ys - y_det`` matches the
        dtype of ``ys``.
    """
    ys_arr = jnp.asarray(ys)
    T_steps = ys_arr.shape[0]
    x_det, y_det = compute_deterministic_path(
        P_aug, c_aug, init_x, Omega, T_steps)
    ys_dev = ys_arr - y_det
    return ys_dev, x_det, y_det

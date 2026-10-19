"""Durbin-Koopman simulation smoother -- single draw per call.

Draws from the posterior distribution of states  p(x_{1:T} | y_{1:T}, theta)
using the Durbin-Koopman (2002) method.  Delegates filtering and RTS smoothing
to KalmanFilter rather than duplicating that code.

Usage (one draw per MCMC parameter draw):

    dk = DKSimulationSmoother(H_obs, init_x, init_P)
    x_draw = dk.draw(T, R, C, ys, key, x_smooth_original)
"""

import jax
import jax.numpy as jnp
from jax import lax, random

from qpm_toolbox.kalman_filter import KalmanFilter, get_dtype

_DK_JITTER = 1e-8


class DKSimulationSmoother:
    """Single-draw Durbin-Koopman simulation smoother.

    Stores only the quantities that stay fixed across parameter draws
    (H, init_x, init_P and their Cholesky factors).  The system matrices
    (T, R, C) that change per draw are passed to ``draw()``.
    """

    def __init__(self, H, init_x, init_P, dtype=None):
        dt = dtype or get_dtype()
        self.H = jnp.asarray(H, dtype=dt)
        self.init_x = jnp.asarray(init_x, dtype=dt)
        self.init_P = jnp.asarray(init_P, dtype=dt)
        self.dt = dt

        self.n_state = self.init_x.shape[0]
        self.n_obs = self.H.shape[0]

        # Cholesky of observation noise covariance
        H_reg = self.H + _DK_JITTER * jnp.eye(self.n_obs, dtype=dt)
        self.L_H = jnp.linalg.cholesky(H_reg)

        # Cholesky of initial state covariance
        P_reg = self.init_P + _DK_JITTER * jnp.eye(self.n_state, dtype=dt)
        self.L0 = jnp.linalg.cholesky(P_reg)

    def draw(self, T, R, C, ys, key, x_smooth_original, c=None):
        """Generate a single draw from the state posterior.

        Args:
            T: state transition matrix  (n_state, n_state)
            R: shock impact matrix      (n_state, n_shocks)
            C: observation matrix        (n_obs, n_state)
            ys: observations             (T_steps, n_obs)
            key: JAX PRNG key
            x_smooth_original: RTS-smoothed states from original data
                               (T_steps, n_state)
            c: state intercept vector    (n_state,), optional

        Returns:
            x_draw: simulated state trajectory (T_steps, n_state)
        """
        dt = self.dt
        T_mat = jnp.asarray(T, dtype=dt)
        R_mat = jnp.asarray(R, dtype=dt)
        C_mat = jnp.asarray(C, dtype=dt)
        c_vec = jnp.zeros(self.n_state, dtype=dt) if c is None \
                else jnp.asarray(c, dtype=dt)

        T_steps = ys.shape[0]
        n_shocks = R_mat.shape[1]

        key_init, key_eps, key_eta = random.split(key, 3)

        # 1. Draw initial state: x0* ~ N(init_x, init_P)
        z0 = random.normal(key_init, (self.n_state,), dtype=dt)
        x0_star = self.init_x + self.L0 @ z0

        # 2. Draw state shocks and forward-simulate
        eps_all = random.normal(key_eps, (T_steps, n_shocks), dtype=dt)

        def state_step(x_prev, eps_t):
            x_t = c_vec + T_mat @ x_prev + R_mat @ eps_t
            return x_t, x_t

        _, x_star_path = lax.scan(state_step, x0_star, eps_all)

        # 3. Generate simulated observations: y* = C x* + eta, eta ~ N(0, H)
        eta_all = random.normal(key_eta, (T_steps, self.n_obs), dtype=dt)
        y_star = x_star_path @ C_mat.T + eta_all @ self.L_H.T

        # 4. Filter + smooth the simulated observations
        kf_star = KalmanFilter(T_mat, R_mat, C_mat, self.H,
                               self.init_x, self.init_P, c=c_vec, dtype=dt)
        x_smooth_star = kf_star.smooth_states_only(ys=y_star)

        # 5. DK correction: x_draw = x* + (x_smooth_orig - x_smooth*)
        x_draw = x_star_path + (x_smooth_original - x_smooth_star)

        return x_draw

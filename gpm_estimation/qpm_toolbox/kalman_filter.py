"""Kalman filter, RTS smoother, and log-likelihood for linear Gaussian
state-space models.  Designed for JAX with steady-state Kalman gain.

State-space form:
    x_t = T x_{t-1} + R eps_t,    eps_t ~ N(0, I)
    y_t = C x_t     + eta_t,      eta_t ~ N(0, H)
"""

import functools

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax
from typing import Dict, Tuple, Optional

_KF_JITTER = 1e-8


# ======================================================================
# DARE solver with implicit differentiation
# ======================================================================
#
# Forward pass:  lax.while_loop Riccati iteration with true early stopping.
# Backward pass: implicit function theorem via adjoint Lyapunov equation.
#
# At the DARE fixed point P, the Riccati map is:
#     R(P; T,Q,C,H) = L (T P T' + Q) L' + K H K'
# where K = P_p C' S^{-1},  L = I - K C,  S = C P_p C' + H,  P_p = T P T' + Q.
#
# By the envelope theorem (K minimizes Tr(P_new)), the Frechet derivative is:
#     R_P[V] = A_cl V A_cl'   where A_cl = (I - K C) T
#
# The adjoint equation is a discrete Lyapunov (Stein) equation:
#     W - A_cl' W A_cl = g_P
# solved via doubling iterations (O(n^3 log(1/tol))).


def _solve_lyapunov_doubling(A, Q, n_iters=30):
    """Solve W - A' W A = Q via doubling.

    Computes W = sum_{k=0}^{2^n_iters - 1} (A')^k Q A^k.
    Converges when spectral radius of A < 1.
    """
    def body(_, state):
        Ak, Bk, Ck = state
        Ck = Ak @ Ck @ Bk + Ck
        Ak = Ak @ Ak
        Bk = Bk @ Bk
        return (Ak, Bk, Ck)

    _, _, W = lax.fori_loop(0, n_iters, body, (A.T, A, Q))
    return W


def _riccati_while_loop(T, Q, C, H, init_P, tol, max_iter):
    """Iterate DARE to convergence via lax.while_loop (true early stopping)."""
    n = T.shape[0]
    n_obs = C.shape[0]
    dt = T.dtype
    I_s = jnp.eye(n, dtype=dt)
    I_o = jnp.eye(n_obs, dtype=dt)
    jitter = _KF_JITTER

    def cond_fn(state):
        _, k, converged = state
        return (~converged) & (k < max_iter)

    def body_fn(state):
        P_f, k, _ = state
        P_p = T @ P_f @ T.T + Q
        P_p = (P_p + P_p.T) * 0.5 + jitter * I_s
        S = C @ P_p @ C.T + H + jitter * I_o
        L_S = jnp.linalg.cholesky(S)
        PCt = P_p @ C.T
        tmp = jax.scipy.linalg.solve_triangular(L_S, PCt.T, lower=True)
        K = jax.scipy.linalg.solve_triangular(
            L_S, tmp, lower=True, trans='T').T
        IKC = I_s - K @ C
        P_f_new = IKC @ P_p @ IKC.T + K @ H @ K.T
        P_f_new = (P_f_new + P_f_new.T) * 0.5
        diff = jnp.max(jnp.abs(P_f_new - P_f))
        return (P_f_new, k + 1, diff < tol)

    P_f_ss, _, _ = lax.while_loop(
        cond_fn, body_fn,
        (init_P, jnp.array(0, jnp.int32), jnp.array(False)))
    return P_f_ss


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _solve_dare(T, Q, C, H, init_P, tol=1e-12, max_iter=1000):
    """Differentiable DARE solver: lax.while_loop + adjoint Lyapunov."""
    return _riccati_while_loop(T, Q, C, H, init_P, tol, max_iter)


def _solve_dare_fwd(T, Q, C, H, init_P, tol, max_iter):
    P_f = _solve_dare(T, Q, C, H, init_P, tol, max_iter)
    return P_f, (T, Q, C, H, P_f)


def _solve_dare_bwd(tol, max_iter, res, g_P):
    """Backward pass via the implicit function theorem on the DARE.

    The adjoint Lyapunov equation:
        W - A_cl' W A_cl = g_P     where A_cl = (I - K C) T

    Parameter gradients (M = L' W L,  L = I - K C):
        g_T = 2 M T P_f
        g_Q = M
        g_C = -2 K' W L P_p
        g_H = K' W K
    """
    T, Q, C, H, P_f = res
    n = T.shape[0]
    n_obs = C.shape[0]
    dt = T.dtype
    I_s = jnp.eye(n, dtype=dt)
    I_o = jnp.eye(n_obs, dtype=dt)
    jitter = _KF_JITTER

    # Recompute steady-state quantities from P_f
    P_p = T @ P_f @ T.T + Q
    P_p = (P_p + P_p.T) * 0.5 + jitter * I_s
    S = C @ P_p @ C.T + H + jitter * I_o
    K = jax.scipy.linalg.solve(S, C @ P_p, assume_a='pos').T   # P_p C' S^{-1}
    L = I_s - K @ C

    # Closed-loop transition (eigenvalues inside unit circle)
    A_cl = L @ T

    # Solve adjoint Lyapunov: W - A_cl' W A_cl = g_P
    g_sym = (g_P + g_P.T) * 0.5
    W = _solve_lyapunov_doubling(A_cl, g_sym)

    # Parameter gradients
    M = L.T @ W @ L
    g_T = 2.0 * M @ T @ P_f
    g_Q = M
    g_C = -2.0 * K.T @ W @ L @ P_p
    g_H = K.T @ W @ K
    g_init_P = jnp.zeros_like(P_f)

    return g_T, g_Q, g_C, g_H, g_init_P


_solve_dare.defvjp(_solve_dare_fwd, _solve_dare_bwd)


def get_dtype():
    """Return the active JAX floating-point dtype."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def prepare_observations(ys, H, fill_value=1e10):
    """Prepare observations with NaN masking for JAX-compatible filtering.

    Replaces NaN entries with 0 and inflates the corresponding diagonal of H
    so the Kalman gain for that observable is effectively zero.  All array
    shapes stay fixed, which keeps everything JAX-traceable.

    Args:
        ys: observations (T, n_obs), may contain NaN
        H:  observation noise covariance (n_obs, n_obs)
        fill_value: large value added to H diagonal for missing obs

    Returns:
        ys_clean: (T, n_obs) with NaN replaced by 0
        H_t: (T, n_obs, n_obs) per-timestep H with inflated diagonal
        has_missing: bool, whether any NaN was found
    """
    dt = ys.dtype
    nan_mask = jnp.isnan(ys)                       # (T, n_obs)
    has_missing = jnp.any(nan_mask)
    ys_clean = jnp.where(nan_mask, 0.0, ys)

    # Build per-timestep H: H_t[t] = H + diag(fill_value * nan_mask[t])
    T_steps, n_obs = ys.shape
    diag_inflate = nan_mask.astype(dt) * fill_value  # (T, n_obs)
    H_base = jnp.broadcast_to(H, (T_steps, n_obs, n_obs))
    # Add inflation to diagonal
    idx = jnp.arange(n_obs)
    H_t = H_base + diag_inflate[:, :, None] * (idx[None, :] == idx[None, None, :]).astype(dt)

    return ys_clean, H_t, has_missing


class KalmanFilter:
    """Kalman filter with pre-computed steady-state gain (like Dynare).

    The constructor solves the discrete-time algebraic Riccati recursion
    to find the steady-state prediction covariance, Kalman gain, and
    innovation covariance.  The per-step filter then uses only
    matrix-vector operations (no Cholesky per step).

    For smoothing, a full time-varying filter is available that stores
    P_pred and P_filt at each step for the RTS backward pass.
    """

    def __init__(self, T, R, C, H, init_x, init_P, c=None, dtype=None):
        dt = dtype or get_dtype()
        self.T = jnp.asarray(T, dtype=dt)
        self.R = jnp.asarray(R, dtype=dt)
        self.C = jnp.asarray(C, dtype=dt)
        self.H = jnp.asarray(H, dtype=dt)
        self.init_x = jnp.asarray(init_x, dtype=dt)
        self.init_P = jnp.asarray(init_P, dtype=dt)
        self.dt = dt

        self.n_state = self.T.shape[0]
        self.n_obs = self.C.shape[0]
        self.Q = self.R @ self.R.T                    # state noise covariance
        self.c = jnp.zeros(self.n_state, dtype=dt) if c is None \
                 else jnp.asarray(c, dtype=dt)
        self.I_s = jnp.eye(self.n_state, dtype=dt)
        self.I_o = jnp.eye(self.n_obs, dtype=dt)

        # Solve steady-state Riccati
        self._solve_steady_state()

    # ------------------------------------------------------------------
    # Steady-state Riccati solver (implicit differentiation via DARE)
    # ------------------------------------------------------------------
    def _solve_steady_state(self, max_iter=1000, tol=1e-12):
        """Solve DARE for steady-state Kalman gain.

        Uses lax.while_loop for the forward pass (true early stopping,
        ~10-20 iterations) and the adjoint Lyapunov equation for the
        backward pass via custom_vjp.

        Stores: K_ss, L_S_ss, log_det_S_ss, S_ss, P_pred_ss, P_filt_ss
        """
        T, C, Q, H = self.T, self.C, self.Q, self.H
        I_s, I_o = self.I_s, self.I_o
        jitter = _KF_JITTER

        # Solve DARE with implicit differentiation
        P_filt_ss = _solve_dare(T, Q, C, H, self.init_P, tol, max_iter)

        # Final steady-state quantities (standard JAX ops, auto-diffable)
        P_pred_ss = T @ P_filt_ss @ T.T + Q
        P_pred_ss = (P_pred_ss + P_pred_ss.T) * 0.5 + jitter * I_s

        S_ss = C @ P_pred_ss @ C.T + H + jitter * I_o
        L_S_ss = jnp.linalg.cholesky(S_ss)

        PCt = P_pred_ss @ C.T
        tmp = jax.scipy.linalg.solve_triangular(L_S_ss, PCt.T, lower=True)
        K_ss = jax.scipy.linalg.solve_triangular(
            L_S_ss, tmp, lower=True, trans='T').T

        log_det_S_ss = 2.0 * jnp.sum(jnp.log(jnp.diag(L_S_ss)))
        ll_const = self.n_obs * jnp.log(2.0 * jnp.pi) + log_det_S_ss

        # Store
        self.K_ss = K_ss
        self.L_S_ss = L_S_ss
        self.S_ss = S_ss
        self.log_det_S_ss = log_det_S_ss
        self.ll_const = ll_const
        self.P_pred_ss = P_pred_ss
        self.P_filt_ss = P_filt_ss

    # ------------------------------------------------------------------
    # Steady-state filter (for log-likelihood -- no P storage)
    # ------------------------------------------------------------------
    def log_likelihood(self, ys, presample=0):
        """Scalar total log-likelihood using steady-state Kalman gain.

        Per-step cost: O(n_state * n_obs) -- no Cholesky per step.

        Args:
            ys: (T, n_obs) observations.
            presample: number of initial periods to exclude from the
                log-likelihood sum.  During these periods the steady-state
                gain is applied to propagate states, but the (biased)
                likelihood contributions are discarded.  Useful when
                init_P differs materially from P_filt_ss (e.g. diffuse
                initialisation for unit-root trends).
        """
        ys_arr = jnp.asarray(ys, dtype=self.dt)
        T, C, K = self.T, self.C, self.K_ss
        c = self.c
        L_S = self.L_S_ss
        ll_const = self.ll_const

        def step(x_filt, y_t):
            x_pred = c + T @ x_filt
            v = y_t - C @ x_pred
            x_filt_new = x_pred + K @ v

            # log-likelihood contribution
            z = jax.scipy.linalg.solve_triangular(L_S, v, lower=True)
            ll_t = -0.5 * (ll_const + jnp.dot(z, z))

            return x_filt_new, ll_t

        _, ll_contributions = lax.scan(step, self.init_x, ys_arr)
        return jnp.sum(ll_contributions[presample:])

    # ------------------------------------------------------------------
    # Time-varying log-likelihood (missing data / shock multipliers)
    # ------------------------------------------------------------------
    def log_likelihood_tv(self, ys, presample=0, H_t=None, Q_t=None):
        """Log-likelihood with per-period observation or state noise.

        Full time-varying Riccati filter at O(n^3 T) per call.  Use when:
          - H varies across periods (missing data / ragged edges)
          - Q varies across periods (shock multipliers)

        For complete data with constant Q, prefer ``log_likelihood()``
        which uses the pre-computed steady-state Kalman gain.

        Args:
            ys: (T, n_obs) observations (NaN should be replaced by 0
                when using per-period H_t from ``prepare_observations``).
            presample: initial periods to exclude from the sum.
            H_t: (T, n_obs, n_obs) per-period observation noise.
                 If None, uses self.H (constant) for all periods.
            Q_t: (T, n_state, n_state) per-period state noise.
                 If None, uses self.Q (constant) for all periods.
        """
        ys_arr = jnp.asarray(ys, dtype=self.dt)
        T_steps = ys_arr.shape[0]
        T_mat, C = self.T, self.C
        c = self.c
        I_s, I_o = self.I_s, self.I_o
        jitter = _KF_JITTER
        n_obs_log2pi = self.n_obs * jnp.log(2.0 * jnp.pi)

        # Broadcast constant matrices to per-period arrays if not provided
        if H_t is None:
            H_t = jnp.broadcast_to(self.H, (T_steps,) + self.H.shape)
        else:
            H_t = jnp.asarray(H_t, dtype=self.dt)

        if Q_t is None:
            Q_t = jnp.broadcast_to(self.Q, (T_steps,) + self.Q.shape)
        else:
            Q_t = jnp.asarray(Q_t, dtype=self.dt)

        def step(carry, inputs):
            x_f, P_f = carry
            y_t, H_step, Q_step = inputs

            x_p = c + T_mat @ x_f
            P_p = T_mat @ P_f @ T_mat.T + Q_step
            P_p = (P_p + P_p.T) * 0.5 + jitter * I_s

            v = y_t - C @ x_p
            S = C @ P_p @ C.T + H_step + jitter * I_o

            L_S = jnp.linalg.cholesky(S)
            PCt = P_p @ C.T
            tmp = jax.scipy.linalg.solve_triangular(
                L_S, PCt.T, lower=True)
            K = jax.scipy.linalg.solve_triangular(
                L_S, tmp, lower=True, trans='T').T

            x_up = x_p + K @ v
            IKC = I_s - K @ C
            P_up = IKC @ P_p @ IKC.T + K @ H_step @ K.T
            P_up = (P_up + P_up.T) * 0.5

            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L_S)))
            z = jax.scipy.linalg.solve_triangular(L_S, v, lower=True)
            ll_t = -0.5 * (n_obs_log2pi + log_det + jnp.dot(z, z))

            return (x_up, P_up), ll_t

        _, ll_all = lax.scan(
            step, (self.init_x, self.init_P), (ys_arr, H_t, Q_t))
        return jnp.sum(ll_all[presample:])

    # ------------------------------------------------------------------
    # Chandrasekhar recursion log-likelihood (stationary models only)
    # ------------------------------------------------------------------
    def log_likelihood_chandrasekhar(self, ys, presample=0, tol=1e-12):
        """Log-likelihood via Chandrasekhar recursions (stationary only).

        Single-pass filter that combines steady-state convergence and
        likelihood evaluation.  K_t and F_t are updated via low-rank
        O(m p^2) recursions instead of full O(m^3) Riccati steps.
        Once K converges the updates freeze for remaining steps.

        Requires spectral radius of T < 1.  For unit-root models use
        log_likelihood().

        Reference: Herbst (2015), Econ. Letters 133, 105-107.

        Args:
            ys: (T_steps, n_obs) observations.
            presample: initial periods excluded from the log-likelihood.
            tol: convergence tolerance for max|Delta K|.
        """
        ys_arr = jnp.asarray(ys, dtype=self.dt)
        T_mat, C, Q, H = self.T, self.C, self.Q, self.H
        c = self.c
        I_o = self.I_o
        n_obs = self.n_obs
        n_obs_log2pi = n_obs * jnp.log(2.0 * jnp.pi)
        jitter = _KF_JITTER

        # Unconditional covariance P = T P T' + Q  (Lyapunov)
        P_unc = _solve_lyapunov_doubling(T_mat.T, Q)
        P_unc = (P_unc + P_unc.T) * 0.5

        # CR initialisation from P_unc
        F_0 = C @ P_unc @ C.T + H + jitter * I_o
        W_0 = T_mat @ P_unc @ C.T                            # (m, p)
        K_0 = jax.scipy.linalg.solve(
            F_0, W_0.T, assume_a='pos').T                     # (m, p)
        M_0 = -jax.scipy.linalg.solve(
            F_0, I_o, assume_a='pos')                         # (p, p)

        def step(carry, y_t):
            x_pred, K, F, W, M, converged = carry

            v = y_t - C @ x_pred

            # Log-likelihood (current F_t)
            L_F = jnp.linalg.cholesky(F + jitter * I_o)
            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L_F)))
            z = jax.scipy.linalg.solve_triangular(
                L_F, v, lower=True)
            ll_t = -0.5 * (n_obs_log2pi + log_det + jnp.dot(z, z))

            # State prediction (K here is the prediction gain T K_filt)
            x_pred_new = c + T_mat @ x_pred + K @ v

            # --- CR factor updates O(m p^2) ---
            CW = C @ W                                        # (p, p)
            TW = T_mat @ W                                    # (m, p)
            MCWt = M @ CW.T                                   # (p, p)

            F_new = F + CW @ MCWt
            F_new = (F_new + F_new.T) * 0.5

            KF_plus = K @ F + TW @ MCWt                       # (m, p)
            K_new = jax.scipy.linalg.solve(
                F_new + jitter * I_o, KF_plus.T,
                assume_a='pos').T                              # (m, p)

            W_new = TW - K_new @ CW                           # (m, p)

            invF_CW = jax.scipy.linalg.cho_solve(
                (L_F, True), CW)                              # (p, p)
            M_new = M + MCWt @ invF_CW @ M
            M_new = (M_new + M_new.T) * 0.5

            # Convergence: freeze when gain stops changing
            delta_K = jnp.max(jnp.abs(K_new - K))
            converged_new = converged | (delta_K < tol)

            K_out = jnp.where(converged, K, K_new)
            F_out = jnp.where(converged, F, F_new)
            W_out = jnp.where(converged, W, W_new)
            M_out = jnp.where(converged, M, M_new)

            return (x_pred_new, K_out, F_out, W_out, M_out,
                    converged_new), ll_t

        init_carry = (self.init_x, K_0, F_0, W_0, M_0,
                      jnp.array(False))
        _, ll_all = lax.scan(step, init_carry, ys_arr)
        return jnp.sum(ll_all[presample:])

    # ------------------------------------------------------------------
    # Chandrasekhar recursion full filter (stationary models only)
    # ------------------------------------------------------------------
    def filter_full_chandrasekhar(self, ys, tol=1e-12):
        """Full filter via Chandrasekhar recursions (stationary only).

        Like filter_full() but uses O(m^2 p) CR updates instead of
        O(m^3) Riccati propagation.  Stores P_pred and P_filt at every
        step for the RTS smoother backward pass.

        Requires spectral radius of T < 1.  For unit-root models use
        filter_full().

        Args:
            ys: (T_steps, n_obs) observations.
            tol: convergence tolerance for max|Delta K|.

        Returns:
            dict with keys: x_pred, P_pred, x_filt, P_filt, v, ll
        """
        ys_arr = jnp.asarray(ys, dtype=self.dt)
        T_mat, C, Q, H = self.T, self.C, self.Q, self.H
        c = self.c
        I_s, I_o = self.I_s, self.I_o
        n_obs = self.n_obs
        n_obs_log2pi = n_obs * jnp.log(2.0 * jnp.pi)
        jitter = _KF_JITTER

        # Unconditional covariance (Lyapunov)
        P_unc = _solve_lyapunov_doubling(T_mat.T, Q)
        P_unc = (P_unc + P_unc.T) * 0.5

        # CR initialisation
        F_0 = C @ P_unc @ C.T + H + jitter * I_o
        W_0 = T_mat @ P_unc @ C.T
        K_0 = jax.scipy.linalg.solve(
            F_0, W_0.T, assume_a='pos').T
        M_0 = -jax.scipy.linalg.solve(
            F_0, I_o, assume_a='pos')

        def step(carry, y_t):
            x_pred, P_pred, K, F, W, M, converged = carry

            v = y_t - C @ x_pred

            # Log-likelihood
            L_F = jnp.linalg.cholesky(F + jitter * I_o)
            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L_F)))
            z = jax.scipy.linalg.solve_triangular(
                L_F, v, lower=True)
            ll_t = -0.5 * (n_obs_log2pi + log_det + jnp.dot(z, z))

            # Filtered state: x_filt = x_pred + P C' F^{-1} v
            invF_v = jax.scipy.linalg.cho_solve(
                (L_F, True), v)                                # (p,)
            x_filt = x_pred + P_pred @ (C.T @ invF_v)         # (m,)

            # Filtering covariance: P_filt = P - P C' F^{-1} C P
            B = P_pred @ C.T                                   # (m, p)
            invF_B = jax.scipy.linalg.cho_solve(
                (L_F, True), B.T).T                            # (m, p)
            P_filt = P_pred - invF_B @ B.T
            P_filt = (P_filt + P_filt.T) * 0.5

            # State prediction for next step
            x_pred_new = c + T_mat @ x_filt

            # --- CR factor updates ---
            CW = C @ W
            TW = T_mat @ W
            MCWt = M @ CW.T

            F_new = F + CW @ MCWt
            F_new = (F_new + F_new.T) * 0.5

            KF_plus = K @ F + TW @ MCWt
            K_new = jax.scipy.linalg.solve(
                F_new + jitter * I_o, KF_plus.T,
                assume_a='pos').T

            W_new = TW - K_new @ CW

            invF_CW = jax.scipy.linalg.cho_solve(
                (L_F, True), CW)
            M_new = M + MCWt @ invF_CW @ M
            M_new = (M_new + M_new.T) * 0.5

            # Prediction covariance for next step: rank-p update
            WMWt = W @ M @ W.T                                # (m, m)
            P_pred_new = P_pred + WMWt
            P_pred_new = (P_pred_new + P_pred_new.T) * 0.5

            # Convergence check and freeze
            delta_K = jnp.max(jnp.abs(K_new - K))
            converged_new = converged | (delta_K < tol)

            K_out = jnp.where(converged, K, K_new)
            F_out = jnp.where(converged, F, F_new)
            W_out = jnp.where(converged, W, W_new)
            M_out = jnp.where(converged, M, M_new)
            P_pred_out = jnp.where(converged, P_pred, P_pred_new)

            out = {
                'x_pred': x_pred, 'P_pred': P_pred,
                'x_filt': x_filt, 'P_filt': P_filt,
                'v': v, 'll': ll_t,
            }
            return (x_pred_new, P_pred_out, K_out, F_out,
                    W_out, M_out, converged_new), out

        init_carry = (self.init_x, P_unc, K_0, F_0, W_0, M_0,
                      jnp.array(False))
        _, results = lax.scan(step, init_carry, ys_arr)
        return results

    # ------------------------------------------------------------------
    # Full time-varying filter (stores P_pred, P_filt for smoother)
    # ------------------------------------------------------------------
    def filter_full(self, ys, n_tv=None, H_t=None, Q_t=None):
        """Full Kalman filter storing all intermediate quantities.

        Needed for the RTS smoother backward pass.

        When ``n_tv`` is set, the filter runs in two phases (a la Dynare):

        * **Phase 1** (t = 0 .. n_tv-1): full time-varying Riccati update,
          O(n^3) per step.  This handles the transient where init_P differs
          from the steady-state covariance.
        * **Phase 2** (t = n_tv .. T-1): steady-state Kalman gain K_ss.
          Only matrix-vector operations, O(n^2) per step.  The stored
          P_pred and P_filt are the pre-computed steady-state matrices,
          which is correct because K has converged.

        For the Brazil model (n_aug=38, T=84, n_tv=10) this removes 74
        Cholesky factorisations and two O(n^3) matrix products per step.

        When ``H_t`` or ``Q_t`` is provided the system is time-varying and
        the steady-state phase is disabled (full Riccati for all T).

        Args:
            ys: (T, n_obs) observations.
            n_tv: number of initial time-varying steps.  ``None`` (default)
                runs the full time-varying filter for all T periods.
                Ignored when H_t or Q_t is provided.
            H_t: optional (T, n_obs, n_obs) per-period observation noise.
                Use ``prepare_observations`` to build this from NaN data.
            Q_t: optional (T, n_state, n_state) per-period state noise.
                Use for time-varying shock multipliers.

        Returns:
            dict with keys: x_pred, P_pred, x_filt, P_filt, v, ll
        """
        ys_arr = jnp.asarray(ys, dtype=self.dt)
        T_steps = ys_arr.shape[0]

        # Dispatch: if per-period H or Q is provided, use the TV-only path
        if H_t is not None or Q_t is not None:
            return self._filter_full_tv(ys_arr, T_steps, H_t, Q_t)

        T_mat, C, Q, H = self.T, self.C, self.Q, self.H
        c = self.c
        I_s, I_o = self.I_s, self.I_o
        jitter = _KF_JITTER
        n_obs_log2pi = self.n_obs * jnp.log(2.0 * jnp.pi)

        # ---- time-varying step (O(n^3)) ----
        def tv_step(carry, y_t):
            x_f, P_f = carry

            x_p = c + T_mat @ x_f
            P_p = T_mat @ P_f @ T_mat.T + Q
            P_p = (P_p + P_p.T) * 0.5 + jitter * I_s

            v = y_t - C @ x_p
            S = C @ P_p @ C.T + H + jitter * I_o

            L_S = jnp.linalg.cholesky(S)
            PCt = P_p @ C.T
            tmp = jax.scipy.linalg.solve_triangular(
                L_S, PCt.T, lower=True)
            K = jax.scipy.linalg.solve_triangular(
                L_S, tmp, lower=True, trans='T').T

            x_up = x_p + K @ v
            IKC = I_s - K @ C
            P_up = IKC @ P_p @ IKC.T + K @ H @ K.T
            P_up = (P_up + P_up.T) * 0.5

            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L_S)))
            z = jax.scipy.linalg.solve_triangular(L_S, v, lower=True)
            ll_t = -0.5 * (n_obs_log2pi + log_det + jnp.dot(z, z))

            out = {
                'x_pred': x_p, 'P_pred': P_p,
                'x_filt': x_up, 'P_filt': P_up,
                'v': v, 'll': ll_t,
            }
            return (x_up, P_up), out

        # Full time-varying: original behaviour
        if n_tv is None or n_tv >= T_steps:
            _, results = lax.scan(
                tv_step, (self.init_x, self.init_P), ys_arr)
            return results

        # ---- steady-state step (O(n^2)) ----
        K_ss = self.K_ss
        P_pred_ss = self.P_pred_ss
        P_filt_ss = self.P_filt_ss
        L_S_ss = self.L_S_ss
        log_det_ss = self.log_det_S_ss

        def ss_step(x_f, y_t):
            x_p = c + T_mat @ x_f
            v = y_t - C @ x_p
            x_up = x_p + K_ss @ v

            z = jax.scipy.linalg.solve_triangular(L_S_ss, v, lower=True)
            ll_t = -0.5 * (n_obs_log2pi + log_det_ss + jnp.dot(z, z))

            out = {
                'x_pred': x_p, 'P_pred': P_pred_ss,
                'x_filt': x_up, 'P_filt': P_filt_ss,
                'v': v, 'll': ll_t,
            }
            return x_up, out

        n_tv = max(1, n_tv)

        # Phase 1: time-varying burn-in
        (x_last, _), results1 = lax.scan(
            tv_step, (self.init_x, self.init_P), ys_arr[:n_tv])

        # Phase 2: steady-state
        _, results2 = lax.scan(ss_step, x_last, ys_arr[n_tv:])

        # Concatenate both phases
        results = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=0),
            results1, results2)
        return results

    def _filter_full_tv(self, ys_arr, T_steps, H_t=None, Q_t=None):
        """Full time-varying filter with per-period H and/or Q.

        Internal method called by ``filter_full`` when time-varying
        observation noise (missing data) or state noise (shock multipliers)
        is provided.  Always runs full Riccati for all T periods.
        """
        T_mat, C = self.T, self.C
        c = self.c
        I_s, I_o = self.I_s, self.I_o
        jitter = _KF_JITTER
        n_obs_log2pi = self.n_obs * jnp.log(2.0 * jnp.pi)

        if H_t is None:
            H_t = jnp.broadcast_to(self.H, (T_steps,) + self.H.shape)
        else:
            H_t = jnp.asarray(H_t, dtype=self.dt)

        if Q_t is None:
            Q_t = jnp.broadcast_to(self.Q, (T_steps,) + self.Q.shape)
        else:
            Q_t = jnp.asarray(Q_t, dtype=self.dt)

        def step(carry, inputs):
            x_f, P_f = carry
            y_t, H_step, Q_step = inputs

            x_p = c + T_mat @ x_f
            P_p = T_mat @ P_f @ T_mat.T + Q_step
            P_p = (P_p + P_p.T) * 0.5 + jitter * I_s

            v = y_t - C @ x_p
            S = C @ P_p @ C.T + H_step + jitter * I_o

            L_S = jnp.linalg.cholesky(S)
            PCt = P_p @ C.T
            tmp = jax.scipy.linalg.solve_triangular(
                L_S, PCt.T, lower=True)
            K = jax.scipy.linalg.solve_triangular(
                L_S, tmp, lower=True, trans='T').T

            x_up = x_p + K @ v
            IKC = I_s - K @ C
            P_up = IKC @ P_p @ IKC.T + K @ H_step @ K.T
            P_up = (P_up + P_up.T) * 0.5

            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L_S)))
            z = jax.scipy.linalg.solve_triangular(L_S, v, lower=True)
            ll_t = -0.5 * (n_obs_log2pi + log_det + jnp.dot(z, z))

            out = {
                'x_pred': x_p, 'P_pred': P_p,
                'x_filt': x_up, 'P_filt': P_up,
                'v': v, 'll': ll_t,
            }
            return (x_up, P_up), out

        _, results = lax.scan(
            step, (self.init_x, self.init_P), (ys_arr, H_t, Q_t))
        return results

    # ------------------------------------------------------------------
    # RTS smoother
    # ------------------------------------------------------------------
    def smooth(self, ys=None, filter_results=None, n_tv=None,
               H_t=None, Q_t=None):
        """RTS backward smoother.  Returns (x_smooth, P_smooth).

        Args:
            ys: observations (T, n_obs).  Ignored if filter_results given.
            filter_results: output of filter_full() (avoids re-running).
            n_tv: passed to filter_full(); see its docstring.
            H_t: optional per-period H; passed to filter_full().
            Q_t: optional per-period Q; passed to filter_full().
        """
        if filter_results is None:
            filter_results = self.filter_full(
                ys, n_tv=n_tv, H_t=H_t, Q_t=Q_t)
        return self._rts_smooth(filter_results, compute_P=True)

    def smooth_states_only(self, ys=None, filter_results=None, n_tv=None,
                           H_t=None, Q_t=None):
        """RTS smoother returning only x_smooth (skips P_smooth).

        Saves O(n_state^3) per time step.  Used by DK simulation smoother.

        Args:
            ys: observations (T, n_obs).  Ignored if filter_results given.
            filter_results: output of filter_full() (avoids re-running).
            n_tv: passed to filter_full(); see its docstring.
            H_t: optional per-period H; passed to filter_full().
            Q_t: optional per-period Q; passed to filter_full().
        """
        if filter_results is None:
            filter_results = self.filter_full(
                ys, n_tv=n_tv, H_t=H_t, Q_t=Q_t)
        x_smooth, _ = self._rts_smooth(filter_results, compute_P=False)
        return x_smooth

    # ------------------------------------------------------------------
    # Koopman (1993) disturbance smoother
    # ------------------------------------------------------------------
    def smooth_disturbances(self, ys=None, filter_results=None, n_tv=None,
                            H_t=None, Q_t=None, shock_scale_t=None):
        """Koopman (1993) disturbance smoother for structural shocks.

        Given the state-space model:
            x_t = T x_{t-1} + R_t eps_t,  eps_t ~ N(0, I)
            y_t = C x_t + eta_t,          eta_t ~ N(0, H)

        Returns the posterior mean and variance of each shock at each t:
            eps_hat_{j,t} = E[eps_{j,t} | y_{1:T}]
            V_{j,t}       = Var(eps_{j,t} | y_{1:T})

        These are the key inputs for the fixed-point calibration
        (conditional second moment q_{j,t} = eps_hat^2_{j,t} + V_{j,t}).

        Args:
            ys: observations (T_steps, n_obs).  Ignored if filter_results given.
            filter_results: output of filter_full() (avoids re-running).
            n_tv: passed to filter_full(); see its docstring.
            H_t: optional per-period H; passed to filter_full().
            Q_t: optional per-period Q; passed to filter_full().
            shock_scale_t: optional (T_steps, n_shock) per-period
                multiplicative scaling on the shock loadings.  When
                provided, the backward pass uses
                ``R_t[:, j] = R[:, j] * shock_scale_t[t, j]`` so that
                ``eps_hat`` and ``V_diag`` reflect the time-varying
                shock variance ``Q_t[j,j] = (sigma_j * shock_scale_t[t,j])^2``.
                If ``Q_t`` is not also supplied, the matching
                ``Q_t = R_t @ R_t.T`` is built and threaded into the
                forward filter automatically.

        Returns:
            eps_hat:   (T_steps, n_shock)  smoothed shock means
            V_eps:     (T_steps, n_shock)  smoothed shock variances (diagonal)
        """
        T_mat, R, C, H = self.T, self.R, self.C, self.H
        I_s, I_o = self.I_s, self.I_o
        jitter = _KF_JITTER
        n_state, n_shock = R.shape

        # If a per-period scale is supplied without an explicit Q_t, build Q_t
        # consistently so the forward filter sees the same heteroscedasticity.
        if shock_scale_t is not None:
            scale_arr = jnp.asarray(shock_scale_t, dtype=self.dt)
            if Q_t is None:
                R_t_full = R[None, :, :] * scale_arr[:, None, :]
                Q_t = jnp.einsum('tsi,tji->tsj', R_t_full, R_t_full)
        else:
            scale_arr = None

        if filter_results is None:
            filter_results = self.filter_full(
                ys, n_tv=n_tv, H_t=H_t, Q_t=Q_t)

        v = filter_results['v']            # (T_steps, n_obs)
        P_pred = filter_results['P_pred']  # (T_steps, n_state, n_state)
        T_steps = v.shape[0]

        # Reverse time for backward scan
        v_rev = v[::-1]
        Pp_rev = P_pred[::-1]
        # Per-step shock scale (default 1) reversed for the backward scan
        if scale_arr is None:
            scale_rev = jnp.ones((T_steps, n_shock), dtype=self.dt)
        else:
            scale_rev = scale_arr[::-1]

        def bwd_step(carry, inputs):
            r, N = carry         # r: (n_state,), N: (n_state, n_state)
            v_t, Pp_t, scale_t = inputs

            # --- Recompute S_t, K_t from stored P_pred_t ---
            S_t = C @ Pp_t @ C.T + H + jitter * I_o
            L_S = jnp.linalg.cholesky(S_t)

            # S^{-1} v  and  S^{-1} C  via Cholesky
            invS_v = jax.scipy.linalg.cho_solve((L_S, True), v_t)     # (n_obs,)
            invS_C = jax.scipy.linalg.cho_solve((L_S, True), C)       # (n_obs, n_state)

            # L_t = T (I - K_t C),  K_t = P_pred C' S^{-1}
            KtC = Pp_t @ (C.T @ invS_C)                               # (n_state, n_state)
            L_t = T_mat @ (I_s - KtC)

            # --- Backward update: r_{t-1}, N_{t-1} ---
            r_new = C.T @ invS_v + L_t.T @ r
            N_new = C.T @ invS_C + L_t.T @ N @ L_t
            N_new = (N_new + N_new.T) * 0.5

            # --- Shock smoother with per-step R_t = R * scale_t ---
            # eps_hat_t = R_t.T @ r_new = scale_t * (R.T @ r_new)
            # V_diag_t = 1 - sum((R_t.T @ N) * R_t.T, axis=1)
            #          = 1 - scale_t^2 * sum((R.T @ N) * R.T, axis=1)
            R_t_T = R.T * scale_t[:, None]                            # (n_shock, n_state)
            eps_hat_t = R_t_T @ r_new                                 # (n_shock,)
            RtN = R_t_T @ N_new                                       # (n_shock, n_state)
            V_diag_t = 1.0 - jnp.sum(RtN * R_t_T, axis=1)             # (n_shock,)

            return (r_new, N_new), (eps_hat_t, V_diag_t)

        init_carry = (jnp.zeros(n_state, dtype=self.dt),
                      jnp.zeros((n_state, n_state), dtype=self.dt))
        _, (eps_hat_rev, V_diag_rev) = lax.scan(
            bwd_step, init_carry, (v_rev, Pp_rev, scale_rev))

        # Reverse back to forward time
        return eps_hat_rev[::-1], V_diag_rev[::-1]

    # ------------------------------------------------------------------
    # RTS smoother internals
    # ------------------------------------------------------------------
    def _rts_smooth(self, fr, compute_P):
        """Backward RTS pass (single scan)."""
        T = self.T
        I_s = self.I_s
        jitter = _KF_JITTER

        x_pred = fr['x_pred']
        P_pred = fr['P_pred']
        x_filt = fr['x_filt']
        P_filt = fr['P_filt']
        N = x_filt.shape[0]

        # Reversed scan inputs (indices 0..N-2)
        scan_in = (
            P_pred[1:N][::-1],
            P_filt[0:N-1][::-1],
            x_pred[1:N][::-1],
            x_filt[0:N-1][::-1],
        )

        if compute_P:
            def bwd_step(carry, inputs):
                x_s_next, P_s_next = carry
                Pp_next, Pf_t, xp_next, xf_t = inputs

                Pf_sym = (Pf_t + Pf_t.T) * 0.5
                Pp_reg = Pp_next + jitter * I_s

                # J_t = P_filt_t T' P_pred_{t+1}^{-1}
                # solve(P, T @ Pf) gives P^{-1} T Pf = J', so .T gives J
                J = jax.scipy.linalg.solve(
                    Pp_reg, T @ Pf_sym, assume_a='pos').T

                x_s = xf_t + J @ (x_s_next - xp_next)
                P_s = Pf_sym + J @ (P_s_next - Pp_next) @ J.T
                P_s = (P_s + P_s.T) * 0.5

                return (x_s, P_s), (x_s, P_s)

            init_carry = (x_filt[-1], P_filt[-1])
            _, (x_rev, P_rev) = lax.scan(bwd_step, init_carry, scan_in)

            x_smooth = jnp.concatenate([x_rev[::-1], x_filt[N-1:N]], axis=0)
            P_smooth = jnp.concatenate([P_rev[::-1], P_filt[N-1:N]], axis=0)
        else:
            def bwd_step(carry, inputs):
                x_s_next, P_s_next = carry
                Pp_next, Pf_t, xp_next, xf_t = inputs

                Pf_sym = (Pf_t + Pf_t.T) * 0.5
                Pp_reg = Pp_next + jitter * I_s

                # J_t = P_filt_t T' P_pred_{t+1}^{-1}
                J = jax.scipy.linalg.solve(
                    Pp_reg, T @ Pf_sym, assume_a='pos').T

                x_s = xf_t + J @ (x_s_next - xp_next)

                return (x_s, P_s_next), x_s

            init_carry = (x_filt[-1], P_filt[-1])
            _, x_rev = lax.scan(bwd_step, init_carry, scan_in)

            x_smooth = jnp.concatenate([x_rev[::-1], x_filt[N-1:N]], axis=0)
            P_smooth = None

        return x_smooth, P_smooth


# ======================================================================
# Exact Diffuse Kalman Filter / Smoother
# ======================================================================
#
# Implements the univariate exact diffuse filter of Durbin & Koopman
# (2012), Chapters 5 and 7, following the Koopman & Durbin (2003)
# treatment.  Processes one scalar observation at a time, avoiding
# matrix inversions and handling unit-root states exactly.
#
# Key differences from KalmanFilter:
#   - No DARE solve: no assumption that T is stable.
#   - Initialization via Pinf (unit-root directions) + Pstar (stationary).
#   - Two-phase forward pass: diffuse (while rank(Pinf) > 0) then standard.
#   - Two-phase backward smoother: standard r/N backward then diffuse r0/r1/N0.
#   - Log-likelihood excludes diffuse observations (correct exact diffuse LL).
#
# JAX implementation notes:
#   - Both phases run as a single lax.scan over T periods using a boolean
#     carry flag `diffuse_active` to mask updates.  No data-dependent
#     loop length is needed.
#   - An inner lax.scan over the n_obs observations processes each scalar
#     observable independently (univariate filter).
#   - The inner obs_step closes over `diffuse_active` from the outer carry,
#     which is valid because JAX traces through nested functions with
#     abstract values.
#
# References:
#   Durbin, J. & Koopman, S.J. (2012). Time Series Analysis by State Space
#   Methods, 2nd ed. Oxford: OUP. Chapters 5, 7.
#   Koopman, S.J. & Durbin, J. (2003). Filtering and Smoothing of State
#   Vector for Diffuse State-Space Models. J. Time Series Analysis 24(1),
#   85-98.


class DiffuseKalmanFilter:
    """Exact diffuse Kalman filter / smoother for models with unit roots.

    State-space form::

        x_t = c + T x_{t-1} + R eps_t,   eps_t ~ N(0, I)
        y_t = C x_t          + eta_t,     eta_t ~ N(0, H)

    Initialization::

        x_{1|0} ~ N(init_a, Pstar + kappa * Pinf),  kappa -> inf

    ``Pinf`` is a positive semi-definite matrix whose column space spans
    the unit-root (non-stationary) subspace.  ``Pstar`` is the stationary
    initial covariance (typically the Lyapunov solution for the stationary
    block).

    **Restriction**: ``H`` must be diagonal (measurement errors are
    independent across observables).  This is satisfied by all QPM models
    in this toolbox.

    Use ``build_diffuse_init()`` to construct ``Pinf`` and ``Pstar`` from
    the solved state-transition matrix ``P_aug``.
    """

    def __init__(self, T, R, C, H, Pinf, Pstar, init_a=None, c=None,
                 dtol=1e-6, ktol=1e-10, dtype=None):
        """
        Args:
            T: (n_state, n_state) state transition matrix.
            R: (n_state, n_shock) shock impact matrix.
            C: (n_obs, n_state) observation matrix.
            H: (n_obs, n_obs) observation noise covariance (must be diagonal).
            Pinf: (n_state, n_state) diffuse initial covariance.
                  Columns span the non-stationary subspace.
            Pstar: (n_state, n_state) stationary initial covariance.
                   Lyapunov solution for the stationary block; zero rows/cols
                   for unit-root states.
            init_a: (n_state,) initial state mean.  Defaults to zeros.
            c: (n_state,) state intercept.  Defaults to zeros.
            dtol: tolerance for declaring f_inf negligible (diffuse phase ends).
            ktol: tolerance for declaring f_star negligible (skip update).
        """
        dt = dtype or get_dtype()
        self.T = jnp.asarray(T, dtype=dt)
        self.R = jnp.asarray(R, dtype=dt)
        self.C = jnp.asarray(C, dtype=dt)
        self.H = jnp.asarray(H, dtype=dt)
        self.Pinf = jnp.asarray(Pinf, dtype=dt)
        self.Pstar = jnp.asarray(Pstar, dtype=dt)
        self.dt = dt
        self.dtol = dtol
        self.ktol = ktol

        self.n_state = self.T.shape[0]
        self.n_obs = self.C.shape[0]
        self.Q = self.R @ self.R.T

        self.init_a = (jnp.zeros(self.n_state, dtype=dt) if init_a is None
                       else jnp.asarray(init_a, dtype=dt))
        self.c = (jnp.zeros(self.n_state, dtype=dt) if c is None
                  else jnp.asarray(c, dtype=dt))

        # Diagonal of H (univariate filter requires independent obs noise)
        self.h_diag = jnp.diag(self.H)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def _forward(self, ys):
        """Univariate exact diffuse forward pass.

        Stores per-period and per-observation quantities required by the
        backward smoother.

        Returns:
            fwd: dict of arrays indexed by [T_steps, ...]:
                'a'             (T, n_state)       predicted state a_{t|t-1}
                'Pstar'         (T, n_state, n_state)
                'Pinf'          (T, n_state, n_state)
                'was_diffuse'   (T,) bool - True while diffuse phase active
                'v'             (T, n_obs) innovations
                'f_inf'         (T, n_obs) diffuse innovation variance
                'f_star'        (T, n_obs) stationary innovation variance
                'k_inf'         (T, n_obs, n_state) diffuse Kalman gain
                'k_star'        (T, n_obs, n_state) stationary Kalman gain
                'is_diffuse_obs'(T, n_obs) bool
                'is_stat_obs'   (T, n_obs) bool
            ll: scalar log-likelihood (exact diffuse)
        """
        T_mat, C, Q, c = self.T, self.C, self.Q, self.c
        h_diag = self.h_diag
        n_state = self.n_state
        dt = self.dt
        dtol, ktol = self.dtol, self.ktol

        # One-step-ahead initialization (matching Octave convention):
        #   a_{1|0} = c + T @ init_a
        #   Pstar_{1|0} = T @ Pstar_{0|0} @ T' + Q
        #   Pinf_{1|0}  = T @ Pinf_{0|0} @ T'
        # For pure random walks T @ I @ T' = I, so Pinf is unchanged;
        # for the stationary block this gives the proper predictive Pstar.
        a0 = c + T_mat @ self.init_a
        Pstar0 = T_mat @ self.Pstar @ T_mat.T + Q
        Pstar0 = (Pstar0 + Pstar0.T) * 0.5
        Pinf0 = T_mat @ self.Pinf @ T_mat.T
        Pinf0 = (Pinf0 + Pinf0.T) * 0.5

        # NaN masking
        nan_mask = jnp.isnan(ys)            # (T_steps, n_obs)
        ys_clean = jnp.where(nan_mask, 0.0, ys)

        # ---- inner observation step --------------------------------
        # Closed over: diffuse_active (from outer carry), C, h_diag, dtol, ktol
        def _obs_step(carry_obs, obs_inp):
            """Update state for one scalar observation (univariate filter)."""
            a_o, Pstar_o, Pinf_o, ll_o, diffuse_active = carry_obs
            zi, h_ii, y_i, missing_i = obs_inp   # zi: (n_state,), scalars

            vi = y_i - jnp.dot(zi, a_o)
            f_inf_i  = jnp.dot(zi, Pinf_o @ zi)
            f_star_i = jnp.dot(zi, Pstar_o @ zi) + h_ii
            k_inf_i  = Pinf_o @ zi          # (n_state,)
            k_star_i = Pstar_o @ zi         # (n_state,)

            is_diffuse_obs = diffuse_active & (f_inf_i > dtol) & ~missing_i
            is_stat_obs    = (~is_diffuse_obs) & (f_star_i > ktol) & ~missing_i

            f_inf_safe  = jnp.where(f_inf_i  > dtol, f_inf_i,  1.0)
            f_star_safe = jnp.where(f_star_i > ktol, f_star_i, 1.0)

            # State update: diffuse case takes priority
            a_diff = a_o + k_inf_i * (vi / f_inf_safe)
            a_stat = a_o + k_star_i * (vi / f_star_safe)
            a_new = jnp.where(is_diffuse_obs, a_diff,
                    jnp.where(is_stat_obs,    a_stat, a_o))

            # Pstar update (D&K 2012, eq. 5.7 / Koopman-Durbin 2003 eq. 2.4)
            Pstar_diff = (Pstar_o
                + jnp.outer(k_inf_i, k_inf_i) * (f_star_i / f_inf_safe ** 2)
                - jnp.outer(k_star_i, k_inf_i) / f_inf_safe
                - jnp.outer(k_inf_i, k_star_i) / f_inf_safe)
            Pstar_stat = Pstar_o - jnp.outer(k_star_i, k_star_i) / f_star_safe
            Pstar_new = jnp.where(is_diffuse_obs, Pstar_diff,
                        jnp.where(is_stat_obs,    Pstar_stat, Pstar_o))
            Pstar_new = (Pstar_new + Pstar_new.T) * 0.5

            # Pinf update: only for diffuse observations (D&K 2012, eq. 5.7)
            Pinf_new = Pinf_o - jnp.outer(k_inf_i, k_inf_i) / f_inf_safe
            Pinf_new = jnp.where(is_diffuse_obs, Pinf_new, Pinf_o)
            Pinf_new = (Pinf_new + Pinf_new.T) * 0.5

            # Log-likelihood: only stationary observations contribute
            # (diffuse obs with f_inf > dtol are excluded - exact diffuse LL)
            ll_i = jnp.where(
                is_stat_obs,
                -0.5 * (jnp.log(2.0 * jnp.pi)
                        + jnp.log(f_star_safe)
                        + vi ** 2 / f_star_safe),
                0.0)

            stored = (vi, f_inf_i, f_star_i, k_inf_i, k_star_i,
                      is_diffuse_obs, is_stat_obs)
            return (a_new, Pstar_new, Pinf_new, ll_o + ll_i,
                    diffuse_active), stored

        # ---- outer period step ------------------------------------
        def _period_step(carry, t_inp):
            a, Pstar, Pinf, ll, diffuse_active = carry
            y_t, nm_t = t_inp   # (n_obs,), (n_obs,) bool

            # Store pre-update predicted quantities
            a_pre     = a
            Pstar_pre = Pstar
            Pinf_pre  = Pinf

            # Univariate obs scan (inner loop over p observations)
            # diffuse_active is captured from carry via closure
            obs_inputs = (C, h_diag, y_t, nm_t)  # each (n_obs,) or (n_obs,n_state)
            init_obs = (a, Pstar, Pinf, 0.0, diffuse_active)
            (a_up, Pstar_up, Pinf_up, ll_t, _), obs_out = lax.scan(
                _obs_step, init_obs, obs_inputs)

            # One-step prediction
            a_pred     = c + T_mat @ a_up
            Pstar_pred = T_mat @ Pstar_up @ T_mat.T + Q
            Pstar_pred = (Pstar_pred + Pstar_pred.T) * 0.5
            Pinf_pred  = T_mat @ Pinf_up @ T_mat.T
            Pinf_pred  = (Pinf_pred + Pinf_pred.T) * 0.5

            # Diffuse phase ends when Pinf becomes numerically zero
            diffuse_active_new = (diffuse_active
                                  & (jnp.max(jnp.abs(Pinf_pred)) > dtol))

            period_out = {
                'a':              a_pre,
                'Pstar':          Pstar_pre,
                'Pinf':           Pinf_pre,
                'was_diffuse':    diffuse_active,
                'v':              obs_out[0],    # (n_obs,)
                'f_inf':          obs_out[1],    # (n_obs,)
                'f_star':         obs_out[2],    # (n_obs,)
                'k_inf':          obs_out[3],    # (n_obs, n_state)
                'k_star':         obs_out[4],    # (n_obs, n_state)
                'is_diffuse_obs': obs_out[5],    # (n_obs,) bool
                'is_stat_obs':    obs_out[6],    # (n_obs,) bool
            }
            return (a_pred, Pstar_pred, Pinf_pred,
                    ll + ll_t, diffuse_active_new), period_out

        init_carry = (a0, Pstar0, Pinf0, jnp.array(0.0, dtype=dt),
                      jnp.array(True))
        (_, _, _, ll, _), fwd = lax.scan(
            _period_step, init_carry,
            (ys_clean, nan_mask))

        return fwd, ll

    # ------------------------------------------------------------------
    # Log-likelihood (forward pass only)
    # ------------------------------------------------------------------
    def log_likelihood(self, ys):
        """Exact diffuse log-likelihood.

        Excludes contributions from diffuse observations (those for which
        f_inf > dtol during the diffuse phase), consistent with the exact
        diffuse log-likelihood of Durbin & Koopman (2012), Section 7.2.

        Args:
            ys: (T, n_obs) observations, may contain NaN for missing data.

        Returns:
            Scalar log-likelihood.
        """
        _, ll = self._forward(jnp.asarray(ys, dtype=self.dt))
        return ll

    # ------------------------------------------------------------------
    # Compact forward pass (no per-period storage)
    # ------------------------------------------------------------------
    def _forward_compact(self, ys):
        """Univariate exact diffuse forward pass without per-period storage.

        Identical to ``_forward`` but returns only the final carry
        ``(a_end, Pstar_end, ll)`` - no per-period arrays are allocated.
        This halves JAX memory use and is used as Phase 1 in
        ``log_likelihood_fast``.

        ``Pstar_end`` is the **predicted** covariance for the next period
        (i.e. ``P_{n_diffuse+1 | n_diffuse}``), ready to be used as the
        starting P_pred for the Phase 2 steady-state filter.

        Returns:
            a_end:     (n_state,) predicted state for period n_diffuse+1.
            Pstar_end: (n_state, n_state) predicted covariance (~= P_pred_ss).
            ll:        scalar log-likelihood for the n_diffuse periods.
        """
        T_mat, C, Q, c = self.T, self.C, self.Q, self.c
        h_diag = self.h_diag
        dt = self.dt
        dtol, ktol = self.dtol, self.ktol

        a0     = c + T_mat @ self.init_a
        Pstar0 = T_mat @ self.Pstar @ T_mat.T + Q
        Pstar0 = (Pstar0 + Pstar0.T) * 0.5
        Pinf0  = T_mat @ self.Pinf @ T_mat.T
        Pinf0  = (Pinf0 + Pinf0.T) * 0.5

        nan_mask = jnp.isnan(ys)
        ys_clean = jnp.where(nan_mask, 0.0, ys)

        def _obs_step(carry_obs, obs_inp):
            a_o, Pstar_o, Pinf_o, ll_o, diffuse_active = carry_obs
            zi, h_ii, y_i, missing_i = obs_inp

            vi       = y_i - jnp.dot(zi, a_o)
            f_inf_i  = jnp.dot(zi, Pinf_o  @ zi)
            f_star_i = jnp.dot(zi, Pstar_o @ zi) + h_ii
            k_inf_i  = Pinf_o  @ zi
            k_star_i = Pstar_o @ zi

            is_diffuse_obs = diffuse_active & (f_inf_i > dtol) & ~missing_i
            is_stat_obs    = (~is_diffuse_obs) & (f_star_i > ktol) & ~missing_i

            f_inf_safe  = jnp.where(f_inf_i  > dtol, f_inf_i,  1.0)
            f_star_safe = jnp.where(f_star_i > ktol, f_star_i, 1.0)

            a_diff = a_o + k_inf_i  * (vi / f_inf_safe)
            a_stat = a_o + k_star_i * (vi / f_star_safe)
            a_new  = jnp.where(is_diffuse_obs, a_diff,
                     jnp.where(is_stat_obs,    a_stat, a_o))

            Pstar_diff = (Pstar_o
                + jnp.outer(k_inf_i, k_inf_i) * (f_star_i / f_inf_safe ** 2)
                - jnp.outer(k_star_i, k_inf_i) / f_inf_safe
                - jnp.outer(k_inf_i, k_star_i) / f_inf_safe)
            Pstar_stat = Pstar_o - jnp.outer(k_star_i, k_star_i) / f_star_safe
            Pstar_new  = jnp.where(is_diffuse_obs, Pstar_diff,
                         jnp.where(is_stat_obs,    Pstar_stat, Pstar_o))
            Pstar_new  = (Pstar_new + Pstar_new.T) * 0.5

            Pinf_new = Pinf_o - jnp.outer(k_inf_i, k_inf_i) / f_inf_safe
            Pinf_new = jnp.where(is_diffuse_obs, Pinf_new, Pinf_o)
            Pinf_new = (Pinf_new + Pinf_new.T) * 0.5

            ll_i = jnp.where(
                is_stat_obs,
                -0.5 * (jnp.log(2.0 * jnp.pi)
                        + jnp.log(f_star_safe)
                        + vi ** 2 / f_star_safe),
                0.0)

            return (a_new, Pstar_new, Pinf_new, ll_o + ll_i,
                    diffuse_active), None

        def _period_step(carry, t_inp):
            a, Pstar, Pinf, ll, diffuse_active = carry
            y_t, nm_t = t_inp

            init_obs = (a, Pstar, Pinf, 0.0, diffuse_active)
            (a_up, Pstar_up, Pinf_up, ll_t, _), _ = lax.scan(
                _obs_step, init_obs, (C, h_diag, y_t, nm_t))

            a_pred     = c + T_mat @ a_up
            Pstar_pred = T_mat @ Pstar_up @ T_mat.T + Q
            Pstar_pred = (Pstar_pred + Pstar_pred.T) * 0.5
            Pinf_pred  = T_mat @ Pinf_up  @ T_mat.T
            Pinf_pred  = (Pinf_pred + Pinf_pred.T) * 0.5

            diffuse_active_new = (diffuse_active
                                  & (jnp.max(jnp.abs(Pinf_pred)) > dtol))

            return (a_pred, Pstar_pred, Pinf_pred,
                    ll + ll_t, diffuse_active_new), None

        init_carry = (a0, Pstar0, Pinf0, jnp.array(0.0, dtype=dt),
                      jnp.array(True))
        (a_end, Pstar_end, _, ll, _), _ = lax.scan(
            _period_step, init_carry, (ys_clean, nan_mask))

        return a_end, Pstar_end, ll

    # ------------------------------------------------------------------
    # Fast two-segment log-likelihood
    # ------------------------------------------------------------------
    def log_likelihood_fast(self, ys, n_diffuse=10):
        """Two-segment exact diffuse log-likelihood for MCMC (fast path).

        Phase 1 - **exact diffuse filter** for the first ``n_diffuse``
        periods (univariate, Pinf + Pstar propagation, O(n^2*p*n_diffuse)).
        Phase 2 - **univariate steady-state filter** for the remaining
        ``T - n_diffuse`` periods using per-observable scalar gains
        precomputed from the hand-off ``Pstar`` (O(n*p*(T-n_diffuse))).

        The hand-off ``Pstar_end`` returned by Phase 1 is already the
        *predicted* covariance ``P_{n_diffuse+1 | n_diffuse}``.  After
        ``n_diffuse`` periods the Pinf has decayed to zero and Pstar has
        converged close to the DARE steady-state, so the per-obs gains
        are essentially constant for the rest of the sample.

        NaN observations are handled correctly in both phases: Phase 1
        via the ``missing_i`` flag in ``_forward_compact``; Phase 2 via
        zeroing the innovation and suppressing the LL contribution.

        **Approximation**: Phase 2 uses fixed per-obs gains derived from
        ``Pstar_end`` rather than re-solving DARE.  For ``n_diffuse >= 8``
        this introduces LL errors < 0.01 nats (negligible for MCMC).

        Args:
            ys: (T, n_obs) observations, may contain NaN.
            n_diffuse: static Python int - upper bound for the diffuse
                phase.  Must be >= the true diffuse phase length (= number
                of periods until Pinf ~= 0).  Default 10 is ample for QPM
                models (empirically 3-5 periods suffice).

        Returns:
            Scalar log-likelihood.
        """
        ys_arr  = jnp.asarray(ys, dtype=self.dt)
        ys1     = ys_arr[:n_diffuse]
        ys2     = ys_arr[n_diffuse:]

        # ---- Phase 1: exact diffuse (compact - no per-period storage) ----
        a_end, Pstar_end, ll1 = self._forward_compact(ys1)
        # a_end:    predicted state   a_{n_diffuse+1 | n_diffuse}
        # Pstar_end predicted cov  P_{n_diffuse+1 | n_diffuse} ~= P_pred_ss

        # ---- Precompute per-obs steady-state scalar gains ----------------
        # Sequential univariate update of P_pred starting from Pstar_end.
        # After n_diffuse Riccati steps, Pstar_end ~= P_pred_ss, so these
        # gains are essentially the true steady-state values.
        C      = self.C
        h_diag = self.h_diag
        T_mat  = self.T
        c      = self.c
        ktol   = self.ktol

        def _precompute_obs(P, obs_inp):
            zi, h_ii = obs_inp
            f_i  = jnp.dot(zi, P @ zi) + h_ii
            k_i  = P @ zi                      # (n_state,) unnormalized gain
            f_safe = jnp.where(f_i > ktol, f_i, 1.0)
            P_new  = P - jnp.outer(k_i, k_i) / f_safe
            P_new  = (P_new + P_new.T) * 0.5
            return P_new, (f_i, k_i)

        _, (f_ss, k_ss) = lax.scan(_precompute_obs, Pstar_end, (C, h_diag))
        # f_ss: (n_obs,)          - scalar innovation variances
        # k_ss: (n_obs, n_state)  - unnormalized Kalman gains

        # ---- Phase 2: univariate steady-state filter --------------------
        nan_mask2 = jnp.isnan(ys2)
        ys2_clean = jnp.where(nan_mask2, 0.0, ys2)

        def _obs_step2(carry, obs_inp):
            a_o, ll_o = carry
            zi, y_i, h_ii, f_i, k_i, missing_i = obs_inp

            v_i    = y_i - jnp.dot(zi, a_o)
            f_safe = jnp.where(f_i > ktol, f_i, 1.0)
            a_new  = jnp.where(missing_i, a_o,
                               a_o + k_i * (v_i / f_safe))
            ll_i   = jnp.where(
                (~missing_i) & (f_i > ktol),
                -0.5 * (jnp.log(2.0 * jnp.pi)
                        + jnp.log(f_safe)
                        + v_i ** 2 / f_safe),
                0.0)
            return (a_new, ll_o + ll_i), None

        def _period_step2(carry, t_inp):
            a, ll = carry          # a is predicted state for this period
            y_t, nm_t = t_inp

            obs_inp = (C, y_t, h_diag, f_ss, k_ss, nm_t)
            (a_up, ll_t), _ = lax.scan(_obs_step2, (a, 0.0), obs_inp)

            a_pred = c + T_mat @ a_up   # predicted state for next period
            return (a_pred, ll + ll_t), None

        (_, ll2), _ = lax.scan(
            _period_step2,
            (a_end, jnp.array(0.0, dtype=self.dt)),
            (ys2_clean, nan_mask2))

        return ll1 + ll2

    # ------------------------------------------------------------------
    # Full smoother
    # ------------------------------------------------------------------
    def smooth(self, ys):
        """Exact diffuse smoother.

        Returns smoothed states, smoothed shocks, and posterior shock
        variances.  Implements the two-phase backward pass of Durbin &
        Koopman (2012): standard r/N recursion for the non-diffuse phase
        and r0/r1/N0 for the diffuse phase.

        Args:
            ys: (T, n_obs) observations, may contain NaN.

        Returns:
            dict with:
              'alphahat': (T, n_state) smoothed states
              'etahat':   (T, n_shock) smoothed shocks
              'V_eta':    (T, n_shock) diagonal of shock posterior variance
              'loglik':   scalar log-likelihood
              'a':        (T, n_state) predicted states a_{t|t-1}
              'd':        scalar (approximate) last diffuse period index
        """
        ys_arr = jnp.asarray(ys, dtype=self.dt)
        fwd, ll = self._forward(ys_arr)
        bwd = self._backward(fwd)
        return {**bwd, 'loglik': ll, 'a': fwd['a']}

    # ------------------------------------------------------------------
    # API parity with KalmanFilter: smoothed states only
    # ------------------------------------------------------------------
    def smooth_states_only(self, ys=None, filter_results=None,
                            n_tv=None, H_t=None, Q_t=None):
        """Drop-in compatible with ``KalmanFilter.smooth_states_only``.

        Returns the smoothed state mean array ``alphahat`` of shape
        ``(T, n_state)`` from the two-phase diffuse smoother.

        ``H_t`` / ``Q_t`` / ``n_tv`` time-varying matrices are NOT yet
        threaded through the diffuse forward pass; pass ``None`` (or
        omit) when ``use_diffuse=True``.  Callers that need a
        time-varying R should fall back to the standard filter.
        """
        if H_t is not None or Q_t is not None or n_tv is not None:
            raise NotImplementedError(
                "DiffuseKalmanFilter.smooth_states_only does not yet "
                "support H_t / Q_t / n_tv.  Pass use_diffuse=False to "
                "use the standard filter for time-varying matrices.")
        if filter_results is None:
            ys_arr = jnp.asarray(ys, dtype=self.dt)
            filter_results, _ = self._forward(ys_arr)
        bwd = self._backward(filter_results)
        return bwd['alphahat']

    # ------------------------------------------------------------------
    # Disturbance smoother (Koopman 1993; D&K 2012)
    # ------------------------------------------------------------------
    def smooth_disturbances(self, ys=None, filter_results=None,
                            shock_scale_t=None):
        """Two-phase disturbance smoother for structural shocks.

        Returns the posterior mean and diagonal variance of each shock at
        each period in unit-variance space (``eps_t ~ N(0, I)``, with
        sigma absorbed in ``R``).  The recursions branch by phase using
        the per-period ``was_diffuse`` flag stored by the forward pass:

          * Standard phase (``Pinf = 0``)::

                eps_hat_t = R' r_t,    V_t = I - R' N_t  R

          * Diffuse phase  (``Pinf != 0``)::

                eps_hat_t = R' r0_t,   V_t = I - R' N0_t R

        See Durbin & Koopman (2012), eqs. 4.69 (standard) and 5.27/5.30
        (diffuse).  This is a thin wrapper around :meth:`_backward`,
        which already computes the right quantities and selects the
        correct ``r``/``N`` variant per period.

        Drop-in compatible with :meth:`KalmanFilter.smooth_disturbances`
        (same return signature and unit-variance convention).

        Args:
            ys: (T, n_obs) observations.  Ignored when ``filter_results``
                is supplied.  May contain NaN.
            filter_results: pre-computed ``_forward(ys)`` output, to
                avoid re-running the forward pass.
            shock_scale_t: time-varying shock scale.  Not yet supported
                in the diffuse path - must be ``None``.  Calibrators
                that need a deterministic per-window sigma scaling (e.g.
                COVID dummy) should fall back to the standard filter
                via ``use_diffuse=False`` until this is implemented.

        Returns:
            eps_hat:   (T, n_shock) smoothed shock means
            V_diag:    (T, n_shock) smoothed shock variances (diagonal)
        """
        if shock_scale_t is not None:
            raise NotImplementedError(
                "DiffuseKalmanFilter.smooth_disturbances: shock_scale_t "
                "is not yet supported.  Pass shock_scale_t=None, or use "
                "the standard filter (use_diffuse=False) when a "
                "scale_schedule is required.\n"
                "TODO: thread time-varying R = R_base * scale through "
                "_forward (Q_t) and _backward (R_t' factor in the "
                "disturbance map).")

        if filter_results is None:
            ys_arr = jnp.asarray(ys, dtype=self.dt)
            filter_results, _ = self._forward(ys_arr)

        bwd = self._backward(filter_results)
        return bwd['etahat'], bwd['V_eta']

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def _backward(self, fwd):
        """Two-phase backward smoother.

        Phase 1 (non-diffuse, t > d):  standard r, N recursion.
        Phase 2 (diffuse,    t <= d):  r0, r1, N0 recursion.

        The transition is detected via the stored `was_diffuse` flag.
        The scan is unified over all T periods (reversed), using
        `jnp.where` to select the appropriate update at each step.
        """
        T_mat, R, Q = self.T, self.R, self.Q
        n_state, n_shock = R.shape
        dt = self.dt
        dtol, ktol = self.dtol, self.ktol

        # Unpack forward storage
        a_all     = fwd['a']              # (T, n_state)
        Pstar_all = fwd['Pstar']          # (T, n_state, n_state)
        Pinf_all  = fwd['Pinf']           # (T, n_state, n_state)
        wd_all    = fwd['was_diffuse']    # (T,) bool
        v_all     = fwd['v']              # (T, n_obs)
        f_inf_all = fwd['f_inf']          # (T, n_obs)
        f_star_all = fwd['f_star']        # (T, n_obs)
        k_inf_all  = fwd['k_inf']         # (T, n_obs, n_state)
        k_star_all = fwd['k_star']        # (T, n_obs, n_state)
        id_obs_all = fwd['is_diffuse_obs']  # (T, n_obs) bool
        is_obs_all = fwd['is_stat_obs']     # (T, n_obs) bool

        n_obs    = self.n_obs
        C        = self.C
        I_s      = jnp.eye(n_state, dtype=dt)

        # ---- backward observation step --------------------------------
        # Closed over: was_diffuse_t (from outer backward carry via closure)
        def _bwd_obs_step(carry_obs, obs_inp):
            r, N, r0, r1, N0, was_diff_t = carry_obs
            zi, vi, f_inf_i, f_star_i, k_inf_i, k_star_i, id_i, is_i = obs_inp

            f_star_safe = jnp.where(f_star_i > ktol, f_star_i, 1.0)
            f_inf_safe  = jnp.where(f_inf_i  > dtol, f_inf_i,  1.0)

            # --- Non-diffuse period: update r, N ---
            L_stat  = I_s - jnp.outer(k_star_i, zi) / f_star_safe
            r_stat  = zi * (vi / f_star_safe) + L_stat.T @ r
            N_stat  = (jnp.outer(zi, zi) / f_star_safe
                       + L_stat.T @ N @ L_stat)
            N_stat  = (N_stat + N_stat.T) * 0.5

            # Apply only for non-diffuse period, stationary obs
            do_rN = is_i & ~was_diff_t
            r = jnp.where(do_rN, r_stat, r)
            N = jnp.where(do_rN, N_stat, N)

            # --- Diffuse period: update r0, r1, N0 ---
            # Diffuse observation (f_inf > dtol): D&K 2012 eq. 5.21
            L_inf = I_s - jnp.outer(k_inf_i, zi) / f_inf_safe
            L0    = (jnp.outer(k_inf_i * (f_star_i / f_inf_safe)
                               - k_star_i, zi) / f_inf_safe)

            # r1 uses old r0 (order matters, D&K 2012 eq. 5.21)
            r1_diff = zi * (vi / f_inf_safe) + L0.T @ r0 + L_inf.T @ r1
            r0_diff = L_inf.T @ r0
            N0_diff = L_inf.T @ N0 @ L_inf
            N0_diff = (N0_diff + N0_diff.T) * 0.5

            # Stationary observation in diffuse period: D&K 2012 eq. 5.21
            r0_stat_d = zi * (vi / f_star_safe) + L_stat.T @ r0
            N0_stat_d = (jnp.outer(zi, zi) / f_star_safe
                         + L_stat.T @ N0 @ L_stat)
            N0_stat_d = (N0_stat_d + N0_stat_d.T) * 0.5

            r0 = jnp.where(id_i & was_diff_t, r0_diff,
                 jnp.where(is_i & was_diff_t, r0_stat_d, r0))
            r1 = jnp.where(id_i & was_diff_t, r1_diff, r1)
            N0 = jnp.where(id_i & was_diff_t, N0_diff,
                 jnp.where(is_i & was_diff_t, N0_stat_d, N0))

            return (r, N, r0, r1, N0, was_diff_t), None

        # ---- backward period step -------------------------------------
        def _bwd_period_step(carry_bwd, t_inp):
            r, N, r0, r1, N0, in_diffuse_bwd = carry_bwd
            (a_t, Pstar_t, Pinf_t, was_diff_t,
             v_t, f_inf_t, f_star_t, k_inf_t, k_star_t,
             id_t, is_t) = t_inp

            # --- Transition into the diffuse phase (backward) ---
            # First diffuse period encountered in backward scan:
            # inherit r -> r0, N -> N0 and reset r1 = 0.
            entering = was_diff_t & ~in_diffuse_bwd
            r0 = jnp.where(entering, r,                   r0)
            r1 = jnp.where(entering, jnp.zeros_like(r),   r1)
            N0 = jnp.where(entering, N,                   N0)
            in_diffuse_bwd = in_diffuse_bwd | was_diff_t

            # --- Backward observation scan (reverse observation order) ---
            obs_inp = (C[::-1], v_t[::-1], f_inf_t[::-1], f_star_t[::-1],
                       k_inf_t[::-1], k_star_t[::-1],
                       id_t[::-1], is_t[::-1])
            init_obs = (r, N, r0, r1, N0, was_diff_t)
            (r_up, N_up, r0_up, r1_up, N0_up, _), _ = lax.scan(
                _bwd_obs_step, init_obs, obs_inp)

            # --- Smoothed state (D&K 2012 eq. 4.44 / 5.23) ---
            alphahat_t = jnp.where(
                was_diff_t,
                a_t + Pstar_t @ r0_up + Pinf_t @ r1_up,
                a_t + Pstar_t @ r_up)

            # --- Smoothed shocks (D&K 2012 eq. 4.69 / p.135) ---
            # etahat_t = R' r  (non-diffuse) or R' r0  (diffuse)
            r_for_shocks  = jnp.where(was_diff_t, r0_up, r_up)
            N_for_shocks  = jnp.where(was_diff_t, N0_up, N_up)
            etahat_t      = R.T @ r_for_shocks           # (n_shock,)
            RtN           = R.T @ N_for_shocks            # (n_shock, n_state)
            V_eta_diag_t  = jnp.maximum(
                0.0, 1.0 - jnp.sum(RtN * R.T, axis=1))  # (n_shock,)

            # --- Time propagation (across periods) ---
            r_prop  = T_mat.T @ r_up
            N_prop  = T_mat.T @ N_up  @ T_mat
            r0_prop = T_mat.T @ r0_up
            r1_prop = T_mat.T @ r1_up
            N0_prop = T_mat.T @ N0_up @ T_mat

            return (r_prop, N_prop, r0_prop, r1_prop,
                    N0_prop, in_diffuse_bwd), (alphahat_t, etahat_t, V_eta_diag_t)

        # Reversed inputs for backward scan
        bwd_inp = (
            a_all[::-1], Pstar_all[::-1], Pinf_all[::-1], wd_all[::-1],
            v_all[::-1], f_inf_all[::-1], f_star_all[::-1],
            k_inf_all[::-1], k_star_all[::-1],
            id_obs_all[::-1], is_obs_all[::-1])

        init_bwd = (
            jnp.zeros(n_state, dtype=dt),             # r
            jnp.zeros((n_state, n_state), dtype=dt),  # N
            jnp.zeros(n_state, dtype=dt),             # r0
            jnp.zeros(n_state, dtype=dt),             # r1
            jnp.zeros((n_state, n_state), dtype=dt),  # N0
            jnp.array(False))                          # in_diffuse_bwd

        _, (alpha_rev, eta_rev, V_rev) = lax.scan(
            _bwd_period_step, init_bwd, bwd_inp)

        # Approximate last diffuse period (for diagnostics)
        d_approx = jnp.sum(wd_all.astype(jnp.int32))

        return {
            'alphahat': alpha_rev[::-1],   # (T, n_state)
            'etahat':   eta_rev[::-1],     # (T, n_shock)
            'V_eta':    V_rev[::-1],       # (T, n_shock)
            'd':        d_approx,
        }


# ======================================================================
# Utility: build Pinf and Pstar from solved state-transition matrix
# ======================================================================

def build_diffuse_init(P_aug, R_aug, unit_root_threshold=0.999,
                       dtype=None):
    """Build diffuse initialization matrices from the solved state-space.

    Identifies unit-root states via spectral analysis of ``P_aug`` and
    constructs:

    * ``Pinf``:  projection onto the unit-root subspace.  Column space
      spanned by the eigenvectors for eigenvalues with modulus >=
      ``unit_root_threshold``.  For QPM models the trend block has exact
      unit-root eigenvalues (= 1.000); the stationary block's most
      persistent state (e.g. LZ_GAP) has eigenvalue ~= 0.90 in the solved
      RE system and is correctly treated as stationary.
    * ``Pstar``: stationary initial covariance.  Computed by zeroing
      unit-root rows/cols in T and Q and solving the discrete Lyapunov
      equation with ``scipy.linalg.solve_discrete_lyapunov``.

    This is a **one-time** computation, not part of the inner MCMC loop.
    The unit-root structure is a model-design property that does not change
    across parameter draws for QPM models.

    Args:
        P_aug: (n_state, n_state) solved state-transition matrix.
        R_aug: (n_state, n_shock) shock-impact matrix.
        unit_root_threshold: eigenvalue modulus threshold.  Default 0.999
            identifies exact unit roots (eigenvalue = 1.000) while leaving
            near-unit-root stationary states (eigenvalue ~= 0.90) in the
            Pstar Lyapunov solve.  Raise to 0.9999 for stricter detection.

    Returns:
        Pinf:  (n_state, n_state) float array - diffuse covariance.
        Pstar: (n_state, n_state) float array - stationary covariance.
        unit_root_mask: (n_state,) bool array - True for non-stationary
            state directions.
    """
    import numpy as _np
    import scipy.linalg as _sp

    dt = dtype or get_dtype()
    P_np = _np.array(P_aug, dtype=_np.float64)
    R_np = _np.array(R_aug, dtype=_np.float64)
    Q_np = R_np @ R_np.T
    n = P_np.shape[0]

    # --- Identify unit-root eigenvalues ---
    evals, evecs = _np.linalg.eig(P_np)
    ur_eval_mask = _np.abs(evals) >= unit_root_threshold
    n_ur = int(ur_eval_mask.sum())

    # --- Unit-root state mask ---
    # For block-diagonal P_aug (P_stat | P_trend), the unit-root eigenvectors
    # are localized to the trend block and have real eigenvalue = 1.
    # We identify state i as a unit root if its projection onto the unit-root
    # eigenspace Pinf[i,i] > 0.5 (i.e., more than half its "mass" is
    # in the non-stationary subspace).
    if n_ur == 0:
        Pinf_np = _np.zeros((n, n))
        ur_state_mask_np = _np.zeros(n, dtype=bool)
    else:
        # Orthonormalize unit-root eigenvectors (take real part; for P_aug
        # with real unit-root eigenvalues the eigenvectors are also real).
        M = evecs[:, ur_eval_mask].real     # (n, n_ur)
        M, _ = _np.linalg.qr(M)            # orthonormal columns
        Pinf_np = M @ M.T
        Pinf_np = 0.5 * (Pinf_np + Pinf_np.T)
        ur_state_mask_np = _np.diag(Pinf_np) > 0.5

    # --- Pstar: discrete Lyapunov on stationary subspace ---
    # Zero out unit-root rows/cols so those directions decouple with
    # zero variance contribution.  The remaining stationary block has
    # spectral radius < 1, so solve_discrete_lyapunov converges.
    ur_idx = _np.where(ur_state_mask_np)[0]
    P_mod = P_np.copy()
    Q_mod = Q_np.copy()
    if len(ur_idx) > 0:
        P_mod[_np.ix_(ur_idx, _np.arange(n))] = 0.0
        P_mod[_np.ix_(_np.arange(n), ur_idx)] = 0.0
        Q_mod[_np.ix_(ur_idx, _np.arange(n))] = 0.0
        Q_mod[_np.ix_(_np.arange(n), ur_idx)] = 0.0

    # scipy solves  A X A' - X + Q = 0  <=>  X = A X A' + Q  (the Lyapunov)
    Pstar_np = _sp.solve_discrete_lyapunov(P_mod, Q_mod)
    Pstar_np = 0.5 * (Pstar_np + Pstar_np.T)

    return (jnp.asarray(Pinf_np,          dtype=dt),
            jnp.asarray(Pstar_np,         dtype=dt),
            jnp.asarray(ur_state_mask_np, dtype=jnp.bool_))


# ======================================================================
# JAX-native discrete Lyapunov solver (doubling algorithm)
# ======================================================================
#
# Solves  X = A X A' + Q  via the matrix-doubling recurrence:
#
#     A_{k+1} = A_k  @  A_k
#     Q_{k+1} = Q_k  +  A_k  @  Q_k  @  A_k'
#
# which computes  X = sum_{k=0}^{2^n - 1}  A^k Q (A')^k  in n passes.
# For spectral radius rho(A) < 1 this converges to the Lyapunov solution.
#
# Use: pass A and Q with unit-root rows/cols already zeroed; the zero
# mask is respected exactly throughout the iteration.
#
# This function is JIT-traceable and fully AD-compatible via jax.lax.scan.
# It is designed for use inside the MCMC log-likelihood, where A = P_aug
# (masked) and Q = R_aug @ R_aug' (masked), both of which change with
# parameters and require gradients.
#
# Reference:
#   Hammarling, S.J. (1982). Numerical Solution of the Stable, Non-negative
#   Definite Lyapunov Equation. IMA J. Numer. Anal. 2(3), 303-323.
#   (Doubling iteration, p. 315.)

def solve_lyapunov_diffuse_jax(T_masked, Q_masked, n_iter=60, dtype=None):
    """JAX-native discrete Lyapunov solver for the diffuse Kalman filter.

    Solves  X = T_masked @ X @ T_masked' + Q_masked  using the matrix-
    doubling algorithm.  Designed for use inside JIT-compiled MCMC loops.

    ``T_masked`` and ``Q_masked`` must have the unit-root rows and columns
    already set to zero (use ``ur_mask`` from ``build_diffuse_init``).
    This ensures those directions contribute zero variance and the
    spectral radius of the remaining block is strictly less than 1.

    Args:
        T_masked: (n, n) state transition matrix with unit-root rows/cols
                  zeroed.  Typically ``jnp.where(stat_2d, P_aug, 0.0)``
                  where ``stat_2d = jnp.outer(~ur_mask, ~ur_mask)``.
        Q_masked: (n, n) process noise covariance with unit-root rows/cols
                  zeroed.  Typically ``jnp.where(stat_2d, R_aug @ R_aug', 0.0)``.
        n_iter:   number of doubling steps.  60 iterations give
                  machine-precision accuracy for rho(T) <= 0.99.
        dtype:    float dtype.  Defaults to ``get_dtype()``.

    Returns:
        Pstar: (n, n) symmetric positive semi-definite Lyapunov solution.
               Zero rows/cols for unit-root states.

    Notes:
        This function is intended to be called inside ``log_likelihood_diffuse``
        at each MCMC iteration.  Compile with JAX JIT for efficiency.
    """
    dt = dtype or get_dtype()
    A = jnp.asarray(T_masked, dtype=dt)
    G = jnp.asarray(Q_masked, dtype=dt)

    def _step(carry, _):
        Ai, Gi = carry
        Gi_new = Gi + Ai @ Gi @ Ai.T
        Ai_new = Ai @ Ai
        return (Ai_new, Gi_new), None

    (_, Pstar), _ = jax.lax.scan(_step, (A, G), None, length=n_iter)
    return (Pstar + Pstar.T) * jnp.array(0.5, dtype=dt)

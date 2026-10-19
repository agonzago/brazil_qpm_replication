"""Model wrapper: parse a .qpm/.dyn file, solve, and compute log-likelihood."""

from typing import Dict, Any

import jax
import jax.numpy as jnp

from qpm_toolbox.kalman_filter import (
    KalmanFilter, DiffuseKalmanFilter, build_diffuse_init,
    solve_lyapunov_diffuse_jax, get_dtype,
)
from qpm_toolbox.parser import (
    parse_lambdify_and_order_model,
    build_trend_matrices as build_trend_matrices_lambdified,
    build_observation_matrix as build_observation_matrix_lambdified,
    extract_trend_declarations, extract_trend_equations,
    extract_observation_declarations, extract_measurement_equations,
    extract_trend_shock_stderrs, extract_stationary_shock_stderrs,
    extract_reporting_equations,
    solve_quadratic_matrix_equation_jax, compute_Q_jax,
)

_LARGE_NEG = -1e10


class DynareModelWithLambdified:
    """Parse a .dyn file, solve, and provide log-likelihood."""

    def __init__(self, mod_file_path: str, verbose: bool = False):
        self.mod_file_path = mod_file_path
        self.verbose = verbose
        self._parse_model()

    # ---- parsing ----
    def _parse_model(self):
        with open(self.mod_file_path, 'r', encoding='utf-8') as f:
            model_def = f.read()

        # parse_lambdify_and_order_model returns 13 elements:
        #   [0..9]  original outputs
        #   [10]    elim_var_names     : list[str]  - eliminated stationary var names
        #   [11]    func_S             : callable(params...) -> (n_elim, n_reduced) jnp array
        #                               or None when no elimination occurred
        #   [12]    elim_subs_symbolic : dict {var_name: sympy_expr}
        (self.func_A_stat, self.func_B_stat, self.func_C_stat, self.func_D_stat,
         self.ordered_stat_vars, self.stat_shocks, self.param_names_for_stat_funcs,
         self.param_assignments_stat, _, self.initial_info_stat,
         self.elim_var_names, self.func_S, self.elim_subs_symbolic,
         ) = parse_lambdify_and_order_model(model_def, verbose=self.verbose)

        self.trend_vars, self.trend_shocks = extract_trend_declarations(model_def)
        self.trend_equations = extract_trend_equations(model_def)
        self.trend_stderr_params = extract_trend_shock_stderrs(model_def)
        self.obs_vars = extract_observation_declarations(model_def)
        self.measurement_equations = extract_measurement_equations(model_def)

        # Combined parameter list
        current_names = list(self.param_names_for_stat_funcs)
        current_assigns = self.param_assignments_stat.copy()
        for p, v in self.trend_stderr_params.items():
            current_assigns.setdefault(p, v)
            if p not in current_names:
                current_names.append(p)
        for shk in self.trend_shocks:
            sig = f"sigma_{shk}"
            current_assigns.setdefault(sig, 1.0)
            if sig not in current_names:
                current_names.append(sig)

        self.all_param_names = list(dict.fromkeys(current_names))
        self.default_param_assignments = current_assigns

        # Trend / observation matrix generators
        (self.func_P_trends, self.func_Q_trends, self.func_c_trends,
         self.ordered_trend_state_vars, self.contemp_trend_defs,
         ) = build_trend_matrices_lambdified(
            self.trend_equations, self.trend_vars, self.trend_shocks,
            self.all_param_names, self.default_param_assignments,
            verbose=self.verbose)

        # Omega is built over the *reduced* stationary state.
        # Eliminated stationary variables are substituted out symbolically so
        # that Omega's columns correspond to (ordered_stat_vars, ordered_trend_state_vars).
        # Zero-padding for eliminated vars is added in solve().
        (self.func_Omega, self.ordered_obs_vars,
         ) = build_observation_matrix_lambdified(
            self.measurement_equations, self.obs_vars,
            self.ordered_stat_vars, self.ordered_trend_state_vars,
            self.contemp_trend_defs,
            self.all_param_names, self.default_param_assignments,
            verbose=self.verbose,
            eliminated_stat_defs=self.elim_subs_symbolic)

        # Dimensions
        # n_stat_reduced: stationary vars after elimination (solved by QME)
        # n_stat_elim   : eliminated stationary vars (reconstructed via S)
        # n_stat        : full stationary count (public state vector)
        self.n_stat_reduced = len(self.ordered_stat_vars)
        self.n_stat_elim    = len(self.elim_var_names)
        self.n_stat         = self.n_stat_reduced + self.n_stat_elim
        self.n_trend        = len(self.ordered_trend_state_vars)
        self.n_aug          = self.n_stat + self.n_trend
        self.n_s_shock      = len(self.stat_shocks)
        self.n_t_shock      = len(self.trend_shocks)
        self.n_aug_shock    = self.n_s_shock + self.n_t_shock
        self.n_obs          = len(self.ordered_obs_vars)

        # Full state variable ordering: reduced stat, then eliminated stat, then trend.
        # Eliminated vars are appended after the reduced stat block so existing code
        # that slices [:n_stat_reduced] still works.
        self.aug_state_vars = (self.ordered_stat_vars + self.elim_var_names
                               + self.ordered_trend_state_vars)
        self.aug_shocks     = self.stat_shocks + self.trend_shocks

        # Reporting variables - post-processing linear combinations of aug states.
        # Not part of the Kalman filter; computed after smoothing.
        self.reporting_vars, self.reporting_eqs = extract_reporting_equations(model_def)

    # ---- solve ----
    def solve(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
        """Solve the model for given parameters.

        Returns dict with P_aug, R_aug, Omega, solution_valid.

        When static variable elimination is active (n_stat_elim > 0), the
        full state vector is ordered as::

            [reduced_stat_vars | elim_stat_vars | trend_vars]

        The transition block for the full stationary partition is::

            P_stat_full = [[P_r,     0 ],
                           [S @ P_r, 0 ]]

        where P_r solves the reduced QME and S is the back-substitution map
        returned by the parser.  The shock-impact block is correspondingly::

            Q_stat_full = [[Q_r  ],
                           [S @ Q_r]]

        Omega is zero-padded in the eliminated-var columns because
        build_observation_matrix already substituted them out symbolically.
        """
        dt = get_dtype()

        stat_vals = [param_dict.get(p, self.default_param_assignments.get(p, 1.0))
                     for p in self.param_names_for_stat_funcs]
        A = jnp.asarray(self.func_A_stat(*stat_vals), dtype=dt)
        B = jnp.asarray(self.func_B_stat(*stat_vals), dtype=dt)
        C_mat = jnp.asarray(self.func_C_stat(*stat_vals), dtype=dt)
        D = jnp.asarray(self.func_D_stat(*stat_vals), dtype=dt)

        P_sol, _, _, converged = solve_quadratic_matrix_equation_jax(
            A, B, C_mat, tol=1e-12, max_iter=10)
        valid = converged & jnp.all(jnp.isfinite(P_sol))

        Q_sol = compute_Q_jax(A, B, D, P_sol)
        valid = valid & jnp.all(jnp.isfinite(Q_sol))

        all_vals = [param_dict.get(p, self.default_param_assignments.get(p, 1.0))
                    for p in self.all_param_names]
        P_trend = jnp.asarray(self.func_P_trends(*all_vals), dtype=dt)
        Q_trend = jnp.asarray(self.func_Q_trends(*all_vals), dtype=dt)
        c_trend = jnp.asarray(self.func_c_trends(*all_vals), dtype=dt).ravel()

        # Raw Omega covers (n_stat_reduced + n_trend) columns.
        Omega_raw = jnp.asarray(self.func_Omega(*all_vals), dtype=dt)

        # ---- Expand stationary block for eliminated variables ----
        if self.n_stat_elim > 0:
            # S: (n_elim, n_r) back-substitution map  x_elim = S @ x_r
            S = jnp.asarray(self.func_S(*stat_vals), dtype=dt)

            n_r = self.n_stat_reduced
            n_e = self.n_stat_elim

            # Full transition matrix for [x_r; x_e]:
            #   x_r(t+1) = P_r @ x_r(t)   (top-left only; top-right zero)
            #   x_e(t+1) = S @ P_r @ x_r(t)  (bottom-left; bottom-right zero)
            P_top    = jnp.concatenate(
                [P_sol, jnp.zeros((n_r, n_e), dtype=dt)], axis=1)
            P_bottom = jnp.concatenate(
                [S @ P_sol, jnp.zeros((n_e, n_e), dtype=dt)], axis=1)
            P_stat_full = jnp.concatenate([P_top, P_bottom], axis=0)

            # Full shock-impact matrix for [x_r; x_e]:
            Q_stat_full = jnp.concatenate([Q_sol, S @ Q_sol], axis=0)

            # Pad Omega: insert zero columns for eliminated vars between
            # the reduced-stat columns and the trend columns.
            Omega_full = jnp.concatenate([
                Omega_raw[:, :n_r],
                jnp.zeros((self.n_obs, n_e), dtype=dt),
                Omega_raw[:, n_r:],
            ], axis=1)
        else:
            P_stat_full = P_sol
            Q_stat_full = Q_sol
            Omega_full  = Omega_raw

        # Shock R matrices  (Q @ diag(sigma))
        def _get_sigma(shk):
            return jnp.maximum(jnp.abs(
                param_dict.get(f"sigma_{shk}",
                               self.default_param_assignments.get(f"sigma_{shk}", 1.0))
            ), 1e-9)

        stat_sigmas  = jnp.array([_get_sigma(s) for s in self.stat_shocks],  dtype=dt)
        trend_sigmas = jnp.array([_get_sigma(s) for s in self.trend_shocks], dtype=dt)

        R_stat = Q_stat_full @ jnp.diag(stat_sigmas) if self.n_s_shock > 0 \
            else jnp.zeros((self.n_stat, 0), dtype=dt)
        R_trend = Q_trend @ jnp.diag(trend_sigmas) if self.n_t_shock > 0 \
            else jnp.zeros((self.n_trend, 0), dtype=dt)

        P_aug = jax.scipy.linalg.block_diag(P_stat_full, P_trend)
        R_aug = jnp.zeros((self.n_aug, self.n_aug_shock), dtype=dt)
        if self.n_s_shock > 0:
            R_aug = R_aug.at[:self.n_stat, :self.n_s_shock].set(R_stat)
        if self.n_t_shock > 0:
            R_aug = R_aug.at[self.n_stat:, self.n_s_shock:].set(R_trend)

        # State intercept: c_aug = [0_stat ; c_trend]
        c_aug = jnp.concatenate([jnp.zeros(self.n_stat, dtype=dt), c_trend])

        valid = valid & jnp.all(jnp.isfinite(P_aug)) & jnp.all(jnp.isfinite(R_aug))

        # Blanchard-Kahn stationarity check: spectral radius of the
        # stationary block must be strictly < 1.  The trend block has
        # unit roots by design and is excluded from this check.
        stat_eigvals = jnp.linalg.eigvals(P_stat_full)
        spectral_radius = jnp.max(jnp.abs(stat_eigvals))
        valid = valid & (spectral_radius < 1.0)

        return {
            "P_aug": P_aug, "R_aug": R_aug, "Omega": Omega_full, "c_aug": c_aug,
            "solution_valid": jnp.asarray(valid, dtype=jnp.bool_),
            "spectral_radius": spectral_radius,
        }

    # ---- log-likelihood (single solve) ----
    def log_likelihood(self, param_dict, ys, H_obs, init_x, init_P,
                       presample=0, H_t=None, shock_multipliers=None):
        """Compute log-likelihood via the Kalman filter.

        Selects the fastest available filter:

        * **No missing data, no multipliers** -> steady-state Kalman gain
          (O(n^2 T), no Cholesky per step).
        * **Missing data (H_t provided) and/or shock multipliers** ->
          full time-varying Riccati (O(n^3 T) per call).

        Args:
            param_dict: parameter name -> value.
            ys: (T, n_obs) observations.  When H_t is provided, NaN
                entries must already be replaced by 0 (use
                ``prepare_observations``).
            H_obs: (n_obs, n_obs) measurement noise covariance.
            init_x: (n_aug,) initial state mean.
            init_P: (n_aug, n_aug) initial state covariance.
            presample: number of initial periods to discard from the
                log-likelihood sum (see KalmanFilter.log_likelihood).
            H_t: optional (T, n_obs, n_obs) per-period observation noise
                covariance.  Built by ``prepare_observations`` for
                missing-data handling (NaN -> inflated diagonal).
            shock_multipliers: optional (T, n_shock) array of per-period
                scale factors for structural shocks.  Entry [t,j] scales
                column j of the impact matrix R at time t.
        """
        dt = get_dtype()
        sol = self.solve(param_dict)

        def ll_valid():
            kf = KalmanFilter(T=sol["P_aug"], R=sol["R_aug"], C=sol["Omega"],
                              H=H_obs, init_x=init_x, init_P=init_P,
                              c=sol["c_aug"], dtype=dt)

            if H_t is not None or shock_multipliers is not None:
                # Build per-period Q_t from shock multipliers
                Q_t = None
                if shock_multipliers is not None:
                    mult = jnp.asarray(shock_multipliers, dtype=dt)
                    R_aug = sol["R_aug"]
                    R_scaled = R_aug[None, :, :] * mult[:, None, :]
                    Q_t = jnp.einsum('tij,tkj->tik', R_scaled, R_scaled)
                ll = kf.log_likelihood_tv(
                    ys, presample=presample, H_t=H_t, Q_t=Q_t)
            else:
                ll = kf.log_likelihood(ys, presample=presample)

            return jnp.where(jnp.isfinite(ll), ll,
                             jnp.array(_LARGE_NEG, dtype=dt))

        def ll_invalid():
            return jnp.array(_LARGE_NEG, dtype=dt)

        return jax.lax.cond(sol["solution_valid"], ll_valid, ll_invalid)

    # ---- diffuse initialization (one-time, per parameter vector) ----
    def build_diffuse_matrices(self, param_dict, H_obs,
                               unit_root_threshold=0.999):
        """Build Pinf and Pstar for the exact diffuse Kalman filter.

        Solves the model for ``param_dict``, then identifies unit-root
        states via spectral analysis of the augmented transition matrix
        ``P_aug``.  Returns the diffuse covariance ``Pinf`` (projector onto
        the non-stationary subspace), the stationary covariance ``Pstar``
        (Lyapunov solution), and a boolean mask indicating which state
        directions are treated as non-stationary.

        This is typically called once per MCMC chain (or once per
        calibration run), not inside the likelihood loop, because the
        unit-root structure is a model-design property that does not change
        across parameter draws.

        Args:
            param_dict: parameter name -> value.
            H_obs: (n_obs, n_obs) observation noise covariance (diagonal).
            unit_root_threshold: eigenvalue modulus threshold.  0.999
                captures exact unit roots (eigenvalue = 1.000).  Note:
                ``LZ_GAP`` appears structurally as a unit root in the .qpm
                file but has eigenvalue ~= 0.903 in the solved RE system
                (Phillips-curve and UIP feedback create strong mean
                reversion) and is correctly classified as stationary.

        Returns:
            Pinf: (n_aug, n_aug) diffuse covariance.
            Pstar: (n_aug, n_aug) stationary initial covariance.
            unit_root_mask: (n_aug,) bool mask.
        """
        sol = self.solve(param_dict)
        Pinf, Pstar, mask = build_diffuse_init(
            sol["P_aug"], sol["R_aug"],
            unit_root_threshold=unit_root_threshold,
            dtype=get_dtype())
        return Pinf, Pstar, mask

    # ---- exact diffuse log-likelihood ----
    def log_likelihood_diffuse(self, param_dict, ys, H_obs,
                               Pinf=None, Pstar=None, ur_mask=None,
                               unit_root_threshold=0.999,
                               init_a=None, lyap_iters=60,
                               n_diffuse=10):
        """Log-likelihood via the exact diffuse Kalman filter.

        Uses ``DiffuseKalmanFilter`` for models with unit-root states.
        Produces the exact diffuse log-likelihood of Durbin & Koopman
        (2012), Section 7.2, excluding contributions from diffuse
        observations.

        **Three initialization modes** (selected automatically):

        1. **MCMC mode** (``Pinf`` provided, ``ur_mask`` provided,
           ``Pstar=None``): ``Pinf`` is a pre-computed constant (the
           unit-root projector, which is parameter-independent).
           ``Pstar`` is computed at each call via the JAX-native doubling
           algorithm (``solve_lyapunov_diffuse_jax``), which is fully
           JIT-traceable and AD-compatible.  Use this mode for NUTS
           estimation.

        2. **Point-estimate mode** (``Pinf`` and ``Pstar`` provided):
           Both initialisation matrices are passed as constants.  No
           Lyapunov solve is performed.  Appropriate when parameters are
           fixed (e.g. post-estimation smoothing at the posterior mean).

        3. **Standalone mode** (all ``None``): ``Pinf`` and ``Pstar`` are
           computed internally via ``build_diffuse_init()`` (scipy-based,
           not JIT-traceable).  Appropriate for one-off likelihood
           evaluations outside MCMC.

        Args:
            param_dict: parameter name -> value.
            ys: (T, n_obs) observations.  NaN entries treated as missing.
            H_obs: (n_obs, n_obs) diagonal observation noise covariance.
            Pinf: (n_aug, n_aug) diffuse covariance (unit-root projector).
                  Pre-compute once with ``build_diffuse_matrices()`` and
                  pass here for MCMC.  If None, computed internally.
            Pstar: (n_aug, n_aug) stationary initial covariance.  If None
                  and ``ur_mask`` is provided, computed via JAX Lyapunov
                  (MCMC mode).  If None and ``ur_mask`` is None, computed
                  via scipy alongside ``Pinf``.
            ur_mask: (n_aug,) boolean array - True for unit-root states.
                  Must be provided together with ``Pinf`` for MCMC mode.
                  Pre-compute once with ``build_diffuse_matrices()``.
            unit_root_threshold: eigenvalue modulus threshold for
                  unit-root detection when Pinf/Pstar are auto-computed.
            init_a: (n_aug,) initial state mean.  Defaults to zeros.
            lyap_iters: number of doubling iterations for JAX Lyapunov
                  solver (MCMC mode).  Use ``check_lyap_convergence()``
                  to determine the appropriate value (typically 8-10).
            n_diffuse: static Python int - number of periods for the exact
                  diffuse phase in ``log_likelihood_fast``.  Must be >= the
                  true diffuse phase length (periods until Pinf ~= 0).
                  Default 10 is ample for QPM models.

        Returns:
            Scalar log-likelihood.
        """
        dt = get_dtype()
        sol = self.solve(param_dict)

        def ll_valid():
            _Pinf = Pinf
            _Pstar = Pstar

            if _Pinf is None:
                # Standalone mode: scipy-based, not JIT-traceable.
                _Pinf, _Pstar, _ = build_diffuse_init(
                    sol["P_aug"], sol["R_aug"],
                    unit_root_threshold=unit_root_threshold,
                    dtype=dt)
            elif _Pstar is None:
                # MCMC mode: Pstar via JAX doubling (JIT-traceable, AD-ok).
                # Zero out unit-root rows/cols so those directions decouple.
                _ur = jnp.asarray(ur_mask, dtype=jnp.bool_)
                stat_2d = jnp.outer(~_ur, ~_ur)   # True = stationary pair
                Q_aug = sol["R_aug"] @ sol["R_aug"].T
                T_masked = jnp.where(stat_2d, sol["P_aug"],
                                     jnp.zeros_like(sol["P_aug"]))
                Q_masked = jnp.where(stat_2d, Q_aug,
                                     jnp.zeros_like(Q_aug))
                _Pstar = solve_lyapunov_diffuse_jax(
                    T_masked, Q_masked, n_iter=lyap_iters, dtype=dt)

            _init_a = (jnp.zeros(self.n_aug, dtype=dt)
                       if init_a is None
                       else jnp.asarray(init_a, dtype=dt))

            kf = DiffuseKalmanFilter(
                T=sol["P_aug"], R=sol["R_aug"],
                C=sol["Omega"], H=H_obs,
                Pinf=_Pinf, Pstar=_Pstar,
                init_a=_init_a, c=sol["c_aug"],
                dtype=dt)

            # Use the two-segment fast path: exact diffuse for n_diffuse
            # periods, then univariate steady-state for the rest.
            # Roughly 5x fewer FLOPs than the full diffuse scan.
            ll = kf.log_likelihood_fast(ys, n_diffuse=n_diffuse)
            return jnp.where(jnp.isfinite(ll), ll,
                             jnp.array(_LARGE_NEG, dtype=dt))

        def ll_invalid():
            return jnp.array(_LARGE_NEG, dtype=dt)

        return jax.lax.cond(sol["solution_valid"], ll_valid, ll_invalid)

    # ---- exact diffuse smoother ----
    def smooth_diffuse(self, param_dict, ys, H_obs,
                       Pinf=None, Pstar=None,
                       unit_root_threshold=0.999,
                       init_a=None):
        """Smoothed states via the exact diffuse Kalman smoother.

        Returns the full Durbin-Koopman two-phase backward smoother output,
        including smoothed states, smoothed shocks, and posterior shock
        variances.

        Args:
            param_dict: parameter name -> value.
            ys: (T, n_obs) observations (NaN = missing).
            H_obs: (n_obs, n_obs) diagonal observation noise covariance.
            Pinf: (n_aug, n_aug) diffuse covariance.  None -> auto-computed.
            Pstar: (n_aug, n_aug) stationary covariance.  None -> auto-computed.
            unit_root_threshold: threshold for unit-root detection.
            init_a: (n_aug,) initial state mean.  Defaults to zeros.

        Returns:
            dict with keys:
              'alphahat': (T, n_aug) smoothed states
              'etahat':   (T, n_shock) smoothed shocks
              'V_eta':    (T, n_shock) diagonal posterior shock variance
              'loglik':   scalar log-likelihood
              'a':        (T, n_aug) predicted states a_{t|t-1}
              'd':        approximate last diffuse period
        """
        dt = get_dtype()
        sol = self.solve(param_dict)

        _Pinf, _Pstar = Pinf, Pstar
        if _Pinf is None or _Pstar is None:
            _Pinf, _Pstar, _ = build_diffuse_init(
                sol["P_aug"], sol["R_aug"],
                unit_root_threshold=unit_root_threshold,
                dtype=dt)

        _init_a = (jnp.zeros(self.n_aug, dtype=dt)
                   if init_a is None
                   else jnp.asarray(init_a, dtype=dt))

        kf = DiffuseKalmanFilter(
            T=sol["P_aug"], R=sol["R_aug"],
            C=sol["Omega"], H=H_obs,
            Pinf=_Pinf, Pstar=_Pstar,
            init_a=_init_a, c=sol["c_aug"],
            dtype=dt)

        return kf.smooth(ys)

    def check_lyap_convergence(self, param_dict, H_obs,
                                tol=1e-6, max_iter=30):
        """Find minimum lyap_iters for JAX Lyapunov solver at given params.

        Compares JAX matrix-doubling Pstar to ``scipy.linalg.solve_discrete_lyapunov``
        reference.  Prints a table of max-absolute-error vs k for
        k = 2, 4, 6, ..., and returns the smallest k where the error < ``tol``.

        Run this *once* before ``estimate()`` (outside JAX) to choose an
        appropriate ``lyap_iters``.  Typical DSGE models converge in 4-10
        doubling steps.

        Args:
            param_dict: parameter name -> value (e.g. prior modes or
                calibrated parameters).
            H_obs: (n_obs, n_obs) observation noise covariance (diagnostic
                only; needed to call build_diffuse_matrices).
            tol: convergence tolerance for max|Pstar_jax - Pstar_scipy|.
            max_iter: maximum k to try (in steps of 2).

        Returns:
            int: recommended lyap_iters.
        """
        import numpy as np
        import scipy.linalg

        dt = get_dtype()

        # Reference Pstar and masked matrices from build_diffuse_matrices
        Pinf, Pstar_ref, ur_mask = self.build_diffuse_matrices(
            param_dict, H_obs)
        Pstar_ref = np.array(Pstar_ref)

        sol = self.solve(param_dict)
        T_np = np.array(sol["P_aug"])
        R_np = np.array(sol["R_aug"])
        Q_np = R_np @ R_np.T
        stat_2d = np.outer(~ur_mask, ~ur_mask)
        T_masked = T_np * stat_2d
        Q_masked = Q_np * stat_2d

        print("  Lyapunov doubling convergence check:")
        print(f"  {'k':>4}   max|Pstar_jax - Pstar_scipy|")
        recommended = max_iter
        for k in range(2, max_iter + 1, 2):
            Pstar_k = np.array(
                solve_lyapunov_diffuse_jax(T_masked, Q_masked,
                                           n_iter=k, dtype=dt))
            err = float(np.max(np.abs(Pstar_k - Pstar_ref)))
            print(f"  {k:4d}   {err:.2e}")
            if err < tol and recommended == max_iter:
                recommended = k
        print(f"  => Recommended lyap_iters = {recommended} (tol={tol:.0e})")
        return recommended

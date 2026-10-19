"""Shock variance calibration: SNR and fixed-point methods.

Two methods for calibrating the standard deviations of structural shocks:

1. **SNR (Signal-to-Noise Ratio)** with AR(1) corrections.  Uses the
   first- and second-difference standard deviations of the observables
   together with user-specified persistence and signal-to-noise ratios
   to set InverseGamma priors for shock stds.  See :func:`calibrate_snr`.

2. **Fixed-point (Koopman disturbance smoother)** EM iteration.  Damped
   log-step update with optional bound-freeze, pinned shocks, robust
   q-aggregator, and a heteroscedastic shock-scale schedule.  See
   :func:`calibrate_fixed_point`.

Both methods can be run sequentially: SNR provides initial values for the
fixed-point iteration, and the fixed-point calibrated values are used to
build tighter priors for MCMC.

Usage:
    from qpm_toolbox.calibration import calibrate_snr, calibrate_fixed_point

    # SNR only
    sigma_priors = calibrate_snr(
        ys, obs_names, cycle_map, trend_snr_map,
        cycle_persistence=0.85, drift_persistence=0.90)

    # Full pipeline: SNR -> fixed-point -> tighter priors
    sigma_priors, calibrated_sigmas = calibrate(
        model, ys, obs_names, cycle_map, trend_snr_map,
        base_param_dict, H_obs, init_x, init_P,
        cycle_persistence=0.85, drift_persistence=0.90,
        method='both')
"""

from typing import Dict, List, Optional, Tuple, Any
import math

import numpy as np
import jax.numpy as jnp

from qpm_toolbox.kalman_filter import (
    KalmanFilter, DiffuseKalmanFilter, build_diffuse_init, get_dtype)
from qpm_toolbox.trends import compute_deterministic_path


# ======================================================================
# Prior mode computation (analytical)
# ======================================================================

def compute_prior_modes(priors: List[dict]) -> Dict[str, float]:
    """Compute the analytical mode of each prior distribution.

    Supports: beta, invgamma, halfnormal, normal, gamma.

    Args:
        priors: list of prior spec dicts with 'name', 'prior', 'args'.

    Returns:
        dict mapping parameter name to its prior mode.
    """
    modes = {}
    for p in priors:
        name = p["name"]
        kind = p["prior"].lower()
        args = p["args"]
        if kind == "beta":
            a, b = args["concentration1"], args["concentration2"]
            modes[name] = (a - 1) / (a + b - 2) if a > 1 and b > 1 else a / (a + b)
        elif kind == "invgamma":
            modes[name] = args["rate"] / (args["concentration"] + 1)
        elif kind == "halfnormal":
            modes[name] = args["scale"] * 0.7979  # sqrt(2/pi) * scale
        elif kind == "normal":
            modes[name] = args.get("loc", 0.0)
        elif kind == "gamma":
            modes[name] = max((args["concentration"] - 1) / args["rate"], 1e-6)
        else:
            modes[name] = 0.5
    return modes


# ======================================================================
# SNR calibration
# ======================================================================

def calibrate_snr(
    ys: np.ndarray,
    obs_names: List[str],
    cycle_map: Dict[str, tuple],
    trend_snr_map: Dict[str, tuple],
    *,
    cycle_persistence: float,
    drift_persistence: float,
    manual_cycle_priors: Optional[List[dict]] = None,
    alpha: float = 3.0,
    verbose: bool = True,
) -> List[dict]:
    """Calibrate shock variance priors via the SNR methodology.

    The unconditional cycle variance is anchored on first differences of
    the observable, with an analytical correction for the AR(1) cycle
    persistence ``rho`` (Harvey 1989, Sec 2.5; Sec 6.1).  Without this
    correction, the legacy ``mode = std(dy)`` choice implicitly imposes
    ``rho = 0`` on the cycle population variance and ``rho = 1/3`` on
    the cycle-shock variance, producing a 3x underestimate of cycle
    volatility at realistic macro persistences (``rho ~= 0.85``) and a
    correspondingly tight, biased trend-shock prior.

    AR(1) algebra (slow-trend approximation ``dy ~= dyhat``):

        Var(yhat)        ~= Var(dy) / [2(1-rho)]              (level cycle var)
        sigma_yhat^2    ~= 1/2(1+rho) * Var(dy)                 (cycle innovation)
        sigma_tau^2     = (1/SNR) * Var(yhat)                   (trend, level ref)
        sigma_g^2     ~= 1/2(1+rho_g) * Var(d^2y) / SNR            (trend, growth ref)

    The shock-side correction is small at typical persistences (~0.93x)
    while the level-cycle correction is large (3.3x at rho=0.85),
    propagating into the trend-shock anchor through the SNR step.

    Stage 1 - cycle priors:
        For every (obs, shock) pair in ``cycle_map``,
            mode_c = sqrt(1/2(1+rho)) * std(dy_obs) * scale.
        IG(alpha, beta_c) with beta_c = (alpha+1) * mode_c.

    Stage 2 - trend priors via SNR:
        For 'level' references:
            mode_tau = std(dy_ref) / sqrt(2(1-rho) * SNR).
        For 'growth' references:
            mode_tau = sqrt(1/2(1+rho_g)) * std(d^2y_ref) / sqrt(SNR).
        IG(alpha, beta_tau) with beta_tau = (alpha+1) * mode_tau.

    Args:
        ys:                 (T, n_obs) observation matrix (numpy).
        obs_names:          list of observable names matching ys columns.
        cycle_map:          {obs_name: (shock_name, scale_factor)}.
        trend_snr_map:      {shock_name: (ref_obs, shock_class, snr, ref_type)}
                            with ``ref_type`` in {'level', 'growth'}.
        cycle_persistence:  rho in (0, 1) - assumed AR(1) persistence of the
                            output gap / cyclical components.  Required;
                            no silent default.  Typical macro values
                            0.75-0.90; the deeper the cycle, the higher.
        drift_persistence:  rho_g in (0, 1) - assumed AR(1) persistence of
                            the trend-growth process.  Required.  For
                            smooth-trend QPMs typical values are 0.85-0.95.
        manual_cycle_priors: additional cycle priors without a dedicated
                             observable (appended directly).
        alpha:              InverseGamma concentration parameter (default 3.0).
        verbose:            print calibration tables.

    Returns:
        list of prior dicts for all ``sigma_SHK_*`` parameters.

    Raises:
        ValueError: if ``cycle_persistence`` or ``drift_persistence`` is
            outside ``(0, 1)``.
    """
    if not 0.0 < cycle_persistence < 1.0:
        raise ValueError(
            f"cycle_persistence must be in (0, 1); got {cycle_persistence!r}")
    if not 0.0 < drift_persistence < 1.0:
        raise ValueError(
            f"drift_persistence must be in (0, 1); got {drift_persistence!r}")

    # AR(1) corrections (see derivation in the docstring).
    cycle_shock_factor = np.sqrt(0.5 * (1.0 + cycle_persistence))
    drift_shock_factor = np.sqrt(0.5 * (1.0 + drift_persistence))
    cycle_var_factor   = 1.0 / (2.0 * (1.0 - cycle_persistence))

    # Stage 1: empirical first- and second-difference standard deviations.
    fd_std = {}
    fd2_std = {}
    for i, name in enumerate(obs_names):
        col = ys[:, i]
        valid = col[~np.isnan(col)]
        d1 = np.diff(valid)
        d2 = np.diff(d1)
        fd_std[name] = float(np.std(d1))
        fd2_std[name] = float(np.std(d2))

    if verbose:
        print(f"\nSNR Calibration  (rho_cycle = {cycle_persistence:.3f}, "
              f"rho_drift = {drift_persistence:.3f})")
        print(f"  AR(1) corrections:")
        print(f"    cycle-shock factor   sqrt(1/2(1+rho))    = {cycle_shock_factor:.4f}")
        print(f"    drift-shock factor   sqrt(1/2(1+rho_g))  = {drift_shock_factor:.4f}")
        print(f"    cycle-var factor     1/[2(1-rho)]      = {cycle_var_factor:.4f}")
        print()
        print("Stage 1 (cycle priors from data):")
        print(f"  {'Observable':25s}  {'std(dy)':>10s}  {'std(d2y)':>10s}  "
              f"{'Cycle shock':30s}  {'mode':>8s}  {'IG rate':>8s}")

    cycle_priors = []
    for obs_name, (shk_name, scale) in cycle_map.items():
        # sigma_{tilde y} ~= sqrt(1/2(1+rho)) * std(dy) * scale
        mode_c = fd_std[obs_name] * cycle_shock_factor * scale
        beta_c = (alpha + 1) * mode_c
        cycle_priors.append({
            "name": f"sigma_{shk_name}",
            "prior": "invgamma",
            "args": {"concentration": alpha, "rate": beta_c},
        })
        if verbose:
            suffix = f'  (scale={scale})' if scale != 1.0 else ''
            print(f"  {obs_name:25s}  {fd_std[obs_name]:10.4f}  "
                  f"{fd2_std[obs_name]:10.4f}  "
                  f"sigma_{shk_name:23s}  {mode_c:8.4f}  {beta_c:8.4f}"
                  f"{suffix}")

    if manual_cycle_priors:
        cycle_priors.extend(manual_cycle_priors)
        if verbose:
            for mp in manual_cycle_priors:
                mode_m = mp["args"]["rate"] / (mp["args"]["concentration"] + 1)
                print(f"  {'(manual)':25s}  {'':>10s}  {'':>10s}  "
                      f"{mp['name']:30s}  {mode_m:8.4f}  "
                      f"{mp['args']['rate']:8.4f}")

    if verbose:
        print(f"\nStage 2 (trend priors via SNR):")
        print(f"  {'Trend shock':30s}  {'Class':20s}  {'SNR':>6s}  "
              f"{'ref':>7s}  {'mode_ref':>10s}  {'mode_tau':>10s}  "
              f"{'IG rate':>8s}")

    trend_priors = []
    for shk_name, (ref_obs, shk_class, snr, ref_type) in trend_snr_map.items():
        if ref_type == 'level':
            # sigma_tau ~= std(dy_ref) / sqrt(2(1-rho) * SNR)
            mode_ref = fd_std[ref_obs]
            mode_tau = mode_ref * np.sqrt(cycle_var_factor / snr)
        elif ref_type == 'growth':
            # sigma_g ~= sqrt(1/2(1+rho_g)) * std(d^2y_ref) / sqrt(SNR)
            mode_ref = fd2_std[ref_obs]
            mode_tau = mode_ref * drift_shock_factor / np.sqrt(snr)
        else:
            raise ValueError(
                f"trend_snr_map[{shk_name!r}]: ref_type must be 'level' or "
                f"'growth', got {ref_type!r}")
        beta_tau = (alpha + 1) * mode_tau
        trend_priors.append({
            "name": f"sigma_{shk_name}",
            "prior": "invgamma",
            "args": {"concentration": alpha, "rate": beta_tau},
        })
        if verbose:
            print(f"  sigma_{shk_name:23s}  {shk_class:20s}  {snr:6d}  "
                  f"{ref_type:>7s}  {mode_ref:10.4f}  {mode_tau:10.6f}  "
                  f"{beta_tau:8.4f}")

    return cycle_priors + trend_priors


# ======================================================================
# Fixed-point calibration (Koopman disturbance smoother)
# ======================================================================

def expand_scale_schedule(
    schedule: Optional[Dict[str, list]],
    dates,
    shock_names: List[str],
) -> Optional[np.ndarray]:
    """Expand a per-shock, per-window scale schedule into a ``(T, n_shock)``
    array of multiplicative factors.

    Args:
        schedule: ``{shock_name: [(start, end, factor), ...]}`` mapping.
            Each window scales the named shock's std by ``factor`` over
            the inclusive period range ``[start, end]``.  Multiple
            windows per shock are allowed.  Overlap resolution is
            *last-write-wins*: later entries in the list override
            earlier ones.  Shocks not in the schedule have scale 1
            everywhere.  Returns ``None`` if ``schedule`` is None or
            empty (caller should treat as no time-variation).
        dates: ``pd.PeriodIndex`` of length ``T`` OR a list/array of
            string period labels (e.g. ``["2020Q1", ...]``).  Window
            endpoints are matched to this index by string equality.
        shock_names: full ordered list of shock names matching the model's
            shock index space.

    Returns:
        ``(T, n_shock)`` numpy array of scale factors, or ``None`` if
        schedule is empty.

    Raises:
        ValueError: window endpoint not found in ``dates`` or shock not
            in ``shock_names``.
    """
    if not schedule:
        return None
    try:
        date_list = [str(d) for d in dates]
    except TypeError:
        raise ValueError("dates must be iterable (PeriodIndex or list of strings)")
    T = len(date_list)
    n_s = len(shock_names)
    arr = np.ones((T, n_s), dtype=np.float64)
    date_idx = {d: i for i, d in enumerate(date_list)}
    name_idx = {nm: j for j, nm in enumerate(shock_names)}
    for shk_name, windows in schedule.items():
        if shk_name not in name_idx:
            raise ValueError(
                f"scale_schedule: unknown shock {shk_name!r}; "
                f"expected one of model.aug_shocks")
        j = name_idx[shk_name]
        for entry in windows:
            if len(entry) != 3:
                raise ValueError(
                    f"scale_schedule[{shk_name!r}]: each window must be "
                    f"(start, end, factor); got {entry!r}")
            start, end, factor = entry
            if str(start) not in date_idx:
                raise ValueError(
                    f"scale_schedule[{shk_name!r}]: start {start!r} not in dates")
            if str(end) not in date_idx:
                raise ValueError(
                    f"scale_schedule[{shk_name!r}]: end {end!r} not in dates")
            i0 = date_idx[str(start)]
            i1 = date_idx[str(end)]
            if i1 < i0:
                raise ValueError(
                    f"scale_schedule[{shk_name!r}]: end {end!r} precedes start {start!r}")
            arr[i0:i1 + 1, j] = float(factor)   # last-write-wins on overlap
    return arr


def _aggregate_q(qj_t: np.ndarray, kind: str,
                 kwargs: Optional[Dict[str, Any]] = None) -> float:
    """Aggregate the per-period contributions ``qj_t = eps_hat^2 + V``
    into a scalar realised second moment.

    Three flavours, all consistent at convergence (``q ~= 1`` under
    correctly-specified Gaussian shocks):

    - ``mean``: standard EM update ``mean(qj_t)``.
    - ``trimmed_mean``: drop top/bottom ``alpha`` fraction of periods,
      then mean.  Recommended for unknown anomalous periods (regime
      shifts, surprise crises) where a few outlier periods would
      otherwise inflate q.
    - ``winsorized``: clamp top/bottom ``alpha`` fractions at the
      quantile and then mean.  Less aggressive than trimming.
    """
    kw = kwargs or {}
    if kind == "mean":
        return float(np.mean(qj_t))
    if kind == "trimmed_mean":
        alpha = float(kw.get("alpha", 0.10))
        if not 0.0 <= alpha < 0.5:
            raise ValueError(f"trimmed_mean alpha must be in [0, 0.5); got {alpha}")
        if alpha == 0.0:
            return float(np.mean(qj_t))
        lo = np.quantile(qj_t, alpha)
        hi = np.quantile(qj_t, 1.0 - alpha)
        kept = qj_t[(qj_t >= lo) & (qj_t <= hi)]
        return float(np.mean(kept)) if kept.size > 0 else float(np.mean(qj_t))
    if kind == "winsorized":
        alpha = float(kw.get("alpha", 0.05))
        if not 0.0 <= alpha < 0.5:
            raise ValueError(f"winsorized alpha must be in [0, 0.5); got {alpha}")
        if alpha == 0.0:
            return float(np.mean(qj_t))
        lo = np.quantile(qj_t, alpha)
        hi = np.quantile(qj_t, 1.0 - alpha)
        return float(np.mean(np.clip(qj_t, lo, hi)))
    raise ValueError(f"unknown q_estimator: {kind!r}; "
                     f"choose one of 'mean', 'trimmed_mean', 'winsorized'")


def calibrate_fixed_point(
    model,
    base_param_dict: Dict[str, float],
    sigma_names: List[str],
    initial_sigmas: Dict[str, float],
    ys,
    H_obs,
    init_x,
    init_P,
    *,
    max_iter: int = 50,
    tol: float = 1e-4,
    damping: float = 0.5,
    sigma_bounds: Optional[Dict[str, tuple]] = None,
    verbose: bool = True,
    max_log_step: float = math.log(3.0),
    bound_freeze_steps: int = 2,
    pins_fixed: Optional[Dict[str, float]] = None,
    q_estimator: str = "mean",
    q_estimator_kwargs: Optional[Dict[str, Any]] = None,
    scale_schedule: Optional[Dict[str, list]] = None,
    dates=None,
    use_diffuse: bool = True,
) -> Dict[str, float]:
    """Fixed-point calibration of shock standard deviations.

    Iterates between running the Kalman disturbance smoother and updating
    shock stds using conditional second moments.  The state-space
    convention has ``eps_t ~ N(0, I)`` with sigma absorbed into
    ``R = Q @ diag(sigma)``, so the smoothed shock contributions
    ``eps_hat^2 + V`` are dimensionless and equal 1 in expectation at
    the truth.  At convergence ``q_j = 1`` for all calibrated j.

    At each iteration:
      1. Solve model with current sigmas.
      2. Run disturbance smoother -> eps_hat, V_diag (unit-variance space).
      3. Aggregate per-period contributions: ``q_j = aggregator(eps_hat^2 + V)``.
      4. Damped log-step update: ``log(sigma_new) - log(sigma_old) =
         (damping/2) * log(q)``, clipped to ``[-max_log_step, max_log_step]``.
      5. Clamp to ``sigma_bounds`` and check the two-step bound-freezing
         rule.

    Args:
        model:           DynareModelWithLambdified instance.
        base_param_dict: structural + fixed params (everything except sigmas).
        sigma_names:     list of 'sigma_SHK_*' names to calibrate.
        initial_sigmas:  dict {name: value} initial sigma values.
        ys:              (T, n_obs) observations (JAX array).
        H_obs:           (n_obs, n_obs) measurement noise covariance.
        init_x:          (n_aug,) initial state mean.
        init_P:          (n_aug, n_aug) initial state covariance.
        max_iter:        maximum iterations.
        tol:             convergence tolerance (max relative change in
                         active sigmas, i.e. unfrozen and unpinned).
        damping:         update step size (0 < damping <= 1).
        sigma_bounds:    dict {name: (lo, hi)} or None for defaults
                         ``(1e-8, 100.0)``.

        max_log_step:    per-iteration multiplicative step bound on sigma
                         in log-space.  Default ``log(3) ~= 1.099`` clips
                         any single-step ratio to ``[1/3, 3]``.  Set to
                         ``math.inf`` to disable (pure undamped EM step).
        bound_freeze_steps: freeze a sigma after it has been at a
                         ``sigma_bounds`` boundary for this many
                         consecutive iterations.  Default 2.  Set to 0
                         to disable freezing.
        pins_fixed:      dict {name: value} of sigmas to pin and exclude
                         from the EM update (treated as frozen from
                         iter 0).  Default None (no pins).

        q_estimator:     'mean' (default), 'trimmed_mean', or 'winsorized'.
                         See :func:`_aggregate_q`.
        q_estimator_kwargs: optional kwargs forwarded to the aggregator.
                         For 'trimmed_mean': ``{'alpha': 0.10}``.

        scale_schedule:  optional ``{shock_name: [(start, end, factor), ...]}``
                         deterministic per-window scale on the shock std.
                         Expanded to a ``(T, n_shock)`` array via
                         :func:`expand_scale_schedule` and threaded into
                         the smoother as ``shock_scale_t``.  Honoured by
                         both the forward filter (Q_t) and the disturbance
                         smoother (R_t).  Falls back to the model's
                         ``shock_scale_schedule`` attribute if set and
                         this arg is ``None``.
        dates:           required when ``scale_schedule`` is provided;
                         a ``pd.PeriodIndex`` or list of strings of length
                         T matching ``ys`` rows.

        use_diffuse:     if ``True`` (default), use the exact diffuse
                         Kalman smoother (Durbin & Koopman 2012, ch. 5)
                         for the EM step.  Required for unbiased trend-
                         shock recovery on models with unit-root states:
                         a non-diffuse prior on integrated trends causes
                         initial-state mismatch to leak into early-period
                         smoothed shocks, biasing trend-shock second
                         moments upward and preventing convergence.  When
                         ``False``, falls back to the standard Kalman
                         smoother with the supplied ``init_P`` (kept as
                         an alternative for stationary models or when a
                         ``scale_schedule`` is required, since the
                         diffuse path does not yet thread time-varying
                         shock scales).  Mirrors the Dynare/IRIS default
                         of using diffuse initialisation for unit-root
                         models.

        verbose:         print iteration progress.

    Returns:
        dict {name: calibrated_sigma}.  Pinned entries appear with their
        pinned values.
    """
    dt = get_dtype()
    sigmas = {k: float(v) for k, v in initial_sigmas.items()}

    if sigma_bounds is None:
        sigma_bounds = {}
    pins_fixed = pins_fixed or {}

    # Resolve scale_schedule -> (T, n_shock) shock_scale_t array.
    # Caller-supplied takes precedence; fall back to model attribute.
    effective_schedule = scale_schedule
    if effective_schedule is None:
        effective_schedule = getattr(model, "shock_scale_schedule", None)
    shock_scale_t = None
    if effective_schedule:
        if dates is None:
            raise ValueError(
                "calibrate_fixed_point: scale_schedule is set but dates=None; "
                "pass a pd.PeriodIndex or list of strings of length ys.shape[0]")
        shock_scale_t = expand_scale_schedule(
            effective_schedule, dates, list(model.aug_shocks))

    # Diffuse path does not yet thread time-varying R; surface the
    # incompatibility before entering the loop so the message is clear.
    if use_diffuse and shock_scale_t is not None:
        raise NotImplementedError(
            "calibrate_fixed_point: use_diffuse=True is incompatible with "
            "scale_schedule (time-varying shock scaling).  Pass "
            "use_diffuse=False to fall back to the standard Kalman "
            "smoother, or remove the schedule.  TODO: thread time-varying "
            "R through DiffuseKalmanFilter._forward / _backward.")

    # Apply pins before the loop; they're frozen from iter 0.
    for pname, pval in pins_fixed.items():
        if pname not in sigmas:
            raise ValueError(
                f"pins_fixed[{pname!r}]: not in initial_sigmas / sigma_names")
        sigmas[pname] = float(pval)

    # Per-shock state vectors keyed by ``sigma_names`` index.
    n = len(sigma_names)
    frozen = np.zeros(n, dtype=bool)
    pinned = np.zeros(n, dtype=bool)
    at_bound_history: List[np.ndarray] = []  # rolling window of at-bound flags

    for j, name in enumerate(sigma_names):
        if name in pins_fixed:
            pinned[j] = True
            frozen[j] = True

    if verbose:
        print(f"\nFixed-point calibration (max_iter={max_iter}, "
              f"damping={damping}, tol={tol}, "
              f"max_log_step={max_log_step:.3f}, "
              f"bound_freeze_steps={bound_freeze_steps}, "
              f"q_estimator={q_estimator!r})")
        print(f"  {'Shock':30s}  {'init sigma':>10s}  "
              f"{'bounds':>20s}  {'note':>10s}")
        for name in sigma_names:
            lo, hi = sigma_bounds.get(name, (1e-8, 100.0))
            note = "(pinned)" if name in pins_fixed else ""
            print(f"  {name:30s}  {sigmas[name]:10.6f}  "
                  f"[{lo:.2e}, {hi:.2e}]  {note:>10s}")

    shock_names = model.aug_shocks
    # Map sigma_names index -> shock_names index for q lookup.
    shk_to_jq = {f"sigma_{shk}": j for j, shk in enumerate(shock_names)}
    q_idx = np.array([shk_to_jq.get(name, -1) for name in sigma_names], dtype=int)

    # ------------------------------------------------------------------
    # Deterministic detrending for the calibration's steady-state filter.
    # P_aug and c_aug depend only on structural parameters (held at modes
    # during calibration), not on sigmas, so we compute the deterministic
    # path once before the EM loop.  Subtracting it from ys gives a
    # deviation-form input on which the time-varying Riccati recursion
    # remains numerically well-behaved on unit-root systems - see
    # qpm_toolbox.trends for the derivation.
    # ------------------------------------------------------------------
    sol_init = model.solve({**base_param_dict, **sigmas})
    if not bool(sol_init["solution_valid"]):
        raise RuntimeError(
            "calibrate_fixed_point: model fails to solve at initial sigmas; "
            "cannot build the deterministic detrending path")
    _, y_det = compute_deterministic_path(
        sol_init["P_aug"], sol_init.get("c_aug"), init_x,
        sol_init["Omega"], T_steps=int(np.asarray(ys).shape[0]))
    ys_dev = jnp.asarray(ys, dtype=dt) - y_det
    init_x_zero = jnp.zeros_like(jnp.asarray(init_x, dtype=dt))

    max_rel_change = 0.0  # ensure defined for the no-iter case
    for it in range(max_iter):
        pdict = {**base_param_dict, **sigmas}

        sol = model.solve(pdict)
        if not sol["solution_valid"]:
            if verbose:
                print(f"  iter {it}: model does not solve!")
            break

        P_aug, R_aug, Omega = sol["P_aug"], sol["R_aug"], sol["Omega"]

        # Deviation-form filter: data is already detrended, so c=0 and
        # init_x=0 keep the filter's predicted mean at zero.  The
        # disturbance smoother's q diagnostic is invariant to this
        # shift, so the EM update is mathematically unchanged.
        if use_diffuse:
            # Pinf is determined by P_aug's eigenstructure (parameter-
            # independent) and Pstar by the Lyapunov solution on the
            # stationary block (depends on R_aug, so rebuilt per iter).
            # build_diffuse_init runs in numpy/scipy and is fast.
            Pinf, Pstar, _ = build_diffuse_init(P_aug, R_aug, dtype=dt)
            kf = DiffuseKalmanFilter(
                T=P_aug, R=R_aug, C=Omega, H=H_obs,
                Pinf=Pinf, Pstar=Pstar,
                init_a=init_x_zero, c=None, dtype=dt)
        else:
            kf = KalmanFilter(
                T=P_aug, R=R_aug, C=Omega, H=H_obs,
                init_x=init_x_zero, init_P=init_P,
                c=None, dtype=dt)
        eps_hat, V_diag = kf.smooth_disturbances(
            ys=ys_dev, shock_scale_t=shock_scale_t)

        eps_hat_np = np.asarray(eps_hat)
        V_diag_np = np.asarray(V_diag)
        qj_t_all = eps_hat_np**2 + V_diag_np   # (T, n_shock)

        old_sigmas = dict(sigmas)
        new_sigmas = dict(sigmas)
        at_bound = np.zeros(n, dtype=bool)
        for i, name in enumerate(sigma_names):
            if frozen[i]:
                continue
            jq = q_idx[i]
            if jq < 0:
                continue   # sigma_name not in shock_names - leave alone
            q_scalar = _aggregate_q(qj_t_all[:, jq], q_estimator,
                                    q_estimator_kwargs)
            # Damped log-step update with clip:
            #   log(sigma_new) = log(sigma_old) + clip( (damping/2) * log(q), +/-max_log_step )
            if q_scalar <= 0.0:
                step = -max_log_step    # collapse if q vanishes
            else:
                step = (damping / 2.0) * math.log(q_scalar)
            step = max(-max_log_step, min(max_log_step, step))
            new_val = sigmas[name] * math.exp(step)
            lo, hi = sigma_bounds.get(name, (1e-8, 100.0))
            new_val = max(lo, min(hi, new_val))
            new_sigmas[name] = new_val
            at_bound[i] = (new_val <= lo + 1e-10) or (new_val >= hi - 1e-10)
        sigmas = new_sigmas

        # Bound-freeze: mark frozen if at a bound for ``bound_freeze_steps``
        # consecutive iterations.
        if bound_freeze_steps > 0:
            at_bound_history.append(at_bound.copy())
            if len(at_bound_history) > bound_freeze_steps:
                at_bound_history.pop(0)
            if len(at_bound_history) >= bound_freeze_steps:
                consec = np.all(at_bound_history, axis=0)
                newly_frozen = consec & ~frozen
                if newly_frozen.any():
                    if verbose:
                        for i in np.where(newly_frozen)[0]:
                            print(f"  iter {it}: freezing {sigma_names[i]} "
                                  f"at {sigmas[sigma_names[i]]:.4g} "
                                  f"(at bound {bound_freeze_steps} consecutive iters)")
                    frozen = frozen | newly_frozen

        # Convergence test: max relative change over ACTIVE entries
        # (non-frozen, non-pinned).
        max_rel_change = 0.0
        for i, name in enumerate(sigma_names):
            if frozen[i] or pinned[i]:
                continue
            if old_sigmas[name] > 1e-12:
                rel = abs(sigmas[name] - old_sigmas[name]) / old_sigmas[name]
                max_rel_change = max(max_rel_change, rel)

        if verbose and (it < 5 or it % 5 == 0 or max_rel_change < tol):
            n_active = int(np.sum(~frozen))
            print(f"  iter {it:3d}:  max_rel_delta={max_rel_change:.6f}  "
                  f"(active={n_active}/{n}  frozen={int(np.sum(frozen))}  "
                  f"pinned={int(np.sum(pinned))})")

        if max_rel_change < tol:
            if verbose:
                print(f"  Converged at iteration {it}")
            break
    else:
        if verbose:
            print(f"  Did not converge in {max_iter} iterations "
                  f"(last max_rel_delta={max_rel_change:.6f})")

    if verbose:
        print(f"\n  Fixed-point calibrated sigmas:")
        print(f"  {'Shock':30s}  {'init':>10s}  {'final':>10s}  "
              f"{'ratio':>8s}  {'note':>10s}")
        for i, name in enumerate(sigma_names):
            ini = initial_sigmas[name]
            fin = sigmas[name]
            ratio = fin / ini if ini != 0 else float('nan')
            note = "(pinned)" if pinned[i] else ("(frozen)" if frozen[i] else "")
            print(f"  {name:30s}  {ini:10.6f}  {fin:10.6f}  "
                  f"{ratio:8.4f}  {note:>10s}")

    return sigmas


# ======================================================================
# Combined calibration pipeline
# ======================================================================

def calibrate(
    model,
    ys: np.ndarray,
    obs_names: List[str],
    cycle_map: Dict[str, tuple],
    trend_snr_map: Dict[str, tuple],
    base_param_dict: Dict[str, float],
    H_obs,
    init_x,
    init_P,
    *,
    cycle_persistence: float,
    drift_persistence: float,
    method: str = 'both',
    manual_cycle_priors: Optional[List[dict]] = None,
    alpha: float = 3.0,
    fixed_point_config: Optional[dict] = None,
    fp_prior_alpha: float = 5.0,
    verbose: bool = True,
    pins_fixed: Optional[Dict[str, float]] = None,
    pin_trends_from_snr: bool = True,
    q_estimator: str = "mean",
    q_estimator_kwargs: Optional[Dict[str, Any]] = None,
    scale_schedule: Optional[Dict[str, list]] = None,
    dates=None,
    use_diffuse: bool = True,
) -> Tuple[List[dict], Dict[str, float]]:
    """Combined SNR + fixed-point calibration pipeline.

    When method='both' (default):
      1. Run SNR calibration -> initial sigma priors for cycles + trends
      2. Extract sigma modes as initial values
      3. Pin trend sigmas at their SNR-derived modes (auto;
         ``pin_trends_from_snr=True`` by default - see below)
      4. Run fixed-point calibration on the remaining (cycle) shocks
      5. Build tighter priors (alpha=fp_prior_alpha) centered on
         calibrated values

    Args:
        model:             DynareModelWithLambdified instance.
        ys:                (T, n_obs) observations (numpy).
        obs_names:         list of observable names.
        cycle_map:         SNR cycle shock mapping.
        trend_snr_map:     SNR trend shock mapping.
        base_param_dict:   structural + fixed params (no sigmas).
        H_obs:             measurement noise covariance.
        init_x:            initial state mean.
        init_P:            initial state covariance.
        cycle_persistence: rho for cycle AR(1) - required.
        drift_persistence: rho_g for drift AR(1) - required.
        method:            'snr', 'fixed_point', or 'both'.
        manual_cycle_priors: manual cycle prior specs.
        alpha:             InverseGamma alpha for SNR priors.
        fixed_point_config: dict with keys 'max_iter', 'tol', 'damping',
                            'sigma_bounds', 'max_log_step',
                            'bound_freeze_steps'.
        fp_prior_alpha:    InverseGamma alpha for post-FP priors (tighter).
        pins_fixed:        forwarded to :func:`calibrate_fixed_point`.
                           Caller-supplied pins override the auto-built
                           trend pins (see ``pin_trends_from_snr``).
        pin_trends_from_snr: if ``True`` (default), pin every shock in
                           ``trend_snr_map`` at its SNR-derived mode and
                           leave only cycle shocks active in the
                           fixed-point EM.  This matches the workflow
                           used in the MATLAB/Dynare reference
                           implementations
                           (``gpm7_estimation/code/run_calibration.m``):
                           trends are anchored by economically
                           motivated signal-to-noise ratios while the
                           data informs only the cycle stds.  Set to
                           ``False`` to recover the legacy behaviour
                           where the EM updates every shock std
                           simultaneously (typically produces large
                           drifts in trend sigmas because the EM has
                           no economic anchor for them).
        q_estimator:       forwarded to :func:`calibrate_fixed_point`.
        q_estimator_kwargs: forwarded to :func:`calibrate_fixed_point`.
        scale_schedule:    forwarded to :func:`calibrate_fixed_point`.
        dates:             required when ``scale_schedule`` is set.
        use_diffuse:       forwarded to :func:`calibrate_fixed_point`.
                           Default ``True`` (exact diffuse smoother for
                           the EM step - required for unbiased trend-
                           shock recovery on unit-root models).
        verbose:           print progress.

    Returns:
        (sigma_priors, calibrated_sigmas):
          - sigma_priors: list of prior dicts for MCMC
          - calibrated_sigmas: dict of {sigma_name: calibrated_value}
    """
    if fixed_point_config is None:
        fixed_point_config = {}

    # Step 1: SNR calibration
    sigma_priors = calibrate_snr(
        ys, obs_names, cycle_map, trend_snr_map,
        cycle_persistence=cycle_persistence,
        drift_persistence=drift_persistence,
        manual_cycle_priors=manual_cycle_priors,
        alpha=alpha, verbose=verbose)

    # Extract initial sigmas from SNR modes
    snr_sigmas = {}
    for p in sigma_priors:
        mode = p["args"]["rate"] / (p["args"]["concentration"] + 1)
        snr_sigmas[p["name"]] = mode

    if method == 'snr':
        return sigma_priors, snr_sigmas

    # Step 2: Auto-pin trend shocks at their SNR-derived modes so the
    # EM only updates the cycle shocks.  Mirrors the MATLAB workflow in
    # gpm7_estimation/code/run_calibration.m where ``pins`` is built
    # from the trend SNR map and ``em_shocks`` is the set difference.
    # Caller-supplied ``pins_fixed`` always wins on conflicts.
    effective_pins: Dict[str, float] = {}
    if pin_trends_from_snr and trend_snr_map:
        for trend_shock_name in trend_snr_map.keys():
            sigma_name = f"sigma_{trend_shock_name}"
            if sigma_name in snr_sigmas:
                effective_pins[sigma_name] = snr_sigmas[sigma_name]
    if pins_fixed:
        effective_pins.update(pins_fixed)   # caller pins override
    if verbose and effective_pins:
        n_auto = (len(effective_pins) -
                  len(pins_fixed) if pins_fixed else len(effective_pins))
        print(f"\n  Pinning {len(effective_pins)} trend shock std(s) "
              f"at SNR modes ({n_auto} auto-built from trend_snr_map, "
              f"{len(pins_fixed) if pins_fixed else 0} caller-supplied):")
        for name, val in effective_pins.items():
            print(f"    {name:30s}  pinned at {val:.6f}")

    # Step 3: Fixed-point calibration on the unpinned (cycle) shocks
    sigma_names = list(snr_sigmas.keys())
    calibrated = calibrate_fixed_point(
        model=model,
        base_param_dict=base_param_dict,
        sigma_names=sigma_names,
        initial_sigmas=snr_sigmas,
        ys=ys,
        H_obs=H_obs,
        init_x=init_x,
        init_P=init_P,
        max_iter=fixed_point_config.get('max_iter', 50),
        tol=fixed_point_config.get('tol', 1e-4),
        damping=fixed_point_config.get('damping', 0.5),
        sigma_bounds=fixed_point_config.get('sigma_bounds'),
        max_log_step=fixed_point_config.get('max_log_step', math.log(3.0)),
        bound_freeze_steps=fixed_point_config.get('bound_freeze_steps', 2),
        pins_fixed=effective_pins if effective_pins else None,
        q_estimator=q_estimator,
        q_estimator_kwargs=q_estimator_kwargs,
        scale_schedule=scale_schedule,
        dates=dates,
        use_diffuse=use_diffuse,
        verbose=verbose,
    )

    if method == 'fixed_point':
        # Build priors from fixed-point values but keep SNR alpha
        fp_priors = []
        for name in sigma_names:
            mode_fp = calibrated[name]
            beta_fp = (alpha + 1) * mode_fp
            fp_priors.append({
                "name": name,
                "prior": "invgamma",
                "args": {"concentration": alpha, "rate": beta_fp},
            })
        return fp_priors, calibrated

    # method == 'both': build tighter priors from FP values
    fp_priors = []
    for name in sigma_names:
        mode_fp = calibrated[name]
        beta_fp = (fp_prior_alpha + 1) * mode_fp
        fp_priors.append({
            "name": name,
            "prior": "invgamma",
            "args": {"concentration": fp_prior_alpha, "rate": beta_fp},
        })

    if verbose:
        print(f"\n  Updated priors (alpha={fp_prior_alpha}, "
              f"centered on FP values):")
        for p in fp_priors:
            mode = p["args"]["rate"] / (p["args"]["concentration"] + 1)
            print(f"    {p['name']:30s}  mode={mode:.6f}")

    return fp_priors, calibrated

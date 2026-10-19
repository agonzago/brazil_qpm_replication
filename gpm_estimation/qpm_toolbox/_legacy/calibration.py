"""Shock variance calibration: SNR and fixed-point methods.

Two methods for calibrating the standard deviations of structural shocks:

1. **SNR (Signal-to-Noise Ratio)**: Uses data volatility and user-specified
   SNR ratios to set InverseGamma priors for shock stds.

2. **Fixed-point (Koopman disturbance smoother)**: Iterates until the
   assumed shock variances equal the data-revealed conditional second
   moments. At convergence q_j = 1 for all shocks j.

Both methods can be run sequentially: SNR provides initial values for the
fixed-point iteration, and the fixed-point calibrated values are used to
build tighter priors for MCMC.

Usage:
    from qpm_toolbox.calibration import calibrate_snr, calibrate_fixed_point

    # SNR only
    sigma_priors = calibrate_snr(ys, obs_names, cycle_map, trend_snr_map)

    # Full pipeline: SNR -> fixed-point -> tighter priors
    sigma_priors, calibrated_sigmas = calibrate(
        model, ys, obs_names, cycle_map, trend_snr_map,
        base_param_dict, H_obs, init_x, init_P,
        method='both')
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from qpm_toolbox.kalman_filter import KalmanFilter, get_dtype


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
    manual_cycle_priors: Optional[List[dict]] = None,
    alpha: float = 3.0,
    verbose: bool = True,
) -> List[dict]:
    """Calibrate shock variance priors using the SNR methodology.

    Stage 1: Compute std(delta y) for each observable -> cycle shock mode.
             Also compute std(delta^2 y) for growth-rate trend references.
    Stage 2: Derive trend shock modes via mode_tau = mode_ref / sqrt(SNR),
             using the appropriate reference (level or growth).

    Args:
        ys:             (T, n_obs) observation matrix (numpy).
        obs_names:      list of observable names matching ys columns.
        cycle_map:      {obs_name: (shock_name, scale_factor)} for cycle shocks.
        trend_snr_map:  {shock_name: (ref_obs, shock_class, snr, ref_type)}
                        ref_type is 'level' or 'growth'.
        manual_cycle_priors: additional cycle priors without a dedicated
                             observable (appended directly).
        alpha:          InverseGamma concentration parameter (default 3.0).
        verbose:        print calibration tables.

    Returns:
        list of prior dicts for all sigma_SHK_* parameters.
    """
    # Stage 1: first-difference and second-difference std
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
        print("\nSNR Calibration - Stage 1 (cycle priors from data):")
        print(f"  {'Observable':25s}  {'std(dy)':>10s}  {'std(d2y)':>10s}  "
              f"{'Cycle shock':30s}  {'mode':>8s}  {'IG rate':>8s}")

    cycle_priors = []
    for obs_name, (shk_name, scale) in cycle_map.items():
        mode_c = fd_std[obs_name] * scale
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

    # Add manual cycle priors (shocks without a dedicated observable)
    if manual_cycle_priors:
        cycle_priors.extend(manual_cycle_priors)
        if verbose:
            for mp in manual_cycle_priors:
                mode_m = mp["args"]["rate"] / (mp["args"]["concentration"] + 1)
                print(f"  {'(manual)':25s}  {'':>10s}  {'':>10s}  "
                      f"{mp['name']:30s}  {mode_m:8.4f}  "
                      f"{mp['args']['rate']:8.4f}")

    # Stage 2: trend shock priors via SNR
    if verbose:
        print(f"\nSNR Calibration - Stage 2 (trend priors via SNR):")
        print(f"  {'Trend shock':30s}  {'Class':20s}  {'SNR':>6s}  "
              f"{'ref':>7s}  {'mode_ref':>8s}  {'mode_tau':>10s}  "
              f"{'IG rate':>8s}")

    trend_priors = []
    for shk_name, (ref_obs, shk_class, snr, ref_type) in trend_snr_map.items():
        if ref_type == 'growth':
            mode_ref = fd2_std[ref_obs]
        else:
            mode_ref = fd_std[ref_obs]

        mode_tau = mode_ref / np.sqrt(snr)
        beta_tau = (alpha + 1) * mode_tau
        trend_priors.append({
            "name": f"sigma_{shk_name}",
            "prior": "invgamma",
            "args": {"concentration": alpha, "rate": beta_tau},
        })
        if verbose:
            print(f"  sigma_{shk_name:23s}  {shk_class:20s}  {snr:6d}  "
                  f"{ref_type:>7s}  {mode_ref:8.4f}  {mode_tau:10.6f}  "
                  f"{beta_tau:8.4f}")

    return cycle_priors + trend_priors


# ======================================================================
# Fixed-point calibration (Koopman disturbance smoother)
# ======================================================================

def calibrate_fixed_point(
    model,
    base_param_dict: Dict[str, float],
    sigma_names: List[str],
    initial_sigmas: Dict[str, float],
    ys,
    H_obs,
    init_x,
    init_P,
    max_iter: int = 50,
    tol: float = 1e-4,
    damping: float = 0.5,
    sigma_bounds: Optional[Dict[str, tuple]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Fixed-point calibration of shock standard deviations.

    Iterates between running the Kalman disturbance smoother and updating
    shock stds using conditional second moments.

    At each iteration:
      1. Solve model with current sigmas
      2. Run disturbance smoother -> eps_hat, V_diag
      3. Compute q_j = mean_t(eps_hat_{j,t}^2 + V_{j,t})
      4. Update sigma_j = sigma_j * q_j^(damping/2)
      5. Apply bounds

    At convergence, q_j = 1 for all j (assumed = revealed variance).

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
        tol:             convergence tolerance (max relative change).
        damping:         update step size (0 < damping <= 1).
        sigma_bounds:    dict {name: (lo, hi)} or None for defaults.
        verbose:         print iteration progress.

    Returns:
        dict {name: calibrated_sigma}.
    """
    dt = get_dtype()
    sigmas = {k: float(v) for k, v in initial_sigmas.items()}

    if sigma_bounds is None:
        sigma_bounds = {}

    if verbose:
        print(f"\nFixed-point calibration (max_iter={max_iter}, "
              f"damping={damping}, tol={tol})")
        print(f"  {'Shock':30s}  {'init sigma':>10s}  "
              f"{'bounds':>20s}")
        for name in sigma_names:
            lo, hi = sigma_bounds.get(name, (1e-8, 100.0))
            print(f"  {name:30s}  {sigmas[name]:10.6f}  "
                  f"[{lo:.2e}, {hi:.2e}]")

    shock_names = model.aug_shocks

    for it in range(max_iter):
        pdict = {**base_param_dict, **sigmas}

        sol = model.solve(pdict)
        if not sol["solution_valid"]:
            print(f"  iter {it}: model does not solve!")
            break

        P_aug, R_aug, Omega = sol["P_aug"], sol["R_aug"], sol["Omega"]

        kf = KalmanFilter(T=P_aug, R=R_aug, C=Omega, H=H_obs,
                          init_x=init_x, init_P=init_P, dtype=dt)
        eps_hat, V_diag = kf.smooth_disturbances(ys=ys)

        eps_hat_np = np.asarray(eps_hat)
        V_diag_np = np.asarray(V_diag)
        q = np.mean(eps_hat_np**2 + V_diag_np, axis=0)

        old_sigmas = dict(sigmas)
        for j, shk in enumerate(shock_names):
            sname = f"sigma_{shk}"
            if sname in sigmas:
                new_val = sigmas[sname] * float(q[j]) ** (damping / 2.0)
                lo, hi = sigma_bounds.get(sname, (1e-8, 100.0))
                sigmas[sname] = max(lo, min(hi, new_val))

        max_rel_change = 0.0
        for name in sigma_names:
            if old_sigmas[name] > 1e-12:
                rel = abs(sigmas[name] - old_sigmas[name]) / old_sigmas[name]
                max_rel_change = max(max_rel_change, rel)

        if verbose and (it < 5 or it % 5 == 0 or max_rel_change < tol):
            print(f"  iter {it:3d}:  max_rel_delta={max_rel_change:.6f}")

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
              f"{'ratio':>8s}")
        for name in sigma_names:
            ini = initial_sigmas[name]
            fin = sigmas[name]
            print(f"  {name:30s}  {ini:10.6f}  {fin:10.6f}  "
                  f"{fin/ini:8.4f}")

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
    method: str = 'both',
    manual_cycle_priors: Optional[List[dict]] = None,
    alpha: float = 3.0,
    fixed_point_config: Optional[dict] = None,
    fp_prior_alpha: float = 5.0,
    verbose: bool = True,
) -> Tuple[List[dict], Dict[str, float]]:
    """Combined SNR + fixed-point calibration pipeline.

    When method='both' (default):
      1. Run SNR calibration -> initial sigma priors
      2. Extract sigma modes as initial values
      3. Run fixed-point calibration starting from SNR modes
      4. Build tighter priors (alpha=fp_prior_alpha) centered on
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
        method:            'snr', 'fixed_point', or 'both'.
        manual_cycle_priors: manual cycle prior specs.
        alpha:             InverseGamma alpha for SNR priors.
        fixed_point_config: dict with keys 'max_iter', 'tol', 'damping',
                            'sigma_bounds'.
        fp_prior_alpha:    InverseGamma alpha for post-FP priors (tighter).
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
        manual_cycle_priors=manual_cycle_priors,
        alpha=alpha, verbose=verbose)

    # Extract initial sigmas from SNR modes
    snr_sigmas = {}
    for p in sigma_priors:
        mode = p["args"]["rate"] / (p["args"]["concentration"] + 1)
        snr_sigmas[p["name"]] = mode

    if method == 'snr':
        return sigma_priors, snr_sigmas

    # Step 2: Fixed-point calibration
    sigma_names = list(snr_sigmas.keys())
    calibrated = calibrate_fixed_point(
        model=model,
        base_param_dict=base_param_dict,
        sigma_names=sigma_names,
        initial_sigmas=snr_sigmas,
        ys=ys if hasattr(ys, 'dtype') and str(ys.dtype).startswith('float')
        else ys,
        H_obs=H_obs,
        init_x=init_x,
        init_P=init_P,
        max_iter=fixed_point_config.get('max_iter', 50),
        tol=fixed_point_config.get('tol', 1e-4),
        damping=fixed_point_config.get('damping', 0.5),
        sigma_bounds=fixed_point_config.get('sigma_bounds'),
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

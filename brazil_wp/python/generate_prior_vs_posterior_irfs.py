"""Prior vs Posterior IRFs for the monetary policy shock.

Samples parameter vectors from the prior, solves the model for each,
computes IRFs, and overlays prior uncertainty bands with posterior
uncertainty bands.  This shows how much the data narrows the structural
uncertainty relative to the prior.

Usage:
    python generate_prior_vs_posterior_irfs.py
    python generate_prior_vs_posterior_irfs.py --n-prior 2000
"""

import os
import sys
import argparse

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX_DIR = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX_DIR)

from qpm_toolbox import QPMModel
from qpm_toolbox.calibration import compute_prior_modes

from _runner_loader import MODEL_LABELS, load_runner, results_dir as _results_dir

# Prior distribution samplers (NumPy)
_PRIOR_SAMPLERS = {
    'beta': lambda rng, args: rng.beta(
        args['concentration1'], args['concentration2']),
    'gamma': lambda rng, args: rng.gamma(
        args['concentration']) / args['rate'],
    'halfnormal': lambda rng, args: abs(
        rng.normal(0, args['scale'])),
    'invgamma': lambda rng, args: 1.0 / rng.gamma(
        args['concentration'], 1.0 / args['rate']),
}


def sample_from_prior(rng, structural_priors, sigma_priors):
    """Draw one parameter vector from the joint prior."""
    params = {}
    for spec in structural_priors:
        name = spec['name']
        prior_type = spec['prior'].lower()
        params[name] = float(_PRIOR_SAMPLERS[prior_type](rng, spec['args']))
    for spec in sigma_priors:
        name = spec['name']
        prior_type = spec['prior'].lower()
        params[name] = float(_PRIOR_SAMPLERS[prior_type](rng, spec['args']))
    return params


def compute_irfs_for_params(model, pdict, shocks, variables, horizon):
    """Solve model and compute IRFs for one parameter set."""
    sol = model.solve(pdict)
    if not sol['solution_valid']:
        return None

    P_aug = np.asarray(sol['P_aug'])
    R_aug = np.asarray(sol['R_aug'])
    Omega = np.asarray(sol['Omega'])

    all_shocks = model.aug_shocks
    all_vars = model.aug_state_vars
    all_obs = model.ordered_obs_vars

    results = {}
    for shk in shocks:
        shk_idx = all_shocks.index(shk)
        # State IRF
        state_irf = np.zeros((horizon, P_aug.shape[0]))
        impulse = R_aug[:, shk_idx]
        x = impulse.copy()
        for t in range(horizon):
            state_irf[t] = x
            x = P_aug @ x
        # Observable IRF
        obs_irf = state_irf @ Omega.T

        irf_vals = np.zeros((horizon, len(variables)))
        for k, v in enumerate(variables):
            if v in all_vars:
                idx = all_vars.index(v)
                irf_vals[:, k] = state_irf[:, idx]
            elif v in all_obs:
                idx = all_obs.index(v)
                irf_vals[:, k] = obs_irf[:, idx]
        results[shk] = irf_vals
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='A', choices=MODEL_LABELS,
                        help='Model label (default: A)')
    parser.add_argument('--n-prior', type=int, default=1000,
                        help='Number of prior draws (default: 1000)')
    parser.add_argument('--n-posterior', type=int, default=200,
                        help='Posterior draws for IRFs (default: 200)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--results-dir', type=str, default=None)
    args = parser.parse_args()

    runner = load_runner(args.model)
    MOD_FILE = runner.MOD_FILE
    STRUCTURAL_PRIORS = runner.STRUCTURAL_PRIORS
    FIXED_PARAMS = runner.FIXED_PARAMS
    MEAS_NOISE_STD = runner.MEAS_NOISE_STD
    CYCLE_MAP = runner.CYCLE_MAP
    TREND_SNR_MAP = runner.TREND_SNR_MAP
    MANUAL_CYCLE_PRIORS = runner.MANUAL_CYCLE_PRIORS

    results_dir = args.results_dir or _results_dir(args.model)
    out_dir = os.path.join(results_dir, 'irf')
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load model ----
    print(f"Loading Model {args.model} from {results_dir}...")
    ys, dates, obs_names = runner.load_brazil_data()
    m = QPMModel(MOD_FILE, results_dir=results_dir)
    ss_params = m.model.default_param_assignments
    init_config = runner.compute_initial_conditions(ys, obs_names, ss_params)
    m.set_data(ys, dates=dates, obs_names=obs_names)
    m.set_priors(
        structural_priors=STRUCTURAL_PRIORS,
        fixed_params=FIXED_PARAMS,
        meas_noise_std=MEAS_NOISE_STD,
        cycle_map=CYCLE_MAP,
        trend_snr_map=TREND_SNR_MAP,
        manual_cycle_priors=MANUAL_CYCLE_PRIORS,
    )
    m.set_initial_conditions(init_state_config=init_config)
    m.calibrate(method='both', cycle_persistence=0.85, drift_persistence=0.90)

    # ---- Config ----
    shock_name = 'SHK_RS'
    variables = ['GDP_GAP', 'DA_LCPIX', 'RS', 'LZ_GAP']
    var_labels = ['Output Gap (pp)', 'Core Inflation (pp ann.)',
                  'Policy Rate (pp ann.)', 'REER Gap (pp)']
    horizon = 30

    # ---- Prior IRFs ----
    print(f"Sampling {args.n_prior} prior draws...")
    rng = np.random.default_rng(args.seed)
    sigma_priors = m._priors  # includes sigma priors from calibration
    # Separate structural from sigma priors
    structural_names = {s['name'] for s in STRUCTURAL_PRIORS}
    sigma_only = [p for p in sigma_priors if p['name'] not in structural_names]

    prior_irfs = []
    n_invalid = 0
    for i in range(args.n_prior):
        pdict = dict(m.model.default_param_assignments)
        pdict.update(FIXED_PARAMS)
        prior_draw = sample_from_prior(rng, STRUCTURAL_PRIORS, sigma_only)
        pdict.update(prior_draw)

        result = compute_irfs_for_params(
            m.model, pdict, [shock_name], variables, horizon)
        if result is None:
            n_invalid += 1
            continue
        prior_irfs.append(result[shock_name])

    prior_irfs = np.stack(prior_irfs)  # (N_valid, horizon, n_vars)
    print(f"  Valid prior draws: {len(prior_irfs)}/{args.n_prior} "
          f"({n_invalid} invalid)")

    # ---- Posterior IRFs ----
    print(f"Computing posterior IRFs ({args.n_posterior} draws)...")
    m.load_draws()
    post_samples = m._posterior_samples
    total = post_samples[next(iter(post_samples))].shape[0]
    step = max(1, total // args.n_posterior)
    indices = list(range(0, total, step))[:args.n_posterior]

    post_irfs = []
    for idx in indices:
        pdict = dict(m.model.default_param_assignments)
        pdict.update(FIXED_PARAMS)
        if m._calibrated_sigmas:
            pdict.update(m._calibrated_sigmas)
        for name in m._estimated_names:
            pdict[name] = float(post_samples[name][idx])

        result = compute_irfs_for_params(
            m.model, pdict, [shock_name], variables, horizon)
        if result is not None:
            post_irfs.append(result[shock_name])

    post_irfs = np.stack(post_irfs)
    print(f"  Valid posterior draws: {len(post_irfs)}")

    # ---- Plot ----
    print("Plotting prior vs posterior IRFs...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    quarters = np.arange(horizon)

    for k, (ax, label) in enumerate(zip(axes, var_labels)):
        # Prior bands (90%)
        prior_lo = np.percentile(prior_irfs[:, :, k], 5, axis=0)
        prior_hi = np.percentile(prior_irfs[:, :, k], 95, axis=0)
        prior_med = np.median(prior_irfs[:, :, k], axis=0)

        # Posterior bands (90%)
        post_lo = np.percentile(post_irfs[:, :, k], 5, axis=0)
        post_hi = np.percentile(post_irfs[:, :, k], 95, axis=0)
        post_med = np.median(post_irfs[:, :, k], axis=0)

        # Prior: light gray
        ax.fill_between(quarters, prior_lo, prior_hi,
                        color='#cccccc', alpha=0.6, label='Prior 90% CI')
        ax.plot(quarters, prior_med, '--', color='gray',
                linewidth=1.0, label='Prior median')

        # Posterior: blue
        ax.fill_between(quarters, post_lo, post_hi,
                        color='#4488cc', alpha=0.4, label='Posterior 90% CI')
        ax.plot(quarters, post_med, '-', color='#1a5599',
                linewidth=1.5, label='Posterior median')

        ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Quarters')

        if k == 0:
            ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('Monetary Policy Shock: Prior vs Posterior IRFs',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(out_dir, 'prior_vs_posterior_irf_SHK_RS.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also save the data for potential reuse
    np.savez(os.path.join(out_dir, 'prior_vs_posterior_irf_data.npz'),
             prior_irfs=prior_irfs, post_irfs=post_irfs,
             variables=variables, shock=shock_name)


if __name__ == '__main__':
    main()

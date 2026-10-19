"""Post-process posterior draws to report spectral radius distribution.

Loads saved MCMC draws, solves the model for each, and reports the
distribution of rho(T_stat) across all posterior draws.

Usage:
    python compute_spectral_radius.py
    python compute_spectral_radius.py --results-dir results_v2/model_a
"""

import os
import sys
import argparse

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX_DIR = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX_DIR)

from qpm_toolbox import QPMModel
from _runner_loader import MODEL_LABELS, load_runner, results_dir as _results_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='A', choices=MODEL_LABELS,
                        help='Model label (default: A)')
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--n-draws', type=int, default=1000,
                        help='Subsample of posterior draws to evaluate '
                             '(default: 1000; use 0 for all draws).')
    parser.add_argument('--seed', type=int, default=42)
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

    print(f"Loading Model {args.model} and posterior draws from {results_dir}...")
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
    m.load_draws()

    samples = m._posterior_samples
    n_total = samples[next(iter(samples))].shape[0]
    if args.n_draws and args.n_draws < n_total:
        rng = np.random.default_rng(args.seed)
        draw_idx = rng.choice(n_total, size=args.n_draws, replace=False)
        print(f"  {n_total} posterior draws loaded; subsampling {args.n_draws} "
              f"(seed={args.seed}).")
    else:
        draw_idx = np.arange(n_total)
        print(f"  {n_total} posterior draws loaded; using all of them.")

    print("Computing spectral radius for each draw...")
    rhos = []
    n_invalid = 0
    for i in draw_idx:
        pdict = dict(m.model.default_param_assignments)
        pdict.update(FIXED_PARAMS)
        if m._calibrated_sigmas:
            pdict.update(m._calibrated_sigmas)
        for name in m._estimated_names:
            pdict[name] = float(samples[name][int(i)])

        sol = m.model.solve(pdict)
        if not sol['solution_valid']:
            n_invalid += 1
            continue
        rhos.append(float(sol['spectral_radius']))

    rhos = np.array(rhos)

    print(f"\n{'=' * 50}")
    print(f"  Spectral Radius Distribution (stationary block)")
    print(f"{'=' * 50}")
    print(f"  Valid draws:  {len(rhos)} / {len(draw_idx)} evaluated "
          f"(of {n_total} posterior draws total)")
    print(f"  Invalid:      {n_invalid}")
    print(f"  Minimum:      {rhos.min():.6f}")
    print(f"  Median:       {np.median(rhos):.6f}")
    print(f"  Mean:         {rhos.mean():.6f}")
    print(f"  Maximum:      {rhos.max():.6f}")
    print(f"  Std:          {rhos.std():.6f}")
    print(f"  P95:          {np.percentile(rhos, 95):.6f}")
    print(f"  P99:          {np.percentile(rhos, 99):.6f}")
    print(f"  All < 1.0:    {bool(np.all(rhos < 1.0))}")
    print(f"{'=' * 50}")

    # Save
    out_path = os.path.join(results_dir, 'estimation',
                            'spectral_radius.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, rhos=rhos)
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()

"""Generate filter diagnostic plots at the posterior median.

Loads saved MCMC draws and re-runs the Kalman smoother to produce
diag_observables.png, diag_states.png, diag_trends.png evaluated
at the posterior (not the initial calibration).

Usage:
    python generate_posterior_diagnostics.py                # Model A (default)
    python generate_posterior_diagnostics.py --model F      # Any of A, A0, B, C, D, E, F
"""

import argparse
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX_DIR = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX_DIR)

from _runner_loader import MODEL_LABELS, load_runner, results_dir as _results_dir
from qpm_toolbox import QPMModel


def main():
    import jax
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='A', choices=MODEL_LABELS,
                        help='Model label (default: A)')
    parser.add_argument('--results-dir', default=None,
                        help='Override results dir (defaults to '
                             'results_v2_diffuse/model_<x>)')
    args = parser.parse_args()

    runner = load_runner(args.model)
    MOD_FILE = runner.MOD_FILE
    RESULTS_DIR = args.results_dir or _results_dir(args.model)

    print("=" * 60)
    print(f"  POSTERIOR FILTER DIAGNOSTICS  [Model {args.model}]")
    print(f"  Results dir: {RESULTS_DIR}")
    print("=" * 60)

    # 1. Load data and model
    ys, dates, obs_names = runner.load_brazil_data()
    print(f"  Data: {ys.shape[0]}Q x {ys.shape[1]} obs")

    m = QPMModel(MOD_FILE, results_dir=RESULTS_DIR)
    ss_params = m.model.default_param_assignments
    init_config = runner.compute_initial_conditions(ys, obs_names, ss_params)

    m.set_data(ys, dates=dates, obs_names=obs_names)
    m.set_priors(
        structural_priors=runner.STRUCTURAL_PRIORS,
        fixed_params=runner.FIXED_PARAMS,
        meas_noise_std=runner.MEAS_NOISE_STD,
        cycle_map=runner.CYCLE_MAP,
        trend_snr_map=runner.TREND_SNR_MAP,
        manual_cycle_priors=runner.MANUAL_CYCLE_PRIORS,
    )
    m.set_initial_conditions(init_state_config=init_config)
    m.solution_check()

    # 2. Load saved posterior draws
    draws_path = os.path.join(RESULTS_DIR, "estimation", "draws.npz")
    if not os.path.isfile(draws_path):
        raise FileNotFoundError(f"No draws found at {draws_path}")
    m.load_draws(draws_path)
    print(f"  Loaded draws from {draws_path}")

    # 3. Run filter diagnostic (uses posterior median via _resolve_params)
    print("  Running Kalman smoother at posterior median...")
    result = m.filter_diagnostic(obs_trend_map=runner.OBS_TREND_MAP)
    print(f"  Log-likelihood at posterior median: {result['ll']:.1f}")
    print(f"  Plots saved to {RESULTS_DIR}/diagnostic/")
    print("=" * 60)


if __name__ == "__main__":
    main()

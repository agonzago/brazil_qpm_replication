"""Generate conditional forecast spaghetti plots for all QPM variants.

Loads saved posterior draws for each model and runs only the conditional
rolling-window forecast evaluation (conditioning on external variables
and energy/food inflation, matching the BVAR conditioning set).

Usage:
    python run_conditional_eval.py              # all models
    python run_conditional_eval.py --model b    # single model
    python run_conditional_eval.py --model a b c  # subset
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'gpm_estimation'))
sys.path.insert(0, _THIS_DIR)

from qpm_toolbox import QPMModel

# ======================================================================
# Conditioning specification (matches BVAR conditioning set)
# ======================================================================
CONDITION_ON = [
    ('obs', 'OBS_LRWOIL_GAP',     'SHK_LRWOIL_GAP'),
    ('obs', 'OBS_LRWFOOD_GAP',    'SHK_LRWFOOD_GAP'),
    ('obs', 'OBS_RS_US',          'SHK_RS_US'),
    ('obs', 'OBS_GDP_GAP_US',     'SHK_GAP_US'),
    ('obs', 'OBS_DA_LCPI_ENERGY', 'SHK_DA_LCPI_ENERGY'),
    ('obs', 'OBS_DA_LCPI_US',     'SHK_DA_LCPI_US'),
    ('obs', 'OBS_RR_BAR_US',      'SHK_RR_BAR_US'),
]

EVAL_VARS = ['OBS_DA_LCPI', 'OBS_LGDP', 'OBS_RS', 'OBS_DA_LCPIX']
FIRST_ORIGIN = '2017-01-01'
LAST_ORIGIN = '2022-10-01'
HORIZONS = [1, 2, 3, 4, 8]
MAX_HORIZON = 8

# ======================================================================
# Model specifications
# ======================================================================
MODELS = {
    'a': {
        'label': 'Model A (TV-R*)',
        'mod_file': os.path.join(_THIS_DIR, 'model', 'brazil_qpm_mr_v2.qpm'),
        'results_dir': os.path.join(_THIS_DIR, 'results_v2_diffuse', 'model_a'),
        'draws_path': os.path.join(_THIS_DIR, 'results_v2_diffuse', 'model_a',
                                   'estimation', 'draws.npz'),
        'config_module': 'run_brazil_wp_v2',
    },
    'a0': {
        'label': 'Model A0 (Const-R*)',
        'mod_file': os.path.join(_THIS_DIR, 'model', 'brazil_qpm_mr_v2.qpm'),
        'results_dir': os.path.join(_THIS_DIR, 'results_v2_diffuse',
                                    'model_a_const_rr'),
        'draws_path': os.path.join(_THIS_DIR, 'results_v2_diffuse',
                                    'model_a_const_rr',
                                   'estimation', 'draws.npz'),
        'config_module': 'run_brazil_model_a_const_rr_v2',
    },
    'b': {
        'label': 'Model B (Koyck IS)',
        'mod_file': os.path.join(_THIS_DIR, 'model', 'brazil_qpm_model_b_v2.qpm'),
        'results_dir': os.path.join(_THIS_DIR, 'results_v2_diffuse', 'model_b'),
        'draws_path': os.path.join(_THIS_DIR, 'results_v2_diffuse', 'model_b',
                                   'estimation', 'draws.npz'),
        'config_module': 'run_brazil_model_b_v2',
    },
    'c': {
        'label': 'Model C (Koyck Phillips)',
        'mod_file': os.path.join(_THIS_DIR, 'model', 'brazil_qpm_model_c_v2.qpm'),
        'results_dir': os.path.join(_THIS_DIR, 'results_v2_diffuse', 'model_c'),
        'draws_path': os.path.join(_THIS_DIR, 'results_v2_diffuse', 'model_c',
                                   'estimation', 'draws.npz'),
        'config_module': 'run_brazil_model_c_v2',
    },
    'd': {
        'label': 'Model D (Simple Taylor)',
        'mod_file': os.path.join(_THIS_DIR, 'model', 'brazil_qpm_model_d_v2.qpm'),
        'results_dir': os.path.join(_THIS_DIR, 'results_v2_diffuse', 'model_d'),
        'draws_path': os.path.join(_THIS_DIR, 'results_v2_diffuse', 'model_d',
                                   'estimation', 'draws.npz'),
        'config_module': 'run_brazil_model_d_v2',
    },
}


def run_model(key: str, spec: dict):
    """Load saved draws and run conditional forecast evaluation."""
    print("\n" + "=" * 70)
    print(f"  {spec['label']}")
    print("=" * 70)

    # Check draws exist
    if not os.path.exists(spec['draws_path']):
        print(f"  SKIP: draws not found at {spec['draws_path']}")
        return

    # Import model-specific configuration
    cfg = __import__(spec['config_module'])

    # Load data
    ys, dates, obs_names = cfg.load_brazil_data()
    print(f"  Data: {ys.shape[0]} quarters, {ys.shape[1]} observables")

    # Create model
    m = QPMModel(spec['mod_file'], results_dir=spec['results_dir'])

    # Initial conditions
    ss_params = m.model.default_param_assignments
    init_config = cfg.compute_initial_conditions(ys, obs_names, ss_params)

    # Configure
    m.set_data(ys, dates=dates, obs_names=obs_names)
    m.set_priors(
        structural_priors=cfg.STRUCTURAL_PRIORS,
        fixed_params=cfg.FIXED_PARAMS,
        meas_noise_std=cfg.MEAS_NOISE_STD,
        cycle_map=cfg.CYCLE_MAP,
        trend_snr_map=cfg.TREND_SNR_MAP,
        manual_cycle_priors=cfg.MANUAL_CYCLE_PRIORS,
    )
    m.set_initial_conditions(init_state_config=init_config)

    # Load saved draws
    print(f"  Loading draws from {spec['draws_path']}")
    m.load_draws(spec['draws_path'])

    # Run conditional forecast evaluation
    print("\n  Running conditional forecast evaluation...")
    ev_cond = m.conditional_forecast_evaluation(
        variables=EVAL_VARS,
        condition_on=CONDITION_ON,
        first_origin=FIRST_ORIGIN,
        last_origin=LAST_ORIGIN,
        horizons=HORIZONS,
        max_horizon=MAX_HORIZON,
    )

    print(f"\n  Done: {spec['label']}")
    return ev_cond


def main():
    parser = argparse.ArgumentParser(
        description="Conditional forecast evaluation for all QPM variants")
    parser.add_argument("--model", nargs='+', default=None,
                        choices=list(MODELS.keys()),
                        help="Model(s) to run (default: all)")
    args = parser.parse_args()

    models_to_run = args.model or list(MODELS.keys())

    print("=" * 70)
    print("  CONDITIONAL FORECAST EVALUATION")
    print(f"  Models: {models_to_run}")
    print(f"  Conditioning on: {[c[1] for c in CONDITION_ON]}")
    print("=" * 70)

    results = {}
    for key in models_to_run:
        spec = MODELS[key]
        try:
            results[key] = run_model(key, spec)
        except Exception as e:
            print(f"\n  ERROR on {spec['label']}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparative RMSE summary
    print("\n" + "=" * 70)
    print("  CONDITIONAL FORECAST RMSE COMPARISON")
    print("=" * 70)

    for var in EVAL_VARS:
        print(f"\n  {var}:")
        header = f"  {'Horizon':>8s}"
        for key in models_to_run:
            header += f"  {key.upper():>8s}"
        print(header)
        print("  " + "-" * len(header))
        for h in HORIZONS:
            row = f"  {f'h={h}':>8s}"
            for key in models_to_run:
                if results.get(key) is not None:
                    val = results[key].rmse[var].get(h, np.nan)
                    row += f"  {val:8.3f}"
                else:
                    row += f"  {'---':>8s}"
            print(row)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

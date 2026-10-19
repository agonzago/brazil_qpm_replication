"""Compute GDP_GAP and PREM_GAP filtration stats for one v5 sweep config.

Run as: python _compute_gap_stats.py <model> <results_dir>

Imports the right runner module (env vars must be set BEFORE invocation), builds
the model, runs the Kalman smoother at the posterior median, extracts
smoothed GDP_GAP and PREM_GAP, and prints summary stats as JSON.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX)
sys.path.insert(0, SCRIPT_DIR)

MODEL_RUNNER = {
    "A": "run_brazil_wp_v2",
    "B": "run_brazil_model_b_v2",
    "C": "run_brazil_model_c_v2",
    "D": "run_brazil_model_d_v2",
    "E": "run_brazil_model_e_v2",
    "F": "run_brazil_model_f_v2",
}


def main(model: str, results_dir: str) -> int:
    import importlib
    runner = importlib.import_module(MODEL_RUNNER[model])
    from qpm_toolbox import QPMModel

    ys, dates, obs_names = runner.load_brazil_data()
    m = QPMModel(runner.MOD_FILE, results_dir=results_dir)
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

    draws_path = os.path.join(results_dir, "estimation", "draws.npz")
    m.load_draws(draws_path)

    # Run filter at posterior median
    res = m.filter_diagnostic(obs_trend_map=runner.OBS_TREND_MAP)
    x_smooth = res["x_smooth"]   # (T, n_aug)

    # Find state indices
    n_stat = m.model.n_stat
    stat_names = m.model.ordered_stat_vars
    trend_names = m.model.ordered_trend_state_vars

    out: dict = {}

    if "GDP_GAP" in stat_names:
        idx = stat_names.index("GDP_GAP")
        s = pd.Series(np.asarray(x_smooth[:, idx])).dropna()
        out["GDP_GAP"] = {
            "max_abs": float(s.abs().max()),
            "min": float(s.min()),
            "max": float(s.max()),
            "std": float(s.std()),
            "share_gt_4": float((s.abs() > 4.0).mean()),
            "share_gt_2": float((s.abs() > 2.0).mean()),
        }

    if "PREM_GAP" in stat_names:
        idx = stat_names.index("PREM_GAP")
        s = pd.Series(np.asarray(x_smooth[:, idx])).dropna()
        out["PREM_GAP"] = {
            "ar1": float(s.autocorr(lag=1)) if len(s) > 1 else None,
            "ar4": float(s.autocorr(lag=4)) if len(s) > 4 else None,
            "abs_mean": float(s.abs().mean()),
            "std": float(s.std()),
        }

    # PREM_BAR (trend) range to see if structural decline captured
    if "PREM_BAR" in trend_names:
        idx = n_stat + trend_names.index("PREM_BAR")
        s = pd.Series(np.asarray(x_smooth[:, idx])).dropna()
        out["PREM_BAR"] = {
            "first": float(s.iloc[0]),
            "last": float(s.iloc[-1]),
            "range": float(s.iloc[-1] - s.iloc[0]),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    # TT_BAR (trend) shape
    if "TT_BAR" in trend_names:
        idx = n_stat + trend_names.index("TT_BAR")
        s = pd.Series(np.asarray(x_smooth[:, idx])).dropna()
        out["TT_BAR"] = {
            "std": float(s.std()),
            "range": float(s.max() - s.min()),
        }

    if "TT_GAP" in stat_names:
        idx = stat_names.index("TT_GAP")
        s = pd.Series(np.asarray(x_smooth[:, idx])).dropna()
        out["TT_GAP"] = {
            "std": float(s.std()),
            "max_abs": float(s.abs().max()),
            "ar1": float(s.autocorr(lag=1)) if len(s) > 1 else None,
        }

    out["log_likelihood"] = float(res["ll"])
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python _compute_gap_stats.py <model> <results_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

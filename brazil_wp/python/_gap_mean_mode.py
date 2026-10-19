"""Compute smoothed GDP_GAP at posterior MEAN and MEDIAN params.

Run as: python _gap_mean_mode.py <model> <results_dir>

Outputs JSON with two series: gap_mean (smoothed at param posterior mean)
and gap_median (smoothed at param posterior median = MAP/mode proxy).
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


def build_pdict(m, summary: str) -> dict:
    """Build a param dict using posterior `mean` or `median` of each draw."""
    pdict = dict(m.model.default_param_assignments)
    if m._fixed_params:
        pdict.update(m._fixed_params)
    if m._calibrated_sigmas:
        pdict.update(m._calibrated_sigmas)
    for name, draws in m._posterior_samples.items():
        arr = np.asarray(draws)
        if summary == "mean":
            pdict[name] = float(np.mean(arr))
        elif summary == "median":
            pdict[name] = float(np.median(arr))
        else:
            raise ValueError(summary)
    return pdict


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
    m.load_draws(os.path.join(results_dir, "estimation", "draws.npz"))

    stat_names = m.model.ordered_stat_vars
    gap_idx = stat_names.index("GDP_GAP")
    series = {}
    for summary in ("mean", "median"):
        pdict = build_pdict(m, summary)
        res = m.filter_diagnostic(params=pdict, obs_trend_map=runner.OBS_TREND_MAP)
        xs = np.asarray(res["x_smooth"])
        series[summary] = xs[:, gap_idx].tolist()
    out = {
        "dates": [str(d) for d in dates],
        "gap_mean":   series["mean"],
        "gap_median": series["median"],
    }
    print("JSON_BEGIN")
    print(json.dumps(out))
    print("JSON_END")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python _gap_mean_mode.py <model> <results_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

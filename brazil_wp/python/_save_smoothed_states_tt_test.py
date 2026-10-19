"""Save smoothed states for the TT-test estimation.

Mirrors _save_smoothed_states.py for Model A but uses the patched
load_brazil_data from run_tt_russia_constrained_test (two-tranche HP
TT trend).  Writes diagnostic/smoothed_states.npz with the same schema
as the baseline _save_smoothed_states.py.

Usage:
    python _save_smoothed_states_tt_test.py <results_dir>
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("V5_TT_MODE", "tworegime")

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX)
sys.path.insert(0, SCRIPT_DIR)


def build_pdict(m, summary: str) -> dict:
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


def main(results_dir: str) -> int:
    import run_brazil_wp_v2 as base
    import run_tt_russia_constrained_test as test_wrapper
    # Apply the wrapper's patched loader
    base.load_brazil_data = test_wrapper.load_brazil_data_constrained
    from qpm_toolbox import QPMModel

    ys, dates, obs_names = base.load_brazil_data()
    m = QPMModel(base.MOD_FILE, results_dir=results_dir)
    ss_params = m.model.default_param_assignments
    init_config = base.compute_initial_conditions(ys, obs_names, ss_params)
    m.set_data(ys, dates=dates, obs_names=obs_names)
    m.set_priors(
        structural_priors=base.STRUCTURAL_PRIORS,
        fixed_params=base.FIXED_PARAMS,
        meas_noise_std=base.MEAS_NOISE_STD,
        cycle_map=base.CYCLE_MAP,
        trend_snr_map=base.TREND_SNR_MAP,
        manual_cycle_priors=base.MANUAL_CYCLE_PRIORS,
    )
    m.set_initial_conditions(init_state_config=init_config)
    m.solution_check()
    m.load_draws(os.path.join(results_dir, "estimation", "draws.npz"))

    state_names = list(m.model.ordered_stat_vars)
    trend_names = list(m.model.ordered_trend_state_vars)
    obs_names_l = list(m.model.ordered_obs_vars)

    out: dict = {
        "dates":       np.array([str(d) for d in dates]),
        "state_names": np.array(state_names),
        "trend_names": np.array(trend_names),
        "obs_names":   np.array(obs_names_l),
    }
    for summary in ("mean", "median"):
        pdict = build_pdict(m, summary)
        res = m.filter_diagnostic(params=pdict,
                                  obs_trend_map=base.OBS_TREND_MAP)
        x_smooth = np.asarray(res["x_smooth"])
        obs_smooth = np.asarray(res["obs_smooth"])
        out[f"x_smooth_{summary}"]   = x_smooth.astype(np.float32)
        out[f"obs_smooth_{summary}"] = obs_smooth.astype(np.float32)
        out[f"log_likelihood_{summary}"] = float(res["ll"])
        out[f"params_{summary}_names"]  = np.array(list(pdict.keys()))
        out[f"params_{summary}_values"] = np.array(list(pdict.values()),
                                                   dtype=np.float64)

    out_path = os.path.join(results_dir, "diagnostic", "smoothed_states.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"OK  saved {out_path}  (T={len(dates)}, "
          f"n_stat={len(state_names)}, n_trend={len(trend_names)})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python _save_smoothed_states_tt_test.py <results_dir>",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

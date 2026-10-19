"""Compute and save smoothed states for one v5 sweep config.

Saves results to <results_dir>/diagnostic/smoothed_states.npz with keys:
  - dates                : (T,) ISO-string array
  - state_names          : (n_stat,)  ordered stationary state names
  - trend_names          : (n_trend,) ordered trend state names
  - obs_names            : (n_obs,)
  - x_smooth_mean        : (T, n_aug) smoothed states at posterior MEAN params
  - x_smooth_median      : (T, n_aug) smoothed states at posterior MEDIAN params
  - obs_smooth_mean      : (T, n_obs) reconstructed observables (Omega @ x) at MEAN
  - obs_smooth_median    : (T, n_obs) reconstructed observables at MEDIAN
  - log_likelihood_mean  : scalar
  - log_likelihood_median: scalar

Note: n_aug = n_stat + n_trend; states are stacked [stat ; trend].
Run as: python _save_smoothed_states.py <model> <results_dir>
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX)
sys.path.insert(0, SCRIPT_DIR)

MODEL_RUNNER = {
    "A":    "run_brazil_wp_v2",
    "A0":   "run_brazil_model_a_const_rr_v2",
    "B":    "run_brazil_model_b_v2",
    "C":    "run_brazil_model_c_v2",
    "D":    "run_brazil_model_d_v2",
    "E":    "run_brazil_model_e_v2",
    "F":    "run_brazil_model_f_v2",
    # v5 robustness wrappers (thin overrides on run_brazil_wp_v2).
    "RUB1": "run_rub1_v5",
    "RUB3": "run_rub3_v5",
    "RUB4": "run_rub4_v5",
    "RUB5": "run_rub5_v5",
}


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


def main(model: str, results_dir: str) -> int:
    import importlib
    runner = importlib.import_module(MODEL_RUNNER[model])
    from qpm_toolbox import QPMModel

    # The RUB wrappers import run_brazil_wp_v2 as `base` and only override
    # STRUCTURAL_PRIORS / FIXED_PARAMS / MEAS_NOISE_STD / MOD_FILE.  They
    # inherit everything else (load_brazil_data, OBS_TREND_MAP, CYCLE_MAP,
    # TREND_SNR_MAP, MANUAL_CYCLE_PRIORS, compute_initial_conditions) from
    # base.  Fall through to `runner.base` (set by the wrapper) when an
    # attribute isn't on the wrapper itself.
    def _get(name):
        if hasattr(runner, name):
            return getattr(runner, name)
        base = getattr(runner, "base", None)
        if base is not None and hasattr(base, name):
            return getattr(base, name)
        raise AttributeError(
            f"{MODEL_RUNNER[model]!r} has no attribute {name!r} "
            f"(and no `base` fallback exposes it)")

    ys, dates, obs_names = _get("load_brazil_data")()
    m = QPMModel(_get("MOD_FILE"), results_dir=results_dir)
    ss_params = m.model.default_param_assignments
    init_config = _get("compute_initial_conditions")(ys, obs_names, ss_params)
    m.set_data(ys, dates=dates, obs_names=obs_names)
    m.set_priors(
        structural_priors=_get("STRUCTURAL_PRIORS"),
        fixed_params=_get("FIXED_PARAMS"),
        meas_noise_std=_get("MEAS_NOISE_STD"),
        cycle_map=_get("CYCLE_MAP"),
        trend_snr_map=_get("TREND_SNR_MAP"),
        manual_cycle_priors=_get("MANUAL_CYCLE_PRIORS"),
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
                                  obs_trend_map=_get("OBS_TREND_MAP"))
        x_smooth = np.asarray(res["x_smooth"])      # (T, n_aug)
        obs_smooth = np.asarray(res["obs_smooth"])  # (T, n_obs)
        out[f"x_smooth_{summary}"]   = x_smooth.astype(np.float32)
        out[f"obs_smooth_{summary}"] = obs_smooth.astype(np.float32)
        out[f"log_likelihood_{summary}"] = float(res["ll"])
        # Also save the param dict used (as flat name/value arrays) for traceability
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
    if len(sys.argv) != 3:
        print("usage: python _save_smoothed_states.py <model> <results_dir>",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

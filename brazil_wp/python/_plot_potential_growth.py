"""Potential growth fan chart with HDI bands.

Adapted from results_v2_diffuse_v5/_scripts/make_potential_growth_plot.py to
accept any results_dir whose draws.npz was produced by run_brazil_wp_v2.
Re-runs the Kalman smoother at N_DRAWS samples to build the HDI for G_TREND
(annualized) and overlays realised GDP growth.  All env vars used at
estimation time MUST be set in the calling shell -- this script does NOT
patch the runner; it relies on the runner reading them at import time.

Usage:
    <env vars...> python _plot_potential_growth.py <results_dir> [n_draws]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLBOX = SCRIPT_DIR.parent.parent / "gpm_estimation"
sys.path.insert(0, str(TOOLBOX))
sys.path.insert(0, str(SCRIPT_DIR))

DATA = SCRIPT_DIR / "data" / "estimation_data.csv"

SEED = 0


def hdi(arr, mass=0.68):
    n, T = arr.shape
    lo = np.empty(T); hi = np.empty(T)
    k = int(np.ceil(mass * n))
    for t in range(T):
        col = np.sort(arr[:, t])
        widths = col[k:] - col[:n - k]
        i = int(np.argmin(widths))
        lo[t] = col[i]; hi[t] = col[i + k]
    return lo, hi


def build_pdict(m, j, names, samples):
    pdict = dict(m.model.default_param_assignments)
    if m._fixed_params:
        pdict.update(m._fixed_params)
    if m._calibrated_sigmas:
        pdict.update(m._calibrated_sigmas)
    for name, d in zip(names, samples):
        pdict[name] = float(d[j])
    return pdict


def main(results_dir: str, n_draws: int = 100) -> int:
    rd = Path(results_dir)
    if not rd.is_absolute():
        rd = SCRIPT_DIR / rd
    out = rd / "diagnostic" / "potential_growth.png"

    import importlib
    runner = importlib.import_module("run_brazil_wp_v2")
    from qpm_toolbox import QPMModel

    ys, dates, obs_names = runner.load_brazil_data()
    m = QPMModel(runner.MOD_FILE, results_dir=str(rd))
    ss = m.model.default_param_assignments
    init_config = runner.compute_initial_conditions(ys, obs_names, ss)
    m.set_data(ys, dates=dates, obs_names=obs_names)
    m.set_priors(structural_priors=runner.STRUCTURAL_PRIORS,
                 fixed_params=runner.FIXED_PARAMS,
                 meas_noise_std=runner.MEAS_NOISE_STD,
                 cycle_map=runner.CYCLE_MAP,
                 trend_snr_map=runner.TREND_SNR_MAP,
                 manual_cycle_priors=runner.MANUAL_CYCLE_PRIORS)
    m.set_initial_conditions(init_state_config=init_config)
    m.solution_check()
    m.load_draws(os.path.join(str(rd), "estimation", "draws.npz"))

    names = list(m._posterior_samples.keys())
    samples = [np.asarray(m._posterior_samples[k]) for k in names]
    n_total = samples[0].shape[0]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_total, size=min(n_draws, n_total), replace=False)

    sn = list(m.model.ordered_stat_vars)
    tn = list(m.model.ordered_trend_state_vars)
    i_g = (sn + tn).index("G_TREND")

    T = len(dates)
    g_draws = np.empty((len(idx), T))
    print(f"[potential_growth] smoothing G_TREND at {len(idx)} draws -> {rd}")
    t0 = time.time()
    for k, j in enumerate(idx):
        pdict = build_pdict(m, j, names, samples)
        res = m.filter_diagnostic(params=pdict,
                                  obs_trend_map=runner.OBS_TREND_MAP)
        g_draws[k] = np.asarray(res["x_smooth"])[:, i_g] * 4.0
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(idx)}  ({time.time()-t0:.1f}s)")

    g_med = np.median(g_draws, axis=0)
    g_lo68, g_hi68 = hdi(g_draws, 0.68)
    g_lo90, g_hi90 = hdi(g_draws, 0.90)

    plot_dates = pd.to_datetime([str(d) for d in dates])
    df = pd.read_csv(DATA, parse_dates=["date"]).set_index("date")
    realised = (df["OBS_LGDP"].diff() * 4.0).reindex(plot_dates)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(plot_dates, realised, color="0.55", lw=1.2, alpha=0.85,
            label=r"Realised GDP growth, $\Delta^a y_t$ (DA_LGDP)")
    ax.fill_between(plot_dates, g_lo90, g_hi90, color="C0", alpha=0.15,
                    label="90% HDI")
    ax.fill_between(plot_dates, g_lo68, g_hi68, color="C0", alpha=0.30,
                    label="68% HDI")
    ax.plot(plot_dates, g_med, color="C0", lw=2.2,
            label=r"Potential growth, $\bar g_t$ (G_TREND posterior median)")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.6)
    ax.set_ylabel("Annualized, percent")
    ax.set_title(f"Potential growth fan chart  ({rd.name})", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(plot_dates.min(), plot_dates.max())
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: <env vars> python _plot_potential_growth.py <results_dir> [n_draws]")
        sys.exit(2)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    sys.exit(main(sys.argv[1], n))

"""Standard LZ + DA_LZ diagnostic plot.

For a given results_dir holding diagnostic/smoothed_states.npz, draws:

  Panel 1  LZ (= LZ_BAR + LZ_GAP, level) and LZ_BAR (trend).
           LZ_GAP appears as the residual.
  Panel 2  DA_LZ (= 4*(LZ_t - LZ_{t-1}), annualized) and
           DA_LZ_BAR (= 4*(LZ_BAR_t - LZ_BAR_{t-1}), annualized trend).
  Panel 3  RR_BAR (domestic real natural rate) decomposed into
           RR_BAR_US + PREM_BAR + 4*d_lz_ss (UIP-implied) for context.

These four variables are all smoothed but none of them is in varobs
(the observed quantity is OBS_DA_LS = nominal depreciation, from which
LZ is reconstructed via the CPI inflation differential identity).

Saves to <results_dir>/diagnostic/lz_diagnostic.png.

Usage:
    python _plot_lz_diagnostic.py <results_dir> [d_lz_ss]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_plot(results_dir: Path, d_lz_ss: float = 0.50) -> Path:
    npz = results_dir / "diagnostic" / "smoothed_states.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)
    d = np.load(npz, allow_pickle=True)
    sn = list(d["state_names"]); tn = list(d["trend_names"])
    dates = pd.to_datetime([str(s) for s in d["dates"]])
    x_mean = np.asarray(d.get("x_smooth_mean", d.get("x_smooth")))
    x_med  = np.asarray(d.get("x_smooth_median", x_mean))
    n_stat = len(sn)

    def get(arr, name):
        if name in sn:
            return pd.Series(arr[:, sn.index(name)], index=dates)
        if name in tn:
            return pd.Series(arr[:, n_stat + tn.index(name)], index=dates)
        return None

    lz_bar  = get(x_mean, "LZ_BAR")
    lz_gap  = get(x_mean, "LZ_GAP")
    if lz_bar is None or lz_gap is None:
        raise RuntimeError(f"LZ_BAR/LZ_GAP not found in states for {results_dir}")
    lz_lvl  = lz_bar + lz_gap
    lz_med  = get(x_med, "LZ_BAR") + get(x_med, "LZ_GAP")

    da_lz       = 4 * lz_lvl.diff()
    da_lz_bar   = 4 * lz_bar.diff()

    rr_bar    = get(x_mean, "RR_BAR")
    rr_bar_us = get(x_mean, "RR_BAR_US")
    prem_bar  = get(x_mean, "PREM_BAR")
    implied_rr = (rr_bar_us + prem_bar + 4 * d_lz_ss
                  if rr_bar_us is not None and prem_bar is not None else None)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    ax = axes[0]
    ax.plot(dates, lz_lvl.values, "k-", lw=1.0, alpha=0.7,
            label="LZ = LZ_BAR + LZ_GAP (level, posterior mean)")
    ax.plot(dates, lz_bar.values, "r--", lw=1.8,
            label="LZ_BAR (trend, posterior mean)")
    ax.plot(dates, lz_med.values, "k:", lw=0.8, alpha=0.5,
            label="LZ (posterior median)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(
        f"REER level (100·log): LZ vs LZ_BAR.   "
        f"LZ_GAP mean={lz_gap.mean():+.2f}, share<0={(lz_gap<0).mean()*100:.0f}%, "
        f"min={lz_gap.min():+.2f}, max={lz_gap.max():+.2f}",
        fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(dates, da_lz.values, "k-", lw=1.0, alpha=0.8,
            label="DA_LZ = 4·ΔLZ (annualized REER change)")
    ax.plot(dates, da_lz_bar.values, "r--", lw=1.8,
            label="DA_LZ_BAR = 4·ΔLZ_BAR (annualized trend REER change)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(
        f"Annualized REER change.   "
        f"DA_LZ_BAR mean={da_lz_bar.mean():+.2f}, "
        f"DA_LZ mean={da_lz.mean():+.2f}",
        fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("pp / year")

    ax = axes[2]
    if rr_bar is not None:
        ax.plot(dates, rr_bar.values, "C0-", lw=1.8, label="RR_BAR (real natural rate)")
    if rr_bar_us is not None:
        ax.plot(dates, rr_bar_us.values, "C2-", lw=1.0, alpha=0.7, label="RR_BAR_US")
    if prem_bar is not None:
        ax.plot(dates, prem_bar.values, "C1-", lw=1.0, alpha=0.7, label="PREM_BAR")
    if implied_rr is not None:
        ax.plot(dates, implied_rr.values, "k--", lw=1.0, alpha=0.7,
                label=f"RR_BAR_US + PREM_BAR + 4·d_lz_ss   (d_lz_ss={d_lz_ss})")
    title = ("Natural rate decomposition (annualized %): " +
             (f"RR_BAR end={rr_bar.iloc[-1]:.2f}%, mean={rr_bar.mean():.2f}%"
              if rr_bar is not None else ""))
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("date")
    ax.set_ylabel("annualized %")

    fig.tight_layout()
    out = results_dir / "diagnostic" / "lz_diagnostic.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python _plot_lz_diagnostic.py <results_dir> [d_lz_ss]")
        return 2
    rd = Path(argv[1])
    if not rd.is_absolute():
        rd = Path(__file__).resolve().parent / rd
    d_lz_ss = float(argv[2]) if len(argv) > 2 else 0.50
    out = make_plot(rd, d_lz_ss=d_lz_ss)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

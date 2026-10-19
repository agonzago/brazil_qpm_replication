"""Overlay GDP_GAP from the three TT experiments on the BCB output-gap envelope.

For comparison, also overlay the smoothed TT_GAP and TT_BAR across the three
specs so we can see how the trend/cycle decomposition shifts when TT_BAR is
observed.

Outputs (under results_tt_exp/):
  gap_vs_bcb.png           — GDP_GAP, posterior MEAN (3 specs) vs BCB Low–High
  gap_vs_bcb_median.png    — same, posterior MEDIAN
  tt_decomposition.png     — TT_GAP and TT_BAR (3 specs) vs the data OBS_TT
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


THIS_DIR = Path(__file__).resolve().parent
OUT_DIR  = THIS_DIR / "results_tt_exp"
BCB_CSV  = THIS_DIR / "data" / "BCB_outputgap_Garf2.2.8.csv"
DATA_CSV = THIS_DIR / "data" / "estimation_data.csv"
# Paper-quality (4×1000+1000) two-regime estimation lives outside results_tt_exp.
PAPER_TT2_DIR = THIS_DIR / "results_v2_diffuse" / "model_a_tt2"

EXPS = [
    ("no_filter",           "1. TT_BAR free (no anchor)",                                 "#1f77b4", None),
    ("anchor_no_filter",    "2. HP-anchor hard (std=0.10)",                               "#d62728", None),
    ("anchor_endpoint",     "3. HP + cycle=0 at T-1 (hard)",                              "#9467bd", None),
    ("anchor_meas2",        "4. HP-anchor soft, std=2.0",                                 "#17becf", None),
    ("anchor_tworegime",    "5. Two-regime (short-chain 300+500)",                        "#bcbd22", None),
    ("PAPER_TT2",           "6. PAPER: two-regime, 4 chains × 1000+1000 (TT2)",           "#000000", PAPER_TT2_DIR),
]


def load_bcb() -> pd.DataFrame:
    df = pd.read_csv(BCB_CSV, skiprows=9, header=None)
    df = df.iloc[:, :6].rename(columns={0: "quarter", 1: "Low", 2: "p25",
                                        3: "p75", 4: "High", 5: "Reference"})
    df = df.dropna(subset=["quarter"])
    df["quarter"] = df["quarter"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["quarter"], format="%b-%y", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    for c in ("Low", "p25", "p75", "High", "Reference"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = df.index + pd.DateOffset(months=-2)   # quarter-end → quarter-start
    return df[["Low", "p25", "p75", "High", "Reference"]].sort_index()


def _smoothed_dir(exp_id: str) -> Path:
    """Map experiment id to its smoothed_states.npz parent dir.  Honors the
    optional override path stored as the 4th tuple element in EXPS."""
    for tag, _, _, override in EXPS:
        if tag == exp_id and override is not None:
            return override
    return OUT_DIR / f"model_a_{exp_id}"


def load_state(exp: str, name: str, which: str) -> pd.Series:
    """Pull one state by name from smoothed_states.npz of one experiment."""
    p = _smoothed_dir(exp) / "diagnostic" / "smoothed_states.npz"
    z = np.load(p, allow_pickle=True)
    state_names = list(z["state_names"])
    trend_names = list(z["trend_names"])
    n_stat = len(state_names)
    x = z[f"x_smooth_{which}"]
    dates = pd.to_datetime([str(d) for d in z["dates"]])
    if name in state_names:
        idx = state_names.index(name)
        return pd.Series(np.asarray(x[:, idx]), index=dates)
    if name in trend_names:
        idx = n_stat + trend_names.index(name)
        return pd.Series(np.asarray(x[:, idx]), index=dates)
    raise KeyError(name)


def plot_gap(which: str) -> Path:
    bcb = load_bcb()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(bcb.index, bcb["Low"], bcb["High"], color="#cccccc",
                    alpha=0.50, label="BCB Low–High envelope")
    ax.fill_between(bcb.index, bcb["p25"], bcb["p75"], color="#999999",
                    alpha=0.45, label="BCB 25th–75th")
    for exp, label, color, _override in EXPS:
        s = load_state(exp, "GDP_GAP", which)
        ax.plot(s.index, s.values, color=color, lw=1.8, label=label)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_title(f"Model A — GDP_GAP (posterior {which.upper()}) across TT specs",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("% of potential")
    ax.set_xlabel("Year")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    suffix = "" if which == "mean" else "_median"
    p = OUT_DIR / f"gap_vs_bcb{suffix}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_tt_decomp(which: str = "mean") -> Path:
    df = pd.read_csv(DATA_CSV, index_col=0)
    df.index = pd.to_datetime(df.index)
    obs_tt    = df["OBS_TT"]
    tt_bar_hp = df["TT_BAR"]   # HP-trend of 100*log(IC_Br_USD)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # --- TT_BAR (trend) panel ---
    ax1.plot(obs_tt.index, obs_tt.values, color="black", lw=1.0,
             alpha=0.55, label="OBS_TT (100·log IC_Br_USD)")
    ax1.plot(tt_bar_hp.index, tt_bar_hp.values, color="#888888", lw=1.2,
             ls="--", label="HP-trend (λ=1600)")
    for exp, label, color, _override in EXPS:
        s = load_state(exp, "TT_BAR", which)
        ax1.plot(s.index, s.values, color=color, lw=1.6, label=label)
    ax1.set_title(f"TT_BAR (trend) — Model A, posterior {which.upper()}",
                  fontsize=11, fontweight="bold")
    ax1.set_ylabel("100·log level")
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # --- TT_GAP (cycle) panel ---
    for exp, label, color, _override in EXPS:
        s = load_state(exp, "TT_GAP", which)
        ax2.plot(s.index, s.values, color=color, lw=1.6, label=label)
    ax2.axhline(0, color="black", lw=0.4)
    ax2.set_title(f"TT_GAP (cycle) — Model A, posterior {which.upper()}",
                  fontsize=11, fontweight="bold")
    ax2.set_ylabel("pp")
    ax2.set_xlabel("Year")
    ax2.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator(3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    p = OUT_DIR / "tt_decomposition.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_lz_decomp(which: str = "mean") -> Path:
    """Plot LZ_BAR (trend) and LZ_GAP (cycle) across the 4 TT specs.

    Helps diagnose the joint TT/LZ identification: anchoring TT_BAR
    re-balances the LZ_BAR decomposition through the cycle equations.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    for exp, label, color, _override in EXPS:
        lz_bar = load_state(exp, "LZ_BAR", which)
        lz_gap = load_state(exp, "LZ_GAP", which)
        ax1.plot(lz_bar.index, lz_bar.values, color=color, lw=1.6, label=label)
        ax2.plot(lz_gap.index, lz_gap.values, color=color, lw=1.6, label=label)
    ax1.axhline(0, color="black", lw=0.4)
    ax1.set_title(f"LZ_BAR — real-exchange-rate trend (posterior {which.upper()})",
                  fontsize=11, fontweight="bold")
    ax1.set_ylabel("100·log dev. from 2004Q1 base")
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0, color="black", lw=0.4)
    ax2.set_title(f"LZ_GAP — real-exchange-rate cycle (posterior {which.upper()})",
                  fontsize=11, fontweight="bold")
    ax2.set_ylabel("pp (negative = BRL stronger than trend)")
    ax2.set_xlabel("Year")
    ax2.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator(3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    p = OUT_DIR / "lz_decomposition.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    p1 = plot_gap("mean")
    p2 = plot_gap("median")
    p3 = plot_tt_decomp("mean")
    p4 = plot_lz_decomp("mean")
    print(f"Saved {p1}")
    print(f"Saved {p2}")
    print(f"Saved {p3}")
    print(f"Saved {p4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

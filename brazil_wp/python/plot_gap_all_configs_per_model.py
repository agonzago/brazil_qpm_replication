"""For each model (A, C, E, F), produce ONE figure showing the smoothed
GDP_GAP (posterior median) across all 6 configs, overlaid on Brazil's
output-gap envelope (High / Low / Reference scenario = median).

Uses the smoothed_states.npz files already saved at
results_smoke/<model>_<cfg>/diagnostic/smoothed_states.npz — no
re-estimation needed.

Outputs:
  results_smoke/bcb_compare/gap_all_configs_<model>.png  (one per model)
  results_smoke/bcb_compare/gap_all_configs_grid.png     (2×2 grid)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

THIS_DIR = Path(__file__).resolve().parent
SMOKE = THIS_DIR / "results_smoke"
BCB_CSV = THIS_DIR / "data" / "BCB_outputgap_Garf2.2.8.csv"
OUT_DIR = SMOKE / "bcb_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["A", "B", "C", "D", "E", "F"]
MODEL_DIRNAME = {"A": "model_a", "B": "model_b", "C": "model_c", "D": "model_d", "E": "model_e", "F": "model_f"}
CONFIGS = ["TLPL", "TLPT", "TTPL", "TTPT", "TTPT_p15", "TTPT_c5"]
CONFIG_COLOR = {
    "TLPL":     "#1f77b4",   # blue
    "TLPT":     "#ff7f0e",   # orange
    "TTPL":     "#2ca02c",   # green
    "TTPT":     "#d62728",   # red
    "TTPT_p15": "#9467bd",   # purple
    "TTPT_c5":  "#8c564b",   # brown
}
CONFIG_DESC = {
    "TLPL":     "TT loose / PREM loose",
    "TLPT":     "TT loose / PREM α=20",
    "TTPL":     "TT tight / PREM loose",
    "TTPT":     "TT tight / PREM α=20",
    "TTPT_p15": "TT tight / PREM α=20, SNR=15",
    "TTPT_c5":  "TT tight cap=5 / PREM α=20",
}


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
    # BCB dates are quarter-end (e.g., Mar-04 = 2004-03-01). Shift to quarter-start.
    df.index = df.index + pd.DateOffset(months=-2)
    return df[["Low", "p25", "p75", "High", "Reference"]].sort_index()


def load_gap(model: str, cfg: str, which: str = "median") -> pd.Series | None:
    """Read GDP_GAP at posterior MEDIAN (default) or MEAN from smoothed_states.npz."""
    p = SMOKE / f"{MODEL_DIRNAME[model]}_{cfg}" / "diagnostic" / "smoothed_states.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    state_names = list(z["state_names"])
    if "GDP_GAP" not in state_names:
        return None
    idx = state_names.index("GDP_GAP")
    x = z[f"x_smooth_{which}"]
    dates = pd.to_datetime([str(d) for d in z["dates"]])
    return pd.Series(np.asarray(x[:, idx]), index=dates)


def plot_one_model(model: str, bcb: pd.DataFrame, ax, which: str = "median") -> None:
    common = None
    # BCB envelope (filled). The Reference scenario line was intentionally
    # dropped — it is a conservative outlier (HP-trend-style methodology
    # that holds 2014-19 in negative territory) and disagrees with most of
    # BCB's own internal methodologies (the 75th percentile is positive
    # through 2019).  The Low-High range remains as the BCB-published
    # envelope across all their methods.
    ax.fill_between(bcb.index, bcb["Low"], bcb["High"], color="#cccccc",
                    alpha=0.45, label="BCB Low–High envelope")
    # Overlay each config
    for cfg in CONFIGS:
        gap = load_gap(model, cfg, which=which)
        if gap is None:
            continue
        c = CONFIG_COLOR[cfg]
        ax.plot(gap.index, gap.values, color=c, lw=1.5,
                label=f"{cfg} — {CONFIG_DESC[cfg]}")
        if common is None:
            common = gap.index
    ax.axhline(0, color="black", lw=0.4)
    ax.set_title(f"Model {model} — GDP_GAP (posterior {which}) across all 6 configs",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("% of potential")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9, ncol=2)


def main() -> int:
    bcb = load_bcb()
    for which in ("median", "mean"):
        # Grid figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
        for ax, model in zip(axes.flat, MODELS):
            plot_one_model(model, bcb, ax, which=which)
        fig.suptitle(f"GDP_GAP across configurations vs BCB output-gap envelope  "
                     f"(posterior {which.upper()})",
                     fontsize=14, fontweight="bold", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = OUT_DIR / f"gap_all_configs_grid_{which}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {p}")
        # Per-model standalone
        for model in MODELS:
            fig, ax = plt.subplots(figsize=(13, 5.5))
            plot_one_model(model, bcb, ax, which=which)
            ax.set_xlabel("Year")
            fig.tight_layout()
            p = OUT_DIR / f"gap_all_configs_{model}_{which}.png"
            fig.savefig(p, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

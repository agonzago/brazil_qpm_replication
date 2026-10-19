"""Build the v5 BCB-envelope exhibits (Master Inventory §6.5).

Outputs:
  results_v2_diffuse_v5/_tables/bcb_coverage_v5.csv     (per-spec coverage stats)
  results_v2_diffuse_v5/_figures/gap_vs_bcb_v5.png      (cross-spec overlay, posterior mean)
  results_v2_diffuse_v5/_figures/gap_vs_bcb_v5_median.png (same, posterior median)

Reads ``smoothed_states.npz`` from each per-spec diagnostic/ subdirectory
(must already exist; produced by save_all_smoothed_states_v5.sh).
Compares the posterior median/mean GDP_GAP to the BCB output-gap envelope
(``data/BCB_outputgap_Garf2.2.8.csv``).
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
V5       = THIS_DIR / "results_v2_diffuse_v5"
BCB_CSV  = THIS_DIR / "data" / "BCB_outputgap_Garf2.2.8.csv"
TABLES   = V5 / "_tables"
FIGURES  = V5 / "_figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Body specifications and RUBs, in the order they appear in the inventory.
# Subdirectories are the v5+ free_kappa result trees (headline treatment).
SPECS = [
    ("A",    "model_a_v5plus_free_kappa",   "Model A (full Koyck)",           "#1f77b4"),
    ("B",    "model_b_v5plus_free_kappa",   "Model B (no QRR)",               "#ff7f0e"),
    ("C",    "model_c_v5plus_free_kappa",   "Model C (baseline)",             "#2ca02c"),
    ("D",    "model_d_v5plus_free_kappa",   "Model D (no Koyck + 1Q Taylor)", "#d62728"),
    ("E",    "model_e_v5plus_free_kappa",   "Model E (COVID-mask)",           "#9467bd"),
    ("RUB1", "rub1_c_v5plus_free_kappa",    "RUB1 (wider priors)",            "#8c564b"),
    ("RUB3", "rub3_c_v5plus_free_kappa",    "RUB3 (weak Taylor)",             "#e377c2"),
    ("RUB4", "rub4_c_v5plus_free_kappa",    "RUB4 (alpha_pi=0)",              "#7f7f7f"),
    ("RUB5", "rub5_c_v5plus_free_kappa",    "RUB5 (Focus IPCA)",              "#bcbd22"),
]

# Specs to show in the cross-spec figure (body only, RUBs would crowd it).
FIGURE_SPECS = ["A", "B", "C", "D", "E"]


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
    # BCB dates are quarter-end (Mar/Jun/Sep/Dec); shift -2 months -> quarter-start.
    df.index = df.index + pd.DateOffset(months=-2)
    df["mid"] = (df["Low"] + df["High"]) / 2.0
    return df.sort_index()


def load_gap(subdir: str, which: str) -> pd.Series | None:
    p = V5 / subdir / "diagnostic" / "smoothed_states.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    sn = list(z["state_names"])
    if "GDP_GAP" not in sn:
        return None
    x = z[f"x_smooth_{which}"]
    dates = pd.to_datetime([str(d) for d in z["dates"]])
    return pd.Series(np.asarray(x[:, sn.index("GDP_GAP")]), index=dates)


def coverage_stats(g: pd.Series, bcb: pd.DataFrame) -> dict:
    common = g.index.intersection(bcb.index)
    gc = g.loc[common]; bc = bcb.loc[common]
    no_covid = (common < pd.Timestamp("2020-01-01")) | (common >= pd.Timestamp("2022-01-01"))
    gc_nc = gc.loc[no_covid]; bc_nc = bc.loc[no_covid]
    return {
        "n_quarters":          len(common),
        "in_low_high_pct":     float(((gc >= bc["Low"]) & (gc <= bc["High"])).mean() * 100),
        "in_iqr_pct":          float(((gc >= bc["p25"]) & (gc <= bc["p75"])).mean() * 100),
        "in_low_high_pct_noCOVID": float(((gc_nc >= bc_nc["Low"]) & (gc_nc <= bc_nc["High"])).mean() * 100),
        "mae_vs_mid":          float((gc - bc["mid"]).abs().mean()),
        "rmse_vs_mid":         float(np.sqrt(((gc - bc["mid"]) ** 2).mean())),
        "mean_gap":            float(gc.mean()),
        "mean_gap_noCOVID":    float(gc_nc.mean()),
        "abs_max_gap":         float(gc.abs().max()),
    }


def build_coverage_table() -> Path:
    bcb = load_bcb()
    rows = []
    for label, sub, _, _ in SPECS:
        for which in ("mean", "median"):
            g = load_gap(sub, which)
            if g is None:
                print(f"  skip {label}/{which}: smoothed_states.npz missing")
                continue
            d = coverage_stats(g, bcb)
            d.update({"spec": label, "subdir": sub, "summary": which})
            rows.append(d)
    out = TABLES / "bcb_coverage_v5.csv"
    cols = ["spec", "subdir", "summary", "n_quarters",
            "in_low_high_pct", "in_low_high_pct_noCOVID", "in_iqr_pct",
            "mae_vs_mid", "rmse_vs_mid", "mean_gap", "mean_gap_noCOVID",
            "abs_max_gap"]
    pd.DataFrame(rows)[cols].to_csv(out, index=False, float_format="%.3f")
    print(f"  {out}: {len(rows)} rows")
    return out


def plot_cross_spec(which: str) -> Path:
    bcb = load_bcb()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(bcb.index, bcb["Low"], bcb["High"], color="#cccccc",
                    alpha=0.50, label="BCB Low-High envelope")
    ax.fill_between(bcb.index, bcb["p25"], bcb["p75"], color="#999999",
                    alpha=0.45, label="BCB 25th-75th")

    spec_idx = {label: (sub, name, color) for label, sub, name, color in SPECS}
    for label in FIGURE_SPECS:
        sub, name, color = spec_idx[label]
        g = load_gap(sub, which)
        if g is None:
            continue
        ax.plot(g.index, g.values, color=color, lw=1.6, label=name)

    ax.axhline(0, color="black", lw=0.4)
    ax.set_title(f"Model GDP gap across body specifications vs BCB envelope  "
                 f"(posterior {which.upper()})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("% of potential")
    ax.set_xlabel("Year")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9, ncol=2)
    fig.tight_layout()
    suffix = "" if which == "mean" else "_median"
    p = FIGURES / f"gap_vs_bcb{suffix}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    print(f"Building v5 BCB exhibits under {V5}")
    build_coverage_table()
    for which in ("mean", "median"):
        p = plot_cross_spec(which)
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

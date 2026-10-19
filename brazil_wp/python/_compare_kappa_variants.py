"""3-variant side-by-side comparison: kappa_g_tt baseline vs free vs zero.

Compares the three Model-A v5+ specifications:
  - baseline     : kappa_g_tt = 0.02 fixed (v5+ headline)
  - free_kappa   : kappa_g_tt estimated   (Gamma prior, mode 0.020)
  - no_coupling  : kappa_g_tt = 0 fixed   (TT-trend -> G channel off)

Output: results_v2_diffuse_v5/_kappa_variants_compare/kappa_variants_compare.png
        and  comparison_table.csv / summary.md
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RES = BASE / "results_v2_diffuse_v5"
OUT = RES / "_kappa_variants_compare"
OUT.mkdir(parents=True, exist_ok=True)
BCB_CSV = BASE / "data" / "BCB_outputgap_Garf2.2.8.csv"
DATA = BASE / "data" / "estimation_data.csv"

RUNS = [
    ("baseline (k=0.02 fixed)",
     RES / "model_a_v5plus"),
    ("free_kappa (k estimated ~0.034)",
     RES / "model_a_v5plus_free_kappa"),
    ("no_coupling (k=0)",
     RES / "model_a_v5plus_no_coupling"),
]


def load_run(label, p):
    npz = p / "diagnostic/smoothed_states.npz"
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    sn = list(d["state_names"]); tn = list(d["trend_names"])
    dates = pd.to_datetime([str(s) for s in d["dates"]])
    x = np.asarray(d.get("x_smooth_mean", d.get("x_smooth")))
    def g(name):
        if name in sn: return pd.Series(x[:, sn.index(name)], index=dates)
        if name in tn: return pd.Series(x[:, len(sn)+tn.index(name)], index=dates)
        return None
    return {"label": label, "dir": p, "dates": dates, "g": g}


def load_bcb():
    df = pd.read_csv(BCB_CSV, skiprows=9, header=None).iloc[:, :6]
    df.columns = ["q","Low","p25","p75","High","Reference"]
    df["date"] = pd.to_datetime(df["q"].astype(str).str.strip(),
                                format="%b-%y", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    df.index = df.index.to_period("Q").start_time
    for c in ["Low","p25","p75","High","Reference"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()


def metric_row(run, obs_tt_data, bcb):
    g = run["g"]
    row = {"variant": run["label"]}
    rr_bar = g("RR_BAR"); g_tr = g("G_TREND"); gdp_gap = g("GDP_GAP")
    lz_bar = g("LZ_BAR"); lz_gap = g("LZ_GAP")
    if rr_bar is not None:
        row["RR_BAR end"]  = float(rr_bar.iloc[-1])
        row["RR_BAR mean"] = float(rr_bar.mean())
    if g_tr is not None:
        row["G_TREND range (ann pp)"] = float(4 * (g_tr.max() - g_tr.min()))
        row["G_TREND end (ann %)"]    = float(4 * g_tr.iloc[-1])
    if gdp_gap is not None:
        a = pd.concat([gdp_gap.rename("gap"), bcb], axis=1).dropna(subset=["gap"])
        aa = a.dropna(subset=["Low", "High"])
        row["BCB cov (%)"]    = float(((aa["gap"] >= aa["Low"])
                                       & (aa["gap"] <= aa["High"])).mean() * 100)
        ar = a.dropna(subset=["Reference"])
        row["MAE vs BCB Ref"] = float((ar["gap"] - ar["Reference"]).abs().mean())
    if lz_gap is not None:
        row["LZ_GAP mean"]      = float(lz_gap.mean())
        row["LZ_GAP share<0 %"] = float((lz_gap < 0).mean() * 100)
    return row


def main() -> int:
    df_data = pd.read_csv(DATA, index_col=0)
    df_data.index = pd.PeriodIndex(df_data.index, freq="Q").to_timestamp()
    obs_tt = df_data["OBS_TT"]
    bcb = load_bcb()

    runs = [load_run(lab, p) for lab, p in RUNS]
    runs = [r for r in runs if r is not None]

    rows = [metric_row(r, obs_tt, bcb) for r in runs]
    table = pd.DataFrame(rows)
    table_path = OUT / "comparison_table.csv"
    table.to_csv(table_path, index=False)
    summary_md = OUT / "summary.md"
    with open(summary_md, "w") as f:
        f.write("# 3-variant comparison: kappa_g_tt baseline vs free vs zero\n\n")
        f.write(table.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # 4-row figure: LZ, DA_LZ, G_TREND, GDP_GAP vs BCB
    n = len(runs)
    fig, axes = plt.subplots(4, n, figsize=(4.5 * n, 13), sharex="col")
    common_idx = pd.date_range("2004-01-01", "2026-01-01", freq="QS")
    aligned_bcb = bcb.reindex(common_idx)

    for j, r in enumerate(runs):
        g = r["g"]
        lz_bar = g("LZ_BAR"); lz_gap = g("LZ_GAP")
        g_tr = g("G_TREND"); gdp_gap = g("GDP_GAP")

        # Row 1: LZ + LZ_BAR
        ax = axes[0, j]
        if lz_bar is not None and lz_gap is not None:
            lz = lz_bar + lz_gap
            ax.plot(lz.index, lz.values, "k-", lw=1.0, alpha=0.7, label="LZ")
            ax.plot(lz_bar.index, lz_bar.values, "r--", lw=1.4, label="LZ_BAR")
            ax.set_title(r["label"], fontsize=10)
            ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel("REER (100*log)")

        # Row 2: DA_LZ + DA_LZ_BAR
        ax = axes[1, j]
        if lz_bar is not None and lz_gap is not None:
            lz = lz_bar + lz_gap
            ax.plot(lz.index, (4 * lz.diff()).values, "k-", lw=0.9, alpha=0.8,
                    label="DA_LZ")
            ax.plot(lz_bar.index, (4 * lz_bar.diff()).values, "r--", lw=1.4,
                    label="DA_LZ_BAR")
            ax.axhline(0, color="grey", lw=0.5)
            ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel("4*ΔREER (ann pp)")

        # Row 3: G_TREND annualised
        ax = axes[2, j]
        if g_tr is not None:
            ax.plot(g_tr.index, 4 * g_tr.values, "C0-", lw=1.4,
                    label="potential growth (ann %)")
            ax.axhline(2.0, color="grey", lw=0.5, ls=":")
            ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel("4*G_TREND (ann %)")

        # Row 4: GDP_GAP vs BCB
        ax = axes[3, j]
        ax.fill_between(aligned_bcb.index, aligned_bcb["Low"],
                        aligned_bcb["High"], color="lightgray", alpha=0.7,
                        label="BCB Low-High")
        ax.plot(aligned_bcb.index, aligned_bcb["Reference"], "k--",
                lw=1.0, alpha=0.7, label="BCB Reference")
        if gdp_gap is not None:
            gg = gdp_gap.reindex(common_idx)
            ax.plot(gg.index, gg.values, "C0-", lw=1.4, label="model")
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("GDP gap (%)")

    fig.suptitle("Model A v5+ — three kappa_g_tt variants "
                 "(baseline / free / no-coupling)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = OUT / "kappa_variants_compare.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved table  -> {table_path}")
    print(f"Saved md     -> {summary_md}")
    print(f"Saved figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

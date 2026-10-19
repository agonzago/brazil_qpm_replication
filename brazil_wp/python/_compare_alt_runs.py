"""Side-by-side comparison of three TT-trend MCMC runs.

Loads smoothed_states.npz from each results dir and computes:
  - LZ_GAP mean / median / share<0 / min / max
  - RR_BAR end-of-sample, mean
  - G_TREND annualized range
  - TT_BAR / OBS_TT range ratio
  - LZ_BAR / LZ range ratio
  - DA_LZ_BAR mean (annualized trend REER depreciation)
  - BCB envelope coverage (output gap)
  - MAE vs BCB Reference
  - Posterior-mean LL

Builds the comparison table and a 3-row figure showing:
  Row 1: LZ vs LZ_BAR across runs
  Row 2: DA_LZ vs DA_LZ_BAR across runs
  Row 3: GDP_GAP vs BCB envelope across runs

Writes outputs to results_v2_diffuse_v5/_alt_runs_compare/.

Usage:
    python _compare_alt_runs.py
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
OUT = RES / "_alt_runs_compare"
OUT.mkdir(parents=True, exist_ok=True)
BCB_CSV = BASE / "data" / "BCB_outputgap_Garf2.2.8.csv"
DATA = BASE / "data" / "estimation_data.csv"

RUNS = [
    # Legacy opt1/opt2 smokes (kept for back-comparison).
    ("opt1 (d_lz_ss=0.50, nu=0.0)  *legacy*",
     RES / "model_a_opt1_nudlz0_smoke"),
    # v5+ paper-quality runs (TT-trend headline)
    ("Model A   v5+ (tt_trend)", RES / "model_a_v5plus"),
    ("Model A0  v5+ (tt_trend)", RES / "model_a0_v5plus"),
    ("Model B   v5+ (tt_trend)", RES / "model_b_v5plus"),
    ("Model C   v5+ (tt_trend)", RES / "model_c_v5plus"),
    ("Model D   v5+ (tt_trend)", RES / "model_d_v5plus"),
    ("Model E   v5+ (tt_trend)", RES / "model_e_v5plus"),
    ("RUB1      v5+ (wider priors)",        RES / "rub1_v5plus"),
    ("RUB3      v5+ (weaker Taylor prior)", RES / "rub3_v5plus"),
    ("RUB4      v5+ (alpha_pi=0)",          RES / "rub4_v5plus"),
    ("RUB5      v5+ (Focus IPCA)",          RES / "rub5_v5plus"),
]


def load_run(label, p):
    npz = p / "diagnostic/smoothed_states.npz"
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    sn = list(d["state_names"]); tn = list(d["trend_names"])
    dates = pd.to_datetime([str(s) for s in d["dates"]])
    x = np.asarray(d.get("x_smooth_mean", d.get("x_smooth")))
    obs_s = (np.asarray(d["obs_smooth_mean"])
             if "obs_smooth_mean" in d.files else None)
    on = list(d["obs_names"])
    def g(name):
        if name in sn: return pd.Series(x[:, sn.index(name)], index=dates)
        if name in tn: return pd.Series(x[:, len(sn)+tn.index(name)], index=dates)
        return None
    return {"label": label, "dir": p, "dates": dates,
            "g": g, "obs_s": obs_s, "obs_names": on}


def metric_row(run, obs_tt_data, obs_lgdp_data, bcb):
    g = run["g"]
    tt_bar = g("TT_BAR"); lz_bar = g("LZ_BAR"); lz_gap = g("LZ_GAP")
    g_tr = g("G_TREND"); lgdp_bar = g("LGDP_BAR"); rr_bar = g("RR_BAR")
    gdp_gap = g("GDP_GAP")
    row = {"name": run["label"]}
    if lz_gap is not None:
        row["LZ_GAP mean"]   = float(lz_gap.mean())
        row["LZ_GAP share<0"] = float((lz_gap < 0).mean()*100)
    if rr_bar is not None:
        row["RR_BAR end"] = float(rr_bar.iloc[-1])
        row["RR_BAR mean"] = float(rr_bar.mean())
    if lz_bar is not None and lz_gap is not None:
        lz = lz_bar + lz_gap
        row["LZ_BAR/LZ range"] = float((lz_bar.max()-lz_bar.min()) /
                                       (lz.max()-lz.min()))
        da_lz_bar = 4 * lz_bar.diff()
        row["DA_LZ_BAR mean"] = float(da_lz_bar.mean())
    if tt_bar is not None and obs_tt_data is not None:
        common = obs_tt_data.index.intersection(tt_bar.index)
        row["TT_BAR/OBS_TT range"] = float(
            (tt_bar.reindex(common).max()-tt_bar.reindex(common).min()) /
            (obs_tt_data.reindex(common).max()-obs_tt_data.reindex(common).min()))
    if g_tr is not None:
        row["G_TREND range (ann)"] = float(4*(g_tr.max()-g_tr.min()))
    if lgdp_bar is not None and obs_lgdp_data is not None:
        common = obs_lgdp_data.index.intersection(lgdp_bar.index)
        row["LGDP_BAR/OBS_LGDP range"] = float(
            (lgdp_bar.reindex(common).max()-lgdp_bar.reindex(common).min()) /
            (obs_lgdp_data.reindex(common).max()-obs_lgdp_data.reindex(common).min()))
    if gdp_gap is not None:
        a = pd.concat([gdp_gap.rename("gap"), bcb], axis=1).dropna(subset=["gap"])
        aa = a.dropna(subset=["Low","High"])
        row["BCB coverage"] = float(((aa["gap"]>=aa["Low"]) & (aa["gap"]<=aa["High"])).mean())
        ar = a.dropna(subset=["Reference"])
        row["MAE vs Ref"] = float((ar["gap"] - ar["Reference"]).abs().mean())
    return row


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


def main() -> int:
    df_data = pd.read_csv(DATA, index_col=0)
    df_data.index = pd.PeriodIndex(df_data.index, freq="Q").to_timestamp()
    obs_tt = df_data["OBS_TT"]
    obs_lgdp = df_data["OBS_LGDP"]
    bcb = load_bcb()

    runs = [load_run(lab, p) for lab, p in RUNS]
    runs = [r for r in runs if r is not None]

    rows = [metric_row(r, obs_tt, obs_lgdp, bcb) for r in runs]
    table = pd.DataFrame(rows)
    table_path = OUT / "comparison_table.csv"
    table.to_csv(table_path, index=False)

    summary_md = OUT / "summary.md"
    with open(summary_md, "w") as f:
        f.write("# Alt-run comparison\n\n")
        f.write(table.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # 3-row comparison figure
    n = len(runs)
    fig, axes = plt.subplots(3, n, figsize=(4.2*n, 10), sharex="col")
    common_idx = pd.date_range("2004-01-01", "2026-01-01", freq="QS")
    aligned_bcb = bcb.reindex(common_idx)
    for j, r in enumerate(runs):
        g = r["g"]
        # Row 1: LZ + LZ_BAR
        ax = axes[0, j] if n > 1 else axes[0]
        lz_bar = g("LZ_BAR"); lz_gap = g("LZ_GAP")
        if lz_bar is not None and lz_gap is not None:
            lz = lz_bar + lz_gap
            ax.plot(lz.index, lz.values, "k-", lw=1.0, alpha=0.7, label="LZ")
            ax.plot(lz_bar.index, lz_bar.values, "r--", lw=1.4, label="LZ_BAR")
            ax.set_title(r["label"], fontsize=9)
            ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3)
            ax.set_ylabel("REER level" if j == 0 else "")
        # Row 2: DA_LZ + DA_LZ_BAR
        ax = axes[1, j] if n > 1 else axes[1]
        if lz_bar is not None and lz_gap is not None:
            lz = lz_bar + lz_gap
            ax.plot(lz.index, (4*lz.diff()).values, "k-", lw=0.9, alpha=0.8, label="DA_LZ")
            ax.plot(lz_bar.index, (4*lz_bar.diff()).values, "r--", lw=1.4, label="DA_LZ_BAR")
            ax.axhline(0, color="grey", lw=0.5)
            ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3)
            ax.set_ylabel("4·ΔREER (pp/yr)" if j == 0 else "")
        # Row 3: GDP_GAP vs BCB
        ax = axes[2, j] if n > 1 else axes[2]
        ax.fill_between(aligned_bcb.index, aligned_bcb["Low"],
                        aligned_bcb["High"], color="lightgray", alpha=0.7)
        ax.plot(aligned_bcb.index, aligned_bcb["Reference"], "k--",
                lw=1.0, alpha=0.7)
        gdp_gap = g("GDP_GAP")
        if gdp_gap is not None:
            gg = gdp_gap.reindex(common_idx)
            ax.plot(gg.index, gg.values, "C0-", lw=1.4, label="model")
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3)
        ax.set_ylabel("GDP gap" if j == 0 else "")

    fig.tight_layout()
    fig_path = OUT / "alt_runs_compare.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved table  -> {table_path}")
    print(f"Saved md     -> {summary_md}")
    print(f"Saved figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

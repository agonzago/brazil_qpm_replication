"""All v5+ body specs + RUBs: kappa_g_tt sensitivity (3-way comparison).

Builds the headline summary table comparing the TT-trend coupling treatments
- kappa_g_tt = 0    (no coupling)
- kappa_g_tt = 0.02 (fixed at calibrated value)
- kappa_g_tt free   (estimated under tight Gamma prior)
across the 6 body specs (A/A0/B/C/D/E) and the 4 robustness checks
(RUB1/3/4/5).

Output:
  results_v2_diffuse_v5/_coupling_compare/comparison_table.csv
  results_v2_diffuse_v5/_coupling_compare/summary.md
  results_v2_diffuse_v5/_coupling_compare/bcb_coverage_bars.png
  results_v2_diffuse_v5/_coupling_compare/gap_grid_by_spec.png
  results_v2_diffuse_v5/_coupling_compare/kappa_posteriors.png
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
OUT = RES / "_coupling_compare"
OUT.mkdir(parents=True, exist_ok=True)
BCB_CSV = BASE / "data" / "BCB_outputgap_Garf2.2.8.csv"

SPECS = [
    ("Model A",  "model_a"),
    ("Model B",  "model_b"),
    ("Model C",  "model_c"),
    ("Model D",  "model_d"),
    ("Model E",  "model_e"),
    ("RUB1",     "rub1_c"),
    ("RUB3",     "rub3_c"),
    ("RUB4",     "rub4_c"),
    ("RUB5",     "rub5_c"),
]


def _load_smoothed(p: Path):
    npz = p / "diagnostic" / "smoothed_states.npz"
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
    return {"g": g}


def _load_bcb() -> pd.DataFrame:
    df = pd.read_csv(BCB_CSV, skiprows=9, header=None).iloc[:, :6]
    df.columns = ["q", "Low", "p25", "p75", "High", "Reference"]
    df["date"] = pd.to_datetime(df["q"].astype(str).str.strip(),
                                format="%b-%y", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    df.index = df.index.to_period("Q").start_time
    for c in ("Low", "p25", "p75", "High", "Reference"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()


def _rhat_ess(p: Path):
    s = p / "estimation" / "arviz_summary.csv"
    if not s.exists():
        return None, None
    df = pd.read_csv(s, index_col=0)
    return float(df.r_hat.max()), float(df.ess_bulk.min())


def _spec_row(label: str, sub: str, variant: str, suffix: str, bcb):
    """variant in {'kappa=0.02', 'kappa=0'}; suffix is '_v5plus' or '_v5plus_no_coupling'."""
    p = RES / f"{sub}{suffix}"
    r = _load_smoothed(p)
    row = {"spec": label, "variant": variant}
    rhat, ess = _rhat_ess(p)
    row["max_rhat"] = rhat
    row["min_ess_bulk"] = ess
    if r is None:
        row.update({k: np.nan for k in
                    ("BCB_cov", "MAE_Ref", "RR_BAR_end", "G_TREND_range_ann",
                     "LZ_GAP_share_neg")})
        return row
    g = r["g"]
    rr_bar = g("RR_BAR"); g_tr = g("G_TREND"); gdp_gap = g("GDP_GAP")
    lz_gap = g("LZ_GAP")
    if rr_bar is not None:
        row["RR_BAR_end"] = float(rr_bar.iloc[-1])
    if g_tr is not None:
        row["G_TREND_range_ann"] = float(4*(g_tr.max() - g_tr.min()))
    if gdp_gap is not None:
        a = pd.concat([gdp_gap.rename("gap"), bcb], axis=1).dropna(subset=["gap"])
        aa = a.dropna(subset=["Low", "High"])
        row["BCB_cov"]  = float(((aa["gap"]>=aa["Low"]) & (aa["gap"]<=aa["High"])).mean()*100)
        ar = a.dropna(subset=["Reference"])
        row["MAE_Ref"]  = float((ar["gap"] - ar["Reference"]).abs().mean())
    if lz_gap is not None:
        row["LZ_GAP_share_neg"] = float((lz_gap<0).mean()*100)
    return row


def _kappa_posterior(p: Path) -> tuple[float | None, float | None]:
    """Return (mean, sd) of kappa_g_tt from arviz_summary.csv, or (None, None)."""
    s = p / "estimation" / "arviz_summary.csv"
    if not s.exists():
        return None, None
    df = pd.read_csv(s, index_col=0)
    if "kappa_g_tt" not in df.index:
        return None, None
    return float(df.loc["kappa_g_tt", "mean"]), float(df.loc["kappa_g_tt", "sd"])


def build_table() -> pd.DataFrame:
    bcb = _load_bcb()
    rows = []
    for label, sub in SPECS:
        for variant_label, suffix in (
            ("k=0    (TT-trend off)",       "_v5plus_no_coupling"),
            ("k free (TT-trend estimated)", "_v5plus_free_kappa"),
        ):
            row = _spec_row(label, sub, variant_label, suffix, bcb)
            k_mean, k_sd = _kappa_posterior(RES / f"{sub}{suffix}")
            row["kappa_mean"] = k_mean
            row["kappa_sd"]   = k_sd
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def plot_bcb_bars(df: pd.DataFrame) -> Path:
    """Bar chart: BCB coverage for each spec, 2 variants side-by-side."""
    fig, ax = plt.subplots(figsize=(13, 5))
    specs = df.spec.unique()
    x = np.arange(len(specs))
    w = 0.40
    off  = df[df.variant.str.contains("k=0  ", regex=False)].set_index("spec").reindex(specs)["BCB_cov"]
    free = df[df.variant.str.contains("k free")].set_index("spec").reindex(specs)["BCB_cov"]
    ax.bar(x - w/2, off.values,  w, label="kappa = 0    (TT-trend off)",        color="C0")
    ax.bar(x + w/2, free.values, w, label="kappa free (TT-trend estimated)",    color="C2")
    for i, (a, c) in enumerate(zip(off.values, free.values)):
        if not np.isnan(a): ax.text(i - w/2, a + 1, f"{a:.1f}", ha="center", fontsize=7)
        if not np.isnan(c): ax.text(i + w/2, c + 1, f"{c:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(specs, rotation=0, fontsize=9)
    ax.set_ylabel("BCB envelope coverage (%)")
    ax.set_title("kappa_g_tt sensitivity: BCB envelope coverage across specifications")
    ax.axhline(85, color="grey", lw=0.5, ls=":")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out = OUT / "bcb_coverage_bars.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_kappa_posteriors(df: pd.DataFrame) -> Path:
    """Error-bar chart of kappa_g_tt posterior (mean +/- sd) across 10 specs."""
    f = df[df.variant.str.contains("k free")].set_index("spec")
    specs = list(f.index)
    means = f["kappa_mean"].values
    sds   = f["kappa_sd"].values
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(specs))
    ax.errorbar(x, means, yerr=sds, fmt="o", color="C0",
                ecolor="C0", capsize=5, ms=8, label="posterior mean +/- sd")
    ax.axhline(0.020, color="grey", lw=1.0, ls="--",
               label="prior mode (0.020)")
    ax.axhline(0,     color="black", lw=0.5)
    for i, (m, s) in enumerate(zip(means, sds)):
        if not np.isnan(m):
            ax.text(i, m + s + 0.003, f"{m:.4f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(specs, rotation=0, fontsize=9)
    ax.set_ylabel("kappa_g_tt (TT-trend coupling on G)")
    ax.set_title("kappa_g_tt posterior across specifications "
                 "(estimated under Gamma prior, mode 0.020)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(0.07, np.nanmax(means + sds) * 1.2))
    fig.tight_layout()
    out = OUT / "kappa_posteriors.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_gap_grid() -> Path:
    """Body-spec x 2 grid: GDP_GAP vs BCB, two columns (k=0 / k free)."""
    bcb = _load_bcb()
    common = pd.date_range("2004-01-01", "2026-01-01", freq="QS")
    aligned = bcb.reindex(common)
    body_specs = [("Model A","model_a"),
                  ("Model B","model_b"),("Model C","model_c"),
                  ("Model D","model_d"),("Model E","model_e")]
    fig, axes = plt.subplots(len(body_specs), 2, figsize=(11, 14), sharex=True, sharey=True)
    columns = (
        ("_v5plus_no_coupling", "kappa = 0  (TT-trend off)",       "C0"),
        ("_v5plus_free_kappa",  "kappa free (TT-trend estimated)", "C2"),
    )
    for i, (label, sub) in enumerate(body_specs):
        for j, (suffix, vlbl, color) in enumerate(columns):
            ax = axes[i, j]
            ax.fill_between(aligned.index, aligned["Low"], aligned["High"],
                            color="lightgray", alpha=0.7, label="BCB Low-High")
            ax.plot(aligned.index, aligned["Reference"], "k--", lw=1.0, alpha=0.7)
            r = _load_smoothed(RES / f"{sub}{suffix}")
            if r is not None:
                gg = r["g"]("GDP_GAP")
                if gg is not None:
                    gg_r = gg.reindex(common)
                    ax.plot(common, gg_r.values, color + "-",
                            lw=1.4, label="model")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{label}  /  {vlbl}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel("GDP gap (%)")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Body specifications: GDP gap vs BCB envelope, by $\\kappa_{g,tt}$ treatment",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / "gap_grid_by_spec.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    df = build_table()
    df.to_csv(OUT / "comparison_table.csv", index=False)
    md = OUT / "summary.md"
    with open(md, "w") as f:
        f.write("# v5+ kappa_g_tt = 0.02  vs  kappa_g_tt = 0 — all paper specs\n\n")
        # Pivot for readability
        pivot = df.pivot_table(index="spec",
                               columns="variant",
                               values=["BCB_cov", "MAE_Ref", "RR_BAR_end",
                                       "G_TREND_range_ann", "max_rhat",
                                       "min_ess_bulk"],
                               sort=False)
        f.write(pivot.to_markdown(floatfmt=".3f"))
        f.write("\n\n## Long-format (raw)\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    bp = plot_bcb_bars(df)
    print(f"  -> {bp}")
    gp = plot_gap_grid()
    print(f"  -> {gp}")
    kp = plot_kappa_posteriors(df)
    print(f"  -> {kp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

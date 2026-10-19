"""Per-model 2-row comparison: v5 baseline vs v5+ (TT-trend headline).

Compares the v5 paper baseline output (`<spec>/`) with the v5+ TT-trend
output (`<spec>_v5plus/`) for Models A, A0, B, C, D, E.

Outputs (under results_v2_diffuse_v5/_figures/):
  v5_baseline_vs_v5plus_gap_by_model.png       (2 x 3 grid -- GDP_GAP vs BCB)
  v5_baseline_vs_v5plus_g_trend_by_model.png   (2 x 3 grid -- G_TREND annualised)
  v5_baseline_vs_v5plus_rr_bar_by_model.png    (2 x 3 grid -- RR_BAR)
  v5_baseline_vs_v5plus_lz_decomp_by_model.png (2 x 3 grid -- LZ vs LZ_BAR)

Per model, the top row shows the v5 baseline (kept for paper comparison);
the bottom row shows the v5+ headline (TT-trend coupling).
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
OUT = RES / "_figures"
OUT.mkdir(parents=True, exist_ok=True)
BCB_CSV = BASE / "data" / "BCB_outputgap_Garf2.2.8.csv"
DATA_CSV = BASE / "data" / "estimation_data.csv"

MODELS = [
    ("Model A",  "model_a",  "model_a_v5plus_free_kappa"),
    ("Model A0", "model_a0", "model_a0_v5plus_free_kappa"),
    ("Model B",  "model_b",  "model_b_v5plus_free_kappa"),
    ("Model C",  "model_c",  "model_c_v5plus_free_kappa"),
    ("Model D",  "model_d",  "model_d_v5plus_free_kappa"),
    ("Model E",  "model_e",  "model_e_v5plus_free_kappa"),
]


def _load_smoothed(p: Path) -> dict | None:
    """Load smoothed_states.npz, returning a dict with name-indexed accessors."""
    npz = p / "diagnostic" / "smoothed_states.npz"
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    sn = list(d["state_names"])
    tn = list(d["trend_names"])
    dates = pd.to_datetime([str(s) for s in d["dates"]])
    x = np.asarray(d.get("x_smooth_mean", d.get("x_smooth")))

    def g(name: str) -> pd.Series | None:
        if name in sn:
            return pd.Series(x[:, sn.index(name)], index=dates)
        if name in tn:
            return pd.Series(x[:, len(sn) + tn.index(name)], index=dates)
        return None

    return {"dates": dates, "g": g}


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


def plot_gap_by_model() -> Path:
    bcb = _load_bcb()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True, sharey=True)
    common = pd.date_range("2004-01-01", "2026-01-01", freq="QS")
    aligned_bcb = bcb.reindex(common)

    for idx, (label, base, plus) in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        ax.fill_between(aligned_bcb.index, aligned_bcb["Low"],
                        aligned_bcb["High"], color="lightgray", alpha=0.7,
                        label="BCB Low-High")
        ax.plot(aligned_bcb.index, aligned_bcb["Reference"], "k--",
                lw=1.0, alpha=0.7, label="BCB Reference")
        ax.axhline(0, color="grey", lw=0.5)

        r_base = _load_smoothed(RES / base)
        r_plus = _load_smoothed(RES / plus)
        if r_base is not None:
            gg = r_base["g"]("GDP_GAP")
            if gg is not None:
                ax.plot(gg.index, gg.values, "C1-", lw=1.2,
                        alpha=0.85, label="v5 baseline")
        if r_plus is not None:
            gg = r_plus["g"]("GDP_GAP")
            if gg is not None:
                ax.plot(gg.index, gg.values, "C0-", lw=1.6,
                        label="v5+ (TT-trend)")

        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower left")
        if idx % 3 == 0:
            ax.set_ylabel("GDP gap (%)")

    fig.suptitle("v5 baseline vs v5+ (TT-trend coupling): GDP gap vs BCB envelope",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "v5_baseline_vs_v5plus_gap_by_model.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")
    return out


def plot_g_trend_by_model() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True, sharey=True)
    for idx, (label, base, plus) in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        ax.axhline(2.0, color="grey", lw=0.5, ls=":")  # 2% ss potential
        for r_dir, color, lw, lbl in (
            (RES / base, "C1", 1.2, "v5 baseline"),
            (RES / plus, "C0", 1.6, "v5+ (TT-trend)"),
        ):
            r = _load_smoothed(r_dir)
            if r is None:
                continue
            g_tr = r["g"]("G_TREND")
            if g_tr is None:
                continue
            ax.plot(g_tr.index, 4 * g_tr.values, color + "-", lw=lw,
                    alpha=0.85, label=lbl)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")
        if idx % 3 == 0:
            ax.set_ylabel("Potential growth (ann. %)")

    fig.suptitle("v5 baseline vs v5+ (TT-trend coupling): potential growth",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "v5_baseline_vs_v5plus_g_trend_by_model.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")
    return out


def plot_rr_bar_by_model() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True, sharey=True)
    for idx, (label, base, plus) in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        for r_dir, color, lw, lbl in (
            (RES / base, "C1", 1.2, "v5 baseline"),
            (RES / plus, "C0", 1.6, "v5+ (TT-trend)"),
        ):
            r = _load_smoothed(r_dir)
            if r is None:
                continue
            rr = r["g"]("RR_BAR")
            if rr is None:
                continue
            ax.plot(rr.index, rr.values, color + "-", lw=lw,
                    alpha=0.85, label=lbl)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")
        if idx % 3 == 0:
            ax.set_ylabel("Neutral real rate (ann. %)")

    fig.suptitle("v5 baseline vs v5+ (TT-trend coupling): RR_BAR",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "v5_baseline_vs_v5plus_rr_bar_by_model.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")
    return out


def plot_lz_decomp_by_model() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True, sharey=False)
    for idx, (label, base, plus) in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        for r_dir, color, lw, lbl in (
            (RES / base, "C1", 1.2, "v5 baseline LZ_BAR"),
            (RES / plus, "C0", 1.6, "v5+ LZ_BAR"),
        ):
            r = _load_smoothed(r_dir)
            if r is None:
                continue
            lz_bar = r["g"]("LZ_BAR")
            lz_gap = r["g"]("LZ_GAP")
            if lz_bar is None or lz_gap is None:
                continue
            lz = lz_bar + lz_gap
            ax.plot(lz.index, lz.values, "k-", lw=0.7, alpha=0.4)
            ax.plot(lz_bar.index, lz_bar.values, color + "--", lw=lw, label=lbl)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")
        if idx % 3 == 0:
            ax.set_ylabel("REER (100*log)")

    fig.suptitle("v5 baseline vs v5+ (TT-trend coupling): LZ vs LZ_BAR",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "v5_baseline_vs_v5plus_lz_decomp_by_model.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")
    return out


def main() -> int:
    plot_gap_by_model()
    plot_g_trend_by_model()
    plot_rr_bar_by_model()
    plot_lz_decomp_by_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

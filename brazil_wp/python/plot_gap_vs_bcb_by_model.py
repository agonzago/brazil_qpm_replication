"""One plot per model showing GDP_GAP (posterior mean + median) against
BCB Low/High/Reference envelope.

For each of the four models (A, C, E, F), uses the best-by-BCB-coverage
config for that model and overlays:
  - BCB Low/High filled envelope
  - BCB Reference line (dashed)
  - Model posterior MEAN smoothed GDP_GAP (solid)
  - Model posterior MEDIAN smoothed GDP_GAP (dot-dashed, MAP/mode proxy)

Outputs:
  results_smoke/bcb_compare/gap_by_model.png       (one figure, 2x2 grid)
  results_smoke/bcb_compare/by_model/<model>.png   (per-model files)
"""
from __future__ import annotations

import json
import os
import subprocess
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
OUT_DIR = SMOKE / "bcb_compare" / "by_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Best config per model from the BCB ranking
BEST_BY_MODEL = {
    "A": "TLPL",
    "C": "TLPL",
    "E": "TLPT",
    "F": "TTPT_p15",
}

MODEL_DIRNAME = {"A": "model_a", "B": "model_b", "C": "model_c", "D": "model_d", "E": "model_e", "F": "model_f"}
_ENV_BY_CFG = {
    "TLPL":     {"V5_TT_PIN": "loose", "V5_PREM_PIN": "loose", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TLPT":     {"V5_TT_PIN": "loose", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TTPL":     {"V5_TT_PIN": "tight", "V5_PREM_PIN": "loose", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TTPT":     {"V5_TT_PIN": "tight", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TTPT_p15": {"V5_TT_PIN": "tight", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "15", "V5_TT_GAP_CAP": "3.0"},
    "TTPT_c5":  {"V5_TT_PIN": "tight", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "5.0"},
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
    # BCB dates are quarter-end (Mar/Jun/Sep/Dec on 1st); shift to quarter-start
    df.index = df.index + pd.DateOffset(months=-2)
    return df[["Low", "p25", "p75", "High", "Reference"]].sort_index()


def get_gap_series(model: str, cfg: str) -> dict | None:
    cfg_dir = SMOKE / f"{MODEL_DIRNAME[model]}_{cfg}"
    if not cfg_dir.is_dir():
        return None
    env = dict(os.environ)
    env.update(_ENV_BY_CFG[cfg])
    env.setdefault("JAX_PLATFORMS", "cpu")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(v, "1")
    helper = THIS_DIR / "_gap_mean_mode.py"
    try:
        proc = subprocess.run(
            [sys.executable, str(helper), model, str(cfg_dir)],
            env=env, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        print(f"  helper failed for {model}_{cfg}: rc={proc.returncode}")
        print(proc.stderr[:400])
        return None
    txt = proc.stdout
    if "JSON_BEGIN" not in txt:
        return None
    body = txt.split("JSON_BEGIN", 1)[1].split("JSON_END", 1)[0].strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def plot_one(model: str, cfg: str, bcb: pd.DataFrame,
             gap_data: dict, ax) -> None:
    dates = pd.to_datetime(gap_data["dates"])
    g_mean   = pd.Series(gap_data["gap_mean"],   index=dates)
    g_median = pd.Series(gap_data["gap_median"], index=dates)
    common = dates.intersection(bcb.index)
    if len(common) == 0:
        ax.set_title(f"{model}/{cfg} — no overlap with BCB dates")
        return
    b = bcb.loc[common]
    g_mean_c   = g_mean.loc[common]
    g_median_c = g_median.loc[common]

    # BCB envelope (Reference dropped — see plot_gap_all_configs_per_model.py)
    ax.fill_between(common, b["Low"], b["High"], color="#bbbbbb", alpha=0.35,
                    label="BCB Low–High")
    ax.fill_between(common, b["p25"], b["p75"], color="#888888", alpha=0.35,
                    label="BCB 25th–75th")
    # Model gap
    ax.plot(common, g_mean_c,   color="#1a5599", lw=2.0,
            label=f"{model}_{cfg} (posterior MEAN)")
    ax.plot(common, g_median_c, color="#cc4422", lw=1.5, ls=":",
            label=f"{model}_{cfg} (posterior MEDIAN / MAP proxy)")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"Model {model} · {cfg}", fontsize=12, fontweight="bold")
    ax.set_ylabel("% of potential")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9, ncol=2)


def main() -> int:
    if not BCB_CSV.is_file():
        print(f"ERROR: BCB CSV not found at {BCB_CSV}", file=sys.stderr)
        return 1
    bcb = load_bcb()
    print(f"BCB envelope: {len(bcb)} quarters "
          f"{bcb.index[0].strftime('%Y-Q%q')} – {bcb.index[-1].strftime('%Y-Q%q')}")

    # Compute gap series for each best (model, cfg)
    gap_data: dict[str, dict] = {}
    for model, cfg in BEST_BY_MODEL.items():
        print(f"Smoothing {model}_{cfg} at mean + median params...")
        d = get_gap_series(model, cfg)
        if d is None:
            print(f"  FAILED for {model}_{cfg}")
            continue
        gap_data[model] = {"cfg": cfg, **d}

    # 2x2 grid figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, model in zip(axes, ["A", "C", "E", "F"]):
        if model in gap_data:
            plot_one(model, gap_data[model]["cfg"], bcb, gap_data[model], ax)
        else:
            ax.set_title(f"Model {model} — no data")

    fig.suptitle(
        "GDP_GAP estimates vs BCB output-gap envelope  "
        "(best config per model by BCB coverage)",
        fontsize=14, fontweight="bold", y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = SMOKE / "bcb_compare" / "gap_by_model.png"
    fig.savefig(grid_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {grid_path}")

    # Per-model standalone files
    for model in ["A", "C", "E", "F"]:
        if model not in gap_data:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_one(model, gap_data[model]["cfg"], bcb, gap_data[model], ax)
        ax.set_xlabel("Year")
        fig.tight_layout()
        p = OUT_DIR / f"{model}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

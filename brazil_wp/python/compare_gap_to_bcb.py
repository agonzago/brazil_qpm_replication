"""Compare v5 sweep GDP_GAP estimates to the BCB published range.

Source: brazil_wp/python/data/BCB_outputgap_Garf2.2.8.csv (Figure 2.2.8 of
the latest BCB Inflation Report).

For each of the 24 v5 sweep configs, the script:
  1. Recomputes the smoothed GDP_GAP at the posterior median (via
     filter_diagnostic on a fresh QPMModel built with the right env vars).
  2. Aligns it to the BCB quarter index (2003Q4 – 2026Q1).
  3. Computes:
       - share of quarters where our GAP is inside [BCB Low, BCB High]
       - share where it is inside [BCB 25th, BCB 75th] (the IQR band)
       - mean abs distance vs BCB Reference scenario
       - max abs distance vs Reference
  4. Writes a per-config CSV with quarter-aligned series, a side-by-side
     plot, and a summary table ranking the configs.

Outputs:
  - results_smoke/bcb_compare/SUMMARY.md
  - results_smoke/bcb_compare/<model>_<cfg>/aligned.csv
  - results_smoke/bcb_compare/<model>_<cfg>/gap_vs_bcb.png
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

THIS_DIR = Path(__file__).resolve().parent
SMOKE = THIS_DIR / "results_smoke"
BCB_CSV = THIS_DIR / "data" / "BCB_outputgap_Garf2.2.8.csv"
OUT = SMOKE / "bcb_compare"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["A", "B", "C", "D", "E", "F"]
MODEL_DIRNAME = {"A": "model_a", "B": "model_b", "C": "model_c", "D": "model_d", "E": "model_e", "F": "model_f"}
CONFIGS = ["TLPL", "TLPT", "TTPL", "TTPT", "TTPT_p15", "TTPT_c5"]

_ENV_BY_CFG = {
    "TLPL":     {"V5_TT_PIN": "loose", "V5_PREM_PIN": "loose", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TLPT":     {"V5_TT_PIN": "loose", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TTPL":     {"V5_TT_PIN": "tight", "V5_PREM_PIN": "loose", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TTPT":     {"V5_TT_PIN": "tight", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TTPT_p15": {"V5_TT_PIN": "tight", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "15", "V5_TT_GAP_CAP": "3.0"},
    "TTPT_c5":  {"V5_TT_PIN": "tight", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "5.0"},
}


def load_bcb() -> pd.DataFrame:
    """Parse BCB output-gap CSV; return DataFrame indexed by quarter timestamp."""
    df = pd.read_csv(BCB_CSV, skiprows=9, header=None)
    df = df.iloc[:, :6].rename(columns={0: "quarter", 1: "Low", 2: "p25",
                                        3: "p75", 4: "High", 5: "Reference"})
    df = df.dropna(subset=["quarter"])
    df["quarter"] = df["quarter"].astype(str).str.strip()
    # Parse "Mmm-yy" → first-of-quarter timestamp
    df["date"] = pd.to_datetime(df["quarter"], format="%b-%y", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    for c in ("Low", "p25", "p75", "High", "Reference"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["Low", "p25", "p75", "High", "Reference"]].sort_index()


def extract_smoothed_gdp_gap(model: str, cfg: str) -> pd.Series | None:
    """Spawn helper subprocess (so env vars apply at import) and read the
    smoothed GDP_GAP series via a small in-line script.

    For speed we reuse the _compute_gap_stats logic but extend it to print
    the full series.
    """
    cfg_dir = SMOKE / f"{MODEL_DIRNAME[model]}_{cfg}"
    if not cfg_dir.is_dir():
        return None
    env = dict(os.environ)
    env.update(_ENV_BY_CFG[cfg])
    env.setdefault("JAX_PLATFORMS", "cpu")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(v, "1")
    script = r"""
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS","cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath('"""+str(THIS_DIR)+r"""/_compute_gap_stats.py'))
sys.path.insert(0, '"""+str(THIS_DIR.parent.parent / "gpm_estimation")+r"""')
sys.path.insert(0, '"""+str(THIS_DIR)+r"""')
import importlib
runner = importlib.import_module({MODELS_TO_RUNNER!r}[sys.argv[1]])
from qpm_toolbox import QPMModel
import numpy as np, pandas as pd
ys, dates, obs_names = runner.load_brazil_data()
m = QPMModel(runner.MOD_FILE, results_dir=sys.argv[2])
ss = m.model.default_param_assignments
init_config = runner.compute_initial_conditions(ys, obs_names, ss)
m.set_data(ys, dates=dates, obs_names=obs_names)
m.set_priors(structural_priors=runner.STRUCTURAL_PRIORS, fixed_params=runner.FIXED_PARAMS,
             meas_noise_std=runner.MEAS_NOISE_STD, cycle_map=runner.CYCLE_MAP,
             trend_snr_map=runner.TREND_SNR_MAP, manual_cycle_priors=runner.MANUAL_CYCLE_PRIORS)
m.set_initial_conditions(init_state_config=init_config)
m.solution_check()
m.load_draws(os.path.join(sys.argv[2], 'estimation', 'draws.npz'))
res = m.filter_diagnostic(obs_trend_map=runner.OBS_TREND_MAP)
xs = np.asarray(res['x_smooth'])
sn = m.model.ordered_stat_vars
i = sn.index('GDP_GAP')
gap = xs[:, i].tolist()
print('JSON_BEGIN')
print(json.dumps({'gap': gap, 'dates': [str(d) for d in dates]}))
print('JSON_END')
""".replace("{MODELS_TO_RUNNER!r}", repr({
        "A": "run_brazil_wp_v2", "C": "run_brazil_model_c_v2",
        "E": "run_brazil_model_e_v2", "F": "run_brazil_model_f_v2",
    }))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, model, str(cfg_dir)],
            env=env, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        print(f"  helper failed for {model}_{cfg}: rc={proc.returncode}")
        print(proc.stderr[:500] if proc.stderr else "")
        return None
    # Extract the JSON between markers
    txt = proc.stdout
    if "JSON_BEGIN" not in txt:
        return None
    body = txt.split("JSON_BEGIN", 1)[1].split("JSON_END", 1)[0].strip()
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        return None
    s = pd.Series(d["gap"], index=pd.to_datetime(d["dates"]))
    return s


def compare_one(model: str, cfg: str, bcb: pd.DataFrame) -> dict:
    print(f"  comparing {model}_{cfg}...")
    gap = extract_smoothed_gdp_gap(model, cfg)
    if gap is None:
        return {"model": model, "cfg": cfg, "complete": False}
    # Align: BCB index runs Dec-03 to Mar-26 at quarter ENDs (we'll use
    # quarter-start; reindex bcb to gap's index)
    gap.index = pd.to_datetime(gap.index)
    # BCB CSV's "Mmm-yy" pattern parses to month-1-of-quarter; gap dates
    # are also quarter-start. So they should align as-is.
    common = gap.index.intersection(bcb.index)
    if len(common) == 0:
        # Try shifting either by 2 months in case of end-of-quarter mismatch
        bcb_shift = bcb.copy()
        bcb_shift.index = bcb_shift.index + pd.DateOffset(months=-2)
        common = gap.index.intersection(bcb_shift.index)
        bcb_use = bcb_shift
    else:
        bcb_use = bcb
    if len(common) < 10:
        return {"model": model, "cfg": cfg, "complete": False,
                "error": f"only {len(common)} common quarters"}
    g = gap.loc[common]
    b = bcb_use.loc[common]
    inside_full = ((g >= b["Low"]) & (g <= b["High"])).mean()
    inside_iqr  = ((g >= b["p25"]) & (g <= b["p75"])).mean()
    mae_ref     = (g - b["Reference"]).abs().mean()
    max_dev_ref = (g - b["Reference"]).abs().max()
    # Plot
    out_dir = OUT / f"{MODEL_DIRNAME[model]}_{cfg}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_aligned = pd.DataFrame({
        "model_gap": g,
        "bcb_low":   b["Low"],
        "bcb_p25":   b["p25"],
        "bcb_p75":   b["p75"],
        "bcb_high":  b["High"],
        "bcb_ref":   b["Reference"],
    })
    df_aligned.to_csv(out_dir / "aligned.csv")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(common, b["Low"], b["High"], color="#aaaaaa", alpha=0.3,
                    label="BCB Low-High envelope")
    ax.fill_between(common, b["p25"], b["p75"], color="#777777", alpha=0.4,
                    label="BCB 25th-75th")
    ax.plot(common, b["Reference"], "k--", lw=1.2, label="BCB Reference")
    ax.plot(common, g, "b-", lw=1.8, label=f"v5 sweep {model}_{cfg} (posterior median)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"GDP_GAP vs BCB output-gap envelope · {model}_{cfg}")
    ax.set_ylabel("% (percent of potential)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gap_vs_bcb.png", dpi=140)
    plt.close(fig)
    return {
        "model": model, "cfg": cfg, "complete": True,
        "n_common": int(len(common)),
        "inside_full":  float(inside_full),
        "inside_iqr":   float(inside_iqr),
        "mae_ref":      float(mae_ref),
        "max_dev_ref":  float(max_dev_ref),
    }


def write_summary(rows: list[dict]) -> None:
    df = pd.DataFrame([r for r in rows if r.get("complete")])
    df = df.sort_values(by=["inside_full", "mae_ref"],
                        ascending=[False, True]).reset_index(drop=True)
    lines = [
        "# GDP_GAP vs BCB envelope — v5 sweep comparison",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        f"BCB source: `data/BCB_outputgap_Garf2.2.8.csv` (Figure 2.2.8, "
        f"latest BCB Inflation Report). 5 series: Low / 25th / 75th / "
        f"High / Reference scenario; ranges 2003Q4 – 2026Q1.",
        "",
        "## Coverage criteria",
        "",
        "- **inside_full**: share of common quarters where the model's smoothed "
        "GDP_GAP falls within BCB's Low–High envelope.",
        "- **inside_iqr**:  share within the 25th–75th percentile band.",
        "- **mae_ref**:     mean absolute distance from BCB Reference scenario.",
        "- **max_dev_ref**: max absolute distance from BCB Reference.",
        "",
        "All series are at the posterior median of the v5 sweep estimation.",
        "",
        "## Ranked comparison",
        "",
        "| Rank | Model | Config | n quarters | inside Low–High | inside IQR | MAE vs Ref | max dev vs Ref |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in df.iterrows():
        lines.append(
            f"| {i+1} | {r['model']} | {r['cfg']} | {r['n_common']} | "
            f"{100*r['inside_full']:.0f}% | {100*r['inside_iqr']:.0f}% | "
            f"{r['mae_ref']:.2f} | {r['max_dev_ref']:.2f} |"
        )
    lines += [
        "",
        "## Per-config plots",
        "",
        "Each config has a side-by-side plot showing its smoothed GDP_GAP "
        "(blue) overlaid on the BCB Low–High envelope (grey wide), 25th-75th "
        "band (grey narrow), and Reference scenario (black dashed).",
        "",
        "Files: `bcb_compare/<model>_<cfg>/gap_vs_bcb.png` and `aligned.csv`.",
    ]
    out = OUT / "SUMMARY.md"
    out.write_text("\n".join(lines))
    print(f"\nWrote {out}")


def main() -> int:
    if not BCB_CSV.is_file():
        print(f"ERROR: BCB CSV not found at {BCB_CSV}", file=sys.stderr)
        return 1
    bcb = load_bcb()
    print(f"Loaded BCB envelope: {len(bcb)} quarters, "
          f"{bcb.index[0].strftime('%Y-Q')} – {bcb.index[-1].strftime('%Y-Q')}")
    rows = []
    for m in MODELS:
        for c in CONFIGS:
            rows.append(compare_one(m, c, bcb))
    write_summary(rows)
    n_ok = sum(1 for r in rows if r.get("complete"))
    print(f"\n{n_ok}/{len(rows)} configs compared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

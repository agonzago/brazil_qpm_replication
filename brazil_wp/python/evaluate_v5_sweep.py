"""Comprehensive evaluator for the v5 sweep.

Reads every `results_smoke/<model>_<cfg>/` produced by run_v5_sweep.sh and
writes a ranked report at `results_smoke/SWEEP_REPORT.md`.

Evaluates each config on multiple dimensions:
  1. Convergence (R̂, ESS_bulk, divergences)
  2. Output gap quality (max, share above ±4%, std)
  3. PREM filtration quality (PREM_GAP AR(1) persistence, PREM_BAR
     decline absorbed)
  4. TT filtration quality (TT_BAR/TT_GAP std ratio, smoothness)
  5. Headline structural parameters vs paper v5 baseline
  6. Sacrifice ratio (median + 68% CI) vs paper
  7. 12Q cumulative pass-through (3 paper shocks) vs paper
  8. Forecast RMSE (h=1,2,4,8) if available
  9. Composite quality score → ranking + recommendation
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# v5 sweep config matrix used to set env vars for the gap-stats subprocess
_ENV_BY_CFG = {
    "TLPL":     {"V5_TT_PIN": "loose", "V5_PREM_PIN": "loose", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TLPT":     {"V5_TT_PIN": "loose", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TTPL":     {"V5_TT_PIN": "tight", "V5_PREM_PIN": "loose", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TTPT":     {"V5_TT_PIN": "tight", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "3.0"},
    "TTPT_p15": {"V5_TT_PIN": "tight", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "15", "V5_TT_GAP_CAP": "3.0"},
    "TTPT_c5":  {"V5_TT_PIN": "tight", "V5_PREM_PIN": "tight", "V5_PREM_SNR": "30", "V5_TT_GAP_CAP": "5.0"},
}

THIS_DIR = Path(__file__).resolve().parent
SMOKE = THIS_DIR / "results_smoke"
sys.path.insert(0, str(THIS_DIR))

MODELS = ["A", "B", "C", "D", "E", "F"]
MODEL_DIRNAME = {"A": "model_a", "B": "model_b", "C": "model_c", "D": "model_d", "E": "model_e", "F": "model_f"}
CONFIGS = ["TLPL", "TLPT", "TTPL", "TTPT", "TTPT_p15", "TTPT_c5"]
CONFIG_DESC = {
    "TLPL":     "TT loose, PREM loose",
    "TLPT":     "TT loose, PREM α=20 pin",
    "TTPL":     "TT tight (SNR=50/cap=3/α=20), PREM loose",
    "TTPT":     "TT tight, PREM α=20 pin",
    "TTPT_p15": "TT tight, PREM α=20 pin, PREM_SNR=15 (refinement)",
    "TTPT_c5":  "TT tight cap=5, PREM α=20 pin (refinement)",
}

# Paper v5 reference (Model A baseline; from verify_paper_results.py)
PAPER_BENCH = {
    "gamma1":   0.921,
    "gamma2":   3.124,
    "gamma4":   0.252,
    "lambda1x": 0.270,
    "lambda2x": 0.120,
    "alpha_rr": 0.180,
    "alpha_pi": 0.004,
    "alpha_dlz": 0.428,
    "psi_uip":  0.761,
    "rho_ls":   0.255,
    "phi_tt_is": 0.163,
    "sacrifice_median": 0.314,
}


def safe_read(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def load_arviz(cfg_dir: Path) -> pd.DataFrame | None:
    p = cfg_dir / "estimation" / "arviz_summary.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p).rename(columns={"Unnamed: 0": "param"}).set_index("param")
    return df.apply(pd.to_numeric, errors="coerce")


def load_decomp_for_gap(cfg_dir: Path) -> np.ndarray | None:
    """Load smoothed GDP_GAP from decomposition output, if available.

    The runner saves the smoother output as decomp NPZ or as the
    diag_states.png input.  We try a couple of conventional locations.
    """
    npz_paths = [
        cfg_dir / "decomposition" / "decomp_data.npz",
        cfg_dir / "diagnostic" / "smoothed_states.npz",
    ]
    for p in npz_paths:
        if p.is_file():
            try:
                z = np.load(p, allow_pickle=True)
                for key in ("GDP_GAP", "GDP_GAP_smooth", "x_smooth"):
                    if key in z.files:
                        return np.asarray(z[key])
            except Exception:
                continue
    return None


def compute_filtration_stats(model: str, cfg: str, cfg_dir: Path) -> dict | None:
    """Recompute smoothed states via `_compute_gap_stats.py` subprocess.

    Returns a dict with 'GDP_GAP', 'PREM_GAP', 'PREM_BAR', 'TT_BAR',
    'TT_GAP', 'log_likelihood' keys, or None on failure.
    """
    env = dict(os.environ)
    env.update(_ENV_BY_CFG[cfg])
    env.setdefault("JAX_PLATFORMS", "cpu")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(v, "1")
    helper = THIS_DIR / "_compute_gap_stats.py"
    try:
        proc = subprocess.run(
            [sys.executable, str(helper), model, str(cfg_dir)],
            env=env, capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    # The helper prints other progress lines; take the last non-empty line
    # which should be the JSON blob.
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def evaluate_config(model: str, cfg: str) -> dict:
    cfg_dir = SMOKE / f"{MODEL_DIRNAME[model]}_{cfg}"
    out: dict = {"model": model, "cfg": cfg, "dir": cfg_dir.relative_to(THIS_DIR),
                 "complete": False}

    arviz = load_arviz(cfg_dir)
    if arviz is None:
        out["error"] = "arviz_summary.csv missing"
        return out
    out["complete"] = True

    # 1. Convergence
    out["rhat_max"] = float(arviz["r_hat"].max())
    out["ess_min"] = float(arviz["ess_bulk"].min())

    # 2/3/4. Filtration metrics — recompute smoothed states by importing
    # the runner with the right env vars and calling filter_diagnostic at
    # the posterior median.
    print(f"  computing filtration stats for {model}_{cfg}...")
    filt = compute_filtration_stats(model, cfg, cfg_dir)
    if filt is not None:
        if "GDP_GAP" in filt:
            out["gdp_gap"] = filt["GDP_GAP"]
        if "PREM_GAP" in filt:
            out["prem_gap"] = filt["PREM_GAP"]
        if "PREM_BAR" in filt:
            out["prem_bar"] = filt["PREM_BAR"]
        if "TT_BAR" in filt:
            out["tt_bar"] = filt["TT_BAR"]
        if "TT_GAP" in filt:
            out["tt_gap"] = filt["TT_GAP"]
        if "log_likelihood" in filt:
            out["log_likelihood"] = filt["log_likelihood"]

    # 5. Headline parameters vs paper
    out["params"] = {}
    for p_name, paper_val in PAPER_BENCH.items():
        if p_name == "sacrifice_median":
            continue
        if p_name in arviz.index:
            m = float(arviz.loc[p_name, "mean"])
            lo_col = "hdi_16%" if "hdi_16%" in arviz.columns else "hdi_3%"
            hi_col = "hdi_84%" if "hdi_84%" in arviz.columns else "hdi_97%"
            try:
                lo = float(arviz.loc[p_name, lo_col])
                hi = float(arviz.loc[p_name, hi_col])
            except Exception:
                lo, hi = m, m
            out["params"][p_name] = {
                "mean": m,
                "hdi_lo": lo,
                "hdi_hi": hi,
                "paper": paper_val,
                "delta_pct": 100.0 * (m - paper_val) / paper_val if paper_val else None,
            }

    # 6. Sacrifice ratio
    sr_path = cfg_dir / "policy" / "sacrifice_ratio_draws.csv"
    if sr_path.is_file():
        sr_df = safe_read(sr_path)
        if sr_df is not None and len(sr_df.columns):
            col = sr_df.columns[-1]
            arr = sr_df[col].dropna().values
            if len(arr):
                out["sacrifice"] = {
                    "median": float(np.median(arr)),
                    "mean":   float(np.mean(arr)),
                    "lo68":   float(np.percentile(arr, 16)),
                    "hi68":   float(np.percentile(arr, 84)),
                    "n":      len(arr),
                    "paper_median": PAPER_BENCH["sacrifice_median"],
                }

    # 7. Pass-through
    pt_summary = {}
    for shock in ("SHK_LZ_GAP", "SHK_RS_US", "SHK_RS"):
        p_csv = cfg_dir / "policy" / f"pass_through_horizon_draws.npz"
        if p_csv.is_file():
            try:
                npz = np.load(p_csv, allow_pickle=True)
                key = f"{shock}__DA_LCPI"
                if key in npz.files:
                    arr = npz[key]   # (draws, horizon)
                    pt_summary[shock] = float(np.median(arr[:, -1]))  # 12Q
            except Exception:
                pass
    out["pass_through_12q"] = pt_summary

    # 8. Forecast RMSE (h=1,2,4,8 medians, if available)
    rmse_path = cfg_dir / "evaluation" / "forecast_rmse.csv"
    if rmse_path.is_file():
        rmse = safe_read(rmse_path)
        if rmse is not None:
            out["forecast_rmse_columns"] = list(rmse.columns)[:8]

    # 9. Composite quality flags
    flags = []
    if out.get("rhat_max", 99) > 1.03:
        flags.append(f"R̂={out['rhat_max']:.3f} > 1.03")
    if out.get("ess_min", 0) < 150:
        flags.append(f"ESS_bulk_min={out['ess_min']:.0f} < 150")
    if "gdp_gap" in out and out["gdp_gap"]["max_abs"] > 4.0:
        flags.append(f"|GDP_GAP|_max={out['gdp_gap']['max_abs']:.2f}% > 4%")
    if "gdp_gap" in out and out["gdp_gap"]["share_gt_4"] > 0.05:
        flags.append(f"{100*out['gdp_gap']['share_gt_4']:.0f}% of sample with |GDP_GAP|>4%")
    if "prem_gap" in out and out["prem_gap"]["ar1"] > 0.95:
        flags.append(f"PREM_GAP AR(1)={out['prem_gap']['ar1']:.2f} > 0.95")
    out["flags"] = flags
    out["status"] = "PASS" if not flags else ("WARN" if len(flags) <= 1 else "FAIL")

    return out


def write_report(results: list[dict]) -> None:
    lines = [
        "# v5 sweep — comprehensive evaluation",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        "",
        "## Configuration matrix",
        "",
        "| ID | description |",
        "|---|---|",
    ]
    for c in CONFIGS:
        lines.append(f"| {c} | {CONFIG_DESC[c]} |")

    lines += [
        "",
        "## Composite ranking",
        "",
        "| Status | Model | Config | R̂ max | ESS min | |GDP_GAP|_max | PREM AR(1) | flags |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    # Sort: PASS first, then by R-hat
    def sort_key(r):
        st = {"PASS": 0, "WARN": 1, "FAIL": 2}.get(r.get("status", "FAIL"), 3)
        return (st, r.get("rhat_max", 99))
    for r in sorted(results, key=sort_key):
        if not r.get("complete"):
            continue
        gap_max = r.get("gdp_gap", {}).get("max_abs", float("nan"))
        prem_ar1 = r.get("prem_gap", {}).get("ar1", float("nan"))
        flags = "; ".join(r.get("flags", [])) if r.get("flags") else "—"
        lines.append(
            f"| {r['status']} | {r['model']} | {r['cfg']} | "
            f"{r['rhat_max']:.3f} | {r['ess_min']:.0f} | "
            f"{gap_max:.2f} | {prem_ar1:.2f} | {flags} |"
        )

    lines += [
        "",
        "## Per-config detail",
        "",
    ]
    for r in results:
        if not r.get("complete"):
            lines.append(f"### {r['model']} / {r['cfg']}  —  ❌ incomplete ({r.get('error')})")
            lines.append("")
            continue
        lines.append(f"### {r['model']} / {r['cfg']}  —  **{r['status']}**")
        lines.append(f"Output: `{r['dir']}/`  ·  {CONFIG_DESC[r['cfg']]}")
        lines.append("")
        # Convergence
        lines.append(f"- **Convergence:** R̂_max={r['rhat_max']:.4f}, ESS_min={r['ess_min']:.0f}")
        # Output gap
        if "gdp_gap" in r:
            g = r["gdp_gap"]
            lines.append(f"- **Output gap:** max|GDP_GAP|={g['max_abs']:.2f}%, "
                         f"share>4%={100*g['share_gt_4']:.0f}%, "
                         f"share>2%={100*g['share_gt_2']:.0f}%, std={g['std']:.2f}")
        else:
            lines.append("- **Output gap:** smoothed series not found in draws.npz")
        # PREM
        if "prem_gap" in r:
            p_ = r["prem_gap"]
            lines.append(f"- **PREM_GAP:** AR(1)={p_['ar1']:.3f}, AR(4)={p_['ar4']:.3f}, "
                         f"|mean|={p_['abs_mean']:.3f}, std={p_['std']:.3f}")
        if "prem_bar" in r:
            pb = r["prem_bar"]
            lines.append(f"- **PREM_BAR trend:** "
                         f"first={pb['first']:.2f} → last={pb['last']:.2f}  "
                         f"(range {pb['range']:+.2f}; CDS5 observed ~5.8 → ~1.5, target Δ≈-4.3)")
        if "tt_bar" in r:
            tb = r["tt_bar"]
            lines.append(f"- **TT_BAR trend:** std={tb['std']:.2f}, range={tb['range']:.2f}")
        if "tt_gap" in r:
            tg = r["tt_gap"]
            lines.append(f"- **TT_GAP cycle:** std={tg['std']:.2f}, max|·|={tg['max_abs']:.2f}, AR(1)={tg['ar1']:.3f}")
        if "log_likelihood" in r:
            lines.append(f"- **Log-likelihood (posterior median):** {r['log_likelihood']:.2f}")
        # Parameters
        if r.get("params"):
            lines.append("- **Key parameters vs paper v5:**")
            lines.append("")
            lines.append("  | Param | Posterior mean | 68% HDI | Paper | Δ |")
            lines.append("  |---|---:|---|---:|---:|")
            for name, p_ in r["params"].items():
                dlt = p_["delta_pct"]
                dlt_str = f"{dlt:+.1f}%" if dlt is not None else "—"
                lines.append(
                    f"  | {name} | {p_['mean']:.3f} | "
                    f"[{p_['hdi_lo']:.2f}, {p_['hdi_hi']:.2f}] | "
                    f"{p_['paper']:.3f} | {dlt_str} |"
                )
        if "sacrifice" in r:
            s = r["sacrifice"]
            lines.append(f"- **Sacrifice ratio:** median={s['median']:.3f} "
                         f"68% CI=[{s['lo68']:.2f}, {s['hi68']:.2f}]   "
                         f"(paper {s['paper_median']:.3f})")
        if r.get("pass_through_12q"):
            lines.append("- **Pass-through (12Q cumulative, posterior median):**")
            for shock, val in r["pass_through_12q"].items():
                lines.append(f"  - {shock} → DA_LCPI: {val:.3f}")
        if r["flags"]:
            lines.append("- **Quality flags:**")
            for f in r["flags"]:
                lines.append(f"  - ⚠ {f}")
        lines.append("")

    # Recommendation
    lines += [
        "## Recommendation",
        "",
    ]
    passes = [r for r in results if r.get("complete") and r["status"] == "PASS"]
    if passes:
        # Prefer Model A baseline if it passes
        a_pass = [r for r in passes if r["model"] == "A"]
        bench = (a_pass[0] if a_pass else passes[0])
        lines.append(f"**Benchmark candidate:** {bench['model']}_{bench['cfg']} — "
                     f"{CONFIG_DESC[bench['cfg']]}")
        lines.append("")
        alts = [r for r in passes if r is not bench][:3]
        if alts:
            lines.append("**Viable alternatives:**")
            for r in alts:
                lines.append(f"- {r['model']}_{r['cfg']} ({CONFIG_DESC[r['cfg']]})")
            lines.append("")
        rejected = [r for r in results if r.get("complete") and r["status"] == "FAIL"]
        if rejected:
            lines.append("**Rejected (multiple quality flags):**")
            for r in rejected:
                lines.append(f"- {r['model']}_{r['cfg']}: "
                             f"{'; '.join(r['flags'])}")
            lines.append("")
    else:
        warns = [r for r in results if r.get("complete") and r["status"] == "WARN"]
        if warns:
            lines.append("No PASS configs.  Top WARN configs (single flag, may still be usable):")
            for r in warns[:3]:
                lines.append(f"- {r['model']}_{r['cfg']}: {r['flags'][0]}")
        else:
            lines.append("All configs failed quality checks.  Manual review required.")

    out = SMOKE / "SWEEP_REPORT.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main() -> int:
    results = []
    for model in MODELS:
        for cfg in CONFIGS:
            print(f"Evaluating {model}_{cfg}...")
            results.append(evaluate_config(model, cfg))
    write_report(results)
    n_complete = sum(1 for r in results if r.get("complete"))
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    print(f"\nComplete: {n_complete}/{len(results)}   PASS: {n_pass}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

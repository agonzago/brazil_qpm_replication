"""Build the three cross-spec aggregation tables for paper v5.

Outputs (under results_v2_diffuse_v5/_tables/):
  posterior_summary_v5.csv          (Table 3 — convergence + key parameter posteriors across A, A0, B, C, D, E)
  forecast_rmse_v5.csv              (Table 4 — unconditional forecast RMSE, h=1..8, plus BVAR baselines)
  conditional_forecast_rmse_v5.csv  (Table 5 — conditional forecast RMSE)

Each file is a long-format CSV that downstream LaTeX-table scripts can pivot
to the v4 layout (variable × horizon × spec).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


V5 = Path(__file__).resolve().parent / "results_v2_diffuse_v5"
OUT = V5 / "_tables"
OUT.mkdir(parents=True, exist_ok=True)

# Specs and their result subdirectories.  Tables 3/4/5 cover the 6 body
# specs (A, A0, B, C, D, E); RUB1/3/4/5 appear in the appendix robustness
# table built separately.
BODY = [
    ("A",  "model_a"),
    ("A0", "model_a0"),
    ("B",  "model_b"),
    ("C",  "model_c"),
    ("D",  "model_d"),
    ("E",  "model_e"),
]
RUBS = [
    ("RUB1", "rub1"),
    ("RUB3", "rub3"),
    ("RUB4", "rub4"),
    ("RUB5", "rub5"),
]
ALL_SPECS = BODY + RUBS

# Parameters reported in Table 3 (matches v4 §4.5 cross-spec table).
HEADLINE_PARAMS = [
    "beta1", "beta2", "beta3", "beta4", "beta5",
    "phi_tt_is", "phi_tt_pc",
    "rho_QRR", "rho_QTT", "rho_QZ", "rho_QZ_E", "rho_QZ_F",
    "lambda1x", "lambda2x", "lambda4x", "lambda5x", "lambda6x",
    "kappa_z",
    "gamma1", "gamma2", "gamma4",
    "rho_ls", "psi_uip", "rho_prem_gap",
    "alpha_g", "alpha_dlz", "alpha_prem", "alpha_rr", "alpha_pi",
    "alpha_rr_us", "alpha_pi_us",
    "sigma_SHK_TT_GAP", "sigma_SHK_TT_BAR",
    "sigma_SHK_LZ_BAR", "sigma_SHK_DLZ_TREND",
    "sigma_SHK_GDP_GAP", "sigma_SHK_DA_LCPIX",
    "sigma_SHK_RS",
]


def _read_summary(d: Path) -> pd.DataFrame | None:
    p = d / "estimation" / "arviz_summary.csv"
    return pd.read_csv(p, index_col=0) if p.is_file() else None


def build_posterior_summary() -> Path:
    rows = []
    for label, sub in ALL_SPECS:
        df = _read_summary(V5 / sub)
        if df is None:
            print(f"  skip posterior {label}: arviz_summary.csv missing")
            continue
        # Convergence headlines.
        rows.append({
            "spec": label, "param": "_DIAG_max_rhat",
            "mean": float(df["r_hat"].max()), "sd": np.nan,
            "hdi_3pct": np.nan, "hdi_97pct": np.nan,
            "ess_bulk": float(df["ess_bulk"].min()),
            "ess_tail": float(df["ess_tail"].min()),
            "r_hat": float(df["r_hat"].max()),
        })
        # Headline parameter posteriors.
        for p in HEADLINE_PARAMS:
            if p not in df.index:
                continue
            r = df.loc[p]
            rows.append({
                "spec": label, "param": p,
                "mean":     float(r["mean"]),
                "sd":       float(r["sd"]),
                "hdi_3pct":  float(r["hdi_3%"]) if "hdi_3%" in r.index else np.nan,
                "hdi_97pct": float(r["hdi_97%"]) if "hdi_97%" in r.index else np.nan,
                "ess_bulk": float(r["ess_bulk"]),
                "ess_tail": float(r["ess_tail"]),
                "r_hat":    float(r["r_hat"]),
            })
    out = OUT / "posterior_summary_v5.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.5f")
    print(f"  {out}: {len(rows)} rows")
    return out


def _read_rmse(d: Path, name: str) -> pd.DataFrame | None:
    p = d / "evaluation" / name
    return pd.read_csv(p) if p.is_file() else None


def _read_bvar_rmse(cond: bool) -> pd.DataFrame | None:
    """Load the 14-variable BVAR forecast RMSE produced by
    ``run_forecast_evaluation.py``.  These are the §6 benchmark rows
    (not the 4-var/5-var sign-restricted BVARs used for the sacrifice
    ratio cross-check)."""
    name = "cond_bvar_forecast_rmse.csv" if cond else "bvar_forecast_rmse.csv"
    p = V5.parent / "results" / "evaluation" / name
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    # Variable-name map: BVAR uses raw QPM names; QPM evaluation/forecast_rmse.csv
    # uses OBS_* names.  Align so the merged Table 4/5 stacks cleanly.
    var_map = {"DA_LGDP": "OBS_LGDP", "DA_LCPI": "OBS_DA_LCPI",
               "DA_LCPIX": "OBS_DA_LCPIX", "RS": "OBS_RS"}
    df["variable"] = df["variable"].map(var_map).fillna(df["variable"])
    spec = "BVAR_Conditional" if cond else "BVAR"
    df = df.rename(columns={"model": "spec"})
    df["spec"] = spec
    keep = [c for c in ("spec", "variable", "horizon",
                        "rmse", "mae", "bias") if c in df.columns]
    return df[keep]


def build_forecast_rmse(cond: bool = False) -> Path:
    rows = []
    fname = "conditional_forecast_rmse.csv" if cond else "forecast_rmse.csv"
    for label, sub in ALL_SPECS:
        df = _read_rmse(V5 / sub, fname)
        if df is None:
            print(f"  skip {fname} for {label}: missing")
            continue
        df = df.copy()
        df["spec"] = label
        rows.append(df)
    # Append the 14-variable BVAR benchmark row (paper Section 6 Tables 4 & 5).
    bvar = _read_bvar_rmse(cond)
    if bvar is not None and not bvar.empty:
        rows.append(bvar)
        print(f"  + BVAR rows folded in ({len(bvar)} rows)")
    name = ("conditional_forecast_rmse_v5.csv" if cond
            else "forecast_rmse_v5.csv")
    out = OUT / name
    if not rows:
        pd.DataFrame(columns=["spec", "variable", "horizon",
                              "rmse", "mae", "bias"]).to_csv(out, index=False)
    else:
        long = pd.concat(rows, ignore_index=True)
        cols = ["spec", "variable", "horizon", "rmse", "mae", "bias"]
        long = long[[c for c in cols if c in long.columns]]
        long.to_csv(out, index=False, float_format="%.4f")
    print(f"  {out}: {sum(len(r) for r in rows)} rows")
    return out


def main() -> int:
    print("Building v5 aggregation tables under", OUT)
    build_posterior_summary()
    build_forecast_rmse(cond=False)
    build_forecast_rmse(cond=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

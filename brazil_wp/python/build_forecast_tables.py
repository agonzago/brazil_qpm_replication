#!/usr/bin/env python3
"""
Generate the four Section 6 forecast-RMSE LaTeX tables directly from the
canonical per-spec evaluation CSVs, with fully explicit provenance.

Motivation
----------
The Section 6 forecast tables were historically hand-transcribed from a
scattered set of CSVs.  That made them impossible to trace and led to
silent inconsistencies (different model rows ended up sourced from
different estimation treatments).  This script centralises the mapping:
every cell's (file, variable, column) origin is declared once, the LaTeX
is emitted mechanically, and a provenance manifest plus an anomaly report
are printed so the source of every number is auditable.

Usage
-----
    ${PYTHON:-python3} build_forecast_tables.py
    ${PYTHON:-python3} build_forecast_tables.py --treatment _v5plus    # fixed-kappa
    ${PYTHON:-python3} build_forecast_tables.py --write                 # write .tex fragments

Outputs (with --write), under paper_v5/draft/_generated/:
    forecast_table_uncond.tex
    forecast_table_uncond_excovid.tex
    forecast_table_cond.tex
    forecast_table_cond_excovid.tex

Each generated .tex is a complete table environment; Section 6 can
\\input it in place of the hand-written tabular.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
V5 = THIS_DIR / "results_v2_diffuse_v5"
BVAR_DIR = THIS_DIR / "results" / "evaluation"
OUT_DIR = (THIS_DIR / ".." / "paper" / "paper_v5" / "draft"
           / "_generated").resolve()

HORIZONS = [1, 2, 3, 4, 8]

# QPM specifications and their result-directory stems (the treatment
# suffix is appended at runtime).  Model C is the headline baseline.
SPECS = [("A", "model_a"), ("B", "model_b"), ("C", "model_c"), ("D", "model_d")]
BASELINE = "C"

# Display rows, in order.  For each: display label, the QPM variable code
# (read directly from the model's reporting output -- GDP growth uses the
# model-transformed DA_LGDP_ANN, not a separate side file), and the BVAR
# variable code (BVAR files use bare names, no OBS_ prefix).
VARIABLES = [
    ("Headline CPI inflation", "OBS_DA_LCPI",  "DA_LCPI"),
    ("Core CPI inflation",     "OBS_DA_LCPIX", "DA_LCPIX"),
    ("GDP growth",             "DA_LGDP_ANN",  "DA_LGDP"),
    ("Selic policy rate",      "OBS_RS",       "RS"),
]

# Per-table source declaration.  Every QPM cell -- including GDP growth --
# comes from a single per-spec file carrying both full-sample (rmse_full)
# and COVID-excluded (rmse_excovid) RMSE for all four variables.  This is
# the canonical source the paper draws from; the legacy forecast_rmse.csv
# / forecast_rmse_da_lgdp.csv side files are NOT used.
TABLES = {
    "uncond": {
        "caption": "Unconditional Forecast RMSE, Full Sample",
        "label": "tab:forecast_uncond",
        "qpm_file": "forecast_rmse_full_excovid.csv",
        "qpm_col": "rmse_full",
        "bvar_file": "bvar_forecast_rmse.csv",
        "bvar_col": "rmse",
    },
    "uncond_excovid": {
        "caption": "Unconditional Forecast RMSE, COVID-Excluded Sample",
        "label": "tab:forecast_uncond_excovid",
        "qpm_file": "forecast_rmse_full_excovid.csv",
        "qpm_col": "rmse_excovid",
        "bvar_file": "bvar_forecast_rmse.csv",
        "bvar_col": "rmse_excovid",
    },
    # The two conditional tables report the RELATIVE RMSE REDUCTION from
    # conditioning -- (uncond - cond) / uncond * 100 -- because that is the
    # quantity the paper argues (the BVAR depends heavily on ex-post
    # conditioning while the QPM changes little).  Positive = conditioning
    # lowers RMSE; negative = conditioning worsens it.
    "cond": {
        "caption": "Relative RMSE Reduction from Conditioning, Full Sample (\\%)",
        "label": "tab:forecast_cond",
        "qpm_file": "conditional_forecast_rmse_full_excovid.csv",
        "qpm_col": "rmse_full",
        "bvar_file": "cond_bvar_forecast_rmse.csv",
        "bvar_col": "rmse",
        "relative_gain": True,
        "uncond_qpm_file": "forecast_rmse_full_excovid.csv",
        "uncond_qpm_col": "rmse_full",
        "uncond_bvar_file": "bvar_forecast_rmse.csv",
        "uncond_bvar_col": "rmse",
    },
    "cond_excovid": {
        "caption": "Relative RMSE Reduction from Conditioning, COVID-Excluded Sample (\\%)",
        "label": "tab:forecast_cond_excovid",
        "qpm_file": "conditional_forecast_rmse_full_excovid.csv",
        "qpm_col": "rmse_excovid",
        "bvar_file": "cond_bvar_forecast_rmse.csv",
        "bvar_col": "rmse_excovid",
        "relative_gain": True,
        "uncond_qpm_file": "forecast_rmse_full_excovid.csv",
        "uncond_qpm_col": "rmse_excovid",
        "uncond_bvar_file": "bvar_forecast_rmse.csv",
        "uncond_bvar_col": "rmse_excovid",
    },
}

# Notes reproduce the paper wording; {origins} etc. kept generic.
NOTES = {
    "uncond": (
        "Root mean squared forecast errors in annualised percentage "
        "points (annualised quarter-on-quarter inflation, GDP growth and "
        "Selic policy rate), computed over $24$ forecast origins "
        "(2017Q1--2022Q4) and all (origin, horizon) pairs whose target "
        "falls inside the sample.  QPM specifications use the full-sample "
        "posterior median parameter vector; the BVAR is re-estimated at "
        "each origin under a Minnesota prior.  Baseline (Model~C) labelled "
        "in bold."),
    "uncond_excovid": (
        "As Table~\\ref{tab:forecast_uncond} but excluding forecast "
        "observations for which either the origin or target quarter lies "
        "in 2020Q1--Q4.  After the filter the remaining sample sizes per "
        "horizon are $n_{h=1}=19$, $n_{h=2}=18$, $n_{h=3}=17$, "
        "$n_{h=4}=16$ and $n_{h=8}=16$ (out of the $24$ origins)."),
    "cond": (
        "Relative RMSE reduction from conditioning, in percent: "
        "$100\\times(\\text{RMSE}_{\\text{uncond}}-\\text{RMSE}_{\\text{cond}})"
        "/\\text{RMSE}_{\\text{uncond}}$ against the matching "
        "unconditional RMSE in Table~\\ref{tab:forecast_uncond}.  Each "
        "model receives the realised future path of the seven external "
        "and exogenous variables (oil and food commodity gaps, U.S.\\ "
        "CPI, U.S.\\ output gap, U.S.\\ rates, domestic energy CPI) via "
        "the \\citet{WaggonerZha1999} hard-constraint algorithm and "
        "forecasts the remaining seven domestic variables freely.  A "
        "positive entry means conditioning lowers RMSE; a negative entry "
        "means it raises RMSE.  Sample as Table~\\ref{tab:forecast_uncond}."),
    "cond_excovid": (
        "As Table~\\ref{tab:forecast_cond} (relative RMSE reduction from "
        "conditioning, percent) but excluding forecast observations for "
        "which either the origin or target quarter lies in 2020Q1--Q4, "
        "against the matching unconditional RMSE in "
        "Table~\\ref{tab:forecast_uncond_excovid}."),
}


class Loader:
    """Cache CSVs and resolve a single (file, variable, horizon, column)
    cell, recording every file actually opened for the manifest."""

    def __init__(self, treatment: str):
        self.treatment = treatment
        self._cache: dict[Path, pd.DataFrame | None] = {}
        self.used_files: set[str] = set()
        self.missing: set[str] = set()

    def _read(self, path: Path) -> pd.DataFrame | None:
        if path not in self._cache:
            self._cache[path] = pd.read_csv(path) if path.is_file() else None
            if path.is_file():
                self.used_files.add(str(path.relative_to(THIS_DIR)))
            else:
                self.missing.add(str(path.relative_to(THIS_DIR)))
        return self._cache[path]

    def qpm(self, spec_stem: str, fname: str, var: str, h: int,
            col: str) -> float | None:
        path = V5 / f"{spec_stem}{self.treatment}" / "evaluation" / fname
        df = self._read(path)
        if df is None:
            return None
        sel = df[(df["variable"] == var) & (df["horizon"] == h)]
        return None if sel.empty else float(sel.iloc[0][col])

    def bvar(self, fname: str, var: str, h: int, col: str) -> float | None:
        path = BVAR_DIR / fname
        df = self._read(path)
        if df is None:
            return None
        sel = df[(df["variable"] == var) & (df["horizon"] == h)]
        return None if sel.empty else float(sel.iloc[0][col])


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def build_table(key: str, ld: Loader) -> tuple[str, list[str]]:
    """Return (latex, anomaly_messages) for one table."""
    cfg = TABLES[key]
    anomalies: list[str] = []
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{cfg['caption']}}}",
        f"\\label{{{cfg['label']}}}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Variable} & \\textbf{Model} & $h=1$ & $h=2$ & $h=3$ "
        "& $h=4$ & $h=8$ \\\\",
        "\\midrule",
    ]

    gain = cfg.get("relative_gain", False)

    def _gain(uncond, cond):
        """Percent RMSE reduction from conditioning."""
        if uncond is None or cond is None or uncond == 0:
            return None
        return 100.0 * (uncond - cond) / uncond

    for vi, (label, qvar, bvar) in enumerate(VARIABLES):
        fname = cfg["qpm_file"]
        lines.append(f"\\multirow{{5}}{{*}}{{\\textbf{{{label}}}}}")
        for code, stem in SPECS:
            cond_vals = [ld.qpm(stem, fname, qvar, h, cfg["qpm_col"])
                         for h in HORIZONS]
            for h, v in zip(HORIZONS, cond_vals):
                if v is None:
                    anomalies.append(
                        f"[{key}] MISSING {code} {qvar} h={h} in {fname}")
            if gain:
                unc = [ld.qpm(stem, cfg["uncond_qpm_file"], qvar, h,
                              cfg["uncond_qpm_col"]) for h in HORIZONS]
                vals = [_gain(u, c) for u, c in zip(unc, cond_vals)]
            else:
                vals = cond_vals
            name = (f"\\textbf{{{code} (baseline)}}"
                    if code == BASELINE else code)
            cells = " & ".join(fmt(v) for v in vals)
            lines.append(f"   & {name:<26} & {cells} \\\\")
        # BVAR row
        bcond = [ld.bvar(cfg["bvar_file"], bvar, h, cfg["bvar_col"])
                 for h in HORIZONS]
        for h, v in zip(HORIZONS, bcond):
            if v is None:
                anomalies.append(
                    f"[{key}] MISSING BVAR {bvar} h={h} "
                    f"in {cfg['bvar_file']}:{cfg['bvar_col']}")
        if gain:
            bunc = [ld.bvar(cfg["uncond_bvar_file"], bvar, h,
                            cfg["uncond_bvar_col"]) for h in HORIZONS]
            bvals = [_gain(u, c) for u, c in zip(bunc, bcond)]
        else:
            bvals = bcond
        lines.append("   & BVAR                       & "
                     + " & ".join(fmt(v) for v in bvals) + " \\\\")
        if vi < len(VARIABLES) - 1:
            lines.append("\\addlinespace")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption*{{\\footnotesize Note: {NOTES[key]}\\par}}",
        "\\end{table}",
    ]
    return "\n".join(lines), anomalies


def check_excovid_duplicates(ld: Loader) -> list[str]:
    """Flag any (spec,var,h) whose COVID-excluded RMSE equals the
    full-sample RMSE -- a symptom of a stale/duplicated excovid file."""
    msgs = []
    for code, stem in SPECS:
        for label, qvar, _ in VARIABLES:
            for h in HORIZONS:
                full = ld.qpm(stem, "forecast_rmse_full_excovid.csv",
                              qvar, h, "rmse_full")
                xc = ld.qpm(stem, "forecast_rmse_full_excovid.csv",
                            qvar, h, "rmse_excovid")
                if full is not None and xc is not None and abs(full - xc) < 1e-9:
                    msgs.append(
                        f"excovid == full for {code} {qvar} h={h} "
                        f"(={full:.4f}); excovid file likely stale")
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--treatment", default="_v5plus_free_kappa",
                    help="result-dir suffix (default: headline free_kappa)")
    ap.add_argument("--write", action="store_true",
                    help="write .tex fragments to _generated/")
    args = ap.parse_args()

    ld = Loader(args.treatment)
    print("=" * 72)
    print(f"  Forecast-table generator  (treatment = {args.treatment})")
    print("=" * 72)

    all_anom: list[str] = []
    rendered: dict[str, str] = {}
    for key in ("uncond", "uncond_excovid", "cond", "cond_excovid"):
        latex, anom = build_table(key, ld)
        rendered[key] = latex
        all_anom += anom
        print(f"\n----- {key}  ({TABLES[key]['label']}) -----")
        print(latex)

    print("\n" + "=" * 72)
    print("  PROVENANCE MANIFEST (files actually read)")
    print("=" * 72)
    for f in sorted(ld.used_files):
        print("  read   ", f)
    for f in sorted(ld.missing):
        print("  MISSING", f)

    all_anom += check_excovid_duplicates(ld)
    print("\n" + "=" * 72)
    print(f"  ANOMALY REPORT  ({len(all_anom)} issues)")
    print("=" * 72)
    for m in all_anom:
        print("  !", m)
    if not all_anom:
        print("  none")

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        names = {
            "uncond": "forecast_table_uncond.tex",
            "uncond_excovid": "forecast_table_uncond_excovid.tex",
            "cond": "forecast_table_cond.tex",
            "cond_excovid": "forecast_table_cond_excovid.tex",
        }
        for key, fn in names.items():
            (OUT_DIR / fn).write_text(rendered[key] + "\n")
            print(f"  wrote {OUT_DIR / fn}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

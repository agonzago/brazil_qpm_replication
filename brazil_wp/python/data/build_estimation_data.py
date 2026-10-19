"""Build estimation_data.csv from consolidated_quarterly_data.csv.

Python port of the MATLAB pipeline at
    application_brazil/Model_/scripts_brazil_qom_tot_v2/app_data_processing.m

The MATLAB script applies a sequence of transformations (rebase CPI indices,
take log levels, compute annualized log differences, HP-filter the
external commodity gaps, derive PREM_BAR / PREM_GAP from the CDS5 spread,
and create OBS_* aliases) to produce the estimation file consumed by the
Python QPM runners (load_brazil_data() in run_brazil_*_v2.py).

Outputs to brazil_wp/python/data/estimation_data.csv with the same column
schema as the previously hand-curated / fix_obs_tt-patched file.

The OBS_TT (terms-of-trade observable) is sourced from IC_Br_USD, the
BCB's Brazilian-export-weighted commodity price index in USD (BCB SGS
29042) — NOT from SGS 11753, which was incorrectly labelled as ToT but
is in fact the IPCA-deflated bilateral real exchange rate.  This matches
line 81 of the original MATLAB script.

Usage:
    python data/build_estimation_data.py
    python data/build_estimation_data.py --start 2004Q1 --end 2025Q2 --no-write
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(THIS_DIR)            # brazil_wp/python/
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "estimation_data",
                             "consolidated_quarterly_data.csv")
DEFAULT_OUTPUT = os.path.join(THIS_DIR, "estimation_data.csv")

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
HP_LAMBDA = 1600           # HP filter smoothing parameter (quarterly default)
REBASE_QUARTER = "2018-01-01"   # 2018Q1 = 100 for CPI-style indices
DEFAULT_START = "2004-01-01"    # 2004Q1
DEFAULT_END = "2025-06-30"      # 2025Q2  (matches current estimation_data.csv)

# ----------------------------------------------------------------------
# MATLAB → consolidated column mapping (verified to machine epsilon)
# ----------------------------------------------------------------------
SHORT_NAME_TO_CONSOLIDATED = {
    # Levels (CPI-style, get rebased to 2018Q1=100 then take log)
    "CPI":         "CPI_Brazil",
    "CPIX":        "Core_IPCA_Average",
    "CPI_FOOD":    "CPI_FOOD",
    "CPI_ENERGY":  "CPI_ENERGY",
    "CPI_US":      "US_CPI",
    # Macro variables passed through as-is
    "RS":          "Brazil_Policy_Rate",
    "RS_US":       "US_Fed_Funds_Rate",
    "PREM":        "CDS5_Brazil",
    # GDP_GAP_US: use the CBO-based FRED gap (100*(GDPC1-GDPPOT)/GDPPOT,
    # computed in fred.py).  HLW also publishes an "Output Gap" column
    # (US_Output_Gap_HLW) but its values are not a standard business-cycle
    # gap: it reaches +7.2 in COVID 2020 and only -2.6 at the GFC trough,
    # both economically implausible.  The FRED/CBO-based gap matches
    # standard intuition (-8.8 in 2020Q2, -5.0 in 2009Q2, +1.5 recent).
    "GDP_GAP_US":  "US_Output_Gap",
    "RR_BAR_US":   "US_Natural_Rate",
    "DA_LCPI_TAR": "Brazil_Inflation_Target",
    "FOCUS_IPCA_12M": "IPCA_12M_Expectation",
    # Levels (real-economy series; log only, no rebase)
    "GDP_RAW":     "GDP_Brazil_SA",
    "S_RAW":       "Exchange_Rate",
    # Commodity indices (USD)
    "IC_BR_USD":   "IC_Br_USD",
    "PFOODW":      "Global_Food_Price_Index",
    "POILAPSP":    "Global_Oil_Price_WTI",
}

# ----------------------------------------------------------------------
# OBS_* aliases.  The QPM runners' load_brazil_data() reads exactly these
# 15 observables (see e.g. obs_names list in run_brazil_wp_v2.py:418).
# Level CPI series (LCPI, LCPIX, LCPI_US, etc.) are NOT directly observed
# by the QPM — only their annualised log differences (DA_*) are.  LCPI_US
# in particular is kept in the file only as an intermediate for the
# real-commodity deflation step (LRWOIL = 100*log(POIL) - LCPI_US).
# ----------------------------------------------------------------------
OBS_VARS = [
    "LGDP",
    "DA_LCPI", "DA_LCPIX", "DA_LCPI_ENERGY", "DA_LCPI_US", "DA_LS",
    "LRWOIL_GAP", "LRWFOOD_GAP",
    "GDP_GAP_US",
    "PREM",
    "RS", "RS_US",
    "TT",
    "DA_LCPI_TAR",   # inflation target (called D4A_LCPI_TND in MATLAB)
    "RR_BAR_US",
    "TT_BAR",        # HP-trend of 100*log IC_Br_USD; used by TT-anchor variant
]

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def rebase_to_100(s: pd.Series, base_date: str = REBASE_QUARTER) -> pd.Series:
    """Rebase an index series so that base_date = 100.  Returns a copy."""
    base_date = pd.Timestamp(base_date)
    if base_date not in s.index or pd.isna(s.loc[base_date]) or s.loc[base_date] == 0:
        return s.copy()
    return s * (100.0 / float(s.loc[base_date]))


def log100(s: pd.Series) -> pd.Series:
    """Return 100 * log(s), with non-positive values set to NaN."""
    return 100.0 * np.log(s.where(s > 0))


def diff_annualized(s: pd.Series) -> pd.Series:
    """Annualized log difference (4 * one-period log diff for quarterly data)."""
    return 4.0 * s.diff()


def hp_filter(s: pd.Series, lamb: float = HP_LAMBDA) -> tuple[pd.Series, pd.Series]:
    """HP filter via statsmodels — returns (cycle, trend)."""
    from statsmodels.tsa.filters.hp_filter import hpfilter
    s_clean = s.dropna()
    cycle, trend = hpfilter(s_clean, lamb=lamb)
    # Reindex back to the original index so callers can join easily
    return cycle.reindex(s.index), trend.reindex(s.index)


# ----------------------------------------------------------------------
# Main build
# ----------------------------------------------------------------------
def build(consolidated: pd.DataFrame,
          start: str = DEFAULT_START,
          end: str = DEFAULT_END) -> pd.DataFrame:
    """Replicate app_data_processing.m on a consolidated quarterly DataFrame.

    Parameters
    ----------
    consolidated : DataFrame
        Output of ``data/consolidate.py`` — DatetimeIndex (quarter starts),
        columns matching ``variable_registry.REGISTRY`` canonical names.
    start, end : str
        Estimation-window bounds (inclusive).

    Returns
    -------
    DataFrame indexed at quarter-start with all level / log-level / DA
    series + OBS_* aliases used by the QPM runners.
    """
    df = consolidated.copy()
    df.index = pd.to_datetime(df.index)

    # ---- Step 1 -- rebase CPI/price indices to 2018Q1 = 100 -----------
    cpi_rebase = ["CPI_Brazil", "Core_IPCA_Average",
                  "CPI_FOOD", "CPI_ENERGY", "US_CPI"]
    for col in cpi_rebase:
        if col in df.columns:
            df[col] = rebase_to_100(df[col])

    # Output frame (build column-by-column, then concat at end to avoid
    # the "highly fragmented DataFrame" pandas PerformanceWarning)
    cols: dict[str, pd.Series] = {}

    # ---- Step 2 -- short-name aliases for level series ----------------
    # CPI* (already rebased above)
    cols["CPI"]         = df["CPI_Brazil"]
    cols["CPIX"]        = df["Core_IPCA_Average"]
    cols["CPI_FOOD"]    = df["CPI_FOOD"]
    cols["CPI_ENERGY"]  = df["CPI_ENERGY"]
    if "US_CPI" in df.columns:
        cols["CPI_US"] = df["US_CPI"]
    # Macro pass-throughs
    cols["RS"]           = df["Brazil_Policy_Rate"]
    cols["RS_US"]        = df["US_Fed_Funds_Rate"]
    cols["PREM"]         = df["CDS5_Brazil"]
    cols["GDP_GAP_US"]   = df["US_Output_Gap"]   # CBO-based; see registry note
    cols["RR_BAR_US"]    = df["US_Natural_Rate"]
    cols["DA_LCPI_TAR"]  = df["Brazil_Inflation_Target"]
    if "IPCA_12M_Expectation" in df.columns:
        cols["FOCUS_IPCA_12M"] = df["IPCA_12M_Expectation"]

    # ---- Step 3 -- log levels (100*log) -------------------------------
    cols["LGDP"]        = log100(df["GDP_Brazil_SA"])
    cols["LS"]          = log100(df["Exchange_Rate"])
    cols["LCPI"]        = log100(cols["CPI"])
    cols["LCPIX"]       = log100(cols["CPIX"])
    cols["LCPI_FOOD"]   = log100(cols["CPI_FOOD"])
    cols["LCPI_ENERGY"] = log100(cols["CPI_ENERGY"])
    if "CPI_US" in cols:
        cols["LCPI_US"] = log100(cols["CPI_US"])

    # ---- Step 4 -- annualized log differences -------------------------
    cols["DA_LGDP"]        = diff_annualized(cols["LGDP"])
    cols["DA_LS"]          = diff_annualized(cols["LS"])
    cols["DA_LCPI"]        = diff_annualized(cols["LCPI"])
    cols["DA_LCPIX"]       = diff_annualized(cols["LCPIX"])
    cols["DA_LCPI_FOOD"]   = diff_annualized(cols["LCPI_FOOD"])
    cols["DA_LCPI_ENERGY"] = diff_annualized(cols["LCPI_ENERGY"])
    if "LCPI_US" in cols:
        cols["DA_LCPI_US"] = diff_annualized(cols["LCPI_US"])

    # ---- Step 5 -- external commodity gaps (real, USD-deflated) -------
    # Only computable when US_CPI is present.  When LCPI_US is missing
    # (e.g. FRED unavailable) these columns become NaN columns; the
    # caller can decide whether to error or fall back to a cached file.
    if "LCPI_US" in cols and "Global_Food_Price_Index" in df.columns:
        lrwfood = log100(df["Global_Food_Price_Index"]) - cols["LCPI_US"]
        cols["LRWFOOD"] = lrwfood
        cycle, trend = hp_filter(lrwfood)
        cols["LRWFOOD_GAP"] = cycle
        cols["LRWFOOD_BAR"] = trend

    if "LCPI_US" in cols and "Global_Oil_Price_WTI" in df.columns:
        lrwoil = log100(df["Global_Oil_Price_WTI"]) - cols["LCPI_US"]
        cols["LRWOIL"] = lrwoil
        # MATLAB script restricts the oil HP filter to start at 2004Q1 to
        # avoid a long pre-sample skewing the trend.  Replicate that.
        lrwoil_post2004 = lrwoil[lrwoil.index >= pd.Timestamp("2004-01-01")]
        cycle, trend = hp_filter(lrwoil_post2004)
        cols["LRWOIL_GAP"] = cycle.reindex(lrwoil.index)
        cols["LRWOIL_BAR"] = trend.reindex(lrwoil.index)

    # ---- Step 6 -- terms-of-trade proxy from IC_Br_USD ----------------
    # ALWAYS use IC_Br_USD (BCB SGS 29042), not SGS 11753.  See module
    # docstring + audit comment in variable_registry.py.
    if "IC_Br_USD" not in df.columns:
        raise RuntimeError(
            "IC_Br_USD missing from consolidated data; cannot build OBS_TT. "
            "Re-run the BCB SGS fetcher.")
    tot = log100(df["IC_Br_USD"])
    cols["TOT"] = tot
    cycle, trend = hp_filter(tot)
    cols["TT_GAP"] = cycle
    cols["TT_BAR"] = trend
    cols["TT"] = tot   # observable: the level itself (HP-trend handled in QPM)

    # ---- Step 7 -- premium decomposition ------------------------------
    cycle, trend = hp_filter(cols["PREM"])
    cols["PREM_GAP"] = cycle
    cols["PREM_BAR"] = trend

    # ---- Step 8 -- subset to estimation window ------------------------
    out = pd.concat(cols, axis=1)
    out.index = pd.to_datetime(out.index)
    out = out.loc[(out.index >= pd.Timestamp(start)) &
                  (out.index <= pd.Timestamp(end))]

    # ---- Step 9 -- create OBS_* aliases for the QPM runners -----------
    for name in OBS_VARS:
        if name in out.columns:
            out[f"OBS_{name}"] = out[name]
        else:
            print(f"  WARNING: {name} missing — OBS_{name} not created")

    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"Path to consolidated_quarterly_data.csv "
                             f"(default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output estimation_data.csv "
                             f"(default: {DEFAULT_OUTPUT})")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"Estimation window start (default: {DEFAULT_START})")
    parser.add_argument("--end", default=DEFAULT_END,
                        help=f"Estimation window end   (default: {DEFAULT_END})")
    parser.add_argument("--no-write", action="store_true",
                        help="Build, summarise and exit without writing.")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip backing up the existing output file.")
    args = parser.parse_args()

    print(f"Loading consolidated data: {args.input}")
    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1
    con = pd.read_csv(args.input, index_col=0)
    print(f"  Shape: {con.shape}, index {con.index[0]} .. {con.index[-1]}")

    print(f"\nBuilding estimation data ({args.start} .. {args.end})...")
    est = build(con, start=args.start, end=args.end)
    print(f"  Output shape: {est.shape}")
    print(f"  Columns ({len(est.columns)}): {list(est.columns)[:10]}...")

    if args.no_write:
        print("\n[no-write]  Skipping CSV write.")
        return 0

    # Backup existing file before overwriting
    if os.path.exists(args.output) and not args.no_backup:
        bak = args.output + ".bak.pre_build"
        if not os.path.exists(bak):
            shutil.copy2(args.output, bak)
            print(f"  Backup -> {bak}")

    est_save = est.copy()
    est_save.index.name = "date"
    est_save.to_csv(args.output)
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

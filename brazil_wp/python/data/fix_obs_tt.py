"""Patch estimation_data.csv: rebuild OBS_TT from IC_Br_USD instead of SGS 11753.

Background
----------
SGS 11753 was incorrectly treated as a terms-of-trade index throughout the
data pipeline.  It is in fact the IPCA-deflated bilateral real exchange
rate (BRL/USD, base Jun-1994 = 100); higher = more depreciated BRL.
Empirical fingerprint (2008Q4 ≈ 458, 2011Q1 ≈ 419, 2020Q2 ≈ 501) tracks
BRL strength one-for-one and runs *opposite* to global commodity prices
(corr with WTI = -0.53).

The correct commodity-cycle observable for a Brazilian QPM is the IC-Br
USD (BCB SGS 29042): the BCB's official commodity-price index, weighted
by the composition of Brazilian exports.  This script rebuilds the `TT`
and `OBS_TT` columns of estimation_data.csv as 100·log(IC_Br_USD)
aggregated to quarterly means.

Two CSVs are patched:
  - brazil_wp/python/data/estimation_data.csv
  - gpm_estimation/data/brazil/estimation_data.csv  (toolbox copy)
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(THIS_DIR)                         # brazil_wp/python/
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))        # gpm_ideas/

CSV_PRIMARY  = os.path.join(SCRIPT_DIR, "data", "estimation_data.csv")
CSV_TOOLBOX  = os.path.join(REPO_ROOT, "gpm_estimation", "data", "brazil",
                            "estimation_data.csv")
RAW_MONTHLY  = os.path.join(SCRIPT_DIR, "shared_data", "raw_monthly_data.csv")


def build_quarterly_obs_tt() -> pd.Series:
    """Build OBS_TT as 100*log(IC_Br_USD) aggregated to quarterly means."""
    raw = pd.read_csv(RAW_MONTHLY, index_col=0)
    raw.index = pd.to_datetime(raw.index)
    if "IC_Br_USD" not in raw.columns:
        raise RuntimeError("IC_Br_USD column not found in raw_monthly_data.csv")
    monthly = raw["IC_Br_USD"].dropna()
    # log * 100 (paper convention)
    log100 = 100.0 * np.log(monthly)
    # Quarterly mean
    quarterly = log100.resample("QS").mean()
    quarterly.name = "OBS_TT"
    return quarterly


def patch_csv(csv_path: str, obs_tt_q: pd.Series, *, write: bool) -> None:
    """Replace TT and OBS_TT columns in csv_path with the IC-Br-USD-derived series."""
    if not os.path.exists(csv_path):
        print(f"  SKIP  {csv_path}  (not found)")
        return

    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="Q").to_timestamp()

    if "TT" not in df.columns or "OBS_TT" not in df.columns:
        print(f"  SKIP  {csv_path}  (no TT/OBS_TT columns)")
        return

    # Align IC-Br-USD-derived series to the CSV's date index
    aligned = obs_tt_q.reindex(df.index)
    n_aligned = aligned.notna().sum()
    n_target  = df["OBS_TT"].notna().sum()
    print(f"  {os.path.relpath(csv_path, REPO_ROOT)}")
    print(f"     aligned IC-Br quarters: {n_aligned} / {n_target} required")
    print(f"     OLD OBS_TT range: [{df['OBS_TT'].min():7.2f}, {df['OBS_TT'].max():7.2f}], "
          f"mean {df['OBS_TT'].mean():7.2f}")
    print(f"     NEW OBS_TT range: [{aligned.min():7.2f}, {aligned.max():7.2f}], "
          f"mean {aligned.mean():7.2f}")

    df_new = df.copy()
    df_new["TT"]     = aligned.values
    df_new["OBS_TT"] = aligned.values

    if write:
        bak = csv_path + ".bak.pre_obstt_fix"
        if not os.path.exists(bak):
            shutil.copy2(csv_path, bak)
            print(f"     backup -> {os.path.relpath(bak, REPO_ROOT)}")
        # Write back with the date column named 'date'
        df_save = df_new.copy()
        df_save.index.name = "date"
        df_save.to_csv(csv_path)
        print(f"     written.")
    else:
        print("     [dry-run]  no write")


def main() -> int:
    obs_tt_q = build_quarterly_obs_tt()
    print(f"\nBuilt new OBS_TT from IC_Br_USD (BCB SGS 29042):")
    print(f"  Quarterly observations: {obs_tt_q.dropna().size}")
    print(f"  Range: [{obs_tt_q.min():.2f}, {obs_tt_q.max():.2f}], "
          f"mean {obs_tt_q.mean():.2f}")
    print()

    write = "--write" in sys.argv
    if not write:
        print("DRY RUN.  Use --write to actually patch the CSVs.\n")

    print("Patching CSV files:")
    for csv_path in (CSV_PRIMARY, CSV_TOOLBOX):
        patch_csv(csv_path, obs_tt_q, write=write)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

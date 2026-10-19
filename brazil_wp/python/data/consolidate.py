"""Consolidate all data sources into a single quarterly DataFrame.

Runs all fetchers in parallel, merges results, computes derived
variables (Core IPCA Average, US Output Gap), and saves the
consolidated quarterly CSV.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .fetchers.base import fetch_all_parallel
from .fetchers.bcb_sgs import BCBSGSFetcher
from .fetchers.bcb_expectations import BCBExpectationsFetcher
from .fetchers.sidra import SIDRAFetcher
from .fetchers.fred import FREDFetcher
from .fetchers.nyfed import NYFedFetcher
from .fetchers.local_csv import LocalCSVFetcher
# from .fetchers.fiscal import FiscalFetcher  # SGS codes returning 404 as of 2026-02
from .variable_registry import REGISTRY

# Default output directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_DIR = os.path.normpath(os.path.join(
    _THIS_DIR, "..",
    "estimation_data",
))


def _compute_core_ipca_average(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Core_IPCA_Average by averaging q/q log-growth rates
    across available core measures and cumulating into a composite index.

    This avoids the pitfall of averaging index LEVELS that start at 100
    at different dates.
    """
    core_cols = [c for c in df.columns if c.startswith("Core_IPCA_")]
    if not core_cols:
        return df

    m = df[core_cols].copy()
    # q/q log-differences for each core index
    dlog_m = np.log(m).diff()
    # Average growth rates across available measures
    avg_dlog = dlog_m.mean(axis=1)

    # Cumulate into composite index (first valid period = 100)
    idx = pd.Series(np.nan, index=df.index)
    first_valid = avg_dlog.first_valid_index()
    if first_valid is not None:
        pos = df.index.get_loc(first_valid)
        start = max(0, pos - 1)
        idx.iloc[start] = 100.0
        for i in range(start + 1, len(idx)):
            if not np.isnan(avg_dlog.iloc[i]) and not np.isnan(idx.iloc[i - 1]):
                idx.iloc[i] = idx.iloc[i - 1] * np.exp(avg_dlog.iloc[i])

    df["Core_IPCA_Average"] = idx
    return df


def consolidate(
    start: str = "1990-01-01",
    end: str = "2025-12-31",
    max_workers: int = 7,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Run all fetchers in parallel and produce consolidated quarterly data.

    Parameters
    ----------
    start, end : str
        Date range for fetching.
    max_workers : int
        ThreadPoolExecutor workers.
    output_dir : str or None
        Directory to save CSVs (default: python/estimation_data/).

    Returns
    -------
    pd.DataFrame
        Consolidated quarterly DataFrame (PeriodIndex).
    """
    if output_dir is None:
        output_dir = _CSV_DIR

    t0 = time.perf_counter()

    # Instantiate all fetchers
    fetchers = [
        BCBSGSFetcher(),
        BCBExpectationsFetcher(),
        SIDRAFetcher(),
        FREDFetcher(),
        NYFedFetcher(),
        LocalCSVFetcher(),
        # FiscalFetcher(),  # SGS codes returning 404 as of 2026-02
    ]

    # Parallel fetch
    results = fetch_all_parallel(fetchers, start, end,
                                 max_workers=max_workers)

    if not results:
        print("  No data fetched from any source.")
        return pd.DataFrame()

    # Merge all DataFrames on the quarterly PeriodIndex
    frames = list(results.values())
    consolidated = frames[0]
    for df in frames[1:]:
        consolidated = consolidated.join(df, how="outer")

    consolidated = consolidated.sort_index()

    # Compute derived variables
    consolidated = _compute_core_ipca_average(consolidated)

    # Quality checks
    dup_cols = consolidated.columns[consolidated.columns.duplicated()]
    if len(dup_cols) > 0:
        print(f"  WARNING: duplicate columns: {list(dup_cols)}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Consolidation complete in {elapsed:.1f}s")
    print(f"  Shape: {consolidated.shape[0]} quarters x "
          f"{consolidated.shape[1]} variables")
    if len(consolidated) > 0:
        print(f"  Range: {consolidated.index[0]} to {consolidated.index[-1]}")

    # Completeness
    n = len(consolidated)
    complete = (consolidated.notna().sum() / n * 100).round(1)
    n_full = (complete == 100).sum()
    n_80 = ((complete >= 80) & (complete < 100)).sum()
    n_50 = ((complete >= 50) & (complete < 80)).sum()
    n_low = (complete < 50).sum()
    print(f"  Completeness: {n_full} at 100%, {n_80} at 80-99%, "
          f"{n_50} at 50-79%, {n_low} below 50%")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "consolidated_quarterly_data.csv")
    df_save = consolidated.copy()
    df_save.index = df_save.index.to_timestamp()
    df_save.index.name = "date"
    df_save.to_csv(out_path)
    print(f"  Saved: {out_path}")

    return consolidated

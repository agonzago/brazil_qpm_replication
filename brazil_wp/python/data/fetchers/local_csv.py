"""Local CSV file fetcher.

Reads static data files that cannot be downloaded programmatically:
  - CDS_5yr_10yr.csv  (Bloomberg-licensed, manual update)
  - Inflation_target.csv  (BCB policy decisions)

These files live in python/data/static_data/.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from ..variable_registry import VariableDef, get_by_source
from .base import BaseFetcher, to_quarterly

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.normpath(os.path.join(
    _THIS_DIR, "..",
    "static_data",
))


# ---------------------------------------------------------------------------
# CDS 5-year spread
# ---------------------------------------------------------------------------

def _load_cds(start: str, end: str) -> pd.DataFrame:
    """Load CDS5 from static CSV.

    File format: index=YYYYM#, CDS5, CDS10 (monthly, basis points).
    R pipeline takes last monthly value per quarter, divides by 100.
    """
    path = os.path.join(_STATIC_DIR, "CDS_5yr_10yr.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CDS file not found: {path}")

    df = pd.read_csv(path, index_col=0)

    # Parse YYYYM# index → datetime
    dates = []
    for code in df.index:
        m = re.match(r"(\d{4})M(\d{1,2})", str(code))
        if m:
            dates.append(pd.Timestamp(int(m.group(1)), int(m.group(2)), 1))
        else:
            dates.append(pd.NaT)
    df.index = pd.DatetimeIndex(dates)
    df = df.dropna(subset=["CDS5"])

    # Filter date range
    df = df.loc[start:end]
    if df.empty:
        return pd.DataFrame()

    # Last monthly value per quarter, then divide by 100
    # (R pipeline: select last month per quarter, then /100)
    df_q = df[["CDS5"]].resample("QS").last()
    df_q["CDS5_Brazil"] = df_q["CDS5"] / 100.0
    df_q = df_q[["CDS5_Brazil"]].dropna()
    df_q.index = df_q.index.to_period("Q")
    return df_q


# ---------------------------------------------------------------------------
# Inflation target
# ---------------------------------------------------------------------------

def _load_inflation_target(start: str, end: str) -> pd.DataFrame:
    """Load BCB inflation target (center, lower, upper bounds).

    File format: Year, Date (announcement), Target (%), Interval Size,
                 Lower Bound, Upper Bound.
    We create a step function: for each year, the target announced
    for that year applies for the entire year.
    """
    path = os.path.join(_STATIC_DIR, "Inflation_target.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Inflation target file not found: {path}")

    raw = pd.read_csv(path)
    # Columns: Year, Date, Target (%), Interval Size +/- (p.p.),
    #          Lower Bound, Upper Bound

    # Build quarterly series: each year gets a constant target
    records = []
    for _, row in raw.iterrows():
        year = int(row["Year"])
        target = float(row["Target (%)"])
        lower = float(row["Lower Bound"])
        upper = float(row["Upper Bound"])
        for q in range(4):
            month = q * 3 + 1
            dt = pd.Timestamp(year=year, month=month, day=1)
            records.append({
                "date": dt,
                "Brazil_Inflation_Target": target,
                "Brazil_Inflation_Target_Lower": lower,
                "Brazil_Inflation_Target_Upper": upper,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("date").sort_index()
    df = df.loc[start:end]

    if df.empty:
        return pd.DataFrame()

    df.index = df.index.to_period("Q")
    # Remove duplicate quarters (keep first, as year-based)
    df = df[~df.index.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# LocalCSVFetcher
# ---------------------------------------------------------------------------

class LocalCSVFetcher(BaseFetcher):
    """Load static data files (CDS, inflation targets)."""

    @property
    def name(self) -> str:
        return "LOCAL"

    @property
    def variables(self) -> list[VariableDef]:
        return get_by_source("LOCAL")

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        # CDS
        try:
            cds = _load_cds(start, end)
            if not cds.empty:
                frames.append(cds)
                print(f"    LOCAL CDS5: {len(cds)} rows")
        except Exception as exc:
            print(f"    LOCAL CDS5: {exc}")

        # Inflation target
        try:
            target = _load_inflation_target(start, end)
            if not target.empty:
                frames.append(target)
                print(f"    LOCAL Inflation Target: {len(target)} rows")
        except Exception as exc:
            print(f"    LOCAL Inflation Target: {exc}")

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=1)
        return df

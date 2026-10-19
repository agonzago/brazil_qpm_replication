"""NY Fed HLW natural rate fetcher.

Fetches Holston-Laubach-Williams (HLW) estimates of the US natural
rate of interest (r*), trend growth, and output gap from the
New York Federal Reserve.

Primary: download Excel from NY Fed website.
Fallback: read static CSV in python/data/static_data/.

Source: https://www.newyorkfed.org/research/policy/rstar
"""

from __future__ import annotations

import io
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd

from ..variable_registry import VariableDef, get_by_source
from .base import BaseFetcher

_HLW_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/"
    "economists/williams/data/"
    "Holston_Laubach_Williams_current_estimates.xlsx"
)

# Resolve the project root for static fallback
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STATIC_CSV = os.path.normpath(os.path.join(
    _THIS_DIR, "..",
    "static_data", "nyfed_US_Natural_Rate.csv",
))


def _download_hlw_excel() -> pd.DataFrame:
    """Download the HLW Excel file and parse the US estimates sheet.

    The Excel sheet "HLW Estimates" has:
        Column A (row 7+): Date
        Column C: US Trend Growth (g)
        Column K: US Natural Rate (r*)
        Column O: US Output Gap
    """
    req = urllib.request.Request(_HLW_URL)
    req.add_header("User-Agent", "Python/brazil_wp_pipeline")

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()

    # Write to a temp file because openpyxl needs a seekable stream
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    try:
        tmp.write(data)
        tmp.close()

        # Read specific columns from known cell ranges
        dates = pd.read_excel(tmp.name, sheet_name="HLW Estimates",
                              usecols="A", skiprows=5, header=None,
                              names=["date"])
        trend = pd.read_excel(tmp.name, sheet_name="HLW Estimates",
                              usecols="C", skiprows=5, header=None,
                              names=["US_Trend_Growth"])
        rstar = pd.read_excel(tmp.name, sheet_name="HLW Estimates",
                              usecols="K", skiprows=5, header=None,
                              names=["US_Natural_Rate"])
        gap = pd.read_excel(tmp.name, sheet_name="HLW Estimates",
                            usecols="O", skiprows=5, header=None,
                            names=["US_Output_Gap_HLW"])
    finally:
        os.unlink(tmp.name)

    df = pd.concat([dates, trend, rstar, gap], axis=1)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S",
                                errors="coerce")
    df = df.dropna(subset=["date", "US_Natural_Rate"])
    df = df.set_index("date").sort_index()
    return df


def _load_static_csv() -> pd.DataFrame:
    """Load the static fallback CSV."""
    if not os.path.isfile(_STATIC_CSV):
        raise FileNotFoundError(f"Static HLW CSV not found: {_STATIC_CSV}")
    df = pd.read_csv(_STATIC_CSV, parse_dates=["date"], index_col="date")
    return df.sort_index()


class NYFedFetcher(BaseFetcher):
    """Fetch HLW natural rate estimates from NY Fed."""

    @property
    def name(self) -> str:
        return "NYFED"

    @property
    def variables(self) -> list[VariableDef]:
        return get_by_source("NYFED")

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        # Try download first, fall back to static CSV
        try:
            df = _download_hlw_excel()
            print("    NYFED: downloaded fresh HLW data")
            # Update static CSV for future fallback
            try:
                df.to_csv(_STATIC_CSV)
            except Exception:
                pass
        except Exception as exc:
            print(f"    NYFED download failed ({exc}), using static CSV")
            df = _load_static_csv()

        # Data is already quarterly; convert index to PeriodIndex
        df.index = pd.to_datetime(df.index)
        df = df.loc[start:end]

        if df.empty:
            return pd.DataFrame()

        df.index = df.index.to_period("Q")
        return df

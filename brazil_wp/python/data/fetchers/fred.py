"""FRED API fetcher.

Fetches US macroeconomic data from the Federal Reserve Economic Data
(FRED) API using stdlib urllib (no fredapi package required).

API: https://api.stlouisfed.org/fred/series/observations
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pandas as pd

from ..variable_registry import VariableDef, get_by_source
from .base import BaseFetcher, to_quarterly

# ---------------------------------------------------------------------------
# FRED series to fetch
# ---------------------------------------------------------------------------

_FRED_SERIES: dict[str, str] = {
    "US_Potential_GDP": "GDPPOT",
    "US_Real_GDP": "GDPC1",
    "US_NAIRU": "NROU",
    "US_CPI": "CPIAUCSL",
    "US_Fed_Funds_Rate": "FEDFUNDS",
    "Global_Oil_Price_WTI": "POILBREUSDM",
    "Global_Food_Price_Index": "PFOODINDEXM",
}

# Series that are already quarterly (no need to aggregate)
_QUARTERLY_SERIES = {"US_Potential_GDP", "US_Real_GDP", "US_NAIRU"}


def _get_fred_api_key() -> str:
    """Retrieve FRED API key from environment or ~/.Renviron."""
    key = os.environ.get("FRED_API_KEY", "")
    if key:
        return key

    # Fall back to ~/.Renviron (shared with R pipeline)
    renviron = os.path.expanduser("~/.Renviron")
    if os.path.isfile(renviron):
        with open(renviron) as f:
            for line in f:
                line = line.strip()
                if line.startswith("FRED_API_KEY"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        return parts[1].strip().strip("'\"")
    raise RuntimeError(
        "FRED_API_KEY not found. Set it as an env var or in ~/.Renviron."
    )


def _fetch_fred_json(series_id: str, api_key: str,
                     start: str | None = None,
                     end: str | None = None) -> list[dict]:
    """Fetch a single FRED series as JSON observations."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
    )
    if start:
        url += f"&observation_start={start}"
    if end:
        url += f"&observation_end={end}"

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    return data.get("observations", [])


def _fred_to_series(obs: list[dict], name: str) -> pd.Series:
    """Convert FRED observations JSON to a pandas Series."""
    dates, vals = [], []
    for rec in obs:
        try:
            dt = pd.to_datetime(rec["date"])
            val_str = rec.get("value", ".")
            if val_str == ".":
                continue
            val = float(val_str)
            dates.append(dt)
            vals.append(val)
        except (ValueError, KeyError):
            continue
    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=name)
    return s.sort_index()


class FREDFetcher(BaseFetcher):
    """Fetch US macro series from FRED API."""

    @property
    def name(self) -> str:
        return "FRED"

    @property
    def variables(self) -> list[VariableDef]:
        return get_by_source("FRED")

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        api_key = _get_fred_api_key()

        quarterly_frames: list[pd.Series] = []
        monthly_frames: list[pd.Series] = []

        for var_name, series_id in _FRED_SERIES.items():
            try:
                obs = _fetch_fred_json(series_id, api_key, start, end)
                s = _fred_to_series(obs, var_name)
                if var_name in _QUARTERLY_SERIES:
                    quarterly_frames.append(s)
                else:
                    monthly_frames.append(s)
            except Exception as exc:
                print(f"    FRED {var_name} ({series_id}): {exc}")
                continue

        frames_q: list[pd.DataFrame] = []

        # Already-quarterly series: convert to PeriodIndex directly
        if quarterly_frames:
            df_q_raw = pd.concat(quarterly_frames, axis=1)
            df_q_raw.index = pd.to_datetime(df_q_raw.index)
            df_q_raw.index = df_q_raw.index.to_period("Q")
            frames_q.append(df_q_raw)

        # Monthly series: aggregate to quarterly mean
        if monthly_frames:
            df_m = pd.concat(monthly_frames, axis=1)
            df_m.index = pd.to_datetime(df_m.index)
            df_m_q = to_quarterly(df_m, method="mean")
            frames_q.append(df_m_q)

        if not frames_q:
            return pd.DataFrame()

        df_q = pd.concat(frames_q, axis=1)

        # Compute US Output Gap = 100 * (Real - Potential) / Potential
        if "US_Potential_GDP" in df_q.columns and "US_Real_GDP" in df_q.columns:
            pot = df_q["US_Potential_GDP"]
            real = df_q["US_Real_GDP"]
            gap = 100.0 * (real - pot) / pot
            df_q["US_Output_Gap"] = gap

        return df_q

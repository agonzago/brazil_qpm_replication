"""BCB SGS API fetcher.

Fetches Brazilian macroeconomic series from the Banco Central do Brasil
Sistema Gerenciador de Séries Temporais (SGS).

API: https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from ..variable_registry import VariableDef, get_by_source
from .base import BaseFetcher, to_quarterly


# SGS series codes for this fetcher
_SGS_SERIES: dict[str, int] = {
    # Commodity indices
    "IC_Br_BRL": 27574,
    "IC_Br_USD": 29042,
    "IC_Br_Agro_BRL": 27575,
    "IC_Br_Agro_USD": 29041,
    "IC_Br_Energy_BRL": 27577,
    "IC_Br_Energy_USD": 29039,
    "IC_Br_Metal_BRL": 27576,
    "IC_Br_Metal_USD": 29040,
    # Terms of trade
    "Terms_of_Trade": 11753,
    # Exchange rate
    "Exchange_Rate": 1,
    # CPI (monthly % variation)
    "CPI_Brazil": 433,
    # GDP seasonally adjusted
    "GDP_Brazil_SA": 22109,
    # IBC-Br
    "IBC_Br": 24363,
    # Core IPCA measures (monthly % variation)
    "Core_IPCA_EX0": 11427,
    "Core_IPCA_EX3": 27839,
    "Core_IPCA_Trimmed_Smoothed": 4466,
    "Core_IPCA_Double_Weight": 16122,
    "Core_IPCA_P55": 28750,
    # Policy rate
    "Brazil_Policy_Rate": 1178,
}

# Variables whose raw data is monthly % variation and must be cumulated
# into a price index (base=100).
_CUMULATE_TO_INDEX = {
    "CPI_Brazil",
    "Core_IPCA_EX0",
    "Core_IPCA_EX3",
    "Core_IPCA_Trimmed_Smoothed",
    "Core_IPCA_Double_Weight",
    "Core_IPCA_P55",
}


def _fetch_sgs_json(
    code: int,
    start: str = "01/01/1990",
    end: str | None = None,
) -> list[dict]:
    """Fetch a single SGS series as JSON.

    BCB enforces a ~10-year window for daily series (Exchange Rate,
    Selic). We chunk into 9-year windows to stay within the limit.
    """
    if end is None:
        end = datetime.now().strftime("%d/%m/%Y")

    start_dt = datetime.strptime(start, "%d/%m/%Y")
    end_dt = datetime.strptime(end, "%d/%m/%Y")

    all_records: list[dict] = []
    chunk_start = start_dt

    while chunk_start <= end_dt:
        # 9-year window (safely under the 10-year API limit)
        chunk_end = min(
            chunk_start.replace(year=chunk_start.year + 9) - pd.Timedelta(days=1),
            end_dt,
        )
        s_fmt = chunk_start.strftime("%d/%m/%Y")
        e_fmt = chunk_end.strftime("%d/%m/%Y")

        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}"
            f"/dados?formato=json&dataInicial={s_fmt}&dataFinal={e_fmt}"
        )
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")

        # Retry on transient 5xx errors; skip 404 chunks (series may
        # not exist for the requested date range)
        raw = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except HTTPError as e:
                if e.code == 404:
                    break  # no data for this chunk — skip it
                if e.code >= 500 and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise

        if raw is not None:
            records = json.loads(raw)
            if isinstance(records, list):
                all_records.extend(records)

        chunk_start = chunk_end + pd.Timedelta(days=1)

    return all_records


def _sgs_to_series(data: list[dict], name: str) -> pd.Series:
    """Convert SGS JSON to a pandas Series with DatetimeIndex."""
    dates, vals = [], []
    for rec in data:
        try:
            dt = pd.to_datetime(rec["data"], dayfirst=True)
            val = float(rec["valor"].replace(",", "."))
            dates.append(dt)
            vals.append(val)
        except (ValueError, KeyError):
            continue
    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=name)
    return s.sort_index()


def _cumulate_pct_to_index(pct_series: pd.Series) -> pd.Series:
    """Cumulate monthly % variations into a price index (base=100)."""
    factors = 1.0 + pct_series.dropna() / 100.0
    idx = 100.0 * factors.cumprod()
    idx.name = pct_series.name
    return idx


class BCBSGSFetcher(BaseFetcher):
    """Fetch ~20 series from BCB SGS API."""

    @property
    def name(self) -> str:
        return "BCB_SGS"

    @property
    def variables(self) -> list[VariableDef]:
        return get_by_source("BCB_SGS")

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        start_fmt = pd.Timestamp(start).strftime("%d/%m/%Y")
        end_fmt = pd.Timestamp(end).strftime("%d/%m/%Y")

        monthly_frames: list[pd.Series] = []

        for var_name, code in _SGS_SERIES.items():
            try:
                raw = _fetch_sgs_json(code, start_fmt, end_fmt)
                s = _sgs_to_series(raw, var_name)

                if var_name in _CUMULATE_TO_INDEX:
                    s = _cumulate_pct_to_index(s)

                monthly_frames.append(s)
            except Exception as exc:
                print(f"    BCB SGS {var_name} (code={code}): {exc}")
                continue

        if not monthly_frames:
            return pd.DataFrame()

        # Combine all monthly series
        df_monthly = pd.concat(monthly_frames, axis=1)
        df_monthly.index = pd.to_datetime(df_monthly.index)

        # Aggregate to quarterly (mean)
        df_q = to_quarterly(df_monthly, method="mean")

        return df_q

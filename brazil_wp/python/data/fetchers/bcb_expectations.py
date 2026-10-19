"""BCB Olinda API fetcher — IPCA 12-month expectations.

Fetches median IPCA 12-month inflation expectations from the BCB
Olinda (OData) API.

API: https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/
     ExpectativasMercadoInflacao12Meses
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from ..variable_registry import VariableDef, get_by_source
from .base import BaseFetcher, to_quarterly


_OLINDA_BASE = (
    "https://olinda.bcb.gov.br/olinda/servico/Expectativas"
    "/versao/v1/odata/ExpectativasMercadoInflacao12Meses"
)


def _fetch_expectations_json() -> list[dict]:
    """Fetch IPCA 12M expectations from BCB Olinda API."""
    params = (
        "$top=100000"
        "&$format=json"
        "&$select=Data,Mediana"
        "&$filter=Indicador%20eq%20'IPCA'"
    )
    url = f"{_OLINDA_BASE}?{params}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")

    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    return data.get("value", [])


class BCBExpectationsFetcher(BaseFetcher):
    """Fetch IPCA 12-month median expectations from BCB Olinda API."""

    @property
    def name(self) -> str:
        return "BCB_OLINDA"

    @property
    def variables(self) -> list[VariableDef]:
        return get_by_source("BCB_OLINDA")

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        records = _fetch_expectations_json()

        dates, vals = [], []
        for rec in records:
            try:
                dt = pd.to_datetime(str(rec["Data"])[:10])
                val = float(rec["Mediana"])
                dates.append(dt)
                vals.append(val)
            except (ValueError, KeyError, TypeError):
                continue

        if not dates:
            return pd.DataFrame()

        s = pd.Series(vals, index=pd.DatetimeIndex(dates),
                       name="IPCA_12M_Expectation")
        s = s.sort_index()

        # Daily data: take median per day (multiple respondents), then
        # aggregate to quarterly mean.
        daily = s.groupby(s.index).median()
        df_daily = daily.to_frame()

        # Filter to requested date range
        df_daily = df_daily.loc[start:end]

        if df_daily.empty:
            return pd.DataFrame()

        df_q = to_quarterly(df_daily, method="mean")
        return df_q

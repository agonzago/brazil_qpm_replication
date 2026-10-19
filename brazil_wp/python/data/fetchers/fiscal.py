"""BCB SGS fiscal series fetcher.

Fetches Brazilian fiscal data from the Banco Central do Brasil
SGS API (same endpoint as bcb_sgs.py but for fiscal variables):
  - Nominal GDP
  - Gross Debt (General Government)
  - Net Debt (Public Sector)
  - Primary Balance (Consolidated Public Sector)

API: https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..variable_registry import VariableDef, get_by_source
from .base import BaseFetcher, to_quarterly
from .bcb_sgs import _fetch_sgs_json, _sgs_to_series

# Fiscal SGS codes (subset of what the R pipeline fetches)
_FISCAL_SERIES: dict[str, int] = {
    "Nominal_GDP_BRL_Mil": 4191,
    "Gross_Debt_GG_PctGDP_NewMethod": 13762,
    "Net_Debt_PS_PctGDP": 4503,
    "Primary_Balance_Consolidated_PctGDP": 4506,
}


class FiscalFetcher(BaseFetcher):
    """Fetch fiscal series from BCB SGS API."""

    @property
    def name(self) -> str:
        return "FISCAL"

    @property
    def variables(self) -> list[VariableDef]:
        return [v for v in get_by_source("BCB_SGS") if v.group == "fiscal"]

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        start_fmt = pd.Timestamp(start).strftime("%d/%m/%Y")
        end_fmt = pd.Timestamp(end).strftime("%d/%m/%Y")

        all_series: list[pd.Series] = []

        for var_name, code in _FISCAL_SERIES.items():
            try:
                raw = _fetch_sgs_json(code, start_fmt, end_fmt)
                s = _sgs_to_series(raw, var_name)
                all_series.append(s)
            except Exception as exc:
                print(f"    FISCAL {var_name} (code={code}): {exc}")
                continue

        if not all_series:
            return pd.DataFrame()

        df = pd.concat(all_series, axis=1)
        df.index = pd.to_datetime(df.index)

        # Detect frequency: if >6 obs/year, it's monthly → aggregate
        if len(df) > 0:
            year_span = (df.index.max() - df.index.min()).days / 365.25
            obs_per_year = len(df) / max(year_span, 1)
            if obs_per_year > 6:
                df_q = to_quarterly(df, method="mean")
            else:
                # Already quarterly
                df_q = df.copy()
                df_q.index = df_q.index.to_period("Q")
        else:
            df_q = pd.DataFrame()

        return df_q

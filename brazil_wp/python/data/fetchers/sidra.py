"""IBGE SIDRA API fetcher.

Fetches Brazilian GDP and CPI component data from IBGE's SIDRA
(Sistema IBGE de Recuperacao Automatica) REST API.

API: https://apisidra.ibge.gov.br/values/t/{table}/...

Field mapping (from SIDRA JSON):
  D1C = Brasil code         D1N = Brasil name
  D2C = Variable code       D2N = Variable name
  D3C = Period code          D3N = Period name
  D4C = Classification code  D4N = Classification name
  V   = Value
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from ..variable_registry import VariableDef, get_by_source
from .base import BaseFetcher, to_quarterly

# ---------------------------------------------------------------------------
# SIDRA API helpers
# ---------------------------------------------------------------------------

_SIDRA_BASE = "https://apisidra.ibge.gov.br/values"


def _sidra_get(table: int, variable: int, period: str = "all",
               classific: str | None = None,
               category: str | None = None) -> list[dict]:
    """Fetch a SIDRA table and return data rows (header row removed)."""
    url = f"{_SIDRA_BASE}/t/{table}/n1/all/v/{variable}/p/{period}"
    if classific and category:
        url += f"/{classific}/{category}"

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")

    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = resp.read().decode("utf-8")
    rows = json.loads(raw)

    # First element is column-name metadata (V="Valor", etc.) — skip it
    if rows and isinstance(rows[0], dict):
        first_v = rows[0].get("V", "")
        if isinstance(first_v, str) and not first_v.replace(".", "").replace("-", "").replace(",", "").isdigit():
            rows = rows[1:]
    return rows


# ---------------------------------------------------------------------------
# GDP — Table 1620
# ---------------------------------------------------------------------------

def _fetch_gdp() -> pd.DataFrame:
    """Quarterly GDP chained volume index (1995=100) from Table 1620."""
    rows = _sidra_get(1620, 583, period="all",
                      classific="c11255", category="90707")

    dates, vals = [], []
    for r in rows:
        try:
            code = r.get("D3C", "")          # Quarter code e.g. "199601"
            val_str = r.get("V", "")
            if val_str in ("", "-", "...", None):
                continue
            val = float(val_str)
            year = int(code[:4])
            qtr = int(code[4:6])
            month = (qtr - 1) * 3 + 1
            dt = pd.Timestamp(year=year, month=month, day=1)
            dates.append(dt)
            vals.append(val)
        except (ValueError, KeyError, TypeError):
            continue

    if not dates:
        return pd.DataFrame()

    df = pd.DataFrame({"GDP_Brazil_Quarterly": vals},
                      index=pd.DatetimeIndex(dates))
    df = df.sort_index()
    df.index = df.index.to_period("Q")
    return df


# ---------------------------------------------------------------------------
# CPI components — Tables 7060, 1419, 2938, 655, 656
# ---------------------------------------------------------------------------

_VAR_VARIATION = 63    # Monthly variation %
_VAR_WEIGHT = 66       # Monthly weight %

_CAT_FOOD = "7170"     # Alimentacao e bebidas
_CAT_ELEC = "7485"     # Energia eletrica residencial
_CAT_FUELS = "7656"    # Combustiveis (veiculos)

_CPI_TABLES = [
    (7060, [_VAR_VARIATION, _VAR_WEIGHT]),   # 2020-present
    (1419, [_VAR_VARIATION, _VAR_WEIGHT]),   # 2012-2019
    (2938, [_VAR_VARIATION, _VAR_WEIGHT]),   # 2006-2011
    (655,  [_VAR_VARIATION]),                 # 1999-2005 variation
    (656,  [_VAR_WEIGHT]),                    # 1999-2005 weight
]


def _fetch_cpi_components() -> pd.DataFrame:
    """CPI Food, Energy indices and weights from SIDRA tables."""
    all_rows: list[dict] = []

    for table, variables in _CPI_TABLES:
        for var in variables:
            cats = ",".join([_CAT_FOOD, _CAT_ELEC, _CAT_FUELS])
            try:
                rows = _sidra_get(table, var, period="all",
                                  classific="c315", category=cats)
                all_rows.extend(rows)
            except Exception:
                continue

    if not all_rows:
        return pd.DataFrame()

    # Parse into structured records
    # SIDRA field mapping:
    #   D3C = Month code (e.g. "202001")
    #   D2C = Variable code (e.g. "63" for variation, "66" for weight)
    #   D4C = Category code (e.g. "7170" for food)
    #   V   = Value
    records: list[dict] = []
    for r in all_rows:
        try:
            month_code = r.get("D3C", "")
            var_code = str(r.get("D2C", ""))
            cat_code = str(r.get("D4C", ""))
            val = r.get("V", "")
            if val in ("", "-", "...", None):
                continue
            val = float(val)
            year = int(month_code[:4])
            month = int(month_code[4:6])
            dt = pd.Timestamp(year=year, month=month, day=1)
            records.append({
                "date": dt,
                "var_code": var_code,
                "cat_code": cat_code,
                "value": val,
            })
        except (ValueError, KeyError, TypeError):
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Map component and kind
    comp_map = {_CAT_FOOD: "FOOD", _CAT_ELEC: "ELEC", _CAT_FUELS: "FUELS"}
    df["comp"] = df["cat_code"].map(comp_map)
    df["kind"] = df["var_code"].apply(
        lambda x: "variation" if str(x) == str(_VAR_VARIATION) else "weight"
    )
    df = df.dropna(subset=["comp"])

    # Pivot
    wide = df.pivot_table(index="date", columns=["comp", "kind"],
                          values="value", aggfunc="first")
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.sort_index()

    result_frames = []

    # --- FOOD index ---
    if "FOOD_variation" in wide.columns:
        food_var = wide["FOOD_variation"].dropna()
        food_idx = _cumulate_variation(food_var)
        food_idx.name = "CPI_FOOD"
        result_frames.append(food_idx)

    # --- ENERGY index (weighted avg of ELEC + FUELS) ---
    elec_v = wide.get("ELEC_variation", pd.Series(dtype=float))
    fuels_v = wide.get("FUELS_variation", pd.Series(dtype=float))
    elec_w = wide.get("ELEC_weight", pd.Series(dtype=float))
    fuels_w = wide.get("FUELS_weight", pd.Series(dtype=float))

    if not elec_v.empty and not fuels_v.empty:
        combined = pd.DataFrame({
            "ev": elec_v, "fv": fuels_v,
            "ew": elec_w, "fw": fuels_w,
        }).dropna()
        wsum = combined["ew"] + combined["fw"]
        energy_var = (combined["ew"] * combined["ev"] +
                      combined["fw"] * combined["fv"]) / wsum.replace(0, np.nan)
        energy_idx = _cumulate_variation(energy_var.dropna())
        energy_idx.name = "CPI_ENERGY"
        result_frames.append(energy_idx)

    # --- Weights ---
    if "FOOD_weight" in wide.columns:
        w_food = wide["FOOD_weight"].copy()
        w_food.name = "W_FOOD"
        result_frames.append(w_food)

    if "ELEC_weight" in wide.columns and "FUELS_weight" in wide.columns:
        w_energy = (wide["ELEC_weight"].fillna(0) +
                    wide["FUELS_weight"].fillna(0))
        w_energy.name = "W_ENERGY"
        result_frames.append(w_energy)

    if not result_frames:
        return pd.DataFrame()

    df_monthly = pd.concat(result_frames, axis=1)
    df_q = to_quarterly(df_monthly, method="mean")
    return df_q


def _cumulate_variation(pct_series: pd.Series, base: float = 100.0) -> pd.Series:
    """Cumulate monthly % variations into a price index (base=100)."""
    factors = 1.0 + pct_series / 100.0
    idx = base * factors.cumprod()
    return idx


# ---------------------------------------------------------------------------
# SIDRAFetcher
# ---------------------------------------------------------------------------

class SIDRAFetcher(BaseFetcher):
    """Fetch GDP + CPI components from IBGE SIDRA API."""

    @property
    def name(self) -> str:
        return "SIDRA"

    @property
    def variables(self) -> list[VariableDef]:
        return get_by_source("SIDRA")

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        # GDP
        try:
            gdp = _fetch_gdp()
            if not gdp.empty:
                frames.append(gdp)
                print(f"    SIDRA GDP: {len(gdp)} rows")
        except Exception as exc:
            print(f"    SIDRA GDP: {exc}")

        # CPI components
        try:
            cpi = _fetch_cpi_components()
            if not cpi.empty:
                frames.append(cpi)
                print(f"    SIDRA CPI components: {cpi.shape}")
        except Exception as exc:
            print(f"    SIDRA CPI components: {exc}")

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=1)
        # Filter date range
        start_p = pd.Period(start, freq="Q")
        end_p = pd.Period(end, freq="Q")
        df = df.loc[(df.index >= start_p) & (df.index <= end_p)]
        return df

"""Canonical variable registry for the Brazil working paper.

Single source of truth for variable names, sources, transformations,
and cross-pipeline mappings (consolidated → QPM → BVAR).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VariableDef:
    """Definition of a single data variable."""

    name: str
    source: str          # BCB_SGS | BCB_OLINDA | SIDRA | FRED | NYFED | LOCAL
    series_code: str     # API code or filename
    description: str
    frequency: str       # daily | monthly | quarterly
    unit: str            # index | pct_pa | pct_mom | bps | brl_usd | pct_gdp | usd
    aggregation: str     # mean | last | sum  (monthly/daily → quarterly)
    transforms: tuple[str, ...] = ()
    qpm_name: str | None = None
    bvar_q_name: str | None = None     # quarterly BVAR column
    bvar_m_name: str | None = None     # monthly BVAR column
    group: str = "brazil"              # brazil | us | fiscal | local


# ======================================================================== #
#  Brazil — BCB SGS
# ======================================================================== #

_BCB_SGS: list[VariableDef] = [
    # ── Commodity Indices ──
    VariableDef("IC_Br_BRL", "BCB_SGS", "27574",
                "IC-Br Commodity Index — General (BRL)",
                "monthly", "index", "mean",
                ("log", "DA", "log100_hp", "log100_hamilton", "log100_lt"),
                bvar_q_name="dlog_commod_brl", bvar_m_name="dlog_commod_brl"),
    # OBS_TT (commodity-exporter terms-of-trade proxy) is sourced from the
    # IC-Br USD index: BCB's official commodity-price index, weighted by the
    # composition of Brazilian exports (iron ore, soy, oil, sugar, etc.).
    # Earlier versions of the pipeline mistakenly mapped OBS_TT to SGS 11753,
    # which is the bilateral real exchange rate (IPCA-deflated, Jun-1994=100,
    # vs USD), not the terms of trade.  See `.claude/memory.md` for the audit
    # log.  USD denomination is used to keep the observable a pure external
    # commodity-cycle driver; the BRL-side response is endogenous via UIP /
    # LZ_GAP.
    VariableDef("IC_Br_USD", "BCB_SGS", "29042",
                "IC-Br Commodity Index — General (USD); BCB Brazilian-export-weighted commodity basket",
                "monthly", "index", "mean",
                ("log", "DA", "log100_hp", "log100_hamilton", "log100_lt"),
                qpm_name="OBS_TT"),
    VariableDef("IC_Br_Agro_BRL", "BCB_SGS", "27575",
                "IC-Br Commodity Index — Agriculture (BRL)",
                "monthly", "index", "mean",
                ("log", "DA", "log100_hp", "log100_hamilton", "log100_lt")),
    VariableDef("IC_Br_Agro_USD", "BCB_SGS", "29041",
                "IC-Br Commodity Index — Agriculture (USD)",
                "monthly", "index", "mean",
                ("log", "DA", "log100_hp", "log100_hamilton", "log100_lt")),
    VariableDef("IC_Br_Energy_BRL", "BCB_SGS", "27577",
                "IC-Br Commodity Index — Energy (BRL)",
                "monthly", "index", "mean",
                ("log", "DA", "log100_hp", "log100_hamilton", "log100_lt")),
    VariableDef("IC_Br_Energy_USD", "BCB_SGS", "29039",
                "IC-Br Commodity Index — Energy (USD)",
                "monthly", "index", "mean",
                ("log", "DA", "log100_hp", "log100_hamilton", "log100_lt")),
    VariableDef("IC_Br_Metal_BRL", "BCB_SGS", "27576",
                "IC-Br Commodity Index — Metals (BRL)",
                "monthly", "index", "mean",
                ("log", "DA", "log100_hp", "log100_hamilton", "log100_lt")),
    VariableDef("IC_Br_Metal_USD", "BCB_SGS", "29040",
                "IC-Br Commodity Index — Metals (USD)",
                "monthly", "index", "mean",
                ("log", "DA", "log100_hp", "log100_hamilton", "log100_lt")),

    # WARNING: SGS 11753 is the IPCA-deflated bilateral real exchange rate
    # (BRL/USD, base Jun-1994 = 100), NOT the terms of trade.  The legacy
    # name "Terms_of_Trade" is preserved here only because both the fetcher
    # dict (bcb_sgs.py::_SGS_SERIES) and the cached raw_monthly_data.csv
    # column key on this name; renaming would break the cache.  Higher
    # values = more depreciated BRL in real terms (so this series moves
    # opposite to commodity prices, not with them).  No qpm_name binding:
    # OBS_TT is now sourced from IC_Br_USD above.  Audit: 2026-05-09.
    VariableDef("Terms_of_Trade", "BCB_SGS", "11753",
                "[MISLABELED LEGACY NAME] BCB SGS 11753 = bilateral real exchange rate BRL/USD (IPCA, Jun-1994=100), NOT terms of trade",
                "monthly", "index", "mean",
                ("log", "log100_hp", "log100_hamilton", "log100_lt")),

    # ── Exchange Rate ──
    VariableDef("Exchange_Rate", "BCB_SGS", "1",
                "Exchange Rate BRL/USD (buy, daily average)",
                "daily", "brl_usd", "mean",
                ("log", "DA", "D4A", "log100_hp", "log100_hamilton", "log100_lt"),
                qpm_name="OBS_DA_LS",
                bvar_q_name="dlog_brl_usd", bvar_m_name="dlog_brl_usd"),

    # ── Headline CPI ──
    VariableDef("CPI_Brazil", "BCB_SGS", "433",
                "IPCA headline CPI (monthly % variation → index)",
                "monthly", "pct_mom", "mean",
                ("log", "DA", "D4A"),
                qpm_name="OBS_DA_LCPI",
                bvar_q_name="ipca_demeaned"),

    # ── GDP ──
    VariableDef("GDP_Brazil_SA", "BCB_SGS", "22109",
                "GDP Seasonally Adjusted (BCB monthly proxy)",
                "monthly", "index", "mean",
                ("log", "DA", "D4A", "log100_hp", "log100_hamilton", "log100_lt"),
                qpm_name="OBS_LGDP",
                bvar_q_name="gdp_gap"),

    # ── IBC-Br ──
    VariableDef("IBC_Br", "BCB_SGS", "24363",
                "IBC-Br Economic Activity Index (SA)",
                "monthly", "index", "mean",
                (),
                bvar_m_name="ibc_growth"),

    # ── Core IPCA measures ──
    VariableDef("Core_IPCA_EX0", "BCB_SGS", "11427",
                "Core IPCA EX0 (monthly % variation → index)",
                "monthly", "pct_mom", "mean",
                ("log", "DA", "D4A")),
    VariableDef("Core_IPCA_EX3", "BCB_SGS", "27839",
                "Core IPCA EX3 (monthly % variation → index)",
                "monthly", "pct_mom", "mean",
                ("log", "DA", "D4A")),
    VariableDef("Core_IPCA_Trimmed_Smoothed", "BCB_SGS", "4466",
                "Core IPCA Smoothed Trimmed Mean (monthly % var → index)",
                "monthly", "pct_mom", "mean",
                ("log", "DA", "D4A")),
    VariableDef("Core_IPCA_Double_Weight", "BCB_SGS", "16122",
                "Core IPCA Double Weighted (monthly % var → index)",
                "monthly", "pct_mom", "mean",
                ("log", "DA", "D4A")),
    VariableDef("Core_IPCA_P55", "BCB_SGS", "28750",
                "Core IPCA P55 (monthly % var → index)",
                "monthly", "pct_mom", "mean",
                ("log", "DA", "D4A")),

    # ── Policy Rate ──
    VariableDef("Brazil_Policy_Rate", "BCB_SGS", "1178",
                "Selic Target Rate (% p.a.)",
                "daily", "pct_pa", "mean",
                (),
                qpm_name="OBS_RS",
                bvar_q_name="selic_rate", bvar_m_name="selic_rate"),
]


# ======================================================================== #
#  Brazil — BCB Olinda (Expectations)
# ======================================================================== #

_BCB_OLINDA: list[VariableDef] = [
    VariableDef("IPCA_12M_Expectation", "BCB_OLINDA", "ExpectativasMercadoInflacao12Meses",
                "IPCA 12-Month Ahead Median Expectation (% p.a.)",
                "daily", "pct_pa", "mean",
                (),
                bvar_q_name="exp_inf"),
]


# ======================================================================== #
#  Brazil — IBGE SIDRA
# ======================================================================== #

_SIDRA: list[VariableDef] = [
    VariableDef("GDP_Brazil_Quarterly", "SIDRA", "t1620/v583",
                "Quarterly GDP — Chained Volume Index (1995=100)",
                "quarterly", "index", "direct",
                ("log", "DA", "D4A")),
    VariableDef("CPI_FOOD", "SIDRA", "t7060",
                "Food & Beverages CPI Component (index, from multiple table vintages)",
                "monthly", "index", "mean",
                ("log", "DA", "D4A")),
    VariableDef("CPI_ENERGY", "SIDRA", "t7060",
                "Energy CPI Component — Electricity + Fuels (index)",
                "monthly", "index", "mean",
                ("log", "DA", "D4A"),
                qpm_name="OBS_DA_LCPI_ENERGY"),
    VariableDef("W_FOOD", "SIDRA", "t7060",
                "Food Weight in CPI Basket (%)",
                "monthly", "pct_pa", "mean", ()),
    VariableDef("W_ENERGY", "SIDRA", "t7060",
                "Energy Weight in CPI Basket (%)",
                "monthly", "pct_pa", "mean", ()),
]


# ======================================================================== #
#  US — FRED
# ======================================================================== #

_FRED: list[VariableDef] = [
    VariableDef("US_Potential_GDP", "FRED", "GDPPOT",
                "US CBO Potential GDP (Billions of Chained 2017 Dollars)",
                "quarterly", "index", "direct", ()),
    VariableDef("US_Real_GDP", "FRED", "GDPC1",
                "US Real GDP (Billions of Chained 2017 Dollars)",
                "quarterly", "index", "direct", ()),
    VariableDef("US_NAIRU", "FRED", "NROU",
                "US Natural Rate of Unemployment (%)",
                "quarterly", "pct_pa", "direct", ()),
    VariableDef("US_CPI", "FRED", "CPIAUCSL",
                "US CPI All Urban Consumers (1982-84=100)",
                "monthly", "index", "mean",
                ("log", "DA", "D4A"),
                qpm_name="OBS_DA_LCPI_US"),
    VariableDef("US_Fed_Funds_Rate", "FRED", "FEDFUNDS",
                "US Federal Funds Effective Rate (% p.a.)",
                "monthly", "pct_pa", "mean",
                (),
                qpm_name="OBS_RS_US"),
    VariableDef("Global_Oil_Price_WTI", "FRED", "POILBREUSDM",
                "Oil Price — Brent Crude (USD/barrel, monthly avg)",
                "monthly", "usd", "mean",
                ("log", "DA", "log100_hp", "log100_hamilton", "log100_lt"),
                qpm_name="OBS_LRWOIL_GAP"),
    VariableDef("Global_Food_Price_Index", "FRED", "PFOODINDEXM",
                "IMF Global Food Price Index (2016=100)",
                "monthly", "index", "mean",
                ("log", "DA", "log100_hp", "log100_hamilton", "log100_lt"),
                qpm_name="OBS_LRWFOOD_GAP"),
]


# ======================================================================== #
#  US — NY Fed HLW
# ======================================================================== #

_NYFED: list[VariableDef] = [
    VariableDef("US_Natural_Rate", "NYFED", "rstar",
                "Holston-Laubach-Williams Natural Rate of Interest (r*)",
                "quarterly", "pct_pa", "direct", ()),
    VariableDef("US_Trend_Growth", "NYFED", "trend_growth",
                "HLW Trend Growth Rate (%)",
                "quarterly", "pct_pa", "direct", ()),
    VariableDef("US_Output_Gap_HLW", "NYFED", "output_gap",
                "HLW Output Gap (%)",
                "quarterly", "pct_pa", "direct",
                (),
                qpm_name="OBS_GDP_GAP_US"),
]


# ======================================================================== #
#  Local / Static files (manual update)
# ======================================================================== #

_LOCAL: list[VariableDef] = [
    VariableDef("CDS5_Brazil", "LOCAL", "CDS_5yr_10yr.csv",
                "5-Year CDS Spread (basis points, Bloomberg)",
                "monthly", "bps", "last",
                (),
                qpm_name="OBS_PREM",
                bvar_q_name="cds5", bvar_m_name="cds5",
                group="local"),
    VariableDef("Brazil_Inflation_Target", "LOCAL", "Inflation_target.csv",
                "BCB Inflation Target Center (% p.a.)",
                "quarterly", "pct_pa", "mean",
                (),
                qpm_name="OBS_DA_LCPI_TAR",
                group="local"),
    VariableDef("Brazil_Inflation_Target_Upper", "LOCAL", "Inflation_target.csv",
                "BCB Inflation Target Upper Bound (% p.a.)",
                "quarterly", "pct_pa", "mean", (), group="local"),
    VariableDef("Brazil_Inflation_Target_Lower", "LOCAL", "Inflation_target.csv",
                "BCB Inflation Target Lower Bound (% p.a.)",
                "quarterly", "pct_pa", "mean", (), group="local"),
]


# ======================================================================== #
#  Fiscal — BCB SGS
# ======================================================================== #

_FISCAL: list[VariableDef] = [
    VariableDef("Nominal_GDP_BRL_Mil", "BCB_SGS", "4191",
                "Nominal Quarterly GDP (R$ millions)",
                "quarterly", "index", "direct",
                ("log", "DA", "D4A"), group="fiscal"),
    VariableDef("Gross_Debt_GG_PctGDP_NewMethod", "BCB_SGS", "13762",
                "Gross Debt (General Government) % GDP — New Method",
                "quarterly", "pct_gdp", "direct", (), group="fiscal"),
    VariableDef("Net_Debt_PS_PctGDP", "BCB_SGS", "4503",
                "Net Debt (Public Sector) % GDP",
                "quarterly", "pct_gdp", "direct", (), group="fiscal"),
    VariableDef("Primary_Balance_Consolidated_PctGDP", "BCB_SGS", "4506",
                "Primary Balance (Consolidated PS) % GDP",
                "quarterly", "pct_gdp", "direct", (), group="fiscal"),
]


# ======================================================================== #
#  Computed / Derived variables (created during consolidation)
# ======================================================================== #

_COMPUTED: list[VariableDef] = [
    VariableDef("Core_IPCA_Average", "COMPUTED", "mean(EX0,EX3,Trimmed,DW,P55)",
                "Average of BCB Core IPCA Measures (index)",
                "quarterly", "index", "direct",
                ("log", "DA", "D4A")),
    VariableDef("US_Output_Gap", "COMPUTED", "100*(GDPC1-GDPPOT)/GDPPOT",
                "US Output Gap (CBO-based, %)",
                "quarterly", "pct_pa", "direct", ()),
]


# ======================================================================== #
#  Master registry
# ======================================================================== #

REGISTRY: tuple[VariableDef, ...] = tuple(
    _BCB_SGS + _BCB_OLINDA + _SIDRA + _FRED + _NYFED + _LOCAL + _FISCAL + _COMPUTED
)

# Fast lookup by canonical name
_BY_NAME: dict[str, VariableDef] = {v.name: v for v in REGISTRY}
_BY_QPM: dict[str, VariableDef] = {v.qpm_name: v for v in REGISTRY if v.qpm_name}
_BY_BVAR_Q: dict[str, VariableDef] = {v.bvar_q_name: v for v in REGISTRY if v.bvar_q_name}


def get(name: str) -> VariableDef:
    """Look up a variable by canonical name."""
    return _BY_NAME[name]


def get_by_qpm(qpm_name: str) -> VariableDef:
    """Look up a variable by QPM observable name."""
    return _BY_QPM[qpm_name]


def get_by_source(source: str) -> list[VariableDef]:
    """Return all variables from a given source."""
    return [v for v in REGISTRY if v.source == source]


def get_all_sgs_codes() -> dict[str, int]:
    """Return {canonical_name: SGS_code} for all BCB SGS variables."""
    return {
        v.name: int(v.series_code)
        for v in REGISTRY
        if v.source == "BCB_SGS"
    }


def get_all_fred_codes() -> dict[str, str]:
    """Return {canonical_name: FRED_series_id} for all FRED variables."""
    return {v.name: v.series_code for v in REGISTRY if v.source == "FRED"}


def print_registry_table() -> None:
    """Print the full registry as a markdown table."""
    print("| Name | Source | Code | Description | Freq | Unit | QPM | BVAR(Q) | BVAR(M) |")
    print("|------|--------|------|-------------|------|------|-----|---------|---------|")
    for v in REGISTRY:
        print(
            f"| {v.name} | {v.source} | {v.series_code} | "
            f"{v.description[:40]} | {v.frequency} | {v.unit} | "
            f"{v.qpm_name or ''} | {v.bvar_q_name or ''} | {v.bvar_m_name or ''} |"
        )


if __name__ == "__main__":
    print_registry_table()

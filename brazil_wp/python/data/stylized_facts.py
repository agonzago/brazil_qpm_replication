"""Stylized facts plots for the Brazil working paper.

Replicates the R stylized_facts_plots.R output using matplotlib.
Produces dual-axis z-scored time series comparisons and
inflation-vs-target plots.

Output: PNG files in paper/plots/stylized_facts/
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.normpath(os.path.join(
    _THIS_DIR, "..", "..",
    "paper", "plots", "stylized_facts",
))
_CSV_DIR = os.path.normpath(os.path.join(
    _THIS_DIR, "..",
    "estimation_data",
))


# ======================================================================== #
#  Plotting helpers
# ======================================================================== #

def _zscore(s: pd.Series) -> pd.Series:
    """Standardise a series to zero mean, unit variance."""
    m, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - m) / sd


def plot_two_series(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    title: str = "",
    y1_label: str | None = None,
    y2_label: str | None = None,
    zscore: bool = True,
    color1: str = "#2E86AB",
    color2: str = "#A23B72",
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot two series on the same axes with optional z-scoring."""
    if var1 not in df.columns or var2 not in df.columns:
        return None

    s1 = df[var1].dropna()
    s2 = df[var2].dropna()
    if len(s1) < 5 or len(s2) < 5:
        return None

    if zscore:
        s1 = _zscore(s1)
        s2 = _zscore(s2)
        suffix = " (Z-scores)"
    else:
        suffix = ""

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    dates1 = s1.index.to_timestamp() if hasattr(s1.index, "to_timestamp") else s1.index
    dates2 = s2.index.to_timestamp() if hasattr(s2.index, "to_timestamp") else s2.index

    ax.plot(dates1, s1.values, color=color1, linewidth=1.5, alpha=0.9,
            label=y1_label or var1)
    ax.plot(dates2, s2.values, color=color2, linewidth=1.5, alpha=0.9,
            label=y2_label or var2)
    ax.set_title(f"{title}{suffix}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_inflation_vs_target(
    df: pd.DataFrame,
    infl_var: str,
    tgt_var: str,
    low_var: str,
    up_var: str,
    title: str = "Inflation vs Target",
    infl_color: str = "#D55E00",
    target_color: str = "#0072B2",
) -> plt.Figure | None:
    """Plot inflation against the BCB target band."""
    required = [infl_var, tgt_var, low_var, up_var]
    if not all(c in df.columns for c in required):
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    dates = df.index.to_timestamp() if hasattr(df.index, "to_timestamp") else df.index

    # Target band
    lo = df[low_var].values
    hi = df[up_var].values
    ax.fill_between(dates, lo, hi, alpha=0.2, color="gray")

    # Target center and inflation
    ax.plot(dates, df[tgt_var].values, color=target_color, linewidth=1.5,
            label="Target")
    ax.plot(dates, df[infl_var].values, color=infl_color, linewidth=1.5,
            label="Inflation")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Percent", fontsize=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ======================================================================== #
#  Plot specifications (mirrors R stylized_facts_plots.R)
# ======================================================================== #

def _get_plot_specs() -> list[dict]:
    """Return the list of plot specifications matching the R pipeline."""
    target = "Brazil_Inflation_Target"
    lower = "Brazil_Inflation_Target_Lower"
    upper = "Brazil_Inflation_Target_Upper"

    infl = "D4A_CPI_Brazil"          # YoY inflation (matches BCB target definition)
    core = "D4A_Core_IPCA_Average"    # YoY core inflation
    gdp = "DA_GDP_Brazil_SA"
    cds5 = "CDS5_Brazil"
    fx = "DA_Exchange_Rate"
    interest = "Brazil_Policy_Rate"
    commod = "DA_IC_Br_BRL"
    exp_var = "IPCA_12M_Expectation"
    d4a_cpi = "D4A_CPI_Brazil"
    da_energy = "DA_CPI_ENERGY"
    da_food = "DA_CPI_FOOD"

    gdp_gap = "log100_hamilton_GDP_Brazil_SA"
    fx_gap = "log100_hamilton_Exchange_Rate"
    commod_gap = "log100_hamilton_IC_Br_BRL"
    commod_gap_usd = "log100_hamilton_IC_Br_USD"
    commod_agro_gap = "log100_hamilton_IC_Br_Agro_BRL"
    commod_energy_gap = "log100_hamilton_IC_Br_Energy_BRL"
    commod_metal_gap = "log100_hamilton_IC_Br_Metal_BRL"
    oil_gap = "log100_hamilton_Global_Oil_Price_WTI"
    food_gap = "log100_hamilton_Global_Food_Price_Index"

    specs = [
        # --- Inflation vs target ---
        {"type": "target", "infl": infl, "tgt": target, "lo": lower, "hi": upper,
         "title": "Inflation vs Target", "name": "inflation_vs_target"},
        {"type": "target", "infl": core, "tgt": target, "lo": lower, "hi": upper,
         "title": "Core Inflation vs Target", "name": "core_inflation_vs_target"},

        # --- Two-series z-score plots ---
        {"v1": core, "v2": gdp, "title": "Core Inflation vs GDP Growth",
         "name": "core_inflation_vs_gdp_growth"},
        {"v1": gdp, "v2": cds5, "title": "GDP Growth vs 5-Year CDS",
         "name": "gdp_growth_vs_cds5"},
        {"v1": fx, "v2": cds5, "title": "Exchange Rate vs 5-Year CDS",
         "name": "exchange_rate_vs_cds5"},
        {"v1": interest, "v2": gdp, "title": "Brazil Interest Rate vs GDP Growth",
         "name": "interest_rate_vs_gdp_growth"},
        {"v1": cds5, "v2": fx, "title": "CDS vs DA Exchange Rate",
         "name": "cds5_vs_exchange_rate"},
        {"v1": cds5, "v2": commod, "title": "CDS vs Commodity Prices",
         "name": "cds5_vs_commodity_prices"},

        # --- Fact 2 analysis ---
        {"v1": cds5, "v2": fx_gap, "title": "CDS Level vs FX Gap",
         "name": "cds_level_vs_fx_gap"},
        {"v1": cds5, "v2": "DA_IC_Br_USD",
         "title": "CDS Level vs Commodity Price Growth (USD)",
         "name": "cds_level_vs_commodity_growth_usd"},
        {"v1": cds5, "v2": gdp, "title": "CDS Level vs GDP Growth",
         "name": "cds_level_vs_gdp_growth"},
        {"v1": fx, "v2": "DA_IC_Br_USD",
         "title": "FX Growth vs Commodity Price Growth (USD)",
         "name": "fx_growth_vs_commodity_growth_usd"},

        # --- Expectations ---
        {"v1": exp_var, "v2": interest,
         "title": "Inflation Expectations vs Interest Rates",
         "name": "inflation_exp_vs_interest_rates"},
        {"v1": exp_var, "v2": d4a_cpi,
         "title": "Inflation Expectations vs YoY Headline Inflation",
         "name": "inflation_exp_vs_yoy_inflation"},

        # --- Gap comparisons ---
        {"v1": gdp_gap, "v2": interest, "title": "GDP Gap vs Interest Rates",
         "name": "gdp_gap_vs_interest_rates"},
        {"v1": gdp_gap, "v2": commod_gap,
         "title": "GDP Gap vs Commodity Prices (BRL, Hamilton Filter)",
         "name": "gdp_gap_vs_commodity_prices"},
        {"v1": gdp_gap, "v2": commod_gap_usd,
         "title": "GDP Gap vs Commodity Prices (USD, Hamilton Filter)",
         "name": "gdp_gap_vs_commodity_prices_usd"},
        {"v1": gdp_gap, "v2": commod_energy_gap,
         "title": "GDP Gap vs Energy Commodity Prices (BRL)",
         "name": "gdp_gap_vs_energy_commodities"},
        {"v1": gdp_gap, "v2": commod_agro_gap,
         "title": "GDP Gap vs Agricultural Commodity Prices (BRL)",
         "name": "gdp_gap_vs_agro_commodities"},
        {"v1": gdp_gap, "v2": commod_metal_gap,
         "title": "GDP Gap vs Metal Commodity Prices (BRL)",
         "name": "gdp_gap_vs_metal_commodities"},

        # --- FX gap ---
        {"v1": fx_gap, "v2": commod_gap,
         "title": "FX Gap vs Commodity Prices (BRL)",
         "name": "fx_gap_vs_commodity_prices"},
        {"v1": fx_gap, "v2": commod_gap_usd,
         "title": "FX Gap vs Commodity Prices (USD)",
         "name": "fx_gap_vs_commodity_prices_usd",
         "color1": "#A23B72", "color2": "#2E86AB"},  # FX gap keeps same color as in CDS vs FX plot

        # --- Component inflation ---
        {"v1": da_energy, "v2": oil_gap,
         "title": "Energy Inflation vs Global Oil Gap",
         "name": "energy_inflation_vs_global_oil_gap",
         "y1_label": "Energy CPI (QoQ ann.)",
         "y2_label": "Global Oil Gap (Hamilton)"},
        {"v1": da_food, "v2": food_gap,
         "title": "Food Inflation vs Global Food Gap",
         "name": "food_inflation_vs_global_food_gap",
         "y1_label": "Food CPI (QoQ ann.)",
         "y2_label": "Global Food Gap (Hamilton)"},
    ]
    return specs


# ======================================================================== #
#  Main entry point
# ======================================================================== #

def generate_stylized_facts(
    transformed: pd.DataFrame | None = None,
    input_path: str | None = None,
    output_dir: str | None = None,
    start: str = "2004-01-01",
    end: str = "2025-12-31",
) -> int:
    """Generate all stylized fact plots.

    Parameters
    ----------
    transformed : DataFrame or None
        If provided, use directly (PeriodIndex).
    input_path : str or None
        Path to transformed_data.csv.
    output_dir : str or None
        Directory for PNG output.
    start, end : str
        Date range filter.

    Returns
    -------
    int
        Number of plots successfully created.
    """
    if output_dir is None:
        output_dir = _IMG_DIR

    # Load data
    if transformed is None:
        if input_path is None:
            input_path = os.path.join(_CSV_DIR, "transformed_data.csv")
        print(f"  Loading: {input_path}")
        df = pd.read_csv(input_path, index_col="date", parse_dates=True)
        df.index = df.index.to_period("Q")
    else:
        df = transformed.copy()

    # Filter date range
    start_p = pd.Period(start, freq="Q")
    end_p = pd.Period(end, freq="Q")
    df = df.loc[(df.index >= start_p) & (df.index <= end_p)]

    print(f"  Data: {df.shape[0]} quarters, {df.shape[1]} columns")
    print(f"  Range: {df.index[0]} to {df.index[-1]}")

    os.makedirs(output_dir, exist_ok=True)

    specs = _get_plot_specs()
    n_ok = 0

    for spec in specs:
        name = spec["name"]
        try:
            if spec.get("type") == "target":
                fig = plot_inflation_vs_target(
                    df, spec["infl"], spec["tgt"],
                    spec["lo"], spec["hi"],
                    title=spec["title"],
                )
            else:
                kwargs = dict(title=spec["title"], zscore=True)
                if "color1" in spec:
                    kwargs["color1"] = spec["color1"]
                if "color2" in spec:
                    kwargs["color2"] = spec["color2"]
                if "y1_label" in spec:
                    kwargs["y1_label"] = spec["y1_label"]
                if "y2_label" in spec:
                    kwargs["y2_label"] = spec["y2_label"]
                fig = plot_two_series(
                    df, spec["v1"], spec["v2"], **kwargs,
                )

            if fig is not None:
                path = os.path.join(output_dir, f"{name}.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                n_ok += 1
            else:
                print(f"    Skipped {name}: missing variables")
        except Exception as exc:
            print(f"    Failed {name}: {exc}")

    print(f"  Generated {n_ok}/{len(specs)} plots in {output_dir}")

    # Print key correlations for substitution into the tex file
    report_correlations(df)

    return n_ok


def report_correlations(df: pd.DataFrame) -> dict:
    """Compute and print key pairwise correlations for the stylized facts."""
    pairs = [
        ("log100_hamilton_GDP_Brazil_SA", "log100_hamilton_IC_Br_USD",
         "GDP gap vs Commodity (USD)"),
        ("log100_hamilton_GDP_Brazil_SA", "log100_hamilton_IC_Br_BRL",
         "GDP gap vs Commodity (BRL)"),
        ("log100_hamilton_GDP_Brazil_SA", "log100_hamilton_IC_Br_Energy_BRL",
         "GDP gap vs Energy Commodity (BRL)"),
        ("log100_hamilton_GDP_Brazil_SA", "log100_hamilton_IC_Br_Energy_USD",
         "GDP gap vs Energy Commodity (USD)"),
        ("CDS5_Brazil", "log100_hamilton_Exchange_Rate",
         "CDS vs FX gap"),
        ("log100_hamilton_Exchange_Rate", "log100_hamilton_IC_Br_USD",
         "FX gap vs Commodity (USD)"),
    ]
    results = {}
    print("\n  Key correlations (Hamilton filter):")
    for v1, v2, label in pairs:
        if v1 in df.columns and v2 in df.columns:
            r = df[[v1, v2]].dropna().corr().iloc[0, 1]
            results[label] = r
            print(f"    {label}: r = {r:.2f}")
        else:
            missing = [v for v in (v1, v2) if v not in df.columns]
            print(f"    {label}: MISSING {missing}")
    return results

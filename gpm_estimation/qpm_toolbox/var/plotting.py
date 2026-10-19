"""Plotting utilities for VAR impulse responses and forecast fan charts."""

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .types import BVARForecastResult, BVARIRFResult, IRFResult


# ======================================================================== #
#  Style defaults
# ======================================================================== #

_COLOURS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _subplot_grid(n: int) -> Tuple[int, int]:
    """Determine a compact subplot grid for *n* panels."""
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


# ======================================================================== #
#  IRF plots
# ======================================================================== #

def plot_irf(
    irf_result,
    shocks: Optional[List[int]] = None,
    responses: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title_prefix: str = "",
) -> List[plt.Figure]:
    """Plot impulse-response functions.

    Handles both :class:`IRFResult` (classical) and
    :class:`BVARIRFResult` (Bayesian with credible bands).

    Args:
        irf_result: IRFResult or BVARIRFResult.
        shocks: Indices of shocks to plot (default: all).
        responses: Indices of responses to plot (default: all).
        figsize: Figure size per shock page.
        title_prefix: Prefix for figure titles.

    Returns:
        List of matplotlib Figure objects (one per shock).
    """
    is_bayesian = isinstance(irf_result, BVARIRFResult)

    if is_bayesian:
        median = irf_result.median     # (K_resp, K_shock, H+1)
        p16 = irf_result.p16
        p84 = irf_result.p84
    else:
        # Classical: irf is (H+1, K_resp, K_shock)
        median = irf_result.irf.transpose(1, 2, 0)  # -> (K_resp, K_shock, H+1)
        p16 = p84 = None

    K_resp, K_shock, Hp1 = median.shape
    H = Hp1 - 1
    horizons = np.arange(Hp1)

    if shocks is None:
        shocks = list(range(K_shock))
    if responses is None:
        responses = list(range(K_resp))

    var_names = irf_result.var_names or [f"Var {i}" for i in range(K_resp)]
    shock_names = irf_result.shock_names or [f"Shock {s}" for s in shocks]

    figs = []
    for si_plot, si in enumerate(shocks):
        n_resp = len(responses)
        nrows, ncols = _subplot_grid(n_resp)
        if figsize is None:
            figsize_use = (4.5 * ncols, 3.0 * nrows)
        else:
            figsize_use = figsize

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize_use)
        if n_resp == 1:
            axes = np.array([axes])
        axes = axes.ravel()

        for ri_plot, ri in enumerate(responses):
            ax = axes[ri_plot]
            ax.plot(horizons, median[ri, si, :],
                    color=_COLOURS[0], lw=1.5, label="Median" if is_bayesian else "IRF")
            if p16 is not None and p84 is not None:
                ax.fill_between(horizons, p16[ri, si, :], p84[ri, si, :],
                                color=_COLOURS[0], alpha=0.2, label="68% band")
            ax.axhline(0, color="black", lw=0.5, ls="--")
            ax.set_title(var_names[ri])
            ax.set_xlabel("Horizon")
            if ri_plot == 0:
                ax.legend(fontsize=7)

        for j in range(len(responses), len(axes)):
            axes[j].set_visible(False)

        sname = shock_names[si_plot] if si_plot < len(shock_names) else f"Shock {si}"
        fig.suptitle(f"{title_prefix}Shock: {sname}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figs.append(fig)

    return figs


# ======================================================================== #
#  IRF comparison overlay
# ======================================================================== #

def plot_irf_comparison(
    irf_results: Sequence,
    labels: List[str],
    shocks: Optional[List[int]] = None,
    responses: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> List[plt.Figure]:
    """Overlay IRFs from multiple models (e.g. pre- vs post-COVID).

    Each entry in *irf_results* can be an :class:`IRFResult` or
    :class:`BVARIRFResult`.  Median lines are plotted for each; bands
    are drawn only for the first result.

    Args:
        irf_results: Sequence of IRF results.
        labels: Display labels (one per result).
        shocks: Shock indices to plot.
        responses: Response indices to plot.
        figsize: Figure size.

    Returns:
        List of Figure objects.
    """
    # Normalise all to (K_resp, K_shock, H+1) median arrays
    medians = []
    bands = []
    for res in irf_results:
        if isinstance(res, BVARIRFResult):
            medians.append(res.median)
            bands.append((res.p16, res.p84))
        else:
            medians.append(res.irf.transpose(1, 2, 0))
            bands.append((None, None))

    K_resp, K_shock, Hp1 = medians[0].shape
    horizons = np.arange(Hp1)

    if shocks is None:
        shocks = list(range(K_shock))
    if responses is None:
        responses = list(range(K_resp))

    ref = irf_results[0]
    var_names = ref.var_names or [f"Var {i}" for i in range(K_resp)]
    shock_names = ref.shock_names or [f"Shock {s}" for s in shocks]

    figs = []
    for si_plot, si in enumerate(shocks):
        n_resp = len(responses)
        nrows, ncols = _subplot_grid(n_resp)
        if figsize is None:
            figsize_use = (4.5 * ncols, 3.0 * nrows)
        else:
            figsize_use = figsize

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize_use)
        if n_resp == 1:
            axes = np.array([axes])
        axes = axes.ravel()

        for ri_plot, ri in enumerate(responses):
            ax = axes[ri_plot]
            for idx, (med, lab) in enumerate(zip(medians, labels)):
                col = _COLOURS[idx % len(_COLOURS)]
                ax.plot(horizons, med[ri, si, :], color=col, lw=1.3, label=lab)
                lo, hi = bands[idx]
                if lo is not None and hi is not None and idx == 0:
                    ax.fill_between(horizons, lo[ri, si, :], hi[ri, si, :],
                                    color=col, alpha=0.15)
            ax.axhline(0, color="black", lw=0.5, ls="--")
            ax.set_title(var_names[ri])
            ax.set_xlabel("Horizon")
            if ri_plot == 0:
                ax.legend(fontsize=7)

        for j in range(len(responses), len(axes)):
            axes[j].set_visible(False)

        sname = shock_names[si_plot] if si_plot < len(shock_names) else f"Shock {si}"
        fig.suptitle(f"IRF Comparison - Shock: {sname}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figs.append(fig)

    return figs


# ======================================================================== #
#  Forecast fan charts
# ======================================================================== #

def plot_fan_chart(
    forecast_result: BVARForecastResult,
    var_names: Optional[List[str]] = None,
    percentiles: Tuple[int, ...] = (10, 25, 50, 75, 90),
    figsize: Optional[Tuple[float, float]] = None,
    history: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Fan-chart visualisation of BVAR forecast distributions.

    Args:
        forecast_result: BVARForecastResult.
        var_names: Override variable names.
        percentiles: Percentiles to shade (outermost to innermost).
        figsize: Figure size.
        history: Optional (T_hist, K) array of historical data to prepend.

    Returns:
        matplotlib Figure.
    """
    fc = forecast_result.forecasts  # (N_draws, H, K)
    H = forecast_result.horizon
    K = fc.shape[2]

    if var_names is None:
        var_names = forecast_result.var_names or [f"Var {i}" for i in range(K)]

    pcts = sorted(percentiles)
    # Group into symmetric pairs around the median
    pairs = []
    for pct in pcts:
        comp = 100 - pct
        if comp in pcts and pct < comp:
            pairs.append((pct, comp))

    nrows, ncols = _subplot_grid(K)
    if figsize is None:
        figsize = (4.5 * ncols, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if K == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    h_axis = np.arange(1, H + 1)

    for j in range(K):
        ax = axes[j]
        fc_j = fc[:, :, j]  # (N_draws, H)

        # Fan bands
        for lo_pct, hi_pct in pairs:
            lo = np.percentile(fc_j, lo_pct, axis=0)
            hi = np.percentile(fc_j, hi_pct, axis=0)
            alpha = 0.15 + 0.10 * (50 - lo_pct) / 50
            ax.fill_between(h_axis, lo, hi, color=_COLOURS[0], alpha=alpha)

        # Median
        med = np.percentile(fc_j, 50, axis=0)
        ax.plot(h_axis, med, color=_COLOURS[0], lw=1.5)

        # History
        if history is not None:
            T_hist = history.shape[0]
            h_hist = np.arange(-T_hist + 1, 1)
            ax.plot(h_hist, history[:, j], color="black", lw=1.0, alpha=0.7)
            ax.axvline(0.5, color="grey", ls="--", lw=0.5)

        ax.set_title(var_names[j])
        ax.set_xlabel("Horizon")

    for j in range(K, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("BVAR Forecast Fan Charts", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

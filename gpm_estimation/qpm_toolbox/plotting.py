"""Plotting utilities for IRFs and smoothed states."""

from typing import Dict, List, Optional

import numpy as onp
import matplotlib.pyplot as plt

from qpm_toolbox.parser import irf


def plot_model_irfs(
    model,
    param_dict: Dict[str, float],
    shocks: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    horizon: int = 40,
    shock_size: float = 1.0,
    ncols: int = 3,
    nrows: int = 3,
    figsize_per_ax: tuple = (4.0, 2.5),
):
    """Compute and plot IRFs for selected shocks and variables.

    Args:
        model: parsed DynareModelWithLambdified instance.
        param_dict: parameter values (merged with model defaults).
        shocks: list of shock names to plot (default: all augmented shocks).
        variables: list of augmented-state variable names to plot
                   (default: all stationary + trend state variables).
        horizon: number of IRF periods.
        shock_size: size of the impulse (applied as a multiplier).
        ncols/nrows: subplot grid layout per figure (default 3x3).
        figsize_per_ax: (width, height) inches per subplot.

    Returns:
        dict  {shock_name: (irf_array, var_names)}  -- raw IRF data.
    """
    sol = model.solve(param_dict)
    if not sol["solution_valid"]:
        print("WARNING: model did not solve -- cannot compute IRFs")
        return {}

    P_aug = sol["P_aug"]
    R_aug = sol["R_aug"]

    all_shocks = model.aug_shocks
    all_vars = model.aug_state_vars

    # Resolve which shocks to plot
    if shocks is None:
        shocks = list(all_shocks)
    bad = [s for s in shocks if s not in all_shocks]
    if bad:
        raise ValueError(f"Unknown shocks: {bad}. Available: {all_shocks}")

    # Resolve which variables to plot
    if variables is None:
        variables = list(all_vars)
    var_indices = []
    for v in variables:
        if v not in all_vars:
            raise ValueError(f"Unknown variable '{v}'. Available: {all_vars}")
        var_indices.append(all_vars.index(v))

    per_page = nrows * ncols
    results = {}

    for shk in shocks:
        shk_idx = all_shocks.index(shk)
        irf_raw = onp.asarray(irf(P_aug, R_aug, shk_idx, horizon))
        irf_scaled = irf_raw[:, var_indices] * shock_size
        results[shk] = (irf_scaled, variables)

        n_vars = len(variables)
        n_figs = (n_vars + per_page - 1) // per_page

        for fig_i in range(n_figs):
            start = fig_i * per_page
            end = min(start + per_page, n_vars)
            n_this = end - start

            actual_rows = (n_this + ncols - 1) // ncols
            fig, axes = plt.subplots(
                actual_rows, ncols,
                figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * actual_rows),
                squeeze=False,
            )
            fig.suptitle(f"IRF to {shk} (size={shock_size})", fontsize=13)

            t = onp.arange(horizon)
            for k in range(n_this):
                ax = axes[k // ncols, k % ncols]
                ax.plot(t, irf_scaled[:, start + k], linewidth=1.4)
                ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
                ax.set_title(variables[start + k], fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.4)
                if k // ncols == actual_rows - 1:
                    ax.set_xlabel("Period")

            # Hide unused subplots
            for k in range(n_this, actual_rows * ncols):
                axes[k // ncols, k % ncols].set_visible(False)

            fig.tight_layout(rect=[0, 0, 1, 0.94])

    plt.show()
    return results


def plot_smoothed_var(ax, time_ax, draws, title,
                      data=None, true_state=None, trend_draws=None,
                      trend_label="Trend"):
    """Plot median + HDI for one variable."""
    import arviz as az
    median = onp.median(draws, axis=0)
    hdi_68 = az.hdi(onp.asarray(draws), hdi_prob=0.68)
    hdi_90 = az.hdi(onp.asarray(draws), hdi_prob=0.90)

    if data is not None:
        ax.plot(time_ax, data, 'k-', alpha=0.5, lw=1, label="Observed")
    if true_state is not None:
        ax.plot(time_ax, true_state, 'g--', alpha=0.7, lw=1.2, label="True")
    ax.plot(time_ax, median, color='mediumblue', lw=1.5, label="Median")
    ax.fill_between(time_ax, hdi_90[:, 0], hdi_90[:, 1],
                    color='lightskyblue', alpha=0.35, label="90% HDI")
    ax.fill_between(time_ax, hdi_68[:, 0], hdi_68[:, 1],
                    color='cornflowerblue', alpha=0.55, label="68% HDI")
    if trend_draws is not None:
        ax.plot(time_ax, onp.median(trend_draws, axis=0),
                color='orangered', ls=':', lw=1.5,
                label=f"{trend_label} (median)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)
    ax.legend(fontsize='xx-small', loc='upper left')

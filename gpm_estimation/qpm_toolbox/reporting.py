"""Reusable MCMC reporting utilities for Bayesian estimation.

Provides standard post-estimation diagnostics:
  - Chain diagnostics (divergences, acceptance rate, ESS, R-hat)
  - ArviZ summary with ESS, R-hat, HDI
  - Trace plots (structural params, alpha params, sigma params)
  - Prior vs posterior comparison table (mode, mean, HDI)

All functions are model-agnostic and operate on the MCMC object
and a list of prior specification dicts.

Usage:
    from qpm_toolbox.reporting import mcmc_report

    mcmc_report(mcmc, user_priors, results_dir,
                trace_groups={'structural': [...], 'alphas': [...]})
"""

import os
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arviz as az


# ======================================================================
# Prior distribution helpers
# ======================================================================

def prior_mode(spec: dict) -> float:
    """Compute the mode of a prior distribution from its specification.

    Supports: beta, gamma, invgamma, halfnormal.
    """
    a = spec["args"]
    if spec["prior"] == "beta":
        c1, c2 = a["concentration1"], a["concentration2"]
        if c1 > 1 and c2 > 1:
            return (c1 - 1) / (c1 + c2 - 2)
        return 0.0
    elif spec["prior"] == "gamma":
        alpha, rate = a["concentration"], a["rate"]
        if alpha > 1:
            return (alpha - 1) / rate
        return 0.0
    elif spec["prior"] == "invgamma":
        return a["rate"] / (a["concentration"] + 1)
    elif spec["prior"] == "halfnormal":
        return 0.0
    return 0.0


def prior_std(spec: dict) -> float:
    """Compute the standard deviation of a prior distribution."""
    a = spec["args"]
    if spec["prior"] == "beta":
        c1, c2 = a["concentration1"], a["concentration2"]
        var = c1 * c2 / ((c1 + c2)**2 * (c1 + c2 + 1))
        return var**0.5
    elif spec["prior"] == "gamma":
        alpha, rate = a["concentration"], a["rate"]
        return (alpha**0.5) / rate
    elif spec["prior"] == "invgamma":
        alpha = a["concentration"]
        beta = a["rate"]
        if alpha > 2:
            var = beta**2 / ((alpha - 1)**2 * (alpha - 2))
            return var**0.5
        return float('inf')
    elif spec["prior"] == "halfnormal":
        return a["scale"] * (1 - 2 / np.pi)**0.5
    return 0.0


def posterior_mode_kde(samples: np.ndarray) -> float:
    """Estimate the posterior mode via Gaussian KDE."""
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(samples)
        x_grid = np.linspace(samples.min(), samples.max(), 500)
        return float(x_grid[np.argmax(kde(x_grid))])
    except Exception:
        return float(np.median(samples))


# ======================================================================
# Prior vs posterior table
# ======================================================================

def prior_posterior_table(
    posterior_samples: Dict[str, Any],
    priors: List[dict],
    hdi_prob: float = 0.68,
) -> pd.DataFrame:
    """Build a prior vs posterior comparison table.

    For each estimated parameter, reports:
      - prior distribution type and mode/std
      - posterior mean, mode (KDE), and HDI

    Args:
        posterior_samples:  {param_name: array(n_draws,)} from MCMC.
        priors:             list of prior spec dicts with 'name', 'prior', 'args'.
        hdi_prob:           HDI probability (default 0.68 for 1-sigma equivalent).

    Returns:
        DataFrame with columns: parameter, prior_dist, prior_mode, prior_std,
        post_mean, post_mode, hdi_lo, hdi_hi.
    """
    lo_pct = 100 * (1 - hdi_prob) / 2
    hi_pct = 100 * (1 + hdi_prob) / 2

    rows = []
    for p in priors:
        name = p["name"]
        if name not in posterior_samples:
            continue
        s = np.asarray(posterior_samples[name])
        rows.append({
            'parameter': name,
            'prior_dist': p["prior"],
            'prior_mode': prior_mode(p),
            'prior_std': prior_std(p),
            'post_mean': float(s.mean()),
            'post_mode': posterior_mode_kde(s),
            'hdi_lo': float(np.percentile(s, lo_pct)),
            'hdi_hi': float(np.percentile(s, hi_pct)),
        })

    return pd.DataFrame(rows)


def print_prior_posterior_table(df: pd.DataFrame, hdi_prob: float = 0.68):
    """Pretty-print the prior vs posterior table."""
    lo_label = f"HDI{int(100*(1-hdi_prob)/2)}"
    hi_label = f"HDI{int(100*(1+hdi_prob)/2)}"

    header = (f"{'Parameter':>25s}  {'Prior':>10s}  {'Pr.Mode':>8s}  "
              f"{'Pr.Std':>7s}  {'Po.Mean':>8s}  {'Po.Mode':>8s}  "
              f"{lo_label:>8s}  {hi_label:>8s}")
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        print(f"{r['parameter']:>25s}  {r['prior_dist']:>10s}  "
              f"{r['prior_mode']:8.4f}  {r['prior_std']:7.4f}  "
              f"{r['post_mean']:8.4f}  {r['post_mode']:8.4f}  "
              f"{r['hdi_lo']:8.4f}  {r['hdi_hi']:8.4f}")


# ======================================================================
# Chain diagnostics
# ======================================================================

def print_chain_diagnostics(
    mcmc,
    az_summary_df: pd.DataFrame,
    rhat_threshold: float = 1.01,
    ess_bulk_threshold: int = 400,
    ess_tail_threshold: int = 100,
):
    """Print MCMC chain convergence diagnostics to console.

    Extracts sampler statistics (divergences, acceptance rate, step size,
    tree depth) from MCMC extra fields and flags problematic parameters
    based on R-hat and ESS from the ArviZ summary.

    Args:
        mcmc:               numpyro MCMC object (after .run()).
        az_summary_df:      ArviZ summary DataFrame (from arviz_summary()).
        rhat_threshold:     R-hat above this value flags non-convergence.
        ess_bulk_threshold: bulk ESS below this flags poor mixing.
        ess_tail_threshold: tail ESS below this flags poor tail estimation.
    """
    print("\n" + "=" * 70)
    print("  MCMC CHAIN DIAGNOSTICS")
    print("=" * 70)

    # --- Sampler configuration ---
    num_chains = mcmc.num_chains
    num_samples = mcmc.num_samples
    num_warmup = (mcmc._warmup_state.num_steps
                  if getattr(mcmc, '_warmup_state', None) is not None
                  else mcmc.num_warmup)
    total_draws = num_chains * num_samples
    print(f"\n  Chains: {num_chains}  |  Warmup: {num_warmup}  |  "
          f"Samples: {num_samples}  |  Total draws: {total_draws}")

    # --- Extract extra fields (divergences, accept prob, etc.) ---
    try:
        extra = mcmc.get_extra_fields()
    except Exception:
        extra = {}

    # Divergences
    if 'diverging' in extra:
        diverging = np.asarray(extra['diverging'])
        n_div = int(diverging.sum())
        pct_div = 100 * n_div / diverging.size
        if n_div == 0:
            print(f"  Divergences: 0 / {diverging.size} (OK)")
        else:
            print(f"  Divergences: {n_div} / {diverging.size} "
                  f"({pct_div:.1f}%) *** WARNING ***")
            if pct_div > 1:
                print("    -> High divergence rate. Consider reparameterising "
                      "or increasing target_accept_prob.")
    else:
        print("  Divergences: not available (extra_fields not collected)")

    # Acceptance probability
    if 'mean_accept_prob' in extra:
        accept = np.asarray(extra['mean_accept_prob'])
        print(f"  Mean accept prob: {accept.mean():.3f}  "
              f"(min={accept.min():.3f}, max={accept.max():.3f})")

    # Step size (lives inside adapt_state on the last HMC state)
    try:
        last_state = mcmc.last_state
        ss = np.asarray(last_state.adapt_state.step_size)
        if ss.ndim == 0:
            print(f"  Step size (final): {float(ss):.5f}")
        else:
            print(f"  Step size (final): {ss.mean():.5f}  "
                  f"(range [{ss.min():.5f}, {ss.max():.5f}])")
    except Exception:
        pass  # step size not accessible in this NumPyro version

    # Tree depth (num leapfrog steps per sample)
    if 'num_steps' in extra:
        depths = np.asarray(extra['num_steps'])
        max_depth = int(depths.max())
        mean_depth = float(depths.mean())
        print(f"  Tree depth: mean={mean_depth:.1f}, max={max_depth}")

    # --- R-hat diagnostics (requires >= 2 chains) ---
    print(f"\n  --- Convergence (R-hat threshold: {rhat_threshold}) ---")
    if num_chains < 2:
        print("  R-hat: not available (need >= 2 chains)")
        print("    -> Run with chains >= 2 for convergence diagnostics.")
    elif 'r_hat' in az_summary_df.columns:
        # ArviZ 1.x stores summary columns as string dtype (formatted for
        # display); coerce to numeric for comparison and reduction.
        rhat = pd.to_numeric(az_summary_df['r_hat'], errors='coerce')
        bad_rhat_mask = rhat > rhat_threshold
        bad_rhat = az_summary_df[bad_rhat_mask.fillna(False)]
        if len(bad_rhat) == 0:
            print(f"  R-hat: all parameters <= {rhat_threshold} (OK)")
            print(f"    max R-hat = {rhat.max():.4f} "
                  f"({az_summary_df.index[rhat.argmax()]})")
        else:
            print(f"  R-hat: {len(bad_rhat)} parameter(s) above "
                  f"{rhat_threshold} *** WARNING ***")
            for idx, row in bad_rhat.iterrows():
                print(f"    {idx:>25s}  R-hat = "
                      f"{float(row['r_hat']):.4f}")
    else:
        print("  R-hat: column not found in ArviZ summary")

    # --- ESS diagnostics ---
    print(f"\n  --- Effective Sample Size (ESS) ---")
    _print_ess_column(az_summary_df, 'ess_bulk', 'Bulk ESS',
                      ess_bulk_threshold, total_draws)
    _print_ess_column(az_summary_df, 'ess_tail', 'Tail ESS',
                      ess_tail_threshold, total_draws)

    print("=" * 70 + "\n")


def _print_ess_column(
    df: pd.DataFrame,
    col: str,
    label: str,
    threshold: int,
    total_draws: int,
):
    """Helper to print ESS diagnostics for a single column."""
    if col not in df.columns:
        print(f"  {label}: column not found")
        return

    # ArviZ 1.x stores summary columns as string dtype; coerce to
    # numeric for reductions and comparisons.
    ess = pd.to_numeric(df[col], errors='coerce')
    min_ess = ess.min()
    min_param = df.index[ess.argmin()]
    ratio = min_ess / total_draws if total_draws > 0 else 0.0

    low_mask = (ess < threshold).fillna(False)
    low = df[low_mask]
    if len(low) == 0:
        print(f"  {label}: all >= {threshold} (OK)")
        print(f"    min = {min_ess:.0f} ({min_param}), "
              f"ESS/draw = {ratio:.2f}")
    else:
        print(f"  {label}: {len(low)} parameter(s) below "
              f"{threshold} *** WARNING ***")
        for idx, row in low.iterrows():
            print(f"    {idx:>25s}  {label} = "
                  f"{float(row[col]):.0f}")
        print(f"    -> Consider more samples or reparameterisation.")


# ======================================================================
# ArviZ summary
# ======================================================================

def arviz_summary(
    mcmc,
    priors: List[dict],
    hdi_prob: float = 0.68,
) -> pd.DataFrame:
    """Compute ArviZ summary table (ESS, R-hat, HDI).

    Args:
        mcmc:       numpyro MCMC object (after .run()).
        priors:     list of prior spec dicts (used to select var names).
        hdi_prob:   HDI probability for the summary.

    Returns:
        ArviZ summary DataFrame.
    """
    az_data = az.from_numpyro(mcmc)
    posterior_samples = mcmc.get_samples()
    var_names = [p["name"] for p in priors if p["name"] in posterior_samples]
    # ArviZ 1.x renamed ``hdi_prob`` to ``ci_prob`` in az.summary; older
    # versions (0.x) only accept ``hdi_prob``.  Pick the kwarg the
    # installed version supports so the toolbox works on both.
    import inspect as _inspect
    _summary_params = _inspect.signature(az.summary).parameters
    if "ci_prob" in _summary_params:
        return az.summary(az_data, ci_prob=hdi_prob, var_names=var_names)
    return az.summary(az_data, hdi_prob=hdi_prob, var_names=var_names)


# ======================================================================
# Trace plots
# ======================================================================

def save_trace_plots(
    mcmc,
    trace_groups: Dict[str, List[str]],
    results_dir: str,
):
    """Save ArviZ trace plots for groups of parameters.

    Args:
        mcmc:           numpyro MCMC object.
        trace_groups:   dict mapping group_name -> list of param names.
                        E.g. {'structural': ['beta1', ...], 'alphas': [...]}
        results_dir:    directory where PNGs are saved.
    """
    az_data = az.from_numpyro(mcmc)
    posterior_samples = mcmc.get_samples()

    # ArviZ 0.x plot_trace accepts figsize; 1.x dropped it (size now
    # set via pc_kwargs / matplotlib defaults).  Inspect the signature
    # so the toolbox works on both.
    import inspect as _inspect
    _trace_params = _inspect.signature(az.plot_trace).parameters
    _supports_figsize = "figsize" in _trace_params

    for group_name, var_names in trace_groups.items():
        valid = [v for v in var_names if v in posterior_samples]
        if not valid:
            continue
        plot_kwargs = {}
        if _supports_figsize:
            plot_kwargs["figsize"] = (14, 2.5 * len(valid))
        result = az.plot_trace(az_data, var_names=valid, **plot_kwargs)
        fname = f"trace_{group_name}.png"
        out_path = os.path.join(results_dir, fname)
        # ArviZ 1.x returns a PlotCollection with a .savefig method;
        # 0.x returns a numpy array of matplotlib axes and relies on
        # pyplot.savefig of the current figure.
        if hasattr(result, "savefig"):
            try:
                result.savefig(out_path, dpi=100)
            except TypeError:
                result.savefig(out_path)
        else:
            plt.tight_layout()
            plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close('all')
        print(f"  Saved {fname}")


# ======================================================================
# Main reporting entry point
# ======================================================================

def mcmc_report(
    mcmc,
    priors: List[dict],
    results_dir: str,
    trace_groups: Optional[Dict[str, List[str]]] = None,
    hdi_prob: float = 0.68,
):
    """Run full MCMC reporting suite.

    Produces:
      1. ArviZ summary CSV (ESS, R-hat, HDI)
      2. Chain diagnostics (divergences, acceptance rate, ESS, R-hat flags)
      3. Trace plots for each parameter group
      4. Prior vs posterior comparison table (printed + CSV)

    Args:
        mcmc:           numpyro MCMC object (after .run()).
        priors:         list of prior spec dicts.
        results_dir:    output directory for files.
        trace_groups:   optional dict of group_name -> [param_names] for
                        trace plots. If None, a single 'all' group is used.
        hdi_prob:       HDI probability (default 0.68).

    Returns:
        dict with keys 'az_summary', 'prior_posterior' (DataFrames).
    """
    os.makedirs(results_dir, exist_ok=True)
    posterior_samples = mcmc.get_samples()

    # 1. ArviZ summary
    print("\n--- ArviZ Summary ---")
    az_sum = arviz_summary(mcmc, priors, hdi_prob=hdi_prob)
    az_sum.to_csv(os.path.join(results_dir, "arviz_summary.csv"))
    print(f"  Saved arviz_summary.csv ({len(az_sum)} parameters)")

    # 2. Chain diagnostics (divergences, ESS, R-hat)
    print_chain_diagnostics(mcmc, az_sum)

    # 3. Trace plots
    if trace_groups is None:
        all_names = [p["name"] for p in priors if p["name"] in posterior_samples]
        trace_groups = {'all': all_names}
    save_trace_plots(mcmc, trace_groups, results_dir)

    # 4. Prior vs posterior table
    print(f"\n--- Prior vs Posterior Table (HDI {int(hdi_prob*100)}%) ---")
    df_pp = prior_posterior_table(posterior_samples, priors, hdi_prob=hdi_prob)
    print_prior_posterior_table(df_pp, hdi_prob=hdi_prob)
    df_pp.to_csv(os.path.join(results_dir, "prior_posterior_table.csv"),
                 index=False, float_format='%.6f')
    print(f"\n  Saved prior_posterior_table.csv")

    return {'az_summary': az_sum, 'prior_posterior': df_pp}

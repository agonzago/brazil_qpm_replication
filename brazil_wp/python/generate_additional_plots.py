"""Generate additional plots for the paper:

1. Historical shock decomposition of GDP growth (OBS_LGDP first-differenced)
   - Shows trend shocks (SHK_LGDP_BAR, SHK_G_TREND) alongside cyclical shocks
2. Real interest rate vs. estimated natural rate with posterior bands

Usage:
    python generate_additional_plots.py                # Model A (default)
    python generate_additional_plots.py --model F      # Any of A, A0, B, C, D, E, F
    python generate_additional_plots.py --model B --results-dir /tmp/x
"""

import argparse
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX_DIR = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX_DIR)

from _runner_loader import MODEL_LABELS, load_runner, results_dir as _results_dir
from qpm_toolbox import QPMModel
from qpm_toolbox.kalman_filter import (
    KalmanFilter, DiffuseKalmanFilter, build_diffuse_init, get_dtype,
)
from qpm_toolbox.calibration import compute_prior_modes


# Extended shock groups: add trend shocks
SHOCK_GROUPS_GROWTH = {
    'Demand':         ['SHK_GDP_GAP'],
    'Cost-push':      ['SHK_DA_LCPIX', 'SHK_DA_LCPI_ENERGY',
                       'SHK_DA_LCPI_FOOD', 'SHK_DA_LCPI'],
    'Monetary':       ['SHK_RS'],
    'FX & premium':   ['SHK_LZ_GAP', 'SHK_PREM_GAP', 'SHK_PREM_BAR'],
    'Terms of trade': ['SHK_TT_GAP', 'SHK_TT_BAR'],
    'Foreign':        ['SHK_GAP_US', 'SHK_DA_LCPI_US', 'SHK_RS_US',
                       'SHK_RR_BAR_US', 'SHK_PI_TREND_US'],
    'Commodities':    ['SHK_LRWOIL_GAP', 'SHK_LRWFOOD_GAP'],
    'Trend (output)': ['SHK_LGDP_BAR', 'SHK_G_TREND'],
    'Trend (other)':  ['SHK_LZ_BAR', 'SHK_DLZ_TREND',
                       'SHK_RR_BAR', 'SHK_PI_TREND'],
}


def build_pdict_for_draw(m, draw_idx):
    """Build full parameter dict for a single posterior draw."""
    pdict = dict(m.model.default_param_assignments)
    if m._fixed_params:
        pdict.update(m._fixed_params)
    if m._structural_priors:
        modes = compute_prior_modes(m._structural_priors)
        pdict.update(modes)
    if m._calibrated_sigmas:
        pdict.update(m._calibrated_sigmas)
    for name, samples in m._posterior_samples.items():
        pdict[name] = float(np.asarray(samples)[draw_idx])
    return pdict


def run_smoother_single(m, pdict, ys_jax, H_t, use_diffuse=True):
    """Solve + Kalman smooth for one parameter draw. Returns smoothed states.

    Uses the exact diffuse smoother (Durbin & Koopman 2012) by default —
    required for unit-root QPM models, and handles NaN observations natively
    via its univariate observation step.  Falls back to the standard
    Kalman filter when ``H_t`` is supplied (time-varying H is not supported
    by the diffuse path)."""
    dt = get_dtype()
    sol = m.model.solve(pdict)
    if not sol["solution_valid"]:
        return None
    if use_diffuse and H_t is None:
        try:
            Pinf, Pstar, _ = build_diffuse_init(sol["P_aug"], sol["R_aug"], dtype=dt)
            kf = DiffuseKalmanFilter(
                T=sol["P_aug"], R=sol["R_aug"], C=sol["Omega"], H=m._H_obs,
                Pinf=Pinf, Pstar=Pstar,
                init_a=m._init_x, c=sol["c_aug"], dtype=dt,
            )
            x_smooth = np.asarray(kf.smooth_states_only(ys=ys_jax))
            if np.isnan(x_smooth).any():
                return None
            return x_smooth
        except Exception:
            pass  # fall through to standard filter
    kf = KalmanFilter(
        T=sol["P_aug"], R=sol["R_aug"], C=sol["Omega"], H=m._H_obs,
        init_x=m._init_x, init_P=m._init_P,
        c=sol["c_aug"], dtype=dt,
    )
    x_smooth = np.asarray(kf.smooth_states_only(ys=ys_jax, H_t=H_t))
    if np.isnan(x_smooth).any():
        return None
    return x_smooth


def main():
    import jax
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='A', choices=MODEL_LABELS,
                        help='Model label (default: A)')
    parser.add_argument('--results-dir', default=None,
                        help='Override results dir (defaults to '
                             'results_v2_diffuse/model_<x>)')
    args = parser.parse_args()

    runner = load_runner(args.model)
    MOD_FILE = runner.MOD_FILE
    RESULTS_DIR = args.results_dir or _results_dir(args.model)

    N_DRAWS = 200

    print("=" * 60)
    print(f"  ADDITIONAL PLOTS: GDP GROWTH DECOMP + NATURAL RATE  [Model {args.model}]")
    print(f"  Results dir: {RESULTS_DIR}")
    print("=" * 60)

    # 1. Load data and model
    ys, dates, obs_names = runner.load_brazil_data()
    m = QPMModel(MOD_FILE, results_dir=RESULTS_DIR)
    ss_params = m.model.default_param_assignments
    init_config = runner.compute_initial_conditions(ys, obs_names, ss_params)

    m.set_data(ys, dates=dates, obs_names=obs_names)
    m.set_priors(
        structural_priors=runner.STRUCTURAL_PRIORS,
        fixed_params=runner.FIXED_PARAMS,
        meas_noise_std=runner.MEAS_NOISE_STD,
        cycle_map=runner.CYCLE_MAP,
        trend_snr_map=runner.TREND_SNR_MAP,
        manual_cycle_priors=runner.MANUAL_CYCLE_PRIORS,
    )
    m.set_initial_conditions(init_state_config=init_config)
    m.solution_check()

    draws_path = os.path.join(RESULTS_DIR, "estimation", "draws.npz")
    m.load_draws(draws_path)

    # ================================================================
    # Part 1: GDP Growth Shock Decomposition
    # ================================================================
    print("\n  1. GDP growth shock decomposition...")

    # Run decomposition on OBS_LGDP (log GDP level)
    # The toolbox decomposes the observable into shock contributions.
    # We'll then first-difference to get growth contributions.
    result = m.decomposition(
        variables=['OBS_LGDP'],
        shock_groups=SHOCK_GROUPS_GROWTH,
        plot=False,  # we'll make our own plot
    )

    # Extract the decomposition for OBS_LGDP from obs_contrib
    # result.obs_contrib: (n_shock, T, n_obs), result.shock_names, result.obs_names
    obs_var_idx = result.obs_names.index('OBS_LGDP')
    shock_to_idx = {s: i for i, s in enumerate(result.shock_names)}

    # Aggregate shock contributions by group and first-difference for growth
    contrib_growth = {}
    for grp, shk_list in SHOCK_GROUPS_GROWTH.items():
        level_contrib = np.zeros(result.obs_contrib.shape[1])  # (T,)
        for s in shk_list:
            if s in shock_to_idx:
                level_contrib += result.obs_contrib[shock_to_idx[s], :, obs_var_idx]
        contrib_growth[grp] = np.diff(level_contrib)  # (T-1,)

    # Add baseline (initial conditions + deterministic) as residual
    baseline_growth = np.diff(result.obs_baseline[:, obs_var_idx])

    # Observed GDP growth (first difference of log GDP)
    obs_idx = {n: i for i, n in enumerate(obs_names)}
    gdp_growth = np.diff(ys[:, obs_idx['OBS_LGDP']])
    dates_growth = dates[1:]

    # Plot stacked bar decomposition
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {
        'Demand':         '#1f77b4',
        'Cost-push':      '#ff7f0e',
        'Monetary':       '#2ca02c',
        'FX & premium':   '#d62728',
        'Terms of trade': '#9467bd',
        'Foreign':        '#8c564b',
        'Commodities':    '#e377c2',
        'Trend (output)': '#7f7f7f',
        'Trend (other)':  '#bcbd22',
        'Initial cond.':  '#17becf',
    }

    # Add baseline to the groups
    contrib_growth['Initial cond.'] = baseline_growth
    groups = list(contrib_growth.keys())
    n_t = len(dates_growth)
    pos_bottom = np.zeros(n_t)
    neg_bottom = np.zeros(n_t)

    for grp in groups:
        vals = contrib_growth[grp]
        pos = np.maximum(vals, 0)
        neg = np.minimum(vals, 0)
        ax.bar(dates_growth, pos, bottom=pos_bottom, width=80,
               color=colors.get(grp, '#333333'), label=grp, alpha=0.85)
        ax.bar(dates_growth, neg, bottom=neg_bottom, width=80,
               color=colors.get(grp, '#333333'), alpha=0.85)
        pos_bottom += pos
        neg_bottom += neg

    ax.plot(dates_growth, gdp_growth, 'k-', lw=1.2, label='GDP growth (QoQ)')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title('Historical Shock Decomposition: GDP Growth', fontsize=12,
                 fontweight='bold')
    ax.set_ylabel('Quarter-on-quarter change (log points)')
    ax.legend(fontsize=7, loc='lower left', ncol=3, framealpha=0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    out1 = os.path.join(RESULTS_DIR, "decomposition", "decomp_GDP_GROWTH.png")
    fig.savefig(out1, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out1}")

    # ================================================================
    # Part 2: Real Interest Rate vs. Natural Rate with Posterior Bands
    # ================================================================
    print("\n  2. Real interest rate vs. natural rate...")

    ys_jax, H_obs, H_t, mult = m._resolve_ys_H_multipliers()

    n_stat = m.model.n_stat
    trend_names = m.model.ordered_trend_state_vars
    stat_names = m.model.ordered_stat_vars
    trend_idx = {name: n_stat + i for i, name in enumerate(trend_names)}
    stat_idx = {name: i for i, name in enumerate(stat_names)}

    # We need: RR_BAR (neutral real rate) and RS - PI_TREND (actual real rate)
    # From model: RS is the nominal rate gap, but OBS_RS = RS + RR_BAR + PI_TREND
    # So actual real rate = OBS_RS - inflation ≈ RR_BAR + RR_GAP
    # More precisely: real rate = i_t - E[pi_{t+1}] = rr_bar + rr_gap
    # where rr_gap = RS - pi_{t+1} (from Fisher equation)
    # But the cleanest: actual_rr = RR_BAR + RR_GAP (from smoothed states)

    total_draws = len(next(iter(m._posterior_samples.values())))
    rng = np.random.default_rng(42)
    draw_indices = rng.choice(total_draws, size=N_DRAWS, replace=False)

    T_periods = ys.shape[0]

    # Collect RR_BAR and RR_GAP draws
    rr_bar_draws = np.zeros((N_DRAWS, T_periods))
    rr_gap_draws = np.zeros((N_DRAWS, T_periods))
    actual_rr_draws = np.zeros((N_DRAWS, T_periods))
    gdp_gap_draws = np.zeros((N_DRAWS, T_periods))

    rr_bar_idx = trend_idx['RR_BAR']
    rr_gap_idx = stat_idx['RR_GAP']
    gdp_gap_idx = stat_idx['GDP_GAP']

    print(f"  Running {N_DRAWS} Kalman smoothers...")
    n_ok = 0
    for j, di in enumerate(draw_indices):
        if (j + 1) % 50 == 0:
            print(f"    Draw {j+1}/{N_DRAWS}...")
        pdict = build_pdict_for_draw(m, di)
        xs = run_smoother_single(m, pdict, ys_jax, H_t)
        if xs is None:
            continue
        rr_bar_draws[n_ok] = xs[:, rr_bar_idx]
        rr_gap_draws[n_ok] = xs[:, rr_gap_idx]
        actual_rr_draws[n_ok] = xs[:, rr_bar_idx] + xs[:, rr_gap_idx]
        gdp_gap_draws[n_ok] = xs[:, gdp_gap_idx]
        n_ok += 1

    rr_bar_draws = rr_bar_draws[:n_ok]
    rr_gap_draws = rr_gap_draws[:n_ok]
    actual_rr_draws = actual_rr_draws[:n_ok]
    gdp_gap_draws = gdp_gap_draws[:n_ok]
    print(f"  Completed: {n_ok} successful draws")

    # Quantiles
    def q(arr):
        return {
            'median': np.median(arr, axis=0),
            'lo68': np.percentile(arr, 16, axis=0),
            'hi68': np.percentile(arr, 84, axis=0),
            'lo90': np.percentile(arr, 5, axis=0),
            'hi90': np.percentile(arr, 95, axis=0),
        }

    rr_bar_q = q(rr_bar_draws)
    actual_rr_q = q(actual_rr_draws)
    gdp_gap_q = q(gdp_gap_draws)

    # --- Plot: Real rate vs. natural rate ---
    fig, ax = plt.subplots(figsize=(12, 5))

    # Natural rate bands
    ax.fill_between(dates, rr_bar_q['lo90'], rr_bar_q['hi90'],
                    color='blue', alpha=0.08, label='Natural rate 90% HDI')
    ax.fill_between(dates, rr_bar_q['lo68'], rr_bar_q['hi68'],
                    color='blue', alpha=0.20, label='Natural rate 68% HDI')
    ax.plot(dates, rr_bar_q['median'], 'b-', lw=2.0,
            label=r'Natural rate $\bar{r}_t$ (median)')

    # Actual real rate
    ax.plot(dates, actual_rr_q['median'], 'r-', lw=1.5, alpha=0.85,
            label=r'Real interest rate $r_t = \bar{r}_t + \tilde{r}_t$')
    ax.fill_between(dates, actual_rr_q['lo68'], actual_rr_q['hi68'],
                    color='red', alpha=0.12)

    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.set_title('Real Interest Rate and Estimated Natural Rate',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Percent (annualized)')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out2 = os.path.join(RESULTS_DIR, "diagnostic", "real_rate_natural_rate.png")
    fig.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out2}")

    # --- Plot: Output gap with posterior bands ---
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.fill_between(dates, gdp_gap_q['lo90'], gdp_gap_q['hi90'],
                     color='blue', alpha=0.08, label='90% HDI')
    ax2.fill_between(dates, gdp_gap_q['lo68'], gdp_gap_q['hi68'],
                     color='blue', alpha=0.20, label='68% HDI')
    ax2.plot(dates, gdp_gap_q['median'], 'b-', lw=2.0, label='Output gap (median)')
    ax2.axhline(0, color='black', lw=0.5, ls='--')
    ax2.set_title('Estimated Output Gap with Posterior Confidence Bands',
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percent of potential')
    ax2.legend(fontsize=9, loc='lower left', framealpha=0.9)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.tick_params(axis='both', labelsize=9)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    out3 = os.path.join(RESULTS_DIR, "diagnostic", "output_gap_posterior.png")
    fig2.savefig(out3, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {out3}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

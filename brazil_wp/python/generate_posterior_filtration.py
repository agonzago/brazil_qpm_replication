"""Generate filtration plots with posterior confidence bands.

For each observable with an estimated trend, plots:
  - Observed data (black)
  - Posterior median trend (red)
  - 68% and 90% posterior bands around the trend

Focuses on observables from the measurement equations that have
non-trivial trend decomposition.

Usage:
    python generate_posterior_filtration.py                # Model A (default)
    python generate_posterior_filtration.py --model F      # Any of A, A0, B, C, D, E, F
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


# Observables with estimated trends (skip pure gap variables)
OBS_WITH_TRENDS = {
    'OBS_LGDP':           'LGDP_BAR',
    'OBS_DA_LCPI':        'PI_TREND',
    'OBS_DA_LCPIX':       'PI_TREND',
    'OBS_DA_LCPI_ENERGY': 'PI_TREND',
    'OBS_DA_LCPI_US':     'PI_TREND_US',
    'OBS_PREM':           'PREM_BAR',
    'OBS_RS':             'RR_BAR',       # neutral rate (via i_bar = rr_bar + pi_trend)
    'OBS_TT':             'TT_BAR',
    'OBS_DA_LCPI_TAR':    'PI_TREND',
}

# Nice labels for plots
LABELS = {
    'OBS_LGDP':           'Log GDP',
    'OBS_DA_LCPI':        'Headline CPI Inflation',
    'OBS_DA_LCPIX':       'Core CPI Inflation',
    'OBS_DA_LCPI_ENERGY': 'Energy CPI Inflation',
    'OBS_DA_LCPI_US':     'U.S. CPI Inflation',
    'OBS_PREM':           'Risk Premium (EMBI)',
    'OBS_RS':             'Policy Rate (Selic)',
    'OBS_TT':             'Terms of Trade',
    'OBS_DA_LCPI_TAR':    'Inflation Target',
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
    # Override with this specific draw
    for name, samples in m._posterior_samples.items():
        pdict[name] = float(np.asarray(samples)[draw_idx])
    return pdict


def run_smoother_single(m, pdict, ys_jax, H_t, use_diffuse=True):
    """Solve model + run Kalman smoother for one parameter draw.

    Uses the exact diffuse smoother (Durbin & Koopman 2012) by default —
    required for unit-root QPM models, and handles NaN observations natively.
    Falls back to the standard Kalman filter when ``H_t`` is supplied."""
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

    N_DRAWS = 200  # subsample for speed

    print("=" * 60)
    print(f"  POSTERIOR FILTRATION WITH CONFIDENCE BANDS  [Model {args.model}]")
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

    # 2. Load draws
    draws_path = os.path.join(RESULTS_DIR, "estimation", "draws.npz")
    m.load_draws(draws_path)
    total_draws = len(next(iter(m._posterior_samples.values())))
    print(f"  Total draws: {total_draws}, using {N_DRAWS}")

    # Subsample draw indices
    rng = np.random.default_rng(42)
    draw_indices = rng.choice(total_draws, size=N_DRAWS, replace=False)

    # 3. Get observation data and H_t
    ys_jax, H_obs, H_t, mult = m._resolve_ys_H_multipliers()

    # Build index maps
    n_stat = m.model.n_stat
    trend_names = m.model.ordered_trend_state_vars
    obs_idx = {name: i for i, name in enumerate(obs_names)}
    trend_idx = {name: n_stat + i for i, name in enumerate(trend_names)}

    # Also build "neutral nominal rate" index for OBS_RS
    # i_bar = rr_bar + pi_trend
    stat_names = m.model.ordered_stat_vars
    print(f"  Trend states: {trend_names}")

    T_periods = ys.shape[0]

    # 4. Run smoother for each draw
    print(f"  Running {N_DRAWS} Kalman smoothers...")
    all_x_smooth = []
    n_failed = 0
    for j, di in enumerate(draw_indices):
        if (j + 1) % 50 == 0:
            print(f"    Draw {j+1}/{N_DRAWS}...")
        pdict = build_pdict_for_draw(m, di)
        xs = run_smoother_single(m, pdict, ys_jax, H_t)
        if xs is None:
            n_failed += 1
            continue
        all_x_smooth.append(xs)

    print(f"  Completed: {len(all_x_smooth)} successful, {n_failed} failed")
    x_stack = np.stack(all_x_smooth, axis=0)  # (N, T, n_aug)

    # 5. Compute observables from smoothed states: obs = x @ Omega.T
    # We need Omega at the posterior median for the obs reconstruction
    pdict_median = m._resolve_params()
    sol_median = m.model.solve(pdict_median)
    Omega = np.asarray(sol_median["Omega"])

    # Smoothed observables for each draw: (N, T, n_obs)
    obs_stack = np.einsum('ntj,oj->nto', x_stack, Omega)

    # 6. For each observable with trend, extract trend state
    # Compute quantiles
    def quantiles(arr, axis=0):
        """Return dict of quantiles along axis."""
        return {
            'median': np.median(arr, axis=axis),
            'lo68': np.percentile(arr, 16, axis=axis),
            'hi68': np.percentile(arr, 84, axis=axis),
            'lo90': np.percentile(arr, 5, axis=axis),
            'hi90': np.percentile(arr, 95, axis=axis),
        }

    # 7. Plot
    plot_obs = [k for k in OBS_WITH_TRENDS if k in obs_idx]
    # Remove OBS_DA_LCPI_TAR (it IS the trend, not interesting to decompose)
    plot_obs = [k for k in plot_obs if k != 'OBS_DA_LCPI_TAR']

    n_plots = len(plot_obs)
    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows),
                             squeeze=False)

    for idx, obs_name in enumerate(plot_obs):
        ax = axes[idx // ncols, idx % ncols]
        oi = obs_idx[obs_name]
        trend_name = OBS_WITH_TRENDS[obs_name]

        # Observed data
        y_obs = ys[:, oi]

        # Smoothed observable (posterior bands)
        obs_q = quantiles(obs_stack[:, :, oi])

        # Trend state
        if trend_name in trend_idx:
            ti = trend_idx[trend_name]
            trend_draws = x_stack[:, :, ti]  # (N, T)

            # For OBS_RS, the trend is i_bar = rr_bar + pi_trend
            if obs_name == 'OBS_RS' and 'PI_TREND' in trend_idx:
                pi_ti = trend_idx['PI_TREND']
                trend_draws = x_stack[:, :, ti] + x_stack[:, :, pi_ti]

            trend_q = quantiles(trend_draws)
        else:
            trend_q = None

        # Plot
        ax.plot(dates, y_obs, 'k-', lw=1.0, alpha=0.7, label='Observed')

        if trend_q is not None:
            ax.plot(dates, trend_q['median'], 'r-', lw=1.5, label='Trend (median)')
            ax.fill_between(dates, trend_q['lo68'], trend_q['hi68'],
                            color='red', alpha=0.25, label='68% HDI')
            ax.fill_between(dates, trend_q['lo90'], trend_q['hi90'],
                            color='red', alpha=0.10, label='90% HDI')

        ax.set_title(LABELS.get(obs_name, obs_name), fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.8)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for idx in range(n_plots, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('Posterior Trend--Cycle Decomposition with 68% and 90% HDI Bands',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "diagnostic", "filtration_posterior.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # --- Also generate a separate trend-only plot ---
    # Focus on key unobserved trends: potential GDP, neutral rate,
    # inflation trend, REER trend, risk premium trend
    key_trends = {
        'LGDP_BAR':  'Potential Output (log)',
        'G_TREND':   'Potential Growth Rate',
        'RR_BAR':    'Neutral Real Rate',
        'PI_TREND':  'Inflation Trend',
        'LZ_BAR':    'REER Trend (log)',
        'DLZ_TREND': 'REER Trend Depreciation',
        'PREM_BAR':  'Risk Premium Trend',
        'TT_BAR':    'Terms of Trade Trend',
    }
    key_trends = {k: v for k, v in key_trends.items() if k in trend_idx}

    n_t = len(key_trends)
    ncols_t = 2
    nrows_t = (n_t + ncols_t - 1) // ncols_t

    fig2, axes2 = plt.subplots(nrows_t, ncols_t, figsize=(14, 3.5 * nrows_t),
                                squeeze=False)

    for idx, (tname, tlabel) in enumerate(key_trends.items()):
        ax = axes2[idx // ncols_t, idx % ncols_t]
        ti = trend_idx[tname]
        t_draws = x_stack[:, :, ti]  # (N, T)
        tq = quantiles(t_draws)

        ax.plot(dates, tq['median'], 'b-', lw=1.5, label='Median')
        ax.fill_between(dates, tq['lo68'], tq['hi68'],
                        color='blue', alpha=0.25, label='68% HDI')
        ax.fill_between(dates, tq['lo90'], tq['hi90'],
                        color='blue', alpha=0.10, label='90% HDI')
        ax.set_title(tlabel, fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.8)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_t, nrows_t * ncols_t):
        axes2[idx // ncols_t, idx % ncols_t].set_visible(False)

    fig2.suptitle('Posterior Trend Estimates with 68% and 90% HDI Bands',
                  fontsize=13, fontweight='bold', y=1.01)
    fig2.tight_layout()

    out_path2 = os.path.join(RESULTS_DIR, "diagnostic", "trends_posterior.png")
    fig2.savefig(out_path2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {out_path2}")

    print("=" * 60)


if __name__ == "__main__":
    main()

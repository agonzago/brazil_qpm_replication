"""Compare smoothed states between Model A and Model F at posterior median.

Produces:
  - results_smoke/compare_af/states_AvF.csv    : long-format DataFrame with
      columns [date, state, model_a, model_f]
  - results_smoke/compare_af/state_<X>.png     : per-state side-by-side line
      plot, A in solid blue, F in dashed red.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX_DIR = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX_DIR)

from qpm_toolbox import QPMModel
from qpm_toolbox.kalman_filter import (
    DiffuseKalmanFilter, build_diffuse_init, get_dtype,
)


MODELS = {
    'A': {
        'mod_file': os.path.join(SCRIPT_DIR, 'model', 'brazil_qpm_mr_v2.qpm'),
        'results':  os.path.join(SCRIPT_DIR, 'results_v2_diffuse', 'model_a'),
    },
    'F': {
        'mod_file': os.path.join(SCRIPT_DIR, 'model', 'brazil_qpm_mr_v2.qpm'),
        'results':  os.path.join(SCRIPT_DIR, 'results_v2_diffuse', 'model_f'),
    },
}

OUT_DIR = os.path.join(SCRIPT_DIR, 'results_smoke', 'compare_af')
os.makedirs(OUT_DIR, exist_ok=True)


def smoothed_states_at_median(model_label, info):
    """Load draws, compute posterior median, run diffuse smoother."""
    print(f"\n=== Model {model_label} ===")
    m = QPMModel(info['mod_file'], results_dir=info['results'])

    # Use the model's loaders from the runner -- we have to recreate the
    # data setup.  Easiest is to use load_data + measurement noise.
    # Read estimation_data.csv directly.
    csv = os.path.join(SCRIPT_DIR, 'data', 'estimation_data.csv')
    if not os.path.exists(csv):
        csv = os.path.join(TOOLBOX_DIR, 'data', 'brazil', 'estimation_data.csv')

    df = pd.read_csv(csv, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp()

    # Map raw RR_BAR_US (HLW r*) to OBS_RR_BAR_US (mirrors runner)
    if 'RR_BAR_US' in df.columns and 'OBS_RR_BAR_US' not in df.columns:
        df['OBS_RR_BAR_US'] = df['RR_BAR_US']

    obs_names = list(m.model.ordered_obs_vars)
    ys = df[obs_names].values.astype(np.float64)

    # Load draws
    draws_path = os.path.join(info['results'], 'estimation', 'draws.npz')
    npz = np.load(draws_path)
    pdict_med = {}
    for name in npz.files:
        arr = npz[name]
        if arr.dtype.kind in ('U', 'O', 'S'):
            continue   # skip _meta_json_ etc.
        if arr.ndim == 0 or arr.size == 0:
            continue
        pdict_med[name] = float(np.median(arr))

    # Add fixed parameters (the runners' FIXED_PARAMS)
    pdict_med.setdefault('lambda1s', 0.1581)
    pdict_med.setdefault('lambda2s', 0.1142)
    pdict_med.setdefault('sigma_SHK_TT_BAR', 3.0)

    # Model F has Taylor coefficients fixed
    if model_label == 'F':
        pdict_med.setdefault('gamma1', 0.85)
        pdict_med.setdefault('gamma2', 1.50)
        pdict_med.setdefault('gamma4', 0.50)

    # Solve at posterior median
    sol = m.model.solve(pdict_med)
    if not bool(sol['solution_valid']):
        raise RuntimeError(f"Model {model_label}: median draw is not valid")

    P_aug  = np.asarray(sol['P_aug'])
    R_aug  = np.asarray(sol['R_aug'])
    Omega  = np.asarray(sol['Omega'])
    c_aug  = np.asarray(sol['c_aug'])

    # Initial conditions: zero mean, diffuse
    n_aug = P_aug.shape[0]
    init_x = np.zeros(n_aug)
    H_obs = np.diag(np.zeros(len(obs_names)))   # diagonal zero meas noise
    # Use canonical small floor on H to avoid singular Cholesky
    H_obs = H_obs + 1e-6 * np.eye(len(obs_names))

    dt = get_dtype()
    Pinf, Pstar, _ = build_diffuse_init(P_aug, R_aug, dtype=dt)

    dkf = DiffuseKalmanFilter(
        T=P_aug, R=R_aug, C=Omega, H=H_obs,
        Pinf=Pinf, Pstar=Pstar, init_a=init_x, c=c_aug, dtype=dt)
    x_smooth = np.asarray(dkf.smooth_states_only(ys=ys))

    state_names = list(m.model.aug_state_vars)
    return df.index, state_names, x_smooth


def main():
    dates_A, names_A, x_A = smoothed_states_at_median('A', MODELS['A'])
    dates_F, names_F, x_F = smoothed_states_at_median('F', MODELS['F'])

    assert len(dates_A) == len(dates_F)
    assert names_A == names_F
    states = names_A
    dates  = dates_A

    # Long-format CSV
    rows = []
    for s_idx, sname in enumerate(states):
        for t_idx, dt_ in enumerate(dates):
            rows.append({
                'date': dt_, 'state': sname,
                'model_a': float(x_A[t_idx, s_idx]),
                'model_f': float(x_F[t_idx, s_idx]),
                'diff_F_minus_A': float(x_F[t_idx, s_idx] - x_A[t_idx, s_idx]),
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'states_AvF.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Per-state plots
    plot_states = ['LGDP_BAR', 'GDP_GAP', 'G_TREND', 'PI_TREND',
                   'RR_BAR', 'PREM_BAR', 'LZ_BAR', 'TT_BAR',
                   'RESN_GDP_GAP', 'RESN_RS']
    plot_states = [s for s in plot_states if s in states]
    print("\n=== Plotting states ===")
    print(f"  Available: {[s for s in plot_states]}")
    print(f"  All state names containing 'RES': "
          f"{[s for s in states if 'RES' in s.upper()]}")

    n_plot = len(plot_states)
    ncols = 2
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, 3.2 * nrows), sharex=True)
    axes = axes.flatten()
    for ax, sname in zip(axes, plot_states):
        s_idx = states.index(sname)
        ax.plot(dates, x_A[:, s_idx], label='Model A', color='steelblue', lw=1.5)
        ax.plot(dates, x_F[:, s_idx], label='Model F',
                color='crimson', lw=1.2, ls='--')
        ax.set_title(sname, fontsize=11)
        ax.axhline(0, color='k', lw=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, framealpha=0.7)
    for ax in axes[n_plot:]:
        ax.set_visible(False)
    fig.suptitle("Smoothed states at posterior median: Model A vs Model F",
                 fontsize=13)
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'states_AvF.png')
    fig.savefig(plot_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {plot_path}")

    # Summary stats
    print("\n=== State summary (mean | std | range) ===")
    print(f"{'State':<14} {'A mean':>9} {'A std':>9} "
          f"{'F mean':>9} {'F std':>9} {'corr(A,F)':>11}")
    for sname in plot_states:
        s_idx = states.index(sname)
        a = x_A[:, s_idx]; f = x_F[:, s_idx]
        corr = float(np.corrcoef(a, f)[0,1])
        print(f"{sname:<14} {a.mean():>9.3f} {a.std():>9.3f} "
              f"{f.mean():>9.3f} {f.std():>9.3f} {corr:>11.3f}")


if __name__ == "__main__":
    main()

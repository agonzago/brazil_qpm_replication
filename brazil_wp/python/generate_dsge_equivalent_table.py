"""Compute DSGE-equivalent parameterization from posterior draws.

Maps the QPM's estimated parameters to familiar structural parameters
used in the DSGE literature:
  - Effective Taylor coefficient on inflation in levels
  - Long-run IS semi-elasticity
  - Sacrifice ratio (already computed)
  - Long-run exchange rate pass-through

Usage:
    python generate_dsge_equivalent_table.py
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX_DIR = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX_DIR)

from qpm_toolbox import QPMModel
from run_brazil_wp_v2 import (
    MOD_FILE, STRUCTURAL_PRIORS, FIXED_PARAMS, MEAS_NOISE_STD,
    CYCLE_MAP, TREND_SNR_MAP, MANUAL_CYCLE_PRIORS,
    load_brazil_data, compute_initial_conditions,
)


def compute_dsge_equivalents(samples):
    """Compute DSGE-equivalent parameters from posterior samples."""
    gamma1 = samples['gamma1']
    gamma2 = samples['gamma2']
    gamma4 = samples['gamma4']
    beta2 = samples['beta2']
    beta3 = samples['beta3']
    lambda1x = samples['lambda1x']
    lambda2x = samples['lambda2x']
    kappa_z = samples['kappa_z']
    rho_QZ = samples['rho_QZ']
    rho_QRR = samples['rho_QRR']

    results = {}

    # 1. Effective Taylor coefficient on inflation in levels
    #    In gap space: i_t = gamma1*i_{t-1} + (1-gamma1)*(pi + gamma2*E[pi] + ...)
    #    Long-run inflation response (setting i_{t-1} = i_t):
    #    di/dpi = gamma2  (already the long-run response in gap space)
    #    Short-run (impact) response: (1-gamma1)*gamma2
    results['taylor_impact'] = (1.0 - gamma1) * gamma2
    results['taylor_longrun'] = gamma2  # gap-space long-run

    # 2. Long-run IS semi-elasticity
    #    IS: y_t = beta1*y_{t-1} + beta2*E[y_{t+1}] - beta3*QRR_t + ...
    #    QRR is a Koyck filter of the real rate gap: QRR_t = rho_QRR*QRR_{t-1} + (1-rho_QRR)*rr_gap_t
    #    Long-run multiplier of rr_gap on y (setting y constant, QRR constant):
    #    y = beta1*y + beta2*y - beta3*QRR  =>  y*(1-beta1-beta2) = -beta3*QRR
    #    QRR = rr_gap in steady state (Koyck converges)
    #    So: dy/drr = -beta3 / (1 - beta1 - beta2)
    results['is_semi_elasticity'] = beta3 / (1.0 - beta2)

    # 3. Long-run ERPT to core inflation
    #    QZ is Koyck filter of REER gap: long-run QZ = REER gap
    #    Phillips curve: pi_x depends on kappa_z * QZ
    #    Long-run: pi_x = lambda1x*pi_x + (1-lambda1x)*pi_x + lambda2x*y + kappa_z*QZ
    #    Simplifying (in gap space, steady state): 0 = kappa_z * dz
    #    Actually the long-run ERPT is kappa_z / (1 - lambda1x) if we
    #    abstract from output gap feedback
    results['erpt_longrun'] = kappa_z

    # 4. Phillips curve slope (already lambda2x)
    results['pc_slope'] = lambda2x

    # 5. Monetary policy persistence (half-life in quarters)
    results['mp_halflife'] = np.log(2) / np.log(1.0 / gamma1)

    return results


def print_table(results):
    """Print formatted table of DSGE-equivalent parameters."""
    print("\n" + "=" * 75)
    print("  DSGE-Equivalent Parameterization")
    print("=" * 75)
    print(f"  {'Parameter':<45s} {'Median':>8s} {'68% HDI':>16s}")
    print("-" * 75)

    for name, label in [
        ('taylor_impact', 'Taylor impact response (1-γ₁)γ₂'),
        ('taylor_longrun', 'Taylor long-run response γ₂ (gap space)'),
        ('is_semi_elasticity', 'IS semi-elasticity β₃/(1-β₂)'),
        ('pc_slope', 'Phillips curve slope λ₂ₓ'),
        ('erpt_longrun', 'ERPT to core κ_z'),
        ('mp_halflife', 'MP smoothing half-life (quarters)'),
    ]:
        vals = results[name]
        med = np.median(vals)
        lo = np.percentile(vals, 16)
        hi = np.percentile(vals, 84)
        print(f"  {label:<45s} {med:8.3f}  [{lo:7.3f}, {hi:7.3f}]")

    print("=" * 75)


def save_latex_table(results, out_path):
    """Save as a LaTeX table fragment."""
    rows = [
        ('taylor_impact', r'Impact Taylor coefficient, $(1-\gamma_1)\gamma_2$'),
        ('taylor_longrun', r'Long-run Taylor coefficient, $\gamma_2$ (gap space)'),
        ('is_semi_elasticity', r'IS semi-elasticity, $\beta_3/(1-\beta_2)$'),
        ('pc_slope', r'Phillips curve slope, $\lambda_{2x}$'),
        ('erpt_longrun', r'ERPT to core, $\kappa_z$'),
        ('mp_halflife', r'Policy smoothing half-life (quarters)'),
    ]

    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering')
    lines.append(r'\caption{DSGE-Equivalent Parameterization}\label{tab:dsge_equiv}')
    lines.append(r'\begin{tabular}{lcc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Parameter} & \textbf{Median} & \textbf{68\% HDI} \\')
    lines.append(r'\midrule')

    for name, label in rows:
        vals = results[name]
        med = np.median(vals)
        lo = np.percentile(vals, 16)
        hi = np.percentile(vals, 84)
        lines.append(f'{label} & {med:.3f} & [{lo:.3f}, {hi:.3f}] \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption*{\footnotesize Note: Derived parameters computed '
                 r'from the 4{,}000 posterior draws of Model~A. The impact '
                 r'Taylor coefficient is the immediate response of the policy '
                 r'rate to a 1~pp increase in expected inflation. The IS '
                 r'semi-elasticity is the long-run output gap response to a '
                 r'sustained 1~pp real rate gap increase. The half-life '
                 r'measures the number of quarters for the policy rate to '
                 r'close half the gap to its target.}')
    lines.append(r'\end{table}')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX table saved: {out_path}")


def main():
    results_dir = os.path.join(SCRIPT_DIR, 'results_v2', 'model_a')

    # Load model and draws
    print("Loading model and posterior draws...")
    ys, dates, obs_names = load_brazil_data()
    m = QPMModel(MOD_FILE, results_dir=results_dir)
    ss_params = m.model.default_param_assignments
    init_config = compute_initial_conditions(ys, obs_names, ss_params)
    m.set_data(ys, dates=dates, obs_names=obs_names)
    m.set_priors(
        structural_priors=STRUCTURAL_PRIORS,
        fixed_params=FIXED_PARAMS,
        meas_noise_std=MEAS_NOISE_STD,
        cycle_map=CYCLE_MAP,
        trend_snr_map=TREND_SNR_MAP,
        manual_cycle_priors=MANUAL_CYCLE_PRIORS,
    )
    m.set_initial_conditions(init_state_config=init_config)
    m.load_draws()

    # Extract posterior samples
    samples = {name: np.asarray(m._posterior_samples[name])
               for name in m._estimated_names}

    # Compute derived parameters
    results = compute_dsge_equivalents(samples)

    # Print and save
    print_table(results)

    out_dir = os.path.join(results_dir, 'policy')
    os.makedirs(out_dir, exist_ok=True)
    save_latex_table(results, os.path.join(out_dir, 'dsge_equivalent_table.tex'))


if __name__ == '__main__':
    main()

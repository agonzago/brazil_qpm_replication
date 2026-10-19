"""Side-by-side comparison of Model A and Model F posteriors with
economic interpretation.

Builds a table comparing posterior mean / std / 68% HDI between A and F
for all common parameters, grouped by economic role.  Saves CSV and a
markdown summary.
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
A_PATH = os.path.join(SCRIPT_DIR, 'results_v2_diffuse/model_a/estimation/arviz_summary.csv')
F_PATH = os.path.join(SCRIPT_DIR, 'results_v2_diffuse/model_f/estimation/arviz_summary.csv')
OUT_DIR = os.path.join(SCRIPT_DIR, 'results_smoke/compare_af')
os.makedirs(OUT_DIR, exist_ok=True)


GROUPS = {
    'Phillips curve (inflation dynamics)': [
        'lambda1x', 'lambda2x', 'lambda4x', 'lambda5x', 'lambda6x',
        'lambda_E1', 'lambda_F1', 'kappa_z',
    ],
    'IS curve (output gap dynamics)': [
        'beta1', 'beta2', 'beta3', 'beta4', 'beta5',
        'rho_QRR', 'rho_QZ', 'rho_QZ_E', 'rho_QZ_F', 'rho_QTT',
    ],
    'Taylor rule (FIXED in F)': [
        'gamma1', 'gamma2', 'gamma4',
    ],
    'FX / UIP / pass-through': [
        'rho_ls', 'psi_uip', 'rho_prem_gap', 'theta_E', 'theta_F',
    ],
    'Trend convergence speeds': [
        'alpha_g', 'alpha_pi', 'alpha_dlz', 'alpha_rr', 'alpha_prem',
        'alpha_pi_us', 'alpha_rr_us',
    ],
    'TT / commodities': [
        'phi_tt_gap', 'phi_tt_is', 'phi_tt_pc',
        'rho_tt', 'rho_lrwoil_gap', 'rho_lrwfood_gap',
    ],
    'Foreign block': [
        'beta1_US', 'beta2_US', 'beta3_us',
        'gamma2_us', 'gamma4_us',
        'rho_US', 'rho_pie_us',
    ],
    'Cycle shock std (data-driven)': [
        'sigma_SHK_GDP_GAP', 'sigma_SHK_DA_LCPI', 'sigma_SHK_DA_LCPIX',
        'sigma_SHK_DA_LCPI_ENERGY', 'sigma_SHK_DA_LCPI_FOOD',
        'sigma_SHK_DA_LCPI_US', 'sigma_SHK_LZ_GAP',
        'sigma_SHK_LRWOIL_GAP', 'sigma_SHK_LRWFOOD_GAP',
        'sigma_SHK_GAP_US', 'sigma_SHK_PREM_GAP',
        'sigma_SHK_RS', 'sigma_SHK_RS_US', 'sigma_SHK_TT_GAP',
    ],
    'Trend shock std (SNR-pinned)': [
        'sigma_SHK_LGDP_BAR', 'sigma_SHK_G_TREND',
        'sigma_SHK_LZ_BAR', 'sigma_SHK_DLZ_TREND',
        'sigma_SHK_PREM_BAR', 'sigma_SHK_RR_BAR',
        'sigma_SHK_PI_TREND', 'sigma_SHK_RR_BAR_US', 'sigma_SHK_PI_TREND_US',
    ],
}


def load(path):
    df = pd.read_csv(path).rename(columns={'Unnamed: 0': 'param'})
    df = df.set_index('param')
    # Coerce all numeric (some files have stringy ESS)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def fmt(val, decimals=3):
    if pd.isna(val):
        return '   --   '
    if abs(val) >= 100:
        return f'{val:7.1f}'
    elif abs(val) >= 10:
        return f'{val:7.2f}'
    elif abs(val) >= 1:
        return f'{val:7.3f}'
    else:
        return f'{val:7.4f}'


def main():
    A = load(A_PATH)
    F = load(F_PATH)

    rows = []
    for grp, params in GROUPS.items():
        for p in params:
            in_A = p in A.index
            in_F = p in F.index
            if not in_A and not in_F:
                continue
            a_mean = A.at[p, 'mean'] if in_A else np.nan
            a_sd   = A.at[p, 'sd']   if in_A else np.nan
            f_mean = F.at[p, 'mean'] if in_F else np.nan
            f_sd   = F.at[p, 'sd']   if in_F else np.nan

            if pd.notna(a_mean) and pd.notna(f_mean):
                pct = (f_mean - a_mean) / abs(a_mean) * 100 if a_mean != 0 else np.nan
            else:
                pct = np.nan

            rows.append({
                'group': grp,
                'param': p,
                'A_mean': a_mean, 'A_sd': a_sd,
                'F_mean': f_mean, 'F_sd': f_sd,
                'F_minus_A': f_mean - a_mean if (pd.notna(a_mean) and pd.notna(f_mean)) else np.nan,
                'pct_change': pct,
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'params_AvF.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}\n")

    # Also pretty-print table per group
    out_lines = []
    out_lines.append('# Model A vs Model F: parameter comparison\n')
    out_lines.append(f"Source: A = `{A_PATH}`")
    out_lines.append(f"Source: F = `{F_PATH}`\n")

    for grp, params in GROUPS.items():
        out_lines.append(f'\n## {grp}\n')
        out_lines.append('| Parameter | A mean | A sd | F mean | F sd | Δ (F-A) | % change |')
        out_lines.append('|---|---:|---:|---:|---:|---:|---:|')
        any_in = False
        for p in params:
            in_A = p in A.index
            in_F = p in F.index
            if not in_A and not in_F:
                continue
            any_in = True
            a_mean = A.at[p, 'mean'] if in_A else None
            a_sd   = A.at[p, 'sd']   if in_A else None
            f_mean = F.at[p, 'mean'] if in_F else None
            f_sd   = F.at[p, 'sd']   if in_F else None
            if a_mean is not None and f_mean is not None:
                d = f_mean - a_mean
                pct = (d / abs(a_mean) * 100) if a_mean != 0 else float('nan')
                pct_str = f'{pct:+.0f}%' if not pd.isna(pct) else '--'
                d_str = fmt(d)
            else:
                d_str, pct_str = '--', '--'
            out_lines.append(
                f'| `{p}` | {fmt(a_mean)} | {fmt(a_sd)} | '
                f'{fmt(f_mean) if f_mean is not None else "(fixed)"} | '
                f'{fmt(f_sd) if f_sd is not None else "(fixed)"} | '
                f'{d_str} | {pct_str} |'
            )
        if not any_in:
            out_lines.pop(); out_lines.pop()

    md = '\n'.join(out_lines)
    md_path = os.path.join(OUT_DIR, 'params_AvF.md')
    with open(md_path, 'w') as fh:
        fh.write(md)
    print(f"Saved: {md_path}")
    print('\n' + md)


if __name__ == '__main__':
    main()

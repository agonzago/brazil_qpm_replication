"""Cross-check every paper-reported number against the freshly re-estimated outputs.

Output: results_smoke/paper_check/paper_check.md  -- side-by-side table
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(SCRIPT_DIR, 'results_v2_diffuse')
OUT_DIR = os.path.join(SCRIPT_DIR, 'results_smoke', 'paper_check')
os.makedirs(OUT_DIR, exist_ok=True)


def load_summary(model_dir):
    csv = os.path.join(RES, model_dir, 'estimation', 'arviz_summary.csv')
    df = pd.read_csv(csv).rename(columns={'Unnamed: 0': 'param'}).set_index('param')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def get(df, p, col='mean'):
    if p in df.index and col in df.columns:
        v = df.at[p, col]
        return float(v) if pd.notna(v) else None
    return None


def hdi_str(df, p):
    lo = get(df, p, 'eti68_lb')
    hi = get(df, p, 'eti68_ub')
    if lo is not None and hi is not None:
        return f'[{lo:.2f}, {hi:.2f}]'
    return '—'


def main():
    A = load_summary('model_a')
    A0 = load_summary('model_a_const_rr')
    F = load_summary('model_f')

    # --- Sacrifice ratio (pull from CSV) ---
    sr_a = pd.read_csv(os.path.join(RES, 'model_a/policy/sacrifice_ratio_draws.csv')).iloc[:, 0]
    sr_f = pd.read_csv(os.path.join(RES, 'model_f/policy/sacrifice_ratio_draws.csv')).iloc[:, 0]
    sr_a0 = pd.read_csv(os.path.join(RES, 'model_a_const_rr/policy/sacrifice_ratio_draws.csv')).iloc[:, 0]

    # --- Pass-through ---
    pt_a = pd.read_csv(os.path.join(RES, 'model_a/policy/pass_through_draws.csv'))
    pt_f = pd.read_csv(os.path.join(RES, 'model_f/policy/pass_through_draws.csv'))

    def pt_med(df, sh, cv='DA_LCPI'):
        g = df[(df['shock'] == sh) & (df['cpi_var'] == cv)]['cumulative_pt']
        return f'{g.median():.2f} [{g.quantile(0.16):.2f}, {g.quantile(0.84):.2f}]' if len(g) else '—'

    out = []
    out.append('# Paper Results Verification: Canonical (paper) vs New (re-estimated)\n')
    out.append('Note: Paper numbers are from `Brazil_model_version4.tex`. New numbers from `results_v2_diffuse/model_<x>/`.\n')

    # ----- Table 1 (Model A baseline parameters) -----
    out.append('\n## Table 1: Model A baseline structural parameters\n')
    out.append('| Param | Paper mean [HDI68] | New mean [HDI68] | Δ mean | Status |')
    out.append('|---|---|---|---:|---|')

    table1 = [
        # (label, runner_var, paper_mean, paper_HDI)
        ('β₁',  'beta1',     0.394, '[0.26, 0.48]'),
        ('β₃',  'beta3',     0.177, '[0.06, 0.24]'),
        ('λ₁ₓ', 'lambda1x',  0.270, '[0.20, 0.34]'),
        ('λ₂ₓ', 'lambda2x',  0.120, '[0.06, 0.16]'),
        ('γ₁',  'gamma1',    0.921, '[0.91, 0.94]'),
        ('γ₂',  'gamma2',    3.124, '[2.08, 3.91]'),
        ('γ₄',  'gamma4',    0.252, '[0.08, 0.31]'),
        ('ρ_LS','rho_ls',    0.255, '[0.10, 0.34]'),
        ('ψUIP','psi_uip',   0.761, '[0.40, 0.96]'),
        ('α_rr','alpha_rr',  0.180, '[0.07, 0.25]'),
        ('α_dz','alpha_dlz', 0.428, '[0.33, 0.55]'),
        ('α_pi','alpha_pi',  0.004, '[0.001, 0.006]'),
    ]
    for lbl, var, p_mean, p_hdi in table1:
        n_mean = get(A, var, 'mean')
        n_hdi  = hdi_str(A, var)
        if n_mean is None:
            out.append(f'| {lbl} (`{var}`) | {p_mean} {p_hdi} | — | — | ❌ missing |')
            continue
        d = n_mean - p_mean
        pct = (d / abs(p_mean) * 100) if p_mean else float('nan')
        flag = '✓' if abs(pct) < 10 else ('~' if abs(pct) < 30 else '⚠')
        out.append(f'| {lbl} (`{var}`) | {p_mean:.3f} {p_hdi} | {n_mean:.3f} {n_hdi} | {d:+.3f} ({pct:+.0f}%) | {flag} |')

    # ----- Table 2 (Sacrifice ratio) -----
    out.append('\n## Table 2: Sacrifice Ratio (Model A baseline)\n')
    out.append('| Stat | Paper | New | Δ |')
    out.append('|---|---:|---:|---:|')
    paper_sr = {'median': 0.314, 'mean': 0.339, 'std': 0.155}
    out.append(f'| Median | {paper_sr["median"]:.3f} | {sr_a.median():.3f} | {sr_a.median()-paper_sr["median"]:+.3f} |')
    out.append(f'| Mean   | {paper_sr["mean"]:.3f} | {sr_a.mean():.3f} | {sr_a.mean()-paper_sr["mean"]:+.3f} |')
    out.append(f'| Std    | {paper_sr["std"]:.3f} | {sr_a.std():.3f} | {sr_a.std()-paper_sr["std"]:+.3f} |')
    out.append(f'| 68% CI | [0.20, 0.47] | [{sr_a.quantile(0.16):.2f}, {sr_a.quantile(0.84):.2f}] | — |')
    out.append(f'| 90% CI | [0.14, 0.65] | [{sr_a.quantile(0.05):.2f}, {sr_a.quantile(0.95):.2f}] | — |')

    # ----- Table 3 (Pass-through) -----
    out.append('\n## Table 3: Exchange Rate Pass-Through (12Q cumulative)\n')
    out.append('| Source shock | Paper median [HDI68] | New median [HDI68] |')
    out.append('|---|---|---|')
    out.append(f'| Direct ER (SHK_LZ_GAP)  | 0.14 [0.08, 0.21] | {pt_med(pt_a, "SHK_LZ_GAP")} |')
    out.append(f'| US MP (SHK_RS_US)        | 1.50 [0.93, 2.19] | {pt_med(pt_a, "SHK_RS_US")} |')
    out.append(f'| Domestic MP (SHK_RS)     | 1.85 [1.14, 3.27] | {pt_med(pt_a, "SHK_RS")} |')

    # ----- Model A0 spot-check -----
    out.append('\n## Model A0 (constant R*) — spot-check\n')
    out.append('| Param | Paper mean | New mean | Δ |')
    out.append('|---|---:|---:|---:|')
    a0_table = [
        ('γ₂', 'gamma2', 3.315),
        ('λ₁ₓ', 'lambda1x', None),  # not reported in Table 1 differently
        ('ψUIP', 'psi_uip', 0.948),
    ]
    for lbl, var, p_mean in a0_table:
        n_mean = get(A0, var, 'mean')
        if p_mean and n_mean:
            d = n_mean - p_mean
            out.append(f'| {lbl} (`{var}`) | {p_mean:.3f} | {n_mean:.3f} | {d:+.3f} |')
        elif n_mean:
            out.append(f'| {lbl} (`{var}`) | (—) | {n_mean:.3f} | — |')
    a0_sr_med = sr_a0.median()
    out.append(f'| Sacrifice ratio (median) | (—) | {a0_sr_med:.3f} | — |')

    # ----- Model F (counterfactual) -----
    out.append('\n## Model F (Taylor fixed) — counterfactual results\n')
    out.append('| Quantity | Model A (baseline) | Model F (Taylor fixed) | Insight |')
    out.append('|---|---:|---:|---|')
    out.append(f'| λ₁ₓ          | {get(A, "lambda1x"):.3f} | {get(F, "lambda1x"):.3f} | Phillips less backward when Taylor fixed |')
    out.append(f'| β₄ (Koyck QZ_E) | {get(A, "beta4"):.3f} | {get(F, "beta4"):.3f} | REER channel takes over |')
    out.append(f'| ψUIP         | {get(A, "psi_uip"):.3f} | {get(F, "psi_uip"):.3f} | UIP discrepancy bigger |')
    out.append(f'| σ_GDP_GAP    | {get(A, "sigma_SHK_GDP_GAP"):.3f} | {get(F, "sigma_SHK_GDP_GAP"):.3f} | Cycle absorbs more |')
    out.append(f'| σ_LGDP_BAR   | {get(A, "sigma_SHK_LGDP_BAR"):.3f} | {get(F, "sigma_SHK_LGDP_BAR"):.3f} | Trend less variable |')
    out.append(f"| Sacrifice ratio (median) | {sr_a.median():.3f} | {sr_f.median():.3f} | F higher (no aggressive policy) |")
    out.append(f"| PT SHK_RS-> DA_LCPI       | {pt_med(pt_a, 'SHK_RS')} | {pt_med(pt_f, 'SHK_RS')} | F lower |")

    # ----- Convergence check -----
    out.append('\n## Convergence diagnostics (Model A)\n')
    rhat_max = float(A['r_hat'].max()) if 'r_hat' in A.columns else None
    ess_min  = float(A['ess_bulk'].min()) if 'ess_bulk' in A.columns else None
    out.append(f'- max R̂ across all parameters: **{rhat_max:.3f}** (paper threshold ≤ 1.01)')
    out.append(f'- min ESS_bulk: **{ess_min:.0f}** (paper threshold ≥ 640)')

    # ----- Things still STALE (need re-running) -----
    out.append('\n## Still STALE — need re-running\n')
    out.append('| What | Paper section | Generator | Status |')
    out.append('|---|---|---|---|')
    out.append('| Forecast RMSE Table 4 | Section 5 | `run_forecast_evaluation.py` | **STALE** — depends on all 7 models + BVAR re-running |')
    out.append('| Forecast conditioning gains Table 5 | Section 5.2 | `run_forecast_evaluation.py` | **STALE** |')
    out.append('| Robustness Table 10 (RUB1-5) | Appendix D | `run_robustness_v2.py` + `run_brazil_rub5_focus_v2.py` | **STALE** |')
    out.append('| BVAR sacrifice ratio Table 9 | Appendix C.2 | `run_bvar_analysis.py` | likely OK (BVAR independent of QPM priors) |')
    out.append('| Spectral radius (max 0.998) | text + appendix | `compute_spectral_radius.py` | **STALE** |')
    out.append('| Prior vs posterior IRF Fig 6 | Section 4.2 | `generate_prior_vs_posterior_irfs.py` | **STALE** |')
    out.append('| Posterior diagnostics (Figs 21-24, 25-26) | Appendix G | `generate_posterior_diagnostics.py` + `generate_posterior_filtration.py` | **STALE** |')
    out.append('| `output_gap_posterior.png` Fig 11 | Section 4.3 | `generate_additional_plots.py` | **STALE** for some models |')
    out.append('| `potential_growth.png` Fig 12 | Section 4.3 | `generate_additional_plots.py` | **STALE** |')
    out.append('| Taylor decomposition Fig 9 | Section 4.2 | `plot_taylor_decomposition` (in runner) | should be FRESH (in runner output) |')
    out.append('| `decomp_GDP_GROWTH.png` Fig 14 | Section 4.4 | `generate_additional_plots.py` | **STALE** for some models |')

    md = '\n'.join(out)
    md_path = os.path.join(OUT_DIR, 'paper_check.md')
    with open(md_path, 'w') as fh:
        fh.write(md)
    print(f"Saved: {md_path}\n")
    print(md)


if __name__ == '__main__':
    main()

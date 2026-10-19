"""Side study: which sacrifice-ratio definition should a gap-form QPM report?

NOT part of the paper pipeline.  Writes only to
``results_v2_diffuse_v5/_sr_study/`` and touches no paper artefact.

Background.  In a gap-form QPM the cyclical block is written in
deviations from the inflation target, so a permanent target reduction is
neutral for the gaps and the Ball (1994) / Cecchetti-Rich (2001)
experiment (cumulative output loss per unit of *permanent* disinflation)
cannot be run.  The toolbox therefore falls back to a cost-push shock and
reports a ratio of cumulative responses.  Under an endogenous rule that
IRF is not single-signed, so the placement of the absolute value matters.
This script quantifies how much it matters, and computes two additional
statistics that do not depend on the convention at all.

Four objects, all on the same posterior draws:

  A. The three sign conventions of ``policy_analysis.sacrifice_ratio``
     ('symmetric', 'mixed', 'net'), for reference.

  B. The Appendix B.5 experiment: monetary-policy shock, headline
     inflation, horizons 12 and 16, under two conventions.  'signed' is
     the ratio of cumsums that ``run_bvar_analysis.compute_sacrifice_ratio``
     computes today.  'symmetric' applies the paper's single definition
     to the same IRF.  Reporting the latter on both the QPM and the BVAR
     side turns Appendix B.5 into a one-definition comparison instead of
     one that crosses three differences at once (shock, horizon,
     convention).

  C. The counterfactual sacrifice ratio: extra output lost because the
     rule responded, divided by the inflation that response avoided,

         SR_cf = |sum(y_active) - sum(y_passive)|
                 / (sum(pi_passive) - sum(pi_active))

     with the passive rule obtained by switching off the inflation
     response (gamma2 = 0) and holding everything else fixed.  Both
     numerator and denominator are single-signed by construction under a
     stabilising rule, so the sign question disappears.  This is the
     object the sacrifice-ratio literature actually means: output cost
     per unit of inflation avoided.  It is scale-invariant, so the shock
     size does not need normalising.

  D. Diagnostics for why the conventions diverge: the share of the
     absolute inflation IRF mass that is negative (the policy-induced
     undershoot) and the number of sign changes.

Usage:
    JAX_PLATFORMS=cpu ${PYTHON:-python3} \
        sr_convention_study.py [--spec model_c] [--draws 200]
"""

import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation"))
sys.path.insert(0, SCRIPT_DIR)

from qpm_toolbox import QPMModel                                  # noqa: E402
from qpm_toolbox.parser import irf as _irf                        # noqa: E402
from qpm_toolbox.parser import irf_observables as _irf_obs        # noqa: E402

RES_BASE = os.path.join(SCRIPT_DIR, "results_v2_diffuse_v5")
OUT_DIR = os.path.join(RES_BASE, "_sr_study")

MODELS = {
    'model_c': "brazil_qpm_model_c_tt_trend_v2.qpm",
    'model_b': "brazil_qpm_model_b_tt_trend_v2.qpm",
    'model_d': "brazil_qpm_model_d_tt_trend_v2.qpm",
}

# Weights of the CPI components; fixed, not estimated (see FIXED_PARAMS
# in the runners).  Everything else comes from the model defaults plus
# the posterior draw.
FIXED_PARAMS = {'lambda1s': 0.1581, 'lambda2s': 0.1142}

GAP, CORE, HEADLINE = 'GDP_GAP', 'DA_LCPIX', 'DA_LCPI'
CP_SHOCK, MP_SHOCK = 'SHK_DA_LCPIX', 'SHK_RS'
H_SR = 120          # quarters, matches horizon_years=30
H_BVAR_MAX = 16     # run_bvar_analysis reports h=16 in App B.5


def build(spec):
    mod_file = os.path.join(SCRIPT_DIR, "model", MODELS[spec])
    m = QPMModel(mod_file, results_dir=OUT_DIR)
    return m.model


def paths(model, sol, shock, horizon, variables):
    P, R, Om = sol["P_aug"], sol["R_aug"], sol["Omega"]
    st, sh, ob = (model.aug_state_vars, model.aug_shocks,
                  model.ordered_obs_vars)
    j = sh.index(shock)
    s_irf = np.asarray(_irf(P, R, j, horizon))
    o_irf = np.asarray(_irf_obs(P, R, Om, j, horizon))
    out = {}
    for v in variables:
        out[v] = (s_irf[:, st.index(v)] if v in st
                  else o_irf[:, ob.index(v)])
    return out


def one_draw(model, pdict):
    """All four objects for a single parameter vector."""
    sol = model.solve(pdict)
    if not sol["solution_valid"]:
        return None

    # --- A + D: cost-push experiment, normalised to +1pp on impact ----
    pr = paths(model, sol, CP_SHOCK, H_SR, [GAP, CORE])
    pi, y = pr[CORE], pr[GAP]
    if abs(pi[0]) < 1e-12:
        return None
    sc = 1.0 / pi[0]
    pi, y = pi * sc, y * sc

    sum_y, sum_abs_y = float(pi.sum() * 0 + y.sum()), float(np.abs(y).sum())
    sum_pi, sum_abs_pi = float(pi.sum()), float(np.abs(pi).sum())

    row = {
        'sr_symmetric': sum_abs_y / sum_abs_pi,
        'sr_mixed': abs(sum_y) / sum_abs_pi,
        'sr_net': abs(sum_y) / abs(sum_pi) if abs(sum_pi) > 1e-12 else np.nan,
        'pi_negative_mass_share': float(np.abs(pi[pi < 0]).sum()
                                        / sum_abs_pi),
        'pi_sign_changes': int(np.sum(
            np.diff(np.sign(pi[np.abs(pi) > 1e-10])) != 0)),
        'pi_first_zero_cross': int(np.argmax(pi < 0)) if (pi < 0).any()
                               else -1,
    }

    # --- B: same experiment as the BVAR cross-check -------------------
    # Monetary policy shock, headline inflation, horizons 12 and 16.
    # 'signed' is what run_bvar_analysis currently computes; 'symmetric'
    # applies the paper's single definition to the same IRF, which is
    # what makes Appendix B.5 a one-definition comparison.
    pb = paths(model, sol, MP_SHOCK, H_BVAR_MAX + 1, [GAP, HEADLINE, CORE])
    for h in (12, 16):
        yb, pib = pb[GAP][:h + 1], pb[HEADLINE][:h + 1]
        row[f'sr_mp_signed_h{h}'] = (float(yb.sum() / pib.sum())
                                     if abs(pib.sum()) > 1e-12 else np.nan)
        row[f'sr_mp_symmetric_h{h}'] = (
            float(np.abs(yb).sum() / np.abs(pib).sum())
            if np.abs(pib).sum() > 1e-12 else np.nan)

    # --- C: counterfactual (estimated rule vs no inflation response) --
    q = dict(pdict)
    q['gamma2'] = 0.0
    sol_p = model.solve(q)
    if sol_p["solution_valid"]:
        pp = paths(model, sol_p, CP_SHOCK, H_SR, [GAP, CORE])
        pi_p, y_p = pp[CORE] * sc, pp[GAP] * sc   # same shock scaling
        d_y = float(y.sum() - y_p.sum())          # extra output, <0
        d_pi = float(pi_p.sum() - pi.sum())       # inflation avoided, >0
        row['cf_output_cost'] = d_y
        row['cf_inflation_avoided'] = d_pi
        row['sr_counterfactual'] = (abs(d_y) / d_pi if d_pi > 1e-12
                                    else np.nan)
        row['cf_stable'] = bool(np.max(np.abs(pi_p[-4:])) < 1e-3)
    else:
        row['sr_counterfactual'] = np.nan
        row['cf_stable'] = False
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="model_c", choices=sorted(MODELS))
    ap.add_argument("--tree", default="v5plus_free_kappa")
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()

    res = os.path.join(RES_BASE, f"{args.spec}_{args.tree}")
    model = build(args.spec)

    npz = np.load(os.path.join(res, "estimation", "draws.npz"),
                  allow_pickle=True)
    import json
    meta = json.loads(str(npz['_meta_json_']))
    names = meta.get('estimated_names', [])
    cal_sigmas = {k: float(v)
                  for k, v in meta.get('calibrated_sigmas', {}).items()}
    samples = {k: np.asarray(npz[k]) for k in npz.files if k != '_meta_json_'}
    total = samples[names[0]].shape[0]
    step = max(1, total // args.draws)
    idx = list(range(0, total, step))[:args.draws]
    print(f"  {args.spec}: {total} draws available, using {len(idx)}")

    base = dict(model.default_param_assignments)
    base.update(FIXED_PARAMS)
    base.update(cal_sigmas)

    rows, skipped = [], 0
    for i in idx:
        p = dict(base)
        for n in names:
            if n in samples:
                p[n] = float(samples[n][i])
        r = one_draw(model, p)
        if r is None:
            skipped += 1
        else:
            rows.append(r)
    if skipped:
        print(f"  skipped {skipped} draws (no valid solution)")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f"{args.spec}_{args.tree}_sr_study.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n  {'statistic':28s} {'median':>9s} {'16%':>9s} {'84%':>9s}")
    print("  " + "-" * 58)
    for c in ['sr_symmetric', 'sr_mixed', 'sr_net',
              'sr_mp_signed_h12', 'sr_mp_signed_h16',
              'sr_mp_symmetric_h12', 'sr_mp_symmetric_h16',
              'sr_counterfactual']:
        if c not in df:
            continue
        v = df[c].dropna()
        if v.empty:
            print(f"  {c:28s}  (all NaN)")
            continue
        print(f"  {c:28s} {v.median():9.4f} {v.quantile(.16):9.4f} "
              f"{v.quantile(.84):9.4f}")

    print(f"\n  Why the conventions diverge:")
    print(f"    share of |inflation IRF| mass that is negative: "
          f"{df.pi_negative_mass_share.median():.3f}")
    print(f"    sign changes in the inflation IRF (median): "
          f"{df.pi_sign_changes.median():.0f}")
    print(f"    first zero crossing (quarter, median): "
          f"{df.pi_first_zero_cross.median():.0f}")
    if 'cf_stable' in df:
        print(f"    passive-rule counterfactual settles by H: "
              f"{df.cf_stable.mean():.0%} of draws")
    print(f"\n  Saved {out_csv}")


if __name__ == "__main__":
    main()

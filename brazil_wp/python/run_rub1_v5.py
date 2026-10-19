"""RUB1 (v5): Wider structural priors on the four parameters with the largest
prior-to-posterior movements (lambda1x, gamma2, alpha_rr, lambda2x).  Modes
held fixed; only the dispersion is increased.

This wrapper sits on top of run_brazil_wp_v2.py (Model A baseline) and
overrides only the four prior specifications.  Prior values are taken
verbatim from run_robustness_v2.py:make_rub1_priors() (v4 RUB1 design).

Usage:
    JAX_PLATFORMS=cpu V5_TT_PIN=tight V5_PREM_PIN=tight V5_PREM_SNR=15 \\
    V5_TT_GAP_CAP=3.0 V5_TT_MODE=tworegime \\
        .venv/bin/python3 run_rub1_v5.py --use-diffuse \\
            --warmup 1000 --samples 1000 --chains 4 \\
            --results-dir results_v2_diffuse_v5/rub1
"""
from __future__ import annotations

import copy
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import run_brazil_wp_v2 as base


# v4 RUB1 prior overrides (from run_robustness_v2.py:make_rub1_priors).
# Mode-preserving widening: kappa halved on Beta priors, std doubled on Gamma.
RUB1_OVERRIDES = {
    # lambda1x: mode 0.60, kappa 10 -> 5  (Beta a=2.80, b=2.20)
    "lambda1x": {"prior": "beta",
                 "args": {"concentration1": 2.80, "concentration2": 2.20}},
    # gamma2: mode 1.50, std 1.00 -> 2.00 (Gamma a=1.5625, r=0.3750)
    "gamma2":   {"prior": "gamma",
                 "args": {"concentration": 1.5625, "rate": 0.3750}},
    # alpha_rr: mode 0.07, kappa 10 -> 5 (Beta a=1.21, b=3.79)
    "alpha_rr": {"prior": "beta",
                 "args": {"concentration1": 1.21, "concentration2": 3.79}},
    # lambda2x: mode 0.16, std 0.10 -> 0.20 (Gamma a=1.64, r=4.00)
    "lambda2x": {"prior": "gamma",
                 "args": {"concentration": 1.6400, "rate": 4.0000}},
}


def _apply_overrides():
    new_priors = []
    seen = set()
    for p in base.STRUCTURAL_PRIORS:
        if p["name"] in RUB1_OVERRIDES:
            p = copy.deepcopy(p)
            p.update(RUB1_OVERRIDES[p["name"]])
            seen.add(p["name"])
        new_priors.append(p)
    base.STRUCTURAL_PRIORS = new_priors
    missing = set(RUB1_OVERRIDES) - seen
    if missing:
        raise RuntimeError(f"RUB1 override missed priors: {missing}")
    print(f"  RUB1: widened priors on {sorted(seen)}")


_apply_overrides()


if __name__ == "__main__":
    base.main()

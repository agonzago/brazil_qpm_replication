"""RUB3 (v5): Weaker Taylor rule prior.  Shift gamma_2 mode from 1.50 to 1.00
and widen the prior (std 1.00 -> 1.50).  Tests whether the aggressive
posterior gamma_2 ~ 3.0 is robust to a prior that puts more mass on weak
inflation responses.

Prior values verbatim from run_robustness_v2.py:make_rub3_priors() (v4 RUB3).

Usage:
    JAX_PLATFORMS=cpu V5_TT_PIN=tight V5_PREM_PIN=tight V5_PREM_SNR=15 \\
    V5_TT_GAP_CAP=3.0 V5_TT_MODE=tworegime \\
        .venv/bin/python3 run_rub3_v5.py --use-diffuse \\
            --warmup 1000 --samples 1000 --chains 4 \\
            --results-dir results_v2_diffuse_v5/rub3
"""
from __future__ import annotations

import copy
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import run_brazil_wp_v2 as base


# v4 RUB3 prior override: gamma_2 weakened.
# Gamma: a = (1.00/1.50)^2 + 1 = 1.4444, r = 0.4444/1.00 = 0.4444 -> mode 1.00, std 1.50.
RUB3_OVERRIDES = {
    "gamma2": {"prior": "gamma",
               "args": {"concentration": 1.4444, "rate": 0.4444}},
}


def _apply_overrides():
    new_priors = []
    seen = set()
    for p in base.STRUCTURAL_PRIORS:
        if p["name"] in RUB3_OVERRIDES:
            p = copy.deepcopy(p)
            p.update(RUB3_OVERRIDES[p["name"]])
            seen.add(p["name"])
        new_priors.append(p)
    base.STRUCTURAL_PRIORS = new_priors
    missing = set(RUB3_OVERRIDES) - seen
    if missing:
        raise RuntimeError(f"RUB3 override missed priors: {missing}")
    print(f"  RUB3: weakened Taylor-rule prior on {sorted(seen)}")


_apply_overrides()


if __name__ == "__main__":
    base.main()

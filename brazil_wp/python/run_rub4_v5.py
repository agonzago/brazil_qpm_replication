"""RUB4 (v5): Pure random-walk inflation trend (alpha_pi = 0).  Tests
whether the near-random-walk posterior (alpha_pi ~ 0.03 in the baseline)
has material impact on the structural parameter estimates.

Removes alpha_pi from STRUCTURAL_PRIORS and pins it to 0 in FIXED_PARAMS.
This matches run_robustness_v2.py:make_rub4_priors() / remove_prior(...) (v4 RUB4).

Usage:
    JAX_PLATFORMS=cpu V5_TT_PIN=tight V5_PREM_PIN=tight V5_PREM_SNR=15 \\
    V5_TT_GAP_CAP=3.0 V5_TT_MODE=tworegime \\
        .venv/bin/python3 run_rub4_v5.py --use-diffuse \\
            --warmup 1000 --samples 1000 --chains 4 \\
            --results-dir results_v2_diffuse_v5/rub4
"""
from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import run_brazil_wp_v2 as base


def _apply_overrides():
    before = len(base.STRUCTURAL_PRIORS)
    base.STRUCTURAL_PRIORS = [p for p in base.STRUCTURAL_PRIORS
                              if p["name"] != "alpha_pi"]
    after = len(base.STRUCTURAL_PRIORS)
    if after != before - 1:
        raise RuntimeError(f"RUB4: expected to drop alpha_pi (1 prior); "
                           f"dropped {before - after} instead.")
    # Add alpha_pi to FIXED_PARAMS so it's pinned to 0.
    base.FIXED_PARAMS["alpha_pi"] = 0.0
    print("  RUB4: alpha_pi removed from estimation, pinned at 0 "
          "(pure random-walk inflation trend).")


_apply_overrides()


if __name__ == "__main__":
    base.main()

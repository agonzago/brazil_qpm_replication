"""RUB5 (v5): Focus 12M IPCA as second observation of PI_TREND.

Adds the Focus IPCA 12-month-ahead inflation survey as an 18th observable
(15 baseline + OBS_TT_BAR + OBS_TT_BAR_POST + OBS_FOCUS_IPCA), with
measurement equation `OBS_FOCUS_IPCA = PI_TREND` and measurement noise
`sigma = 0.6 pp` (larger than the 0.3 pp on the inflation target,
reflecting the noisier nature of survey-based expectations).

This corresponds to v4's run_brazil_rub5_focus_v2.py adapted to the v5
two-regime TT-anchor baseline.

Uses the combined QPM file ``brazil_qpm_mr_focus_tt_anchor2_v2.qpm`` which
adds OBS_FOCUS_IPCA = PI_TREND on top of brazil_qpm_mr_tt_anchor2_v2.qpm.

Usage:
    JAX_PLATFORMS=cpu V5_TT_PIN=tight V5_PREM_PIN=tight V5_PREM_SNR=15 \\
    V5_TT_GAP_CAP=3.0 V5_TT_MODE=tworegime \\
        .venv/bin/python3 run_rub5_v5.py --use-diffuse \\
            --warmup 1000 --samples 1000 --chains 4 \\
            --results-dir results_v2_diffuse_v5/rub5
"""
from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pandas as pd

import run_brazil_wp_v2 as base
import _v5_sweep_config as _V5SC


# Override MOD_FILE to the combined Focus + tt-anchor2 QPM.
if _V5SC.V5_TT_MODE != "tworegime":
    raise RuntimeError(
        "RUB5 v5 wrapper requires V5_TT_MODE=tworegime "
        f"(got {_V5SC.V5_TT_MODE!r}). Set the env var and re-run.")
base.MOD_FILE = os.path.join(SCRIPT_DIR, "model",
                             "brazil_qpm_mr_focus_tt_anchor2_v2.qpm")

# Add Focus IPCA to measurement-noise dict (sigma = 0.6 pp; v4 RUB5 value).
base.MEAS_NOISE_STD["OBS_FOCUS_IPCA"] = 0.6

# Map the Focus observable to its anchored trend state for diagnostic plots.
base.OBS_TREND_MAP["OBS_FOCUS_IPCA"] = "PI_TREND"


# Override the data loader to append OBS_FOCUS_IPCA from the FOCUS_IPCA_12M
# column.  Everything else (TT_BAR / TT_BAR_POST under tworegime, OBS_TT, etc.)
# is identical to base.load_brazil_data.
_BASE_LOAD = base.load_brazil_data


def load_brazil_data():
    df = pd.read_csv(base.DATA_CSV, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="Q").to_timestamp()
    df["OBS_RR_BAR_US"] = df["RR_BAR_US"]

    obs_names = [
        "OBS_LGDP", "OBS_DA_LCPI", "OBS_DA_LCPIX", "OBS_DA_LCPI_ENERGY",
        "OBS_DA_LCPI_US", "OBS_DA_LS", "OBS_LRWOIL_GAP", "OBS_LRWFOOD_GAP",
        "OBS_GDP_GAP_US", "OBS_PREM", "OBS_RS", "OBS_RS_US", "OBS_TT",
        "OBS_DA_LCPI_TAR", "OBS_RR_BAR_US",
    ]
    if _V5SC.V5_TT_MODE == "tworegime":
        missing = [c for c in ("TT_BAR_PRE_HP", "TT_BAR_POST_EXTRAP")
                   if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"V5_TT_MODE=tworegime requires columns {missing}")
        df["OBS_TT_BAR"]      = df["TT_BAR_PRE_HP"]
        df["OBS_TT_BAR_POST"] = df["TT_BAR_POST_EXTRAP"]
        obs_names = obs_names + ["OBS_TT_BAR", "OBS_TT_BAR_POST"]

    # RUB5: Focus IPCA 12M as additional PI_TREND observation.
    if "FOCUS_IPCA_12M" not in df.columns:
        raise RuntimeError("RUB5 requires FOCUS_IPCA_12M column in "
                           "estimation_data.csv.")
    df["OBS_FOCUS_IPCA"] = df["FOCUS_IPCA_12M"]
    obs_names = obs_names + ["OBS_FOCUS_IPCA"]

    ys = df[obs_names].values.astype(np.float64)
    dates = df.index
    n_focus_nan = int(np.isnan(df["FOCUS_IPCA_12M"]).sum())
    if n_focus_nan:
        print(f"  RUB5: OBS_FOCUS_IPCA has {n_focus_nan}/{len(dates)} "
              f"NaN entries (handled natively by diffuse smoother).")
    return ys, dates, obs_names


base.load_brazil_data = load_brazil_data
print(f"  RUB5: QPM = {os.path.basename(base.MOD_FILE)}; "
      f"OBS_FOCUS_IPCA added (sigma=0.6 pp).")


if __name__ == "__main__":
    base.main()

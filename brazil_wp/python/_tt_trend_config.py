"""TT-trend variant helper module (v5+ structural rewrite).

Provides shared wiring for the TT-trend coupling adopted as the v5+
headline specification (plan file:
brazil_wp/paper/paper_v5/v5plus_revision_plan.md).

The TT-trend variant introduces a new trend state ``D_TT_BAR`` (ToT
trend growth) that couples to:

  - Potential growth ``G_TREND`` via ``kappa_g_tt * (D_TT_BAR - d_tt_ss)``
    (4GM Gonzalez et al. 2020, eq. 2; Demidenko et al. 2016 IMF QPM).
  - Trend REER depreciation ``DLZ_TREND`` via
    ``-nu_dlz_tt * (D_TT_BAR - d_tt_ss)`` (4GM eq. 9; enters with minus).

The variant nests the v5 baseline (``kappa_g_tt = nu_dlz_tt = 0,
d_tt_ss = 0``).  Identification: ``kappa_g_tt -> G_TREND`` identifies
through ``OBS_LGDP`` and ``OBS_TT``; ``nu_dlz_tt -> DLZ_TREND``
identifies through ``OBS_DA_LS`` (via the DA_LS_TREND identity) and
``OBS_TT``.

Opt1 winner calibration (memorialised here):
  - rho_tt prior Beta(10, 10) -- mode 0.50, sd ~ 0.11; suppresses near-RW
    behaviour of TT_GAP that absorbed trend movement.
  - CYCLE_MAP['OBS_TT'] SNR 0.30  (was 1.0 default, 0.30 tightens the
    cycle, leaving more variance in TT_BAR for tracking the data).
  - FIXED_POINT_CONFIG sigma_SHK_TT_GAP capped at 5.0 with alpha=20
    post-cal pin to prevent the cycle shock from inflating during MCMC.
  - kappa_g_tt fixed at 0.02 (= 4GM/5 -- avoids over-amplification of
    potential growth swings; data alone identify the level).
  - nu_dlz_tt fixed at 0.0 (Brazil less commodity-sensitive on REER than
    Colombia: pinning to 0 sidesteps weak identification).
  - d_lz_ss = 0.50 (verbatim from baseline mod file; the 0.05 variant
    pulled LZ_GAP persistently negative and RR_BAR below its UIP-implied
    target).

Mirrors the pattern of ``_v5_sweep_config.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


# TT-trend mod files (basenames).  Used by is_tt_trend(mod_file_path).
TT_TREND_MOD_FILES: frozenset[str] = frozenset({
    "brazil_qpm_mr_tt_trend_v2.qpm",         # Model A TT-trend
    "brazil_qpm_mr_tt_trend_v3.qpm",         # Model A TT-trend (DA_LS identity fix)
    "brazil_qpm_mr_tt_trend_anchor_v2.qpm",  # Model A TT-trend + tworegime
    "brazil_qpm_model_b_tt_trend_v2.qpm",    # Model B TT-trend (v5+)
    "brazil_qpm_model_c_tt_trend_v2.qpm",    # Model C TT-trend (v5+)
    "brazil_qpm_model_d_tt_trend_v2.qpm",    # Model D TT-trend (v5+)
    "brazil_qpm_mr_focus_tt_trend_v2.qpm",   # RUB5 Focus IPCA + TT-trend (v5+, Model A base)
    "brazil_qpm_model_c_focus_tt_trend_v2.qpm",  # RUB5-C: Focus IPCA + TT-trend (Model C base)
})


def is_tt_trend(mod_file_path: str | Path) -> bool:
    """Return True iff ``mod_file_path`` is a TT-trend variant.

    Accepts an absolute path, relative path, or bare basename.
    """
    return Path(mod_file_path).name in TT_TREND_MOD_FILES


def tt_trend_priors() -> list[dict]:
    """Gamma priors on ``kappa_g_tt`` and ``nu_dlz_tt`` (4GM eqs. 2 / 9).

    Modes 0.10 and 0.20 respectively; sd ~ 0.07 and 0.14.  These match
    the original priors in run_brazil_wp_v2.py; opt-style pinning (e.g.
    kappa_g_tt = 0.02 fixed) is layered on top via
    ``apply_opt1_overrides``.
    """
    return [
        {"name": "kappa_g_tt", "prior": "gamma",
         "args": {"concentration": 3.78, "rate": 27.8}},
        {"name": "nu_dlz_tt",  "prior": "gamma",
         "args": {"concentration": 3.78, "rate": 13.9}},
    ]


def tt_trend_snr_entry() -> tuple[str, tuple[str, str, int, str]]:
    """``TREND_SNR_MAP`` entry for ``SHK_D_TT_BAR``.

    Anchors the ToT trend-growth shock to OBS_TT with SNR=50 in the
    'growth' family (same family as SHK_G_TREND vs OBS_LGDP).
    """
    return ("SHK_D_TT_BAR", ("OBS_TT", "tt_idio_trend", 50, "growth"))


# ----------------------------------------------------------------------
# opt1 calibration defaults (baked-in when V5_TT_MODE=tt_trend)
# ----------------------------------------------------------------------

OPT1_RHO_TT_BETA: tuple[float, float] = (10.0, 10.0)
OPT1_TT_GAP_PIN: str = "tight"           # alpha=20 pin on sigma_SHK_TT_GAP
OPT1_TT_CYCLE_SNR: float = 0.30
OPT1_TT_GAP_CAP: float = 5.0
OPT1_KAPPA_G_TT: float = 0.02
OPT1_NU_DLZ_TT: float = 0.0


def apply_opt1_calibration(
    structural_priors: list[dict],
    cycle_map: dict,
    fixed_point_config: dict,
    fixed_params: dict,
) -> None:
    """Bake the opt1 winner calibration into the runner's data structures.

    Idempotent.  Side-effects:
      - rho_tt prior -> Beta(OPT1_RHO_TT_BETA)
      - CYCLE_MAP['OBS_TT'] SNR -> OPT1_TT_CYCLE_SNR
      - FIXED_POINT_CONFIG sigma_SHK_TT_GAP bound -> (1e-8, OPT1_TT_GAP_CAP)
      - kappa_g_tt prior dropped, FIXED_PARAMS['kappa_g_tt'] -> OPT1_KAPPA_G_TT
      - nu_dlz_tt prior dropped, FIXED_PARAMS['nu_dlz_tt'] -> OPT1_NU_DLZ_TT
      - alpha=20 InvGamma pin on sigma_SHK_TT_GAP applied post-calibration
        is handled by the runner's existing _v5_sweep_config logic when
        V5_TT_PIN='tight' (set via env var by run_all_models_v5plus.sh).

    The function reports each change via print() so the runner log
    captures what was set.
    """
    a, b = OPT1_RHO_TT_BETA
    for p in structural_priors:
        if p["name"] == "rho_tt":
            p["args"]["concentration1"] = a
            p["args"]["concentration2"] = b
            print(f"  v5+ opt1: rho_tt -> Beta({a},{b}) "
                  f"mode={(a-1)/(a+b-2):.3f}")
            break

    if "OBS_TT" in cycle_map:
        shk, _ = cycle_map["OBS_TT"]
        cycle_map["OBS_TT"] = (shk, OPT1_TT_CYCLE_SNR)
        print(f"  v5+ opt1: CYCLE_MAP['OBS_TT'] SNR -> {OPT1_TT_CYCLE_SNR}")

    bounds = fixed_point_config.setdefault("sigma_bounds", {})
    bounds["sigma_SHK_TT_GAP"] = (1e-8, OPT1_TT_GAP_CAP)
    print(f"  v5+ opt1: FIXED_POINT_CONFIG sigma_SHK_TT_GAP bound "
          f"-> (1e-8, {OPT1_TT_GAP_CAP})")

    # Drop priors and pin fixed values for kappa_g_tt and nu_dlz_tt
    # unless V5_TT_FREE_KAPPA=1 is set, in which case kappa_g_tt is left
    # in structural_priors with a tight Gamma centered at 0.02
    # (mode 0.02, sd ~0.015 from Gamma(concentration=3.49, rate=124.57)).
    free_kappa = os.environ.get("V5_TT_FREE_KAPPA", "0").strip() == "1"
    if free_kappa:
        for p in structural_priors:
            if p["name"] == "kappa_g_tt":
                p["args"]["concentration"] = 3.49
                p["args"]["rate"] = 124.57
                print("  v5+ opt1: V5_TT_FREE_KAPPA=1 -> kappa_g_tt prior "
                      "Gamma(3.49, 124.57) mode=0.020 sd~0.015")
                break
        structural_priors[:] = [p for p in structural_priors
                                if p["name"] != "nu_dlz_tt"]
        fixed_params["nu_dlz_tt"] = OPT1_NU_DLZ_TT
        print(f"  v5+ opt1: nu_dlz_tt fixed at {OPT1_NU_DLZ_TT}")
    else:
        structural_priors[:] = [p for p in structural_priors
                                if p["name"] not in ("kappa_g_tt", "nu_dlz_tt")]
        fixed_params["kappa_g_tt"] = OPT1_KAPPA_G_TT
        fixed_params["nu_dlz_tt"] = OPT1_NU_DLZ_TT
        print(f"  v5+ opt1: kappa_g_tt fixed at {OPT1_KAPPA_G_TT}, "
              f"nu_dlz_tt fixed at {OPT1_NU_DLZ_TT}")


def apply_env_overrides(
    structural_priors: list[dict],
    cycle_map: dict,
    fixed_point_config: dict,
    fixed_params: dict,
) -> None:
    """Apply optional ALT_* env-var overrides on top of opt1 defaults.

    Mirrors the env-hook behaviour previously inlined in
    run_brazil_wp_v2.py.  Each ALT_* is optional; an empty string leaves
    the structures unchanged.

    Recognised env vars:
      - ALT_RHO_TT_BETA       : 'a,b'  -> rho_tt prior Beta(a,b)
      - ALT_CYCLE_TT_SNR      : float  -> CYCLE_MAP['OBS_TT'] SNR
      - ALT_KAPPA_G_TT        : float  -> pin kappa_g_tt (drop prior)
      - ALT_NU_DLZ_TT         : float  -> pin nu_dlz_tt (drop prior)
      - ALT_TT_GAP_CAP        : float  -> sigma_SHK_TT_GAP upper bound
    """
    alt_rho = os.environ.get("ALT_RHO_TT_BETA", "").strip()
    if alt_rho:
        a_str, b_str = alt_rho.split(",")
        a, b = float(a_str), float(b_str)
        for p in structural_priors:
            if p["name"] == "rho_tt":
                p["args"]["concentration1"] = a
                p["args"]["concentration2"] = b
                print(f"  ALT_RHO_TT_BETA: rho_tt -> Beta({a},{b}) "
                      f"mode={(a-1)/(a+b-2):.3f}")
                break

    alt_snr = os.environ.get("ALT_CYCLE_TT_SNR", "").strip()
    if alt_snr:
        snr = float(alt_snr)
        if "OBS_TT" in cycle_map:
            shk, _ = cycle_map["OBS_TT"]
            cycle_map["OBS_TT"] = (shk, snr)
            print(f"  ALT_CYCLE_TT_SNR: CYCLE_MAP['OBS_TT'] SNR -> {snr}")

    alt_k = os.environ.get("ALT_KAPPA_G_TT", "").strip()
    if alt_k:
        k = float(alt_k)
        structural_priors[:] = [p for p in structural_priors
                                if p["name"] != "kappa_g_tt"]
        fixed_params["kappa_g_tt"] = k
        print(f"  ALT_KAPPA_G_TT: kappa_g_tt fixed at {k}")

    alt_n = os.environ.get("ALT_NU_DLZ_TT", "").strip()
    if alt_n:
        n = float(alt_n)
        structural_priors[:] = [p for p in structural_priors
                                if p["name"] != "nu_dlz_tt"]
        fixed_params["nu_dlz_tt"] = n
        print(f"  ALT_NU_DLZ_TT: nu_dlz_tt fixed at {n}")

    alt_cap = os.environ.get("ALT_TT_GAP_CAP", "").strip()
    if alt_cap:
        cap = float(alt_cap)
        bounds = fixed_point_config.setdefault("sigma_bounds", {})
        bounds["sigma_SHK_TT_GAP"] = (1e-8, cap)
        print(f"  ALT_TT_GAP_CAP: sigma_SHK_TT_GAP capped at {cap}")


def configure_tt_trend(
    mod_file: str | Path,
    structural_priors: list[dict],
    cycle_map: dict,
    trend_snr_map: dict,
    fixed_point_config: dict,
    fixed_params: dict,
    apply_opt1: bool = True,
) -> bool:
    """Top-level entry point: configure all TT-trend wiring for a runner.

    Returns True iff ``mod_file`` is a TT-trend variant.  When True:
      1. Extends ``structural_priors`` with Gamma priors on
         kappa_g_tt and nu_dlz_tt.
      2. Adds the SHK_D_TT_BAR entry to ``trend_snr_map``.
      3. If ``apply_opt1`` (the default when V5_TT_MODE=tt_trend),
         applies the opt1 winner calibration.
      4. Applies any ALT_* env-var overrides on top.
    """
    if not is_tt_trend(mod_file):
        return False

    structural_priors.extend(tt_trend_priors())
    key, val = tt_trend_snr_entry()
    trend_snr_map[key] = val

    if apply_opt1:
        apply_opt1_calibration(
            structural_priors, cycle_map, fixed_point_config, fixed_params,
        )

    apply_env_overrides(
        structural_priors, cycle_map, fixed_point_config, fixed_params,
    )
    return True

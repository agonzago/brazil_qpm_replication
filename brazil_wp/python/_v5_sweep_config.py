"""v5 sweep config helper.

Reads the env vars that control the v5 model-selection sweep:

- V5_TT_PIN:       'tight' or 'loose'
                   tight ⇒ TT_SNR=50 + cap on σ_SHK_TT_GAP at V5_TT_GAP_CAP
                          + α=20 InvGamma prior pin on both TT shocks
                   loose ⇒ canonical (SNR=100, no cap, default α=3)

- V5_PREM_PIN:     'tight' or 'loose'
                   tight ⇒ α=20 prior pin on σ_SHK_PREM_BAR after calibrate
                   loose ⇒ default α=3

- V5_PREM_SNR:     integer (default 30). Replaces whatever SNR was hardcoded
                   in the runner for SHK_PREM_BAR.

- V5_TT_GAP_CAP:   float (default 3.0). Upper bound on sigma_SHK_TT_GAP
                   used by the fixed-point only when V5_TT_PIN == 'tight'.

- V5_TIGHT_ALPHA:  float (default 20.0). InvGamma α applied to pinned
                   trend shocks after m.calibrate().

All values are resolved at module-import time, so set env vars BEFORE
invoking the runner.
"""
import os


def _read_env() -> dict:
    tt = os.environ.get('V5_TT_PIN', 'tight').lower()
    pp = os.environ.get('V5_PREM_PIN', 'loose').lower()
    ps = int(os.environ.get('V5_PREM_SNR', '30'))
    cap = float(os.environ.get('V5_TT_GAP_CAP', '3.0'))
    alpha = float(os.environ.get('V5_TIGHT_ALPHA', '20.0'))
    mode = os.environ.get('V5_TT_MODE', 'free').lower()
    tt_bar_pre_std  = float(os.environ.get('V5_TT_BAR_PRE_STD',  '0.10'))
    tt_bar_post_std = float(os.environ.get('V5_TT_BAR_POST_STD', '0.50'))
    assert tt in ('tight', 'loose'), f"V5_TT_PIN must be tight/loose, got {tt!r}"
    assert pp in ('tight', 'loose'), f"V5_PREM_PIN must be tight/loose, got {pp!r}"
    assert mode in ('free', 'tworegime', 'hp_full', 'tt_trend'), \
        f"V5_TT_MODE must be 'free', 'tworegime', 'hp_full', or 'tt_trend', got {mode!r}"
    return dict(tt_pin=tt, prem_pin=pp, prem_snr=ps, tt_cap=cap,
                tight_alpha=alpha, tt_mode=mode,
                tt_bar_pre_std=tt_bar_pre_std,
                tt_bar_post_std=tt_bar_post_std)


_CFG = _read_env()
V5_TT_PIN: str = _CFG['tt_pin']
V5_PREM_PIN: str = _CFG['prem_pin']
V5_PREM_SNR: int = _CFG['prem_snr']
V5_TT_GAP_CAP: float = _CFG['tt_cap']
TIGHT_ALPHA: float = _CFG['tight_alpha']
# v5+: TT modelling regime.
#
#   *** PAPER HEADLINE = 'tt_trend' ONLY.  The other modes are
#   *** EXPLORATORY EXPERIMENTS that do NOT feed the v5+ paper.
#
#   'tt_trend'  -> v5+ headline (THE PAPER).  Selects the *_tt_trend_v2.qpm
#                  mod-file variant for the active model (introduces D_TT_BAR
#                  with kappa_g_tt coupling to G_TREND and -nu_dlz_tt coupling
#                  to DLZ_TREND).  Pairs with V5_TT_PIN='tight' to apply opt1
#                  calibration via _tt_trend_config.configure_tt_trend.
#
#   --- experiments below: NOT used anywhere in the paper ---
#   'free'      -> 15 observables, TT_BAR a free latent random walk.  Pre-v5+
#                  exploration; superseded by 'tt_trend'.
#   'tworegime' -> EXPERIMENT, not in the paper.  17 observables: the
#                  constrained-HP two-regime Russia/no-Russia anchor.
#                  OBS_TT_BAR (HP on data <= 2021Q4) anchors the pre-Russia
#                  trend hard (std = V5_TT_BAR_PRE_STD, default 0.10);
#                  OBS_TT_BAR_POST (linear extrapolation forward of that
#                  pre-Russia HP) anchors the post-Russia trend softly
#                  (std = V5_TT_BAR_POST_STD, default 0.50).  Loads the
#                  *_tt_anchor*.qpm variants.  Do not cite in the paper.
#   'hp_full'   -> EXPERIMENT, not in the paper.  Full-sample HP anchor on TT.
V5_TT_MODE: str = _CFG['tt_mode']
V5_TT_BAR_PRE_STD:  float = _CFG['tt_bar_pre_std']
V5_TT_BAR_POST_STD: float = _CFG['tt_bar_post_std']

# Derived: SNR for SHK_TT_BAR and the (low,high) cap for sigma_SHK_TT_GAP
TT_SNR: int = 50 if V5_TT_PIN == 'tight' else 100
TT_CAP = (1e-8, V5_TT_GAP_CAP) if V5_TT_PIN == 'tight' else None


def apply_post_calibration_pins(m) -> None:
    """Bump InvGamma α on selected shocks after calibration."""
    targets = []
    if V5_TT_PIN == 'tight':
        targets.extend(['sigma_SHK_TT_BAR', 'sigma_SHK_TT_GAP'])
        # TT-TREND-ANCHOR: pin sigma_SHK_D_TT_BAR too when present, so the
        # data can't pull D_TT_BAR variance wide and turn it into a
        # high-frequency state.  Only pinned if the prior exists on the
        # model (i.e. the TT-trend variant is active).
        if any(p['name'] == 'sigma_SHK_D_TT_BAR' for p in m._priors):
            targets.append('sigma_SHK_D_TT_BAR')
    if V5_PREM_PIN == 'tight':
        targets.append('sigma_SHK_PREM_BAR')
    if not targets:
        print("  v5 sweep: no posterior prior pins (TT loose + PREM loose)")
        return
    for prior in m._priors:
        if prior['name'] in targets:
            a_old = prior['args']['concentration']
            r_old = prior['args']['rate']
            mode = r_old / (a_old + 1)
            prior['args']['concentration'] = TIGHT_ALPHA
            prior['args']['rate'] = (TIGHT_ALPHA + 1) * mode
            print(f"  v5 sweep pin: {prior['name']} α {a_old:.1f} → "
                  f"{TIGHT_ALPHA:.1f}, mode preserved at {mode:.4f}")


def banner() -> None:
    print("=" * 60)
    print("  v5 sweep config")
    print("=" * 60)
    print(f"  V5_TT_PIN     = {V5_TT_PIN}   (TT_SNR={TT_SNR}, TT_CAP={TT_CAP})")
    print(f"  V5_PREM_PIN   = {V5_PREM_PIN}")
    print(f"  V5_PREM_SNR   = {V5_PREM_SNR}")
    print(f"  V5_TT_GAP_CAP = {V5_TT_GAP_CAP}  (only used when TT_PIN=tight)")
    print(f"  TIGHT_ALPHA   = {TIGHT_ALPHA}")
    print(f"  V5_TT_MODE    = {V5_TT_MODE}", end="")
    if V5_TT_MODE == 'tworegime':
        print(f"   (PRE std={V5_TT_BAR_PRE_STD}, POST std={V5_TT_BAR_POST_STD})")
    else:
        print()
    print("=" * 60)

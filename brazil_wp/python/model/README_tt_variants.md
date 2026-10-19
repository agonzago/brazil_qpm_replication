# TT-trend model-file variants — what feeds the paper, what does not

The `V5_TT_MODE` runner flag (see `_v5_sweep_config.py`) selects which
terms-of-trade (TT) modelling regime, and hence which `.qpm` model file,
a runner loads.  **Only `tt_trend` feeds the v5+ paper.**

## Paper headline — USE THESE

`V5_TT_MODE=tt_trend` loads the `*_tt_trend_v2.qpm` variants:

- `brazil_qpm_model_b_tt_trend_v2.qpm`
- `brazil_qpm_model_c_tt_trend_v2.qpm`   ← headline baseline (Model C)
- `brazil_qpm_model_d_tt_trend_v2.qpm`
- `brazil_qpm_model_c_focus_tt_trend_v2.qpm`
- `brazil_qpm_mr_tt_trend_v2.qpm` / `_v3` (Model A)
- `brazil_qpm_mr_focus_tt_trend_v2.qpm`

These introduce the `D_TT_BAR` trend-growth state with `kappa_g_tt`
coupling to `G_TREND` (and `-nu_dlz_tt` to `DLZ_TREND`).  This is the
v5+ headline structural adaptation.

## Experiments — NOT in the paper

`V5_TT_MODE=tworegime` loads the `*_tt_anchor*.qpm` variants:

- `brazil_qpm_model_b_tt_anchor2_v2.qpm`
- `brazil_qpm_model_c_tt_anchor2_v2.qpm`
- `brazil_qpm_model_d_tt_anchor2_v2.qpm`
- `brazil_qpm_mr_focus_tt_anchor2_v2.qpm`
- `brazil_qpm_mr_tt_anchor2_v2.qpm`
- `brazil_qpm_mr_tt_anchor_v2.qpm`
- `brazil_qpm_mr_tt_trend_anchor_v2.qpm`

These implement the **constrained-HP two-regime Russia/no-Russia anchor**
(`OBS_TT_BAR` on data through 2021Q4, `OBS_TT_BAR_POST` extrapolated
afterward).  This was an exploration only.  **It does not appear in the
v5+ paper** — no prose, no results, no figures — and must not be cited
there.  The only legitimate Russia/Ukraine reference in the paper is the
§5.3 economics of the 2022 terms-of-trade surge, which is unrelated to
this anchor machinery.

See `CLAUDE.md` rule 5b (repo root and `brazil_wp/paper/paper_v5/`).

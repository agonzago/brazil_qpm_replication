#!/usr/bin/env bash
# =============================================================================
# Brazil WP v5 -- post-Phase-1/2 pipeline orchestrator
# =============================================================================
#
# Runs Phases 3, 4, 6, 5 (in that order, longest last) sequentially.
# Designed to run as a single `systemd-run --user` unit, fully detached.
#
# Each phase runs even if a previous one failed (no `set -e`).  Per-phase
# logs go to results_v2_diffuse/v5_pipeline/<phase>.log, and a single
# status file (results_v2_diffuse/v5_pipeline/STATUS.tsv) is updated with
# `<phase>\t<state>\t<ts>` for hourly /loop ticks to inspect.
#
# Usage:
#   systemd-run --user --unit=brazil-qpm-v5-pipeline \
#     --working-directory=. \
#     bash run_v5_pipeline.sh
# -----------------------------------------------------------------------------

set -uo pipefail   # no -e: phases continue past failures

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPTS_DIR/../.." && pwd)"
PY="$REPO/.venv/bin/python3"

OUT_BASE="$SCRIPTS_DIR/results_v2_diffuse"
LOG_DIR="$OUT_BASE/v5_pipeline"
STATUS="$LOG_DIR/STATUS.tsv"
mkdir -p "$LOG_DIR"

export JAX_PLATFORMS=cpu
export N_PARALLEL=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_phase() {
    local name="$1"
    shift
    local log="$LOG_DIR/${name}.log"
    printf "%s\tSTART\t%s\n" "$name" "$(date -Iseconds)" >> "$STATUS"
    {
        echo "============================================================"
        echo "  Phase: $name   started $(date)"
        echo "  Command: $*"
        echo "============================================================"
    } | tee "$log"
    "$@" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    {
        echo "============================================================"
        echo "  Phase $name finished $(date)  rc=$rc"
        echo "============================================================"
    } | tee -a "$log"
    if [[ $rc -eq 0 ]]; then
        printf "%s\tOK\t%s\n" "$name" "$(date -Iseconds)" >> "$STATUS"
    else
        printf "%s\tFAIL(%d)\t%s\n" "$name" "$rc" "$(date -Iseconds)" >> "$STATUS"
    fi
    return $rc
}

cd "$SCRIPTS_DIR"

printf "PIPELINE\tSTART\t%s\n" "$(date -Iseconds)" >> "$STATUS"

# -----------------------------------------------------------------------------
# Phase 3 -- forecast evaluation (Tables 4 and 5)
# Runs both BVAR and QPM rolling-window evaluation; --run-qpm forces refresh
# from results_v2_diffuse/model_a draws.
# -----------------------------------------------------------------------------
run_phase phase3_forecast_eval "$PY" run_forecast_evaluation.py --run-qpm

# -----------------------------------------------------------------------------
# Phase 4 -- BVAR analysis (Tables 7 and 9: sacrifice ratio + sign restrictions)
# -----------------------------------------------------------------------------
run_phase phase4_bvar "$PY" run_bvar_analysis.py

# -----------------------------------------------------------------------------
# Phase 6 -- A vs F comparison (feeds Section 4.7 appendix)
# -----------------------------------------------------------------------------
run_phase phase6_compare_af_params "$PY" compare_a_f_params.py
run_phase phase6_compare_af_states "$PY" compare_a_f_states.py

# -----------------------------------------------------------------------------
# Phase 5 -- robustness (Table 10: RUB1-5 + RUB5 focus)  ← longest, run last
# -----------------------------------------------------------------------------
run_phase phase5_robustness_main "$PY" run_robustness_v2.py \
    --spec all --warmup 1000 --samples 1000 --chains 4

run_phase phase5_robustness_rub5 "$PY" run_brazil_rub5_focus_v2.py \
    --warmup 1000 --samples 1000 --chains 4

# -----------------------------------------------------------------------------
# Bonus -- paper-check verification (feeds Phase 8 numerical updates)
# -----------------------------------------------------------------------------
run_phase phase8_prep_verify "$PY" verify_paper_results.py

printf "PIPELINE\tDONE\t%s\n" "$(date -Iseconds)" >> "$STATUS"
echo "============================================================"
echo "  v5 pipeline complete.  See $STATUS for per-phase status."
echo "============================================================"

#!/usr/bin/env bash
# Retry of failed phases from the first run_v5_pipeline.sh batch.
# - phase4_bvar: data path was wrong (estimation_data/ -> shared_data/)
# - phase5_robustness_main: --use-diffuse is not a valid flag (default IS diffuse)
# - phase5_robustness_rub5: same --use-diffuse issue
# All three coding-only fixes applied; this script re-runs them and appends to
# the same STATUS.tsv as the original orchestrator.

set -uo pipefail

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
        echo "  Phase: $name (RETRY)   started $(date)"
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

printf "RETRY\tSTART\t%s\n" "$(date -Iseconds)" >> "$STATUS"

# Run quick one first so we get visible feedback before the long robustness job
run_phase phase4_bvar "$PY" run_bvar_analysis.py
run_phase phase5_robustness_rub5 "$PY" run_brazil_rub5_focus_v2.py \
    --warmup 1000 --samples 1000 --chains 4
run_phase phase5_robustness_main "$PY" run_robustness_v2.py \
    --spec all --warmup 1000 --samples 1000 --chains 4

printf "RETRY\tDONE\t%s\n" "$(date -Iseconds)" >> "$STATUS"
echo "Retry batch complete; see $STATUS."

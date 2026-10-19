#!/usr/bin/env bash
# =============================================================================
# Brazil WP v5 -- GPU pipeline (RUB5 focus + robustness main RUB1-4)
# =============================================================================
#
# Runs on the .venv-gpu venv (jax[cuda12]).  BK stability check auto-disables
# on GPU per CLAUDE.md (host-callback eig overhead); unstable draws are
# rejected via NaN log-likelihoods downstream.
#
# Single CUDA device (RTX 4070 SUPER, 12 GB).  N_PARALLEL is set to 1 since
# we don't need XLA virtual chain devices on a single physical GPU; numpyro
# runs the chains in vectorized/sequential mode on the device itself.
#
# Logs to results_v2_diffuse/v5_pipeline/<phase>.log and STATUS.tsv (same
# format as the CPU orchestrator).
#
# Usage (via systemd-run):
#   systemd-run --user --unit=brazil-qpm-v5-gpu-pipeline \
#     --working-directory=. \
#     bash run_v5_gpu_pipeline.sh
# -----------------------------------------------------------------------------

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPTS_DIR/../.." && pwd)"
PY="$REPO/.venv-gpu/bin/python3"

OUT_BASE="$SCRIPTS_DIR/results_v2_diffuse"
LOG_DIR="$OUT_BASE/v5_pipeline"
STATUS="$LOG_DIR/STATUS.tsv"
mkdir -p "$LOG_DIR"

# CUDA-flavored env: 'cuda' (not 'gpu' — the latter resolves to rocm first).
export JAX_PLATFORMS=cuda
export N_PARALLEL=1
# Free GPU memory between subprocesses
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ ! -x "$PY" ]]; then
    echo "ERROR: GPU venv not found at $PY" >&2
    exit 1
fi

run_phase() {
    local name="$1"
    shift
    local log="$LOG_DIR/${name}_gpu.log"
    printf "%s\tSTART_GPU\t%s\n" "$name" "$(date -Iseconds)" >> "$STATUS"
    {
        echo "============================================================"
        echo "  Phase: $name (GPU)   started $(date)"
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
        printf "%s\tOK_GPU\t%s\n" "$name" "$(date -Iseconds)" >> "$STATUS"
    else
        printf "%s\tFAIL_GPU(%d)\t%s\n" "$name" "$rc" "$(date -Iseconds)" >> "$STATUS"
    fi
    return $rc
}

cd "$SCRIPTS_DIR"

printf "GPU_PIPELINE\tSTART\t%s\n" "$(date -Iseconds)" >> "$STATUS"

# Order: RUB5 focus first (smaller, exposes any GPU teething issues fast),
# then RUB1-4 main (the long pole).
run_phase phase5_robustness_rub5 "$PY" run_brazil_rub5_focus_v2.py \
    --warmup 1000 --samples 1000 --chains 4

run_phase phase5_robustness_main "$PY" run_robustness_v2.py \
    --spec all --warmup 1000 --samples 1000 --chains 4

printf "GPU_PIPELINE\tDONE\t%s\n" "$(date -Iseconds)" >> "$STATUS"
echo "============================================================"
echo "  GPU pipeline complete; see $STATUS."
echo "============================================================"

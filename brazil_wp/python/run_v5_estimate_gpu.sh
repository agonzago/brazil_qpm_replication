#!/usr/bin/env bash
# =============================================================================
# Brazil WP v5 -- full Bayesian estimation (NUTS) on GPU for ALL 12 specs
# =============================================================================
#
# Sister script to run_v5_calibrate_only.sh.  Same env / backend setup, but
# this one runs the full pipeline (calibration + NUTS + post-estimation
# diagnostics) instead of stopping at calibration.
#
# Per-spec wall-clock on GPU (RTX 4070 SUPER, 12 GB):
#   ~5–15 min depending on model complexity (vmap'd 4-chain NUTS).
#   Total for 12 specs ≈ 1–3 hours wall.
#
# Usage:
#   systemd-run --user --unit=brazil-qpm-v5-estimate \
#     --working-directory=. \
#     bash run_v5_estimate_gpu.sh
# -----------------------------------------------------------------------------

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPTS_DIR/../.." && pwd)"

PY="$REPO/.venv-gpu/bin/python3"
export GPM_USE_GPU=1
export JAX_PLATFORMS=cuda           # NOT 'gpu' (resolves to rocm first)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUT_BASE="$SCRIPTS_DIR/results_v2_diffuse"
LOG_DIR="$OUT_BASE/v5_estimate"
STATUS="$LOG_DIR/STATUS.tsv"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: GPU venv not found at $PY" >&2
    exit 1
fi

# MCMC config (paper quality; 4 vectorized chains x 1000 warmup x 1000 samples)
WARMUP=1000
SAMPLES=1000
CHAINS=4

# -----------------------------------------------------------------------------
# Job definitions:  (label, command...)
# Single-GPU: run sequentially (MAX_PARALLEL=1 by design — see calibration
# script for the OOM rationale).
# -----------------------------------------------------------------------------
declare -a JOBS=(
    "A     run_brazil_wp_v2.py                  --use-diffuse --warmup $WARMUP --samples $SAMPLES --chains $CHAINS"
    "A0    run_brazil_model_a_const_rr_v2.py    --use-diffuse --warmup $WARMUP --samples $SAMPLES --chains $CHAINS"
    "B     run_brazil_model_b_v2.py             --use-diffuse --warmup $WARMUP --samples $SAMPLES --chains $CHAINS"
    "C     run_brazil_model_c_v2.py             --use-diffuse --warmup $WARMUP --samples $SAMPLES --chains $CHAINS"
    "D     run_brazil_model_d_v2.py             --use-diffuse --warmup $WARMUP --samples $SAMPLES --chains $CHAINS"
    "E     run_brazil_model_e_v2.py             --use-diffuse --warmup $WARMUP --samples $SAMPLES --chains $CHAINS"
    "F     run_brazil_model_f_v2.py             --use-diffuse --warmup $WARMUP --samples $SAMPLES --chains $CHAINS"
    "G     run_brazil_model_g_v2.py             --use-diffuse --warmup $WARMUP --samples $SAMPLES --chains $CHAINS"
    "RUB1-3-4  run_robustness_v2.py             --spec all --warmup $WARMUP --samples $SAMPLES --chains $CHAINS"
    "RUB5  run_brazil_rub5_focus_v2.py          --warmup $WARMUP --samples $SAMPLES --chains $CHAINS"
)

run_job() {
    local label="$1"; shift
    local log="$LOG_DIR/${label}.log"
    printf "%s\tSTART\t%s\n" "$label" "$(date -Iseconds)" >> "$STATUS"
    {
        echo "============================================================"
        echo "  Job: $label   started $(date)"
        echo "  Command: $PY $*"
        echo "============================================================"
    } > "$log"
    "$PY" "$@" >> "$log" 2>&1
    local rc=$?
    {
        echo "============================================================"
        echo "  Job $label finished $(date)  rc=$rc"
        echo "============================================================"
    } >> "$log"
    if [[ $rc -eq 0 ]]; then
        printf "%s\tOK\t%s\n" "$label" "$(date -Iseconds)" >> "$STATUS"
    else
        printf "%s\tFAIL(%d)\t%s\n" "$label" "$rc" "$(date -Iseconds)" >> "$STATUS"
    fi
}

cd "$SCRIPTS_DIR"
printf "ESTIMATE\tSTART\t%s\n" "$(date -Iseconds)" >> "$STATUS"
echo "Using backend=gpu, python=$PY, MCMC=4x${WARMUP}+${SAMPLES} chains, sequential"

t0=$(date +%s)

# Sequential execution — one model on the GPU at a time
for job in "${JOBS[@]}"; do
    read -r label rest <<< "$job"
    # shellcheck disable=SC2086
    run_job "$label" $rest
done

elapsed=$(( $(date +%s) - t0 ))
printf "ESTIMATE\tDONE\t%s\n" "$(date -Iseconds)" >> "$STATUS"
echo "============================================================"
echo "  v5 estimation complete in $((elapsed/60)) min."
echo "  Per-job logs:  $LOG_DIR/*.log"
echo "  Status file:   $STATUS"
echo "============================================================"

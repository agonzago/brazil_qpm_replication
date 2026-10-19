#!/usr/bin/env bash
# =============================================================================
# Brazil WP v5 -- calibration-only pass for ALL specs (post OBS_TT data fix)
# =============================================================================
#
# Runs every model variant (A, A0, B, C, D, E, F, G) and every robustness
# spec (RUB1/3/4 via run_robustness_v2.py + RUB5 via dedicated script) with
# `--calibrate-only` so we get filter diagnostics + decomposition at the
# SNR-derived calibrated params, WITHOUT triggering NUTS.
#
# Use this to inspect whether the SNR config (especially for SHK_TT_BAR /
# SHK_TT_GAP) is appropriate for the corrected commodity-flavored OBS_TT,
# before committing the long MCMC runs.
#
# Each calibration is ~3 min CPU.  Running 4 at a time → ~10 min wall.
#
# Usage (via systemd-run, like the earlier pipelines):
#   systemd-run --user --unit=brazil-qpm-v5-calib-only \
#     --working-directory=. \
#     bash run_v5_calibrate_only.sh
# -----------------------------------------------------------------------------

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPTS_DIR/../.." && pwd)"

# Backend selection: pass `cpu` as $1 to fall back to .venv (CPU JAX);
# default is GPU (.venv-gpu, jax[cuda12]).  BK stability check auto-disables
# on GPU per CLAUDE.md — fine for calibration since EM starts from prior
# modes and unstable draws won't be hit.
BACKEND="${1:-gpu}"
if [[ "$BACKEND" == "gpu" ]]; then
    PY="$REPO/.venv-gpu/bin/python3"
    # GPM_USE_GPU=1 makes the runners skip the XLA virtual-CPU-device setup
    # and switch chain_method to 'vectorized' (single-GPU vmap) for any later
    # NUTS step.  For --calibrate-only this just keeps the runtime clean.
    export GPM_USE_GPU=1
    export JAX_PLATFORMS=cuda      # NOT 'gpu' (resolves to rocm first)
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    MAX_PARALLEL=1                 # 1 model at a time on a single 12 GB GPU.
                                   # Each model JIT'd + 4 vmap chains can hit
                                   # 5 GB during NUTS warmup; running two in
                                   # parallel risks OOM.
else
    PY="$REPO/.venv/bin/python3"
    export JAX_PLATFORMS=cpu
    MAX_PARALLEL=4
fi

OUT_BASE="$SCRIPTS_DIR/results_v2_diffuse"
LOG_DIR="$OUT_BASE/v5_calibrate_only"
STATUS="$LOG_DIR/STATUS.tsv"
mkdir -p "$LOG_DIR"

export N_PARALLEL=1       # calibration is single-threaded JAX; no virtual chains needed
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -x "$PY" ]]; then
    echo "ERROR: Python venv not found at $PY" >&2
    exit 1
fi
echo "Using backend=$BACKEND, python=$PY, MAX_PARALLEL=$MAX_PARALLEL"

# -----------------------------------------------------------------------------
# Job definitions:  (label, command...)
# -----------------------------------------------------------------------------
declare -a JOBS=(
    "A     run_brazil_wp_v2.py                  --use-diffuse --calibrate-only"
    "A0    run_brazil_model_a_const_rr_v2.py    --use-diffuse --calibrate-only"
    "B     run_brazil_model_b_v2.py             --use-diffuse --calibrate-only"
    "C     run_brazil_model_c_v2.py             --use-diffuse --calibrate-only"
    "D     run_brazil_model_d_v2.py             --use-diffuse --calibrate-only"
    "E     run_brazil_model_e_v2.py             --use-diffuse --calibrate-only"
    "F     run_brazil_model_f_v2.py             --use-diffuse --calibrate-only"
    "G     run_brazil_model_g_v2.py             --use-diffuse --calibrate-only"
    "RUB1-3-4  run_robustness_v2.py             --spec all --calibrate-only"
    "RUB5  run_brazil_rub5_focus_v2.py          --calibrate-only"
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
printf "CALIB_ONLY\tSTART\t%s\n" "$(date -Iseconds)" >> "$STATUS"

t0=$(date +%s)

# Bounded concurrency without GNU parallel: while-loop with job count
RUNNING=0
PIDS=()
for job in "${JOBS[@]}"; do
    # tokenize: first field = label, rest = command tokens
    read -r label rest <<< "$job"
    # shellcheck disable=SC2086
    run_job "$label" $rest &
    PIDS+=($!)
    RUNNING=$((RUNNING+1))
    if [[ $RUNNING -ge $MAX_PARALLEL ]]; then
        wait -n
        RUNNING=$((RUNNING-1))
    fi
done

# Wait for the last batch
wait

elapsed=$(( $(date +%s) - t0 ))
printf "CALIB_ONLY\tDONE\t%s\n" "$(date -Iseconds)" >> "$STATUS"
echo "============================================================"
echo "  v5 calibration-only complete in $((elapsed/60)) min."
echo "  Per-job logs:  $LOG_DIR/*.log"
echo "  Status file:   $STATUS"
echo "  Diagnostic plots land in:"
echo "    results_v2_diffuse/model_<x>/diagnostic/diag_*.png"
echo "    results_v2_diffuse/model_<x>/decomposition/decomp_*.png"
echo "============================================================"

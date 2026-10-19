#!/usr/bin/env bash
# =============================================================================
# Brazil WP v5 -- CPU estimation, picks up where the GPU run was abandoned
# =============================================================================
#
# After the GPU per-model wall-clock turned out to be ~91 min vs ~30 min on
# CPU (the Brazilian QPM is too small to benefit from vmap'd 4-chain NUTS
# on a single 12 GB consumer GPU), we switched back to CPU.  Model A's
# draws were preserved from the GPU run; this script estimates the
# remaining 11 specs on CPU, two at a time.
#
# Per-spec wall-clock on this 28-core box: ~30-60 min × 11/2 batches ≈
# 3-4 hours total.
#
# Usage (via systemd-run):
#   systemd-run --user --unit=brazil-qpm-v5-estimate \
#     --working-directory=. \
#     bash run_v5_estimate_cpu.sh
# -----------------------------------------------------------------------------

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPTS_DIR/../.." && pwd)"
PY="$REPO/.venv/bin/python3"

OUT_BASE="$SCRIPTS_DIR/results_v2_diffuse"
LOG_DIR="$OUT_BASE/v5_estimate"
STATUS="$LOG_DIR/STATUS.tsv"
mkdir -p "$LOG_DIR"

# CPU env: 4 XLA virtual chain devices, BLAS pinned to 1
export JAX_PLATFORMS=cpu
export N_PARALLEL=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WARMUP=1000
SAMPLES=1000
CHAINS=4
MAX_PARALLEL=2     # 2 model subprocesses x 4 chains = 8 chains in flight

# -----------------------------------------------------------------------------
# Job definitions (Model A is SKIPPED — already done from the GPU run).
# -----------------------------------------------------------------------------
declare -a JOBS=(
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
    printf "%s\tSTART_CPU\t%s\n" "$label" "$(date -Iseconds)" >> "$STATUS"
    {
        echo "============================================================"
        echo "  Job: $label (CPU)   started $(date)"
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
        printf "%s\tOK_CPU\t%s\n" "$label" "$(date -Iseconds)" >> "$STATUS"
    else
        printf "%s\tFAIL_CPU(%d)\t%s\n" "$label" "$rc" "$(date -Iseconds)" >> "$STATUS"
    fi
}

cd "$SCRIPTS_DIR"

# Pre-create the arviz daily-warning stamp file BEFORE parallel imports.
# Multiple Python processes importing arviz simultaneously race in
# arviz.__init__._warn_once_per_day(); the loser sees a FileNotFoundError
# on tmp.replace().  See run_all_models_v2.py:_ensure_arviz_cache.
mkdir -p "$HOME/.cache/arviz"
[[ -f "$HOME/.cache/arviz/daily_warning" ]] || date -I > "$HOME/.cache/arviz/daily_warning"

printf "ESTIMATE_CPU\tSTART\t%s\n" "$(date -Iseconds)" >> "$STATUS"
echo "Using backend=cpu, python=$PY, MAX_PARALLEL=$MAX_PARALLEL, MCMC=4x${WARMUP}+${SAMPLES}"

t0=$(date +%s)

# Bounded concurrency
RUNNING=0
for job in "${JOBS[@]}"; do
    read -r label rest <<< "$job"
    # shellcheck disable=SC2086
    run_job "$label" $rest &
    RUNNING=$((RUNNING+1))
    if [[ $RUNNING -ge $MAX_PARALLEL ]]; then
        wait -n
        RUNNING=$((RUNNING-1))
    fi
done
wait

elapsed=$(( $(date +%s) - t0 ))
printf "ESTIMATE_CPU\tDONE\t%s\n" "$(date -Iseconds)" >> "$STATUS"
echo "============================================================"
echo "  v5 CPU estimation complete in $((elapsed/60)) min."
echo "============================================================"

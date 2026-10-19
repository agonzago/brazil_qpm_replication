#!/usr/bin/env bash
# =============================================================================
# Re-run all 7 models in --skip-estimation mode to regenerate post-MCMC
# artifacts (diagnostic plots, decompositions, IRFs, forecasts, evaluations,
# policy analysis) using the PATCHED toolbox where every smoother is the
# diffuse Kalman smoother by default.
#
# All 7 models already have valid posterior draws on disk from prior full
# runs that used the diffuse likelihood (--use-diffuse).  This script reuses
# those draws (~7 × 1 min of MCMC saved per model) and only redoes the
# post-processing.  Each model takes ~15-30 min in this mode, dominated by
# the rolling-window forecast evaluation.
# =============================================================================

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/.venv/bin/python3"
SCRIPTS_DIR="$REPO/brazil_wp/python"
OUT_BASE="$SCRIPTS_DIR/results_v2_diffuse"
LOG_OVERVIEW="$OUT_BASE/rerun_postprocess.log"

export JAX_PLATFORMS=cpu
export N_PARALLEL=8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# (label, runner, results-subdir-for-log)
MODELS=(
    "A:run_brazil_wp_v2.py:model_a"
    "A0:run_brazil_model_a_const_rr_v2.py:model_a0"
    "B:run_brazil_model_b_v2.py:model_b"
    "C:run_brazil_model_c_v2.py:model_c"
    "D:run_brazil_model_d_v2.py:model_d"
    "E:run_brazil_model_e_v2.py:model_e"
    "F:run_brazil_model_f_v2.py:model_f"
)

mkdir -p "$OUT_BASE"
{
    echo "============================================================"
    echo "  Brazil WP -- post-processing rerun (consistent diffuse)"
    echo "============================================================"
    echo "  Started: $(date)"
    echo "  Mode:    --skip-estimation (reuses draws.npz)"
    echo "============================================================"
} | tee "$LOG_OVERVIEW"

cd "$SCRIPTS_DIR"
t_start=$(date +%s)

for entry in "${MODELS[@]}"; do
    IFS=':' read -r label runner subdir <<< "$entry"
    echo "" | tee -a "$LOG_OVERVIEW"
    echo "--- Model $label ($runner) ---" | tee -a "$LOG_OVERVIEW"
    t0=$(date +%s)
    if "$PY" -u "$runner" --skip-estimation --use-diffuse \
            2>&1 | tee "$OUT_BASE/$subdir/run.log" >/dev/null; then
        rc=0
    else
        rc=${PIPESTATUS[0]}
    fi
    elapsed=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ]; then
        echo "  Model $label: OK ($((elapsed/60)) min)" | tee -a "$LOG_OVERVIEW"
    else
        echo "  Model $label: FAILED (exit $rc, $((elapsed/60)) min)" \
            | tee -a "$LOG_OVERVIEW"
    fi
done

total=$(( $(date +%s) - t_start ))
{
    echo ""
    echo "============================================================"
    echo "  Total wall-clock: $((total/60)) min"
    echo "  Finished:         $(date)"
    echo "============================================================"
} | tee -a "$LOG_OVERVIEW"

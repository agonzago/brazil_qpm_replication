#!/usr/bin/env bash
# =============================================================================
# Brazil WP -- full re-estimation of all paper models with the new calibration
# =============================================================================
#
# New calibration (paper-rev-calibration branch) is now the default in
# QPMModel.calibrate():
#   * use_diffuse=True            (exact diffuse Kalman smoother for the EM)
#   * pin_trends_from_snr=True    (trend sigmas anchored at SNR modes;
#                                  EM updates only cycle shocks)
#
# The runners pick this up automatically via QPMModel.calibrate().  The
# --use-diffuse flag still gates the NUTS likelihood and routes output to
# results_v2_diffuse/.
#
# -----------------------------------------------------------------------------
# Mac tuning (10-core Apple Silicon, 16 GB RAM):
#   * One model at a time (--max-workers 1).  16 GB does not comfortably fit
#     two QPM models with 8 chains each running in parallel, and XLA already
#     uses all 8 virtual chain devices per model.
#   * 8 chains per model via XLA virtual devices.  Native BLAS threading
#     disabled (OMP/MKL/OPENBLAS=1) to avoid contending with XLA.
#   * MCMC settings match the paper: 400 warmup, 500 samples, 8 chains.
#
# Expected wall-clock per model: ~30-60 min.  Total for all 6 models: ~3-6 h.
# -----------------------------------------------------------------------------
#
# Output:
#   results_v2_diffuse/model_<x>/...   (one directory per model variant)
#   results_v2_diffuse/run_estimation.log   (overview + timings)
# -----------------------------------------------------------------------------

set -euo pipefail

# Auto-detect repo root: this script lives in brazil_wp/python/, so the
# repo is two levels up.  Resolves symlinks via cd && pwd.
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPTS_DIR/../.." && pwd)"
PY="$REPO/.venv/bin/python3"
OUT_BASE="$SCRIPTS_DIR/results_v2_diffuse"
LOG_OVERVIEW="$OUT_BASE/run_estimation.log"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: Python venv not found at $PY"
    echo "Create it first with:"
    echo "  cd $REPO && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# -----------------------------------------------------------------------------
# Environment: thread/device budget for an Apple Silicon Mac
# -----------------------------------------------------------------------------
export JAX_PLATFORMS=cpu
export N_PARALLEL=8                  # XLA virtual chain devices per model
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------------------------------------------------------
# MCMC settings (paper quality)
# -----------------------------------------------------------------------------
WARMUP=400
SAMPLES=500
CHAINS=8

# -----------------------------------------------------------------------------
# Optional quick-test mode: pass `quick` as the first arg to use a tiny
# posterior for smoke testing (~5-10 min per model).
# -----------------------------------------------------------------------------
if [[ "${1:-}" == "quick" ]]; then
    WARMUP=50
    SAMPLES=100
    CHAINS=4
    echo ">>> QUICK MODE: warmup=$WARMUP samples=$SAMPLES chains=$CHAINS <<<"
fi

mkdir -p "$OUT_BASE"

{
    echo "============================================================"
    echo "  Brazil WP -- full re-estimation (new calibration)"
    echo "============================================================"
    echo "  Started:    $(date)"
    echo "  Host:       $(hostname)"
    echo "  CPU cores:  $(sysctl -n hw.physicalcpu) physical / $(sysctl -n hw.logicalcpu) logical"
    echo "  Memory:     $(echo "scale=1; $(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc) GB"
    echo "  Python:     $PY"
    echo "  Output:     $OUT_BASE"
    echo "  MCMC:       warmup=$WARMUP, samples=$SAMPLES, chains=$CHAINS"
    echo "============================================================"
} | tee "$LOG_OVERVIEW"

cd "$SCRIPTS_DIR"

t_start=$(date +%s)

# -----------------------------------------------------------------------------
# Run all paper models sequentially via the batch orchestrator.
# Models: A (MR baseline), A0 (constant R*), B (no QRR Koyck),
#         C (no QRR/QZ Koyck), D (simplified Taylor), E (D + COVID masked),
#         F (A with Taylor coefficients fixed).
# -----------------------------------------------------------------------------
{
    echo
    echo "--- Running models A, A0, B, C, D, E, F (run_all_models_v2.py) ---"
} | tee -a "$LOG_OVERVIEW"

"$PY" run_all_models_v2.py \
    --use-diffuse \
    --max-workers 1 \
    --models A A0 B C D E F \
    --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS" \
    2>&1 | tee -a "$LOG_OVERVIEW"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
total=$(( $(date +%s) - t_start ))
{
    echo
    echo "============================================================"
    echo "  Total wall-clock: $((total/60)) min"
    echo "  Finished:         $(date)"
    echo "============================================================"
} | tee -a "$LOG_OVERVIEW"

#!/usr/bin/env bash
# =============================================================================
# Brazil WP -- v5 paper-quality re-estimation (Linux, batched 2-at-a-time)
# =============================================================================
#
# v5 MCMC config (matches the paper's stated 4 x (1000 warmup x 1000 samples)):
#   * --warmup 1000  --samples 1000  --chains 4
#
# Concurrency:
#   * --max-workers 2: at most two model subprocesses run at the same time.
#     Each model launches 4 MCMC chains via XLA virtual devices, so peak
#     in-flight chains = 2 x 4 = 8.  BLAS threads pinned to 1 per chain.
#   * Peak RAM: ~12-20 GB (well within 62 GB available on this host).
#   * 28 cores / 8 chains => generous headroom; avoids the resource pressure
#     that crashed VS Code when running all 7 models in parallel.
#
# Wall-clock estimate: ~3-4 hours (7 models in batches of 2; ~30-60 min/model).
#
# Output:
#   results_v2_diffuse/model_<x>/...
#   results_v2_diffuse/run_estimation_v5.log
#
# Usage:
#   Foreground (blocks the terminal):
#     bash run_estimation_v5_linux.sh
#   Detached via systemd (survives VS Code/SSH disconnects, lingering enabled):
#     systemd-run --user --unit=brazil-qpm-v5 --working-directory=$PWD \
#       bash run_estimation_v5_linux.sh
#   Inspect:
#     systemctl --user status brazil-qpm-v5
#     journalctl --user -u brazil-qpm-v5 -f
#   Stop:
#     systemctl --user stop brazil-qpm-v5
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPTS_DIR/../.." && pwd)"
PY="$REPO/.venv/bin/python3"
OUT_BASE="$SCRIPTS_DIR/results_v2_diffuse"
LOG_OVERVIEW="$OUT_BASE/run_estimation_v5.log"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: Python venv not found at $PY" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Environment: thread / device budget for 28-core Linux
# -----------------------------------------------------------------------------
export JAX_PLATFORMS=cpu
export N_PARALLEL=4                  # XLA virtual chain devices per model = --chains
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------------------------------------------------------
# MCMC settings (paper-quality v5)
# -----------------------------------------------------------------------------
WARMUP=1000
SAMPLES=1000
CHAINS=4
MAX_WORKERS=2

# Optional smoke override: pass `quick` as $1 for a fast end-to-end smoke test.
if [[ "${1:-}" == "quick" ]]; then
    WARMUP=50
    SAMPLES=100
    CHAINS=4
    echo ">>> QUICK MODE: warmup=$WARMUP samples=$SAMPLES chains=$CHAINS <<<"
fi

mkdir -p "$OUT_BASE"

{
    echo "============================================================"
    echo "  Brazil WP -- v5 re-estimation (paper-quality, batched)"
    echo "============================================================"
    echo "  Started:     $(date)"
    echo "  Host:        $(hostname)"
    echo "  CPU cores:   $(nproc)"
    echo "  Memory:      $(free -h | awk '/^Mem:/ {print $2}')"
    echo "  Python:      $PY"
    echo "  Output:      $OUT_BASE"
    echo "  MCMC:        warmup=$WARMUP, samples=$SAMPLES, chains=$CHAINS"
    echo "  Workers:     $MAX_WORKERS  (concurrent model subprocesses)"
    echo "============================================================"
} | tee "$LOG_OVERVIEW"

cd "$SCRIPTS_DIR"

t_start=$(date +%s)

"$PY" run_all_models_v2.py \
    --use-diffuse \
    --max-workers "$MAX_WORKERS" \
    --models A A0 B C D E F \
    --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS" \
    2>&1 | tee -a "$LOG_OVERVIEW"

total=$(( $(date +%s) - t_start ))
{
    echo
    echo "============================================================"
    echo "  Total wall-clock: $((total/60)) min"
    echo "  Finished:         $(date)"
    echo "============================================================"
} | tee -a "$LOG_OVERVIEW"

#!/usr/bin/env bash
# Build smoothed_states.npz under results_v2_diffuse_v5/ for every v5
# specification (5 body models + 4 RUBs).  Each .npz stores smoothed states
# at posterior MEAN and MEDIAN — see _save_smoothed_states.py for the
# array key schema.

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
V5="$SCRIPTS_DIR/results_v2_diffuse_v5"
PY="${PYTHON:-python3}"

export JAX_PLATFORMS=cpu
export N_PARALLEL=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# v5 baseline pins (same env vars used during estimation; the runner reads
# these at import time to pick MOD_FILE, FIXED_PARAMS, MEAS_NOISE_STD).
export V5_TT_PIN=tight
export V5_PREM_PIN=tight
export V5_PREM_SNR=15
export V5_TT_GAP_CAP=3.0
export V5_TT_MODE=tworegime
export V5_TT_BAR_PRE_STD=0.10
export V5_TT_BAR_POST_STD=0.50

# (label-in-MODEL_RUNNER  results-dir-name)
SPECS=(
    "A     model_a"
    "A0    model_a0"
    "B     model_b"
    "C     model_c"
    "D     model_d"
    "E     model_e"
    "RUB1  rub1"
    "RUB3  rub3"
    "RUB4  rub4"
    "RUB5  rub5"
)

mkdir -p "$V5/_logs"
LOG="$V5/_logs/save_all_smoothed_states.log"
: > "$LOG"

t0=$(date +%s)
n_ok=0; n_fail=0; n_skip=0
for row in "${SPECS[@]}"; do
    read -r label dirname <<< "$row"
    outdir="$V5/$dirname"
    if [[ ! -d "$outdir" ]]; then
        echo "SKIP $label  ($dirname missing)" | tee -a "$LOG"
        n_skip=$((n_skip+1))
        continue
    fi
    if [[ -f "$outdir/diagnostic/smoothed_states.npz" ]]; then
        echo "SKIP $label  (already saved)" | tee -a "$LOG"
        n_skip=$((n_skip+1))
        continue
    fi
    echo "[$label]  $dirname  ..." | tee -a "$LOG"
    "$PY" "$SCRIPTS_DIR/_save_smoothed_states.py" "$label" "$outdir" \
        >> "$LOG" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "  OK" | tee -a "$LOG"
        n_ok=$((n_ok+1))
    else
        echo "  FAIL rc=$rc  (see $LOG)" | tee -a "$LOG"
        n_fail=$((n_fail+1))
    fi
done

elapsed=$(( $(date +%s) - t0 ))
echo
echo "Done in $((elapsed/60))m $((elapsed%60))s: $n_ok OK, $n_fail FAIL, $n_skip SKIP" | tee -a "$LOG"

#!/usr/bin/env bash
# Loop over all 24 v5 sweep configs and save smoothed states (at posterior
# MEAN and MEDIAN params) to each config's diagnostic/smoothed_states.npz.

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
SMOKE="$SCRIPTS_DIR/results_smoke"
PY="${PYTHON:-python3}"

# Common env (overridden per-config below)
export JAX_PLATFORMS=cpu
export N_PARALLEL=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

declare -A DIRNAME=([A]="model_a" [B]="model_b" [C]="model_c" [D]="model_d" [E]="model_e" [F]="model_f")

# (cfg, V5_TT_PIN, V5_PREM_PIN, V5_PREM_SNR, V5_TT_GAP_CAP)
declare -a CONFIGS=(
    "TLPL     loose loose 30 3.0"
    "TLPT     loose tight 30 3.0"
    "TTPL     tight loose 30 3.0"
    "TTPT     tight tight 30 3.0"
    "TTPT_p15 tight tight 15 3.0"
    "TTPT_c5  tight tight 30 5.0"
)

t0=$(date +%s)
n_ok=0; n_fail=0; n_skip=0
for cfg_row in "${CONFIGS[@]}"; do
    read -r cfg tt_pin prem_pin prem_snr tt_cap <<< "$cfg_row"
    for model in A B C D E F; do
        outdir="$SMOKE/${DIRNAME[$model]}_${cfg}"
        if [[ ! -d "$outdir" ]]; then
            echo "SKIP ${model}_${cfg} (no dir)"
            n_skip=$((n_skip+1))
            continue
        fi
        if [[ -f "$outdir/diagnostic/smoothed_states.npz" ]]; then
            echo "SKIP ${model}_${cfg} (already saved)"
            n_skip=$((n_skip+1))
            continue
        fi
        echo "[${model}_${cfg}]  ..."
        V5_TT_PIN="$tt_pin" V5_PREM_PIN="$prem_pin" \
        V5_PREM_SNR="$prem_snr" V5_TT_GAP_CAP="$tt_cap" \
            "$PY" "$SCRIPTS_DIR/_save_smoothed_states.py" "$model" "$outdir" \
            >> "$SMOKE/v5_sweep/save_smoothed_states.log" 2>&1
        rc=$?
        if [[ $rc -eq 0 ]]; then
            echo "  OK"
            n_ok=$((n_ok+1))
        else
            echo "  FAIL rc=$rc (see save_smoothed_states.log)"
            n_fail=$((n_fail+1))
        fi
    done
done
elapsed=$(( $(date +%s) - t0 ))
echo
echo "Done in $((elapsed/60))m: $n_ok OK, $n_fail FAIL, $n_skip SKIP"

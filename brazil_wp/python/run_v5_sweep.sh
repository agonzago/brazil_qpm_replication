#!/usr/bin/env bash
# =============================================================================
# v5 sweep -- A/C/E/F × 6 configs (TT prior × PREM prior + refinements)
# =============================================================================
#
# Configs:
#   TLPL     : TT loose, PREM loose
#   TLPT     : TT loose, PREM tight α=20
#   TTPL     : TT tight (cap=3 SNR=50 α=20), PREM loose
#   TTPT     : TT tight, PREM tight α=20         ← Model C reference equiv.
#   TTPT_p15 : TT tight, PREM tight α=20, PREM_SNR=15  ← refinement
#   TTPT_c5  : TT tight cap=5, PREM tight α=20         ← refinement
#
# PREM_SNR=30 for all except TTPT_p15 (which uses 15).
# TT_GAP_CAP=3.0 for all tight configs except TTPT_c5 (cap=5).
# 4 models × 6 configs = 24 runs.  MAX_PARALLEL=2 ⇒ ~12 batches ⇒ ~7 h wall.
#
# Idempotent: skips configs already marked OK in STATUS.tsv.
# -----------------------------------------------------------------------------

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPTS_DIR/../.." && pwd)"
PY="$REPO/.venv/bin/python3"

SMOKE_ROOT="$SCRIPTS_DIR/results_smoke"
LOG_DIR="$SMOKE_ROOT/v5_sweep"
STATUS="$LOG_DIR/STATUS.tsv"
mkdir -p "$LOG_DIR"

# Pre-create the arviz daily-warning stamp to avoid the parallel-import race.
mkdir -p "$HOME/.cache/arviz"
[[ -f "$HOME/.cache/arviz/daily_warning" ]] || date -I > "$HOME/.cache/arviz/daily_warning"

# Common env
export JAX_PLATFORMS=cpu
export N_PARALLEL=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WARMUP=1000
SAMPLES=1000
CHAINS=4
MAX_PARALLEL=2

declare -A SCRIPT
SCRIPT[A]="run_brazil_wp_v2.py"
SCRIPT[B]="run_brazil_model_b_v2.py"
SCRIPT[C]="run_brazil_model_c_v2.py"
SCRIPT[D]="run_brazil_model_d_v2.py"
SCRIPT[E]="run_brazil_model_e_v2.py"
SCRIPT[F]="run_brazil_model_f_v2.py"

declare -A DIRNAME
DIRNAME[A]="model_a"
DIRNAME[B]="model_b"
DIRNAME[C]="model_c"
DIRNAME[D]="model_d"
DIRNAME[E]="model_e"
DIRNAME[F]="model_f"

# (cfg_id, V5_TT_PIN, V5_PREM_PIN, V5_PREM_SNR, V5_TT_GAP_CAP)
declare -a CONFIGS=(
    "TLPL     loose loose 30 3.0"
    "TLPT     loose tight 30 3.0"
    "TTPL     tight loose 30 3.0"
    "TTPT     tight tight 30 3.0"
    "TTPT_p15 tight tight 15 3.0"
    "TTPT_c5  tight tight 30 5.0"
)

already_ok() {
    local model="$1" cfg="$2"
    grep -q "^${model}_${cfg}	OK	" "$STATUS" 2>/dev/null
}

run_one() {
    local model="$1" cfg="$2" tt_pin="$3" prem_pin="$4" prem_snr="$5" tt_cap="$6"
    local script="${SCRIPT[$model]}"
    local outdir="$SMOKE_ROOT/${DIRNAME[$model]}_${cfg}"
    local job="${model}_${cfg}"
    mkdir -p "$outdir" "$outdir/post_estimation"

    printf "%s\tSTART\t%s\tTT=%s PREM=%s PREM_SNR=%s CAP=%s\n" \
        "$job" "$(date -Iseconds)" "$tt_pin" "$prem_pin" "$prem_snr" "$tt_cap" >> "$STATUS"

    # ----- Step 1: full pipeline estimation -----
    V5_TT_PIN="$tt_pin" V5_PREM_PIN="$prem_pin" \
    V5_PREM_SNR="$prem_snr" V5_TT_GAP_CAP="$tt_cap" \
        "$PY" "$SCRIPTS_DIR/$script" \
            --use-diffuse --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS" \
            --results-dir "$outdir" \
            > "$outdir/run.log" 2>&1
    local rc=$?

    if [[ $rc -ne 0 ]]; then
        printf "%s\tFAIL(%d)\t%s\n" "$job" "$rc" "$(date -Iseconds)" >> "$STATUS"
        return
    fi

    # ----- Step 2: post-estimation plots (5 generate_* scripts) -----
    # Each receives the SAME env vars so it imports the runner with the
    # same TREND_SNR_MAP / FIXED_POINT_CONFIG used during estimation.
    for pscript in \
        generate_posterior_diagnostics.py \
        generate_posterior_filtration.py \
        generate_additional_plots.py \
        generate_prior_vs_posterior_irfs.py \
        compute_spectral_radius.py; do
        plog="$outdir/post_estimation/${pscript%.py}.log"
        V5_TT_PIN="$tt_pin" V5_PREM_PIN="$prem_pin" \
        V5_PREM_SNR="$prem_snr" V5_TT_GAP_CAP="$tt_cap" \
            "$PY" "$SCRIPTS_DIR/$pscript" \
                --model "$model" \
                --results-dir "$outdir" \
                > "$plog" 2>&1 || \
            echo "  WARN: $pscript failed for $job (see $plog)" >&2
    done

    printf "%s\tOK\t%s\n" "$job" "$(date -Iseconds)" >> "$STATUS"
}

# Build pending list
declare -a PENDING=()
for cfg_row in "${CONFIGS[@]}"; do
    read -r cfg tt_pin prem_pin prem_snr tt_cap <<< "$cfg_row"
    for model in A B C D E F; do
        if already_ok "$model" "$cfg"; then
            echo "SKIP ${model}_${cfg} (already OK)"
            continue
        fi
        PENDING+=("$model|$cfg|$tt_pin|$prem_pin|$prem_snr|$tt_cap")
    done
done

if [[ ${#PENDING[@]} -eq 0 ]]; then
    echo "Nothing to do."
    printf "SWEEP\tDONE\t%s\n" "$(date -Iseconds)" >> "$STATUS"
    exit 0
fi

printf "SWEEP\tSTART\t%s\t%d configs pending\n" "$(date -Iseconds)" "${#PENDING[@]}" >> "$STATUS"
echo "Pending: ${#PENDING[@]}"

cd "$SCRIPTS_DIR"
t0=$(date +%s)
RUNNING=0
for entry in "${PENDING[@]}"; do
    IFS='|' read -r model cfg tt_pin prem_pin prem_snr tt_cap <<< "$entry"
    echo "Launching ${model}_${cfg}  TT=$tt_pin PREM=$prem_pin PREM_SNR=$prem_snr CAP=$tt_cap"
    run_one "$model" "$cfg" "$tt_pin" "$prem_pin" "$prem_snr" "$tt_cap" &
    RUNNING=$((RUNNING+1))
    if [[ $RUNNING -ge $MAX_PARALLEL ]]; then
        wait -n
        RUNNING=$((RUNNING-1))
    fi
done
wait

elapsed=$(( $(date +%s) - t0 ))
printf "SWEEP\tDONE\t%s\t%d min\n" "$(date -Iseconds)" "$((elapsed/60))" >> "$STATUS"
echo "Sweep complete in $((elapsed/60)) min."

echo "Launching evaluator..."
"$PY" "$SCRIPTS_DIR/evaluate_v5_sweep.py" > "$LOG_DIR/evaluate.log" 2>&1
echo "Evaluator rc=$?  See $SMOKE_ROOT/SWEEP_REPORT.md"

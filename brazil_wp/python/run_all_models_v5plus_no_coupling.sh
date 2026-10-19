#!/usr/bin/env bash
# v5+ no-coupling variant: paper-quality MCMC for the 9 remaining specs.
#
# Same wiring as run_all_models_v5plus.sh, but with ALT_KAPPA_G_TT=0
# forcing the TT-trend -> G_TREND coupling off (kappa_g_tt pinned to 0).
# Model A no_coupling already exists at results_v2_diffuse_v5/model_a_v5plus_no_coupling/.
#
# Usage:
#   bash run_all_models_v5plus_no_coupling.sh smoke  # 500/500/4
#   bash run_all_models_v5plus_no_coupling.sh paper  # 1000/1000/4

set -uo pipefail
cd "$(dirname "$0")"

MODE="${1:-paper}"
case "$MODE" in
    smoke)  WARMUP=500;  SAMPLES=500  ;;
    paper)  WARMUP=1000; SAMPLES=1000 ;;
    *)      echo "USAGE: $0 [smoke|paper]" >&2; exit 2 ;;
esac

CHAINS=4
PY="${PYTHON:-python3}"
RES_BASE=results_v2_diffuse_v5
mkdir -p "$RES_BASE"
export N_PARALLEL=4

SPECS_MODELS=(
    "model_a0     run_brazil_model_a_const_rr_v2.py"
    "model_b      run_brazil_model_b_v2.py"
    "model_c      run_brazil_model_c_v2.py"
    "model_d      run_brazil_model_d_v2.py"
    "model_e      run_brazil_model_e_v2.py"
)
SPECS_RUB=( "rub1" "rub3" "rub4" )
SPECS_FOCUS=( "rub5    run_brazil_rub5_focus_v2.py" )

echo "================================================================"
echo "  v5+ no-coupling (k=0) ${MODE^^} MCMC driver"
echo "  warmup=$WARMUP samples=$SAMPLES chains=$CHAINS"
echo "  ALT_KAPPA_G_TT=0 (TT-trend channel on G off)"
echo "================================================================"

PIDS=(); LAUNCHED=()

launch_model() {
    local name="$1" runner="$2"
    local res="${RES_BASE}/${name}_v5plus_no_coupling"
    local log="${RES_BASE}/${name}_v5plus_no_coupling.log"
    mkdir -p "$res"
    echo "  -> launching $name ($runner) -> $res"
    JAX_PLATFORMS=cpu V5_TT_MODE=tt_trend V5_TT_PIN=tight ALT_KAPPA_G_TT=0 \
        "$PY" "$runner" --use-diffuse \
        --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS" \
        --results-dir "$res" > "$log" 2>&1 &
    PIDS+=("$!"); LAUNCHED+=("$name")
}

launch_rub() {
    local name="$1"
    local res="${RES_BASE}/${name}_v5plus_no_coupling"
    local log="${RES_BASE}/${name}_v5plus_no_coupling.log"
    mkdir -p "$res"
    echo "  -> launching $name (run_robustness_v2.py) -> $res"
    JAX_PLATFORMS=cpu V5_TT_MODE=tt_trend V5_TT_PIN=tight ALT_KAPPA_G_TT=0 \
        V5PLUS_RESULTS_OVERRIDE="$res" \
        "$PY" run_robustness_v2.py --spec "$name" \
        --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS" \
        > "$log" 2>&1 &
    PIDS+=("$!"); LAUNCHED+=("$name")
}

launch_focus() {
    local name="$1" runner="$2"
    local res="${RES_BASE}/${name}_v5plus_no_coupling"
    local log="${RES_BASE}/${name}_v5plus_no_coupling.log"
    mkdir -p "$res"
    echo "  -> launching $name ($runner) -> $res"
    JAX_PLATFORMS=cpu V5_TT_MODE=tt_trend V5_TT_PIN=tight ALT_KAPPA_G_TT=0 \
        V5PLUS_RESULTS_OVERRIDE="$res" \
        "$PY" "$runner" \
        --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS" \
        > "$log" 2>&1 &
    PIDS+=("$!"); LAUNCHED+=("$name")
}

for entry in "${SPECS_MODELS[@]}"; do
    read -r name runner <<< "$entry"; launch_model "$name" "$runner"
done
for spec in "${SPECS_RUB[@]}"; do launch_rub "$spec"; done
for entry in "${SPECS_FOCUS[@]}"; do
    read -r name runner <<< "$entry"; launch_focus "$name" "$runner"
done

echo
echo "  ${#PIDS[@]} subprocesses running (PIDs: ${PIDS[*]})"
echo

EXIT_STATUS=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}; name=${LAUNCHED[$i]}
    if wait "$pid"; then echo "  [OK] $name"; else echo "  [FAIL] $name"; EXIT_STATUS=1; fi
done

echo
echo "  All ${#PIDS[@]} no_coupling specs done. Exit: $EXIT_STATUS"
exit "$EXIT_STATUS"

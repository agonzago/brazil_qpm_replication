#!/usr/bin/env bash
# v5+ smoke / paper-quality MCMC driver for all 10 specs.
#
# Usage:
#   bash run_all_models_v5plus.sh smoke   # warmup 500 / samples 500 / chains 4
#   bash run_all_models_v5plus.sh paper   # warmup 1000 / samples 1000 / chains 4
#
# Spawns up to 10 parallel subprocesses (one per spec).  Linux 28-core box:
# wall-clock ~25 min per spec in parallel for smoke, ~3 hours for paper.
#
# Output:
#   results_v2_diffuse_v5/<spec>_v5plus/  for each spec.
#
# Env defaults baked-in:
#   V5_TT_MODE=tt_trend  (selects the *_tt_trend_v2.qpm mod file per runner)
#   V5_TT_PIN=tight      (applies α=20 post-cal pin on TT_BAR / TT_GAP / D_TT_BAR)
#   JAX_PLATFORMS=cpu

set -uo pipefail

cd "$(dirname "$0")"

MODE="${1:-smoke}"
case "$MODE" in
    smoke)
        WARMUP=500
        SAMPLES=500
        ;;
    paper)
        WARMUP=1000
        SAMPLES=1000
        ;;
    *)
        echo "USAGE: $0 [smoke|paper]" >&2
        exit 2
        ;;
esac

CHAINS=4
PY="${PYTHON:-python3}"
RES_BASE=results_v2_diffuse_v5
mkdir -p "$RES_BASE"

# Cap XLA virtual devices to chain count (10 specs * 8 default = 80; capping
# to 4 keeps total threads near the 28-core ceiling).
export N_PARALLEL=4

# ---------------------------------------------------------------------------
# Spec table: <name> <runner> <extra args>
# ---------------------------------------------------------------------------
# Models A / A0 / B / C / D / E -- standalone runners
SPECS_MODELS=(
    "model_a      run_brazil_wp_v2.py"
    "model_a0     run_brazil_model_a_const_rr_v2.py"
    "model_b      run_brazil_model_b_v2.py"
    "model_c      run_brazil_model_c_v2.py"
    "model_d      run_brazil_model_d_v2.py"
    "model_e      run_brazil_model_e_v2.py"
)

# Robustness specs RUB1 / RUB3 / RUB4 -- routed through run_robustness_v2.py
SPECS_RUB=(
    "rub1"
    "rub3"
    "rub4"
)

# RUB5 has its own runner (Focus IPCA).
SPECS_FOCUS=(
    "rub5    run_brazil_rub5_focus_v2.py"
)

echo "================================================================"
echo "  v5+ ${MODE^^} MCMC driver"
echo "  warmup=$WARMUP samples=$SAMPLES chains=$CHAINS"
echo "  results_base=$RES_BASE  (per-spec: <name>_v5plus)"
echo "================================================================"

PIDS=()
LAUNCHED=()

launch_model() {
    local name="$1" runner="$2"
    local res="${RES_BASE}/${name}_v5plus"
    local log="${RES_BASE}/${name}_v5plus.log"
    mkdir -p "$res"
    echo "  -> launching $name ($runner) -> $res"
    JAX_PLATFORMS=cpu V5_TT_MODE=tt_trend V5_TT_PIN=tight \
        "$PY" "$runner" --use-diffuse \
        --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS" \
        --results-dir "$res" > "$log" 2>&1 &
    PIDS+=("$!")
    LAUNCHED+=("$name")
}

launch_rub() {
    local name="$1"
    local res="${RES_BASE}/${name}_v5plus"
    local log="${RES_BASE}/${name}_v5plus.log"
    mkdir -p "$res"
    echo "  -> launching $name (run_robustness_v2.py) -> $res"
    JAX_PLATFORMS=cpu V5_TT_MODE=tt_trend V5_TT_PIN=tight \
        V5PLUS_RESULTS_OVERRIDE="$res" \
        "$PY" run_robustness_v2.py --spec "$name" \
        --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS" \
        > "$log" 2>&1 &
    PIDS+=("$!")
    LAUNCHED+=("$name")
}

launch_focus() {
    local name="$1" runner="$2"
    local res="${RES_BASE}/${name}_v5plus"
    local log="${RES_BASE}/${name}_v5plus.log"
    mkdir -p "$res"
    echo "  -> launching $name ($runner) -> $res"
    # run_brazil_rub5_focus_v2.py does not accept --use-diffuse / --results-dir;
    # we route via V5PLUS_RESULTS_OVERRIDE env var instead.
    JAX_PLATFORMS=cpu V5_TT_MODE=tt_trend V5_TT_PIN=tight \
        V5PLUS_RESULTS_OVERRIDE="$res" \
        "$PY" "$runner" \
        --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS" \
        > "$log" 2>&1 &
    PIDS+=("$!")
    LAUNCHED+=("$name")
}

# Launch all 10 specs in parallel
for entry in "${SPECS_MODELS[@]}"; do
    read -r name runner <<< "$entry"
    launch_model "$name" "$runner"
done

for spec in "${SPECS_RUB[@]}"; do
    launch_rub "$spec"
done

for entry in "${SPECS_FOCUS[@]}"; do
    read -r name runner <<< "$entry"
    launch_focus "$name" "$runner"
done

echo
echo "  ${#PIDS[@]} subprocesses running"
echo "  PIDs: ${PIDS[*]}"
echo

# Wait for all
EXIT_STATUS=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    name=${LAUNCHED[$i]}
    if wait "$pid"; then
        echo "  [OK] $name (pid=$pid)"
    else
        rc=$?
        echo "  [FAIL] $name (pid=$pid, rc=$rc) -- see ${RES_BASE}/${name}_v5plus.log"
        EXIT_STATUS=1
    fi
done

echo
echo "  All 10 specs done.  Exit status: $EXIT_STATUS"
exit "$EXIT_STATUS"

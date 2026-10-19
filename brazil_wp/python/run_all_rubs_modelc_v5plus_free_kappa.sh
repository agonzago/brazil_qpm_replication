#!/usr/bin/env bash
# Re-estimate RUB1, RUB3, RUB4, RUB5 on top of the Model-C structural form
# (drops rho_QRR, rho_QZ_E, rho_QZ_F Koyck filters; keeps the TT-trend
# coupling kappa_g_tt estimated).
#
# Outputs to results_v2_diffuse_v5/<spec>_c_v5plus_free_kappa/ so the
# existing Model-A-based RUBs in <spec>_v5plus_free_kappa/ are preserved.
#
# Usage:
#   bash run_all_rubs_modelc_v5plus_free_kappa.sh smoke   # 500/500/4
#   bash run_all_rubs_modelc_v5plus_free_kappa.sh paper   # 1000/1000/4

set -uo pipefail
cd "$(dirname "$0")"

MODE=${1:-paper}
case "$MODE" in
    smoke)  WARMUP=500;  SAMPLES=500;  CHAINS=4 ;;
    paper)  WARMUP=1000; SAMPLES=1000; CHAINS=4 ;;
    *)      echo "usage: $0 {smoke|paper}"; exit 1 ;;
esac

PY="${PYTHON:-python3}"
RES_BASE=results_v2_diffuse_v5

SPECS=(rub1 rub3 rub4 rub5)

echo "================================================================"
echo "  v5+ RUB suite -- Model C base ($MODE: $WARMUP/$SAMPLES/$CHAINS)"
echo "================================================================"

PIDS=()
for spec in "${SPECS[@]}"; do
    res="${RES_BASE}/${spec}_c_v5plus_free_kappa"
    log="${RES_BASE}/${spec}_c_v5plus_free_kappa.log"
    mkdir -p "$res"
    echo ""
    echo "  Launching ${spec} -> ${res} (log ${log})"
    V5PLUS_RESULTS_OVERRIDE="$res" \
    V5_TT_MODE=tt_trend V5_TT_PIN=tight V5_TT_FREE_KAPPA=1 \
    RUB_BASE_MODEL=c \
    JAX_PLATFORMS=cpu \
        "$PY" run_robustness_v2.py --spec "$spec" \
        --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "  All ${#PIDS[@]} RUB-C specs launched, waiting..."
EXIT=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || EXIT=$?
done

echo ""
echo "  Done.  Exit: $EXIT"
echo "  Results under: $RES_BASE/<spec>_c_v5plus_free_kappa/"

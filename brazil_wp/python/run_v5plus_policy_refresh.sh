#!/usr/bin/env bash
# Refresh the policy block (sacrifice ratio + pass-through) for every
# v5+ free-kappa spec, without re-running MCMC.
#
# Motivation: the sacrifice ratio moved to the symmetric sign convention
# (sum|y| / sum|pi|) and the pass-through block now also reports the
# impact-normalised elasticity dP(h)/dS(0) alongside the legacy ratio of
# cumulative sums.  Both are read off saved draws, so this is minutes
# rather than the hour-plus of a full estimation pass.
#
# The runners reach the policy block through --skip-estimation
# --only-policy, which also regenerates the forecast-evaluation tables.
# Those are byte-identical on re-run (verified for model_c: only row
# order and two extra reporting variables differ), so this script
# restores evaluation/ from git afterwards to keep the change set
# confined to policy/.
#
# Usage:
#   bash run_v5plus_policy_refresh.sh            # all specs
#   bash run_v5plus_policy_refresh.sh model_c    # a subset

set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
RES_BASE=results_v2_diffuse_v5
export N_PARALLEL=2

# Only the specs that carry a policy block.  rub1/rub3/rub4 have never
# produced sacrifice-ratio or pass-through artefacts (run_robustness_v2.py
# reaches the sacrifice ratio only via --reanalyze, with plot=False, so
# nothing is written to disk), and the paper draws its cross-specification
# ranges from the set below.
SPECS=(
    "model_a      run_brazil_wp_v2.py"
    "model_a0     run_brazil_model_a_const_rr_v2.py"
    "model_b      run_brazil_model_b_v2.py"
    "model_c      run_brazil_model_c_v2.py"
    "model_d      run_brazil_model_d_v2.py"
    "model_e      run_brazil_model_e_v2.py"
    "rub5         run_brazil_rub5_focus_v2.py"
)

WANTED=("$@")
want() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

EXIT_STATUS=0
for entry in "${SPECS[@]}"; do
    read -r name runner <<< "$entry"
    want "$name" || continue

    res="${RES_BASE}/${name}_v5plus_free_kappa"
    log="${RES_BASE}/${name}_v5plus_free_kappa.policy.log"
    if [ ! -f "${res}/estimation/draws.npz" ]; then
        echo "  [SKIP] $name: no draws at ${res}/estimation/draws.npz"
        continue
    fi

    echo "  -> $name ($runner)"
    case "$runner" in
        run_brazil_rub5_focus_v2.py)
            JAX_PLATFORMS=cpu V5_TT_MODE=tt_trend V5_TT_PIN=tight \
            V5_TT_FREE_KAPPA=1 V5PLUS_RESULTS_OVERRIDE="$res" \
                "$PY" "$runner" \
                --skip-estimation --only-policy > "$log" 2>&1
            ;;
        *)
            JAX_PLATFORMS=cpu V5_TT_MODE=tt_trend V5_TT_PIN=tight \
            V5_TT_FREE_KAPPA=1 \
                "$PY" "$runner" --use-diffuse \
                --skip-estimation --only-policy \
                --results-dir "$res" > "$log" 2>&1
            ;;
    esac

    if [ $? -eq 0 ]; then
        echo "     [OK]   $name"
    else
        echo "     [FAIL] $name (see $log)"
        EXIT_STATUS=1
    fi

    # Confine the change set to policy/.
    git checkout -- "${res}/evaluation/" 2>/dev/null || true
done

echo
echo "  Policy refresh done. Exit: $EXIT_STATUS"
exit "$EXIT_STATUS"

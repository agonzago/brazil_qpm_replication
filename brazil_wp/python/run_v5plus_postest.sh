#!/usr/bin/env bash
# v5+ post-estimation artifact regeneration driver.
#
# After run_all_models_v5plus.sh finishes, this script regenerates the
# per-spec diagnostic plots (LZ, potential growth) plus the cross-spec
# tables and v5-baseline-vs-v5+ comparison figures.
#
# Usage:
#   bash run_v5plus_postest.sh

set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
RES=results_v2_diffuse_v5

SPECS=(
    model_a_v5plus
    model_a0_v5plus
    model_b_v5plus
    model_c_v5plus
    model_d_v5plus
    model_e_v5plus
    rub1_v5plus
    rub3_v5plus
    rub4_v5plus
    rub5_v5plus
)

echo "================================================================"
echo "  v5+ post-estimation artifact regeneration"
echo "================================================================"
echo

# 1. Per-spec diagnostic plots: LZ + potential growth fan chart.
for spec in "${SPECS[@]}"; do
    res_dir="${RES}/${spec}"
    if [ ! -d "$res_dir/estimation" ]; then
        echo "  [skip] $spec: no estimation/ dir found"
        continue
    fi
    echo "  -- $spec"
    if [ -f "$res_dir/diagnostic/smoothed_states.npz" ]; then
        echo "     LZ diagnostic..."
        "$PY" _plot_lz_diagnostic.py "$res_dir" \
            > "${res_dir}/_plot_lz.log" 2>&1 || \
            echo "     [WARN] LZ diag failed; see ${res_dir}/_plot_lz.log"
    fi
    if [ -d "$res_dir/estimation" ]; then
        echo "     potential growth fan chart..."
        "$PY" _plot_potential_growth.py "$res_dir" 100 \
            > "${res_dir}/_plot_g.log" 2>&1 || \
            echo "     [WARN] potential growth failed; see ${res_dir}/_plot_g.log"
    fi
done

# 2. Cross-spec aggregation tables (Tables 3, 4, 5 v5+).
echo
echo "  -- aggregation tables"
"$PY" build_v5plus_tables.py 2>&1 | tail -10

# 3. Cross-spec comparison: alt_runs (LZ + GDP_GAP).
echo
echo "  -- alt_runs comparison figure"
"$PY" _compare_alt_runs.py 2>&1 | tail -5

# 4. v5 baseline vs v5+ comparison figures.
echo
echo "  -- v5 baseline vs v5+ comparison figures"
"$PY" _plot_v5plus_vs_v5_baseline.py 2>&1 | tail -10

# 5. verify_paper_results.py is hardcoded to results_v2_diffuse/ (v4 paths)
# and only checks Model A / F.  Not applicable to v5+; skip.

echo
echo "  Post-estimation artifact regeneration complete."
echo "  Results under: $RES/"
echo "  Tables:        $RES/_tables/"
echo "  Figures:       $RES/_figures/"

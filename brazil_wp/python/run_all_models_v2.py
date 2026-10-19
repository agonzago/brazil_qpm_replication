"""Launch all QPM model variant estimations in parallel.

Each model (A through D) is run as a separate subprocess, allowing full
parallelism on multi-core machines.  All CLI arguments are forwarded to
each individual run script, so you can pass --warmup, --samples, --chains,
--skip-forecast, etc. exactly as you would to a single model.

Usage:
    python run_all_models_v2.py                          # all models (A, A0, B, C, D)
    python run_all_models_v2.py --models A B C           # subset of models
    python run_all_models_v2.py --warmup 200 --samples 300 --chains 4
    python run_all_models_v2.py --calibrate-only         # calibration + diagnostics only
    python run_all_models_v2.py --skip-estimation --only-policy
    python run_all_models_v2.py --dry-run                # show commands without running
    python run_all_models_v2.py --use-diffuse            # exact diffuse init (D&K 2012)
    python run_all_models_v2.py --use-diffuse --models A A0 B C D  # diffuse, all models
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_THIS_DIR = Path(__file__).resolve().parent

# Model label -> run script mapping
# A  = MR baseline (TV-R* with EC attractor)
# A0 = Constant R* (alpha_rr=0, pure random walk RR_BAR)
# B/C = Koyck filter variants
# D  = Simplified Taylor rule
# E  = D + COVID-era observations masked (2020Q1-Q2 set to NaN)
# F  = A baseline with Taylor-rule coefficients FIXED at calibrated values
# All models support --use-diffuse for exact diffuse initialization (D&K 2012).
MODELS = {
    "A":     _THIS_DIR / "run_brazil_wp_v2.py",
    "A0":    _THIS_DIR / "run_brazil_model_a_const_rr_v2.py",
    "B":     _THIS_DIR / "run_brazil_model_b_v2.py",
    "C":     _THIS_DIR / "run_brazil_model_c_v2.py",
    "D":     _THIS_DIR / "run_brazil_model_d_v2.py",
    "E":     _THIS_DIR / "run_brazil_model_e_v2.py",
    "F":     _THIS_DIR / "run_brazil_model_f_v2.py",
}


def _ensure_arviz_cache():
    """Pre-create the ArviZ daily-warning stamp file.

    ArviZ writes a daily timestamp to ~/.cache/arviz/daily_warning on first
    import.  When multiple processes import ArviZ concurrently the temp-file
    rename races and raises FileNotFoundError.  Writing the stamp once before
    forking prevents this.
    """
    cache_dir = Path.home() / ".cache" / "arviz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = cache_dir / "daily_warning"
    if not stamp.exists():
        from datetime import date
        stamp.write_text(date.today().isoformat())


def run_model(label, script, extra_args, python, results_base="results_v2"):
    """Run a single model script and return (label, returncode, elapsed_s)."""
    cmd = [python, str(script)] + extra_args
    log_path = _THIS_DIR / results_base / f"model_{label.lower()}" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(_THIS_DIR),
        )
    elapsed = time.time() - t0
    return label, proc.returncode, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run all QPM model variants in parallel",
        epilog="All unrecognised arguments are forwarded to each model script.",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Which models to run (default: all)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Max parallel processes (default: number of models)",
    )
    parser.add_argument(
        "--python", type=str, default=sys.executable,
        help="Python interpreter to use (default: current interpreter)",
    )
    parser.add_argument(
        "--calibrate-only", action="store_true",
        help="Run calibration + filter diagnostic + decomposition only (no NUTS)",
    )
    parser.add_argument(
        "--use-diffuse", action="store_true",
        help="Use exact diffuse Kalman filter (D&K 2012). "
             "Results go to results_v2_diffuse/ instead of results_v2/.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )

    args, extra = parser.parse_known_args()

    # Inject flags into forwarded args
    if args.calibrate_only and '--calibrate-only' not in extra:
        extra.append('--calibrate-only')
    if args.use_diffuse and '--use-diffuse' not in extra:
        extra.append('--use-diffuse')

    # Pre-create ArviZ cache to avoid race condition on parallel import
    _ensure_arviz_cache()

    models_to_run = {k: MODELS[k] for k in args.models}
    n_workers = args.max_workers or len(models_to_run)
    results_base = "results_v2_diffuse" if args.use_diffuse else "results_v2"

    mode = "CALIBRATE-ONLY" if args.calibrate_only else "FULL PIPELINE"
    if args.use_diffuse:
        mode += " + DIFFUSE"
    print("=" * 70)
    print(f"  QPM Model Batch Runner  [{mode}]")
    print("=" * 70)
    print(f"  Models:      {', '.join(models_to_run.keys())}")
    print(f"  Workers:     {n_workers}")
    print(f"  Python:      {args.python}")
    print(f"  Extra args:  {extra or '(none)'}")
    print("=" * 70)

    if args.dry_run:
        print("\n  DRY RUN -- commands that would be executed:\n")
        for label, script in models_to_run.items():
            cmd = [args.python, str(script)] + extra
            print(f"  Model {label}: {' '.join(cmd)}")
        print()
        return

    print()
    t_start = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_model, label, script, extra, args.python, results_base): label
            for label, script in models_to_run.items()
        }
        for label in models_to_run:
            log_path = _THIS_DIR / results_base / f"model_{label.lower()}" / "run.log"
            print(f"  [STARTED]  Model {label}  (log: {log_path})")

        for future in as_completed(futures):
            label, rc, elapsed = future.result()
            status = "OK" if rc == 0 else f"FAILED (exit {rc})"
            mins = elapsed / 60
            print(f"  [{status:>8s}]  Model {label}  ({mins:.1f} min)")
            results[label] = (rc, elapsed)

    total = time.time() - t_start
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    for label in sorted(results):
        rc, elapsed = results[label]
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"    Model {label}: {status:>8s}  ({elapsed/60:.1f} min)")
    print(f"\n  Total wall-clock time: {total/60:.1f} min")

    n_failed = sum(1 for rc, _ in results.values() if rc != 0)
    if n_failed:
        print(f"\n  WARNING: {n_failed} model(s) failed. Check logs above.")
        sys.exit(1)
    else:
        print("\n  All models completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

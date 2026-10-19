"""Run all post-estimation analyses for every model.

Phase 2 of the v5 plan: regenerate per-model post-hoc plots after the
re-estimation in Phase 1.  Each (model, script) combination runs as a
fresh subprocess so JAX/XLA state cannot leak between models.

Scripts invoked per model (each accepts --model {A, A0, B, C, D, E, F}):
  1. generate_posterior_diagnostics.py     — diag_observables/states/trends.png
  2. generate_posterior_filtration.py      — filtration_posterior.png + trends_posterior.png
  3. generate_additional_plots.py          — decomp_GDP_GROWTH.png, real_rate_natural_rate.png,
                                              output_gap_posterior.png
  4. generate_prior_vs_posterior_irfs.py   — prior_vs_posterior_irf_SHK_RS.png
  5. compute_spectral_radius.py            — spectral_radius.npz (text + appendix BK)

Usage:
  # All models, all scripts, sequentially:
  python run_post_estimation_all.py

  # Just two models, in parallel up to 2 at a time:
  python run_post_estimation_all.py --models A F --max-workers 2

  # Skip the slow IRF/spectral-radius pair:
  python run_post_estimation_all.py \
      --scripts diagnostics filtration additional

  # Continue past failures (default is to halt the whole batch):
  python run_post_estimation_all.py --skip-on-error
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO = _THIS_DIR.parent.parent
_PYTHON = str(_REPO / ".venv" / "bin" / "python3")

# Short alias -> script filename
SCRIPTS: dict[str, str] = {
    "diagnostics":  "generate_posterior_diagnostics.py",
    "filtration":   "generate_posterior_filtration.py",
    "additional":   "generate_additional_plots.py",
    "irfs":         "generate_prior_vs_posterior_irfs.py",
    "spectral":     "compute_spectral_radius.py",
}
SCRIPT_ORDER: list[str] = list(SCRIPTS.keys())  # canonical run order

MODEL_LABELS = ["A", "A0", "B", "C", "D", "E", "F"]
MODEL_RESULTS_SUBDIR = {
    "A":  "model_a",
    "A0": "model_a_const_rr",
    "B":  "model_b",
    "C":  "model_c",
    "D":  "model_d",
    "E":  "model_e",
    "F":  "model_f",
}


def _env_for_subprocess() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(var, "1")
    return env


def _log_path(results_base: str, model: str, alias: str) -> Path:
    log_dir = _THIS_DIR / results_base / MODEL_RESULTS_SUBDIR[model] / "post_estimation"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{alias}.log"


def run_one(model: str, alias: str, script: str, results_base: str,
            extra_args: list[str]) -> tuple[str, str, int, float]:
    """Run one (model, script) combination, return (model, alias, rc, elapsed_s)."""
    log_path = _log_path(results_base, model, alias)
    cmd = [_PYTHON, str(_THIS_DIR / script), "--model", model] + extra_args

    t0 = time.time()
    with open(log_path, "w") as fp:
        fp.write(f"# Command: {' '.join(cmd)}\n# Started: {time.strftime('%F %T')}\n\n")
        fp.flush()
        proc = subprocess.run(
            cmd, stdout=fp, stderr=subprocess.STDOUT,
            cwd=str(_THIS_DIR), env=_env_for_subprocess(),
        )
    return model, alias, proc.returncode, time.time() - t0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=MODEL_LABELS, choices=MODEL_LABELS,
        help="Subset of models (default: all 7).",
    )
    parser.add_argument(
        "--scripts", nargs="+", default=SCRIPT_ORDER, choices=SCRIPT_ORDER,
        help="Subset of post-hoc scripts (default: all 5).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="Parallel (model, script) jobs (default: 1, sequential).",
    )
    parser.add_argument(
        "--results-base", default="results_v2_diffuse",
        help="Results directory under brazil_wp/python/ (default: results_v2_diffuse).",
    )
    parser.add_argument(
        "--skip-on-error", action="store_true",
        help="Continue running remaining jobs after a failure.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned commands without running them.",
    )
    args, extra = parser.parse_known_args()

    jobs = [(m, alias, SCRIPTS[alias]) for m in args.models for alias in args.scripts]

    print("=" * 70)
    print("  POST-ESTIMATION DRIVER  (Phase 2)")
    print("=" * 70)
    print(f"  Models:       {', '.join(args.models)}")
    print(f"  Scripts:      {', '.join(args.scripts)}")
    print(f"  Workers:      {args.max_workers}")
    print(f"  Results base: {args.results_base}")
    print(f"  Total jobs:   {len(jobs)}")
    print("=" * 70)

    if args.dry_run:
        for m, alias, script in jobs:
            log = _log_path(args.results_base, m, alias)
            print(f"  [{m:3s} | {alias:11s}] {script}   -> {log.relative_to(_THIS_DIR)}")
        return

    t_start = time.time()
    results: list[tuple[str, str, int, float]] = []

    if args.max_workers == 1:
        for m, alias, script in jobs:
            print(f"  [START]   Model {m}  {alias:11s}  ({script})")
            sys.stdout.flush()
            r = run_one(m, alias, script, args.results_base, extra)
            results.append(r)
            _, _, rc, elapsed = r
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"  [{status:7s}] Model {m}  {alias:11s}  ({elapsed/60:.1f} min) "
                  f"-> {_log_path(args.results_base, m, alias).relative_to(_THIS_DIR)}")
            if rc != 0 and not args.skip_on_error:
                print("\n  Halting batch (use --skip-on-error to continue past failures).")
                break
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {
                ex.submit(run_one, m, alias, script, args.results_base, extra):
                    (m, alias) for m, alias, script in jobs
            }
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                m, alias, rc, elapsed = r
                status = "OK" if rc == 0 else f"FAIL({rc})"
                print(f"  [{status:7s}] Model {m}  {alias:11s}  ({elapsed/60:.1f} min)")
                if rc != 0 and not args.skip_on_error:
                    for f in futures:
                        f.cancel()
                    print("\n  Halting batch (use --skip-on-error to continue past failures).")
                    break

    total = time.time() - t_start
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    n_ok = sum(1 for _, _, rc, _ in results if rc == 0)
    n_fail = len(results) - n_ok
    for m, alias, rc, elapsed in results:
        tag = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"    Model {m:3s}  {alias:11s}  {tag:8s}  ({elapsed/60:.1f} min)")
    print(f"\n  Jobs:        {n_ok} OK / {n_fail} FAILED  (of {len(jobs)} planned)")
    print(f"  Wall-clock:  {total/60:.1f} min")
    print("=" * 70)

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

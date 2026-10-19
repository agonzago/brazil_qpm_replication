#!/usr/bin/env python3
"""Completeness checker for the Brazil QPM replication package.

The checker is static by default. It verifies that the package contains
everything its own scripts and README say is needed, that paths are portable,
that no obvious secrets are committed, and that the Python sources compile.

It understands two layouts:

  flat   README.md, requirements.txt, qpm_toolbox/, model/, data/, scripts/
  repo   README.md, requirements.txt, gpm_estimation/qpm_toolbox/,
         brazil_wp/python/{model,data,shared_data,*.py,.sh}

Optional runtime checks (``--imports``, ``--help-scripts``, ``--smoke``)
require the dependencies in ``requirements.txt`` to be installed.

Usage
-----
    python verify_replication_package.py [PACKAGE_DIR]
    python verify_replication_package.py [PACKAGE_DIR] --strict
    python verify_replication_package.py [PACKAGE_DIR] --json
    python verify_replication_package.py [PACKAGE_DIR] --imports --help-scripts
    python verify_replication_package.py [PACKAGE_DIR] --smoke
    python verify_replication_package.py [PACKAGE_DIR] --compare-source /path/to/gpm_ideas

Exit status is 1 if any FAIL is found, or if any WARN is found with --strict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical content expected in the package
# ---------------------------------------------------------------------------

MAIN_SCRIPTS = [
    "run_brazil_wp_v2.py",
    "run_brazil_model_a_const_rr_v2.py",
    "run_brazil_model_b_v2.py",
    "run_brazil_model_c_v2.py",
    "run_brazil_model_d_v2.py",
    "run_brazil_model_e_v2.py",
    "run_brazil_model_f_v2.py",
    "run_robustness_v2.py",
    "run_brazil_rub5_focus_v2.py",
    "run_all_models_v2.py",
    "run_forecast_evaluation.py",
    "run_bvar_analysis.py",
    "run_conditional_eval.py",
    "generate_posterior_diagnostics.py",
    "generate_posterior_filtration.py",
    "generate_additional_plots.py",
]

HELPER_MODULES = [
    "_v5_sweep_config.py",
    "_tt_trend_config.py",
    "_runner_loader.py",
]

V5_SHELL_SCRIPTS = [
    "run_all_models_v5plus.sh",
    "run_all_models_v5plus_free_kappa.sh",
    "run_all_models_v5plus_no_coupling.sh",
    "run_all_rubs_modelc_v5plus_free_kappa.sh",
    "run_v5plus_postest.sh",
    "save_all_smoothed_states_v5.sh",
]

REQUIRED_FILES = ["README.md", "requirements.txt"]
RECOMMENDED_FILES = ["LICENSE", "CITATION.cff"]

KEY_PACKAGES = [
    "jax",
    "numpyro",
    "arviz",
    "pandas",
    "numpy",
    "scipy",
    "sympy",
    "matplotlib",
    "statsmodels",
]

QPM_RE = re.compile(r"""["']([A-Za-z0-9_./-]+\.qpm)["']""")
ABS_PATH_RE = re.compile(r"(/home/|/Users/|[A-Za-z]:\\\\)")
SECRET_RE = re.compile(
    r"^\s*[A-Za-z_][A-Za-z0-9_]*"
    r"(?:api[_-]?key|token|secret|password)[A-Za-z0-9_]*\s*=\s*['\"][^'\"]{6,}['\"]",
    re.IGNORECASE,
)
DATE_CANDIDATES = ("date", "quarter", "period", "time", "obs_date")

SKIP_SCAN_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
}

TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".tex", ".yaml", ".yml", ".toml", ".cfg"}


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------


@dataclass
class Layout:
    kind: str
    scripts: Path
    toolbox: Path
    model: Path
    data: Path
    shared: Path
    import_path: Path


def detect_layout(root: Path) -> Layout:
    """Return the layout of *root* (flat package, repo mirror, or unknown)."""
    if (root / "scripts").is_dir() and (root / "qpm_toolbox").is_dir():
        return Layout(
            kind="flat",
            scripts=root / "scripts",
            toolbox=root / "qpm_toolbox",
            model=root / "model",
            data=root / "data",
            shared=root / "shared_data",
            import_path=root,
        )

    repo_py = root / "brazil_wp" / "python"
    repo_tb = root / "gpm_estimation" / "qpm_toolbox"
    if repo_py.is_dir() and repo_tb.is_dir():
        return Layout(
            kind="repo",
            scripts=repo_py,
            toolbox=repo_tb,
            model=repo_py / "model",
            data=repo_py / "data",
            shared=repo_py / "shared_data",
            import_path=root / "gpm_estimation",
        )

    # The user may point at brazil_wp/python directly.
    if (root / "run_brazil_wp_v2.py").is_file():
        return Layout(
            kind="scripts-only",
            scripts=root,
            toolbox=root.parent.parent / "gpm_estimation" / "qpm_toolbox",
            model=root / "model",
            data=root / "data",
            shared=root / "shared_data",
            import_path=root.parent.parent / "gpm_estimation",
        )

    return Layout(
        kind="unknown",
        scripts=root / "scripts",
        toolbox=root / "qpm_toolbox",
        model=root / "model",
        data=root / "data",
        shared=root / "shared_data",
        import_path=root,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


@dataclass
class Result:
    status: str  # PASS | FAIL | WARN | SKIP
    check: str
    detail: str = ""


class Checker:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.layout = detect_layout(self.root)
        self.results: list[Result] = []

    # -- helpers ---------------------------------------------------------
    def add(self, status: str, check: str, detail: str = "") -> None:
        self.results.append(Result(status, check, detail))

    def rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return str(path)

    def file(self, path: Path, required: bool = True, hint: str = "") -> None:
        name = self.rel(path)
        if path.is_file():
            self.add("PASS", f"file: {name}", f"{path.stat().st_size:,} bytes")
        elif required:
            self.add("FAIL", f"file: {name}", hint or "missing")
        else:
            self.add("WARN", f"file: {name}", hint or "missing (recommended)")

    def iter_files(self, skip_results: bool = True):
        for dirpath, dirnames, filenames in os.walk(self.root):
            if skip_results:
                dirnames[:] = [d for d in dirnames if not d.startswith("results")]
            for name in filenames:
                yield Path(dirpath) / name

    # -- structural checks ----------------------------------------------
    def check_layout(self) -> None:
        self.add("PASS", "layout", self.layout.kind)
        for rel in REQUIRED_FILES:
            self.file(self.root / rel)
        for rel in RECOMMENDED_FILES:
            self.file(
                self.root / rel,
                required=False,
                hint="missing (recommended for a public repo)",
            )

        required_dirs = [
            ("scripts", self.layout.scripts),
            ("toolbox", self.layout.toolbox),
            ("model", self.layout.model),
            ("data", self.layout.data),
        ]
        for label, path in required_dirs:
            if path.is_dir():
                n = sum(1 for p in path.rglob("*") if p.is_file())
                self.add("PASS", f"dir: {label}/", f"{self.rel(path)} ({n} files)")
            else:
                self.add("FAIL", f"dir: {label}/", f"missing: {self.rel(path)}")

    def check_scripts(self) -> None:
        scripts = self.layout.scripts
        for name in MAIN_SCRIPTS:
            self.file(scripts / name)
        for name in HELPER_MODULES:
            self.file(scripts / name, hint="required by the v5+ runners")

        present = [n for n in V5_SHELL_SCRIPTS if (scripts / n).is_file()]
        if not present:
            self.add(
                "WARN",
                "v5+ shell drivers",
                "none found; the paper's headline v5+ runs use these drivers",
            )
        elif len(present) < len(V5_SHELL_SCRIPTS):
            self.add(
                "WARN",
                "v5+ shell drivers",
                f"{len(present)}/{len(V5_SHELL_SCRIPTS)} present",
            )
        else:
            self.add("PASS", "v5+ shell drivers", f"{len(present)} present")

    def check_models(self) -> None:
        model_dir = self.layout.model
        scripts_dir = self.layout.scripts
        if not model_dir.is_dir() or not scripts_dir.is_dir():
            return

        referenced: dict[str, list[str]] = {}
        for py in sorted(scripts_dir.rglob("*.py")):
            try:
                lines = py.read_text(errors="ignore").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                code = line.split("#", 1)[0]
                for match in QPM_RE.finditer(code):
                    name = Path(match.group(1)).name
                    referenced.setdefault(name, []).append(f"{py.name}:{lineno}")

        if not referenced:
            self.add("WARN", "model references", "no .qpm references found in scripts")
            return

        missing = {
            name: where
            for name, where in referenced.items()
            if not (model_dir / name).is_file()
        }
        if missing:
            for name, where in sorted(missing.items()):
                self.add(
                    "WARN",
                    f"model: {name}",
                    "referenced in " + ", ".join(sorted(set(where))[:3]),
                )
        else:
            self.add(
                "PASS",
                "model references",
                f"all {len(referenced)} referenced .qpm files present",
            )

    def check_data(self) -> None:
        data_dir = self.layout.data
        est = data_dir / "estimation_data.csv"
        if est.is_file():
            self._check_estimation_csv(est)
        else:
            self.add("FAIL", f"data: {self.rel(est)}", "missing")

        self.file(
            data_dir / "bvar_data.csv",
            required=False,
            hint="missing (recommended for the BVAR benchmark)",
        )
        if not (
            (data_dir / "transformed_data.csv").is_file()
            or (self.layout.shared / "transformed_data.csv").is_file()
        ):
            self.add(
                "WARN",
                "BVAR transformed data",
                "transformed_data.csv missing from data/ and shared_data/",
            )

        bvar_script = self.layout.scripts / "run_bvar_analysis.py"
        if bvar_script.is_file():
            text = bvar_script.read_text(errors="ignore")
            if "transformed_data.csv" in text:
                candidates = [
                    data_dir / "transformed_data.csv",
                    self.layout.shared / "transformed_data.csv",
                ]
                if any(p.is_file() for p in candidates):
                    self.add("PASS", "BVAR input data", "transformed_data.csv present")
                else:
                    self.add(
                        "FAIL",
                        "BVAR input data",
                        "run_bvar_analysis.py reads transformed_data.csv, "
                        "which is not in data/ or shared_data/",
                    )

    def _check_estimation_csv(self, est: Path) -> None:
        try:
            with est.open(newline="") as fh:
                rows = list(csv.reader(fh))
        except OSError as exc:
            self.add("FAIL", f"data: {self.rel(est)}", str(exc))
            return

        header = rows[0] if rows else []
        body = [row for row in rows[1:] if any(cell.strip() for cell in row)]
        name = self.rel(est)

        if len(header) < 15:
            self.add(
                "FAIL",
                f"data: {name}",
                f"only {len(header)} columns; the paper uses 15 observables",
            )
        elif len(body) < 80:
            self.add(
                "FAIL",
                f"data: {name}",
                f"only {len(body)} rows; expected about 86 quarters",
            )
        else:
            empty = [
                header[i]
                for i in range(len(header))
                if all((row[i].strip() == "" if i < len(row) else True) for row in body)
            ]
            detail = f"{len(body)} rows x {len(header)} columns"
            if empty:
                self.add(
                    "WARN",
                    f"data: {name}",
                    detail + f"; all-empty columns: {', '.join(empty)}",
                )
            else:
                self.add("PASS", f"data: {name}", detail)

        self._check_dates(header, body)

    def _check_dates(self, header: list[str], body: list[list[str]]) -> None:
        idx = None
        for i, column in enumerate(header):
            if column.strip().lower() in DATE_CANDIDATES:
                idx = i
                break
        if idx is None or not body:
            self.add("SKIP", "data: date range", "no recognisable date column")
            return
        first = body[0][idx].strip() if idx < len(body[0]) else ""
        last = body[-1][idx].strip() if idx < len(body[-1]) else ""
        self.add("PASS", "data: date range", f"{first} to {last}")

    def check_syntax(self) -> None:
        targets: list[Path] = []
        for path in (self.layout.toolbox, self.layout.scripts):
            if path.is_dir():
                targets.extend(sorted(path.rglob("*.py")))
        bad = []
        for py in targets:
            try:
                compile(py.read_text(errors="ignore"), str(py), "exec")
            except SyntaxError as exc:
                bad.append(f"{self.rel(py)}:{exc.lineno}: {exc.msg}")
        if bad:
            for item in bad:
                self.add("FAIL", "syntax", item)
        elif targets:
            self.add("PASS", "syntax", f"{len(targets)} Python files compile")
        else:
            self.add("WARN", "syntax", "no Python files found")

    def check_portability(self) -> None:
        abs_hits: list[str] = []
        secret_hits: list[str] = []
        for path in self.iter_files():
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.name == "verify_replication_package.py":
                continue
            try:
                lines = path.read_text(errors="ignore").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                where = f"{self.rel(path)}:{lineno}"
                if ABS_PATH_RE.search(line):
                    abs_hits.append(where)
                if path.suffix == ".py" and SECRET_RE.match(line):
                    low = line.lower()
                    if not any(
                        token in low
                        for token in ("environ", "getenv", "input(", "todo",
                                      "your_", "<", "example")
                    ):
                        secret_hits.append(where)

        if abs_hits:
            self.add("WARN", "portability", "absolute paths: " + ", ".join(abs_hits[:5]))
        else:
            self.add("PASS", "portability", "no absolute paths found")

        if secret_hits:
            self.add(
                "FAIL",
                "secrets",
                "possible hard-coded secrets: " + ", ".join(secret_hits[:5]),
            )
        else:
            self.add("PASS", "secrets", "no hard-coded secrets found")

    def check_junk(self) -> None:
        junk: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if not d.startswith("results")]
            for name in list(dirnames):
                if name in {"__pycache__", ".ipynb_checkpoints", ".venv", "venv"}:
                    junk.append(self.rel(Path(dirpath) / name))
            for name in filenames:
                if name == ".DS_Store" or name.endswith(".pyc"):
                    junk.append(self.rel(Path(dirpath) / name))
        if junk:
            self.add("WARN", "build junk", f"{len(junk)} entries, e.g. {junk[0]}")
        else:
            self.add("PASS", "build junk", "none")

    def check_requirements(self) -> None:
        req = self.root / "requirements.txt"
        if not req.is_file():
            return
        text = req.read_text(errors="ignore")
        low = text.lower()
        missing = [pkg for pkg in KEY_PACKAGES if pkg not in low]
        if missing:
            self.add("FAIL", "requirements", "missing: " + ", ".join(missing))
        else:
            self.add("PASS", "requirements", "all key packages listed")

        if re.search(r"(^|\s)(-e|--editable)\s", low) or "file:" in low:
            self.add("WARN", "requirements", "contains a local/editable install")

        unpinned = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#") and "==" not in line
        ]
        if unpinned:
            self.add("WARN", "requirements", f"{len(unpinned)} unpinned entries")

        lock = self.root / "requirements-lock.txt"
        if lock.is_file():
            lines = lock.read_text(errors="ignore").splitlines()
            self.add("PASS", "requirements lock", f"{len(lines)} pinned entries")
        else:
            self.add(
                "WARN",
                "requirements lock",
                "missing (recommended for exact reproduction)",
            )

    def check_readme(self) -> None:
        readme = self.root / "README.md"
        if not readme.is_file():
            return
        text = readme.read_text(errors="ignore")
        missing = [script for script in MAIN_SCRIPTS if script not in text]
        if missing:
            self.add(
                "WARN",
                "README coverage",
                "scripts not mentioned: " + ", ".join(missing[:5]),
            )
        else:
            self.add("PASS", "README coverage", "all main scripts mentioned")

        if not re.search(r"v5\+|v5plus|tt[_ -]?trend", text, re.IGNORECASE):
            self.add(
                "WARN",
                "README v5+ workflow",
                "README does not describe the v5+/TT-trend headline runs",
            )
        else:
            self.add("PASS", "README v5+ workflow", "v5+ workflow mentioned")

    def check_reference_outputs(self) -> None:
        candidates = [
            self.root / "reference_outputs",
            self.root / "expected_outputs",
            self.root / "results",
            self.layout.scripts / "results_v2_diffuse_v5",
            self.layout.scripts / "results_v2_diffuse",
        ]
        found = [p for p in candidates if p.is_dir()]
        if found:
            summaries = sum(
                1
                for p in found
                for _ in p.rglob("*")
                if _.is_file() and (_.name.endswith(".csv") or _.name.endswith(".json"))
            )
            self.add(
                "PASS",
                "reference outputs",
                f"{self.rel(found[0])} ({summaries} csv/json files)",
            )
        else:
            self.add(
                "WARN",
                "reference outputs",
                "none found; numerical reproduction needs saved draws or a smoke run",
            )

    # -- freshness against the source repository ------------------------
    def check_freshness(self, source_root: Path) -> None:
        source = detect_layout(source_root)
        if source.kind == "unknown":
            self.add("SKIP", "freshness", f"unrecognised source layout: {source_root}")
            return

        script_pairs = [
            (self.layout.scripts / name, source.scripts / name)
            for name in MAIN_SCRIPTS + HELPER_MODULES
        ]
        script_pairs = [(a, b) for a, b in script_pairs if a.is_file() and b.is_file()]
        script_diff = [a.name for a, b in script_pairs if sha256(a) != sha256(b)]

        pkg_tb = {
            p.relative_to(self.layout.toolbox): p
            for p in self.layout.toolbox.rglob("*.py")
        }
        src_tb = {
            p.relative_to(source.toolbox): p
            for p in source.toolbox.rglob("*.py")
        }
        tb_pairs = [
            (pkg_tb[key], src_tb[key])
            for key in sorted(set(pkg_tb) & set(src_tb))
        ]
        tb_diff = [str(a.relative_to(self.layout.toolbox)) for a, b in tb_pairs
                   if sha256(a) != sha256(b)]

        model_pairs = [
            (self.layout.model / name, source.model / name)
            for name in sorted({p.name for p in self.layout.model.glob("*.qpm")})
            if (source.model / name).is_file()
        ]
        model_diff = [a.name for a, b in model_pairs if sha256(a) != sha256(b)]

        data_pairs = []
        for rel in ("estimation_data.csv", "bvar_data.csv"):
            a = self.layout.data / rel
            b = source.data / rel
            if a.is_file() and b.is_file():
                data_pairs.append((a, b))
        data_diff = [a.name for a, b in data_pairs if sha256(a) != sha256(b)]

        detail = (
            f"scripts {len(script_diff)}/{len(script_pairs)} differ; "
            f"toolbox {len(tb_diff)}/{len(tb_pairs)} differ; "
            f"models {len(model_diff)}/{len(model_pairs)} differ; "
            f"data {len(data_diff)}/{len(data_pairs)} differ"
        )
        if script_diff or tb_diff or model_diff or data_diff:
            examples = (script_diff + tb_diff + model_diff + data_diff)[:5]
            self.add("WARN", "freshness vs source repo", detail + "; e.g. " + ", ".join(examples))
        elif script_pairs or tb_pairs:
            self.add("PASS", "freshness vs source repo", detail)
        else:
            self.add("SKIP", "freshness vs source repo", "no comparable files found")

    # -- optional runtime checks ----------------------------------------
    def _run(self, cmd: list[str], timeout: int) -> tuple[int, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.layout.import_path)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return proc.returncode, (proc.stdout + proc.stderr)[-3000:]
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout}s"
        except OSError as exc:
            return 127, str(exc)

    def check_imports(self) -> None:
        code = (
            "import qpm_toolbox; "
            "import jax, numpyro, arviz, pandas, scipy, sympy, matplotlib; "
            "print('imports ok')"
        )
        rc, out = self._run([sys.executable, "-c", code], timeout=180)
        if rc == 0:
            self.add("PASS", "runtime imports", "qpm_toolbox and dependencies import")
        else:
            last = out.strip().splitlines()[-1] if out.strip() else "failed"
            self.add("FAIL", "runtime imports", last)

    def check_help(self) -> None:
        for name in MAIN_SCRIPTS:
            script = self.layout.scripts / name
            if not script.is_file():
                continue
            rc, out = self._run([sys.executable, str(script), "--help"], timeout=180)
            if rc == 0:
                self.add("PASS", f"help: {name}")
            else:
                last = out.strip().splitlines()[-1] if out.strip() else "failed"
                self.add("FAIL", f"help: {name}", last)

    def check_smoke(self) -> None:
        script = self.layout.scripts / "run_brazil_model_c_v2.py"
        if not script.is_file():
            self.add("SKIP", "smoke: Model C", "runner missing")
            return
        rc, out = self._run(
            [sys.executable, str(script), "--use-diffuse", "--calibrate-only"],
            timeout=1800,
        )
        if rc != 0:
            last = out.strip().splitlines()[-1] if out.strip() else "failed"
            self.add("FAIL", "smoke: Model C calibrate-only", last)
            return
        candidates = [
            self.layout.scripts / "results_v2_diffuse" / "model_c",
            self.layout.scripts / "results_v2_diffuse_v5" / "model_c_v5plus_free_kappa",
        ]
        populated = [p for p in candidates if p.is_dir() and any(p.rglob("*"))]
        if populated:
            n = sum(1 for p in populated[0].rglob("*") if p.is_file())
            self.add("PASS", "smoke: Model C calibrate-only", f"{n} output files")
        else:
            self.add(
                "WARN",
                "smoke: Model C calibrate-only",
                "ran cleanly but no model_c output directory was populated",
            )

    # -- reporting -------------------------------------------------------
    def report(self, as_json: bool) -> int:
        order = {"FAIL": 0, "WARN": 1, "SKIP": 2, "PASS": 3}
        results = sorted(
            self.results, key=lambda r: (order.get(r.status, 9), r.check)
        )
        counts = {
            key: sum(1 for r in results if r.status == key) for key in order
        }
        if as_json:
            print(
                json.dumps(
                    {
                        "root": str(self.root),
                        "layout": self.layout.kind,
                        "counts": counts,
                        "results": [asdict(r) for r in results],
                    },
                    indent=2,
                )
            )
        else:
            print(f"Replication package check: {self.root}")
            print(f"Layout: {self.layout.kind}")
            print("=" * 78)
            for status in ("FAIL", "WARN", "SKIP", "PASS"):
                group = [r for r in results if r.status == status]
                if not group:
                    continue
                print(f"\n{status} ({len(group)})")
                print("-" * 78)
                for result in group:
                    line = f"  {result.check}"
                    if result.detail:
                        line += f" — {result.detail}"
                    print(line)
            print("\n" + "=" * 78)
            print(
                f"Summary: {counts['PASS']} pass, {counts['FAIL']} fail, "
                f"{counts['WARN']} warn, {counts['SKIP']} skip"
            )
        return 1 if counts["FAIL"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package_dir",
        nargs="?",
        default=None,
        help="package root (default: directory containing this script)",
    )
    parser.add_argument("--strict", action="store_true", help="treat warnings as failure")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--imports", action="store_true", help="check runtime imports")
    parser.add_argument(
        "--help-scripts", action="store_true", help="run every script with --help"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="run Model C calibrate-only"
    )
    parser.add_argument(
        "--compare-source",
        default=None,
        metavar="PATH",
        help="compare package files against a source checkout",
    )
    args = parser.parse_args()

    root = (
        Path(args.package_dir)
        if args.package_dir
        else Path(__file__).resolve().parent
    )
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    checker = Checker(root)
    checker.check_layout()
    checker.check_scripts()
    checker.check_models()
    checker.check_data()
    checker.check_syntax()
    checker.check_portability()
    checker.check_junk()
    checker.check_requirements()
    checker.check_readme()
    checker.check_reference_outputs()
    if args.compare_source:
        checker.check_freshness(Path(args.compare_source))
    if args.imports:
        checker.check_imports()
    if args.help_scripts:
        checker.check_help()
    if args.smoke:
        checker.check_smoke()

    rc = checker.report(args.json)
    if args.strict and rc == 0 and any(r.status == "WARN" for r in checker.results):
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

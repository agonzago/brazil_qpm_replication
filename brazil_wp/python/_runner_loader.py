"""Map a model label to its runner module + canonical results subdirectory.

Each `run_brazil_*_v2.py` module exports the same set of module-level
configuration symbols (MOD_FILE, STRUCTURAL_PRIORS, FIXED_PARAMS,
MEAS_NOISE_STD, CYCLE_MAP, TREND_SNR_MAP, MANUAL_CYCLE_PRIORS,
FIXED_POINT_CONFIG, SHOCK_GROUPS, OBS_TREND_MAP, plus the helpers
load_brazil_data and compute_initial_conditions).

Post-estimation scripts that used to be hardcoded to Model A can use
`load_runner(label)` to retrieve the right module and `results_dir(label)`
to land outputs in the matching `results_v2_diffuse/model_<x>/` directory.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Label -> (runner module name, results subdir under results_v2_diffuse/)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "A":  ("run_brazil_wp_v2",                "model_a"),
    "A0": ("run_brazil_model_a_const_rr_v2",  "model_a_const_rr"),
    "B":  ("run_brazil_model_b_v2",           "model_b"),
    "C":  ("run_brazil_model_c_v2",           "model_c"),
    "D":  ("run_brazil_model_d_v2",           "model_d"),
    "E":  ("run_brazil_model_e_v2",           "model_e"),
    "F":  ("run_brazil_model_f_v2",           "model_f"),
}

MODEL_LABELS: list[str] = list(MODEL_REGISTRY.keys())


def load_runner(label: str) -> ModuleType:
    """Import and return the runner module for *label* (e.g. 'A', 'A0', 'F')."""
    if label not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model label {label!r}; expected one of {MODEL_LABELS}"
        )
    module_name, _ = MODEL_REGISTRY[label]
    return importlib.import_module(module_name)


def results_dir(label: str, base: str = "results_v2_diffuse") -> str:
    """Return the canonical results dir for *label* (paper baseline = diffuse)."""
    if label not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model label {label!r}; expected one of {MODEL_LABELS}"
        )
    _, subdir = MODEL_REGISTRY[label]
    return os.path.join(_THIS_DIR, base, subdir)

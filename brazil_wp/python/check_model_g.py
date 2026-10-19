"""Solve check for Model G (three-way ToT decomposition).

Loads brazil_qpm_model_g_v2.qpm, calls QPMModel.solution_check() at the
calibrated parameter values, and reports:

  - parser/lambdify success
  - Blanchard-Kahn (number of unit roots == number of trend states,
    no explosive eigenvalues)
  - stationary-block spectral radius < 1.

No MCMC, no plots.  Purely an architectural sanity check on the .qpm.
"""
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLBOX_DIR = os.path.join(SCRIPT_DIR, "..", "..", "gpm_estimation")
sys.path.insert(0, TOOLBOX_DIR)

from qpm_toolbox import QPMModel  # noqa: E402

MOD_FILE = os.path.join(SCRIPT_DIR, "model", "brazil_qpm_model_g_v2.qpm")

print(f"\n=== Model G solve check ===")
print(f"Model file: {MOD_FILE}\n")

m = QPMModel(MOD_FILE, results_dir=os.path.join(SCRIPT_DIR, "results_v2_diffuse", "model_g"),
             verbose=False, validate=True)

# Solve at calibrated parameter values (defaults from the .qpm file)
result = m.solution_check()

print("\n=== Diagnosis ===")
ok = (
    result['valid']
    and result.get('n_unstable', 0) == 0
    and result.get('n_unit_root', 0) == m.model.n_trend
)
print(f"  Solution valid:           {result['valid']}")
print(f"  No explosive eigenvalues: {result.get('n_unstable', 0) == 0}")
print(f"  Unit roots == n_trend:    {result.get('n_unit_root', 0)} == {m.model.n_trend}")
print(f"  Overall PASS:             {ok}")
sys.exit(0 if ok else 1)

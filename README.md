# Brazil QPM Replication — From Calibration to Estimation

**Andres Gonzalez Gomez and Juan David Prada**

This public repository contains the code, model files, and frozen data used in
*From Calibration to Estimation: A Modern Bayesian Framework for Quarterly
Projection Models*. The headline specification is the v5+ TT-trend model with
a free `kappa_g_tt` (Model C), estimated with full Bayesian methods
(NUTS/NumPyro) on an exact-diffuse Kalman filter.

The repository is **read-only for visitors**: anyone can clone or download it,
but only the owner can push. Estimation outputs, posterior draws, and paper
figures are **not** stored here; the commands below generate them locally.

## Download

```bash
git clone https://github.com/agonzago/brazil_qpm_replication.git
cd brazil_qpm_replication
```

Alternatively, use **Code → Download ZIP** on GitHub.

## Contents

```
.
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── CITATION.cff
├── verify_replication_package.py
├── gpm_estimation/
│   └── qpm_toolbox/                  # estimation library
└── brazil_wp/
    └── python/
        ├── run_*.py                  # model runners and evaluations
        ├── generate_*.py             # figure/diagnostic scripts
        ├── _*.py                     # v5+ helpers
        ├── run_*.sh                  # batch drivers
        ├── model/*.qpm               # QPM model files
        ├── data/                     # frozen estimation data + pipeline
        └── shared_data/              # BVAR inputs
```

The frozen estimation sample is 2004Q1–2025Q2 (86 quarters). The file
`data/estimation_data.csv` is the input used by every runner;
`shared_data/transformed_data.csv` is the BVAR input.

**Not included:** `results_v2_diffuse_v5/`, posterior draws, the paper PDF,
and generated figures. Running the pipeline creates the results tree locally.

## Requirements

- Python 3.12+
- Linux (tested on Ubuntu 22.04). macOS should work; Windows is untested.
- 16 GB RAM minimum (4 MCMC chains run concurrently).
- 4+ CPU cores recommended. No GPU is required (JAX CPU mode).

Install in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export JAX_PLATFORMS=cpu
```

`requirements.txt` lists the direct dependencies. `requirements-lock.txt` is the
exact environment used for the paper run (Python 3.12.3, Linux); install it
instead if you need to reproduce the original package versions:

```bash
pip install -r requirements-lock.txt
```

`qpm_toolbox` is included in this repository and should not be pip-installed.

The batch drivers use `${PYTHON:-python3}`. If you do not activate the
environment, set `PYTHON` explicitly:

```bash
export PYTHON="$PWD/.venv/bin/python"   # or an absolute path
```

## Quick check (no MCMC)

Run calibration, the filter diagnostics, and the shock decomposition for
Model C. This takes a few minutes and verifies that the data, model file,
toolbox, and plotting stack all work:

```bash
cd brazil_wp/python
python run_brazil_model_c_v2.py --use-diffuse --calibrate-only \
    --results-dir results_smoke/model_c
```

The completeness checker can also be run from the repository root:

```bash
python verify_replication_package.py .
```

It will warn that no reference outputs or license are present; that is expected
for this code/data-only repository.

## Replication steps

### 1. Paper headline: v5+ TT-trend with free kappa

The headline suite runs Models A0, B, C, D, E and the RUB1/RUB3/RUB4/RUB5
robustness specifications in parallel:

```bash
cd brazil_wp/python
bash run_all_models_v5plus_free_kappa.sh paper     # 1000 warmup / 1000 draws / 4 chains
```

Use `smoke` instead of `paper` for a shorter 500/500/4 run. Outputs go to
`results_v2_diffuse_v5/<spec>_v5plus_free_kappa/`.

Model A (free kappa) is run separately:

```bash
V5_TT_MODE=tt_trend V5_TT_PIN=tight V5_TT_FREE_KAPPA=1 \
python run_brazil_wp_v2.py --use-diffuse \
    --warmup 1000 --samples 1000 --chains 4 \
    --results-dir results_v2_diffuse_v5/model_a_v5plus_free_kappa
```

The fixed-kappa v5+ suite (the "before" comparison) is:

```bash
bash run_all_models_v5plus.sh paper
```

### 2. Robustness on the Model C structural form

```bash
bash run_all_rubs_modelc_v5plus_free_kappa.sh paper
```

Outputs go to `results_v2_diffuse_v5/<spec>_c_v5plus_free_kappa/`.

### 3. Single-model runs

Every runner accepts the same estimation flags. For example:

```bash
python run_brazil_model_c_v2.py --use-diffuse \
    --warmup 1000 --samples 1000 --chains 4 \
    --results-dir results_v2_diffuse_v5/model_c_v5plus_free_kappa
```

The model runners are:

`run_brazil_wp_v2.py` (A), `run_brazil_model_a_const_rr_v2.py` (A0),
`run_brazil_model_b_v2.py` (B), `run_brazil_model_c_v2.py` (C),
`run_brazil_model_d_v2.py` (D), `run_brazil_model_e_v2.py` (E, COVID-masked),
and `run_brazil_model_f_v2.py` (F, Taylor coefficients fixed).
`run_all_models_v2.py` launches A/A0/B/C/D/E/F as subprocesses and supports
`--dry-run`.

### 4. Forecast evaluation

Each estimation runner writes its per-spec evaluation files under
`results_v2_diffuse_v5/<spec>/evaluation/` (forecast and conditional RMSE,
spaghetti plots). The BVAR rolling-window benchmark is produced by:

```bash
python run_forecast_evaluation.py
```

This writes the BVAR CSVs to `results/evaluation/` and evaluates the
`results_v2_diffuse/model_a` baseline. `run_conditional_eval.py` runs the
conditional evaluation on the `results_v2_diffuse/model_*` baseline trees;
for v5+ results, use the per-spec
`evaluation/conditional_forecast_rmse.csv` files written by the runners.
`build_forecast_tables.py` combines the v5+ QPM files with the BVAR files into
the Section 6 LaTeX tables.

### 5. BVAR benchmark

```bash
python run_bvar_analysis.py
```

This reads `shared_data/transformed_data.csv` and writes its outputs under
`results/`.

### 6. Post-estimation tables and figures

```bash
bash run_v5plus_postest.sh
python build_v5plus_tables.py
python build_forecast_tables.py --treatment _v5plus_free_kappa
python generate_posterior_diagnostics.py --model C \
    --results-dir results_v2_diffuse_v5/model_c_v5plus_free_kappa
python generate_posterior_filtration.py --model C \
    --results-dir results_v2_diffuse_v5/model_c_v5plus_free_kappa
python generate_additional_plots.py --model C \
    --results-dir results_v2_diffuse_v5/model_c_v5plus_free_kappa
python data/stylized_facts.py
```

`build_forecast_tables.py --write` emits the four Section 6 LaTeX tables.
`run_v5plus_postest.sh` regenerates the cross-spec tables and comparison
figures. Stylized-facts plots are written to
`brazil_wp/paper/plots/stylized_facts/`.

## Script-to-output map

| Script | Main outputs |
|---|---|
| `run_brazil_wp_v2.py` | Model A posterior, IRFs, policy analysis |
| `run_brazil_model_a_const_rr_v2.py` | Model A0 (constant natural rate) |
| `run_brazil_model_b_v2.py` | Model B |
| `run_brazil_model_c_v2.py` | Model C (paper headline) |
| `run_brazil_model_d_v2.py` | Model D |
| `run_brazil_model_e_v2.py` | Model E (COVID-masked) |
| `run_brazil_model_f_v2.py` | Model F (fixed Taylor coefficients) |
| `run_robustness_v2.py` | RUB1/RUB3/RUB4/RUB5 prior sensitivity |
| `run_brazil_rub5_focus_v2.py` | RUB5 with Focus expectations |
| `run_all_models_v2.py` | Parallel launcher for A/A0/B/C/D/E/F |
| `run_forecast_evaluation.py` | BVAR rolling-window benchmark and v5-baseline QPM evaluation |
| `run_conditional_eval.py` | Conditional evaluation on the `results_v2_diffuse` baselines |
| `run_bvar_analysis.py` | Sign-restricted BVAR sacrifice ratio and pass-through |
| `generate_posterior_diagnostics.py` | Trace and posterior diagnostics |
| `generate_posterior_filtration.py` | Trend/cycle filtration plots |
| `generate_additional_plots.py` | GDP decomposition and extra figures |

## Outputs

Estimation outputs are written to `results_v2_diffuse_v5/<spec>/`:

- `estimation/arviz_summary.csv` — posterior medians, credible intervals, R-hat, ESS
- `estimation/draws.npz` — posterior draws
- `diagnostic/`, `decomposition/`, `forecast/`, `evaluation/`, `policy/` — figures and tables

These files are generated locally and are not stored in this repository.

## Runtime

The paper reports about two hours for the Model C baseline on a multi-core
workstation (4 chains, 1000 warmup + 1000 draws, CPU-only JAX). The full v5+
suite (nine specifications) takes substantially longer. Use the `smoke` mode
of the batch drivers for a first end-to-end check.

## Data provenance

- `data/estimation_data.csv` — frozen estimation input (2004Q1–2025Q2,
  15 observables plus transforms).
- `data/bvar_data.csv` — legacy BVAR input.
- `shared_data/transformed_data.csv` — BVAR input used by
  `run_bvar_analysis.py`.
- `data/fetchers/`, `data/build_estimation_data.py`, `data/run_data_pipeline.py`
  — pipeline used to rebuild the estimation data from the raw sources.
  Rebuilding requires the source APIs and credentials described in the
  pipeline scripts; the frozen CSV is provided so that estimation does not
  depend on network access.

## Verification

The static checker validates the repository layout, scripts, helper modules,
model/data cross-references, syntax, portable paths, and secrets:

```bash
python verify_replication_package.py .
```

Optional runtime checks require the installed dependencies:

```bash
python verify_replication_package.py . --imports --help-scripts
python verify_replication_package.py . --smoke
```

## License and citation

Citation metadata is in `CITATION.cff`. A license has **not** been selected
for this release. The repository is public and downloadable, but reuse rights
are not granted until a license is added. Add one before citing or reusing the
code.

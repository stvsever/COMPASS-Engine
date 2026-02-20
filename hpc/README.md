# COMPASS HPC Scripts (Slurm + Apptainer + Single GPU)

This folder contains an **example HPC workflow** for running COMPASS on Slurm-based GPU clusters.

It was developed to support **phenotype validation runs** of the COMPASS engine on a **single-GPU node** (example target hardware: **NVIDIA L40S, 48 GB VRAM**). The defaults in these scripts reflect that operating constraint.

It is intentionally written as a reproducible template for:
- Single-participant validation (`04_submit_single.sh`)
- Sequential batch validation (`05_submit_batch.sh`)

Use it as a starting point. For your cluster, adapt partition/GPU/path settings as needed.

## Scope and assumptions

- Scheduler: Slurm
- Container runtime: Apptainer
- Execution mode: single GPU, sequential participant processing (stability-first)
- Main local model: `Qwen/Qwen3-14B-AWQ`
- Embedding model: `Qwen/Qwen3-Embedding-8B`
- Project path on cluster: `~/compass_pipeline/multi_agent_system`

## What these scripts do

- `00_deploy_and_run.sh`: optional helper to copy this repo (and optionally data) to an HPC and SSH in
- `01_check_status.sh`: pre-flight checks (paths, models, Slurm/apptainer availability)
- `02_setup_environment.sh`: create container + venv environment
- `03_download_models.sh`: download/patch models in shared storage
- `04_submit_single.sh`: submit one participant smoke test job
- `05_submit_batch.sh`: submit sequential batch run across participant list in `utils/batch_run.py`
- `HPC_Operational_Guide.ipynb`: didactic notebook explaining HPC components and the end-to-end workflow with these scripts

## Notebook-first onboarding (recommended)

If you are new to HPC execution, start with:

- `hpc/HPC_Operational_Guide.ipynb`

It explains Slurm, login-vs-compute nodes, Apptainer, runtime tradeoffs, monitoring, and migration to other clusters.

## Important: these defaults are examples

These scripts currently include concrete defaults such as:
- `#SBATCH --partition=main`
- `#SBATCH --gres=gpu:l40s:1`
- `PROJECT_DIR="${HOME}/compass_pipeline/multi_agent_system"`

Treat these as sample values and update them for your infrastructure.

Design choices you will likely adapt for other hardware:
- GPU request (`--gres`) and partition/queue names
- memory/time requests (`--mem`, `--time`)
- context and per-role token budgets (longer context and larger outputs increase latency and memory pressure)
- whether to remain sequential (single GPU) or introduce safe parallelism (multiple GPUs or multiple nodes)

## Recommended run order

From `~/compass_pipeline/multi_agent_system`:

```bash
bash hpc/00_deploy_and_run.sh   # optional
bash hpc/01_check_status.sh
bash hpc/02_setup_environment.sh
bash hpc/03_download_models.sh
bash hpc/04_submit_single.sh
bash hpc/05_submit_batch.sh
```

All submission scripts are login-node safe:
- If run on `login*`, they auto-submit to Slurm compute nodes.
- Logs are written to `~/compass_pipeline/multi_agent_system/logs/`.

## Monitoring commands

```bash
squeue -u "$USER"
tail -f logs/compass_single_<JOBID>.out
tail -f logs/compass_batch_<JOBID>.out
cat logs/compass_single_<JOBID>.err
cat logs/compass_batch_<JOBID>.err
```

## Single run vs batch run

### `04_submit_single.sh`

- Validates one participant end-to-end.
- Uses local backend settings tuned for a single-GPU run.
- Intended as the gate before batch execution.

### `05_submit_batch.sh`

- Runs the participant cohort defined in `utils/batch_run.py`.
- Keeps execution **sequential** (single GPU) by design.
- Passes local runtime, token budgets, and prediction task flags through to `main.py` per participant:
  - `PREDICTION_TYPE` (default `binary`)
  - `TARGETS_FILE` must be JSON for binary queue construction (`binary_targets.json` style)
  - Optional `CLASS_LABELS`, `REGRESSION_OUTPUT` (univariate), `REGRESSION_OUTPUTS` (multivariate), `TASK_SPEC_FILE`, `TASK_SPEC_JSON`
  - Optional `ANNOTATIONS_JSON` for non-binary post-hoc validation (`run_validation_metrics.py` / `detailed_analysis.py`)
  - Optional runtime guidance: `GLOBAL_INSTRUCTION`, `ORCHESTRATOR_INSTRUCTION`, `EXECUTOR_INSTRUCTION`, `TOOLS_INSTRUCTION`, `INTEGRATOR_INSTRUCTION`, `PREDICTOR_INSTRUCTION`, `CRITIC_INSTRUCTION`, `COMMUNICATOR_INSTRUCTION`
  - `--max_tokens`
  - `--max_agent_input`, `--max_agent_output`
  - `--max_tool_input`, `--max_tool_output`
  - local engine/quantization settings

### `04_submit_single.sh`

- Uses the same prediction-task controls as Step 05.
- Defaults remain binary (`PREDICTION_TYPE=binary`) so existing case/control workflows still run unchanged.
- You can override to multiclass/regression/hierarchical by exporting:
  - `PREDICTION_TYPE`
  - optional `CLASS_LABELS`, `REGRESSION_OUTPUT` (univariate), `REGRESSION_OUTPUTS` (multivariate), `TASK_SPEC_FILE`, `TASK_SPEC_JSON`

For non-binary post-hoc validation in Step 05, also export:

```bash
ANNOTATIONS_JSON=/path/to/annotated_targets.json
```

Post-hoc analysis output directories are mode-aware:
- binary: `results/analysis/binary_confusion_matrix/` and `results/analysis/details/`
- non-binary: `results/analysis/<prediction_type>_metrics/` and `results/analysis/<prediction_type>_details/`

Non-binary detailed outputs include:
- annotation contract reports (`detailed_annotation_contract_<prediction_type>.json/.txt`)
- row-level evaluation payloads (`detailed_rows_<prediction_type>.json`)
- mode-specific plots (for example multiclass top-confusions, regression residual diagnostics, hierarchical coverage)

## Participant cohort definition

`utils/batch_run.py` contains the participant subset (EIDs + expected label + target string):

- Edit `PARTICIPANTS` in `utils/batch_run.py` to change the batch cohort.
- Data folder is resolved from:
  - `DATA_ROOT` env var if set, else
  - `../data/__FEATURES__/HPC_data`

Expected folder pattern:

```text
.../HPC_data/participant_ID<eid>/
```

## Local backend and public API backend behavior

The pipeline supports both:
- Local model inference (HPC/local GPU)
- Public API inference (OpenRouter/OpenAI)

They are intentionally configurable independently. The HPC scripts in this folder are focused on the **local backend** path for clinical validation runs.

Note: Explainability (XAI) remains binary-root-only. Non-binary prediction modes run normally, but XAI steps are skipped with explicit status metadata.

## Performance notes

- Local open-source inference on 1x L40S is expected to be slower than hosted public APIs.
- Most runtime is model generation latency and repeated agent/tool calls (COMPASS is intentionally multi-step).
- Longer context windows and higher output budgets increase latency.
- `05_submit_batch.sh` is tuned for reliability on single-GPU sequential execution.

## Quick adaptation checklist for other clusters

1. Update Slurm directives in `04_submit_single.sh` and `05_submit_batch.sh`.
2. Update paths (`PROJECT_DIR`, `MODELS_DIR`, `VENV_DIR`, `CONTAINER_IMAGE`).
3. Confirm `apptainer` command availability on compute nodes.
4. Run `01` -> `04` successfully before running `05`.

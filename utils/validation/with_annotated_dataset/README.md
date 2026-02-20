# Annotated Validation Toolkit

Mode-aware validation suite for COMPASS outputs against annotated datasets.

## Structure

- `run_validation_metrics.py`: metrics + visualization entrypoint (all prediction types)
- `detailed_analysis.py`: deep diagnostics entrypoint (all prediction types)
- `core/`
  - `io_utils.py`: annotation/result loading and prediction extraction
  - `metrics.py`: binary/multiclass/regression/hierarchical metrics
  - `visualization.py`: professional plots per prediction type
  - `reports.py`: mode-specific text/JSON reporting
  - `workflows.py`: orchestration logic shared by both CLIs
- `validation_guide.ipynb`: notebook walkthrough
- `annotation_templates/`: recommended annotation storage templates per prediction type
  - all examples are centralized in `annotation_templates/examples/`

## Inputs

- `--results_dir`:
  - directory containing participant outputs (for example `../results/participant_runs`)
- `--prediction_type`:
  - canonical values: `binary`, `multiclass`, `regression_univariate`, `regression_multivariate`, `hierarchical`
  - alias values also accepted: `binary_classification`, `multiclass_classification`, `univariate_regression`, `multivariate_regression`
- `--targets_file`:
  - required for `binary` mode
  - JSON is required (`binary_targets_example.json`)
- `--annotations_json`:
  - required for non-binary modes
  - both CLIs validate presence/existence and print template hints on failure

## Input Template Mapping

- `binary`:
  - `annotation_templates/examples/binary_targets_example.json`
- `multiclass`:
  - `annotation_templates/examples/multiclass_annotations_example.json`
- `regression_univariate`:
  - `annotation_templates/examples/regression_univariate_annotations_example.json`
- `regression_multivariate`:
  - `annotation_templates/examples/regression_multivariate_annotations_example.json`
- `hierarchical`:
  - `annotation_templates/examples/hierarchical_annotations_example.json`

## Prediction-Type Support

- `binary`:
  - confusion matrix
  - calibration and critic-quality diagnostics
- `multiclass`:
  - confusion matrix
  - per-class precision/recall/F1
  - top-label confidence calibration
  - top confusion ranking and confidence/entropy diagnostics
- `regression_univariate` / `regression_multivariate`:
  - MAE/RMSE/R2 (macro + micro)
  - density-gradient parity plots
  - per-output error bars
  - residual distributions and residual-vs-true diagnostics
  - largest absolute error ranking
- `hierarchical`:
  - per-node metrics (classification accuracy / regression R2)
  - macro hierarchical score
  - node score/support visualizations
  - node coverage diagnostics (truth/prediction overlap)
  - node metric heatmap (score/coverage/support matrix)

## Detailed Analysis Artifacts (Non-Binary)

Detailed runs now include:

- `detailed_analysis_<prediction_type>.json`
- `detailed_analysis_<prediction_type>.txt`
- `detailed_rows_<prediction_type>.json` (row-level evaluation payload used by metrics)
- `detailed_annotation_contract_<prediction_type>.json`
- `detailed_annotation_contract_<prediction_type>.txt`
- mode-specific professional plots (multiclass/regression/hierarchical)

The annotation contract artifacts explicitly report annotation validity, issue counts, and concrete issue examples by participant ID.

## Quick Usage

```bash
# Binary
python utils/validation/with_annotated_dataset/run_validation_metrics.py \
  --results_dir ../results/participant_runs \
  --prediction_type binary \
  --targets_file ../data/__TARGETS__/binary_targets.json \
  --output_dir ../results/analysis/binary_confusion_matrix

# Multiclass
python utils/validation/with_annotated_dataset/run_validation_metrics.py \
  --results_dir ../results/participant_runs \
  --prediction_type multiclass \
  --annotations_json ../data/__TARGETS__/annotated_targets.json \
  --output_dir ../results/analysis/multiclass

# Univariate regression
python utils/validation/with_annotated_dataset/run_validation_metrics.py \
  --results_dir ../results/participant_runs \
  --prediction_type regression_univariate \
  --annotations_json ../data/__TARGETS__/annotated_targets.json \
  --output_dir ../results/analysis/univariate_regression

# Multivariate regression
python utils/validation/with_annotated_dataset/run_validation_metrics.py \
  --results_dir ../results/participant_runs \
  --prediction_type regression_multivariate \
  --annotations_json ../data/__TARGETS__/annotated_targets.json \
  --output_dir ../results/analysis/multivariate_regression

# Hierarchical
python utils/validation/with_annotated_dataset/run_validation_metrics.py \
  --results_dir ../results/participant_runs \
  --prediction_type hierarchical \
  --annotations_json ../data/__TARGETS__/annotated_targets.json \
  --output_dir ../results/analysis/hierarchical
```

Use the same mode/input arguments with `detailed_analysis.py` for mode-specific text + JSON diagnostics and additional plots.

> XAI note: non-binary validation excludes XAI metrics because XAI currently supports binary classification only.

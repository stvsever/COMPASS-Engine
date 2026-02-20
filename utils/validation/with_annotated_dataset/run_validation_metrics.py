#!/usr/bin/env python3
"""
COMPASS Annotated Validation Metrics & Visual Diagnostics Runner.

This CLI now delegates to a modular validation core and supports:
- binary classification (integrated + per-disorder confusion matrices)
- multiclass classification (confusion + per-class diagnostics)
- univariate/multivariate regression (parity and error diagnostics)
- hierarchical mixed-task metrics (per-node diagnostics)

Backward-compatible usage is preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    # Package import path (e.g., notebooks/tests importing as module).
    from .core.constants import ACCEPTED_PREDICTION_TYPES, normalize_prediction_type
    from .core.io_utils import (  # noqa: F401
        collect_binary_results as collect_results,
        extract_binary_prediction as extract_prediction,
        extract_generalized_prediction,
        load_generalized_annotations,
        load_ground_truth,
    )
    from .core.input_templates import template_hint_for_mode
    from .core.metrics import (  # noqa: F401
        compute_binary_metrics as compute_metrics,
        compute_hierarchical_metrics as _compute_hierarchical_metrics,
        compute_multiclass_metrics as _compute_multiclass_metrics,
        compute_regression_metrics as _compute_regression_metrics,
    )
    from .core.workflows import parse_disorder_groups, run_metrics_workflow
except ImportError:
    # Direct script execution path.
    from core.constants import ACCEPTED_PREDICTION_TYPES, normalize_prediction_type
    from core.io_utils import (  # noqa: F401
        collect_binary_results as collect_results,
        extract_binary_prediction as extract_prediction,
        extract_generalized_prediction,
        load_generalized_annotations,
        load_ground_truth,
    )
    from core.input_templates import template_hint_for_mode
    from core.metrics import (  # noqa: F401
        compute_binary_metrics as compute_metrics,
        compute_hierarchical_metrics as _compute_hierarchical_metrics,
        compute_multiclass_metrics as _compute_multiclass_metrics,
        compute_regression_metrics as _compute_regression_metrics,
    )
    from core.workflows import parse_disorder_groups, run_metrics_workflow


def _print_binary_summary(summary: Dict[str, Any]) -> None:
    integrated = summary.get("integrated") if isinstance(summary.get("integrated"), dict) else {}
    metrics = integrated.get("metrics") if isinstance(integrated.get("metrics"), dict) else {}
    print("\n═══ Integrated Binary Metrics ═══")
    print(f"  Accuracy:    {float(metrics.get('accuracy') or 0.0):.1%}")
    print(f"  Sensitivity: {float(metrics.get('sensitivity') or 0.0):.1%}")
    print(f"  Specificity: {float(metrics.get('specificity') or 0.0):.1%}")
    print(f"  Precision:   {float(metrics.get('precision') or 0.0):.1%}")
    print(f"  F1:          {float(metrics.get('f1') or 0.0):.1%}")
    print(f"  MCC:         {float(metrics.get('mcc') or 0.0):+.3f}")


def _print_non_binary_summary(summary: Dict[str, Any]) -> None:
    integrated = summary.get("integrated") if isinstance(summary.get("integrated"), dict) else {}
    ptype = str(summary.get("prediction_type") or "unknown")
    n_rows = int(integrated.get("n_rows") or 0)
    contract = integrated.get("annotation_contract") if isinstance(integrated.get("annotation_contract"), dict) else {}
    print(f"\n═══ {ptype} Metrics ═══")
    print(f"  Rows evaluated: {n_rows}")
    if contract:
        print(
            "  Annotation validity: "
            f"{int(contract.get('n_valid_rows') or 0)}/{int(contract.get('n_rows') or 0)} "
            f"({float(contract.get('validity_rate') or 0.0):.1%})"
        )

    metrics = integrated.get("metrics") if isinstance(integrated.get("metrics"), dict) else {}
    if ptype == "multiclass":
        print(f"  Accuracy:   {float(metrics.get('accuracy') or 0.0):.1%}")
        print(f"  Macro F1:   {float(metrics.get('macro_f1') or 0.0):.3f}")
    elif ptype in {"regression_univariate", "regression_multivariate"}:
        print(f"  Macro MAE:  {metrics.get('macro_mae')}")
        print(f"  Macro RMSE: {metrics.get('macro_rmse')}")
        print(f"  Macro R2:   {metrics.get('macro_r2')}")
    elif ptype == "hierarchical":
        print(f"  Macro score: {metrics.get('macro_score')}")
        print(f"  Nodes:       {metrics.get('n_nodes')}")


def main() -> None:
    help_epilog = (
        "Input template examples are centralized in:\n"
        "  utils/validation/with_annotated_dataset/annotation_templates/examples/\n"
        "Use --targets_file for binary and --annotations_json for non-binary modes."
    )
    parser = argparse.ArgumentParser(
        description="COMPASS Validation: Mode-aware metrics and visualization diagnostics",
        epilog=help_epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--results_dir", required=True, help="Path to participant_runs directory")
    parser.add_argument(
        "--targets_file",
        required=False,
        default="",
        help="Path to binary ground-truth JSON file (required for prediction_type=binary)",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for metrics/plots")
    parser.add_argument(
        "--disorder_groups",
        default="",
        help="Comma-separated groups/disorders for per-group outputs",
    )
    parser.add_argument(
        "--prediction_type",
        default="binary",
        choices=sorted(ACCEPTED_PREDICTION_TYPES),
        help="Task type for validation",
    )
    parser.add_argument(
        "--annotations_json",
        default="",
        help="Generalized annotation JSON (required for non-binary modes)",
    )
    args = parser.parse_args()

    mode = normalize_prediction_type(args.prediction_type)
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        parser.error(f"--results_dir does not exist: {results_dir}")

    if mode == "binary":
        targets = Path(str(args.targets_file or "").strip())
        if not str(args.targets_file or "").strip():
            parser.error(
                "binary validation requires --targets_file.\n"
                f"Template example: {template_hint_for_mode(mode)}"
            )
        if not targets.exists():
            parser.error(
                f"--targets_file not found: {targets}\n"
                f"Template example: {template_hint_for_mode(mode)}"
            )
        if targets.suffix.lower() != ".json":
            parser.error(
                "binary validation expects JSON for --targets_file. "
                "Legacy txt is not supported.\n"
                f"Template example: {template_hint_for_mode(mode)}"
            )
    else:
        ann = Path(str(args.annotations_json or "").strip())
        if not str(args.annotations_json or "").strip():
            parser.error(
                f"{mode} validation requires --annotations_json.\n"
                f"Template example: {template_hint_for_mode(mode)}"
            )
        if not ann.exists():
            parser.error(
                f"--annotations_json not found: {ann}\n"
                f"Template example: {template_hint_for_mode(mode)}"
            )

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        summary = run_metrics_workflow(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            prediction_type=mode,
            targets_file=args.targets_file,
            annotations_json=args.annotations_json,
            disorder_groups=parse_disorder_groups(args.disorder_groups),
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if mode == "binary":
        _print_binary_summary(summary)
    else:
        _print_non_binary_summary(summary)
        print("  NOTE: XAI currently supports binary classification only; non-binary validation excludes XAI metrics.")

    print("\n✓ Validation metrics generation complete.")


if __name__ == "__main__":
    main()

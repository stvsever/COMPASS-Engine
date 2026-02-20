#!/usr/bin/env python3
"""
COMPASS Annotated Validation — Detailed Performance Analysis.

Mode-aware detailed analysis entrypoint.

Binary mode outputs (backward-compatible names):
- detailed_analysis.txt
- composite_vs_accuracy.png
- probability_calibration.png
- iteration_improvement.png
- verdict_accuracy.png

Non-binary mode outputs:
- detailed_analysis_{prediction_type}.json
- detailed_analysis_{prediction_type}.txt
- detailed_* diagnostic visualizations per mode
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

try:
    # Package import path (e.g., notebooks/tests importing as module).
    from .core.constants import ACCEPTED_PREDICTION_TYPES, normalize_prediction_type
    from .core.input_templates import template_hint_for_mode
    from .core.workflows import parse_disorder_groups, run_detailed_workflow
except ImportError:
    # Direct script execution path.
    from core.constants import ACCEPTED_PREDICTION_TYPES, normalize_prediction_type
    from core.input_templates import template_hint_for_mode
    from core.workflows import parse_disorder_groups, run_detailed_workflow


def _print_summary(summary: Dict[str, Any]) -> None:
    ptype = str(summary.get("prediction_type") or "unknown")
    outputs = [str(x) for x in (summary.get("outputs") or []) if str(x).strip()]
    groups = summary.get("groups") if isinstance(summary.get("groups"), dict) else {}
    integrated = summary.get("integrated") if isinstance(summary.get("integrated"), dict) else {}
    contract = integrated.get("annotation_contract") if isinstance(integrated.get("annotation_contract"), dict) else {}

    print(f"\n═══ Detailed Analysis Summary ({ptype}) ═══")
    print(f"  Artifacts: {len(outputs)}")
    if contract:
        print(
            "  Annotation validity: "
            f"{int(contract.get('n_valid_rows') or 0)}/{int(contract.get('n_rows') or 0)} "
            f"({float(contract.get('validity_rate') or 0.0):.1%})"
        )
    if groups:
        print(f"  Group-level analyses: {len(groups)}")


def main() -> None:
    help_epilog = (
        "Input template examples are centralized in:\n"
        "  utils/validation/with_annotated_dataset/annotation_templates/examples/\n"
        "Use --targets_file for binary and --annotations_json for non-binary modes."
    )
    parser = argparse.ArgumentParser(
        description="COMPASS Validation: Detailed mode-aware analysis",
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
    parser.add_argument("--output_dir", required=True, help="Output directory for analysis files")
    parser.add_argument(
        "--disorder_groups",
        default="",
        help="Comma-separated groups/disorders for per-group analysis",
    )
    parser.add_argument(
        "--prediction_type",
        default="binary",
        choices=sorted(ACCEPTED_PREDICTION_TYPES),
        help="Task type for detailed analysis",
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
                "binary detailed analysis requires --targets_file.\n"
                f"Template example: {template_hint_for_mode(mode)}"
            )
        if not targets.exists():
            parser.error(
                f"--targets_file not found: {targets}\n"
                f"Template example: {template_hint_for_mode(mode)}"
            )
        if targets.suffix.lower() != ".json":
            parser.error(
                "binary detailed analysis expects JSON for --targets_file. "
                "Legacy txt is not supported.\n"
                f"Template example: {template_hint_for_mode(mode)}"
            )
    else:
        ann = Path(str(args.annotations_json or "").strip())
        if not str(args.annotations_json or "").strip():
            parser.error(
                f"{mode} detailed analysis requires --annotations_json.\n"
                f"Template example: {template_hint_for_mode(mode)}"
            )
        if not ann.exists():
            parser.error(
                f"--annotations_json not found: {ann}\n"
                f"Template example: {template_hint_for_mode(mode)}"
            )

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        summary = run_detailed_workflow(
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

    _print_summary(summary)

    if mode != "binary":
        print("  NOTE: XAI currently supports binary classification only; non-binary detailed analysis excludes XAI metrics.")

    print("\n✓ Detailed analysis complete.")


if __name__ == "__main__":
    main()

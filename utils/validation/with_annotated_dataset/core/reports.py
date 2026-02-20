"""Text and JSON reporting helpers for annotated validation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def write_json(payload: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _header_lines(title: str) -> List[str]:
    sep = "=" * 92
    return [
        sep,
        f"  COMPASS ENGINE — ANNOTATED DATASET VALIDATION",
        f"  {title}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
        "",
    ]


def _format_float(value: Any, fmt: str = ".3f") -> str:
    if value is None:
        return "N/A"
    try:
        return format(float(value), fmt)
    except Exception:
        return "N/A"


def _cohort_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    total = len(rows)
    succeeded = sum(1 for r in rows if str(r.get("status") or "") == "SUCCESS")
    failed = total - succeeded
    out = [
        "1. COHORT SUMMARY",
        f"   Total participants:        {total}",
        f"   Successful predictions:    {succeeded}",
        f"   Failed/Unavailable:        {failed}",
        "",
    ]
    return out


def write_binary_text_report(
    *,
    rows: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: str,
    title: str = "Integrated Binary Validation",
) -> None:
    lines = _header_lines(title)
    lines.extend(_cohort_lines(rows))

    lines.extend(
        [
            "2. BINARY METRICS",
            f"   Accuracy:                {metrics.get('accuracy', 0.0):.1%}",
            f"   Balanced Accuracy:       {metrics.get('balanced_accuracy', 0.0):.1%}",
            f"   Sensitivity (Recall+):   {metrics.get('sensitivity', 0.0):.1%}",
            f"   Specificity (Recall-):   {metrics.get('specificity', 0.0):.1%}",
            f"   Precision:               {metrics.get('precision', 0.0):.1%}",
            f"   F1 Score:                {metrics.get('f1', 0.0):.1%}",
            f"   MCC:                     {metrics.get('mcc', 0.0):+.3f}",
            f"   Brier score:             {_format_float(metrics.get('brier'))}",
            f"   Expected calibration err:{_format_float(metrics.get('ece'))}",
            "",
            "3. CONFUSION COUNTS",
            f"   TP: {int(metrics.get('tp') or 0)}",
            f"   FP: {int(metrics.get('fp') or 0)}",
            f"   TN: {int(metrics.get('tn') or 0)}",
            f"   FN: {int(metrics.get('fn') or 0)}",
            "",
        ]
    )

    verdict_rows = [r for r in rows if r.get("predicted") in {"CASE", "CONTROL"}]
    if verdict_rows:
        sat = [r for r in verdict_rows if "SATIS" in str(r.get("verdict") or "").upper()]
        unsat = [r for r in verdict_rows if "UNSATIS" in str(r.get("verdict") or "").upper()]
        lines.append("4. CRITIC VERDICT QUALITY")
        if sat:
            lines.append(
                f"   SATISFACTORY runs:       {len(sat)} (accuracy {sum(1 for r in sat if r.get('correct')) / len(sat):.1%})"
            )
        if unsat:
            lines.append(
                f"   UNSATISFACTORY runs:     {len(unsat)} (accuracy {sum(1 for r in unsat if r.get('correct')) / len(unsat):.1%})"
            )
        lines.append("")

    lines.append("5. PARTICIPANT TABLE")
    lines.append("   EID            Actual     Predicted   Correct   Prob(CASE)   Verdict")
    lines.append("   " + "-" * 80)
    for row in rows:
        eid = str(row.get("eid") or "-")
        actual = str(row.get("actual") or "-")
        pred = str(row.get("predicted") or "FAILED")
        correct = "YES" if bool(row.get("correct")) else "NO"
        prob = _format_float(row.get("probability"))
        verdict = str(row.get("verdict") or "-")
        lines.append(f"   {eid:<13}{actual:<11}{pred:<12}{correct:<10}{prob:<13}{verdict}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_multiclass_text_report(
    *,
    rows: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: str,
    title: str = "Multiclass Validation",
) -> None:
    lines = _header_lines(title)
    lines.extend(_cohort_lines(rows))
    prob_diag = metrics.get("probability_diagnostics") if isinstance(metrics.get("probability_diagnostics"), dict) else {}
    topk = prob_diag.get("top_k_accuracy") if isinstance(prob_diag.get("top_k_accuracy"), dict) else {}

    lines.extend(
        [
            "2. MULTICLASS METRICS",
            f"   Accuracy:                {metrics.get('accuracy', 0.0):.1%}",
            f"   Balanced Accuracy:       {metrics.get('balanced_accuracy', 0.0):.1%}",
            f"   Macro Precision:         {metrics.get('macro_precision', 0.0):.3f}",
            f"   Macro Recall:            {metrics.get('macro_recall', 0.0):.3f}",
            f"   Macro F1:                {metrics.get('macro_f1', 0.0):.3f}",
            f"   Weighted F1:             {metrics.get('weighted_f1', 0.0):.3f}",
            f"   ECE (top-label):         {_format_float((metrics.get('confidence_calibration') or {}).get('ece'))}",
            f"   Top-1 / Top-2 / Top-3:   {_format_float(topk.get('top1'), '.3f')} / {_format_float(topk.get('top2'), '.3f')} / {_format_float(topk.get('top3'), '.3f')}",
            f"   Mean conf (correct):     {_format_float(prob_diag.get('mean_confidence_correct'))}",
            f"   Mean conf (incorrect):   {_format_float(prob_diag.get('mean_confidence_incorrect'))}",
            f"   Mean entropy (correct):  {_format_float(prob_diag.get('mean_entropy_correct'))}",
            f"   Mean entropy (incorrect):{_format_float(prob_diag.get('mean_entropy_incorrect'))}",
            "",
            "3. PER-CLASS BREAKDOWN",
            "   Label                          Support   Precision   Recall   F1",
            "   " + "-" * 74,
        ]
    )

    per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
    for label in sorted(per_class.keys()):
        block = per_class[label] if isinstance(per_class[label], dict) else {}
        lines.append(
            f"   {label:<30}{int(block.get('support') or 0):>7}"
            f"{float(block.get('precision') or 0.0):>12.3f}"
            f"{float(block.get('recall') or 0.0):>9.3f}"
            f"{float(block.get('f1') or 0.0):>7.3f}"
        )

    confusions = metrics.get("top_confusions") if isinstance(metrics.get("top_confusions"), list) else []
    if confusions:
        lines.extend(["", "4. TOP CONFUSIONS", "   True label  -> Pred label      Count", "   " + "-" * 46])
        for row in confusions[:15]:
            lines.append(
                f"   {str(row.get('actual') or '-'): <12} -> {str(row.get('predicted') or '-'): <14} {int(row.get('count') or 0):>5}"
            )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_regression_text_report(
    *,
    rows: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: str,
    prediction_type: str,
    title: str,
) -> None:
    lines = _header_lines(title)
    lines.extend(_cohort_lines(rows))

    lines.extend(
        [
            f"2. {prediction_type.upper()} METRICS",
            f"   Macro MAE:               {_format_float(metrics.get('macro_mae'))}",
            f"   Macro RMSE:              {_format_float(metrics.get('macro_rmse'))}",
            f"   Macro R2:                {_format_float(metrics.get('macro_r2'))}",
            f"   Micro MAE:               {_format_float((metrics.get('micro') or {}).get('mae'))}",
            f"   Micro RMSE:              {_format_float((metrics.get('micro') or {}).get('rmse'))}",
            f"   Micro R2:                {_format_float((metrics.get('micro') or {}).get('r2'))}",
            f"   Micro error mean/std:    {_format_float(((metrics.get('micro') or {}).get('residual_summary') or {}).get('mean_error'))} / {_format_float(((metrics.get('micro') or {}).get('residual_summary') or {}).get('std_error'))}",
            "",
            "3. PER-OUTPUT REGRESSION TABLE",
            "   Output                         n       MAE      RMSE       R2   Pearson  Spearman   P05err   P95err",
            "   " + "-" * 106,
        ]
    )

    per_output = metrics.get("per_output") if isinstance(metrics.get("per_output"), dict) else {}
    for output_name in sorted(per_output.keys()):
        block = per_output[output_name] if isinstance(per_output[output_name], dict) else {}
        residual = block.get("residual_summary") if isinstance(block.get("residual_summary"), dict) else {}
        lines.append(
            f"   {output_name:<30}"
            f"{int(block.get('n') or 0):>5}"
            f"{_format_float(block.get('mae')):>10}"
            f"{_format_float(block.get('rmse')):>10}"
            f"{_format_float(block.get('r2')):>10}"
            f"{_format_float(block.get('pearson')):>9}"
            f"{_format_float(block.get('spearman')):>10}"
            f"{_format_float(residual.get('p05_error')):>9}"
            f"{_format_float(residual.get('p95_error')):>9}"
        )

    top_errors = metrics.get("largest_absolute_errors") if isinstance(metrics.get("largest_absolute_errors"), list) else []
    if top_errors:
        lines.extend(
            [
                "",
                "4. LARGEST ABSOLUTE ERRORS",
                "   EID            Output                    Actual      Pred      |Err|     SignedErr",
                "   " + "-" * 90,
            ]
        )
        for row in top_errors[:20]:
            lines.append(
                f"   {str(row.get('eid') or '-'): <14}"
                f"{str(row.get('output') or '-'): <26}"
                f"{_format_float(row.get('actual')):>8}"
                f"{_format_float(row.get('predicted')):>10}"
                f"{_format_float(row.get('abs_error')):>11}"
                f"{_format_float(row.get('signed_error')):>13}"
            )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_hierarchical_text_report(
    *,
    rows: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: str,
    title: str = "Hierarchical Validation",
) -> None:
    lines = _header_lines(title)
    lines.extend(_cohort_lines(rows))

    lines.extend(
        [
            "2. HIERARCHICAL AGGREGATES",
            f"   Nodes evaluated:          {int(metrics.get('n_nodes') or 0)}",
            f"   Classification nodes:     {int(metrics.get('n_classification_nodes') or 0)}",
            f"   Regression nodes:         {int(metrics.get('n_regression_nodes') or 0)}",
            f"   Macro classification acc: {_format_float(metrics.get('macro_classification_accuracy'))}",
            f"   Macro regression R2:      {_format_float(metrics.get('macro_regression_r2'))}",
            f"   Composite macro score:    {_format_float(metrics.get('macro_score'))}",
            f"   Macro node coverage:      {_format_float((metrics.get('coverage_summary') or {}).get('macro_node_coverage_rate'))}",
            "",
            "3. PER-NODE TABLE",
            "   Node ID                         Mode                       n      Score    Coverage  TruthOnly PredOnly",
            "   " + "-" * 112,
        ]
    )

    per_node = metrics.get("per_node") if isinstance(metrics.get("per_node"), dict) else {}
    for node_id in sorted(per_node.keys()):
        block = per_node[node_id] if isinstance(per_node[node_id], dict) else {}
        mode = str(block.get("mode") or "")
        n = int(block.get("n") or 0)
        score = block.get("accuracy") if mode.endswith("classification") else block.get("macro_r2")
        coverage = block.get("coverage") if isinstance(block.get("coverage"), dict) else {}
        lines.append(
            f"   {node_id:<32}{mode:<27}{n:>4}{_format_float(score):>11}"
            f"{_format_float(coverage.get('coverage_rate')):>11}"
            f"{int(coverage.get('truth_only') or 0):>10}"
            f"{int(coverage.get('pred_only') or 0):>9}"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

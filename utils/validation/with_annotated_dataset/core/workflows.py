"""End-to-end validation workflows for annotated datasets."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import (
    ACCEPTED_PREDICTION_TYPES,
    PREDICTION_TYPE_LABELS,
    SUPPORTED_PREDICTION_TYPES,
    normalize_prediction_type,
)
from .annotation_contract import summarize_annotation_contract
from .io_utils import (
    collect_binary_results,
    collect_generalized_rows,
    load_generalized_annotations,
)
from .metrics import (
    compute_binary_metrics,
    compute_hierarchical_metrics,
    compute_multiclass_metrics,
    compute_regression_metrics,
)
from .reports import (
    write_binary_text_report,
    write_hierarchical_text_report,
    write_json,
    write_multiclass_text_report,
    write_regression_text_report,
)
from .visualization import (
    plot_binary_composite_vs_accuracy,
    plot_binary_confusion_matrix,
    plot_binary_iteration_improvement,
    plot_binary_probability_calibration,
    plot_binary_verdict_accuracy,
    plot_hierarchical_mode_distribution,
    plot_hierarchical_metric_heatmap,
    plot_hierarchical_node_coverage,
    plot_hierarchical_node_scores,
    plot_multiclass_confidence_calibration,
    plot_multiclass_confidence_diagnostics,
    plot_multiclass_confusion_matrix,
    plot_multiclass_label_distribution,
    plot_multiclass_per_class,
    plot_multiclass_top_confusions,
    plot_regression_error_bars,
    plot_regression_parity,
    plot_regression_residual_distribution,
    plot_regression_residuals_vs_truth,
    plot_regression_top_errors,
)


def parse_disorder_groups(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text or "").strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "group"


def _resolve_multiclass_label(truth: Dict[str, Any]) -> Optional[str]:
    for key in ("label", "classification", "class", "target"):
        value = truth.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _resolve_regression_truth(truth: Dict[str, Any]) -> Dict[str, Any]:
    regression = truth.get("regression")
    if isinstance(regression, dict):
        return regression
    values = truth.get("values")
    if isinstance(values, dict):
        return values
    if "value" in truth:
        output_name = str(truth.get("output_name") or "value")
        return {output_name: truth.get("value")}
    return {}


def build_multiclass_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        truth = row.get("truth") if isinstance(row.get("truth"), dict) else {}
        pred = row.get("pred") if isinstance(row.get("pred"), dict) else {}
        actual = _resolve_multiclass_label(truth)
        predicted = pred.get("predicted_label")
        out.append(
            {
                "eid": row.get("eid"),
                "disorder": row.get("disorder", "UNKNOWN"),
                "actual": actual,
                "predicted": predicted,
                "probabilities": pred.get("probabilities") if isinstance(pred.get("probabilities"), dict) else {},
                "predicted_probability": None,
                "status": row.get("status"),
            }
        )
    return out


def build_regression_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    out: List[Dict[str, Any]] = []
    expected_outputs: set[str] = set()

    for row in rows:
        truth = row.get("truth") if isinstance(row.get("truth"), dict) else {}
        pred = row.get("pred") if isinstance(row.get("pred"), dict) else {}
        actual_values = _resolve_regression_truth(truth)
        predicted_values = pred.get("regression_values") if isinstance(pred.get("regression_values"), dict) else {}
        expected_outputs.update(str(k) for k in actual_values.keys())
        out.append(
            {
                "eid": row.get("eid"),
                "disorder": row.get("disorder", "UNKNOWN"),
                "actual_values": actual_values,
                "predicted_values": predicted_values,
                "status": row.get("status"),
            }
        )

    return out, sorted(expected_outputs)


def build_hierarchical_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        truth = row.get("truth") if isinstance(row.get("truth"), dict) else {}
        pred = row.get("pred") if isinstance(row.get("pred"), dict) else {}
        truth_nodes = truth.get("nodes") if isinstance(truth.get("nodes"), dict) else {}
        pred_nodes = pred.get("nodes") if isinstance(pred.get("nodes"), dict) else {}
        out.append(
            {
                "eid": row.get("eid"),
                "disorder": row.get("disorder", "UNKNOWN"),
                "truth_nodes": truth_nodes,
                "pred_nodes": pred_nodes,
                "status": row.get("status"),
            }
        )
    return out


def _compute_non_binary_metrics(prediction_type: str, rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if prediction_type == "multiclass":
        eval_rows = build_multiclass_rows(rows)
        metrics = compute_multiclass_metrics(eval_rows)
        return metrics, {"eval_rows": eval_rows}

    if prediction_type in {"regression_univariate", "regression_multivariate"}:
        eval_rows, expected_outputs = build_regression_rows(rows)
        metrics = compute_regression_metrics(eval_rows, expected_outputs=expected_outputs)
        return metrics, {"eval_rows": eval_rows, "expected_outputs": expected_outputs}

    if prediction_type == "hierarchical":
        eval_rows = build_hierarchical_rows(rows)
        metrics = compute_hierarchical_metrics(eval_rows)
        return metrics, {"eval_rows": eval_rows}

    raise ValueError(f"Unsupported non-binary prediction_type: {prediction_type}")


def _plot_non_binary_metrics(
    *,
    prediction_type: str,
    metrics: Dict[str, Any],
    output_dir: str,
    prefix: str,
    title_suffix: str,
    detailed: bool = False,
) -> List[str]:
    out_files: List[str] = []

    if prediction_type == "multiclass":
        cm_path = os.path.join(output_dir, f"{prefix}multiclass_confusion_matrix.png")
        plot_multiclass_confusion_matrix(metrics, f"Multiclass confusion matrix{title_suffix}", cm_path)
        out_files.append(cm_path)

        pc_path = os.path.join(output_dir, f"{prefix}multiclass_per_class_metrics.png")
        plot_multiclass_per_class(metrics, f"Multiclass per-class metrics{title_suffix}", pc_path)
        out_files.append(pc_path)

        cal_path = os.path.join(output_dir, f"{prefix}multiclass_confidence_calibration.png")
        plot_multiclass_confidence_calibration(metrics, f"Top-label confidence calibration{title_suffix}", cal_path)
        out_files.append(cal_path)

        if detailed:
            top_conf_path = os.path.join(output_dir, f"{prefix}multiclass_top_confusions.png")
            plot_multiclass_top_confusions(metrics, f"Top multiclass confusions{title_suffix}", top_conf_path)
            out_files.append(top_conf_path)

            diag_path = os.path.join(output_dir, f"{prefix}multiclass_confidence_diagnostics.png")
            plot_multiclass_confidence_diagnostics(metrics, f"Multiclass confidence diagnostics{title_suffix}", diag_path)
            out_files.append(diag_path)

            dist_path = os.path.join(output_dir, f"{prefix}multiclass_label_distribution.png")
            plot_multiclass_label_distribution(metrics, f"Multiclass label prevalence{title_suffix}", dist_path)
            out_files.append(dist_path)

    elif prediction_type in {"regression_univariate", "regression_multivariate"}:
        parity_path = os.path.join(output_dir, f"{prefix}regression_parity.png")
        plot_regression_parity(metrics, f"Regression parity plots{title_suffix}", parity_path)
        out_files.append(parity_path)

        err_path = os.path.join(output_dir, f"{prefix}regression_error_bars.png")
        plot_regression_error_bars(metrics, f"Regression error by output{title_suffix}", err_path)
        out_files.append(err_path)

        if detailed:
            residual_dist_path = os.path.join(output_dir, f"{prefix}regression_residual_distribution.png")
            plot_regression_residual_distribution(metrics, f"Regression residual distributions{title_suffix}", residual_dist_path)
            out_files.append(residual_dist_path)

            residual_vs_true_path = os.path.join(output_dir, f"{prefix}regression_residual_vs_true.png")
            plot_regression_residuals_vs_truth(metrics, f"Residuals vs true values{title_suffix}", residual_vs_true_path)
            out_files.append(residual_vs_true_path)

            top_errors_path = os.path.join(output_dir, f"{prefix}regression_top_errors.png")
            plot_regression_top_errors(metrics, f"Largest regression errors{title_suffix}", top_errors_path)
            out_files.append(top_errors_path)

    elif prediction_type == "hierarchical":
        node_path = os.path.join(output_dir, f"{prefix}hierarchical_node_scores.png")
        plot_hierarchical_node_scores(metrics, f"Hierarchical node performance{title_suffix}", node_path)
        out_files.append(node_path)

        mode_path = os.path.join(output_dir, f"{prefix}hierarchical_mode_distribution.png")
        plot_hierarchical_mode_distribution(metrics, f"Hierarchical node-type distribution{title_suffix}", mode_path)
        out_files.append(mode_path)

        if detailed:
            coverage_path = os.path.join(output_dir, f"{prefix}hierarchical_node_coverage.png")
            plot_hierarchical_node_coverage(metrics, f"Hierarchical node coverage{title_suffix}", coverage_path)
            out_files.append(coverage_path)

            heatmap_path = os.path.join(output_dir, f"{prefix}hierarchical_metric_heatmap.png")
            plot_hierarchical_metric_heatmap(metrics, f"Hierarchical node metric matrix{title_suffix}", heatmap_path)
            out_files.append(heatmap_path)

    return out_files


def _write_annotation_contract_text(contract: Dict[str, Any], output_path: str) -> None:
    lines: List[str] = [
        "COMPASS Annotation Contract Report",
        "=================================",
        "",
        f"Prediction type: {contract.get('prediction_type')}",
        f"Rows checked:    {contract.get('n_rows')}",
        f"Valid rows:      {contract.get('n_valid_rows')}",
        f"Invalid rows:    {contract.get('n_invalid_rows')}",
        f"Validity rate:   {float(contract.get('validity_rate') or 0.0):.1%}",
        "",
    ]

    issue_counts = contract.get("issue_counts") if isinstance(contract.get("issue_counts"), dict) else {}
    if issue_counts:
        lines.append("Issue counts:")
        for key in sorted(issue_counts.keys()):
            lines.append(f"- {key}: {int(issue_counts[key] or 0)}")
        lines.append("")

    examples = contract.get("issue_examples") if isinstance(contract.get("issue_examples"), list) else []
    if examples:
        lines.append("Issue examples:")
        for row in examples[:20]:
            lines.append(
                f"- eid={row.get('eid')} | code={row.get('code')} | detail={row.get('detail')}"
            )
        lines.append("")

    for key in ("class_distribution", "output_presence", "node_mode_distribution"):
        block = contract.get(key)
        if isinstance(block, dict) and block:
            lines.append(f"{key}:")
            for name in sorted(block.keys()):
                lines.append(f"- {name}: {block[name]}")
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _annotation_contract_error_message(contract: Dict[str, Any], prediction_type: str) -> str:
    issue_counts = contract.get("issue_counts") if isinstance(contract.get("issue_counts"), dict) else {}
    if issue_counts:
        top = sorted(issue_counts.items(), key=lambda kv: int(kv[1] or 0), reverse=True)[:5]
        compact = ", ".join(f"{code}={int(count)}" for code, count in top)
        return (
            f"Annotation contract validation found zero valid rows for '{prediction_type}'. "
            f"Top issues: {compact}. "
            "Check annotation_templates/examples for the expected JSON structure."
        )
    return (
        f"Annotation contract validation found zero valid rows for '{prediction_type}'. "
        "Check annotation_templates/examples for the expected JSON structure."
    )


def _hierarchical_schema_error_message(contract: Dict[str, Any]) -> str:
    issue_counts = contract.get("issue_counts") if isinstance(contract.get("issue_counts"), dict) else {}
    keys = [
        "hierarchy_node_set_mismatch",
        "hierarchy_node_mode_mismatch",
        "hierarchy_regression_output_mismatch",
    ]
    present = [f"{k}={int(issue_counts.get(k) or 0)}" for k in keys if int(issue_counts.get(k) or 0) > 0]
    details = ", ".join(present) if present else "schema mismatch detected"
    return (
        "Hierarchical annotation schema mismatch across participants. "
        f"{details}. "
        "Ensure all EIDs share the same node IDs, node modes, and regression output keys per node."
    )


def _write_non_binary_text(
    *,
    prediction_type: str,
    rows: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: str,
    title: str,
) -> None:
    if prediction_type == "multiclass":
        write_multiclass_text_report(rows=rows, metrics=metrics, output_path=output_path, title=title)
    elif prediction_type in {"regression_univariate", "regression_multivariate"}:
        write_regression_text_report(
            rows=rows,
            metrics=metrics,
            output_path=output_path,
            prediction_type=prediction_type,
            title=title,
        )
    else:
        write_hierarchical_text_report(rows=rows, metrics=metrics, output_path=output_path, title=title)


def run_metrics_workflow(
    *,
    results_dir: str,
    output_dir: str,
    prediction_type: str,
    targets_file: str = "",
    annotations_json: str = "",
    disorder_groups: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    prediction_type = normalize_prediction_type(str(prediction_type or "binary"))
    if prediction_type not in SUPPORTED_PREDICTION_TYPES:
        accepted = ", ".join(sorted(ACCEPTED_PREDICTION_TYPES))
        raise ValueError(f"Unsupported prediction_type: {prediction_type}. Accepted values: {accepted}")

    summary: Dict[str, Any] = {
        "prediction_type": prediction_type,
        "outputs": [],
        "groups": {},
    }

    disorders = [x for x in list(disorder_groups or []) if str(x).strip()]

    if prediction_type == "binary":
        if not targets_file:
            raise ValueError("targets_file is required for binary validation")

        rows = collect_binary_results(results_dir=results_dir, targets_file=targets_file)
        if not rows:
            raise ValueError("No binary rows found for validation")

        metrics = compute_binary_metrics(rows)
        summary["integrated"] = {"n_rows": len(rows), "metrics": metrics}

        cm_path = os.path.join(output_dir, "integrated_confusion_matrix.png")
        plot_binary_confusion_matrix(metrics, "Integrated binary confusion matrix", cm_path)
        summary["outputs"].append(cm_path)

        metrics_json = os.path.join(output_dir, "binary_metrics_integrated.json")
        write_json({"prediction_type": prediction_type, "n_rows": len(rows), "metrics": metrics}, metrics_json)
        summary["outputs"].append(metrics_json)

        if disorders:
            for disorder in disorders:
                d_rows = collect_binary_results(results_dir=results_dir, targets_file=targets_file, disorder_filter=[disorder])
                if not d_rows:
                    continue
                d_metrics = compute_binary_metrics(d_rows)
                safe = _safe_slug(disorder)
                path = os.path.join(output_dir, f"{safe}_confusion_matrix.png")
                plot_binary_confusion_matrix(d_metrics, f"Binary confusion matrix — {disorder}", path)
                json_path = os.path.join(output_dir, f"{safe}_binary_metrics.json")
                write_json({"prediction_type": prediction_type, "group": disorder, "n_rows": len(d_rows), "metrics": d_metrics}, json_path)
                summary["groups"][disorder] = {"n_rows": len(d_rows), "metrics": d_metrics, "outputs": [path, json_path]}
                summary["outputs"].extend([path, json_path])

        return summary

    # Non-binary workflows.
    if not annotations_json:
        raise ValueError("annotations_json is required for non-binary validation")

    annotations = load_generalized_annotations(annotations_json)
    rows = collect_generalized_rows(results_dir=results_dir, annotations=annotations)
    if disorders:
        rows = [r for r in rows if str(r.get("disorder") or "") in set(disorders)]

    if not rows:
        raise ValueError("No overlapping prediction/annotation rows found")

    contract = summarize_annotation_contract(rows=rows, prediction_type=prediction_type)
    if int(contract.get("n_valid_rows") or 0) <= 0:
        raise ValueError(_annotation_contract_error_message(contract, prediction_type))
    if prediction_type == "hierarchical":
        issue_counts = contract.get("issue_counts") if isinstance(contract.get("issue_counts"), dict) else {}
        if any(
            int(issue_counts.get(code) or 0) > 0
            for code in (
                "hierarchy_node_set_mismatch",
                "hierarchy_node_mode_mismatch",
                "hierarchy_regression_output_mismatch",
            )
        ):
            raise ValueError(_hierarchical_schema_error_message(contract))

    contract_json = os.path.join(output_dir, f"annotation_contract_{prediction_type}.json")
    write_json(contract, contract_json)
    summary["outputs"].append(contract_json)

    metrics, context = _compute_non_binary_metrics(prediction_type, rows)

    payload = {
        "prediction_type": prediction_type,
        "prediction_type_label": PREDICTION_TYPE_LABELS.get(prediction_type, prediction_type),
        "n_rows": len(rows),
        "n_success": sum(1 for r in rows if r.get("status") == "SUCCESS"),
        "n_failed": sum(1 for r in rows if r.get("status") != "SUCCESS"),
        "annotation_contract": contract,
        "metrics": metrics,
        "xai_status": {
            "status": "skipped",
            "reason": "XAI currently supports binary classification only.",
        },
    }

    metrics_path = os.path.join(output_dir, f"{prediction_type}_metrics.json")
    write_json(payload, metrics_path)
    summary["outputs"].append(metrics_path)
    summary["integrated"] = payload

    summary["outputs"].extend(
        _plot_non_binary_metrics(
            prediction_type=prediction_type,
            metrics=metrics,
            output_dir=output_dir,
            prefix="",
            title_suffix="",
            detailed=False,
        )
    )

    if disorders:
        for disorder in disorders:
            d_rows = [r for r in rows if str(r.get("disorder") or "") == disorder]
            if not d_rows:
                continue
            d_metrics, _ = _compute_non_binary_metrics(prediction_type, d_rows)
            d_contract = summarize_annotation_contract(rows=d_rows, prediction_type=prediction_type)
            safe = _safe_slug(disorder)
            d_payload = {
                "prediction_type": prediction_type,
                "group": disorder,
                "n_rows": len(d_rows),
                "annotation_contract": d_contract,
                "metrics": d_metrics,
            }
            d_json = os.path.join(output_dir, f"{safe}_{prediction_type}_metrics.json")
            write_json(d_payload, d_json)
            d_contract_json = os.path.join(output_dir, f"{safe}_{prediction_type}_annotation_contract.json")
            write_json(d_contract, d_contract_json)
            plots = _plot_non_binary_metrics(
                prediction_type=prediction_type,
                metrics=d_metrics,
                output_dir=output_dir,
                prefix=f"{safe}_",
                title_suffix=f" — {disorder}",
                detailed=False,
            )
            summary["groups"][disorder] = {
                "n_rows": len(d_rows),
                "metrics": d_metrics,
                "annotation_contract": d_contract,
                "outputs": [d_json, d_contract_json, *plots],
            }
            summary["outputs"].extend([d_json, d_contract_json, *plots])

    return summary


def run_detailed_workflow(
    *,
    results_dir: str,
    output_dir: str,
    prediction_type: str,
    targets_file: str = "",
    annotations_json: str = "",
    disorder_groups: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    prediction_type = normalize_prediction_type(str(prediction_type or "binary"))
    if prediction_type not in SUPPORTED_PREDICTION_TYPES:
        accepted = ", ".join(sorted(ACCEPTED_PREDICTION_TYPES))
        raise ValueError(f"Unsupported prediction_type: {prediction_type}. Accepted values: {accepted}")

    disorders = [x for x in list(disorder_groups or []) if str(x).strip()]

    summary: Dict[str, Any] = {
        "prediction_type": prediction_type,
        "outputs": [],
        "groups": {},
    }

    if prediction_type == "binary":
        if not targets_file:
            raise ValueError("targets_file is required for binary detailed analysis")

        rows = collect_binary_results(results_dir=results_dir, targets_file=targets_file)
        if not rows:
            raise ValueError("No binary rows found for detailed analysis")

        metrics = compute_binary_metrics(rows)
        integrated_json = os.path.join(output_dir, "detailed_analysis_binary.json")
        write_json({"prediction_type": prediction_type, "n_rows": len(rows), "metrics": metrics}, integrated_json)
        summary["outputs"].append(integrated_json)
        summary["integrated"] = {"prediction_type": prediction_type, "n_rows": len(rows), "metrics": metrics}

        txt_path = os.path.join(output_dir, "detailed_analysis.txt")
        write_binary_text_report(rows=rows, metrics=metrics, output_path=txt_path, title="Integrated Binary Detailed Analysis")
        summary["outputs"].append(txt_path)

        plot_binary_composite_vs_accuracy(rows, "Composite score vs prediction correctness", os.path.join(output_dir, "composite_vs_accuracy.png"))
        plot_binary_probability_calibration(rows, "Binary probability calibration", os.path.join(output_dir, "probability_calibration.png"))
        plot_binary_iteration_improvement(rows, "Composite improvement across iterations", os.path.join(output_dir, "iteration_improvement.png"))
        plot_binary_verdict_accuracy(rows, "Prediction accuracy by critic verdict", os.path.join(output_dir, "verdict_accuracy.png"))
        summary["outputs"].extend(
            [
                os.path.join(output_dir, "composite_vs_accuracy.png"),
                os.path.join(output_dir, "probability_calibration.png"),
                os.path.join(output_dir, "iteration_improvement.png"),
                os.path.join(output_dir, "verdict_accuracy.png"),
            ]
        )

        if disorders:
            for disorder in disorders:
                d_rows = collect_binary_results(results_dir=results_dir, targets_file=targets_file, disorder_filter=[disorder])
                if not d_rows:
                    continue
                d_metrics = compute_binary_metrics(d_rows)
                safe = _safe_slug(disorder)
                d_txt = os.path.join(output_dir, f"detailed_analysis_{safe}.txt")
                write_binary_text_report(rows=d_rows, metrics=d_metrics, output_path=d_txt, title=f"Binary Detailed Analysis — {disorder}")
                d_json = os.path.join(output_dir, f"detailed_analysis_{safe}.json")
                write_json({"prediction_type": prediction_type, "group": disorder, "n_rows": len(d_rows), "metrics": d_metrics}, d_json)
                summary["groups"][disorder] = {"n_rows": len(d_rows), "metrics": d_metrics, "outputs": [d_txt, d_json]}
                summary["outputs"].extend([d_txt, d_json])

        return summary

    # Non-binary detailed workflows.
    if not annotations_json:
        raise ValueError("annotations_json is required for non-binary detailed analysis")

    annotations = load_generalized_annotations(annotations_json)
    rows = collect_generalized_rows(results_dir=results_dir, annotations=annotations)
    if disorders:
        rows = [r for r in rows if str(r.get("disorder") or "") in set(disorders)]

    if not rows:
        raise ValueError("No overlapping prediction/annotation rows found")

    contract = summarize_annotation_contract(rows=rows, prediction_type=prediction_type)
    if int(contract.get("n_valid_rows") or 0) <= 0:
        raise ValueError(_annotation_contract_error_message(contract, prediction_type))
    if prediction_type == "hierarchical":
        issue_counts = contract.get("issue_counts") if isinstance(contract.get("issue_counts"), dict) else {}
        if any(
            int(issue_counts.get(code) or 0) > 0
            for code in (
                "hierarchy_node_set_mismatch",
                "hierarchy_node_mode_mismatch",
                "hierarchy_regression_output_mismatch",
            )
        ):
            raise ValueError(_hierarchical_schema_error_message(contract))

    contract_json = os.path.join(output_dir, f"detailed_annotation_contract_{prediction_type}.json")
    write_json(contract, contract_json)
    summary["outputs"].append(contract_json)
    contract_txt = os.path.join(output_dir, f"detailed_annotation_contract_{prediction_type}.txt")
    _write_annotation_contract_text(contract, contract_txt)
    summary["outputs"].append(contract_txt)

    metrics, context = _compute_non_binary_metrics(prediction_type, rows)
    payload = {
        "prediction_type": prediction_type,
        "prediction_type_label": PREDICTION_TYPE_LABELS.get(prediction_type, prediction_type),
        "n_rows": len(rows),
        "n_success": sum(1 for r in rows if r.get("status") == "SUCCESS"),
        "n_failed": sum(1 for r in rows if r.get("status") != "SUCCESS"),
        "annotation_contract": contract,
        "metrics": metrics,
        "xai_status": {
            "status": "skipped",
            "reason": "XAI currently supports binary classification only.",
        },
    }

    json_path = os.path.join(output_dir, f"detailed_analysis_{prediction_type}.json")
    write_json(payload, json_path)
    summary["outputs"].append(json_path)
    summary["integrated"] = payload

    eval_rows_path = os.path.join(output_dir, f"detailed_rows_{prediction_type}.json")
    write_json(
        {
            "prediction_type": prediction_type,
            "n_rows": len(context.get("eval_rows") or []),
            "rows": context.get("eval_rows") if isinstance(context.get("eval_rows"), list) else [],
        },
        eval_rows_path,
    )
    summary["outputs"].append(eval_rows_path)

    txt_path = os.path.join(output_dir, f"detailed_analysis_{prediction_type}.txt")
    _write_non_binary_text(
        prediction_type=prediction_type,
        rows=rows,
        metrics=metrics,
        output_path=txt_path,
        title=f"{PREDICTION_TYPE_LABELS.get(prediction_type, prediction_type)} detailed analysis",
    )
    summary["outputs"].append(txt_path)

    summary["outputs"].extend(
        _plot_non_binary_metrics(
            prediction_type=prediction_type,
            metrics=metrics,
            output_dir=output_dir,
            prefix="detailed_",
            title_suffix="",
            detailed=True,
        )
    )

    if disorders:
        for disorder in disorders:
            d_rows = [r for r in rows if str(r.get("disorder") or "") == disorder]
            if not d_rows:
                continue
            d_contract = summarize_annotation_contract(rows=d_rows, prediction_type=prediction_type)
            d_metrics, d_context = _compute_non_binary_metrics(prediction_type, d_rows)
            safe = _safe_slug(disorder)
            d_json = os.path.join(output_dir, f"detailed_analysis_{prediction_type}_{safe}.json")
            d_txt = os.path.join(output_dir, f"detailed_analysis_{prediction_type}_{safe}.txt")
            write_json(
                {
                    "prediction_type": prediction_type,
                    "group": disorder,
                    "n_rows": len(d_rows),
                    "annotation_contract": d_contract,
                    "metrics": d_metrics,
                },
                d_json,
            )
            _write_non_binary_text(
                prediction_type=prediction_type,
                rows=d_rows,
                metrics=d_metrics,
                output_path=d_txt,
                title=f"{PREDICTION_TYPE_LABELS.get(prediction_type, prediction_type)} — {disorder}",
            )
            d_contract_json = os.path.join(output_dir, f"detailed_annotation_contract_{prediction_type}_{safe}.json")
            write_json(d_contract, d_contract_json)
            d_contract_txt = os.path.join(output_dir, f"detailed_annotation_contract_{prediction_type}_{safe}.txt")
            _write_annotation_contract_text(d_contract, d_contract_txt)
            d_rows_json = os.path.join(output_dir, f"detailed_rows_{prediction_type}_{safe}.json")
            write_json(
                {
                    "prediction_type": prediction_type,
                    "group": disorder,
                    "n_rows": len(d_context.get("eval_rows") or []),
                    "rows": d_context.get("eval_rows") if isinstance(d_context.get("eval_rows"), list) else [],
                },
                d_rows_json,
            )
            plots = _plot_non_binary_metrics(
                prediction_type=prediction_type,
                metrics=d_metrics,
                output_dir=output_dir,
                prefix=f"detailed_{safe}_",
                title_suffix=f" — {disorder}",
                detailed=True,
            )
            summary["groups"][disorder] = {
                "n_rows": len(d_rows),
                "metrics": d_metrics,
                "annotation_contract": d_contract,
                "outputs": [d_json, d_txt, d_contract_json, d_contract_txt, d_rows_json, *plots],
            }
            summary["outputs"].extend([d_json, d_txt, d_contract_json, d_contract_txt, d_rows_json, *plots])

    return summary

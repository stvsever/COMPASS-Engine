"""Annotation contract checks for mode-aware annotated validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence

from .io_utils import safe_float


def _extract_label(payload: Dict[str, Any]) -> str:
    for key in ("label", "classification", "class", "target", "predicted_label", "class_label"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("regression"), dict):
        return payload["regression"]
    if isinstance(payload.get("values"), dict):
        return payload["values"]
    if "value" in payload:
        key = str(payload.get("output_name") or "value")
        return {key: payload.get("value")}
    return {}


def _numeric_values(values: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in values.items():
        fv = safe_float(value)
        if fv is not None:
            out[str(key)] = float(fv)
    return out


def _append_issue(issues: Counter, examples: List[Dict[str, Any]], *, eid: str, code: str, detail: str) -> None:
    issues[code] += 1
    if len(examples) < 30:
        examples.append({"eid": eid, "code": code, "detail": detail})


def summarize_annotation_contract(
    *,
    rows: Sequence[Dict[str, Any]],
    prediction_type: str,
) -> Dict[str, Any]:
    """
    Validate annotation payload quality for the selected prediction type.

    `rows` should be the generalized overlap rows from `collect_generalized_rows`.
    """
    issues: Counter = Counter()
    examples: List[Dict[str, Any]] = []

    class_distribution: Counter = Counter()
    output_presence: Counter = Counter()
    node_mode_distribution: Counter = Counter()
    reference_node_ids: set[str] | None = None
    reference_node_modes: Dict[str, str] = {}
    reference_regression_outputs: Dict[str, set[str]] = {}

    n_valid = 0

    for row in rows:
        eid = str(row.get("eid") or "UNKNOWN")
        truth = row.get("truth") if isinstance(row.get("truth"), dict) else {}
        if not truth:
            _append_issue(issues, examples, eid=eid, code="missing_truth_payload", detail="truth payload is absent or invalid")
            continue

        is_valid = True

        if prediction_type == "multiclass":
            label = _extract_label(truth)
            if not label:
                _append_issue(issues, examples, eid=eid, code="missing_multiclass_label", detail="expected one class label")
                is_valid = False
            else:
                class_distribution[label] += 1

        elif prediction_type == "regression_univariate":
            values = _numeric_values(_extract_values(truth))
            if len(values) != 1:
                _append_issue(
                    issues,
                    examples,
                    eid=eid,
                    code="invalid_univariate_values",
                    detail=f"expected exactly one numeric output, got {len(values)}",
                )
                is_valid = False
            else:
                output_presence.update(values.keys())

        elif prediction_type == "regression_multivariate":
            values = _numeric_values(_extract_values(truth))
            if len(values) < 2:
                _append_issue(
                    issues,
                    examples,
                    eid=eid,
                    code="invalid_multivariate_values",
                    detail=f"expected at least two numeric outputs, got {len(values)}",
                )
                is_valid = False
            else:
                output_presence.update(values.keys())

        elif prediction_type == "hierarchical":
            nodes = truth.get("nodes") if isinstance(truth.get("nodes"), dict) else {}
            if not nodes:
                _append_issue(issues, examples, eid=eid, code="missing_hierarchy_nodes", detail="truth.nodes is missing or empty")
                is_valid = False
            else:
                n_valid_nodes = 0
                row_node_ids: set[str] = set()
                row_node_modes: Dict[str, str] = {}
                row_regression_outputs: Dict[str, set[str]] = {}
                for node_id, node_truth in nodes.items():
                    if not isinstance(node_truth, dict):
                        _append_issue(
                            issues,
                            examples,
                            eid=eid,
                            code="invalid_hierarchy_node_payload",
                            detail=f"node '{node_id}' is not a dict",
                        )
                        continue
                    row_node_ids.add(str(node_id))
                    mode = str(node_truth.get("mode") or "").strip()
                    if not mode:
                        # Infer from payload structure.
                        if _numeric_values(_extract_values(node_truth)):
                            mode = "regression"
                        elif _extract_label(node_truth):
                            mode = "classification"
                    if not mode:
                        _append_issue(
                            issues,
                            examples,
                            eid=eid,
                            code="missing_hierarchy_node_mode",
                            detail=f"node '{node_id}' has no mode and cannot be inferred",
                        )
                        continue

                    row_node_modes[str(node_id)] = mode
                    node_mode_distribution[mode] += 1
                    if "classification" in mode:
                        if not _extract_label(node_truth):
                            _append_issue(
                                issues,
                                examples,
                                eid=eid,
                                code="missing_hierarchy_class_label",
                                detail=f"node '{node_id}' classification label missing",
                            )
                            continue
                    elif "regression" in mode:
                        values = _numeric_values(_extract_values(node_truth))
                        if not values:
                            _append_issue(
                                issues,
                                examples,
                                eid=eid,
                                code="missing_hierarchy_regression_values",
                                detail=f"node '{node_id}' regression values missing",
                            )
                            continue
                        row_regression_outputs[str(node_id)] = set(str(k) for k in values.keys())
                        output_presence.update(values.keys())
                    n_valid_nodes += 1

                if n_valid_nodes == 0:
                    is_valid = False
                elif reference_node_ids is None:
                    reference_node_ids = set(row_node_ids)
                    reference_node_modes = dict(row_node_modes)
                    reference_regression_outputs = {k: set(v) for k, v in row_regression_outputs.items()}
                else:
                    missing_nodes = sorted(reference_node_ids - row_node_ids)
                    extra_nodes = sorted(row_node_ids - reference_node_ids)
                    if missing_nodes or extra_nodes:
                        _append_issue(
                            issues,
                            examples,
                            eid=eid,
                            code="hierarchy_node_set_mismatch",
                            detail=f"missing={missing_nodes or []}, extra={extra_nodes or []}",
                        )
                        is_valid = False

                    for node_id in sorted(reference_node_ids & row_node_ids):
                        ref_mode = str(reference_node_modes.get(node_id) or "")
                        row_mode = str(row_node_modes.get(node_id) or "")
                        if ref_mode and row_mode and ref_mode != row_mode:
                            _append_issue(
                                issues,
                                examples,
                                eid=eid,
                                code="hierarchy_node_mode_mismatch",
                                detail=f"node='{node_id}', expected_mode='{ref_mode}', got_mode='{row_mode}'",
                            )
                            is_valid = False

                    for node_id, ref_outputs in reference_regression_outputs.items():
                        row_outputs = row_regression_outputs.get(node_id, set())
                        if ref_outputs != row_outputs:
                            _append_issue(
                                issues,
                                examples,
                                eid=eid,
                                code="hierarchy_regression_output_mismatch",
                                detail=(
                                    f"node='{node_id}', expected_outputs={sorted(ref_outputs)}, "
                                    f"got_outputs={sorted(row_outputs)}"
                                ),
                            )
                            is_valid = False

        if is_valid:
            n_valid += 1

    n_rows = len(rows)
    n_invalid = max(0, n_rows - n_valid)

    summary: Dict[str, Any] = {
        "prediction_type": prediction_type,
        "n_rows": n_rows,
        "n_valid_rows": n_valid,
        "n_invalid_rows": n_invalid,
        "validity_rate": (float(n_valid) / float(n_rows)) if n_rows > 0 else 0.0,
        "issue_counts": dict(sorted(issues.items(), key=lambda x: x[0])),
        "issue_examples": examples,
    }

    if class_distribution:
        summary["class_distribution"] = dict(sorted(class_distribution.items(), key=lambda x: x[0]))
    if output_presence:
        summary["output_presence"] = dict(sorted(output_presence.items(), key=lambda x: x[0]))
    if node_mode_distribution:
        summary["node_mode_distribution"] = dict(sorted(node_mode_distribution.items(), key=lambda x: x[0]))
    if prediction_type == "hierarchical" and reference_node_ids is not None:
        summary["hierarchy_reference_schema"] = {
            "node_ids": sorted(reference_node_ids),
            "node_modes": dict(sorted(reference_node_modes.items(), key=lambda x: x[0])),
            "regression_outputs_by_node": {
                k: sorted(v) for k, v in sorted(reference_regression_outputs.items(), key=lambda x: x[0])
            },
        }

    return summary

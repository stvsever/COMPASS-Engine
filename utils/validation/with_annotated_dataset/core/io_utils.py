"""I/O and extraction helpers for annotated validation workflows."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _strip_leading_zeros(value: str) -> str:
    if not value:
        return value
    if value.isdigit():
        stripped = value.lstrip("0")
        return stripped if stripped else "0"
    return value


def normalized_identifier_variants(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []

    out: List[str] = []

    def _add(x: str) -> None:
        if x and x not in out:
            out.append(x)

    _add(text)
    _add(text.upper())

    digits = re.sub(r"\D", "", text)
    if digits:
        _add(digits)
        _add(_strip_leading_zeros(digits))

    if text.isdigit():
        _add(_strip_leading_zeros(text))

    return out


def extract_participant_id_candidates(dirname: str) -> List[str]:
    name = str(dirname or "").strip()
    if not name:
        return []

    candidates: List[str] = []

    def _add(x: str) -> None:
        if x and x not in candidates:
            candidates.append(x)

    _add(name)

    patterns = [
        r"ID(\d+)",
        r"SUBJ_(\d+)",
        r"participant_(\d+)",
        r"(\d+)$",
    ]
    for pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            _add(m.group(1))

    # Keep SUBJ token variants too, often used in pseudo runs.
    m_subj = re.search(r"(SUBJ_[A-Za-z0-9_]+)", name, flags=re.IGNORECASE)
    if m_subj:
        _add(m_subj.group(1))

    return candidates


def resolve_identifier(candidates: Sequence[str], available_keys: Iterable[str]) -> Optional[str]:
    keys = [str(k) for k in available_keys]
    key_set = set(keys)

    expanded: List[str] = []
    for cand in candidates:
        for v in normalized_identifier_variants(cand):
            if v not in expanded:
                expanded.append(v)

    for c in expanded:
        if c in key_set:
            return c

    # Case-insensitive fallback.
    lower_map = {k.lower(): k for k in keys}
    for c in expanded:
        hit = lower_map.get(c.lower())
        if hit is not None:
            return hit

    return None


def _normalize_binary_truth_label(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if "CASE" in text and "CONTROL" not in text:
        return "CASE"
    if "CONTROL" in text:
        return "CONTROL"
    return None


def _extract_disorder_from_label_text(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"\(([^)]+)\)", text)
    return match.group(1).strip() if match else "UNKNOWN"


def _extract_binary_truth_disorder(row: Dict[str, Any], label_source: Any) -> str:
    for key in ("disorder", "group", "cohort", "phenotype_group"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return _extract_disorder_from_label_text(label_source)


def _register_ground_truth_row(
    gt: Dict[str, Dict[str, str]],
    *,
    eid_raw: str,
    label: str,
    disorder: str,
) -> None:
    for variant in normalized_identifier_variants(eid_raw):
        if variant and variant not in gt:
            gt[variant] = {"label": label, "disorder": disorder or "UNKNOWN"}


def _load_ground_truth_from_json_payload(payload: Any) -> Dict[str, Dict[str, str]]:
    gt: Dict[str, Dict[str, str]] = {}

    if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        payload = payload["annotations"]

    if isinstance(payload, dict):
        for raw_key, value in payload.items():
            row = value if isinstance(value, dict) else {"label": value}
            label_source = (
                row.get("label")
                or row.get("classification")
                or row.get("class")
                or row.get("target")
                or row.get("value")
            )
            label = _normalize_binary_truth_label(label_source)
            if label is None:
                continue
            disorder = _extract_binary_truth_disorder(row, label_source)
            _register_ground_truth_row(gt, eid_raw=str(raw_key), label=label, disorder=disorder)
        return gt

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            key = row.get("eid") or row.get("participant_id") or row.get("id")
            if key is None:
                continue
            label_source = (
                row.get("label")
                or row.get("classification")
                or row.get("class")
                or row.get("target")
                or row.get("value")
            )
            label = _normalize_binary_truth_label(label_source)
            if label is None:
                continue
            disorder = _extract_binary_truth_disorder(row, label_source)
            _register_ground_truth_row(gt, eid_raw=str(key), label=label, disorder=disorder)
        return gt

    raise ValueError("Unsupported JSON structure for binary ground truth")


def load_ground_truth(targets_file: str) -> Dict[str, Dict[str, str]]:
    """Parse binary targets file into {eid: {label, disorder}}.

    JSON-only contract:
    - dict keyed by participant id (value=object or label string), or
    - list of row objects, or
    - {"annotations": [...]}
    """
    path = Path(targets_file)
    with open(path, "r") as f:
        raw = f.read()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Binary targets input must be valid JSON. "
            "Plain-text formats like '01|CASE (...)' are not supported. "
            "Use annotation_templates/examples/binary_targets_example.json."
        ) from exc

    try:
        gt = _load_ground_truth_from_json_payload(payload)
    except ValueError as exc:
        raise ValueError(
            "Invalid binary targets JSON structure. "
            "Expected a dict/list (optionally {'annotations': [...]}) with participant IDs "
            "and CASE/CONTROL labels. See annotation_templates/examples/binary_targets_example.json."
        ) from exc

    if not gt:
        raise ValueError(
            "Binary targets JSON parsed successfully but no valid CASE/CONTROL rows were found. "
            "Each row needs a participant id and a CASE/CONTROL label."
        )

    return gt


def load_generalized_annotations(path: str) -> Dict[str, Dict[str, Any]]:
    """Load generalized annotations for multiclass/regression/hierarchical validation."""
    payload: Any
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if isinstance(payload.get("annotations"), list):
            payload = payload["annotations"]
        else:
            out: Dict[str, Dict[str, Any]] = {}
            for raw_key, value in payload.items():
                key = str(raw_key)
                row = value if isinstance(value, dict) else {"value": value}
                for variant in normalized_identifier_variants(key):
                    if variant and variant not in out:
                        out[variant] = row
            return out

    if isinstance(payload, list):
        out = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            key = row.get("eid") or row.get("participant_id") or row.get("id")
            if key is None:
                continue
            entry = dict(row)
            entry.pop("eid", None)
            entry.pop("participant_id", None)
            entry.pop("id", None)
            for variant in normalized_identifier_variants(str(key)):
                if variant and variant not in out:
                    out[variant] = entry
        return out

    raise ValueError(
        "Unsupported annotation JSON structure. "
        "Expected dict/list or {'annotations': [...]} with participant identifiers "
        "(eid/participant_id/id or object keys)."
    )


def _extract_report_and_perf(result_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    report = None
    perf = None

    report_files = sorted(result_dir.glob("report_*.json"))
    if report_files:
        report = _read_json(report_files[0])

    perf_files = sorted(result_dir.glob("performance_report_*.json"))
    if perf_files:
        perf = _read_json(perf_files[0])

    return report, perf


def _extract_composite_from_eval(eval_block: Dict[str, Any]) -> Optional[float]:
    if not isinstance(eval_block, dict):
        return None
    composite = safe_float(eval_block.get("composite_score"))
    if composite is not None:
        return composite
    passed = safe_float(eval_block.get("checklist_passed"))
    total = safe_float(eval_block.get("checklist_total"))
    if passed is None or total is None or total <= 0:
        return None
    return passed / total


def _extract_iteration_composites(report: Optional[Dict[str, Any]]) -> List[float]:
    if not isinstance(report, dict):
        return []

    out: List[float] = []

    exec_block = report.get("execution")
    if isinstance(exec_block, dict):
        detailed = exec_block.get("detailed_logs")
        if isinstance(detailed, list):
            for row in detailed:
                if not isinstance(row, dict):
                    continue
                eval_block = row.get("evaluation")
                if not isinstance(eval_block, dict):
                    continue
                comp = _extract_composite_from_eval(eval_block)
                if comp is not None:
                    out.append(float(comp))

    if out:
        return out

    exec_summary = report.get("execution_summary")
    if isinstance(exec_summary, dict):
        attempts = exec_summary.get("attempts") or exec_summary.get("iteration_details") or []
        if isinstance(attempts, list):
            for row in attempts:
                if not isinstance(row, dict):
                    continue
                eval_block = row.get("evaluation")
                if not isinstance(eval_block, dict):
                    continue
                comp = _extract_composite_from_eval(eval_block)
                if comp is not None:
                    out.append(float(comp))

    return out


def extract_generalized_prediction(result_dir: Path) -> Optional[Dict[str, Any]]:
    report, perf = _extract_report_and_perf(result_dir)
    if not report:
        return None

    pred_block = report.get("prediction")
    if not isinstance(pred_block, dict):
        pred_block = {}

    if not pred_block:
        exec_summary = report.get("execution_summary")
        if isinstance(exec_summary, dict):
            fp = exec_summary.get("final_prediction")
            if isinstance(fp, dict):
                pred_block = fp

    if not pred_block:
        return None

    task_spec = pred_block.get("prediction_task_spec")
    if not isinstance(task_spec, dict):
        task_spec = {}
    spec_root = task_spec.get("root") if isinstance(task_spec.get("root"), dict) else {}

    root = pred_block.get("root_prediction")
    if not isinstance(root, dict):
        root = {}
    flat = pred_block.get("flat_predictions")
    if not isinstance(flat, list):
        flat = []
    if root and not flat:
        flat = [root]

    root_mode = str(root.get("mode") or spec_root.get("mode") or "").strip() or None

    cls = root.get("classification") if isinstance(root.get("classification"), dict) else {}
    reg = root.get("regression") if isinstance(root.get("regression"), dict) else {}

    probs = cls.get("probabilities") if isinstance(cls.get("probabilities"), dict) else {}
    probabilities: Dict[str, float] = {}
    for k, v in probs.items():
        fv = safe_float(v)
        if fv is not None:
            probabilities[str(k)] = float(fv)

    vals = reg.get("values") if isinstance(reg.get("values"), dict) else {}
    regression_values: Dict[str, float] = {}
    for k, v in vals.items():
        fv = safe_float(v)
        if fv is not None:
            regression_values[str(k)] = float(fv)

    node_map: Dict[str, Dict[str, Any]] = {}
    for node in flat:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or "").strip()
        if not node_id:
            continue
        node_cls = node.get("classification") if isinstance(node.get("classification"), dict) else {}
        node_reg = node.get("regression") if isinstance(node.get("regression"), dict) else {}
        node_map[node_id] = {
            "mode": str(node.get("mode") or "").strip(),
            "predicted_label": node_cls.get("predicted_label"),
            "probabilities": node_cls.get("probabilities") if isinstance(node_cls.get("probabilities"), dict) else {},
            "values": node_reg.get("values") if isinstance(node_reg.get("values"), dict) else {},
        }

    eval_block = report.get("evaluation") if isinstance(report.get("evaluation"), dict) else {}
    exec_block = report.get("execution") if isinstance(report.get("execution"), dict) else {}

    verdict = str(eval_block.get("verdict") or "").strip() or None
    composite_score = _extract_composite_from_eval(eval_block)
    iterations = int(exec_block.get("iterations")) if safe_float(exec_block.get("iterations")) is not None else None
    duration = safe_float(exec_block.get("duration_seconds"))
    tokens = None
    token_block = exec_block.get("tokens_used")
    if isinstance(token_block, dict):
        tokens = int(safe_float(token_block.get("total") or token_block.get("total_tokens") or 0) or 0) or None
    elif safe_float(token_block) is not None:
        tokens = int(float(token_block))

    if perf:
        if duration is None:
            duration = safe_float(perf.get("duration_seconds"))
        if tokens is None:
            tk = perf.get("token_summary")
            if isinstance(tk, dict):
                maybe = safe_float(tk.get("total_tokens"))
                tokens = int(maybe) if maybe is not None else None

    root_conf = safe_float(root.get("confidence_score"))

    return {
        "root_mode": root_mode,
        "predicted_label": cls.get("predicted_label"),
        "probabilities": probabilities,
        "regression_values": regression_values,
        "nodes": node_map,
        "class_labels": spec_root.get("class_labels") if isinstance(spec_root.get("class_labels"), list) else [],
        "root_confidence": root_conf,
        "verdict": verdict,
        "composite_score": composite_score,
        "iterations": iterations,
        "duration": duration,
        "tokens": tokens,
        "iter_composites": _extract_iteration_composites(report),
        "raw": pred_block,
    }


def normalize_binary_prediction_label(
    *,
    predicted_label: Optional[str],
    class_labels: Optional[Sequence[str]] = None,
    binary_alias: Optional[str] = None,
    case_probability: Optional[float] = None,
) -> Optional[str]:
    raw = str(predicted_label or "").strip()
    up = raw.upper()
    if up:
        if "CASE" in up and "CONTROL" not in up:
            return "CASE"
        if "CONTROL" in up or "NON_CASE" in up or "NON-CASE" in up:
            return "CONTROL"

    labels = [str(x).strip() for x in (class_labels or []) if str(x).strip()]
    if raw and len(labels) == 2:
        if raw == labels[0]:
            return "CASE"
        if raw == labels[1]:
            return "CONTROL"
        if up == labels[0].upper():
            return "CASE"
        if up == labels[1].upper():
            return "CONTROL"

    alias = str(binary_alias or "").strip().upper()
    if alias:
        if "CASE" in alias and "CONTROL" not in alias:
            return "CASE"
        if "CONTROL" in alias:
            return "CONTROL"

    if case_probability is not None:
        return "CASE" if case_probability > 0.5 else "CONTROL"

    return None


def extract_binary_prediction(result_dir: Path) -> Optional[Dict[str, Any]]:
    generalized = extract_generalized_prediction(result_dir)
    if generalized is None:
        return None

    report, perf = _extract_report_and_perf(result_dir)

    pred_block = {}
    if isinstance(report, dict) and isinstance(report.get("prediction"), dict):
        pred_block = report["prediction"]

    spec_labels = generalized.get("class_labels") if isinstance(generalized.get("class_labels"), list) else []

    case_prob = None
    pb_prob = safe_float(pred_block.get("probability") if isinstance(pred_block, dict) else None)
    if pb_prob is None:
        pb_prob = safe_float(pred_block.get("probability_score") if isinstance(pred_block, dict) else None)
    if pb_prob is not None:
        case_prob = pb_prob
    else:
        probs = generalized.get("probabilities") if isinstance(generalized.get("probabilities"), dict) else {}
        if len(spec_labels) == 2:
            case_prob = safe_float(probs.get(spec_labels[0]))
        if case_prob is None:
            case_prob = safe_float(probs.get("CASE"))

    predicted = normalize_binary_prediction_label(
        predicted_label=generalized.get("predicted_label"),
        class_labels=spec_labels,
        binary_alias=(pred_block.get("binary_classification") if isinstance(pred_block, dict) else None),
        case_probability=case_prob,
    )

    if predicted is None:
        return None

    if case_prob is not None:
        case_prob = max(0.0, min(1.0, float(case_prob)))

    out = {
        "prediction": predicted,
        "raw_predicted_label": generalized.get("predicted_label"),
        "probability": case_prob,
        "verdict": generalized.get("verdict"),
        "composite_score": generalized.get("composite_score"),
        "iterations": generalized.get("iterations"),
        "duration": generalized.get("duration"),
        "tokens": generalized.get("tokens"),
        "iter_composites": generalized.get("iter_composites") or [],
        "status": "SUCCESS",
    }

    if perf:
        if out.get("duration") is None:
            out["duration"] = safe_float(perf.get("duration_seconds"))
        if out.get("tokens") is None:
            tk = perf.get("token_summary")
            if isinstance(tk, dict):
                maybe = safe_float(tk.get("total_tokens"))
                out["tokens"] = int(maybe) if maybe is not None else None

    return out


def _extract_disorder_from_truth(truth: Dict[str, Any]) -> str:
    if not isinstance(truth, dict):
        return "UNKNOWN"
    for key in ("disorder", "group", "cohort", "phenotype_group"):
        value = truth.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "UNKNOWN"


def collect_binary_results(
    results_dir: str,
    targets_file: str,
    disorder_filter: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    gt = load_ground_truth(targets_file)
    rows: List[Dict[str, Any]] = []

    allowed = set(disorder_filter or [])
    results_path = Path(results_dir)
    if not results_path.exists():
        return rows

    for participant_dir in sorted(results_path.iterdir()):
        if not participant_dir.is_dir():
            continue

        candidates = extract_participant_id_candidates(participant_dir.name)
        key = resolve_identifier(candidates, gt.keys())
        if key is None:
            continue

        truth = gt[key]
        if allowed and truth["disorder"] not in allowed:
            continue

        pred = extract_binary_prediction(participant_dir)
        if pred is None:
            rows.append(
                {
                    "eid": key,
                    "actual": truth["label"],
                    "predicted": None,
                    "disorder": truth["disorder"],
                    "status": "FAILED",
                }
            )
            continue

        rows.append(
            {
                "eid": key,
                "actual": truth["label"],
                "predicted": pred.get("prediction"),
                "probability": pred.get("probability"),
                "verdict": pred.get("verdict"),
                "composite_score": pred.get("composite_score"),
                "iterations": pred.get("iterations"),
                "duration": pred.get("duration"),
                "tokens": pred.get("tokens"),
                "iter_composites": pred.get("iter_composites") or [],
                "disorder": truth["disorder"],
                "status": "SUCCESS",
                "correct": pred.get("prediction") == truth["label"],
            }
        )

    return rows


def collect_generalized_rows(
    results_dir: str,
    annotations: Dict[str, Dict[str, Any]],
    disorder_filter: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    allowed = set(disorder_filter or [])

    results_path = Path(results_dir)
    if not results_path.exists():
        return rows

    for participant_dir in sorted(results_path.iterdir()):
        if not participant_dir.is_dir():
            continue

        candidates = extract_participant_id_candidates(participant_dir.name)
        key = resolve_identifier(candidates, annotations.keys())
        if key is None:
            continue

        truth = annotations[key]
        disorder = _extract_disorder_from_truth(truth)
        if allowed and disorder not in allowed:
            continue

        pred = extract_generalized_prediction(participant_dir)
        if pred is None:
            rows.append(
                {
                    "eid": key,
                    "truth": truth,
                    "pred": None,
                    "disorder": disorder,
                    "status": "FAILED",
                }
            )
            continue

        rows.append(
            {
                "eid": key,
                "truth": truth,
                "pred": pred,
                "disorder": disorder,
                "status": "SUCCESS",
            }
        )

    return rows


def collect_per_disorder(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        disorder = str(row.get("disorder") or "UNKNOWN")
        grouped[disorder].append(dict(row))
    return dict(grouped)

"""Metric computations for binary, multiclass, regression, and hierarchical tasks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .io_utils import safe_float

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is optional in some environments
    HAS_SCIPY = False


def compute_binary_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("predicted") in {"CASE", "CONTROL"}]
    failed = [r for r in rows if r.get("predicted") not in {"CASE", "CONTROL"}]

    tp = fp = tn = fn = 0
    for r in valid:
        actual = str(r.get("actual"))
        predicted = str(r.get("predicted"))
        if actual == "CASE" and predicted == "CASE":
            tp += 1
        elif actual == "CASE" and predicted == "CONTROL":
            fn += 1
        elif actual == "CONTROL" and predicted == "CONTROL":
            tn += 1
        elif actual == "CONTROL" and predicted == "CASE":
            fp += 1

    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    # Brier/ECE from CASE-probability if available.
    prob_rows = [r for r in valid if safe_float(r.get("probability")) is not None]
    brier = None
    ece = None
    calibration_bins: List[Dict[str, Any]] = []
    if prob_rows:
        probs = np.array([float(r["probability"]) for r in prob_rows], dtype=float)
        probs = np.clip(probs, 0.0, 1.0)
        y_true = np.array([1.0 if r.get("actual") == "CASE" else 0.0 for r in prob_rows], dtype=float)
        brier = float(np.mean((probs - y_true) ** 2))

        edges = np.linspace(0.0, 1.0, 11)
        ece_acc = 0.0
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            mask = (probs >= lo) & (probs < hi + (1e-12 if i == len(edges) - 2 else 0.0))
            if not np.any(mask):
                continue
            conf = float(np.mean(probs[mask]))
            obs = float(np.mean(y_true[mask]))
            n_bin = int(np.sum(mask))
            calibration_bins.append(
                {
                    "start": lo,
                    "end": hi,
                    "n": n_bin,
                    "mean_confidence": conf,
                    "observed_case_rate": obs,
                }
            )
            ece_acc += (n_bin / len(probs)) * abs(conf - obs)
        ece = float(ece_acc)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_total": len(rows),
        "n_valid": n,
        "n_failed": len(failed),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "mcc": mcc,
        "brier": brier,
        "ece": ece,
        "calibration_bins": calibration_bins,
    }


def _build_label_index(labels: Sequence[str]) -> Dict[str, int]:
    return {label: i for i, label in enumerate(labels)}


def compute_multiclass_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("actual") is not None and r.get("predicted") is not None]
    failed = [r for r in rows if r.get("actual") is None or r.get("predicted") is None]

    labels = sorted(
        {
            str(r.get("actual"))
            for r in valid
            if str(r.get("actual", "")).strip()
        }
        | {
            str(r.get("predicted"))
            for r in valid
            if str(r.get("predicted", "")).strip()
        }
    )

    if not labels:
        return {
            "n_total": len(rows),
            "n_valid": 0,
            "n_failed": len(failed),
            "labels": [],
            "matrix": [],
            "per_class": {},
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "top_confusions": [],
            "confidence_calibration": {
                "bins": [],
                "ece": None,
                "brier_top_label": None,
            },
        }

    idx = _build_label_index(labels)
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for r in valid:
        i = idx.get(str(r.get("actual")))
        j = idx.get(str(r.get("predicted")))
        if i is None or j is None:
            continue
        mat[i, j] += 1

    n = int(mat.sum())
    accuracy = float(np.trace(mat) / n) if n > 0 else 0.0

    per_class: Dict[str, Dict[str, Any]] = {}
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []
    weights: List[float] = []

    for i, label in enumerate(labels):
        tp = float(mat[i, i])
        fp = float(mat[:, i].sum() - tp)
        fn = float(mat[i, :].sum() - tp)
        support = int(mat[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        weights.append(float(support))

    macro_precision = float(np.mean(precision_values)) if precision_values else 0.0
    macro_recall = float(np.mean(recall_values)) if recall_values else 0.0
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    balanced_accuracy = macro_recall
    weighted_f1 = float(np.average(f1_values, weights=weights)) if weights and sum(weights) > 0 else 0.0

    confusions: List[Tuple[str, str, int]] = []
    for i, a_label in enumerate(labels):
        for j, p_label in enumerate(labels):
            if i == j:
                continue
            count = int(mat[i, j])
            if count > 0:
                confusions.append((a_label, p_label, count))
    confusions.sort(key=lambda t: t[2], reverse=True)

    # Top-label confidence calibration + probability diagnostics.
    conf_rows = []
    conf_correct: List[float] = []
    conf_incorrect: List[float] = []
    entropy_correct: List[float] = []
    entropy_incorrect: List[float] = []
    topk_hits = {1: 0, 2: 0, 3: 0}
    prob_eval_n = 0

    for r in valid:
        probs = r.get("probabilities") if isinstance(r.get("probabilities"), dict) else {}
        actual = str(r.get("actual"))
        pred = str(r.get("predicted"))
        numeric_probs: Dict[str, float] = {}
        for k, v in probs.items():
            fv = safe_float(v)
            if fv is None:
                continue
            if fv < 0:
                continue
            numeric_probs[str(k)] = float(fv)

        conf = safe_float(numeric_probs.get(pred))
        if conf is None:
            conf = safe_float(r.get("predicted_probability"))
        if conf is None:
            continue
        conf = max(0.0, min(1.0, float(conf)))
        is_correct = str(r.get("actual")) == pred
        conf_rows.append(
            {
                "confidence": conf,
                "correct": 1.0 if is_correct else 0.0,
            }
        )
        if is_correct:
            conf_correct.append(conf)
        else:
            conf_incorrect.append(conf)

        if numeric_probs:
            prob_eval_n += 1
            ranked = [k for k, _ in sorted(numeric_probs.items(), key=lambda kv: kv[1], reverse=True)]
            for k in (1, 2, 3):
                k_eff = min(k, len(ranked))
                if k_eff > 0 and actual in ranked[:k_eff]:
                    topk_hits[k] += 1

            probs_arr = np.array(list(numeric_probs.values()), dtype=float)
            s = float(np.sum(probs_arr))
            if s > 0.0 and len(probs_arr) > 1:
                p = probs_arr / s
                entropy = float((-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))) / np.log(len(p))))
                if is_correct:
                    entropy_correct.append(entropy)
                else:
                    entropy_incorrect.append(entropy)

    calibration_bins: List[Dict[str, Any]] = []
    ece = None
    brier_top = None
    if conf_rows:
        conf = np.array([row["confidence"] for row in conf_rows], dtype=float)
        corr = np.array([row["correct"] for row in conf_rows], dtype=float)
        brier_top = float(np.mean((conf - corr) ** 2))

        edges = np.linspace(0.0, 1.0, 11)
        ece_acc = 0.0
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            mask = (conf >= lo) & (conf < hi + (1e-12 if i == len(edges) - 2 else 0.0))
            if not np.any(mask):
                continue
            mean_conf = float(np.mean(conf[mask]))
            mean_acc = float(np.mean(corr[mask]))
            n_bin = int(np.sum(mask))
            calibration_bins.append(
                {
                    "start": lo,
                    "end": hi,
                    "n": n_bin,
                    "mean_confidence": mean_conf,
                    "observed_accuracy": mean_acc,
                }
            )
            ece_acc += (n_bin / len(conf)) * abs(mean_conf - mean_acc)
        ece = float(ece_acc)

    return {
        "n_total": len(rows),
        "n_valid": n,
        "n_failed": len(failed),
        "labels": labels,
        "matrix": mat.tolist(),
        "per_class": per_class,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top_confusions": [
            {"actual": a, "predicted": p, "count": c} for a, p, c in confusions[:20]
        ],
        "confidence_calibration": {
            "bins": calibration_bins,
            "ece": ece,
            "brier_top_label": brier_top,
        },
        "probability_diagnostics": {
            "n_probability_rows": prob_eval_n,
            "top_k_accuracy": {
                "top1": float(topk_hits[1] / prob_eval_n) if prob_eval_n > 0 else None,
                "top2": float(topk_hits[2] / prob_eval_n) if prob_eval_n > 0 else None,
                "top3": float(topk_hits[3] / prob_eval_n) if prob_eval_n > 0 else None,
            },
            "confidence_correct": conf_correct,
            "confidence_incorrect": conf_incorrect,
            "entropy_correct": entropy_correct,
            "entropy_incorrect": entropy_incorrect,
            "mean_confidence_correct": float(np.mean(conf_correct)) if conf_correct else None,
            "mean_confidence_incorrect": float(np.mean(conf_incorrect)) if conf_incorrect else None,
            "mean_entropy_correct": float(np.mean(entropy_correct)) if entropy_correct else None,
            "mean_entropy_incorrect": float(np.mean(entropy_incorrect)) if entropy_incorrect else None,
        },
    }


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) < 2:
        return None
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return None
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if not HAS_SCIPY or len(y_true) < 2:
        return None
    try:
        coef, _ = scipy_stats.spearmanr(y_true, y_pred)
    except Exception:
        return None
    coef_f = safe_float(coef)
    return float(coef_f) if coef_f is not None else None


def compute_regression_metrics(
    rows: Sequence[Dict[str, Any]],
    expected_outputs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    by_output_true: Dict[str, List[float]] = defaultdict(list)
    by_output_pred: Dict[str, List[float]] = defaultdict(list)
    by_output_residuals: Dict[str, List[float]] = defaultdict(list)
    largest_error_rows: List[Dict[str, Any]] = []

    for row in rows:
        actual_values = row.get("actual_values") if isinstance(row.get("actual_values"), dict) else {}
        predicted_values = row.get("predicted_values") if isinstance(row.get("predicted_values"), dict) else {}
        for output_name, true_val in actual_values.items():
            t = safe_float(true_val)
            p = safe_float(predicted_values.get(output_name))
            if t is None or p is None:
                continue
            key = str(output_name)
            by_output_true[key].append(float(t))
            by_output_pred[key].append(float(p))
            err = float(p - t)
            by_output_residuals[key].append(err)
            largest_error_rows.append(
                {
                    "eid": row.get("eid"),
                    "disorder": row.get("disorder"),
                    "output": key,
                    "actual": float(t),
                    "predicted": float(p),
                    "signed_error": err,
                    "abs_error": abs(err),
                }
            )

    if expected_outputs:
        for name in expected_outputs:
            key = str(name)
            by_output_true.setdefault(key, [])
            by_output_pred.setdefault(key, [])

    per_output: Dict[str, Dict[str, Any]] = {}
    mae_list: List[float] = []
    rmse_list: List[float] = []
    r2_list: List[float] = []
    all_true: List[float] = []
    all_pred: List[float] = []

    for output_name in sorted(by_output_true.keys()):
        y_true = np.array(by_output_true[output_name], dtype=float)
        y_pred = np.array(by_output_pred[output_name], dtype=float)
        n = int(len(y_true))

        if n == 0:
            per_output[output_name] = {
                "n": 0,
                "mae": None,
                "rmse": None,
                "mape": None,
                "r2": None,
                "pearson": None,
                "spearman": None,
                "mean_error": None,
                "raw_pairs": {"y_true": [], "y_pred": []},
            }
            continue

        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mean_error = float(np.mean(err))

        non_zero_mask = np.abs(y_true) > 1e-12
        mape = float(np.mean(np.abs(err[non_zero_mask] / y_true[non_zero_mask]))) if np.any(non_zero_mask) else None

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        pearson = _pearson(y_true, y_pred)
        spearman = _spearman(y_true, y_pred)

        per_output[output_name] = {
            "n": n,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "pearson": pearson,
            "spearman": spearman,
            "mean_error": mean_error,
            "residual_summary": {
                "std_error": float(np.std(err)),
                "median_error": float(np.median(err)),
                "p05_error": float(np.quantile(err, 0.05)),
                "p25_error": float(np.quantile(err, 0.25)),
                "p75_error": float(np.quantile(err, 0.75)),
                "p95_error": float(np.quantile(err, 0.95)),
            },
            "raw_residuals": [float(x) for x in err.tolist()],
            "raw_pairs": {
                "y_true": [float(x) for x in y_true.tolist()],
                "y_pred": [float(x) for x in y_pred.tolist()],
            },
        }

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

    # Micro metrics across all output points.
    micro = {
        "n": int(len(all_true)),
        "mae": None,
        "rmse": None,
        "r2": None,
        "pearson": None,
        "spearman": None,
    }
    if all_true:
        y_true_all = np.array(all_true, dtype=float)
        y_pred_all = np.array(all_pred, dtype=float)
        err_all = y_pred_all - y_true_all
        micro["mae"] = float(np.mean(np.abs(err_all)))
        micro["rmse"] = float(np.sqrt(np.mean(err_all ** 2)))
        ss_res_all = float(np.sum((y_true_all - y_pred_all) ** 2))
        ss_tot_all = float(np.sum((y_true_all - float(np.mean(y_true_all))) ** 2))
        micro["r2"] = float(1.0 - ss_res_all / ss_tot_all) if ss_tot_all > 0 else 0.0
        micro["pearson"] = _pearson(y_true_all, y_pred_all)
        micro["spearman"] = _spearman(y_true_all, y_pred_all)
        micro["residual_summary"] = {
            "mean_error": float(np.mean(err_all)),
            "std_error": float(np.std(err_all)),
            "median_error": float(np.median(err_all)),
            "p05_error": float(np.quantile(err_all, 0.05)),
            "p95_error": float(np.quantile(err_all, 0.95)),
        }

    largest_error_rows.sort(key=lambda row: float(row.get("abs_error") or 0.0), reverse=True)

    return {
        "n_rows": len(rows),
        "n_outputs": len(per_output),
        "n_outputs_with_data": int(sum(1 for v in per_output.values() if int(v.get("n") or 0) > 0)),
        "per_output": per_output,
        "macro_mae": float(np.mean(mae_list)) if mae_list else None,
        "macro_rmse": float(np.mean(rmse_list)) if rmse_list else None,
        "macro_r2": float(np.mean(r2_list)) if r2_list else None,
        "micro": micro,
        "largest_absolute_errors": largest_error_rows[:30],
    }


def _extract_truth_label(node: Dict[str, Any]) -> Optional[str]:
    for key in ("predicted_label", "label", "classification", "class_label"):
        value = node.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _extract_truth_values(node: Dict[str, Any]) -> Dict[str, Any]:
    values = node.get("values")
    if isinstance(values, dict):
        return values
    reg = node.get("regression")
    if isinstance(reg, dict):
        inner = reg.get("values")
        if isinstance(inner, dict):
            return inner
    return {}


def compute_hierarchical_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    node_mode: Dict[str, str] = {}
    node_coverage: Dict[str, Dict[str, int]] = defaultdict(lambda: {"truth_present": 0, "pred_present": 0, "both_present": 0})
    node_class_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
            "missing_truth_label": 0,
            "missing_pred_label": 0,
        }
    )
    node_reg_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        truth_nodes = row.get("truth_nodes") if isinstance(row.get("truth_nodes"), dict) else {}
        pred_nodes = row.get("pred_nodes") if isinstance(row.get("pred_nodes"), dict) else {}
        node_ids = set(truth_nodes.keys()) | set(pred_nodes.keys())

        for node_id in node_ids:
            truth_node = truth_nodes.get(node_id) if isinstance(truth_nodes.get(node_id), dict) else {}
            pred_node = pred_nodes.get(node_id) if isinstance(pred_nodes.get(node_id), dict) else {}

            if truth_node:
                node_coverage[node_id]["truth_present"] += 1
            if pred_node:
                node_coverage[node_id]["pred_present"] += 1
            if truth_node and pred_node:
                node_coverage[node_id]["both_present"] += 1

            mode = str((truth_node or {}).get("mode") or (pred_node or {}).get("mode") or "").strip()
            if not mode:
                if _extract_truth_values(truth_node):
                    mode = "multivariate_regression"
                elif _extract_truth_label(truth_node):
                    mode = "multiclass_classification"
            node_mode[node_id] = mode or node_mode.get(node_id, "")

            if mode.endswith("classification"):
                t_label = _extract_truth_label(truth_node) if truth_node else None
                p_label = None
                if pred_node:
                    p_label = pred_node.get("predicted_label")
                    if p_label is None:
                        p_label = _extract_truth_label(pred_node)

                if t_label is None:
                    node_class_counts[node_id]["missing_truth_label"] += 1
                if p_label is None:
                    node_class_counts[node_id]["missing_pred_label"] += 1
                if t_label is None or p_label is None:
                    continue

                node_class_counts[node_id]["total"] += 1
                if str(t_label) == str(p_label):
                    node_class_counts[node_id]["correct"] += 1

            elif mode.endswith("regression"):
                actual_values = _extract_truth_values(truth_node)
                pred_values = pred_node.get("values") if isinstance(pred_node.get("values"), dict) else {}
                node_reg_rows[node_id].append(
                    {
                        "actual_values": actual_values,
                        "predicted_values": pred_values,
                    }
                )

    per_node: Dict[str, Dict[str, Any]] = {}
    class_scores: List[float] = []
    reg_scores: List[float] = []

    all_node_ids = sorted(set(node_mode.keys()) | set(node_coverage.keys()) | set(node_class_counts.keys()) | set(node_reg_rows.keys()))

    for node_id in all_node_ids:
        mode = node_mode.get(node_id) or ""
        coverage = node_coverage.get(node_id, {"truth_present": 0, "pred_present": 0, "both_present": 0})
        truth_present = int(coverage.get("truth_present") or 0)
        pred_present = int(coverage.get("pred_present") or 0)
        both_present = int(coverage.get("both_present") or 0)
        truth_only = max(0, truth_present - both_present)
        pred_only = max(0, pred_present - both_present)
        coverage_rate = float(both_present / truth_present) if truth_present > 0 else None

        if mode.endswith("classification"):
            counts = node_class_counts.get(node_id, {})
            total = int(counts.get("total") or 0)
            correct = int(counts.get("correct") or 0)
            acc = float(correct / total) if total > 0 else None
            per_node[node_id] = {
                "mode": mode,
                "n": total,
                "accuracy": acc,
                "correct": correct,
                "missing_truth_label": int(counts.get("missing_truth_label") or 0),
                "missing_pred_label": int(counts.get("missing_pred_label") or 0),
                "coverage": {
                    "truth_present": truth_present,
                    "pred_present": pred_present,
                    "both_present": both_present,
                    "truth_only": truth_only,
                    "pred_only": pred_only,
                    "coverage_rate": coverage_rate,
                },
            }
            if acc is not None:
                class_scores.append(acc)
        elif mode.endswith("regression"):
            reg_rows = node_reg_rows.get(node_id, [])
            metric = compute_regression_metrics(reg_rows)
            node_r2 = metric.get("macro_r2")
            per_node[node_id] = {
                "mode": mode,
                "n": int(metric.get("micro", {}).get("n") or 0),
                "macro_r2": node_r2,
                "regression": metric,
                "coverage": {
                    "truth_present": truth_present,
                    "pred_present": pred_present,
                    "both_present": both_present,
                    "truth_only": truth_only,
                    "pred_only": pred_only,
                    "coverage_rate": coverage_rate,
                },
            }
            if node_r2 is not None:
                reg_scores.append(float(max(0.0, min(1.0, node_r2))))
        else:
            per_node[node_id] = {
                "mode": mode or "unknown",
                "n": 0,
                "coverage": {
                    "truth_present": truth_present,
                    "pred_present": pred_present,
                    "both_present": both_present,
                    "truth_only": truth_only,
                    "pred_only": pred_only,
                    "coverage_rate": coverage_rate,
                },
            }

    macro_class_acc = float(np.mean(class_scores)) if class_scores else None
    macro_reg_r2 = float(np.mean(reg_scores)) if reg_scores else None

    combined_scores = []
    if macro_class_acc is not None:
        combined_scores.append(macro_class_acc)
    if macro_reg_r2 is not None:
        combined_scores.append(macro_reg_r2)

    macro_score = float(np.mean(combined_scores)) if combined_scores else None

    coverage_rates = [
        float((block.get("coverage") or {}).get("coverage_rate"))
        for block in per_node.values()
        if isinstance(block.get("coverage"), dict) and block.get("coverage", {}).get("coverage_rate") is not None
    ]

    return {
        "n_rows": len(rows),
        "n_nodes": len(per_node),
        "n_classification_nodes": int(sum(1 for _, v in per_node.items() if str(v.get("mode", "")).endswith("classification"))),
        "n_regression_nodes": int(sum(1 for _, v in per_node.items() if str(v.get("mode", "")).endswith("regression"))),
        "macro_classification_accuracy": macro_class_acc,
        "macro_regression_r2": macro_reg_r2,
        "macro_score": macro_score,
        "coverage_summary": {
            "macro_node_coverage_rate": float(np.mean(coverage_rates)) if coverage_rates else None,
            "total_truth_node_instances": int(sum((v.get("coverage") or {}).get("truth_present") or 0 for v in per_node.values())),
            "total_pred_node_instances": int(sum((v.get("coverage") or {}).get("pred_present") or 0 for v in per_node.values())),
            "total_both_node_instances": int(sum((v.get("coverage") or {}).get("both_present") or 0 for v in per_node.values())),
        },
        "per_node": per_node,
    }

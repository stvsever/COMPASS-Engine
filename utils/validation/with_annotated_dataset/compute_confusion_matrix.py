#!/usr/bin/env python3
"""
COMPASS Annotated Validation Metrics Generator.

Computes and plots binary confusion matrices (CASE vs CONTROL) from COMPASS
participant run results, using annotated ground-truth labels.

For non-binary tasks (multiclass, regression, hierarchical), this script
computes generalized metrics and writes JSON summaries.

Outputs:
  - integrated_confusion_matrix.png  (all disorders combined)
  - {disorder}_confusion_matrix.png  (per-disorder, if --disorder_groups given)

Usage:
  python compute_confusion_matrix.py \
      --results_dir /path/to/participant_runs \
      --targets_file /path/to/cases_controls_with_specific_subtypes.txt \
      --output_dir /path/to/output/binary_confusion_matrix

  # With per-disorder breakdown:
  python compute_confusion_matrix.py \
      --results_dir /path/to/participant_runs \
      --targets_file /path/to/cases_controls_with_specific_subtypes.txt \
      --output_dir /path/to/output/binary_confusion_matrix \
      --disorder_groups "MAJOR_DEPRESSIVE_DISORDER,ANXIETY_DISORDERS,SUBSTANCE_USE_DISORDERS"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Data extraction helpers
# ═══════════════════════════════════════════════════════════════════════════

SUPPORTED_PREDICTION_TYPES = {
    "binary",
    "multiclass",
    "regression_univariate",
    "regression_multivariate",
    "hierarchical",
}


def _safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def load_generalized_annotations(path: str) -> Dict[str, dict]:
    """
    Load generalized annotation file for multiclass/regression/hierarchical metrics.

    Accepted formats:
      - Dict keyed by eid: {"123": {...}, "124": {...}}
      - List of dicts with eid/id field: [{"eid":"123", ...}, ...]
    """
    with open(path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        if "annotations" in raw and isinstance(raw["annotations"], list):
            raw = raw["annotations"]
        else:
            out = {}
            for eid, payload in raw.items():
                out[str(eid)] = payload if isinstance(payload, dict) else {"value": payload}
            return out

    if isinstance(raw, list):
        out = {}
        for row in raw:
            if not isinstance(row, dict):
                continue
            eid = row.get("eid") or row.get("participant_id") or row.get("id")
            if eid is None:
                continue
            payload = dict(row)
            payload.pop("eid", None)
            payload.pop("participant_id", None)
            payload.pop("id", None)
            out[str(eid)] = payload
        return out

    raise ValueError("Unsupported annotation JSON structure")


def extract_generalized_prediction(result_dir: Path) -> Optional[dict]:
    """Extract generalized prediction payload from report JSON."""
    report_files = list(result_dir.glob("report_*.json"))
    if not report_files:
        return None
    try:
        with open(report_files[0], "r") as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    pred_block = report.get("prediction", {}) if isinstance(report, dict) else {}
    if not isinstance(pred_block, dict):
        pred_block = {}
    if not pred_block:
        exec_summary = report.get("execution_summary", {}) if isinstance(report, dict) else {}
        if isinstance(exec_summary, dict):
            final_pred = exec_summary.get("final_prediction", {})
            if isinstance(final_pred, dict):
                pred_block = final_pred

    if not pred_block:
        return None

    root = pred_block.get("root_prediction")
    flat = pred_block.get("flat_predictions")
    if not isinstance(flat, list):
        flat = []
    if isinstance(root, dict) and not flat:
        flat = [root]

    root_mode = None
    predicted_label = None
    probabilities = {}
    regression_values = {}
    if isinstance(root, dict):
        root_mode = root.get("mode")
        cls = root.get("classification") if isinstance(root.get("classification"), dict) else {}
        reg = root.get("regression") if isinstance(root.get("regression"), dict) else {}
        predicted_label = cls.get("predicted_label")
        probs = cls.get("probabilities")
        if isinstance(probs, dict):
            probabilities = {str(k): float(v) for k, v in probs.items() if _safe_float(v) is not None}
        values = reg.get("values")
        if isinstance(values, dict):
            regression_values = {str(k): float(v) for k, v in values.items() if _safe_float(v) is not None}

    node_map = {}
    for node in flat:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or "").strip()
        if not node_id:
            continue
        node_mode = str(node.get("mode") or "").strip()
        cls = node.get("classification") if isinstance(node.get("classification"), dict) else {}
        reg = node.get("regression") if isinstance(node.get("regression"), dict) else {}
        node_map[node_id] = {
            "mode": node_mode,
            "predicted_label": cls.get("predicted_label"),
            "probabilities": cls.get("probabilities") if isinstance(cls.get("probabilities"), dict) else {},
            "values": reg.get("values") if isinstance(reg.get("values"), dict) else {},
        }

    return {
        "root_mode": root_mode,
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "regression_values": regression_values,
        "nodes": node_map,
        "raw": pred_block,
    }


def _compute_multiclass_metrics(rows: List[dict]) -> dict:
    valid = [r for r in rows if r.get("actual") is not None and r.get("predicted") is not None]
    labels = sorted(set([str(r["actual"]) for r in valid] + [str(r["predicted"]) for r in valid]))
    if not labels:
        return {"n_valid": 0, "n_failed": len(rows), "labels": [], "matrix": [], "accuracy": 0.0, "macro_f1": 0.0}

    idx = {label: i for i, label in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for r in valid:
        mat[idx[str(r["actual"])], idx[str(r["predicted"])]] += 1

    n = int(mat.sum())
    acc = float(np.trace(mat) / n) if n > 0 else 0.0
    f1s = []
    for i in range(len(labels)):
        tp = float(mat[i, i])
        fp = float(mat[:, i].sum() - tp)
        fn = float(mat[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return {
        "n_valid": n,
        "n_failed": len(rows) - n,
        "labels": labels,
        "matrix": mat.tolist(),
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def _compute_regression_metrics(rows: List[dict], expected_outputs: Optional[List[str]] = None) -> dict:
    by_output_true: Dict[str, List[float]] = defaultdict(list)
    by_output_pred: Dict[str, List[float]] = defaultdict(list)

    for r in rows:
        true_vals = r.get("actual_values") or {}
        pred_vals = r.get("predicted_values") or {}
        for key, true_val in true_vals.items():
            t = _safe_float(true_val)
            p = _safe_float(pred_vals.get(key))
            if t is None or p is None:
                continue
            by_output_true[str(key)].append(t)
            by_output_pred[str(key)].append(p)

    if expected_outputs:
        for key in expected_outputs:
            by_output_true.setdefault(str(key), [])
            by_output_pred.setdefault(str(key), [])

    per_output = {}
    maes = []
    rmses = []
    r2s = []
    for key in sorted(by_output_true.keys()):
        y_true = np.array(by_output_true[key], dtype=float)
        y_pred = np.array(by_output_pred[key], dtype=float)
        if len(y_true) == 0:
            per_output[key] = {"n": 0, "mae": None, "rmse": None, "r2": None}
            continue
        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        per_output[key] = {"n": int(len(y_true)), "mae": mae, "rmse": rmse, "r2": r2}
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    return {
        "per_output": per_output,
        "macro_mae": float(np.mean(maes)) if maes else None,
        "macro_rmse": float(np.mean(rmses)) if rmses else None,
        "macro_r2": float(np.mean(r2s)) if r2s else None,
    }


def _compute_hierarchical_metrics(rows: List[dict]) -> dict:
    node_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    node_reg_true: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    node_reg_pred: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        truth_nodes = row.get("truth_nodes") or {}
        pred_nodes = row.get("pred_nodes") or {}
        for node_id, truth in truth_nodes.items():
            pred = pred_nodes.get(node_id) or {}
            mode = str((truth or {}).get("mode") or "").strip()
            if mode.endswith("classification"):
                t_label = (truth or {}).get("predicted_label") or (truth or {}).get("label")
                p_label = (pred or {}).get("predicted_label")
                if t_label is None or p_label is None:
                    continue
                node_acc[node_id]["total"] += 1
                if str(t_label) == str(p_label):
                    node_acc[node_id]["correct"] += 1
            elif mode.endswith("regression"):
                t_vals = (truth or {}).get("values") or {}
                p_vals = (pred or {}).get("values") or {}
                for key, tv in t_vals.items():
                    t = _safe_float(tv)
                    p = _safe_float(p_vals.get(key))
                    if t is None or p is None:
                        continue
                    node_reg_true[node_id][key].append(t)
                    node_reg_pred[node_id][key].append(p)

    node_metrics = {}
    macro_scores = []

    for node_id, counts in node_acc.items():
        total = counts["total"]
        acc = counts["correct"] / total if total > 0 else None
        node_metrics[node_id] = {"mode": "classification", "n": total, "accuracy": acc}
        if acc is not None:
            macro_scores.append(acc)

    for node_id in sorted(node_reg_true.keys()):
        per_output = {}
        scores = []
        for key in sorted(node_reg_true[node_id].keys()):
            y_true = np.array(node_reg_true[node_id][key], dtype=float)
            y_pred = np.array(node_reg_pred[node_id][key], dtype=float)
            if len(y_true) == 0:
                continue
            err = y_pred - y_true
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            per_output[key] = {"n": int(len(y_true)), "mae": mae, "rmse": rmse, "r2": r2}
            scores.append(max(0.0, min(1.0, r2)))
        if per_output:
            node_metrics[node_id] = {
                "mode": "regression",
                "outputs": per_output,
                "macro_r2": float(np.mean(scores)) if scores else None,
            }
            if scores:
                macro_scores.append(float(np.mean(scores)))

    return {
        "per_node": node_metrics,
        "macro_score": float(np.mean(macro_scores)) if macro_scores else None,
    }

def load_ground_truth(targets_file: str) -> Dict[str, dict]:
    """Parse the annotated targets file into {eid: {label, disorder}}."""
    ground_truth = {}
    with open(targets_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            eid = parts[0].strip()
            if len(parts) < 2:
                continue
            label_part = parts[1].strip()
            # Extract CASE or CONTROL
            if "CASE" in label_part and "CONTROL" not in label_part:
                label = "CASE"
            elif "CONTROL" in label_part:
                label = "CONTROL"
            else:
                continue
            # Extract disorder from parentheses
            disorder_match = re.search(r"\(([^)]+)\)", label_part)
            disorder = disorder_match.group(1) if disorder_match else "UNKNOWN"
            ground_truth[eid] = {"label": label, "disorder": disorder}
    return ground_truth


def extract_prediction(result_dir: Path) -> Optional[dict]:
    """Extract prediction from a participant result folder.

    Handles multiple report schemas:
      - Schema A (current): data["prediction"], data["evaluation"], data["execution"]
      - Schema B (legacy/future): data["execution_summary"]["final_prediction"]
    """
    report_files = list(result_dir.glob("report_*.json"))
    if not report_files:
        return None

    report_file = report_files[0]
    try:
        with open(report_file, "r") as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    prediction = None
    probability = None
    verdict = None
    composite_score = None
    iterations = None

    # ── Schema A: top-level "prediction" / "evaluation" / "execution" ──
    pred_block = report.get("prediction", {})
    if isinstance(pred_block, dict):
        prediction = pred_block.get("classification")
        probability = pred_block.get("probability") or pred_block.get("probability_score")

    eval_block = report.get("evaluation", {})
    if isinstance(eval_block, dict):
        verdict = eval_block.get("verdict")
        composite_score = eval_block.get("composite_score")
        checklist_passed = eval_block.get("checklist_passed")
        checklist_total = eval_block.get("checklist_total")
        if composite_score is None and checklist_passed is not None and checklist_total is not None:
            try:
                composite_score = float(checklist_passed) / float(checklist_total) if float(checklist_total) > 0 else None
            except (ValueError, TypeError):
                pass

    exec_block = report.get("execution", {})
    if isinstance(exec_block, dict):
        iterations = exec_block.get("iterations") or exec_block.get("total_iterations")

    # ── Schema B: execution_summary.final_prediction / final_evaluation ──
    if not prediction:
        exec_summary = report.get("execution_summary", {})
        if isinstance(exec_summary, dict):
            final_pred = exec_summary.get("final_prediction", {})
            if isinstance(final_pred, dict):
                prediction = final_pred.get("classification") or final_pred.get("binary_classification")
                probability = probability or final_pred.get("probability")
            final_eval = exec_summary.get("final_evaluation", {})
            if isinstance(final_eval, dict):
                verdict = verdict or final_eval.get("verdict")
                composite_score = composite_score or final_eval.get("composite_score")
            iterations = iterations or exec_summary.get("iterations")

    # ── Fallback: bare top-level ──
    if not prediction:
        prediction = report.get("classification") or report.get("predicted_label")

    # Clean prediction to binary CASE / CONTROL
    if prediction:
        prediction = str(prediction).upper().strip()
        if "CASE" in prediction and "CONTROL" not in prediction:
            prediction = "CASE"
        elif "CONTROL" in prediction or "NON_CASE" in prediction or "NOT PSYCHIATRIC" in prediction:
            prediction = "CONTROL"
        else:
            if probability is not None:
                try:
                    prediction = "CASE" if float(probability) > 0.5 else "CONTROL"
                except (ValueError, TypeError):
                    prediction = None
            else:
                prediction = None

    if prediction is None:
        return None

    return {
        "prediction": prediction,
        "probability": float(probability) if probability is not None else None,
        "verdict": str(verdict) if verdict else None,
        "composite_score": float(composite_score) if composite_score is not None else None,
        "iterations": int(iterations) if iterations is not None else None,
    }



def collect_results(
    results_dir: str,
    targets_file: str,
    disorder_filter: Optional[List[str]] = None,
) -> List[dict]:
    """Collect all participant results matched against ground truth."""
    ground_truth = load_ground_truth(targets_file)
    results = []

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return results

    for participant_dir in sorted(results_path.iterdir()):
        if not participant_dir.is_dir():
            continue

        # Extract EID
        dir_name = participant_dir.name
        eid_match = re.search(r"ID(\d+)", dir_name)
        if not eid_match:
            continue
        eid = eid_match.group(1)

        # Ground truth lookup
        gt = ground_truth.get(eid)
        if gt is None:
            continue

        # Filter by disorder if specified
        if disorder_filter and gt["disorder"] not in disorder_filter:
            continue

        # Extract prediction
        pred_info = extract_prediction(participant_dir)
        if pred_info is None:
            results.append({
                "eid": eid,
                "actual": gt["label"],
                "predicted": None,
                "disorder": gt["disorder"],
                "status": "FAILED",
            })
            continue

        results.append({
            "eid": eid,
            "actual": gt["label"],
            "predicted": pred_info["prediction"],
            "probability": pred_info["probability"],
            "verdict": pred_info["verdict"],
            "composite_score": pred_info["composite_score"],
            "iterations": pred_info["iterations"],
            "disorder": gt["disorder"],
            "status": "SUCCESS",
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Metrics computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(results: List[dict]) -> dict:
    """Compute binary classification metrics from results.

    Accepts results with either 'predicted' or 'prediction' key.
    """
    valid = [r for r in results if (r.get("predicted") or r.get("prediction")) is not None]
    failed = [r for r in results if (r.get("predicted") or r.get("prediction")) is None]

    tp = fp = tn = fn = 0
    for r in valid:
        actual = r["actual"]
        predicted = r.get("predicted") or r.get("prediction")
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
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # Matthews Correlation Coefficient
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_valid": n,
        "n_failed": len(failed),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "mcc": mcc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Professional confusion matrix plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    metrics: dict,
    title: str,
    output_path: str,
) -> None:
    """Generate a professional confusion matrix plot with metrics sidebar."""
    tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]
    n = metrics["n_valid"]
    matrix = np.array([[tp, fn], [fp, tn]])

    # ── Color scheme ──
    bg_color = "#0D1117"
    card_color = "#161B22"
    text_primary = "#E6EDF3"
    text_secondary = "#8B949E"
    accent_blue = "#58A6FF"
    accent_green = "#3FB950"
    accent_orange = "#D29922"
    accent_red = "#F85149"

    # Custom colormap: dark blue → teal → bright green
    colors_list = ["#0D1117", "#0E4429", "#006D32", "#26A641", "#39D353"]
    cmap = mcolors.LinearSegmentedColormap.from_list("compass", colors_list, N=256)

    fig, (ax_matrix, ax_metrics) = plt.subplots(
        1, 2, figsize=(14, 6.5),
        gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.05},
    )
    fig.patch.set_facecolor(bg_color)

    # ── Matrix ──
    ax_matrix.set_facecolor(card_color)
    max_val = matrix.max() if matrix.max() > 0 else 1

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            pct = val / n * 100 if n > 0 else 0
            norm_val = val / max_val
            cell_color = cmap(norm_val * 0.85 + 0.15) if val > 0 else card_color

            rect = FancyBboxPatch(
                (j - 0.45, i - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=cell_color,
                edgecolor="#30363D",
                linewidth=1.5,
            )
            ax_matrix.add_patch(rect)

            ax_matrix.text(j, i - 0.08, f"{val}", ha="center", va="center",
                          fontsize=28, fontweight="bold", color=text_primary, fontfamily="monospace")
            ax_matrix.text(j, i + 0.28, f"({pct:.1f}%)", ha="center", va="center",
                          fontsize=12, color=text_secondary, fontfamily="monospace")

    # Labels
    categories = ["CASE", "CONTROL"]
    ax_matrix.set_xticks([0, 1])
    ax_matrix.set_yticks([0, 1])
    ax_matrix.set_xticklabels([f"Pred\n{c}" for c in categories], fontsize=11,
                               color=text_primary, fontweight="bold")
    ax_matrix.set_yticklabels([f"Actual\n{c}" for c in categories], fontsize=11,
                               color=text_primary, fontweight="bold", rotation=0, ha="right")
    ax_matrix.tick_params(axis="both", length=0, pad=12)
    ax_matrix.set_xlim(-0.6, 1.6)
    ax_matrix.set_ylim(1.6, -0.6)

    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    # REMOVED TITLE as requested
    # ax_matrix.set_title(title, fontsize=16, fontweight="bold",
    #                     color=accent_blue, pad=20, fontfamily="sans-serif")

    # ── Metrics sidebar ──
    ax_metrics.set_facecolor(bg_color)
    ax_metrics.axis("off")

    metric_items = [
        ("Accuracy",    metrics["accuracy"],    accent_green),
        ("Sensitivity", metrics["sensitivity"], accent_blue),
        ("Specificity", metrics["specificity"], accent_blue),
        ("Precision",   metrics["precision"],   accent_orange),
        ("F1 Score",    metrics["f1"],          accent_orange),
        ("MCC",         metrics["mcc"],         accent_red),
    ]

    y_start = 0.88
    for idx, (name, value, color) in enumerate(metric_items):
        y = y_start - idx * 0.13
        # Metric name
        ax_metrics.text(0.05, y, name, transform=ax_metrics.transAxes,
                       fontsize=13, color=text_secondary, fontweight="bold",
                       fontfamily="sans-serif", va="center")
        # Value
        if name == "MCC":
            val_str = f"{value:+.3f}"
        else:
            val_str = f"{value:.1%}"
        ax_metrics.text(0.95, y, val_str, transform=ax_metrics.transAxes,
                       fontsize=15, color=color, fontweight="bold",
                       fontfamily="monospace", va="center", ha="right")
        # Bar
        bar_y = y - 0.035
        bar_width = max(0, min(1, abs(value))) * 0.9
        ax_metrics.barh(bar_y, bar_width, left=0.05,
                       height=0.018, color=color, alpha=0.3,
                       transform=ax_metrics.transAxes, clip_on=False)

    # REMOVED FOOTER as requested
    # n_failed = metrics["n_failed"]
    # footer = f"N = {n}  |  Failed = {n_failed}  |  TP={tp}  FP={fp}  TN={tn}  FN={fn}"
    # fig.text(0.5, 0.02, footer, ha="center", fontsize=10,
    #          color=text_secondary, fontfamily="monospace")

    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=bg_color, edgecolor="none")
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="COMPASS Validation: Binary Confusion Matrix + Generalized Metrics",
    )
    parser.add_argument("--results_dir", required=True,
                        help="Path to participant_runs directory")
    parser.add_argument("--targets_file", required=False, default="",
                        help="Path to ground-truth annotations file (binary workflow)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for .png files")
    parser.add_argument("--disorder_groups", default="",
                        help="Comma-separated disorder groups for per-group matrices")
    parser.add_argument(
        "--prediction_type",
        default="binary",
        choices=sorted(SUPPORTED_PREDICTION_TYPES),
        help="Task type for metric computation",
    )
    parser.add_argument(
        "--annotations_json",
        default="",
        help="Generalized annotation JSON (required for non-binary modes)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prediction_type = str(args.prediction_type or "binary").strip().lower()

    if prediction_type != "binary":
        if not args.annotations_json:
            print("ERROR: --annotations_json is required for non-binary prediction types.")
            sys.exit(1)
        annotations = load_generalized_annotations(args.annotations_json)
        if not annotations:
            print("ERROR: No annotations loaded from --annotations_json.")
            sys.exit(1)

        rows = []
        results_path = Path(args.results_dir)
        for participant_dir in sorted(results_path.iterdir()):
            if not participant_dir.is_dir():
                continue
            eid_match = re.search(r"ID(\d+)", participant_dir.name)
            if not eid_match:
                continue
            eid = eid_match.group(1)
            truth = annotations.get(eid)
            if truth is None:
                continue
            pred = extract_generalized_prediction(participant_dir)
            if pred is None:
                continue
            rows.append({"eid": eid, "truth": truth, "pred": pred})

        if not rows:
            print("ERROR: No overlapping prediction/annotation rows found.")
            sys.exit(1)

        metrics_payload = {
            "prediction_type": prediction_type,
            "n_rows": len(rows),
        }

        if prediction_type == "multiclass":
            eval_rows = []
            for row in rows:
                truth = row["truth"]
                actual = truth.get("label") or truth.get("classification")
                predicted = row["pred"].get("predicted_label")
                eval_rows.append({"actual": actual, "predicted": predicted})
            metrics_payload["classification_metrics"] = _compute_multiclass_metrics(eval_rows)
        elif prediction_type in {"regression_univariate", "regression_multivariate"}:
            eval_rows = []
            expected_outputs = None
            for row in rows:
                truth = row["truth"]
                actual_values = truth.get("regression") if isinstance(truth.get("regression"), dict) else {}
                if not actual_values and "value" in truth:
                    key = str(truth.get("output_name") or "value")
                    actual_values = {key: truth.get("value")}
                predicted_values = row["pred"].get("regression_values") or {}
                if expected_outputs is None:
                    expected_outputs = sorted(actual_values.keys())
                eval_rows.append({
                    "actual_values": actual_values,
                    "predicted_values": predicted_values,
                })
            metrics_payload["regression_metrics"] = _compute_regression_metrics(eval_rows, expected_outputs=expected_outputs)
        elif prediction_type == "hierarchical":
            eval_rows = []
            for row in rows:
                truth = row["truth"]
                truth_nodes = truth.get("nodes") if isinstance(truth.get("nodes"), dict) else {}
                pred_nodes = row["pred"].get("nodes") if isinstance(row["pred"].get("nodes"), dict) else {}
                eval_rows.append({"truth_nodes": truth_nodes, "pred_nodes": pred_nodes})
            metrics_payload["hierarchical_metrics"] = _compute_hierarchical_metrics(eval_rows)
        else:
            print(f"ERROR: Unsupported non-binary prediction_type: {prediction_type}")
            sys.exit(1)

        out_json = os.path.join(args.output_dir, f"{prediction_type}_metrics.json")
        with open(out_json, "w") as f:
            json.dump(metrics_payload, f, indent=2)
        print(f"✓ Saved generalized metrics: {out_json}")
        print(
            "NOTE: XAI currently supports binary classification only; "
            "non-binary validation here excludes XAI metrics."
        )
        return

    if not args.targets_file:
        print("ERROR: --targets_file is required for binary confusion matrix workflow.")
        sys.exit(1)

    # ── Integrated (all disorders) ──
    print("\n═══ Computing Integrated Confusion Matrix ═══")
    all_results = collect_results(args.results_dir, args.targets_file)
    if not all_results:
        print("ERROR: No results found.")
        sys.exit(1)

    metrics = compute_metrics(all_results)
    plot_confusion_matrix(
        metrics,
        title="COMPASS — Integrated Binary Confusion Matrix",
        output_path=os.path.join(args.output_dir, "integrated_confusion_matrix.png"),
    )

    # Print summary
    print(f"  Accuracy:    {metrics['accuracy']:.1%}")
    print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
    print(f"  Specificity: {metrics['specificity']:.1%}")
    print(f"  Precision:   {metrics['precision']:.1%}")
    print(f"  F1:          {metrics['f1']:.1%}")
    print(f"  MCC:         {metrics['mcc']:+.3f}")

    # ── Per-disorder (if requested) ──
    if args.disorder_groups:
        disorders = [d.strip() for d in args.disorder_groups.split(",") if d.strip()]
        for disorder in disorders:
            print(f"\n═══ {disorder} ═══")
            d_results = collect_results(
                args.results_dir, args.targets_file, disorder_filter=[disorder]
            )
            if not d_results:
                print(f"  ⚠ No results for {disorder}")
                continue

            d_metrics = compute_metrics(d_results)
            safe_name = disorder.lower().replace(" ", "_")
            plot_confusion_matrix(
                d_metrics,
                title=f"COMPASS — {disorder.replace('_', ' ').title()}",
                output_path=os.path.join(args.output_dir, f"{safe_name}_confusion_matrix.png"),
            )
            print(f"  Accuracy: {d_metrics['accuracy']:.1%}  |  N={d_metrics['n_valid']}")

    print("\n✓ Confusion matrix generation complete.")


if __name__ == "__main__":
    main()

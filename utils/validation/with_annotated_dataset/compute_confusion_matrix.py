#!/usr/bin/env python3
"""
COMPASS Clinical Validation — Binary Confusion Matrix Generator.

Computes and plots binary confusion matrices (CASE vs CONTROL) from COMPASS
participant run results, using annotated ground-truth labels.

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
        description="COMPASS Validation: Binary Confusion Matrix Generator",
    )
    parser.add_argument("--results_dir", required=True,
                        help="Path to participant_runs directory")
    parser.add_argument("--targets_file", required=True,
                        help="Path to ground-truth annotations file")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for .png files")
    parser.add_argument("--disorder_groups", default="",
                        help="Comma-separated disorder groups for per-group matrices")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

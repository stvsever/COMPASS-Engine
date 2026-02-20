#!/usr/bin/env python3
"""
COMPASS Clinical Validation — Detailed Performance Analysis.

Generates comprehensive statistical analysis of COMPASS predictions against
annotated ground-truth labels, including:
  - Binary classification metrics (per-disorder + integrated)
  - Generalized analyses for multiclass, regression, and hierarchical tasks
  - Failure analysis
  - Critic composite score vs. prediction accuracy correlation
  - Probability calibration analysis
  - Iteration improvement analysis with significance testing (violin + scatter)
  - Verdict quality distribution

Outputs:
  - detailed_analysis.txt         (comprehensive text report)
  - composite_vs_accuracy.png     (point-biserial correlation plot)
  - probability_calibration.png   (calibration curve + histogram)
  - iteration_improvement.png     (violin plot with significance annotation)
  - verdict_accuracy.png          (verdict category vs accuracy rates)
  - Per-disorder variants of the above (if --disorder_groups specified)

Usage:
  python detailed_analysis.py \\
      --results_dir /path/to/participant_runs \\
      --targets_file /path/to/cases_controls_with_specific_subtypes.txt \\
      --output_dir /path/to/output/details

  # With per-disorder breakdown:
  python detailed_analysis.py \\
      --results_dir /path/to/participant_runs \\
      --targets_file /path/to/cases_controls_with_specific_subtypes.txt \\
      --output_dir /path/to/output/details \\
      --disorder_groups "MAJOR_DEPRESSIVE_DISORDER,ANXIETY_DISORDERS"
"""

import argparse
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime
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

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found — significance tests will be skipped.")

# Import the shared data extraction from our companion script
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
from compute_confusion_matrix import (
    load_ground_truth,
    extract_prediction,
    collect_results,
    compute_metrics,
    load_generalized_annotations,
    extract_generalized_prediction,
    _compute_multiclass_metrics,
    _compute_regression_metrics,
    _compute_hierarchical_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# Plot styling constants
# ═══════════════════════════════════════════════════════════════════════════

BG_COLOR = "#0D1117"
CARD_COLOR = "#161B22"
TEXT_PRIMARY = "#E6EDF3"
TEXT_SECONDARY = "#8B949E"
ACCENT_BLUE = "#58A6FF"
ACCENT_GREEN = "#3FB950"
ACCENT_ORANGE = "#D29922"
ACCENT_RED = "#F85149"
ACCENT_PURPLE = "#BC8CFF"
GRID_COLOR = "#21262D"


def _style_ax(ax, title=""):
    """Apply consistent dark styling to an axis."""
    ax.set_facecolor(CARD_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, alpha=0.15, color=GRID_COLOR)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color=ACCENT_BLUE, pad=12)


# ═══════════════════════════════════════════════════════════════════════════
# Enhanced result extraction (with duration + token info)
# ═══════════════════════════════════════════════════════════════════════════

def extract_extended_result(result_dir: Path) -> Optional[dict]:
    """Extract prediction + performance metadata from results folder.

    Handles multiple report schemas:
      - Schema A (current): data["prediction"], data["evaluation"], data["execution"]
      - Schema B (legacy/future): data["execution_summary"]
    """
    report_files = list(result_dir.glob("report_*.json"))
    perf_files = list(result_dir.glob("performance_report_*.json"))

    info = {}

    # From report
    if report_files:
        try:
            with open(report_files[0], "r") as f:
                report = json.load(f)

            # ── Schema A: top-level "prediction" / "evaluation" / "execution" ──
            pred_block = report.get("prediction", {})
            if isinstance(pred_block, dict):
                pred_raw = pred_block.get("classification")
                if pred_raw:
                    pred_str = str(pred_raw).upper().strip()
                    if "CASE" in pred_str and "CONTROL" not in pred_str:
                        info["prediction"] = "CASE"
                    elif "CONTROL" in pred_str or "NON_CASE" in pred_str or "NOT PSYCHIATRIC" in pred_str:
                        info["prediction"] = "CONTROL"
                    else:
                        prob = pred_block.get("probability")
                        if prob is not None:
                            try:
                                info["prediction"] = "CASE" if float(prob) > 0.5 else "CONTROL"
                            except (ValueError, TypeError):
                                pass
                info["probability"] = pred_block.get("probability") or pred_block.get("probability_score")

            eval_block = report.get("evaluation", {})
            if isinstance(eval_block, dict):
                info["verdict"] = eval_block.get("verdict")
                info["composite_score"] = eval_block.get("composite_score")
                # Derive composite from checklist if not available
                checklist_passed = eval_block.get("checklist_passed")
                checklist_total = eval_block.get("checklist_total")
                if info["composite_score"] is None and checklist_passed is not None and checklist_total is not None:
                    try:
                        if float(checklist_total) > 0:
                            info["composite_score"] = float(checklist_passed) / float(checklist_total)
                    except (ValueError, TypeError):
                        pass

            exec_block = report.get("execution", {})
            if isinstance(exec_block, dict):
                info["iterations"] = exec_block.get("iterations") or exec_block.get("total_iterations")
                info["duration"] = exec_block.get("duration_seconds")
                token_info = exec_block.get("tokens_used", {})
                if isinstance(token_info, dict):
                    info["tokens"] = token_info.get("total_tokens") or token_info.get("total")
                elif isinstance(token_info, (int, float)):
                    info["tokens"] = token_info

                # Per-iteration composites from detailed_logs
                detailed_logs = exec_block.get("detailed_logs", [])
                iter_composites = []
                if isinstance(detailed_logs, list):
                    for att in detailed_logs:
                        if isinstance(att, dict):
                            eval_data = att.get("evaluation", {})
                            if isinstance(eval_data, dict):
                                cs = eval_data.get("composite_score")
                                if cs is not None:
                                    iter_composites.append(float(cs))
                                else:
                                    cp = eval_data.get("checklist_passed")
                                    ct = eval_data.get("checklist_total")
                                    if cp is not None and ct is not None:
                                        try:
                                            if float(ct) > 0:
                                                iter_composites.append(float(cp) / float(ct))
                                        except (ValueError, TypeError):
                                            pass
                info["iter_composites"] = iter_composites

            # ── Schema B: execution_summary fallback ──
            if "prediction" not in info:
                exec_sum = report.get("execution_summary", {})
                if isinstance(exec_sum, dict):
                    final_pred = exec_sum.get("final_prediction", {})
                    if isinstance(final_pred, dict):
                        pred_raw = final_pred.get("classification") or final_pred.get("binary_classification")
                        if pred_raw:
                            pred_str = str(pred_raw).upper().strip()
                            if "CASE" in pred_str and "CONTROL" not in pred_str:
                                info["prediction"] = "CASE"
                            elif "CONTROL" in pred_str or "NON_CASE" in pred_str or "NOT PSYCHIATRIC" in pred_str:
                                info["prediction"] = "CONTROL"
                        info.setdefault("probability", final_pred.get("probability"))

                    final_eval = exec_sum.get("final_evaluation", {})
                    if isinstance(final_eval, dict):
                        info.setdefault("verdict", final_eval.get("verdict"))
                        info.setdefault("composite_score", final_eval.get("composite_score"))

                    info.setdefault("iterations", exec_sum.get("iterations"))
                    info.setdefault("duration", exec_sum.get("duration_seconds"))
                    info.setdefault("tokens", exec_sum.get("total_tokens"))

                    attempts = exec_sum.get("attempts") or exec_sum.get("iteration_details") or []
                    if not info.get("iter_composites") and isinstance(attempts, list):
                        iter_composites = []
                        for att in attempts:
                            if isinstance(att, dict):
                                eval_data = att.get("evaluation", {})
                                if isinstance(eval_data, dict):
                                    cs = eval_data.get("composite_score")
                                    if cs is not None:
                                        iter_composites.append(float(cs))
                        info["iter_composites"] = iter_composites

        except (json.JSONDecodeError, IOError):
            pass

    # From performance report (fallback / enrichment)
    if perf_files:
        try:
            with open(perf_files[0], "r") as f:
                perf = json.load(f)
            if "prediction" not in info:
                pred_raw = perf.get("prediction") or perf.get("classification")
                if pred_raw:
                    pred_str = str(pred_raw).upper().strip()
                    if "CASE" in pred_str and "CONTROL" not in pred_str:
                        info["prediction"] = "CASE"
                    elif "CONTROL" in pred_str or "NON_CASE" in pred_str:
                        info["prediction"] = "CONTROL"

            if info.get("composite_score") is None:
                info["composite_score"] = perf.get("composite_score")
            if info.get("verdict") is None:
                info["verdict"] = perf.get("verdict")
            if info.get("tokens") is None:
                token_info = perf.get("token_summary", {})
                info["tokens"] = token_info.get("total_tokens") if isinstance(token_info, dict) else None
            if info.get("duration") is None:
                info["duration"] = perf.get("duration_seconds")
        except (json.JSONDecodeError, IOError):
            pass

    # Cast types
    for k in ["probability", "composite_score", "duration"]:
        if info.get(k) is not None:
            try:
                info[k] = float(info[k])
            except (ValueError, TypeError):
                info[k] = None
    for k in ["iterations", "tokens"]:
        if info.get(k) is not None:
            try:
                info[k] = int(info[k])
            except (ValueError, TypeError):
                info[k] = None

    return info if "prediction" in info else None


def collect_extended_results(
    results_dir: str,
    targets_file: str,
    disorder_filter: Optional[List[str]] = None,
) -> List[dict]:
    """Collect extended result data for all participants."""
    ground_truth = load_ground_truth(targets_file)
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        return results

    for participant_dir in sorted(results_path.iterdir()):
        if not participant_dir.is_dir():
            continue

        eid_match = re.search(r"ID(\d+)", participant_dir.name)
        if not eid_match:
            continue
        eid = eid_match.group(1)

        gt = ground_truth.get(eid)
        if gt is None:
            continue
        if disorder_filter and gt["disorder"] not in disorder_filter:
            continue

        info = extract_extended_result(participant_dir)
        entry = {
            "eid": eid,
            "actual": gt["label"],
            "disorder": gt["disorder"],
        }
        if info:
            entry.update(info)
            entry["correct"] = entry.get("prediction") == gt["label"]
            entry["status"] = "SUCCESS"
        else:
            entry["prediction"] = None
            entry["correct"] = False
            entry["status"] = "FAILED"

        results.append(entry)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1: Composite Score vs. Prediction Accuracy
# ═══════════════════════════════════════════════════════════════════════════

def plot_composite_vs_accuracy(results: List[dict], output_path: str, title_suffix: str = ""):
    """Point-biserial correlation between composite score and correct/incorrect."""
    valid = [r for r in results if r.get("composite_score") is not None and r.get("prediction") is not None]
    if len(valid) < 3:
        print(f"  ⚠ Not enough data for composite vs accuracy plot (n={len(valid)})")
        return

    composites = np.array([r["composite_score"] for r in valid])
    correct = np.array([1 if r["correct"] else 0 for r in valid])

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    _style_ax(ax, f"Critic Composite Score vs. Prediction Accuracy{title_suffix}")

    # Scatter with jitter on y-axis for visibility
    jitter = np.random.normal(0, 0.03, size=len(correct))
    colors = [ACCENT_GREEN if c else ACCENT_RED for c in correct]
    ax.scatter(composites, correct + jitter, c=colors, s=60, alpha=0.7, edgecolors="white",
               linewidth=0.5, zorder=5)

    # Point-biserial correlation
    if HAS_SCIPY and len(set(correct)) > 1:
        r_pb, p_val = scipy_stats.pointbiserialr(correct, composites)
        # Logistic-like trend line
        x_range = np.linspace(composites.min() - 0.05, composites.max() + 0.05, 100)
        mean_correct = np.mean(correct)
        slope = r_pb * 2  # visual approximation
        trend = mean_correct + slope * (x_range - np.mean(composites))
        trend = np.clip(trend, 0, 1)
        ax.plot(x_range, trend, color=ACCENT_BLUE, linewidth=2, alpha=0.7,
                label=f"r_pb = {r_pb:.3f} (p = {p_val:.4f})")
        ax.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY,
                  fontsize=11, loc="lower right")

        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.text(0.02, 0.95, f"Point-biserial r = {r_pb:.3f}  {sig_marker}",
                transform=ax.transAxes, fontsize=12, color=ACCENT_PURPLE,
                fontweight="bold", va="top")

    ax.set_xlabel("Critic Composite Score", fontsize=12)
    ax.set_ylabel("Prediction Correct (1) / Incorrect (0)", fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Incorrect", "Correct"], fontsize=11, color=TEXT_PRIMARY)

    # Legend markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCENT_GREEN, markersize=10, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCENT_RED, markersize=10, label='Incorrect'),
    ]
    ax.legend(handles=legend_elements, facecolor=CARD_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_PRIMARY, fontsize=10, loc="upper left")

    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 2: Probability Calibration
# ═══════════════════════════════════════════════════════════════════════════

def plot_probability_calibration(results: List[dict], output_path: str, title_suffix: str = ""):
    """Predicted probability vs. observed accuracy (calibration curve + histogram)."""
    valid = [r for r in results if r.get("probability") is not None and r.get("prediction") is not None]
    if len(valid) < 5:
        print(f"  ⚠ Not enough data for calibration plot (n={len(valid)})")
        return

    probs = np.array([r["probability"] for r in valid])
    correct = np.array([1 if r["correct"] else 0 for r in valid])

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor(BG_COLOR)

    # ── Calibration curve ──
    _style_ax(ax_cal, f"Probability Calibration{title_suffix}")

    n_bins = min(10, max(3, len(valid) // 5))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracy = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1] + (1e-9 if i == n_bins - 1 else 0))
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accuracy.append(correct[mask].mean())
            bin_counts.append(mask.sum())

    # Perfect calibration line
    ax_cal.plot([0, 1], [0, 1], "--", color=TEXT_SECONDARY, alpha=0.5, linewidth=1,
                label="Perfect calibration")
    # Actual calibration
    if bin_centers:
        ax_cal.plot(bin_centers, bin_accuracy, "o-", color=ACCENT_BLUE,
                    linewidth=2, markersize=8, label="COMPASS", zorder=5)

    ax_cal.set_ylabel("Observed Accuracy", fontsize=12)
    ax_cal.set_ylim(-0.05, 1.05)
    ax_cal.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY, fontsize=10)

    # ── Histogram ──
    _style_ax(ax_hist)
    correct_probs = probs[correct == 1]
    incorrect_probs = probs[correct == 0]

    ax_hist.hist(correct_probs, bins=15, alpha=0.6, color=ACCENT_GREEN,
                 label="Correct", edgecolor=CARD_COLOR)
    ax_hist.hist(incorrect_probs, bins=15, alpha=0.6, color=ACCENT_RED,
                 label="Incorrect", edgecolor=CARD_COLOR)
    ax_hist.set_xlabel("Predicted Probability", fontsize=12)
    ax_hist.set_ylabel("Count", fontsize=12)
    ax_hist.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY, fontsize=10)

    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 3: Iteration Improvement (Violin + Scatter + Significance)
# ═══════════════════════════════════════════════════════════════════════════

def plot_iteration_improvement(results: List[dict], output_path: str, title_suffix: str = ""):
    """Violin plot comparing composite scores across iterations with significance."""
    # Collect per-iteration composites
    iter_data = defaultdict(list)
    for r in results:
        composites = r.get("iter_composites", [])
        for idx, cs in enumerate(composites):
            iter_data[idx + 1].append(cs)

    if len(iter_data) < 1:
        print(f"  ⚠ No per-iteration composite data available")
        return

    # Sort by iteration number
    iterations = sorted(iter_data.keys())
    data_arrays = [np.array(iter_data[it]) for it in iterations]
    labels = [f"Iteration {it}" for it in iterations]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG_COLOR)
    _style_ax(ax, f"Composite Score Across Iterations{title_suffix}")

    positions = list(range(1, len(iterations) + 1))

    # Custom color gradient for iterations
    iter_colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED, ACCENT_PURPLE]

    # ── Violin plot ──
    if all(len(d) >= 2 for d in data_arrays):
        parts = ax.violinplot(data_arrays, positions=positions, showmeans=False,
                              showmedians=False, showextrema=False)
        for idx, body in enumerate(parts.get("bodies", [])):
            color = iter_colors[idx % len(iter_colors)]
            body.set_facecolor(color)
            body.set_alpha(0.25)
            body.set_edgecolor(color)

    # ── Box plot overlay (quartiles) ──
    bp = ax.boxplot(data_arrays, positions=positions, widths=0.15,
                    patch_artist=True, showfliers=False, zorder=3)
    for idx, box in enumerate(bp["boxes"]):
        color = iter_colors[idx % len(iter_colors)]
        box.set_facecolor(color)
        box.set_alpha(0.5)
        box.set_edgecolor(color)
    for element in ["whiskers", "caps"]:
        for item in bp[element]:
            item.set_color(TEXT_SECONDARY)
            item.set_alpha(0.5)
    for median in bp["medians"]:
        median.set_color(TEXT_PRIMARY)
        median.set_linewidth(2)

    # ── Raw scatter with jitter ──
    for idx, (pos, data) in enumerate(zip(positions, data_arrays)):
        color = iter_colors[idx % len(iter_colors)]
        jitter_x = np.random.normal(0, 0.06, size=len(data))
        ax.scatter(pos + jitter_x, data, s=30, alpha=0.6, color=color,
                   edgecolors="white", linewidth=0.3, zorder=4)

    # ── Significance testing between consecutive iterations ──
    if HAS_SCIPY and len(iterations) >= 2:
        y_max = max(d.max() for d in data_arrays if len(d) > 0) * 1.05
        for i in range(len(iterations) - 1):
            d1 = data_arrays[i]
            d2 = data_arrays[i + 1]
            if len(d1) < 2 or len(d2) < 2:
                continue

            # Wilcoxon rank-sum (Mann-Whitney U) for unpaired comparisons
            try:
                stat, p_val = scipy_stats.mannwhitneyu(d1, d2, alternative="two-sided")
            except ValueError:
                continue

            sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            # Draw significance bracket
            x1, x2 = positions[i], positions[i + 1]
            y_bar = y_max + 0.03 * (i + 1)
            bar_height = 0.01

            ax.plot([x1, x1, x2, x2], [y_bar - bar_height, y_bar, y_bar, y_bar - bar_height],
                    color=TEXT_SECONDARY, linewidth=1.2)
            ax.text((x1 + x2) / 2, y_bar + 0.005, f"{sig_str}\np={p_val:.3f}",
                    ha="center", va="bottom", fontsize=9, color=ACCENT_PURPLE,
                    fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11, color=TEXT_PRIMARY)
    ax.set_xlabel("Orchestration Iteration", fontsize=12)
    ax.set_ylabel("Critic Composite Score", fontsize=12)

    # Add sample size annotation
    for pos, data in zip(positions, data_arrays):
        ax.text(pos, ax.get_ylim()[0] + 0.01, f"n={len(data)}",
                ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY)

    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 4: Verdict vs Accuracy
# ═══════════════════════════════════════════════════════════════════════════

def plot_verdict_accuracy(results: List[dict], output_path: str, title_suffix: str = ""):
    """Bar chart of accuracy rates grouped by verdict category."""
    verdict_groups = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r.get("prediction") is None:
            continue
        v = str(r.get("verdict", "UNKNOWN")).upper()
        if "SATISFACTORY" in v and "UNSATISFACTORY" not in v:
            group = "SATISFACTORY"
        elif "UNSATISFACTORY" in v:
            group = "UNSATISFACTORY"
        else:
            group = "UNKNOWN"
        verdict_groups[group]["total"] += 1
        if r.get("correct"):
            verdict_groups[group]["correct"] += 1

    if not verdict_groups:
        print(f"  ⚠ No verdict data available")
        return

    cats = sorted(verdict_groups.keys())
    accuracies = [verdict_groups[c]["correct"] / verdict_groups[c]["total"]
                  if verdict_groups[c]["total"] > 0 else 0 for c in cats]
    counts = [verdict_groups[c]["total"] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_COLOR)
    _style_ax(ax, f"Prediction Accuracy by Critic Verdict{title_suffix}")

    colors_map = {"SATISFACTORY": ACCENT_GREEN, "UNSATISFACTORY": ACCENT_ORANGE, "UNKNOWN": TEXT_SECONDARY}
    bar_colors = [colors_map.get(c, TEXT_SECONDARY) for c in cats]

    bars = ax.bar(cats, accuracies, color=bar_colors, alpha=0.7, edgecolor="white", linewidth=0.5, width=0.5)

    for bar, acc, cnt in zip(bars, accuracies, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.0%}\n(n={cnt})", ha="center", va="bottom",
                fontsize=11, color=TEXT_PRIMARY, fontweight="bold")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Critic Verdict", fontsize=12)

    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Text Report Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_text_report(
    results: List[dict],
    metrics: dict,
    output_path: str,
    disorder_filter: Optional[List[str]] = None,
    title: str = "Integrated",
) -> None:
    """Generate detailed text analysis report."""
    lines = []
    sep = "=" * 80

    lines.append(sep)
    lines.append(f"  COMPASS ENGINE — DETAILED PERFORMANCE ANALYSIS")
    lines.append(f"  {title}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    # ── 1. Cohort Summary ──
    n_total = len(results)
    n_success = sum(1 for r in results if r.get("status") == "SUCCESS")
    n_failed = sum(1 for r in results if r.get("status") == "FAILED")
    n_cases = sum(1 for r in results if r["actual"] == "CASE")
    n_controls = sum(1 for r in results if r["actual"] == "CONTROL")

    lines.append(f"1. COHORT SUMMARY")
    lines.append(f"   Total participants:  {n_total}")
    lines.append(f"   Successfully processed: {n_success}")
    lines.append(f"   Failed executions:   {n_failed}")
    lines.append(f"   Actual CASE:         {n_cases}")
    lines.append(f"   Actual CONTROL:      {n_controls}")
    lines.append("")

    # ── 2. Classification Metrics ──
    lines.append(f"2. BINARY CLASSIFICATION METRICS")
    lines.append(f"   {'Metric':<20} {'Value':>10}")
    lines.append(f"   {'-'*30}")
    lines.append(f"   {'Accuracy':<20} {metrics['accuracy']:>10.1%}")
    lines.append(f"   {'Sensitivity':<20} {metrics['sensitivity']:>10.1%}")
    lines.append(f"   {'Specificity':<20} {metrics['specificity']:>10.1%}")
    lines.append(f"   {'Precision':<20} {metrics['precision']:>10.1%}")
    lines.append(f"   {'F1 Score':<20} {metrics['f1']:>10.1%}")
    lines.append(f"   {'MCC':<20} {metrics['mcc']:>+10.3f}")
    lines.append(f"   {'TP':<20} {metrics['tp']:>10d}")
    lines.append(f"   {'FP':<20} {metrics['fp']:>10d}")
    lines.append(f"   {'TN':<20} {metrics['tn']:>10d}")
    lines.append(f"   {'FN':<20} {metrics['fn']:>10d}")
    lines.append("")

    # ── 3. Per-Disorder Breakdown ──
    disorders = sorted(set(r["disorder"] for r in results))
    if len(disorders) > 1:
        lines.append(f"3. PER-DISORDER BREAKDOWN")
        lines.append(f"   {'Disorder':<35} {'N':>4} {'Acc':>7} {'Sens':>7} {'Spec':>7} {'F1':>7}")
        lines.append(f"   {'-'*70}")
        for d in disorders:
            d_results = [r for r in results if r["disorder"] == d]
            d_m = compute_metrics(d_results)
            lines.append(
                f"   {d:<35} {d_m['n_valid']:>4} "
                f"{d_m['accuracy']:>7.1%} {d_m['sensitivity']:>7.1%} "
                f"{d_m['specificity']:>7.1%} {d_m['f1']:>7.1%}"
            )
        lines.append("")

    # ── 4. Failure Analysis ──
    failed = [r for r in results if r.get("status") == "FAILED"]
    lines.append(f"4. FAILURE ANALYSIS")
    lines.append(f"   Total failed: {len(failed)} / {n_total} ({len(failed)/n_total*100:.1f}%)" if n_total > 0 else "   Total failed: 0")
    if failed:
        lines.append(f"   Failed EIDs:")
        for r in failed:
            lines.append(f"     - {r['eid']} ({r['disorder']}, actual={r['actual']})")
    lines.append("")

    # ── 5. Verdict Quality ──
    sat_correct = sum(1 for r in results if str(r.get("verdict", "")).upper().startswith("SATISF") and r.get("correct"))
    sat_total = sum(1 for r in results if str(r.get("verdict", "")).upper().startswith("SATISF") and r.get("prediction"))
    unsat_correct = sum(1 for r in results if "UNSATISF" in str(r.get("verdict", "")).upper() and r.get("correct"))
    unsat_total = sum(1 for r in results if "UNSATISF" in str(r.get("verdict", "")).upper() and r.get("prediction"))

    lines.append(f"5. VERDICT QUALITY ANALYSIS")
    lines.append(f"   SATISFACTORY verdicts:   {sat_total} (accuracy: {sat_correct/sat_total:.1%})" if sat_total > 0 else "   SATISFACTORY verdicts:   0")
    lines.append(f"   UNSATISFACTORY verdicts: {unsat_total} (accuracy: {unsat_correct/unsat_total:.1%})" if unsat_total > 0 else "   UNSATISFACTORY verdicts: 0")
    lines.append("")

    # ── 6. Composite Score Correlation ──
    valid_composites = [r for r in results if r.get("composite_score") is not None and r.get("prediction")]
    if len(valid_composites) >= 3:
        composites = np.array([r["composite_score"] for r in valid_composites])
        correct_arr = np.array([1 if r["correct"] else 0 for r in valid_composites])

        lines.append(f"6. COMPOSITE SCORE — PREDICTION ACCURACY CORRELATION")
        lines.append(f"   Correct predictions:    mean composite = {composites[correct_arr==1].mean():.3f}" if correct_arr.sum() > 0 else "   Correct predictions:    no data")
        lines.append(f"   Incorrect predictions:  mean composite = {composites[correct_arr==0].mean():.3f}" if (correct_arr==0).sum() > 0 else "   Incorrect predictions:  no data")

        if HAS_SCIPY and len(set(correct_arr)) > 1:
            r_pb, p_val = scipy_stats.pointbiserialr(correct_arr, composites)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            lines.append(f"   Point-biserial r:       {r_pb:+.3f} (p = {p_val:.4f}) {sig}")
            lines.append(f"   Interpretation: {'Higher composite scores are significantly associated with correct predictions' if p_val < 0.05 else 'No significant correlation between composite score and prediction accuracy'}")
        lines.append("")

    # ── 7. Probability Calibration ──
    valid_probs = [r for r in results if r.get("probability") is not None and r.get("prediction")]
    if valid_probs:
        correct_probs = [r["probability"] for r in valid_probs if r.get("correct")]
        incorrect_probs = [r["probability"] for r in valid_probs if not r.get("correct")]

        lines.append(f"7. PROBABILITY CALIBRATION")
        lines.append(f"   Correct predictions:    mean probability = {np.mean(correct_probs):.3f}" if correct_probs else "   Correct predictions:    no data")
        lines.append(f"   Incorrect predictions:  mean probability = {np.mean(incorrect_probs):.3f}" if incorrect_probs else "   Incorrect predictions:  no data")

        if HAS_SCIPY and correct_probs and incorrect_probs:
            stat, p_val = scipy_stats.mannwhitneyu(correct_probs, incorrect_probs, alternative="two-sided")
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            lines.append(f"   Mann-Whitney U test:    U = {stat:.1f}, p = {p_val:.4f} {sig}")
        lines.append("")

    # ── 8. Iteration Analysis ──
    multi_iter = [r for r in results if r.get("iterations") is not None and r["iterations"] > 1]
    single_iter = [r for r in results if r.get("iterations") is not None and r["iterations"] == 1]
    if multi_iter or single_iter:
        lines.append(f"8. ITERATION ANALYSIS")
        lines.append(f"   Single iteration runs:   {len(single_iter)}")
        lines.append(f"   Multi-iteration runs:    {len(multi_iter)}")
        if multi_iter:
            multi_correct = sum(1 for r in multi_iter if r.get("correct"))
            single_correct = sum(1 for r in single_iter if r.get("correct"))
            lines.append(f"   Single-iter accuracy:    {single_correct/len(single_iter):.1%}" if single_iter else "")
            lines.append(f"   Multi-iter accuracy:     {multi_correct/len(multi_iter):.1%}")

            # Per-iteration composite improvement
            iter_data = defaultdict(list)
            for r in results:
                for idx, cs in enumerate(r.get("iter_composites", [])):
                    iter_data[idx + 1].append(cs)

            if len(iter_data) >= 2:
                lines.append(f"\n   Per-iteration composite scores:")
                for it in sorted(iter_data.keys()):
                    arr = np.array(iter_data[it])
                    lines.append(f"     Iteration {it}: mean = {arr.mean():.3f}, std = {arr.std():.3f}, n = {len(arr)}")

                if HAS_SCIPY:
                    sorted_iters = sorted(iter_data.keys())
                    for i in range(len(sorted_iters) - 1):
                        d1 = iter_data[sorted_iters[i]]
                        d2 = iter_data[sorted_iters[i + 1]]
                        if len(d1) >= 2 and len(d2) >= 2:
                            stat, p_val = scipy_stats.mannwhitneyu(d1, d2, alternative="two-sided")
                            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                            lines.append(f"     Iter {sorted_iters[i]} → {sorted_iters[i+1]}: U = {stat:.1f}, p = {p_val:.4f} {sig}")
        lines.append("")

    # ── 9. Resource Usage ──
    durations = [r["duration"] for r in results if r.get("duration")]
    tokens_list = [r["tokens"] for r in results if r.get("tokens")]
    if durations or tokens_list:
        lines.append(f"9. RESOURCE USAGE")
        if durations:
            lines.append(f"   Duration (sec):  mean = {np.mean(durations):.0f}, median = {np.median(durations):.0f}, max = {np.max(durations):.0f}")
        if tokens_list:
            lines.append(f"   Total tokens:    mean = {np.mean(tokens_list):.0f}, median = {np.median(tokens_list):.0f}, max = {np.max(tokens_list):.0f}")
        lines.append("")

    # ── 10. Per-Participant Results Table ──
    lines.append(f"10. PER-PARTICIPANT RESULTS")
    header = f"   {'EID':<12} {'Disorder':<30} {'Actual':<9} {'Pred':<9} {'Prob':>6} {'Verdict':<15} {'Comp':>6} {'Iter':>5} {'Dur(s)':>7}"
    lines.append(header)
    lines.append(f"   {'-' * len(header)}")

    for r in sorted(results, key=lambda x: x["eid"]):
        pred = r.get("prediction", "FAIL")
        prob = f"{r['probability']:.2f}" if r.get("probability") is not None else "  -"
        verdict = str(r.get("verdict", "-"))[:14]
        comp = f"{r['composite_score']:.2f}" if r.get("composite_score") is not None else "  -"
        itr = str(r.get("iterations", "-"))
        dur = f"{r['duration']:.0f}" if r.get("duration") is not None else "  -"
        marker = " ✓" if r.get("correct") else " ✗" if r.get("prediction") else " !"
        lines.append(
            f"   {r['eid']:<12} {r['disorder']:<30} {r['actual']:<9} {pred:<9} {prob:>6} {verdict:<15} {comp:>6} {itr:>5} {dur:>7}{marker}"
        )
    lines.append("")
    lines.append(sep)

    report_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"  ✓ Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis(results_dir: str, targets_file: str, output_dir: str,
                 disorder_filter: Optional[List[str]] = None,
                 title_suffix: str = ""):
    """Run complete analysis for a set of results."""
    results = collect_extended_results(results_dir, targets_file, disorder_filter)
    if not results:
        print(f"  ⚠ No results found{title_suffix}")
        return

    # Compute metrics for successful predictions
    valid_results = [r for r in results if r.get("prediction") is not None]
    metrics = compute_metrics(valid_results)

    # Generate text report
    generate_text_report(
        results, metrics,
        output_path=os.path.join(output_dir, f"detailed_analysis{title_suffix.replace(' ', '_').lower()}.txt"),
        disorder_filter=disorder_filter,
        title=f"Analysis{title_suffix}" if title_suffix else "Integrated Analysis",
    )

    # Generate plots
    plot_composite_vs_accuracy(
        results,
        output_path=os.path.join(output_dir, f"composite_vs_accuracy{title_suffix.replace(' ', '_').lower()}.png"),
        title_suffix=title_suffix,
    )
    plot_probability_calibration(
        results,
        output_path=os.path.join(output_dir, f"probability_calibration{title_suffix.replace(' ', '_').lower()}.png"),
        title_suffix=title_suffix,
    )
    plot_iteration_improvement(
        results,
        output_path=os.path.join(output_dir, f"iteration_improvement{title_suffix.replace(' ', '_').lower()}.png"),
        title_suffix=title_suffix,
    )
    plot_verdict_accuracy(
        results,
        output_path=os.path.join(output_dir, f"verdict_accuracy{title_suffix.replace(' ', '_').lower()}.png"),
        title_suffix=title_suffix,
    )


def run_generalized_analysis(
    results_dir: str,
    annotations_json: str,
    output_dir: str,
    prediction_type: str,
):
    """Run generalized non-binary analysis and emit JSON/TXT summaries."""
    annotations = load_generalized_annotations(annotations_json)
    rows = []
    for participant_dir in sorted(Path(results_dir).iterdir()):
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
        print("  ⚠ No overlapping prediction/annotation rows found for generalized analysis.")
        return

    payload = {"prediction_type": prediction_type, "n_rows": len(rows)}
    if prediction_type == "multiclass":
        eval_rows = []
        for row in rows:
            truth = row["truth"]
            eval_rows.append({
                "actual": truth.get("label") or truth.get("classification"),
                "predicted": row["pred"].get("predicted_label"),
            })
        payload["classification_metrics"] = _compute_multiclass_metrics(eval_rows)
    elif prediction_type in {"regression_univariate", "regression_multivariate"}:
        eval_rows = []
        expected_outputs = None
        for row in rows:
            truth = row["truth"]
            actual_values = truth.get("regression") if isinstance(truth.get("regression"), dict) else {}
            if not actual_values and "value" in truth:
                key = str(truth.get("output_name") or "value")
                actual_values = {key: truth.get("value")}
            if expected_outputs is None:
                expected_outputs = sorted(actual_values.keys())
            eval_rows.append({
                "actual_values": actual_values,
                "predicted_values": row["pred"].get("regression_values") or {},
            })
        payload["regression_metrics"] = _compute_regression_metrics(eval_rows, expected_outputs=expected_outputs)
    else:
        eval_rows = []
        for row in rows:
            truth_nodes = row["truth"].get("nodes") if isinstance(row["truth"].get("nodes"), dict) else {}
            pred_nodes = row["pred"].get("nodes") if isinstance(row["pred"].get("nodes"), dict) else {}
            eval_rows.append({"truth_nodes": truth_nodes, "pred_nodes": pred_nodes})
        payload["hierarchical_metrics"] = _compute_hierarchical_metrics(eval_rows)

    json_path = os.path.join(output_dir, f"detailed_analysis_{prediction_type}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    txt_path = os.path.join(output_dir, f"detailed_analysis_{prediction_type}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Prediction type: {prediction_type}\n")
        f.write(f"Rows evaluated: {len(rows)}\n\n")
        f.write(json.dumps(payload, indent=2))
        f.write("\n\n")
        f.write("Note: XAI metrics are excluded for non-binary task modes.\n")

    print(f"  ✓ Saved generalized analysis: {json_path}")
    print(f"  ✓ Saved generalized summary: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="COMPASS Validation: Detailed Performance Analysis",
    )
    parser.add_argument("--results_dir", required=True,
                        help="Path to participant_runs directory")
    parser.add_argument("--targets_file", required=False, default="",
                        help="Path to ground-truth annotations file (binary mode)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for analysis files")
    parser.add_argument("--disorder_groups", default="",
                        help="Comma-separated disorder groups for per-group analysis")
    parser.add_argument(
        "--prediction_type",
        default="binary",
        choices=["binary", "multiclass", "regression_univariate", "regression_multivariate", "hierarchical"],
        help="Task type for analysis",
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
            print("ERROR: --annotations_json is required for non-binary analysis.")
            sys.exit(1)
        print(f"\n═══ Running Generalized Analysis ({prediction_type}) ═══")
        run_generalized_analysis(
            results_dir=args.results_dir,
            annotations_json=args.annotations_json,
            output_dir=args.output_dir,
            prediction_type=prediction_type,
        )
        print(
            "\n✓ Generalized detailed analysis complete.\n"
            "Note: XAI metrics are not included for non-binary prediction types."
        )
        return

    if not args.targets_file:
        print("ERROR: --targets_file is required for binary detailed analysis.")
        sys.exit(1)

    # ── Integrated analysis ──
    print("\n═══ Running Integrated Analysis ═══")
    run_analysis(args.results_dir, args.targets_file, args.output_dir)

    # ── Per-disorder analysis ──
    if args.disorder_groups:
        disorders = [d.strip() for d in args.disorder_groups.split(",") if d.strip()]
        for disorder in disorders:
            safe_suffix = f" — {disorder.replace('_', ' ').title()}"
            print(f"\n═══ {disorder} ═══")
            run_analysis(
                args.results_dir, args.targets_file, args.output_dir,
                disorder_filter=[disorder],
                title_suffix=safe_suffix,
            )

    print("\n✓ Detailed analysis complete.")


if __name__ == "__main__":
    main()

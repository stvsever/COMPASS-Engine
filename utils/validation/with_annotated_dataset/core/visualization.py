"""Visualization utilities for annotated validation across prediction modes."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .constants import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    ACCENT_RED,
    BG_COLOR,
    CARD_COLOR,
    GRID_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for validation visualizations") from exc

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False


def _style_ax(ax, title: str = "") -> None:
    ax.set_facecolor(CARD_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.18, linewidth=0.7)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    if title:
        ax.set_title(title, fontsize=13, color=ACCENT_BLUE, fontweight="bold", pad=10)


def _save_fig(fig, output_path: str) -> None:
    fig.patch.set_facecolor(BG_COLOR)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)


def _point_density_values(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return normalized density values for each point (0..1)."""
    n = len(x)
    if n < 8:
        return np.ones(n, dtype=float)

    if HAS_SCIPY:
        try:
            xy = np.vstack([x, y])
            kde = scipy_stats.gaussian_kde(xy)
            dens = kde(xy)
            dens = np.asarray(dens, dtype=float)
            dmin = float(np.min(dens))
            dmax = float(np.max(dens))
            if dmax > dmin:
                return (dens - dmin) / (dmax - dmin)
            return np.ones(n, dtype=float)
        except Exception:
            pass

    # Histogram fallback.
    bins = int(min(50, max(10, np.sqrt(n))))
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if np.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    x_idx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1)
    y_idx = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1)
    dens = hist[x_idx, y_idx].astype(float)
    dmin = float(np.min(dens))
    dmax = float(np.max(dens))
    if dmax > dmin:
        return (dens - dmin) / (dmax - dmin)
    return np.ones(n, dtype=float)


def _density_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    s: float = 24.0,
    alpha: float = 0.9,
    cmap: str = "viridis",
) -> Tuple[np.ndarray, Any]:
    dens = _point_density_values(x, y)
    order = np.argsort(dens)
    x_ord = x[order]
    y_ord = y[order]
    d_ord = dens[order]
    sc = ax.scatter(x_ord, y_ord, c=d_ord, cmap=cmap, s=s, alpha=alpha, edgecolors="none")
    return d_ord, sc


def plot_binary_confusion_matrix(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    tp = int(metrics.get("tp") or 0)
    fn = int(metrics.get("fn") or 0)
    fp = int(metrics.get("fp") or 0)
    tn = int(metrics.get("tn") or 0)

    matrix = np.array([[tp, fn], [fp, tn]], dtype=float)
    n = int(np.sum(matrix))

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    _style_ax(ax, title)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "compass_binary", ["#0D1117", "#0E4429", "#26A641", "#39D353"]
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(1.0, float(np.max(matrix))))

    labels = ["CASE", "CONTROL"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred\n{x}" for x in labels], color=TEXT_PRIMARY, fontweight="bold")
    ax.set_yticklabels([f"Actual\n{x}" for x in labels], color=TEXT_PRIMARY, fontweight="bold")

    for i in range(2):
        for j in range(2):
            count = int(matrix[i, j])
            pct = (100.0 * count / n) if n > 0 else 0.0
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=TEXT_PRIMARY,
                fontsize=12,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_SECONDARY)

    _save_fig(fig, output_path)


def plot_multiclass_confusion_matrix(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    labels = [str(x) for x in metrics.get("labels") or []]
    matrix = np.array(metrics.get("matrix") or [], dtype=float)
    if matrix.size == 0 or len(labels) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(7.5, 0.9 * len(labels) + 4), max(6.5, 0.7 * len(labels) + 3)))
    _style_ax(ax, title)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "compass_multi", ["#0D1117", "#1F6FEB", "#2EA043", "#39D353"]
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(1.0, float(np.max(matrix))))

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", color=TEXT_PRIMARY, fontsize=9)
    ax.set_yticklabels(labels, color=TEXT_PRIMARY, fontsize=9)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    n = float(np.sum(matrix))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = int(matrix[i, j])
            if count == 0:
                continue
            pct = 100.0 * count / n if n > 0 else 0.0
            ax.text(j, i, f"{count}\n{pct:.1f}%", ha="center", va="center", color=TEXT_PRIMARY, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_SECONDARY)

    _save_fig(fig, output_path)


def plot_multiclass_per_class(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
    if not per_class:
        return

    labels = sorted(per_class.keys())
    precision = [float(per_class[l].get("precision") or 0.0) for l in labels]
    recall = [float(per_class[l].get("recall") or 0.0) for l in labels]
    f1 = [float(per_class[l].get("f1") or 0.0) for l in labels]
    support = [int(per_class[l].get("support") or 0) for l in labels]

    x = np.arange(len(labels), dtype=float)
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(labels) + 3), 6.2))
    _style_ax(ax, title)

    ax.bar(x - width, precision, width=width, color=ACCENT_BLUE, alpha=0.8, label="Precision")
    ax.bar(x, recall, width=width, color=ACCENT_GREEN, alpha=0.8, label="Recall")
    ax.bar(x + width, f1, width=width, color=ACCENT_ORANGE, alpha=0.8, label="F1")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", color=TEXT_PRIMARY, fontsize=9)
    ax.set_ylabel("Score")

    ax2 = ax.twinx()
    ax2.plot(x, support, color=ACCENT_PURPLE, marker="o", linewidth=1.6, label="Support")
    ax2.set_ylabel("Support", color=TEXT_SECONDARY)
    ax2.tick_params(axis="y", colors=TEXT_SECONDARY)
    for spine in ax2.spines.values():
        spine.set_color(GRID_COLOR)

    lines, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels1 + labels2, fontsize=9, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)

    _save_fig(fig, output_path)


def plot_multiclass_confidence_calibration(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    calib = metrics.get("confidence_calibration") if isinstance(metrics.get("confidence_calibration"), dict) else {}
    bins = calib.get("bins") if isinstance(calib.get("bins"), list) else []
    if not bins:
        return

    x = [float(b.get("mean_confidence") or 0.0) for b in bins]
    y = [float(b.get("observed_accuracy") or 0.0) for b in bins]
    n = [int(b.get("n") or 0) for b in bins]

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    _style_ax(ax, title)

    ax.plot([0, 1], [0, 1], "--", color=TEXT_SECONDARY, alpha=0.7, linewidth=1.2, label="Perfect calibration")
    sc = ax.scatter(x, y, s=[max(24, 8 + k * 2) for k in n], color=ACCENT_BLUE, alpha=0.9, label="Observed")
    ax.plot(x, y, color=ACCENT_BLUE, alpha=0.45)

    ece = calib.get("ece")
    brier = calib.get("brier_top_label")
    meta = []
    if ece is not None:
        meta.append(f"ECE={float(ece):.3f}")
    if brier is not None:
        meta.append(f"Brier={float(brier):.3f}")
    if meta:
        ax.text(0.03, 0.95, "  |  ".join(meta), transform=ax.transAxes, va="top", color=ACCENT_PURPLE, fontsize=10)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.legend(fontsize=9, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)

    _save_fig(fig, output_path)


def plot_multiclass_top_confusions(metrics: Dict[str, Any], title: str, output_path: str, top_n: int = 12) -> None:
    confusions = metrics.get("top_confusions") if isinstance(metrics.get("top_confusions"), list) else []
    if not confusions:
        return

    confusions = confusions[:top_n]
    labels = [f"{row.get('actual')} -> {row.get('predicted')}" for row in confusions]
    counts = [int(row.get("count") or 0) for row in confusions]

    y = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(labels) + 4.2), max(5.0, 0.35 * len(labels) + 2.2)))
    _style_ax(ax, title)

    bars = ax.barh(y, counts, color=ACCENT_ORANGE, alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT_PRIMARY, fontsize=9)
    ax.set_xlabel("Count")
    ax.invert_yaxis()

    for yi, c in zip(y, counts):
        ax.text(c + 0.1, yi, str(int(c)), va="center", color=TEXT_PRIMARY, fontsize=9)

    _save_fig(fig, output_path)


def plot_multiclass_confidence_diagnostics(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    diag = metrics.get("probability_diagnostics") if isinstance(metrics.get("probability_diagnostics"), dict) else {}
    conf_correct = np.array(diag.get("confidence_correct") or [], dtype=float)
    conf_incorrect = np.array(diag.get("confidence_incorrect") or [], dtype=float)
    ent_correct = np.array(diag.get("entropy_correct") or [], dtype=float)
    ent_incorrect = np.array(diag.get("entropy_incorrect") or [], dtype=float)
    if (len(conf_correct) + len(conf_incorrect)) == 0:
        return

    fig, (ax_conf, ax_ent) = plt.subplots(1, 2, figsize=(12.0, 5.2))
    _style_ax(ax_conf, f"{title} — Confidence")
    _style_ax(ax_ent, f"{title} — Normalized entropy")

    data_conf = [conf_correct, conf_incorrect]
    labels = ["Correct", "Incorrect"]
    colors = [ACCENT_GREEN, ACCENT_RED]

    bp = ax_conf.boxplot(data_conf, tick_labels=labels, patch_artist=True, showfliers=False)
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.5)
        box.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color(TEXT_PRIMARY)
        median.set_linewidth(1.6)
    ax_conf.set_ylim(-0.02, 1.02)
    ax_conf.set_ylabel("Predicted confidence")

    if len(ent_correct) + len(ent_incorrect) > 0:
        data_ent = [ent_correct if len(ent_correct) > 0 else np.array([0.0]), ent_incorrect if len(ent_incorrect) > 0 else np.array([0.0])]
        bp2 = ax_ent.boxplot(data_ent, tick_labels=labels, patch_artist=True, showfliers=False)
        for box, color in zip(bp2["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.5)
            box.set_edgecolor(color)
        for median in bp2["medians"]:
            median.set_color(TEXT_PRIMARY)
            median.set_linewidth(1.6)
    ax_ent.set_ylim(-0.02, 1.02)
    ax_ent.set_ylabel("Entropy (0=confident, 1=diffuse)")

    topk = diag.get("top_k_accuracy") if isinstance(diag.get("top_k_accuracy"), dict) else {}
    txt_parts = []
    for key in ("top1", "top2", "top3"):
        value = topk.get(key)
        if value is not None:
            txt_parts.append(f"{key}={float(value):.1%}")
    if txt_parts:
        fig.text(0.5, 0.01, "  |  ".join(txt_parts), ha="center", color=ACCENT_PURPLE, fontsize=10)

    _save_fig(fig, output_path)


def plot_multiclass_label_distribution(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    labels = [str(x) for x in metrics.get("labels") or []]
    matrix = np.array(metrics.get("matrix") or [], dtype=float)
    if len(labels) == 0 or matrix.size == 0:
        return

    true_counts = matrix.sum(axis=1)
    pred_counts = matrix.sum(axis=0)
    x = np.arange(len(labels), dtype=float)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(labels) + 3), 5.8))
    _style_ax(ax, title)

    ax.bar(x - width / 2, true_counts, width=width, color=ACCENT_GREEN, alpha=0.78, label="True count")
    ax.bar(x + width / 2, pred_counts, width=width, color=ACCENT_BLUE, alpha=0.78, label="Predicted count")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", color=TEXT_PRIMARY, fontsize=9)
    ax.set_ylabel("Count")
    ax.legend(fontsize=9, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)

    _save_fig(fig, output_path)


def plot_binary_composite_vs_accuracy(rows: Sequence[Dict[str, Any]], title: str, output_path: str) -> None:
    valid = [r for r in rows if r.get("composite_score") is not None and r.get("predicted") in {"CASE", "CONTROL"}]
    if len(valid) < 3:
        return

    comps = np.array([float(r["composite_score"]) for r in valid], dtype=float)
    corr = np.array([1.0 if bool(r.get("correct")) else 0.0 for r in valid], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _style_ax(ax, title)

    rng = np.random.default_rng(42)
    jitter = rng.normal(0.0, 0.03, size=len(corr))
    colors = [ACCENT_GREEN if c > 0.5 else ACCENT_RED for c in corr]
    ax.scatter(comps, corr + jitter, color=colors, alpha=0.75, s=45, edgecolors="white", linewidth=0.4)

    if HAS_SCIPY and len(set(corr.tolist())) > 1:
        r_pb, p_val = scipy_stats.pointbiserialr(corr, comps)
        xs = np.linspace(float(np.min(comps)) - 0.02, float(np.max(comps)) + 0.02, 120)
        trend = float(np.mean(corr)) + float(r_pb) * 1.75 * (xs - float(np.mean(comps)))
        trend = np.clip(trend, 0.0, 1.0)
        ax.plot(xs, trend, color=ACCENT_BLUE, linewidth=1.8, alpha=0.8, label=f"r_pb={r_pb:.3f}, p={p_val:.4f}")
        ax.legend(fontsize=9, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY, loc="lower right")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Incorrect", "Correct"], color=TEXT_PRIMARY)
    ax.set_xlabel("Critic composite score")
    ax.set_ylabel("Prediction correctness")

    _save_fig(fig, output_path)


def plot_binary_probability_calibration(rows: Sequence[Dict[str, Any]], title: str, output_path: str) -> None:
    valid = [r for r in rows if r.get("probability") is not None and r.get("predicted") in {"CASE", "CONTROL"}]
    if len(valid) < 5:
        return

    probs = np.clip(np.array([float(r["probability"]) for r in valid], dtype=float), 0.0, 1.0)
    y_true = np.array([1.0 if r.get("actual") == "CASE" else 0.0 for r in valid], dtype=float)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(8.8, 7.2),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
        sharex=True,
    )
    _style_ax(ax_top, title)
    _style_ax(ax_bot)

    edges = np.linspace(0.0, 1.0, 11)
    centers = []
    observed = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (probs >= lo) & (probs < hi + (1e-12 if i == len(edges) - 2 else 0.0))
        if not np.any(mask):
            continue
        centers.append(float(np.mean(probs[mask])))
        observed.append(float(np.mean(y_true[mask])))

    ax_top.plot([0, 1], [0, 1], "--", color=TEXT_SECONDARY, alpha=0.65, linewidth=1.2, label="Perfect")
    if centers:
        ax_top.plot(centers, observed, "o-", color=ACCENT_BLUE, linewidth=1.9, label="Observed")
    ax_top.set_ylabel("Observed CASE rate")
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.legend(fontsize=9, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)

    correct_probs = probs[np.array([bool(r.get("correct")) for r in valid], dtype=bool)]
    incorrect_probs = probs[~np.array([bool(r.get("correct")) for r in valid], dtype=bool)]
    ax_bot.hist(correct_probs, bins=12, alpha=0.65, color=ACCENT_GREEN, edgecolor=CARD_COLOR, label="Correct")
    ax_bot.hist(incorrect_probs, bins=12, alpha=0.65, color=ACCENT_RED, edgecolor=CARD_COLOR, label="Incorrect")
    ax_bot.set_xlabel("Predicted CASE probability")
    ax_bot.set_ylabel("Count")
    ax_bot.legend(fontsize=9, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)

    _save_fig(fig, output_path)


def plot_binary_iteration_improvement(rows: Sequence[Dict[str, Any]], title: str, output_path: str) -> None:
    iter_data: Dict[int, List[float]] = {}
    for r in rows:
        comps = r.get("iter_composites") if isinstance(r.get("iter_composites"), list) else []
        for idx, value in enumerate(comps):
            fv = float(value)
            iter_data.setdefault(idx + 1, []).append(fv)

    if not iter_data:
        return

    order = sorted(iter_data.keys())
    arrays = [np.array(iter_data[i], dtype=float) for i in order]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    _style_ax(ax, title)

    positions = list(range(1, len(order) + 1))
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED, ACCENT_PURPLE]

    if all(len(a) >= 2 for a in arrays):
        parts = ax.violinplot(arrays, positions=positions, showmeans=False, showmedians=False, showextrema=False)
        for i, body in enumerate(parts.get("bodies", [])):
            body.set_facecolor(colors[i % len(colors)])
            body.set_edgecolor(colors[i % len(colors)])
            body.set_alpha(0.22)

    bp = ax.boxplot(arrays, positions=positions, widths=0.18, patch_artist=True, showfliers=False)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i % len(colors)])
        box.set_edgecolor(colors[i % len(colors)])
        box.set_alpha(0.45)
    for median in bp["medians"]:
        median.set_color(TEXT_PRIMARY)
        median.set_linewidth(1.8)

    rng = np.random.default_rng(7)
    for i, arr in enumerate(arrays):
        jitter = rng.normal(0.0, 0.05, size=len(arr))
        ax.scatter(np.full_like(arr, positions[i], dtype=float) + jitter, arr, color=colors[i % len(colors)], alpha=0.6, s=24)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Iteration {i}" for i in order], color=TEXT_PRIMARY)
    ax.set_ylabel("Composite score")

    _save_fig(fig, output_path)


def plot_binary_verdict_accuracy(rows: Sequence[Dict[str, Any]], title: str, output_path: str) -> None:
    groups: Dict[str, Dict[str, int]] = {}
    for r in rows:
        if r.get("predicted") not in {"CASE", "CONTROL"}:
            continue
        verdict = str(r.get("verdict") or "UNKNOWN").upper()
        if "UNSATISFACTORY" in verdict:
            key = "UNSATISFACTORY"
        elif "SATISFACTORY" in verdict:
            key = "SATISFACTORY"
        else:
            key = "UNKNOWN"
        groups.setdefault(key, {"correct": 0, "total": 0})
        groups[key]["total"] += 1
        groups[key]["correct"] += 1 if bool(r.get("correct")) else 0

    if not groups:
        return

    cats = sorted(groups.keys())
    acc = [groups[c]["correct"] / groups[c]["total"] if groups[c]["total"] > 0 else 0.0 for c in cats]
    n = [groups[c]["total"] for c in cats]

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    _style_ax(ax, title)

    color_map = {"SATISFACTORY": ACCENT_GREEN, "UNSATISFACTORY": ACCENT_ORANGE, "UNKNOWN": TEXT_SECONDARY}
    bars = ax.bar(cats, acc, color=[color_map.get(c, ACCENT_BLUE) for c in cats], alpha=0.8)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Accuracy")

    for bar, a, count in zip(bars, acc, n):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{a:.0%}\n(n={count})",
            ha="center",
            va="bottom",
            color=TEXT_PRIMARY,
            fontsize=10,
            fontweight="bold",
        )

    _save_fig(fig, output_path)


def plot_regression_parity(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    per_output = metrics.get("per_output") if isinstance(metrics.get("per_output"), dict) else {}
    outputs = [k for k, v in per_output.items() if int(v.get("n") or 0) > 0]
    if not outputs:
        return

    n_out = len(outputs)
    n_cols = min(3, n_out)
    n_rows = int(np.ceil(n_out / float(n_cols)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.2 * n_rows), squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)

    for idx, output_name in enumerate(outputs):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        _style_ax(ax, output_name)

        pairs = per_output[output_name].get("raw_pairs") if isinstance(per_output[output_name].get("raw_pairs"), dict) else {}
        y_true = np.array(pairs.get("y_true") or [], dtype=float)
        y_pred = np.array(pairs.get("y_pred") or [], dtype=float)
        if len(y_true) == 0:
            continue

        dens, sc = _density_scatter(ax, y_true, y_pred, s=28.0, alpha=0.9)
        lo = float(min(np.min(y_true), np.min(y_pred)))
        hi = float(max(np.max(y_true), np.max(y_pred)))
        if np.isclose(lo, hi):
            lo -= 1.0
            hi += 1.0
        ax.plot([lo, hi], [lo, hi], "--", color=ACCENT_GREEN, alpha=0.7, linewidth=1.3)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

        mae = per_output[output_name].get("mae")
        r2 = per_output[output_name].get("r2")
        txt = []
        if mae is not None:
            txt.append(f"MAE={float(mae):.3f}")
        if r2 is not None:
            txt.append(f"R²={float(r2):.3f}")
        if txt:
            ax.text(0.03, 0.96, " | ".join(txt), transform=ax.transAxes, va="top", color=ACCENT_PURPLE, fontsize=9)

        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("Point density", color=TEXT_SECONDARY, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY, labelsize=7)
        plt.setp(cbar.ax.get_yticklabels(), color=TEXT_SECONDARY)

    # Hide unused axes.
    for idx in range(n_out, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")
        axes[r][c].set_facecolor(BG_COLOR)

    fig.suptitle(title, color=ACCENT_BLUE, fontsize=13, fontweight="bold")
    _save_fig(fig, output_path)


def plot_regression_error_bars(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    per_output = metrics.get("per_output") if isinstance(metrics.get("per_output"), dict) else {}
    labels = [k for k, v in per_output.items() if int(v.get("n") or 0) > 0]
    if not labels:
        return

    mae = [float(per_output[k].get("mae") or 0.0) for k in labels]
    rmse = [float(per_output[k].get("rmse") or 0.0) for k in labels]

    x = np.arange(len(labels), dtype=float)
    width = 0.34

    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(labels) + 3), 5.8))
    _style_ax(ax, title)

    ax.bar(x - width / 2, mae, width=width, color=ACCENT_ORANGE, alpha=0.85, label="MAE")
    ax.bar(x + width / 2, rmse, width=width, color=ACCENT_RED, alpha=0.72, label="RMSE")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", color=TEXT_PRIMARY)
    ax.set_ylabel("Error")
    ax.legend(fontsize=9, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)

    _save_fig(fig, output_path)


def plot_regression_residual_distribution(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    per_output = metrics.get("per_output") if isinstance(metrics.get("per_output"), dict) else {}
    outputs = [k for k, v in per_output.items() if len(v.get("raw_residuals") or []) > 0]
    if not outputs:
        return

    n_out = len(outputs)
    n_cols = min(3, n_out)
    n_rows = int(np.ceil(n_out / float(n_cols)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.8 * n_rows), squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)

    for idx, output_name in enumerate(outputs):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        _style_ax(ax, output_name)

        residuals = np.array(per_output[output_name].get("raw_residuals") or [], dtype=float)
        if len(residuals) == 0:
            continue
        ax.hist(residuals, bins=min(20, max(8, len(residuals) // 2)), color=ACCENT_ORANGE, alpha=0.75, edgecolor=CARD_COLOR)
        ax.axvline(0.0, color=ACCENT_GREEN, linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_xlabel("Residual (pred - true)")
        ax.set_ylabel("Count")

    for idx in range(n_out, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")
        axes[r][c].set_facecolor(BG_COLOR)

    fig.suptitle(title, color=ACCENT_BLUE, fontsize=13, fontweight="bold")
    _save_fig(fig, output_path)


def plot_regression_residuals_vs_truth(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    per_output = metrics.get("per_output") if isinstance(metrics.get("per_output"), dict) else {}
    outputs = [k for k, v in per_output.items() if int(v.get("n") or 0) > 0]
    if not outputs:
        return

    n_out = len(outputs)
    n_cols = min(3, n_out)
    n_rows = int(np.ceil(n_out / float(n_cols)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.9 * n_rows), squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)

    for idx, output_name in enumerate(outputs):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        _style_ax(ax, output_name)

        pairs = per_output[output_name].get("raw_pairs") if isinstance(per_output[output_name].get("raw_pairs"), dict) else {}
        y_true = np.array(pairs.get("y_true") or [], dtype=float)
        y_pred = np.array(pairs.get("y_pred") or [], dtype=float)
        if len(y_true) == 0:
            continue
        residuals = y_pred - y_true
        _, sc = _density_scatter(ax, y_true, residuals, s=24.0, alpha=0.9)
        ax.axhline(0.0, color=ACCENT_GREEN, linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_xlabel("True")
        ax.set_ylabel("Residual")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("Point density", color=TEXT_SECONDARY, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY, labelsize=7)
        plt.setp(cbar.ax.get_yticklabels(), color=TEXT_SECONDARY)

    for idx in range(n_out, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")
        axes[r][c].set_facecolor(BG_COLOR)

    fig.suptitle(title, color=ACCENT_BLUE, fontsize=13, fontweight="bold")
    _save_fig(fig, output_path)


def plot_regression_top_errors(metrics: Dict[str, Any], title: str, output_path: str, top_n: int = 15) -> None:
    rows = metrics.get("largest_absolute_errors") if isinstance(metrics.get("largest_absolute_errors"), list) else []
    if not rows:
        return

    rows = rows[:top_n]
    labels = [f"{row.get('eid')}:{row.get('output')}" for row in rows]
    abs_err = [float(row.get("abs_error") or 0.0) for row in rows]

    y = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.5, 0.5 * len(labels) + 4), max(5.0, 0.35 * len(labels) + 2.2)))
    _style_ax(ax, title)

    ax.barh(y, abs_err, color=ACCENT_RED, alpha=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT_PRIMARY, fontsize=9)
    ax.set_xlabel("Absolute error")
    ax.invert_yaxis()

    for yi, e in zip(y, abs_err):
        ax.text(e + 0.01, yi, f"{e:.3f}", va="center", color=TEXT_PRIMARY, fontsize=8)

    _save_fig(fig, output_path)


def plot_hierarchical_node_scores(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    per_node = metrics.get("per_node") if isinstance(metrics.get("per_node"), dict) else {}
    if not per_node:
        return

    nodes = sorted(per_node.keys())
    scores = []
    support = []
    colors = []

    for node in nodes:
        block = per_node[node] if isinstance(per_node[node], dict) else {}
        mode = str(block.get("mode") or "")
        n = int(block.get("n") or 0)
        support.append(n)

        if mode.endswith("classification"):
            score = block.get("accuracy")
            colors.append(ACCENT_GREEN)
        else:
            score = block.get("macro_r2")
            colors.append(ACCENT_BLUE)

        s_val = 0.0 if score is None else float(max(0.0, min(1.0, score)))
        scores.append(s_val)

    y = np.arange(len(nodes), dtype=float)

    fig, (ax_score, ax_support) = plt.subplots(1, 2, figsize=(12.0, max(5.2, 0.45 * len(nodes) + 2.3)), gridspec_kw={"width_ratios": [2.0, 1.2]})
    _style_ax(ax_score, f"{title} — Node Score")
    _style_ax(ax_support, f"{title} — Node Support")

    ax_score.barh(y, scores, color=colors, alpha=0.8)
    ax_score.set_yticks(y)
    ax_score.set_yticklabels(nodes, color=TEXT_PRIMARY, fontsize=9)
    ax_score.set_xlim(0.0, 1.05)
    ax_score.set_xlabel("Score (accuracy or clipped R²)")

    for yi, sv in zip(y, scores):
        ax_score.text(sv + 0.01, yi, f"{sv:.2f}", va="center", color=TEXT_PRIMARY, fontsize=8)

    ax_support.barh(y, support, color=ACCENT_PURPLE, alpha=0.78)
    ax_support.set_yticks(y)
    ax_support.set_yticklabels([])
    ax_support.set_xlabel("n")
    for yi, n in zip(y, support):
        ax_support.text(n + 0.1, yi, str(int(n)), va="center", color=TEXT_PRIMARY, fontsize=8)

    _save_fig(fig, output_path)


def plot_hierarchical_mode_distribution(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    n_class = int(metrics.get("n_classification_nodes") or 0)
    n_reg = int(metrics.get("n_regression_nodes") or 0)
    if (n_class + n_reg) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    _style_ax(ax, title)

    labels = ["classification", "regression"]
    vals = [n_class, n_reg]
    bars = ax.bar(labels, vals, color=[ACCENT_GREEN, ACCENT_BLUE], alpha=0.82, width=0.55)
    ax.set_ylabel("Node count")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(val), ha="center", va="bottom", color=TEXT_PRIMARY, fontsize=10, fontweight="bold")

    _save_fig(fig, output_path)


def plot_hierarchical_node_coverage(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    per_node = metrics.get("per_node") if isinstance(metrics.get("per_node"), dict) else {}
    if not per_node:
        return

    nodes = sorted(per_node.keys())
    truth_vals = [int((per_node[n].get("coverage") or {}).get("truth_present") or 0) for n in nodes]
    pred_vals = [int((per_node[n].get("coverage") or {}).get("pred_present") or 0) for n in nodes]
    both_vals = [int((per_node[n].get("coverage") or {}).get("both_present") or 0) for n in nodes]
    if max(truth_vals + pred_vals + both_vals, default=0) <= 0:
        return

    y = np.arange(len(nodes), dtype=float)
    h = 0.24

    fig, ax = plt.subplots(figsize=(12.0, max(5.6, 0.42 * len(nodes) + 2.1)))
    _style_ax(ax, title)

    ax.barh(y + h, truth_vals, height=h, color=ACCENT_PURPLE, alpha=0.72, label="truth present")
    ax.barh(y, pred_vals, height=h, color=ACCENT_BLUE, alpha=0.72, label="pred present")
    ax.barh(y - h, both_vals, height=h, color=ACCENT_GREEN, alpha=0.86, label="both present")

    ax.set_yticks(y)
    ax.set_yticklabels(nodes, color=TEXT_PRIMARY, fontsize=9)
    ax.set_xlabel("Node instances")
    ax.legend(fontsize=9, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY, loc="lower right")

    _save_fig(fig, output_path)


def plot_hierarchical_metric_heatmap(metrics: Dict[str, Any], title: str, output_path: str) -> None:
    per_node = metrics.get("per_node") if isinstance(metrics.get("per_node"), dict) else {}
    if not per_node:
        return

    nodes = sorted(per_node.keys())
    if not nodes:
        return

    support = np.array([float(per_node[n].get("n") or 0.0) for n in nodes], dtype=float)
    support_norm = support / np.max(support) if np.max(support) > 0 else np.zeros_like(support)

    score_vals: List[float] = []
    cov_vals: List[float] = []
    for n in nodes:
        block = per_node[n] if isinstance(per_node[n], dict) else {}
        mode = str(block.get("mode") or "")
        if mode.endswith("classification"):
            s = block.get("accuracy")
        elif mode.endswith("regression"):
            s = block.get("macro_r2")
        else:
            s = None
        score_vals.append(0.0 if s is None else float(max(0.0, min(1.0, s))))
        cov = (block.get("coverage") or {}).get("coverage_rate") if isinstance(block.get("coverage"), dict) else None
        cov_vals.append(0.0 if cov is None else float(max(0.0, min(1.0, cov))))

    data = np.vstack([np.array(score_vals), np.array(cov_vals), support_norm]).T
    labels = ["Score", "Coverage", "Support(norm)"]

    fig, ax = plt.subplots(figsize=(8.8, max(5.6, 0.36 * len(nodes) + 2.1)))
    _style_ax(ax, title)
    cmap = mcolors.LinearSegmentedColormap.from_list("compass_heat", ["#0D1117", "#1F6FEB", "#2EA043", "#39D353"])
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_yticklabels(nodes, color=TEXT_PRIMARY, fontsize=9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, color=TEXT_PRIMARY, fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color=TEXT_PRIMARY, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Normalized value", color=TEXT_SECONDARY, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY, labelsize=8)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_SECONDARY)

    _save_fig(fig, output_path)

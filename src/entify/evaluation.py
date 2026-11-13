import json
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import pearsonr

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")


def load_ground_truth(annotation_path):
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    names = {category["id"]: category["name"] for category in coco["categories"]}
    files = {image["id"]: image["file_name"] for image in coco["images"]}
    pairs = [
        (files[annotation["image_id"]], names[annotation["category_id"]])
        for annotation in coco["annotations"]
        if not annotation["iscrowd"]
    ]
    return pairs, sorted(files.values())


def count_grid(pairs, images, categories):
    grid = pd.DataFrame(0, index=images, columns=categories)
    for (image, category), count in Counter(pairs).items():
        grid.loc[image, category] = count
    return grid


def predicted_pairs(predictions, threshold, class_names):
    return [
        (image, class_names[label])
        for image, detections in predictions.items()
        for label, score in zip(detections.labels, detections.scores)
        if score >= threshold
    ]


def evaluate(truth_pairs, predictions, thresholds, class_names):
    images, categories = list(predictions), list(class_names)
    true = count_grid(truth_pairs, images, categories)
    summary, by_category, predicted_grids = [], [], {}
    for threshold in thresholds:
        predicted = count_grid(predicted_pairs(predictions, threshold, class_names), images, categories)
        error = predicted - true
        present = (true + predicted).to_numpy() > 0
        correlation = pearsonr(true.to_numpy()[present], predicted.to_numpy()[present]).statistic
        summary.append({
            "threshold": threshold,
            "true objects": int(true.to_numpy().sum()),
            "predicted objects": int(predicted.to_numpy().sum()),
            "mean absolute error per image": error.abs().sum(axis=1).mean(),
            "mean error per image": error.sum(axis=1).mean(),
            "pearson r": correlation,
        })
        for category in categories:
            if true[category].sum() or predicted[category].sum():
                by_category.append({
                    "threshold": threshold,
                    "category": category,
                    "true": int(true[category].sum()),
                    "predicted": int(predicted[category].sum()),
                    "mean absolute error": error[category].abs().mean(),
                    "mean error": error[category].mean(),
                })
        predicted_grids[threshold] = predicted
    return pd.DataFrame(summary), pd.DataFrame(by_category), true, predicted_grids


def best_threshold(summary):
    return summary.loc[summary["mean absolute error per image"].idxmin(), "threshold"]


def styled_axes(figure):
    figure.set_facecolor(SURFACE)
    axes = figure.add_subplot()
    axes.set_facecolor(SURFACE)
    axes.grid(color=GRIDLINE, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
    axes.tick_params(colors=MUTED_INK, labelcolor=SECONDARY_INK)
    return axes


def plot_error_by_threshold(summary, path):
    figure = Figure(figsize=(7, 4.2))
    axes = styled_axes(figure)
    thresholds = summary["threshold"]
    lines = [("mean absolute error", "mean absolute error per image"), ("mean error (bias)", "mean error per image")]
    for (label, column), color in zip(lines, SERIES):
        axes.plot(thresholds, summary[column], color=color, linewidth=2, marker="o", markersize=6, label=label)
        axes.annotate(label, (thresholds.iloc[-1], summary[column].iloc[-1]), xytext=(8, 0),
                      textcoords="offset points", va="center", color=SECONDARY_INK, fontsize=9)
    axes.axhline(0, color=BASELINE, linewidth=1)
    best = best_threshold(summary)
    axes.axvline(best, color=MUTED_INK, linewidth=1, linestyle=":")
    axes.annotate(f"lowest error at {best:.2f}", (best, axes.get_ylim()[1]), xytext=(4, -12),
                  textcoords="offset points", color=SECONDARY_INK, fontsize=9)
    axes.set_xlabel("confidence threshold", color=SECONDARY_INK)
    axes.set_ylabel("objects per image", color=SECONDARY_INK)
    axes.set_title("Counting error by confidence threshold", color=PRIMARY_INK, loc="left")
    axes.legend(frameon=False, labelcolor=SECONDARY_INK, loc="lower left")
    axes.set_xticks(thresholds)
    axes.set_xlim(thresholds.min() - 0.03, thresholds.max() + 0.15)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def plot_predicted_vs_true(true, predicted, threshold, path):
    figure = Figure(figsize=(5.5, 5.5))
    axes = styled_axes(figure)
    present = (true + predicted).to_numpy() > 0
    true_counts, predicted_counts = true.to_numpy()[present], predicted.to_numpy()[present]
    limit = max(true_counts.max(), predicted_counts.max()) + 1
    axes.plot([0, limit], [0, limit], color=BASELINE, linewidth=1)
    axes.scatter(true_counts, predicted_counts, s=36, color=SERIES[0], alpha=0.55, edgecolors=SURFACE, linewidths=1)
    correlation = pearsonr(true_counts, predicted_counts).statistic
    axes.text(0.03, 0.97, f"{present.sum()} image–category pairs\nPearson r = {correlation:.2f}",
              transform=axes.transAxes, va="top", color=SECONDARY_INK, fontsize=9)
    axes.set_xlim(-0.5, limit)
    axes.set_ylim(-0.5, limit)
    axes.set_xlabel("true count", color=SECONDARY_INK)
    axes.set_ylabel("predicted count", color=SECONDARY_INK)
    axes.set_title(f"Predicted vs. true counts at threshold {threshold:.2f}", color=PRIMARY_INK, loc="left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def summary_table(summary):
    return summary.round({"mean absolute error per image": 2, "mean error per image": 2, "pearson r": 3})


def worst_categories(by_category, threshold, count=5):
    rows = by_category[np.isclose(by_category["threshold"], threshold)]
    return rows.nlargest(count, "mean absolute error").round({"mean absolute error": 2, "mean error": 2})

"""Compare models on one category (default: person) with the original COCO annotations as ground truth.

The images come from COCO val2017 and leave out every image a model could have been trained or checked on: both
COCO subsets of this repository (reviewed and mixed into fine-tuning) and every image listed in a fine-tuned model's
entify.json.
"""
import argparse
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from entify import __version__
from entify.cli import DEFAULT_THRESHOLDS, find_objects, fraction, progress
from entify.images import load_image
from entify.models import DEFAULT_MODEL, ModelError, load_model

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_DIR = BENCHMARK_DIR.parent
DATA_DIR = BENCHMARK_DIR / "data"
RESULTS_FILE = BENCHMARK_DIR / "model_comparison.md"
USED_SUBSETS = [REPO_DIR / "data" / "coco_subset", DATA_DIR / "coco_subset"]
SCORE_FLOOR = 0.05            # detections down to this score are kept, for the precision-recall curve
BANDS = [(1, 2), (3, 6), (7, None)]  # objects per image, the images with the category are spread evenly over these

sys.path.insert(0, str(REPO_DIR / "scripts"))
import fetch_coco_subset as coco_files  # noqa: E402  (download and caching of COCO val2017)


# ---------- held-out images ----------

def used_images(models):
    """File names of COCO images that a model may have seen: the repository's subsets and fine-tuning images."""
    names, sources = set(), []
    for subset in USED_SUBSETS:
        annotations = subset / "annotations.json"
        if annotations.exists():
            found = {image["file_name"] for image in json.loads(annotations.read_text(encoding="utf-8"))["images"]}
            names |= found
            sources.append(f"{len(found)} from `{subset.relative_to(REPO_DIR).as_posix()}`")
    for spec in models:
        settings = Path(spec) / "entify.json"
        if settings.exists():
            images = json.loads(settings.read_text(encoding="utf-8")).get("training", {}).get("images", {})
            found = {Path(path).name for group in images.values() for path in group}
            names |= found
            sources.append(f"{len(found)} used to fine-tune `{Path(spec).name}`")
    return names, sources


def select_images(coco, category_id, count, without, seed, excluded):
    """Seeded choice of images without crowd regions: `without` images with no object of the category, the rest
    spread evenly over 1-2, 3-6 and 7 or more objects of it (topped up from any band if one runs short)."""
    annotations = coco_files.usable_images(coco)
    names = {image["id"]: image["file_name"] for image in coco["images"]}
    ids = sorted(image_id for image_id in annotations if names[image_id] not in excluded)
    random.Random(seed).shuffle(ids)
    objects = {image_id: sum(a["category_id"] == category_id for a in annotations[image_id]) for image_id in ids}

    with_category = count - without
    picked = []
    for index, (low, high) in enumerate(BANDS):
        share = with_category // len(BANDS) + (1 if index < with_category % len(BANDS) else 0)
        band = [i for i in ids if objects[i] >= low and (high is None or objects[i] <= high) and i not in picked]
        picked += band[:share]
    picked += [i for i in ids if objects[i] > 0 and i not in picked][:with_category - len(picked)]
    picked += [i for i in ids if objects[i] == 0][:without]
    return picked, annotations


def holdout_dataset(args, excluded):
    coco = coco_files.load_annotations(DATA_DIR / "cache")
    category_id = next((c["id"] for c in coco["categories"] if c["name"] == args.category), None)
    if category_id is None:
        raise SystemExit(f"compare_models: {args.category!r} is not a COCO category")
    picked, annotations = select_images(coco, category_id, args.images, args.without, args.seed, excluded)
    out_dir = DATA_DIR / f"{args.category.replace(' ', '_')}_holdout"
    images_by_id = {image["id"]: image for image in coco["images"]}
    images = [images_by_id[i] for i in picked]
    print(f"Downloading missing images to {out_dir}")
    for image in progress(images):
        target = out_dir / "images" / image["file_name"]
        if not target.exists():
            coco_files.download(image["coco_url"], target)
    subset = {
        "info": coco["info"], "licenses": coco["licenses"], "categories": coco["categories"],
        "images": images, "annotations": [a for i in picked for a in annotations[i]],
    }
    (out_dir / "annotations.json").write_text(json.dumps(subset), encoding="utf-8")
    coco_files.write_sources(out_dir / "SOURCES.md", coco, images)
    truth = {image["file_name"]: [] for image in images}
    for annotation in subset["annotations"]:
        if annotation["category_id"] == category_id:
            x, y, width, height = annotation["bbox"]
            truth[images_by_id[annotation["image_id"]]["file_name"]].append([x, y, x + width, y + height])
    return out_dir, {name: np.array(boxes, dtype=float).reshape(-1, 4) for name, boxes in truth.items()}


# ---------- scoring ----------

def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def match(boxes, scores, truth, min_iou):
    """True positive flag per detection: greedy by score, each true box matched at most once (as in COCO).

    Because the matching goes by descending score, the flags of the detections above any threshold are the same as
    when only those detections were matched, so one matching serves every threshold.
    """
    hits = np.zeros(len(scores), dtype=bool)
    free = np.ones(len(truth), dtype=bool)
    for index in np.argsort(-scores, kind="stable"):
        if not free.any():
            break
        overlaps = np.where(free, iou(boxes[index], truth), -1) if len(truth) else np.array([])
        if len(overlaps) and overlaps.max() >= min_iou:
            hits[index] = True
            free[overlaps.argmax()] = False
    return hits


def average_precision(scores, hits, positives):
    """Area under the precision-recall curve with COCO's 101-point interpolation."""
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    true_positives = np.cumsum(hits[order])
    precision = true_positives / np.arange(1, len(order) + 1)
    recall = true_positives / positives
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    points = np.linspace(0, 1, 101)
    indices = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in indices]))


def score_model(spec, args, dataset, truth):
    model = load_model(spec, args.labels, SCORE_FLOOR, args.overlap)
    if args.category not in model.categories:
        raise SystemExit(f"compare_models: model {spec} has no category {args.category!r}")
    label = model.categories.index(args.category)
    started = time.perf_counter()
    per_image = {}
    for name in progress(list(truth)):
        detections = find_objects(model, load_image(dataset / "images" / name), args).only(label)
        per_image[name] = (detections.scores, match(detections.boxes, detections.scores, truth[name], args.iou))
    seconds = time.perf_counter() - started

    scores = np.concatenate([s for s, _ in per_image.values()])
    hits = np.concatenate([h for _, h in per_image.values()])
    positives = sum(len(boxes) for boxes in truth.values())
    rows = []
    for threshold in args.thresholds:
        tp = int(hits[scores >= threshold].sum())
        predicted = int((scores >= threshold).sum())
        errors = np.array([(s >= threshold).sum() - len(truth[name]) for name, (s, _) in per_image.items()])
        precision = tp / predicted if predicted else float("nan")
        recall = tp / positives if positives else float("nan")
        rows.append({
            "threshold": threshold, "predicted": predicted, "correct": tp,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if predicted and tp else 0.0,
            "mae": float(np.abs(errors).mean()), "bias": float(errors.mean()),
        })
    return {
        "name": Path(spec).name if Path(spec).exists() else spec,
        "spec": spec,
        "ap50": average_precision(scores, hits, positives),
        "rows": rows,
        "seconds": seconds,
    }


# ---------- report ----------

def write_report(args, results, truth, sources, dataset):
    counts = np.array([len(boxes) for boxes in truth.values()])
    bands = ", ".join(
        f"{int(((counts >= low) & ((counts <= high) if high else True)).sum())} with "
        f"{low}{f'-{high}' if high and high != low else '' if high else ' or more'}"
        for low, high in BANDS
    )
    category = args.category
    settings = [
        ("Date", date.today().isoformat()),
        ("Entify version", __version__),
        ("Category", category),
        ("Ground truth", f"original COCO val2017 annotations, {int(counts.sum())} {category} boxes"),
        ("Images", f"{len(truth)} from COCO val2017 without crowd regions (seed {args.seed}): "
                   f"{int((counts == 0).sum())} without {category}, {bands} {category} objects"),
        ("Left out", "; ".join(sources) if sources else "nothing"),
        ("Match", f"same category and IoU >= {args.iou}, each true box at most once"),
        ("Overlap (IoU) for duplicates", str(args.overlap)),
        ("Tiling", "on" if args.tile else "off"),
        ("Sharpening", f"amount {args.sharpen}" if args.sharpen else "off"),
    ]
    lines = [
        f"# Model comparison: {category}",
        "",
        f"How well each model finds {category} objects on images none of them was trained or checked on, compared "
        f"with the original COCO annotations.",
        "",
        "| Setting | Value |",
        "|---|---|",
        *(f"| {name} | {value} |" for name, value in settings),
        "",
        "## Summary",
        "",
        "| Model | AP50 | Best F1 | at threshold | Precision | Recall | Counting MAE | Seconds per image |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        best = max(result["rows"], key=lambda row: row["f1"])
        lines.append(
            f"| `{result['name']}` | {result['ap50']:.3f} | {best['f1']:.3f} | {best['threshold']:.2f} "
            f"| {best['precision']:.3f} | {best['recall']:.3f} | {best['mae']:.2f} "
            f"| {result['seconds'] / len(truth):.2f} |"
        )
    lines += [
        "",
        "## By confidence threshold",
        "",
        f"| Threshold | Model | Predicted (true: {int(counts.sum())}) | Correct | Precision | Recall | F1 "
        "| Counting MAE | Counting bias |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for index, threshold in enumerate(args.thresholds):
        for result in results:
            row = result["rows"][index]
            lines.append(
                f"| {threshold:.2f} | `{result['name']}` | {row['predicted']} | {row['correct']} "
                f"| {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['mae']:.2f} "
                f"| {row['bias']:+.2f} |"
            )
    lines += [
        "",
        "## How to read this",
        "",
        f"- Precision: share of the predicted {category} boxes that match a true one. Recall: share of the true boxes "
        "that were found. F1 combines both. Higher is better.",
        f"- AP50: average precision over all thresholds from {SCORE_FLOOR} up, at IoU {args.iou} (COCO's 101-point "
        "method), one number for the whole precision-recall curve. Higher is better.",
        "- Counting MAE: mean absolute difference between predicted and true count per image, lower is better. "
        "Counting bias: positive means too many, negative too few.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python benchmarks/compare_models.py " + " ".join(f"-m {result['spec']}" for result in results)
        + (f" --category \"{category}\"" if category != "person" else "")
        + (f" --images {args.images} --without {args.without}" if (args.images, args.without) != (200, 50) else "")
        + (f" --seed {args.seed}" if args.seed else ""),
        "```",
        "",
        f"Images and annotations: `{dataset.relative_to(REPO_DIR).as_posix()}` (not in the repository).",
        "",
    ]
    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Compare models on one COCO category, against the original COCO annotations, on images none of "
                    "them was trained on.")
    parser.add_argument("-m", "--model", dest="models", action="append", metavar="MODEL",
                        help=f"model to compare, repeat for several (default: {DEFAULT_MODEL} and every fine-tuned "
                             f"model in models/)")
    parser.add_argument("--category", default="person", help="COCO category to compare (default: person)")
    parser.add_argument("--images", type=int, default=200, help="number of images (default: 200)")
    parser.add_argument("--without", type=int, default=50,
                        help="of those, images without the category, for false detections (default: 50)")
    parser.add_argument("--seed", type=int, default=0, help="seed for choosing the images (default: 0)")
    parser.add_argument("--thresholds", type=fraction, nargs="+", default=DEFAULT_THRESHOLDS, metavar="VALUE")
    parser.add_argument("--iou", type=fraction, default=0.5, help="overlap a box needs to count as found (default: 0.5)")
    parser.add_argument("--labels", type=Path, metavar="FILE")
    parser.add_argument("-o", "--overlap", type=fraction, default=0.5)
    parser.add_argument("-t", "--tile", action="store_true")
    parser.add_argument("--sharpen", type=float, nargs="?", const=1.0, metavar="AMOUNT")
    args = parser.parse_args()
    args.thresholds = sorted(set(args.thresholds))
    if args.models is None:
        args.models = [DEFAULT_MODEL] + sorted(
            str(path.relative_to(REPO_DIR).as_posix()) for path in (REPO_DIR / "models").glob("*_ft*")
            if (path / "entify.json").exists()
        )
    if not 0 <= args.without <= args.images:
        raise SystemExit("compare_models: --without must be between 0 and --images")

    excluded, sources = used_images(args.models)
    dataset, truth = holdout_dataset(args, excluded)
    results = []
    for spec in args.models:
        print(f"\nScoring {spec}")
        try:
            results.append(score_model(spec, args, dataset, truth))
        except ModelError as error:
            raise SystemExit(f"compare_models: {error}")
    write_report(args, results, truth, sources, dataset)
    print(f"\nResults written to {RESULTS_FILE}")
    print((REPO_DIR / RESULTS_FILE).read_text(encoding="utf-8").split("## By confidence")[0].split("## Summary")[1])


if __name__ == "__main__":
    main()

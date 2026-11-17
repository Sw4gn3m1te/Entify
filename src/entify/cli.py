import argparse
from pathlib import Path

from tqdm import tqdm

from entify import __version__
from entify import evaluation
from entify.counting import count_by_category, totals
from entify.detector import COCO_CLASSES, DEFAULT_PRESET, Detector
from entify.highlighting import render
from entify.images import find_images, load_image, save_image
from entify.sharpening import sharpen
from entify.tiling import detect_tiled

DEFAULT_CSV = Path("outputs/results.csv")
DEFAULT_OUT = Path("outputs")
DEFAULT_DATASET = Path("data/coco_subset")
DEFAULT_EVALUATION_OUT = Path("outputs/evaluation")
DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ALL_SORTS = "*"


def fraction(text):
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{text} is not between 0 and 1")
    return value


def progress(items):
    # One bar over all images, hidden for a single image and when the output is not a terminal.
    return tqdm(items, desc="images", unit="img", disable=None if len(items) > 1 else True)


def find_objects(detector, image, args):
    if args.sharpen:
        image = sharpen(image, args.sharpen)
    return detect_tiled(detector, image, args.overlap) if args.tile else detector.detect(image)


def predict_dataset(detector, dataset, files, args, threshold):
    predictions = {}
    for n, name in enumerate(progress(files), 1):
        predictions[name] = find_objects(detector, load_image(dataset / "images" / name), args)
        tqdm.write(f"[{n}/{len(files)}] {name}: {len(predictions[name].labels)} objects at {threshold}")
    return predictions


def process_images(args, handle_image, classes=None):
    paths = find_images(args.path)
    if not paths:
        raise SystemExit(f"entify: no images found in {args.path}")

    print(f"Loading model {args.preset}")
    detector = Detector(args.preset, args.confidence, args.overlap, args.verbose, classes)
    results = {}
    for n, path in enumerate(progress(paths), 1):
        image = load_image(path)
        results[path.name] = find_objects(detector, image, args)
        tqdm.write(f"[{n}/{len(paths)}] {path.name}: {handle_image(path, image, results[path.name])}")
    return count_by_category(results, COCO_CLASSES)


def save_csv(counts, path):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(path, index=False)
        print(f"\nCounts written to {path}")


def detect(args):
    counts = process_images(args, lambda path, image, detections: f"{len(detections.labels)} objects")
    print()
    print(totals(counts).to_string() if not counts.empty else "No objects detected.")
    save_csv(counts, args.csv)


def highlight(args):
    all_sorts = ALL_SORTS in args.sort
    sorts = list(COCO_CLASSES) if all_sorts else list(dict.fromkeys(args.sort))
    suffix = "all" if all_sorts else "_".join(sort.replace(" ", "_") for sort in sorts)
    classes = None if all_sorts else [COCO_CLASSES.index(sort) for sort in sorts]

    def save_highlight(path, image, detections):
        groups = [(sort, detections.only(COCO_CLASSES.index(sort))) for sort in sorts]
        if all_sorts:
            groups = [(sort, selected) for sort, selected in groups if len(selected.labels)]
        target = args.out / f"{path.stem}_{suffix}.jpg"
        save_image(render(image, groups, args.print_confidence, args.dim), target)
        found = ", ".join(f"{len(selected.labels)} {sort}" for sort, selected in groups) or "nothing found"
        return f"{found} -> {target}"

    counts = process_images(args, save_highlight, classes)
    save_csv(counts, args.csv)


def evaluate(args):
    truth, files = evaluation.load_ground_truth(args.path / "annotations.json")
    thresholds = sorted(set(args.thresholds))
    print(f"Loading model {args.preset}")
    detector = Detector(args.preset, thresholds[0], args.overlap, args.verbose)
    predictions = predict_dataset(detector, args.path, files, args, thresholds[0])

    summary, by_category, true, predicted = evaluation.evaluate(truth, predictions, thresholds, COCO_CLASSES)
    best = evaluation.best_threshold(summary)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    by_category.to_csv(args.out / "errors_by_category.csv", index=False)
    evaluation.plot_error_by_threshold(summary, args.out / "error_by_threshold.png")
    evaluation.plot_predicted_vs_true(true, predicted[best], best, args.out / "predicted_vs_true.png")

    print()
    print(evaluation.summary_table(summary).to_string(index=False))
    print(f"\nLowest counting error at threshold {best:.2f}. Largest errors there:")
    print(evaluation.worst_categories(by_category, best).to_string(index=False))
    print(f"\nResults written to {args.out}")


def build_parser():
    parser = argparse.ArgumentParser(prog="entify", description="Detect, count and highlight objects in images.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    model_options = argparse.ArgumentParser(add_help=False)
    model_options.add_argument(
        "-o", "--overlap", type=fraction, default=0.5,
        help="maximum overlap (IoU) between two detections before the weaker one is removed, 1 keeps all (default: 0.5)",
    )
    model_options.add_argument(
        "-p", "--preset", default=DEFAULT_PRESET,
        help=f"KerasHub preset (default: {DEFAULT_PRESET})",
    )
    model_options.add_argument(
        "-t", "--tile", action="store_true",
        help="detect in overlapping 640 px tiles to find small objects in large images",
    )
    model_options.add_argument(
        "--sharpen", type=float, nargs="?", const=1.0, metavar="AMOUNT",
        help="sharpen images before detection with an unsharp mask (default amount: 1.0)",
    )
    model_options.add_argument("-v", "--verbose", action="store_true", help="show TensorFlow and KerasHub log messages")

    input_options = argparse.ArgumentParser(add_help=False)
    input_options.add_argument("path", type=Path, help="image file or folder of images")
    input_options.add_argument(
        "-c", "--confidence", type=fraction, default=0.5,
        help="minimum confidence score of a detection (default: 0.5)",
    )
    input_options.add_argument(
        "--csv", type=Path, nargs="?", const=DEFAULT_CSV, metavar="FILE",
        help=f"write counts per image and category to a CSV file (default file: {DEFAULT_CSV})",
    )

    detect_parser = commands.add_parser(
        "detect", parents=[input_options, model_options],
        help="detect and count objects in an image or a folder of images",
    )
    detect_parser.set_defaults(handler=detect)

    highlight_parser = commands.add_parser(
        "highlight", parents=[input_options, model_options],
        help="draw boxes around all objects of the chosen categories",
    )
    highlight_parser.add_argument(
        "-s", "--sort", required=True, action="extend", nargs="+", choices=[ALL_SORTS, *sorted(COCO_CLASSES)],
        metavar="CATEGORY",
        help="one or more COCO categories to highlight, e.g. -s bird car or -s bird -s car, * selects all",
    )
    highlight_parser.add_argument(
        "--print-confidence", action="store_true", help="print the confidence score next to each box"
    )
    highlight_parser.add_argument("--dim", action="store_true", help="darken everything outside the boxes")
    highlight_parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"folder for the highlighted images (default: {DEFAULT_OUT})",
    )
    highlight_parser.set_defaults(handler=highlight)

    evaluate_parser = commands.add_parser(
        "evaluate", parents=[model_options], help="compare predicted counts with COCO ground truth"
    )
    evaluate_parser.add_argument(
        "path", type=Path, nargs="?", default=DEFAULT_DATASET,
        help=f"folder with images/ and annotations.json (default: {DEFAULT_DATASET})",
    )
    evaluate_parser.add_argument(
        "--thresholds", type=fraction, nargs="+", default=DEFAULT_THRESHOLDS, metavar="VALUE",
        help=f"confidence thresholds to compare (default: {' '.join(map(str, DEFAULT_THRESHOLDS))})",
    )
    evaluate_parser.add_argument(
        "--out", type=Path, default=DEFAULT_EVALUATION_OUT,
        help=f"folder for tables and plots (default: {DEFAULT_EVALUATION_OUT})",
    )
    evaluate_parser.set_defaults(handler=evaluate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except FileNotFoundError as error:
        raise SystemExit(f"entify: {error}")

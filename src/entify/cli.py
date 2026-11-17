import argparse
from pathlib import Path

from tqdm import tqdm

from entify import __version__
from entify import evaluation
from entify.counting import count_by_category, totals
from entify.highlighting import render
from entify.images import find_images, load_image, save_image
from entify.models import DEFAULT_MODEL, ModelError, load_model
from entify.sharpening import sharpen
from entify.tiling import detect_tiled

DEFAULT_CSV = Path("outputs/results.csv")
DEFAULT_OUT = Path("outputs")
DEFAULT_DATASET = Path("data/coco_subset")
DEFAULT_EVALUATION_OUT = Path("outputs/evaluation")
DEFAULT_REVIEW_DIR = Path("reviews")
DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_MIN_AREA = None
ALL_SORTS = "*"


def fraction(text):
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{text} is not between 0 and 1")
    return value


def progress(items):
    # One bar over all images, hidden for a single image and when the output is not a terminal.
    return tqdm(items, desc="images", unit="img", disable=None if len(items) > 1 else True)


def open_model(args, confidence):
    print(f"Loading model {args.model}")
    try:
        model = load_model(args.model, args.labels, confidence, args.overlap, args.min_area, args.verbose)
    except (ModelError, ValueError) as error:
        raise SystemExit(f"entify: {error}")
    if args.tile and model.kind != "boxes":
        # Segmentation models lose the surrounding context in small tiles and then mislabel whole regions.
        raise SystemExit("entify: --tile only works with box models, not with segmentation models")
    return model


def check_categories(model, names):
    valid = model.selectable_categories
    unknown = [name for name in names if name not in valid]
    if unknown:
        raise SystemExit(
            f"entify: argument -s/--sort: invalid choice: {', '.join(map(repr, unknown))} for model {model.name} "
            f"(choose from {ALL_SORTS}, {', '.join(sorted(valid))})"
        )


def image_paths(args):
    paths = find_images(args.path)
    if not paths:
        raise SystemExit(f"entify: no images found in {args.path}")
    return paths


def find_objects(model, image, args):
    if args.sharpen:
        image = sharpen(image, args.sharpen)
    return detect_tiled(model, image, args.overlap) if args.tile else model.detect(image)


def counts_by_threshold(detections, thresholds):
    # The model runs once at the lowest threshold; higher thresholds only filter the stored scores.
    return ", ".join(f"{int((detections.scores >= threshold).sum())} at {threshold:.2f}" for threshold in thresholds)


def predict_dataset(model, dataset, files, args, thresholds):
    predictions = {}
    for n, name in enumerate(progress(files), 1):
        predictions[name] = find_objects(model, load_image(dataset / "images" / name), args)
        tqdm.write(f"[{n}/{len(files)}] {name}: {counts_by_threshold(predictions[name], thresholds)}")
    return predictions


def process_images(args, model, paths, handle_image):
    results = {}
    for n, path in enumerate(progress(paths), 1):
        image = load_image(path)
        results[path.name] = find_objects(model, image, args)
        tqdm.write(f"[{n}/{len(paths)}] {path.name}: {handle_image(path, image, results[path.name])}")
    return count_by_category(results, model.categories)


def save_csv(counts, path):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(path, index=False)
        print(f"\nCounts written to {path}")


def detect(args):
    paths = image_paths(args)
    model = open_model(args, args.confidence)
    counts = process_images(args, model, paths, lambda path, image, detections: f"{len(detections.labels)} objects")
    print()
    print(totals(counts).to_string() if not counts.empty else "No objects detected.")
    save_csv(counts, args.csv)


def highlight(args):
    paths = image_paths(args)
    model = open_model(args, args.confidence)
    all_sorts = ALL_SORTS in args.sort
    sorts = list(model.selectable_categories) if all_sorts else list(dict.fromkeys(args.sort))
    check_categories(model, sorts)
    model.select(None if all_sorts else sorts)
    suffix = "all" if all_sorts else "_".join(sort.replace(" ", "_") for sort in sorts)

    def save_highlight(path, image, detections):
        groups = [(sort, detections.only(model.categories.index(sort))) for sort in sorts]
        if all_sorts:
            groups = [(sort, selected) for sort, selected in groups if len(selected.labels)]
        target = args.out / f"{path.stem}_{suffix}.jpg"
        save_image(render(image, groups, args.print_confidence, args.dim), target)
        found = ", ".join(f"{len(selected.labels)} {sort}" for sort, selected in groups) or "nothing found"
        return f"{found} -> {target}"

    counts = process_images(args, model, paths, save_highlight)
    save_csv(counts, args.csv)


def evaluate(args):
    truth, files = evaluation.load_ground_truth(args.path / "annotations.json")
    thresholds = sorted(set(args.thresholds))
    model = open_model(args, thresholds[0])
    predictions = predict_dataset(model, args.path, files, args, thresholds)

    summary, by_category, true, predicted = evaluation.evaluate(truth, predictions, thresholds, model.categories)
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


def review(args):
    try:
        from entify.review.server import run_review
    except ImportError:
        raise SystemExit('entify: the review app needs extra packages: pip install -e ".[review]"')
    paths = image_paths(args)
    model = open_model(args, args.confidence)
    out = args.out or DEFAULT_REVIEW_DIR / (args.path.stem if args.path.is_file() else args.path.resolve().name)
    run_review(paths, model, lambda model, image: find_objects(model, image, args), out, args.port,
               not args.no_browser, args.verbose)


def add_model_options(parser):
    parser.add_argument(
        "-m", "--model", "--preset", dest="model", default=DEFAULT_MODEL, metavar="MODEL",
        help=f"KerasHub preset, model folder, .keras file or SavedModel (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--labels", type=Path, metavar="FILE",
        help="category names of the model, one per line or a JSON list (needed for models Entify does not know)",
    )
    parser.add_argument(
        "-o", "--overlap", type=fraction, default=0.5,
        help="maximum overlap (IoU) between two detections before the weaker one is removed, 1 keeps all (default: 0.5)",
    )
    parser.add_argument(
        "--min-area", type=int, default=DEFAULT_MIN_AREA, metavar="PIXELS",
        help="segmentation models: smallest region in pixels that counts as an object (default: 0.05%% of the image area)",
    )
    parser.add_argument(
        "-t", "--tile", action="store_true",
        help="detect in overlapping 640 px tiles to find small objects in large images",
    )
    parser.add_argument(
        "--sharpen", type=float, nargs="?", const=1.0, metavar="AMOUNT",
        help="sharpen images before detection with an unsharp mask (default amount: 1.0)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show TensorFlow and KerasHub log messages")


def build_parser():
    parser = argparse.ArgumentParser(prog="entify", description="Detect, count and highlight objects in images.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    model_options = argparse.ArgumentParser(add_help=False)
    add_model_options(model_options)

    input_options = argparse.ArgumentParser(add_help=False)
    input_options.add_argument("path", type=Path, help="image file or folder of images")
    input_options.add_argument(
        "-c", "--confidence", type=fraction, default=0.5,
        help="minimum confidence score of a detection (default: 0.5)",
    )

    csv_options = argparse.ArgumentParser(add_help=False)
    csv_options.add_argument(
        "--csv", type=Path, nargs="?", const=DEFAULT_CSV, metavar="FILE",
        help=f"write counts per image and category to a CSV file (default file: {DEFAULT_CSV})",
    )

    detect_parser = commands.add_parser(
        "detect", parents=[input_options, csv_options, model_options],
        help="detect and count objects in an image or a folder of images",
    )
    detect_parser.set_defaults(handler=detect)

    highlight_parser = commands.add_parser(
        "highlight", parents=[input_options, csv_options, model_options],
        help="mark all objects of the chosen categories",
    )
    highlight_parser.add_argument(
        "-s", "--sort", required=True, action="extend", nargs="+", metavar="CATEGORY",
        help="one or more categories of the model to highlight, e.g. -s bird car or -s bird -s car, * selects all",
    )
    highlight_parser.add_argument(
        "--print-confidence", action="store_true", help="print the confidence score next to each object"
    )
    highlight_parser.add_argument("--dim", action="store_true", help="darken everything outside the objects")
    highlight_parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"folder for the highlighted images (default: {DEFAULT_OUT})",
    )
    highlight_parser.set_defaults(handler=highlight)

    evaluate_parser = commands.add_parser(
        "evaluate", parents=[model_options], help="compare predicted counts with ground truth in COCO format"
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

    review_parser = commands.add_parser(
        "review", parents=[input_options, model_options],
        help="check and correct the detections in a browser, saved as COCO annotations",
    )
    review_parser.add_argument(
        "--out", type=Path, metavar="FOLDER",
        help=f"folder for annotations.json (default: {DEFAULT_REVIEW_DIR}/<name of PATH>)",
    )
    review_parser.add_argument("--port", type=int, default=8765, help="local port of the web app (default: 8765)")
    review_parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    review_parser.set_defaults(handler=review)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except FileNotFoundError as error:
        raise SystemExit(f"entify: {error}")

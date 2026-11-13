import argparse
from pathlib import Path

from entify import __version__
from entify.counting import count_by_category, totals
from entify.detector import COCO_CLASSES, DEFAULT_PRESET, Detector
from entify.highlighting import render
from entify.images import find_images, load_image, save_image

DEFAULT_CSV = Path("outputs/results.csv")
DEFAULT_OUT = Path("outputs")
ALL_SORTS = "*"


def fraction(text):
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{text} is not between 0 and 1")
    return value


def run(args, handle_image, classes=None):
    paths = find_images(args.path)
    if not paths:
        raise SystemExit(f"entify: no images found in {args.path}")

    print(f"Loading model {args.preset}")
    detector = Detector(args.preset, args.confidence, args.overlap, args.verbose, classes)
    results = {}
    for n, path in enumerate(paths, 1):
        image = load_image(path)
        results[path.name] = detector.detect(image)
        print(f"[{n}/{len(paths)}] {path.name}: {handle_image(path, image, results[path.name])}")
    return count_by_category(results, COCO_CLASSES)


def save_csv(counts, path):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(path, index=False)
        print(f"\nCounts written to {path}")


def detect(args):
    counts = run(args, lambda path, image, detections: f"{len(detections.labels)} objects")
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

    counts = run(args, save_highlight, classes)
    save_csv(counts, args.csv)


def build_parser():
    parser = argparse.ArgumentParser(prog="entify", description="Detect, count and highlight objects in images.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("path", type=Path, help="image file or folder of images")
    common.add_argument(
        "-c", "--confidence", type=fraction, default=0.5,
        help="minimum confidence score of a detection (default: 0.5)",
    )
    common.add_argument(
        "-o", "--overlap", type=fraction, default=0.5,
        help="maximum overlap (IoU) between two detections before the weaker one is removed, 1 keeps all (default: 0.5)",
    )
    common.add_argument(
        "-p", "--preset", default=DEFAULT_PRESET,
        help=f"KerasHub preset (default: {DEFAULT_PRESET})",
    )
    common.add_argument(
        "--csv", type=Path, nargs="?", const=DEFAULT_CSV, metavar="FILE",
        help=f"write counts per image and category to a CSV file (default file: {DEFAULT_CSV})",
    )
    common.add_argument("-v", "--verbose", action="store_true", help="show TensorFlow and KerasHub log messages")

    detect_parser = commands.add_parser(
        "detect", parents=[common], help="detect and count objects in an image or a folder of images"
    )
    detect_parser.set_defaults(handler=detect)

    highlight_parser = commands.add_parser(
        "highlight", parents=[common], help="highlight all objects of one category and dim the rest of the image"
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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except FileNotFoundError as error:
        raise SystemExit(f"entify: {error}")

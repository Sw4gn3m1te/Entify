import argparse
from pathlib import Path

from entify import __version__
from entify.counting import count_by_category, totals
from entify.detector import DEFAULT_PRESET, Detector
from entify.images import find_images, load_image


def fraction(text):
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{text} is not between 0 and 1")
    return value


def detect(args):
    paths = find_images(args.path)
    if not paths:
        raise SystemExit(f"No images found in {args.path}")

    print(f"Loading model {args.preset}")
    detector = Detector(args.preset, args.confidence, args.overlap, args.verbose)
    results = {}
    for n, path in enumerate(paths, 1):
        results[path.name] = detector.detect(load_image(path))
        print(f"[{n}/{len(paths)}] {path.name}: {len(results[path.name].labels)} objects")

    counts = count_by_category(results, detector.class_names)
    print()
    print(totals(counts).to_string() if not counts.empty else "No objects detected.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(args.csv, index=False)
        print(f"\nCounts written to {args.csv}")


def build_parser():
    parser = argparse.ArgumentParser(prog="entify", description="Detect, count and highlight objects in images.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="show TensorFlow and KerasHub log messages")

    detect_parser = commands.add_parser(
        "detect", parents=[common], help="detect and count objects in an image or a folder of images"
    )
    detect_parser.add_argument("path", type=Path, help="image file or folder of images")
    detect_parser.add_argument("-c", "--confidence", type=fraction, default=0.5,
                                help="minimum confidence score of a detection (default: 0.5)"
    )
    detect_parser.add_argument("-o", "--overlap", type=fraction, default=0.5,
                               help="maximum overlap (IoU) between two detections before the weaker one is removed, 1 keeps all (default: 0.5)",
    )
    detect_parser.add_argument("-p", "--preset", default=DEFAULT_PRESET, help=f"KerasHub preset (default: {DEFAULT_PRESET})")
    detect_parser.add_argument("--csv", default="outputs/results.csv", type=Path, help="write counts per image and category to this CSV file (default: outputs/results.csv)")
    detect_parser.set_defaults(handler=detect)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except FileNotFoundError as error:
        raise SystemExit(f"entify: {error}")

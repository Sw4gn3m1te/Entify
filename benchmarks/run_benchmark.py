import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from entify import __version__, evaluation
from entify.cli import DEFAULT_THRESHOLDS, fraction, predict_dataset
from entify.models import DEFAULT_MODEL, load_model

BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
RESULTS_FILE = BENCHMARK_DIR / "benchmark_results.md"
FETCH_SCRIPT = BENCHMARK_DIR.parent / "scripts" / "fetch_coco_subset.py"


def markdown_table(frame, formats):
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = [
        "| " + " | ".join(formats.get(column, "{}").format(value) for column, value in zip(frame.columns, row)) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, divider, *rows])


def fetch_subset(seed):
    subprocess.run([sys.executable, str(FETCH_SCRIPT), "--seed", str(seed), "--data-dir", str(DATA_DIR)], check=True)
    return DATA_DIR / "coco_subset"


def run(args, dataset):
    truth, files = evaluation.load_ground_truth(dataset / "annotations.json")
    thresholds = sorted(set(args.thresholds))
    model = load_model(args.model, args.labels, thresholds[0], args.overlap)
    predictions = predict_dataset(model, dataset, files, args, thresholds[0])
    return truth, evaluation.evaluate(truth, predictions, thresholds, model.categories)


def write_results(args, truth, results, runtime):
    summary, by_category, true, _ = results
    best = evaluation.best_threshold(summary)
    images = len(true.index)
    settings = [
        ("Date", date.today().isoformat()),
        ("Entify version", __version__),
        ("Model", f"`{args.model}`"),
        ("COCO subset seed", str(args.seed)),
        ("Images", f"{images} (COCO val2017, chosen by `scripts/fetch_coco_subset.py`)"),
        ("Ground-truth objects", str(len(truth))),
        ("Overlap (IoU)", str(args.overlap)),
        ("Tiling", "on" if args.tile else "off"),
        ("Sharpening", f"amount {args.sharpen}" if args.sharpen else "off"),
        ("Runtime", f"{runtime:.0f} s"),
    ]
    table = summary.rename(columns={"predicted objects": f"predicted (true: {len(truth)})"}).drop(columns="true objects")
    worst = evaluation.worst_categories(by_category, best).drop(columns="threshold")
    lines = [
        "# Entify benchmark",
        "",
        "Counting accuracy on a COCO val2017 subset, compared with the COCO ground truth.",
        "",
        "| Setting | Value |",
        "|---|---|",
        *(f"| {name} | {value} |" for name, value in settings),
        "",
        "## Counting error by confidence threshold",
        "",
        markdown_table(table, {
            "threshold": "{:.2f}",
            "mean absolute error per image": "{:.2f}",
            "mean error per image": "{:+.2f}",
            "pearson r": "{:.3f}",
        }),
        "",
        f"Lowest counting error at threshold **{best:.2f}**.",
        "",
        f"## Largest errors at threshold {best:.2f}",
        "",
        markdown_table(worst, {"mean absolute error": "{:.2f}", "mean error": "{:+.2f}"}),
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python benchmarks/run_benchmark.py {reproduce_arguments(args)}".rstrip(),
        "```",
        "",
    ]
    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")


def reproduce_arguments(args):
    parts = [f"--seed {args.seed}"]
    if args.model != DEFAULT_MODEL:
        parts.append(f"--model {args.model}")
    if args.overlap != 0.5:
        parts.append(f"--overlap {args.overlap}")
    if args.tile:
        parts.append("--tile")
    if args.sharpen:
        parts.append(f"--sharpen {args.sharpen}")
    if sorted(set(args.thresholds)) != DEFAULT_THRESHOLDS:
        parts.append("--thresholds " + " ".join(map(str, sorted(set(args.thresholds)))))
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Entify's counting accuracy on a COCO val2017 subset.")
    parser.add_argument("--seed", type=int, default=0, help="seed for choosing the COCO subset (default: 0)")
    parser.add_argument("--thresholds", type=fraction, nargs="+", default=DEFAULT_THRESHOLDS, metavar="VALUE")
    parser.add_argument("-m", "--model", "--preset", dest="model", default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, metavar="FILE")
    parser.add_argument("-o", "--overlap", type=fraction, default=0.5)
    parser.add_argument("-t", "--tile", action="store_true")
    parser.add_argument("--sharpen", type=float, nargs="?", const=1.0, metavar="AMOUNT")
    args = parser.parse_args()

    dataset = fetch_subset(args.seed)
    start = time.perf_counter()
    truth, results = run(args, dataset)
    write_results(args, truth, results, time.perf_counter() - start)
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()

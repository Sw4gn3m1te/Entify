# Entify

A Python 3 script for entity identification. Detect, count and highlight objects in images with pretrained models.
Give Entify a picture or a folder of pictures. It detects the objects it knows, counts them per category and highlights all objects of a category you choose.

## Tech stack
Python, TensorFlow / Keras (KerasHub), SciPy, Pandas, Matplotlib.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## Usage

```bash
entify detect PATH [options]                                # detect and count objects
entify highlight PATH -s CATEGORY [CATEGORY ...] [options]  # draw boxes around the chosen categories
entify evaluate [DATASET] [options]                         # compare counts with COCO ground truth
```

`python main.py ...` and `python -m entify ...` work the same way.

| Argument | Long form | Description |
|---|---|---|
| `PATH` | – | `detect`, `highlight`: path to image file or image folder |
| `DATASET` | – | `evaluate` only: folder with `images/` and `annotations.json` (default: `data/coco_subset`) |
| `-c` | `--confidence` | `detect`, `highlight`: minimum detection score, 0 to 1 (default: 0.5) <br> 0 = accept all detections, 1 = reject all detections  |
| `-o` | `--overlap` | Maximum overlap (IoU) of two detections, 0 to 1 (default: 0.5) <br> 0 = allow no overlap, 1 = allow identical boxes |
| `-p` | `--preset` | KerasHub D-FINE preset (default: `dfine_small_coco`) |
| `-t` | `--tile` | Also detect in overlapping 640 px tiles to find small objects in large images (slower) |
| – | `--sharpen` | Sharpen images before detection with an unsharp mask, optional amount (default amount: 1.0) |
| – | `--csv` | `detect`, `highlight`: write results to a CSV file, only when given (default file: `outputs/results.csv`) |
| `-s` | `--sort` | `highlight` only, required: one or more COCO categories to highlight, `*` selects all <br> `-s bird car` and `-s bird -s car` both work, quote names with spaces: `-s "cell phone"` |
| – | `--print-confidence` | `highlight` only: print the confidence score (2 decimals) next to each box |
| – | `--dim` | `highlight` only: darken everything outside the boxes |
| – | `--out` | `highlight`: folder for the highlighted images (default: `outputs`) <br> `evaluate`: folder for tables and plots (default: `outputs/evaluation`) |
| – | `--thresholds` | `evaluate` only: confidence thresholds to compare (default: `0.3 0.4 0.5 0.6 0.7 0.8`) |
| `-v` | `--verbose` | Show TensorFlow and KerasHub log messages |
| `-h` | `--help` | Show the help message |
| – | `--version` | Show the version |

If two detections overlap more than `--overlap`, the one with the lower score is removed. `--overlap 1` keeps all detections.

When several categories are selected, each object gets the selected category with the highest score.
In Git Bash, quote the asterisk (`-s '*'`), otherwise the shell replaces it with file names.
Highlighted images keep the original pixels. Images narrower than 2560 px are saved at a whole-number multiple of their size so that labels stay sharp.
Tiling helps with large scenes full of small objects (crowds, parking lots, herds). On close-ups it can add false detections and is about 7× slower.
`--csv` and `--sharpen` take an optional value, so put them after `PATH`.

Example:

```bash
entify detect photos/ --confidence 0.3 --csv outputs/counts.csv
entify detect photos/ -t --sharpen 1.5
entify highlight photos/ -s person --out outputs/people
entify highlight photos/ -s car truck --print-confidence
entify highlight photos/ -s '*' --dim
entify evaluate --thresholds 0.4 0.5 0.6
```

## Benchmark

```bash
python benchmarks/run_benchmark.py [--seed N] [-t] [--sharpen [AMOUNT]] [--thresholds ...]
```

Downloads a COCO val2017 subset chosen by the seed (default: 0) into `benchmarks/data`, compares Entify's counts with the COCO ground truth and writes the results to [`benchmarks/benchmark_results.md`](benchmarks/benchmark_results.md).
The first run downloads the COCO annotations (about 240 MB).

## License

MIT, see [LICENSE](LICENSE).

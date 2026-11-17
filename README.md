# Entify

A Python 3 script for entity identification. Detect, count and highlight objects in images with pretrained models.
Give Entify a picture or a folder of pictures. It detects the objects it knows, counts them per category and highlights all objects of a category you choose.

### Features
- **Detection and counting:** finds the 80 COCO categories (people, vehicles, animals, everyday objects) with a pretrained D-FINE detector and counts them per image and in total.
- **Batch mode:** processes a whole folder in one run and combines the counts in one table, optionally saved as CSV.
- **Highlighting:** draws boxes around the categories you choose, optionally with confidence scores and everything else dimmed. The original pixels stay untouched.
- **Tiling:** the model sees every image at only 640 × 640 px, so small objects in large photos shrink to a few pixels and get lost. Tiling also scans the full-resolution image in overlapping 640 px tiles and merges the results, which finds people in crowds, cars in parking lots or birds in a flock that a single pass misses.
- **Sharpening:** an optional unsharp mask before detection brings out edges in soft or slightly blurred photos.
- **Evaluation:** compares the counts with COCO ground truth across several confidence thresholds, so you can see how accurate the counts are and which threshold works best.
- **Benchmark and test data:** scripts that fetch a reproducible COCO subset and large Wikimedia Commons photos, with author and licence recorded for every image.

## Tech stack
Python, TensorFlow / Keras (KerasHub, pretrained D-FINE detector), NumPy, SciPy, Pandas, Matplotlib, Pillow.

## Setup

### Requirements
- Python 3.12 or newer, so that TensorFlow installs without issues.
- About 2 GB of disk space for the dependencies, plus about 45 MB for the default model.

### Installation

```bash
git clone https://github.com/Sw4gn3m1te/Entify.git
cd Entify
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Check the installation with `entify --version`.

### Model
The pretrained model is downloaded on first use and stored in `models/<preset>_v<version>/` inside the project, for example `models/dfine_small_coco_v1/`.
Each preset chosen with `--preset` gets its own folder. Delete a folder to download that model again.

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
| `-s` | `--sort` | `highlight` only, required: one or more COCO categories to highlight, `*` selects all (Git Bash: `'*'`) <br> `-s bird car` and `-s bird -s car` both work, quote names with spaces: `-s "cell phone"` |
| `-p` | `--preset` | KerasHub D-FINE preset (default: `dfine_small_coco`) |
| `-t` | `--tile` | Also detect in overlapping 640 px tiles to find small objects in large images (slower) |
| – | `--sharpen` | Sharpen images before detection with an unsharp mask, optional amount (default amount: 1.0) |
| – | `--print-confidence` | `highlight` only: print the confidence score (2 decimals) next to each box |
| – | `--dim` | `highlight` only: darken everything outside the boxes |
| – | `--thresholds` | `evaluate` only: confidence thresholds to compare (default: `0.3 0.4 0.5 0.6 0.7 0.8`) |
| – | `--csv` | `detect`, `highlight`: write results to a CSV file, only when given (default file: `outputs/results.csv`) |
| – | `--out` | `highlight`: folder for the highlighted images (default: `outputs`) <br> `evaluate`: folder for tables and plots (default: `outputs/evaluation`) |
| `-v` | `--verbose` | Show TensorFlow and KerasHub log messages |
| `-h` | `--help` | Show the help message |
| – | `--version` | Show the version |

---

Notes:
- Put `--csv` and `--sharpen` after `PATH`, because their value is optional.
- With several categories in `-s`, each object is labelled with the selected category it scores highest for.
- Tiling is about 7× slower and can add false detections on close-ups, so use it for large scenes only.
- Highlighted images narrower than 2560 px are upscaled by a whole-number factor so the labels stay sharp.

Example:

```bash
entify detect photos/ --confidence 0.3 --csv outputs/counts.csv
entify detect photos/ -t --sharpen 1.5
entify highlight photos/ -s person --out outputs/people
entify highlight photos/ -s car truck --print-confidence
entify highlight photos/ -s '*' --dim
entify evaluate --thresholds 0.4 0.5 0.6
```

## Test images

```bash
# COCO val2017 subset with ground truth for "entify evaluate", chosen by the seed (default: 0), saved to data/coco_subset
python scripts/fetch_coco_subset.py [--seed N]

# Fixed list of large Wikimedia Commons photos, saved to data/highres
python scripts/fetch_highres.py

# Random large photos from a Wikimedia Commons search, saved to data/highres_random
python scripts/fetch_highres.py random [-q QUERY] [-n COUNT] [--min-width PX] [--seed N]
```

The first COCO run downloads the annotations (about 240 MB). Each image folder contains a `SOURCES.md` with author, licence and source of every image.
Wikimedia sometimes rate-limits downloads. Skipped photos are fetched on the next run.

## Benchmark

```bash
python benchmarks/run_benchmark.py [--seed N] [-t] [--sharpen [AMOUNT]] [--thresholds ...]
```

Downloads a COCO val2017 subset chosen by the seed (default: 0) into `benchmarks/data`, compares Entify's counts with the COCO ground truth and writes the results to [`benchmarks/benchmark_results.md`](benchmarks/benchmark_results.md).
The first run downloads the COCO annotations (about 240 MB).

## License

MIT, see [LICENSE](LICENSE).

# Entify

A Python 3 script for entity identification. Detect, count and highlight objects in images with pretrained models.
Give Entify a picture or a folder of pictures. It detects the objects it knows, counts them per category and highlights all objects of a category you choose.

### Features
- **Detection and counting:** finds the 80 COCO categories (people, vehicles, animals, everyday objects) with a pretrained D-FINE detector and counts them per image and in total.
- **Your own models:** works with other Keras / TensorFlow models too, e.g. one trained on your own image domain. Object detectors return boxes, segmentation models return masks, and each connected region counts as one object. See [Using your own model](#using-your-own-model).
- **Batch mode:** processes a whole folder in one run and combines the counts in one table, optionally saved as CSV.
- **Highlighting:** draws boxes or filled masks around the categories you choose, optionally with confidence scores and everything else dimmed. The original pixels stay untouched.
- **Tiling:** the model sees every image at only 640 × 640 px, so small objects in large photos shrink to a few pixels and get lost. Tiling also scans the full-resolution image in overlapping 640 px tiles and merges the results, which finds people in crowds, cars in parking lots or birds in a flock that a single pass misses.
- **Sharpening:** an optional unsharp mask before detection brings out edges in soft or slightly blurred photos.
- **Evaluation:** compares the counts with COCO ground truth across several confidence thresholds, so you can see how accurate the counts are and which threshold works best.
- **Benchmark and test data:** scripts that fetch a reproducible COCO subset and large Wikimedia Commons photos, with author and licence recorded for every image.

## Tech stack
Python, TensorFlow / Keras (KerasHub: pretrained D-FINE detector, DeepLabV3 segmenter), NumPy, SciPy, Pandas, Matplotlib, Pillow.

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
Each preset chosen with `--model` gets its own folder. Delete a folder to download that model again.

## Usage

```bash
entify detect PATH [options]                                # detect and count objects
entify highlight PATH -s CATEGORY [CATEGORY ...] [options]  # mark the objects of the chosen categories
entify evaluate [DATASET] [options]                         # compare counts with ground truth in COCO format
```

`python main.py ...` and `python -m entify ...` work the same way.

| Argument | Long form | Description |
|---|---|---|
| `PATH` | – | `detect`, `highlight`: path to image file or image folder |
| `DATASET` | – | `evaluate` only: folder with `images/` and `annotations.json` (default: `data/coco_subset`) |
| `-c` | `--confidence` | `detect`, `highlight`: minimum detection score, 0 to 1 (default: 0.5) <br> 0 = accept all detections, 1 = reject all detections  |
| `-o` | `--overlap` | Maximum overlap (IoU) of two detections, 0 to 1 (default: 0.5) <br> 0 = allow no overlap, 1 = allow identical boxes |
| `-s` | `--sort` | `highlight` only, required: one or more categories of the model to highlight, `*` selects all (Git Bash: `'*'`) <br> `-s bird car` and `-s bird -s car` both work, quote names with spaces: `-s "cell phone"` |
| `-m` | `--model` | KerasHub preset, model folder, `.keras` file or SavedModel (default: `dfine_small_coco`). `--preset` still works |
| – | `--labels` | Category names for a model Entify does not know: text file with one name per line, or a JSON list |
| – | `--min-area` | Segmentation models: smallest region in pixels that counts as an object (default: 0.05 % of the image area) |
| `-t` | `--tile` | Also detect in overlapping 640 px tiles to find small objects in large images (slower, box models only) |
| – | `--sharpen` | Sharpen images before detection with an unsharp mask, optional amount (default amount: 1.0) |
| – | `--print-confidence` | `highlight` only: print the confidence score (2 decimals) next to each object |
| – | `--dim` | `highlight` only: darken everything outside the objects |
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
entify highlight photos/ -m deeplab_v3_plus_resnet50_pascalvoc -s person
entify evaluate --thresholds 0.4 0.5 0.6
```

## Using your own model

`--model` accepts four kinds of models:

| Model | Example | Supported output |
|---|---|---|
| KerasHub preset | `dfine_medium_coco`, `deeplab_v3_plus_resnet50_pascalvoc` | object detector (boxes) or image segmenter (masks) |
| Preset folder | `models/my_detector/`, saved with `model.save_to_preset(...)` | object detector or image segmenter |
| `.keras` file | `my_model.keras` | KerasHub task, or a plain model with per-pixel scores |
| SavedModel folder | `my_savedmodel/` | plain model with per-pixel scores |

Entify needs the category names in the order of the model's outputs. It knows them for the D-FINE COCO presets and DeepLabV3 (Pascal VOC). For any other model, pass them with `--labels FILE`, or put them in a settings file: `entify.json` inside a model folder, or `<name>.entify.json` next to a `.keras` file.

```json
{
  "labels": ["background", "lesion"],
  "input_size": [256, 256],
  "input_scale": 0.00392156862745098,
  "output": "logits"
}
```

| Key | Meaning | Default |
|---|---|---|
| `labels` | Category names in output order. A category named `background` is never counted | – |
| `input_size` | Height and width the model expects | taken from the model if it has a fixed input size |
| `input_scale` | Factor applied to the 0–255 pixel values, e.g. `1/255` | `1` |
| `output` | `probabilities` or `logits` (softmax, or sigmoid for a single output channel) | `probabilities` |
| `channels` | `1` for grayscale models | taken from the model |

Segmentation models label every pixel, so touching objects of the same category merge into one region and are counted once. Images are scaled to the model input with their aspect ratio kept and the rest padded. Tiling is not available for segmentation models, because they need the surrounding context.

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

# Entify

A Python 3 script for entity identification. Detect, count and highlight objects in images with pretrained models.
Give Entify a picture or a folder of pictures. It detects the objects it knows, counts them per category and highlights all objects of a category you choose.

### Features
- **Detection and counting:** finds the 80 COCO categories (people, vehicles, animals, everyday objects) with a pretrained D-FINE detector and counts them per image and in total.
- **Your own models:** works with other Keras / TensorFlow models too, e.g. one trained on your own image domain. Object detectors return boxes, segmentation models return masks, and each connected region counts as one object. See [Using your own model](#using-your-own-model).
- **Review in the browser:** a local web app shows the detections on each image. Reject wrong ones, draw missed ones and tighten rough boxes with one click (Segment Anything). The corrections are saved as COCO annotations, ready to evaluate the model against them.
- **Fine-tuning:** trains the detector further on the reviewed images and saves it as a new model version, with a before/after comparison of the counting accuracy.
- **Batch mode:** processes a whole folder in one run and combines the counts in one table, optionally saved as CSV.
- **Highlighting:** draws boxes or filled masks around the categories you choose, optionally with confidence scores and everything else dimmed. The original pixels stay untouched.
- **Tiling:** the model sees every image at only 640 × 640 px, so small objects in large photos shrink to a few pixels and get lost. Tiling also scans the full-resolution image in overlapping 640 px tiles and merges the results, which finds people in crowds, cars in parking lots or birds in a flock that a single pass misses.
- **Sharpening:** an optional unsharp mask before detection brings out edges in soft or slightly blurred photos.
- **Evaluation:** compares the counts with COCO ground truth across several confidence thresholds, so you can see how accurate the counts are and which threshold works best.
- **Benchmark and test data:** scripts that fetch a reproducible COCO subset and large Wikimedia Commons photos, with author and licence recorded for every image.

## Tech stack
Python, TensorFlow / Keras (KerasHub: pretrained D-FINE detector, DeepLabV3 segmenter, Segment Anything), NumPy, SciPy, Pandas, Matplotlib, Pillow. Review app: FastAPI and plain JavaScript.

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
pip install -e .                 # or pip install -e ".[review]" for the review web app
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
entify review PATH [options]                                # check and correct detections in the browser
entify finetune REVIEW_FOLDER [options]                     # train the detector further on the corrections
```

`python main.py ...` and `python -m entify ...` work the same way.

| Argument | Long form | Description |
|---|---|---|
| `PATH` | – | `detect`, `highlight`: path to image file or image folder |
| `DATASET` | – | `evaluate` only: folder with `images/` and `annotations.json` (default: `data/coco_subset`) |
| `-c` | `--confidence` | `detect`, `highlight`: minimum detection score, 0 to 1 (default: 0.5) <br> 0 = accept all detections, 1 = reject all detections  |
| `-o` | `--overlap` | Maximum overlap (IoU) of two detections, 0 to 1 (default: 0.5) <br> 0 = allow no overlap, 1 = allow identical boxes |
| `-s` | `--sort` | `highlight`, required: one or more categories of the model to highlight, `*` selects all (Git Bash: `'*'`) <br> `review`, optional: only check these categories (see [Reviewing detections](#reviewing-detections)) <br> `-s bird car` and `-s bird -s car` both work, quote names with spaces: `-s "cell phone"` |
| `-m` | `--model` | KerasHub preset, model folder, `.keras` file or SavedModel (default: `dfine_small_coco`). `--preset` still works |
| – | `--labels` | Category names for a model Entify does not know: text file with one name per line, or a JSON list |
| – | `--min-area` | Segmentation models: smallest region in pixels that counts as an object (default: 0.05 % of the image area) |
| `-t` | `--tile` | Also detect in overlapping 640 px tiles to find small objects in large images (slower, box models only) |
| – | `--sharpen` | Sharpen images before detection with an unsharp mask, optional amount (default amount: 1.0) |
| – | `--print-confidence` | `highlight` only: print the confidence score (2 decimals) next to each object |
| – | `--dim` | `highlight` only: darken everything outside the objects |
| – | `--thresholds` | `evaluate` only: confidence thresholds to compare (default: `0.3 0.4 0.5 0.6 0.7 0.8`) |
| – | `--csv` | `detect`, `highlight`: write results to a CSV file, only when given (default file: `outputs/results.csv`) |
| – | `--out` | `highlight`: folder for the highlighted images (default: `outputs`) <br> `evaluate`: folder for tables and plots (default: `outputs/evaluation`) <br> `review`: folder for `annotations.json` (default: `reviews/<name of PATH>`) |
| – | `--port` | `review` only: local port of the web app (default: 8123) |
| – | `--no-browser` | `review` only: do not open the browser automatically |
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

## Reviewing detections

```bash
pip install -e ".[review]"
entify review photos/ -c 0.4
```

Opens a local web page (http://127.0.0.1:8123) with one image at a time and the model's detections as boxes.

To check only some categories, name them: `entify review photos/ -s person`. Only those are shown at first, and the checkbox then means "all person objects checked". The other categories keep the model's detections, which fine-tuning uses as they are.

| Action | How |
|---|---|
| Reject a wrong detection or restore it | select the box, then `Delete` or the Reject button |
| Remove a box you drew | select it, then `Delete`: it is deleted outright (`Ctrl+Z` brings it back) |
| Add a missed object | drag a box on the image, also inside other boxes; `Shift` + drag always draws, even on the selected box |
| Choose the category of new boxes | `L` or click the category field, type to search, `↑`/`↓` and `Enter` to pick. Recently used and this image's categories come first |
| Change a box's category | select the box, then pick another category the same way |
| Fix a box | select it, then drag it or one of its handles |
| Tighten a rough box | select it, then `S` or Snap: Segment Anything finds the object inside the box |
| Show only some categories | `F` or the Show button: tick the categories to show. Hidden boxes stay saved, nothing is detected again |
| Mark the image as fully checked | `C` or the checkbox: every object of every category is marked (with `-s`: every object of those categories) |
| Undo, save, next/previous image | `Ctrl+Z`, `Ctrl+S`, `→` / `←` (switching images saves) |
| Zoom and pan | mouse wheel, right-drag or `Space` + drag |

The corrections go to `reviews/<name>/annotations.json` in COCO format, with absolute image paths, so the folder works directly as ground truth: `entify evaluate reviews/<name>`. Reopening the same folder continues where you left off.
Snapping downloads the Segment Anything model (`sam_base_sa1b`, about 360 MB) on first use. SAM has to encode each image once (about 10 s on a laptop CPU) before it can snap; the review app does this in the background as soon as an image is open, so snapping is usually ready by the time you need it, and every snap after that takes well under a second.
Segmentation models are reviewed as boxes.

## Fine-tuning on your corrections

```bash
entify finetune reviews/photos [--epochs 5] [--lr 1e-5] [--mix-coco 50]
```

Continues training the detector on the reviewed images and saves the result as a new model, e.g. `models/dfine_small_coco_ft1/`. The original model is never changed. Use the new one with `-m models/dfine_small_coco_ft1`.

- Only checked images are used: on any other image, an object nobody marked would teach the model that it is background. Images reviewed with `-s` (e.g. only for person) count as well: their other objects are the model's own detections as the review saved them, so the model is corrected for the checked categories and stays as it was for the rest.
- The pretrained backbone stays frozen and the learning rate is low, so a few corrections adjust the model instead of overwriting what it knows. `--unfreeze-all` trains the backbone too.
- `--mix-coco` mixes COCO images (from `scripts/fetch_coco_subset.py`) into the training, so the model keeps recognising the categories you did not correct. Reviewed images are left out of it, so no image is trained on both your labels and COCO's. This only applies to models with the COCO categories.
- Before and after training, Entify counts the objects on the reviewed images (part of them held out with `--val-split`, only the checked categories) and on further COCO images (`--check-coco`). The comparison is printed and saved as `finetune_report.md` in the new model folder, and `entify.json` there lists every image used, so benchmarks can leave them out.

Fine-tuning works for object detectors (D-FINE). On a laptop CPU, one training step takes several seconds per image, so 50 COCO plus 20 reviewed images over 5 epochs take about 45 minutes.

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

### Fine-tuned against original

```bash
python benchmarks/compare_models.py -m dfine_small_coco -m models/dfine_small_coco_ft1 [--category person] [--images 200]
```

Compares models on one category (default: person) with the original COCO annotations as ground truth, on 200 COCO val2017 images that none of the models was trained or checked on: both COCO subsets of this repository and every image a fine-tuned model lists in its `entify.json` are left out. 50 of the images have no person, to count false detections; the others have 1-2, 3-6 or 7 and more people in equal shares.
For each model it reports precision, recall and F1 of the boxes (IoU 0.5) and the counting error per confidence threshold, plus AP50, in [`benchmarks/model_comparison.md`](benchmarks/model_comparison.md). Without `-m`, it compares `dfine_small_coco` with every fine-tuned model in `models/`.

## License

MIT, see [LICENSE](LICENSE).

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
entify detect PATH [options]
```

`python main.py detect ...` and `python -m entify detect ...` work the same way.

| Argument | Long form | Description |
|---|---|---|
| `PATH` | – | Path to image file or image folder |
| `-c` | `--confidence` | Minimum detection score, 0 to 1 (default: 0.5) <br> 0 = accept all detections, 1 = reject all detections  |
| `-o` | `--overlap` | Maximum overlap (IoU) of two detections, 0 to 1 (default: 0.5) <br> 0 = allow no overlap, 1 = allow identical boxes |
| `-p` | `--preset` | KerasHub D-FINE preset (default: `dfine_small_coco`) |
| – | `--csv` | Write results to a CSV file (default: `outputs/results.csv`) |
| `-v` | `--verbose` | Show TensorFlow and KerasHub log messages |
| `-h` | `--help` | Show the help message |
| – | `--version` | Show the version |

If two detections overlap more than `--overlap`, the one with the lower score is removed. `--overlap 1` keeps all detections.

Example:

```bash
entify detect photos/ --confidence 0.3 --csv outputs/counts.csv
```

## License

MIT, see [LICENSE](LICENSE).

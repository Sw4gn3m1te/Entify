# Entify

Detect, count and highlight objects in images with pretrained models.

Give Entify a picture or a folder of pictures. It detects the objects it knows, counts them per category and highlights all objects of a category you choose.

> Status: early prototype, work in progress. The features below are planned and not implemented yet.

## Planned features

- Detection with a pretrained COCO detector (Keras / TensorFlow)
- Counts per category, exported as CSV
- Highlighting of one category with boxes or masks, rest of the image dimmed
- Batch mode over a folder of images
- Tiling for large images with many small objects
- Optional sharpening before detection
- Evaluation of predicted counts against COCO validation ground truth
- Optional: small web interface and an open-vocabulary backend for classes outside COCO

## Tech stack

Python, TensorFlow / Keras (KerasHub), SciPy, Pandas, Matplotlib.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## License

MIT, see [LICENSE](LICENSE).

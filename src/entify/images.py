from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(path):
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"No such file or directory: {path}")


def load_image(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def save_image(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=figure.dpi, pil_kwargs={"quality": 90})

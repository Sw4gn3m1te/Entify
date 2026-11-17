from dataclasses import dataclass

import numpy as np
from PIL import Image

from entify.models.labels import BACKGROUND


@dataclass
class Detections:
    """Objects found in one image.

    boxes are x1, y1, x2, y2 in image pixels. masks is None for box-only models. For segmentation models it holds
    one boolean array per object that covers exactly the object's box (boxes are whole pixels then).
    """

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    masks: np.ndarray | None = None

    def where(self, keep):
        masks = None if self.masks is None else self.masks[keep]
        return Detections(self.boxes[keep], self.scores[keep], self.labels[keep], masks)

    def only(self, label):
        return self.where(self.labels == label)

    def shifted(self, x, y):
        return Detections(self.boxes + [x, y, x, y], self.scores, self.labels, self.masks)

    @staticmethod
    def concatenate(parts):
        masks = None
        if all(part.masks is not None for part in parts):
            masks = np.empty(sum(len(part.masks) for part in parts), dtype=object)
            masks[:] = [mask for part in parts for mask in part.masks]
        return Detections(
            np.concatenate([part.boxes for part in parts]).reshape(-1, 4),
            np.concatenate([part.scores for part in parts]),
            np.concatenate([part.labels for part in parts]).astype(int),
            masks,
        )

    @staticmethod
    def empty(with_masks=False):
        masks = np.empty(0, dtype=object) if with_masks else None
        return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), masks)


class EntityModel:
    """Common interface of all models: categories, kind ("boxes" or "masks") and detect(image)."""

    kind = "boxes"

    def __init__(self, name, categories):
        self.name = name
        self.categories = tuple(categories)
        self.selected = None

    @property
    def selectable_categories(self):
        return tuple(category for category in self.categories if category.lower() != BACKGROUND)

    def select(self, categories):
        """Restrict detection to the given category names. None selects all."""
        self.selected = None if categories is None else [self.categories.index(name) for name in categories]

    def detect(self, image):
        raise NotImplementedError


def resize(image, size):
    # Resize to the model input without cropping, because KerasHub preprocessors would center-crop to a square.
    height, width = size
    return np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))

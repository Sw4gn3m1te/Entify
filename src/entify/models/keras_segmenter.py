import numpy as np
from PIL import Image
from scipy import ndimage

from entify.models.base import Detections, EntityModel, resize
from entify.models.labels import BACKGROUND

DEFAULT_MIN_AREA_SHARE = 0.0005


def softmax(values):
    exponents = np.exp(values - values.max(axis=-1, keepdims=True))
    return exponents / exponents.sum(axis=-1, keepdims=True)


def upscale(values, size, resample):
    height, width = size
    return np.asarray(Image.fromarray(values).resize((width, height), resample))


class KerasSegmenter(EntityModel):
    """A model that scores every pixel per category. Each connected region of one category becomes one object."""

    kind = "masks"

    def __init__(self, name, predict, categories, input_size, confidence=0.5, min_area=None,
                 channels=3, input_scale=1.0, logits=False):
        super().__init__(name, categories)
        self.predict = predict
        self.input_size = tuple(input_size)
        self.confidence = confidence
        self.min_area = min_area
        self.channels = channels
        self.input_scale = input_scale
        self.logits = logits

    def class_scores(self, image):
        # Keep the aspect ratio and pad the rest (letterbox): stretching distorts shapes and hurts segmentation.
        input_height, input_width = self.input_size
        scale = min(input_height / image.shape[0], input_width / image.shape[1])
        height, width = max(1, round(image.shape[0] * scale)), max(1, round(image.shape[1] * scale))
        x = np.zeros((input_height, input_width, 3), dtype=np.float32)
        x[:height, :width] = resize(image, (height, width))
        if self.channels == 1:
            x = x.mean(axis=-1, keepdims=True)
        scores = np.asarray(self.predict(x[None] * self.input_scale), dtype=np.float32)[0]
        scores = scores[:round(height * scores.shape[0] / input_height), :round(width * scores.shape[1] / input_width)]
        if scores.shape[-1] == 1:
            # A single output channel is the probability of the one category against everything else.
            foreground = 1 / (1 + np.exp(-scores)) if self.logits else scores
            return np.concatenate([1 - foreground, foreground], axis=-1), (BACKGROUND, self.categories[0])
        return (softmax(scores) if self.logits else scores), self.categories

    def detect(self, image):
        size = image.shape[:2]
        scores, categories = self.class_scores(image)
        if scores.shape[-1] != len(categories):
            raise ValueError(f"model returns {scores.shape[-1]} categories but {len(categories)} names are known")
        class_map = upscale(scores.argmax(axis=-1).astype(np.int32), size, Image.NEAREST)
        confidence_map = upscale(scores.max(axis=-1), size, Image.BILINEAR)

        wanted = [
            index for index, name in enumerate(categories)
            if name.lower() != BACKGROUND and (self.selected is None or self.categories.index(name) in self.selected)
        ]
        min_area = self.min_area if self.min_area is not None else DEFAULT_MIN_AREA_SHARE * size[0] * size[1]
        boxes, object_scores, labels, masks = [], [], [], []
        for index in wanted:
            regions, count = ndimage.label(class_map == index)
            for number, (rows, columns) in enumerate(ndimage.find_objects(regions), 1):
                mask = regions[rows, columns] == number
                score = float(confidence_map[rows, columns][mask].mean())
                if mask.sum() < min_area or score < self.confidence:
                    continue
                boxes.append((columns.start, rows.start, columns.stop, rows.stop))
                object_scores.append(score)
                labels.append(self.categories.index(categories[index]))
                masks.append(mask)
        if not boxes:
            return Detections.empty(with_masks=True)
        mask_array = np.empty(len(masks), dtype=object)
        mask_array[:] = masks
        return Detections(np.array(boxes, dtype=float), np.array(object_scores), np.array(labels), mask_array)

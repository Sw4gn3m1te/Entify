import numpy as np


def areas(boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection_areas(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    widths = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    heights = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    return widths * heights


def iou(box, boxes):
    intersection = intersection_areas(box, boxes)
    return intersection / np.maximum(areas(box)[0] + areas(boxes) - intersection, 1e-9)

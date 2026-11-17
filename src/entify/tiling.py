import numpy as np
from tqdm import tqdm

from entify.detector import Detections
from entify.geometry import areas, intersection_areas, iou

TILE_SIZE = 640
TILE_OVERLAP = 0.2
BORDER_MARGIN = 2
CONTAINED = 0.8
MIN_FRAGMENT_SHARE = 0.03


def tile_origins(length, size, step):
    if length <= size:
        return [0]
    return list(range(0, length - size, step)) + [length - size]


def tiles(image, size=TILE_SIZE, overlap=TILE_OVERLAP):
    height, width = image.shape[:2]
    step = int(size * (1 - overlap))
    for y in tile_origins(height, size, step):
        for x in tile_origins(width, size, step):
            yield x, y, image[y:y + size, x:x + size]


def complete_boxes(detections, x, y, tile_shape, image_shape):
    tile_height, tile_width = tile_shape[:2]
    height, width = image_shape[:2]
    x1, y1, x2, y2 = detections.boxes.T
    keep = (
        ((x == 0) | (x1 > BORDER_MARGIN))
        & ((y == 0) | (y1 > BORDER_MARGIN))
        & ((x + tile_width >= width) | (x2 < tile_width - BORDER_MARGIN))
        & ((y + tile_height >= height) | (y2 < tile_height - BORDER_MARGIN))
    )
    return detections.where(keep)


def without_duplicates(detections, max_overlap):
    order = np.argsort(-detections.scores)
    keep = []
    while order.size:
        best, order = order[0], order[1:]
        keep.append(best)
        order = order[iou(detections.boxes[best], detections.boxes[order]) <= max_overlap]
    return np.array(keep, dtype=int)


def fragments(detections, from_tiles):
    boxes, sizes = detections.boxes, areas(detections.boxes)
    found = np.zeros(len(boxes), dtype=bool)
    for i in np.flatnonzero(from_tiles):
        larger = (detections.labels == detections.labels[i]) & (sizes > sizes[i])
        contained = intersection_areas(boxes[i], boxes[larger]) / max(sizes[i], 1e-9) >= CONTAINED
        found[i] = np.any(contained & (sizes[i] / sizes[larger] >= MIN_FRAGMENT_SHARE))
    return found


def detect_tiled(detector, image, max_overlap):
    if max(image.shape[:2]) <= TILE_SIZE:
        return detector.detect(image)
    parts = [detector.detect(image)]
    grid = list(tiles(image))
    for x, y, tile in tqdm(grid, desc="tiles", unit="tile", leave=False, disable=None):
        found = complete_boxes(detector.detect(tile), x, y, tile.shape, image.shape)
        parts.append(found.shifted(x, y))
    combined = Detections.concatenate(parts)
    from_tiles = np.arange(len(combined.scores)) >= len(parts[0].scores)
    keep = without_duplicates(combined, max_overlap)
    keep = keep[~fragments(combined.where(keep), from_tiles[keep])]
    return combined.where(keep)

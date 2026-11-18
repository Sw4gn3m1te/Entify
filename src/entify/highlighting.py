import math

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy import ndimage

from entify.geometry import intersection_areas

DIM_FACTOR = 0.35
MASK_ALPHA = 0.4
COLORS = (
    "#ffd400", "#00e5ff", "#ff4fd8", "#7cfc00", "#ff9f1a", "#b388ff",
    "#ff6b6b", "#ffffff", "#c6ff00", "#80cbc4", "#ffab91", "#f48fb1",
)
PAD = 0.15
DPI = 1
MIN_OUTPUT_WIDTH = 2560


def mark_object(target, box, mask=None):
    # Set the pixels of one object in a full-size boolean array: its segmentation mask, or its box if it has none.
    height, width = target.shape
    x1, y1, x2, y2 = np.clip(np.round(box).astype(int), 0, [width, height, width, height])
    if mask is None:
        target[y1:y2, x1:x2] = True
    else:
        target[y1:y1 + mask.shape[0], x1:x1 + mask.shape[1]] |= mask[:height - y1, :width - x1]


def object_masks(detections, shape):
    for box, mask in zip(detections.boxes, detections.masks):
        full = np.zeros(shape[:2], dtype=bool)
        mark_object(full, box, mask)
        yield full


def dim(image, groups, factor=DIM_FACTOR):
    inside = np.zeros(image.shape[:2], dtype=bool)
    for _, detections in groups:
        masks = detections.masks if detections.masks is not None else [None] * len(detections.boxes)
        for box, mask in zip(detections.boxes, masks):
            mark_object(inside, box, mask)
    return np.where(inside[..., None], image, (image * factor).astype(np.uint8))


def paint_masks(image, groups, colors, thickness):
    # Fill each segmented object with its category colour and draw its outline.
    painted = image.astype(np.float32)
    for category, detections in groups:
        if detections.masks is None:
            continue
        color = np.array(to_rgb(colors[category])) * 255
        for mask in object_masks(detections, image.shape):
            outline = mask & ~ndimage.binary_erosion(mask, iterations=thickness)
            painted[mask] = (1 - MASK_ALPHA) * painted[mask] + MASK_ALPHA * color
            painted[outline] = color
    return painted.astype(np.uint8)


def overlap_area(rect, rects):
    return float(intersection_areas(rect, rects).sum())


def candidate_positions(box, width, height, snug):
    x1, y1, x2, y2 = box
    adjacent = [
        (x1, y1 - height + snug), (x1, y2 - snug), (x2 - width, y1 - height + snug), (x2 - width, y2 - snug),
        (x1 - width + snug, y1), (x2 - snug, y1), (x1, y1), (x1, y2 - height),
    ]
    displaced = [
        position
        for step in (1, 2, 3)
        for position in (
            (x1, y1 - height * (step + 1)), (x1, y2 + height * step),
            (x1 - width * step, y1 - height), (x2 + width * (step - 1), y1 - height),
            (x1 - width * step, y2), (x2 + width * (step - 1), y2),
        )
    ]
    return adjacent + displaced


def gap_line(rect, box):
    def facing(a1, a2, b1, b2):
        if a2 < b1:
            return a2, b1
        if b2 < a1:
            return a1, b2
        middle = (max(a1, b1) + min(a2, b2)) / 2
        return middle, middle

    return facing(rect[0], rect[2], box[0], box[2]), facing(rect[1], rect[3], box[1], box[3])


def place_label(box, size, image_size, placed_labels, other_boxes, snug):
    width, height = size
    image_width, image_height = image_size
    best, best_cost = None, None
    for x, y in candidate_positions(box, width, height, snug):
        x = min(max(x, 0), image_width - width)
        y = min(max(y, 0), image_height - height)
        rect = (x, y, x + width, y + height)
        cost = 10 * overlap_area(rect, placed_labels) + overlap_area(rect, other_boxes)
        if best_cost is None or cost < best_cost:
            best, best_cost = rect, cost
        if cost == 0:
            break
    return best


class Canvas:
    def __init__(self, image):
        self.height, self.width = image.shape[:2]
        self.scale = max(1, math.ceil(MIN_OUTPUT_WIDTH / self.width))
        self.figure = Figure(figsize=(self.width / DPI, self.height / DPI), dpi=DPI * self.scale)
        self.renderer = FigureCanvasAgg(self.figure).get_renderer()
        self.figure.figimage(image.repeat(self.scale, axis=0).repeat(self.scale, axis=1), origin="upper", zorder=-1)
        self.axes = self.figure.add_axes((0, 0, 1, 1))
        self.axes.set_axis_off()
        self.axes.patch.set_visible(False)
        self.axes.set_xlim(0, self.width)
        self.axes.set_ylim(self.height, 0)
        self.axes.set_autoscale_on(False)

    def label(self, text, color, fontsize):
        artist = self.axes.text(
            0, 0, text, ha="left", va="top", fontsize=fontsize,
            bbox={"boxstyle": f"square,pad={PAD}", "facecolor": color, "edgecolor": "none"},
        )
        extent = artist.get_window_extent(self.renderer)
        pad = PAD * fontsize * DPI / 72
        return artist, (extent.width / self.scale + 2 * pad, extent.height / self.scale + 2 * pad), pad


def points(pixels):
    return pixels * 72 / DPI


def label_fontsize(boxes, image_width):
    median_height = np.median(boxes[:, 3] - boxes[:, 1]) if len(boxes) else image_width
    minimum = min(max(image_width / 60, 8), 12)
    pixels = min(max(0.4 * median_height, minimum), max(minimum, image_width / 60))
    return points(max(pixels / 2, 6))


def draw_boxes(canvas, groups, colors, line_pixels):
    # Segmented objects are already painted, so only box-only detections get a rectangle.
    for category, detections in groups:
        if detections.masks is None:
            for x1, y1, x2, y2 in detections.boxes:
                canvas.axes.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colors[category],
                                                linewidth=points(line_pixels)))


def draw_legend(canvas, groups, colors):
    """The count of each category in the top left corner. Returns the rectangles it covers."""
    x, y = canvas.width * 0.01, canvas.height * 0.01
    fontsize = points(max(8.0, canvas.width / 115))  # 1/115 of the image width, at least 8 px
    occupied = []
    for category, detections in groups:
        artist, (width, height), pad = canvas.label(f"{category}: {len(detections.labels)}", colors[category], fontsize)
        artist.set_position((x + pad, y + pad))
        occupied.append((x, y, x + width, y + height))
        y += height
    return occupied


def draw_labels(canvas, groups, colors, occupied, show_category, show_confidence, line_pixels):
    # A label next to each object, placed where it covers the fewest other labels and boxes, linked by a short line.
    objects = [(category, box, score) for category, detections in groups
               for box, score in zip(detections.boxes, detections.scores)]
    boxes = np.array([box for _, box, _ in objects], dtype=float).reshape(-1, 4)
    fontsize = label_fontsize(boxes, canvas.width)
    for index, (category, box, score) in enumerate(objects):
        text = " ".join(([category] if show_category else []) + ([f"{score:.2f}"] if show_confidence else []))
        artist, size, pad = canvas.label(text, colors[category], fontsize)
        others = np.delete(boxes, index, axis=0)
        rect = place_label(box, size, (canvas.width, canvas.height), occupied, others, line_pixels / 2)
        artist.set_position((rect[0] + pad, rect[1] + pad))
        occupied.append(rect)
        xs, ys = gap_line(rect, box)
        if np.hypot(xs[1] - xs[0], ys[1] - ys[0]) > line_pixels:
            canvas.axes.plot(xs, ys, color=colors[category], linewidth=points(line_pixels / 2))


def render(image, groups, show_confidence=False, dim_rest=False):
    colors = dict(zip((category for category, _ in groups), COLORS * (len(groups) // len(COLORS) + 1)))
    line_pixels = max(1.4, image.shape[1] / 576)  # 1/576 of the image width, at least 1.4 px
    if dim_rest:
        image = dim(image, groups)
    canvas = Canvas(paint_masks(image, groups, colors, max(1, round(line_pixels))))
    draw_boxes(canvas, groups, colors, line_pixels)
    occupied = draw_legend(canvas, groups, colors)
    show_category = len(groups) > 1
    if show_category or show_confidence:
        draw_labels(canvas, groups, colors, occupied, show_category, show_confidence, line_pixels)
    return canvas.figure

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

DIM_FACTOR = 0.35
COLORS = ("#ffd400", "#00e5ff", "#ff4fd8", "#7cfc00", "#ff9f1a", "#b388ff", "#ff6b6b", "#ffffff")
PAD = 0.25
DPI = 100


def dim(image, boxes, factor=DIM_FACTOR):
    height, width = image.shape[:2]
    inside = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in np.clip(boxes.round().astype(int), 0, [width, height, width, height]):
        inside[y1:y2, x1:x2] = True
    return np.where(inside[..., None], image, (image * factor).astype(np.uint8))


def overlap_area(rect, rects):
    if len(rects) == 0:
        return 0.0
    rects = np.asarray(rects)
    widths = np.minimum(rect[2], rects[:, 2]) - np.maximum(rect[0], rects[:, 0])
    heights = np.minimum(rect[3], rects[:, 3]) - np.maximum(rect[1], rects[:, 1])
    return float((np.clip(widths, 0, None) * np.clip(heights, 0, None)).sum())


def candidate_positions(box, width, height):
    x1, y1, x2, y2 = box
    adjacent = [
        (x1, y1 - height), (x1, y2), (x2 - width, y1 - height), (x2 - width, y2),
        (x1 - width, y1), (x2, y1), (x1, y1), (x1, y2 - height),
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

    (label_x, box_x), (label_y, box_y) = facing(rect[0], rect[2], box[0], box[2]), facing(rect[1], rect[3], box[1], box[3])
    return (label_x, box_x), (label_y, box_y)


def place_label(box, size, image_size, placed_labels, other_boxes):
    width, height = size
    image_width, image_height = image_size
    best, best_cost = None, None
    for x, y in candidate_positions(box, width, height):
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
        self.figure = Figure(figsize=(self.width / DPI, self.height / DPI), dpi=DPI)
        self.renderer = FigureCanvasAgg(self.figure).get_renderer()
        self.axes = self.figure.add_axes((0, 0, 1, 1))
        self.axes.set_axis_off()

    def label(self, text, color, fontsize):
        artist = self.axes.text(
            0, 0, text, ha="left", va="top", fontsize=fontsize,
            bbox={"boxstyle": f"square,pad={PAD}", "facecolor": color, "edgecolor": "none"},
        )
        extent = artist.get_window_extent(self.renderer)
        pad = PAD * fontsize * DPI / 72
        return artist, (extent.width + 2 * pad, extent.height + 2 * pad), pad


def label_fontsize(boxes, image_width):
    median_height = np.median(boxes[:, 3] - boxes[:, 1]) if len(boxes) else image_width
    minimum = min(max(image_width / 60, 8), 12)
    pixels = min(max(0.4 * median_height, minimum), max(minimum, image_width / 60))
    return pixels * 72 / DPI


def label_text(category, score, show_category, show_confidence):
    parts = ([category] if show_category else []) + ([f"{score:.2f}"] if show_confidence else [])
    return " ".join(parts)


def render(image, groups, show_confidence=False):
    canvas = Canvas(image)
    colors = dict(zip((category for category, _ in groups), COLORS * (len(groups) // len(COLORS) + 1)))
    boxes = np.concatenate([detections.boxes for _, detections in groups]).reshape(-1, 4)
    canvas.axes.imshow(dim(image, boxes))

    line_width = max(1.0, canvas.width / 800)
    for category, detections in groups:
        for x1, y1, x2, y2 in detections.boxes:
            canvas.axes.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colors[category], linewidth=line_width)
            )

    occupied = []
    legend_y = canvas.height * 0.01
    for category, detections in groups:
        artist, (width, height), pad = canvas.label(f"{category}: {len(detections.labels)}", colors[category],
                                                    max(8.0, canvas.width / 80))
        x = canvas.width * 0.01
        artist.set_position((x + pad, legend_y + pad))
        occupied.append((x, legend_y, x + width, legend_y + height))
        legend_y += height

    show_category = len(groups) > 1
    if not (show_category or show_confidence):
        return canvas.figure

    fontsize = label_fontsize(boxes, canvas.width)
    index = 0
    for category, detections in groups:
        for box, score in zip(detections.boxes, detections.scores):
            others = np.delete(boxes, index, axis=0)
            index += 1
            text = label_text(category, score, show_category, show_confidence)
            artist, size, pad = canvas.label(text, colors[category], fontsize)
            rect = place_label(box, size, (canvas.width, canvas.height), occupied, others)
            artist.set_position((rect[0] + pad, rect[1] + pad))
            occupied.append(rect)
            xs, ys = gap_line(rect, box)
            if np.hypot(xs[1] - xs[0], ys[1] - ys[0]) > line_width:
                canvas.axes.plot(xs, ys, color=colors[category], linewidth=line_width / 2)
    return canvas.figure

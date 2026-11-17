import json
import re
from pathlib import Path

BACKGROUND = "background"

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

PASCAL_VOC_CLASSES = (
    BACKGROUND, "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv monitor",
)

# Category names of KerasHub presets, matched by preset name. Presets not listed here need a label file.
BUILTIN_LABELS = (
    (r"dfine_.*_(coco|obj2coco.*?)(_v\d+)?", COCO_CLASSES),
    (r"deeplab_v3_.*_pascalvoc(_v\d+)?", PASCAL_VOC_CLASSES),
)


def builtin_labels(preset):
    for pattern, labels in BUILTIN_LABELS:
        if re.fullmatch(pattern, Path(str(preset)).name.removesuffix("/")):
            return labels
    return None


def read_labels(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        labels = json.loads(text)
        labels = labels.get("labels", labels) if isinstance(labels, dict) else labels
    else:
        labels = [line.strip() for line in text.splitlines() if line.strip()]
    if not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"{path} must contain a list of category names")
    return tuple(labels)

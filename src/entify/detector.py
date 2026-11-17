import contextlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from entify.quiet import silence_library_logs, silenced_stderr

DEFAULT_PRESET = "dfine_small_coco"

def model_dir():
    # Keep the weights in the project's models/ folder when running from the source tree, otherwise in the user's cache.
    project_root = Path(__file__).resolve().parents[2]
    if (project_root / "pyproject.toml").is_file():
        return project_root / "models"
    return Path.home() / ".cache" / "entify" / "models"


MODEL_DIR = model_dir()
DOWNLOADED = ".downloaded"


def local_preset(preset):
    # Store built-in Kaggle presets as models/<preset>_v<version>/ instead of kagglehub's deeply nested cache.
    # Anything else, such as a local folder or a preset KerasHub does not list, is passed on unchanged.
    try:
        from keras_hub.src.utils.preset_utils import BUILTIN_PRESETS

        handle = BUILTIN_PRESETS[preset]["kaggle_handle"]
    except (ImportError, KeyError, TypeError):
        return preset
    match = re.fullmatch(r"kaggle://(.+)/(\d+)", handle)
    if not match:
        return preset
    target = MODEL_DIR / f"{preset}_v{match[2]}"
    if not (target / DOWNLOADED).exists():
        import kagglehub

        print(f"Downloading model {preset} to {target}")
        kagglehub.model_download(handle.removeprefix("kaggle://"), output_dir=str(target))
        (target / DOWNLOADED).touch()
    return str(target)

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


@dataclass
class Detections:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    def where(self, keep):
        return Detections(self.boxes[keep], self.scores[keep], self.labels[keep])

    def only(self, label):
        return self.where(self.labels == label)

    def shifted(self, x, y):
        return Detections(self.boxes + [x, y, x, y], self.scores, self.labels)

    @staticmethod
    def concatenate(parts):
        return Detections(
            np.concatenate([part.boxes for part in parts]).reshape(-1, 4),
            np.concatenate([part.scores for part in parts]),
            np.concatenate([part.labels for part in parts]),
        )


class Detector:
    def __init__(self, preset=DEFAULT_PRESET, confidence=0.5, overlap=0.5, verbose=False, classes=None):
        if not verbose:
            silence_library_logs()
        os.environ.setdefault("KAGGLEHUB_CACHE", str(MODEL_DIR))
        with contextlib.nullcontext() if verbose else silenced_stderr():
            import keras_hub

            from entify.decoder import SelectedClassesNMS

            self.model = keras_hub.models.ObjectDetector.from_preset(local_preset(preset))
            config = self.model.prediction_decoder.get_config()
            config |= {"confidence_threshold": confidence, "iou_threshold": overlap}
            self.model.prediction_decoder = SelectedClassesNMS(classes=classes, **config)
        self.input_size = tuple(self.model.preprocessor.image_converter.image_size)

    def detect(self, image):
        height, width = image.shape[:2]
        input_height, input_width = self.input_size
        # Resize here because the KerasHub preprocessor would center-crop to a square and cut off the image edges.
        resized = np.asarray(Image.fromarray(image).resize((input_width, input_height), Image.BILINEAR))
        output = self.model.predict(resized[None], verbose=0)

        count = int(output["num_detections"][0])
        y1, x1, y2, x2 = np.asarray(output["boxes"][0][:count]).T
        scale_x, scale_y = width / input_width, height / input_height
        boxes = np.stack([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], axis=1)
        return Detections(boxes, np.asarray(output["confidence"][0][:count]), np.asarray(output["labels"][0][:count]))

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from entify.quiet import silence_library_logs, silenced_stderr

DEFAULT_PRESET = "dfine_small_coco"
MODEL_DIR = Path("models")

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

    def only(self, label):
        keep = self.labels == label
        return Detections(self.boxes[keep], self.scores[keep], self.labels[keep])


class Detector:
    class_names = COCO_CLASSES

    def __init__(self, preset=DEFAULT_PRESET, confidence=0.5, overlap=0.5, verbose=False, classes=None):
        if not verbose:
            silence_library_logs()
        os.environ.setdefault("KAGGLEHUB_CACHE", str(MODEL_DIR.resolve()))
        with contextlib.nullcontext() if verbose else silenced_stderr():
            import keras_hub

            from entify.decoder import SelectedClassesNMS

            self.model = keras_hub.models.ObjectDetector.from_preset(preset)
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

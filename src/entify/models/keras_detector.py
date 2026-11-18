import numpy as np

from entify.models.base import Detections, EntityModel, resize
from entify.models.nms import SelectedClassesNMS


class KerasDetector(EntityModel):
    """A KerasHub ObjectDetector (for example D-FINE) that returns boxes."""

    def __init__(self, name, task, categories, confidence=0.5, overlap=0.5):
        super().__init__(name, categories)
        self.task = task
        self.confidence = confidence
        self.overlap = overlap
        self.input_size = tuple(task.preprocessor.image_converter.image_size)
        self.original_decoder = task.prediction_decoder
        self.configure_decoder()

    def select(self, categories):
        super().select(categories)
        self.configure_decoder()

    def configure_decoder(self):
        config = self.original_decoder.get_config()
        config |= {"confidence_threshold": self.confidence, "iou_threshold": self.overlap}
        self.task.prediction_decoder = SelectedClassesNMS(classes=self.selected, **config)
        self.task.predict_function = None

    def restore_decoder(self):
        # Saved presets must contain KerasHub's own decoder, which loads without Entify.
        self.task.prediction_decoder = self.original_decoder
        self.task.predict_function = None

    def detect(self, image):
        height, width = image.shape[:2]
        input_height, input_width = self.input_size
        output = self.task.predict(resize(image, self.input_size)[None], verbose=0)

        count = int(output["num_detections"][0])
        y1, x1, y2, x2 = np.asarray(output["boxes"][0][:count]).T
        scale_x, scale_y = width / input_width, height / input_height
        boxes = np.stack([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], axis=1)
        labels = np.asarray(output["labels"][0][:count]).astype(int)
        return Detections(boxes, np.asarray(output["confidence"][0][:count]), labels)

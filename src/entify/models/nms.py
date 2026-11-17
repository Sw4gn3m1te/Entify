import numpy as np
from keras import ops
from keras_hub.layers import NonMaxSuppression


class SelectedClassesNMS(NonMaxSuppression):
    def __init__(self, classes=None, **kwargs):
        super().__init__(**kwargs)
        self.classes = classes

    def call(self, box_prediction, class_prediction, images=None):
        if self.classes is not None:
            selected = np.isin(np.arange(class_prediction.shape[-1]), self.classes)
            excluded = -1e9 if self.from_logits else 0.0
            class_prediction = ops.where(selected, class_prediction, excluded)
        return super().call(box_prediction, class_prediction, images=images)

import contextlib
from collections import OrderedDict

import numpy as np
from PIL import Image

from entify.quiet import silenced_stderr

SAM_PRESET = "sam_base_sa1b"
SAM_SIZE = 1024
CACHED_IMAGES = 4


class BoxSnapper:
    """Tightens a rough box around the object inside it with Segment Anything (SAM, KerasHub).

    SAM's image encoder takes several seconds on a CPU, so its result is cached per image and only the cheap prompt
    encoder and mask decoder run for each box.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.backbone = None
        self.embeddings = OrderedDict()

    def load(self):
        if self.backbone is None:
            with contextlib.nullcontext() if self.verbose else silenced_stderr():
                import keras_hub

                from entify.models.loading import local_preset

                self.backbone = keras_hub.models.SAMImageSegmenter.from_preset(local_preset(SAM_PRESET)).backbone

    def embedding(self, key, image):
        if key not in self.embeddings:
            # SAM expects the longest side at 1024 px, padded to a square, with values from 0 to 1.
            scale = SAM_SIZE / max(image.shape[:2])
            height, width = round(image.shape[0] * scale), round(image.shape[1] * scale)
            padded = np.zeros((SAM_SIZE, SAM_SIZE, 3), dtype=np.float32)
            padded[:height, :width] = np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))
            self.embeddings[key] = self.backbone.image_encoder.predict(padded[None] / 255, verbose=0), scale
            if len(self.embeddings) > CACHED_IMAGES:
                self.embeddings.popitem(last=False)
        self.embeddings.move_to_end(key)
        return self.embeddings[key]

    def snap(self, key, image, box):
        import keras

        self.load()
        embedding, scale = self.embedding(key, image)
        x1, y1, x2, y2 = np.asarray(box, dtype=np.float32) * scale
        prompt = self.backbone.prompt_encoder(
            points=keras.ops.zeros((1, 1, 2)),
            labels=-keras.ops.ones((1, 1)),
            boxes=keras.ops.convert_to_tensor(np.array([[[[x1, y1], [x2, y2]]]], dtype=np.float32)),
            masks=keras.ops.zeros((1, 0, 256, 256, 1)),
        )
        output = self.backbone.mask_decoder(image_embeddings=keras.ops.convert_to_tensor(embedding), **prompt)
        logits = np.asarray(output["masks"])[0, 0].astype(np.float32)
        logits = np.asarray(Image.fromarray(logits).resize((SAM_SIZE, SAM_SIZE), Image.BILINEAR))
        rows, columns = np.nonzero(logits > 0)
        if not len(rows):
            return list(map(float, box))
        height, width = image.shape[:2]
        snapped = np.array([columns.min(), rows.min(), columns.max() + 1, rows.max() + 1]) / scale
        return [float(v) for v in np.clip(snapped, 0, [width, height, width, height])]

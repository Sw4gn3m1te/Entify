import contextlib
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image

from entify.quiet import silenced_stderr

SAM_PRESET = "sam_base_sa1b"
SAM_SIZE = 1024
CACHED_IMAGES = 8


class Cancelled(Exception):
    """An image encoding was stopped because another image is needed first."""


def run_encoder(encoder, images, cancelled):
    """Runs a Keras functional model operation by operation and stops between two operations once cancelled().

    SAM's encoder has 16 layers and some graph operations, so a stale encoding stops within about a second instead of
    blocking the next one for 10 seconds. Uses Keras' internal graph runner; if that is not available (another Keras
    version), the model runs in one piece and cannot be stopped.
    """
    try:
        from keras.src.models.functional import operation_fn

        run_graph = encoder._run_through_graph
        standardize = encoder._standardize_inputs
    except (ImportError, AttributeError):
        return encoder.predict(images, verbose=0)

    def checked(operation):
        call = operation_fn(operation, training=False)

        def run(*args, **kwargs):
            if cancelled():
                raise Cancelled
            return call(*args, **kwargs)

        return run

    outputs = run_graph(standardize(images), operation_fn=checked)
    return np.asarray(outputs[0] if isinstance(outputs, (list, tuple)) else outputs)


class BoxSnapper:
    """Tightens a rough box around the object inside it with Segment Anything (SAM, KerasHub).

    SAM's image encoder takes about 10 seconds per image on a CPU, so its result is cached per image and only the
    cheap prompt encoder and mask decoder (well under a second) run for each box. The review server encodes images in
    the background while they are being looked at, so the first snap on an image usually finds the encoding ready.
    Thread-safe: one image is encoded at a time, snaps on encoded images never wait for that, and an encoding stops
    as soon as another image becomes the focus (see focus()).
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.backbone = None
        self.embeddings = OrderedDict()   # key -> (embedding, scale, (height, width))
        self.load_lock = threading.Lock()
        self.encode_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.focused = None

    def focus(self, key):
        """Marks the image that is needed now; a running encoding of any other image stops."""
        self.focused = key

    def load(self):
        with self.load_lock:
            if self.backbone is None:
                with contextlib.nullcontext() if self.verbose else silenced_stderr():
                    import keras_hub

                    from entify.models.loading import local_preset

                    self.backbone = keras_hub.models.SAMImageSegmenter.from_preset(local_preset(SAM_PRESET)).backbone

    def cached(self, key):
        with self.cache_lock:
            if key in self.embeddings:
                self.embeddings.move_to_end(key)
                return self.embeddings[key]
        return None

    def embedding(self, key, load_image):
        """The SAM encoding of an image; load_image() is only called when the image is not cached yet.

        Raises Cancelled if another image gets the focus meanwhile.
        """
        found = self.cached(key)
        if found is not None:
            return found
        self.load()
        with self.encode_lock:
            if self.focused != key:
                raise Cancelled
            found = self.cached(key)
            if found is not None:
                return found
            image = load_image()
            # SAM expects the longest side at 1024 px, padded to a square, with values from 0 to 1.
            scale = SAM_SIZE / max(image.shape[:2])
            height, width = round(image.shape[0] * scale), round(image.shape[1] * scale)
            padded = np.zeros((SAM_SIZE, SAM_SIZE, 3), dtype=np.float32)
            padded[:height, :width] = np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))
            encoded = run_encoder(self.backbone.image_encoder, padded[None] / 255, lambda: self.focused != key)
            found = encoded, scale, image.shape[:2]
            with self.cache_lock:
                self.embeddings[key] = found
                if len(self.embeddings) > CACHED_IMAGES:
                    self.embeddings.popitem(last=False)
            return found

    def snap(self, key, load_image, box):
        import keras

        self.focus(key)
        embedding, scale, (height, width) = self.embedding(key, load_image)
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
        snapped = np.array([columns.min(), rows.min(), columns.max() + 1, rows.max() + 1]) / scale
        return [float(v) for v in np.clip(snapped, 0, [width, height, width, height])]

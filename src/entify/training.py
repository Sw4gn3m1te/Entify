import json
import random
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
from tqdm import tqdm

from entify.evaluation import load_ground_truth
from entify.images import load_image
from entify.models.base import resize
from entify.models.labels import COCO_CLASSES
from entify.models.loading import MODEL_DIR, SETTINGS_FILE
from entify.review.store import ANNOTATIONS_FILE


class Sample:
    """One training image: its path and its objects as (category name, x1, y1, x2, y2) in image pixels.

    checked: the categories whose objects were all checked by a person, None if all of them were.
    """

    def __init__(self, path, objects, source, checked=None):
        self.path = Path(path)
        self.objects = objects
        self.source = source
        self.checked = None if checked is None else frozenset(checked)

    def counts(self, category):
        return self.checked is None or category in self.checked


def reviewed_samples(review_dir):
    """Reviewed images that are fully checked, or checked for some categories (entify review -s).

    An object nobody marked would teach the model that it is background. On an image checked only for some
    categories, the objects of the other categories are therefore the model's own detections as the review saved
    them: pseudo-labels that keep the model as it was for those categories, while the checked ones are corrected.
    """
    coco = json.loads((Path(review_dir) / ANNOTATIONS_FILE).read_text(encoding="utf-8"))
    names = {category["id"]: category["name"] for category in coco["categories"]}
    objects = {image["id"]: [] for image in coco["images"]}
    for annotation in coco["annotations"]:
        x, y, width, height = annotation["bbox"]
        objects[annotation["image_id"]].append((names[annotation["category_id"]], x, y, x + width, y + height))
    return [
        Sample(image["file_name"], objects[image["id"]], "review",
               None if image.get("entify_complete") else image.get("entify_checked"))
        for image in coco["images"] if image.get("entify_complete") or image.get("entify_checked")
    ]


def describe_checked(samples):
    """How many images are fully checked and how many only for some categories, e.g. '4 fully, 30 for person'."""
    groups = {}
    for sample in samples:
        key = "fully" if sample.checked is None else "for " + ", ".join(sorted(sample.checked))
        groups[key] = groups.get(key, 0) + 1
    return ", ".join(f"{count} {key}" for key, count in sorted(groups.items(), key=lambda item: item[0] != "fully"))


def coco_samples(dataset, count, seed, exclude=()):
    """Random images from a COCO-format dataset (data/coco_subset), mixed in so the model keeps the other categories.

    exclude: file names to leave out, e.g. the reviewed images, so none is trained on both its review and COCO's labels.
    """
    dataset = Path(dataset)
    exclude = set(exclude)
    coco = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))
    names = {category["id"]: category["name"] for category in coco["categories"]}
    objects = {image["id"]: [] for image in coco["images"]}
    for annotation in coco["annotations"]:
        if not annotation.get("iscrowd"):
            x, y, width, height = annotation["bbox"]
            objects[annotation["image_id"]].append((names[annotation["category_id"]], x, y, x + width, y + height))
    images = sorted((image for image in coco["images"] if image["file_name"] not in exclude), key=lambda image: image["id"])
    random.Random(seed).shuffle(images)
    return [Sample(dataset / "images" / image["file_name"], objects[image["id"]], "coco") for image in images[:count]]


def split(samples, val_share, seed):
    samples = list(samples)
    random.Random(seed).shuffle(samples)
    count = round(len(samples) * val_share) if len(samples) > 1 else 0
    return samples[count:], samples[:count]


def training_example(sample, model):
    """Image resized to the model input and boxes as normalised (centre x, centre y, width, height).

    D-FINE's loss compares these directly with its predictions. The resize does not crop, so normalised coordinates
    are the same before and after it.
    """
    image = load_image(sample.path)
    height, width = image.shape[:2]
    x = resize(image, model.input_size).astype(np.float32)[None] / 255
    known = [item for item in sample.objects if item[0] in model.categories]
    boxes = np.array(
        [[(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]
         for _, x1, y1, x2, y2 in known], dtype=np.float32,
    ).reshape(1, -1, 4)
    labels = np.array([model.categories.index(name) for name, *_ in known], dtype=np.int32).reshape(1, -1)
    return x, {"boxes": np.clip(boxes, 0, 1), "labels": labels}


def dataset(samples, model, seed):
    import tensorflow as tf

    def generate():
        order = list(samples)
        random.Random(seed).shuffle(order)
        for sample in order:
            x, y = training_example(sample, model)
            if y["labels"].size:
                yield x, y

    height, width = model.input_size
    signature = (
        tf.TensorSpec((1, height, width, 3), tf.float32),
        {"boxes": tf.TensorSpec((1, None, 4), tf.float32), "labels": tf.TensorSpec((1, None), tf.int32)},
    )
    return tf.data.Dataset.from_generator(generate, output_signature=signature)


def next_model_dir(name):
    base = re.sub(r"(_v\d+|_ft\d+)+$", "", Path(name).name)
    number = 1
    while (MODEL_DIR / f"{base}_ft{number}").exists():
        number += 1
    return MODEL_DIR / f"{base}_ft{number}"


class ProgressBar:
    """Keras callback with a tqdm bar per epoch and the loss."""

    def __init__(self, steps):
        import keras

        class Callback(keras.callbacks.Callback):
            def on_epoch_begin(callback, epoch, logs=None):
                callback.bar = tqdm(total=steps, desc=f"epoch {epoch + 1}", unit="img")
                callback.started = time.perf_counter()

            def on_train_batch_end(callback, batch, logs=None):
                callback.bar.update(1)
                callback.bar.set_postfix(loss=f"{(logs or {}).get('loss', float('nan')):.3f}")

            def on_epoch_end(callback, epoch, logs=None):
                callback.bar.close()

        self.callback = Callback()


def finetune(model, train, epochs, learning_rate, freeze_backbone, seed):
    """Fine-tune a KerasDetector in place on the training samples."""
    import keras

    task = model.task
    keras.utils.set_random_seed(seed)
    task.backbone.backbone.trainable = not freeze_backbone
    # With one image per step, batch normalisation would replace its pretrained statistics with those of single
    # images. Frozen BatchNormalization layers run in inference mode and keep them.
    batch_norms = [layer for layer in task._flatten_layers() if isinstance(layer, keras.layers.BatchNormalization)]
    for layer in batch_norms:
        layer.trainable = False
    preprocessor, task.preprocessor = task.preprocessor, None
    try:
        # D-FINE overrides compute_loss(), which Keras calls during fit(); compile() only needs the optimizer.
        task.compile(optimizer=keras.optimizers.Adam(learning_rate, clipnorm=0.1))
        data = dataset(train, model, seed)
        steps = sum(1 for sample in train if any(item[0] in model.categories for item in sample.objects))
        history = task.fit(data.repeat(epochs), epochs=epochs, steps_per_epoch=steps, verbose=0,
                           callbacks=[ProgressBar(steps).callback])
    finally:
        task.preprocessor = preprocessor
        task.backbone.backbone.trainable = True
        for layer in batch_norms:
            layer.trainable = True
    return [float(loss) for loss in history.history["loss"]]


def save_finetuned(model, out_dir, settings):
    model.restore_decoder()
    model.task.save_to_preset(str(out_dir))
    model.configure_decoder()
    settings = {"labels": list(model.categories), "created": date.today().isoformat(), **settings}
    (Path(out_dir) / SETTINGS_FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")


def ground_truth_pairs(samples):
    # Only checked categories: pseudo-labels are the model's own detections and would make it look better than it is.
    return [(sample.path.name, name) for sample in samples for name, *_ in sample.objects if sample.counts(name)]


def uses_coco_categories(model):
    return tuple(model.categories) == COCO_CLASSES

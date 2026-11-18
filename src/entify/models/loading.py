import contextlib
import json
import os
import re
from pathlib import Path

from entify.models.labels import builtin_labels, read_labels
from entify.quiet import silence_library_logs, silenced_stderr

DEFAULT_MODEL = "dfine_small_coco"
SETTINGS_FILE = "entify.json"
DOWNLOADED = ".downloaded"


class ModelError(Exception):
    pass


def model_dir():
    # Keep the weights in the project's models/ folder when running from the source tree, otherwise in the user's cache.
    project_root = Path(__file__).resolve().parents[3]
    if (project_root / "pyproject.toml").is_file():
        return project_root / "models"
    return Path.home() / ".cache" / "entify" / "models"


MODEL_DIR = model_dir()


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


def read_settings(source):
    # Settings for a model live in entify.json inside a model folder, or in <name>.entify.json next to a model file.
    path = Path(source)
    settings = path / SETTINGS_FILE if path.is_dir() else path.with_name(f"{path.stem}.{SETTINGS_FILE}")
    return json.loads(settings.read_text(encoding="utf-8")) if settings.is_file() else {}


def resolve_categories(spec, labels_file, settings):
    if labels_file:
        return read_labels(labels_file)
    if settings.get("labels"):
        return tuple(settings["labels"])
    labels = builtin_labels(spec) or builtin_labels(settings.get("base_model", ""))
    if labels is None:
        raise ModelError(f"category names of model {spec} are unknown, pass them with --labels FILE")
    return labels


def is_plain_model(path):
    return path.suffix.lower() in (".keras", ".h5") or (path / "saved_model.pb").is_file()


def load_plain(path, settings):
    import keras

    if path.suffix.lower() in (".keras", ".h5"):
        network = keras.saving.load_model(path, compile=False)
        input_shape = network.input_shape
        return network, lambda x: network.predict(x, verbose=0), input_shape
    layer = keras.layers.TFSMLayer(str(path), call_endpoint=settings.get("endpoint", "serving_default"))

    def predict(x):
        output = layer(x)
        return next(iter(output.values())) if isinstance(output, dict) else output

    return layer, predict, (None, None, None, None)


def load_task(spec, source):
    # KerasHub only loads presets through a concrete task class, so try the two kinds Entify supports.
    import keras_hub

    errors = []
    for task_class in (keras_hub.models.ObjectDetector, keras_hub.models.ImageSegmenter):
        try:
            return task_class.from_preset(source)
        except (ValueError, FileNotFoundError) as error:
            errors.append(str(error))
    raise ModelError(f"cannot load model {spec}: {errors[-1]}")


def build(spec, source, settings, categories, confidence, overlap, min_area):
    import keras_hub

    from entify.models.keras_segmenter import KerasSegmenter

    path = Path(source)
    if not is_plain_model(path):
        return from_task(spec, load_task(spec, source), categories, confidence, overlap, min_area)
    network, predict, input_shape = load_plain(path, settings)
    if isinstance(network, (keras_hub.models.ObjectDetector, keras_hub.models.ImageSegmenter)):
        return from_task(spec, network, categories, confidence, overlap, min_area)
    # Any other plain model is a segmentation network described by the settings file.
    if settings.get("kind", "masks") != "masks":
        raise ModelError("plain Keras models must be segmentation models; save detectors as KerasHub presets")
    input_size = settings.get("input_size") or input_shape[1:3]
    if not input_size or None in input_size:
        raise ModelError(f"input size of {spec} is unknown, set \"input_size\" in {SETTINGS_FILE}")
    return KerasSegmenter(
        spec, predict, categories, input_size, confidence, min_area,
        channels=settings.get("channels") or input_shape[-1] or 3,
        input_scale=settings.get("input_scale", 1.0),
        logits=settings.get("output") == "logits",
    )


def from_task(spec, task, categories, confidence, overlap, min_area):
    import keras_hub

    from entify.models.keras_detector import KerasDetector
    from entify.models.keras_segmenter import KerasSegmenter

    if isinstance(task, keras_hub.models.ObjectDetector):
        return KerasDetector(spec, task, categories, confidence, overlap)
    if isinstance(task, keras_hub.models.ImageSegmenter):
        return KerasSegmenter(
            spec, lambda x: task.predict(x, verbose=0), categories,
            task.preprocessor.image_converter.image_size, confidence, min_area,
            logits=getattr(task, "activation", None) in (None, "linear"),
        )
    raise ModelError(f"{spec} is a {type(task).__name__}, Entify supports object detectors and image segmenters")


def check_category_count(model):
    expected = getattr(getattr(model, "task", None), "num_classes", None)
    if expected and expected != len(model.categories):
        raise ModelError(f"model {model.name} has {expected} categories but {len(model.categories)} names were given")


def load_model(spec=DEFAULT_MODEL, labels=None, confidence=0.5, overlap=0.5, min_area=None, verbose=False):
    """Load a KerasHub preset, a local preset folder, a .keras file or a SavedModel as an EntityModel."""
    if not verbose:
        silence_library_logs()
    os.environ.setdefault("KAGGLEHUB_CACHE", str(MODEL_DIR))
    spec = str(spec)
    with contextlib.nullcontext() if verbose else silenced_stderr():
        from keras_hub.src.utils.preset_utils import BUILTIN_PRESETS

        if spec in BUILTIN_PRESETS:
            settings = {}
        elif Path(spec).exists():
            settings = read_settings(spec)
        else:
            raise ModelError(f"unknown model {spec}: not a KerasHub preset, model folder, .keras file or SavedModel")
        # Check the category names before downloading, so an unusable preset fails fast.
        categories = resolve_categories(spec, labels, settings)
        source = local_preset(spec) if spec in BUILTIN_PRESETS else spec
        model = build(spec, source, settings, categories, confidence, overlap, min_area)
    check_category_count(model)
    return model

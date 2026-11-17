import io
import threading
import time
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from entify.images import load_image
from entify.review.snap import BoxSnapper, Cancelled
from entify.review.store import ReviewStore

STATIC_DIR = Path(__file__).parent / "static"


class ReviewObject(BaseModel):
    box: list[float]
    category: str
    source: str = "manual"
    score: float | None = None


class ReviewUpdate(BaseModel):
    objects: list[ReviewObject]
    complete: bool = False


class SnapRequest(BaseModel):
    box: list[float]


def image_size(path):
    # Size after EXIF rotation, the same orientation load_image() returns.
    with Image.open(path) as image:
        width, height = image.size
        return (height, width) if image.getexif().get(0x0112) in (5, 6, 7, 8) else (width, height)


class Preparer:
    """Runs slow preparations (SAM model, image encoding, detections) in one background thread.

    want() replaces the waiting tasks, so after a switch to another image its preparations come first. Tasks start
    only after `delay` seconds without a new wish, so paging quickly through images does not start work for each one.
    """

    def __init__(self, delay=0.5):
        self.delay = delay
        self.tasks = []
        self.wished = 0.0
        self.condition = threading.Condition()
        threading.Thread(target=self.work, daemon=True).start()

    def want(self, tasks):
        with self.condition:
            self.tasks = list(tasks)
            self.wished = time.monotonic()
            self.condition.notify()

    def work(self):
        while True:
            with self.condition:
                while not self.tasks or time.monotonic() < self.wished + self.delay:
                    self.condition.wait(None if not self.tasks else self.wished + self.delay - time.monotonic())
                task = self.tasks.pop(0)
            try:
                task()
            except Cancelled:
                pass
            except Exception as error:  # a failed preparation is retried by the request that needs it
                print(f"Preparing in the background failed: {error}")


def create_app(paths, model, find_objects, store, snapper, prepare=True):
    app = FastAPI(title="Entify review")
    model_lock = threading.Lock()
    proposals = {}
    sizes = {index: image_size(path) for index, path in enumerate(paths)}
    preparer = Preparer() if prepare else None

    def image_path(image_id):
        if not 0 <= image_id < len(paths):
            raise HTTPException(404, "no such image")
        return paths[image_id]

    def status(index, path):
        saved = store.get(path)
        return {
            "id": index, "name": path.name, "width": sizes[index][0], "height": sizes[index][1],
            "reviewed": saved is not None, "complete": bool(saved and saved["complete"]),
        }

    def detect(image_id):
        with model_lock:
            if image_id not in proposals:
                detections = find_objects(model, load_image(paths[image_id]))
                proposals[image_id] = [
                    {"box": [float(v) for v in box], "category": model.categories[label], "source": "model",
                     "score": float(score)}
                    for box, score, label in zip(detections.boxes, detections.scores, detections.labels)
                ]
        return proposals[image_id]

    def encode(image_id):
        path = paths[image_id]
        snapper.embedding(str(path), lambda: load_image(path))

    def prepare_around(image_id):
        # While an image is open, encode it for snapping and detect the next one. The next image is not encoded in
        # advance: two encodings at once would slow down the snaps on the open image several times over.
        snapper.focus(str(paths[image_id]))
        if preparer is None:
            return
        tasks = [lambda: encode(image_id)]
        following = image_id + 1
        if following < len(paths) and store.get(paths[following]) is None:
            tasks.append(lambda: detect(following))
        preparer.want(tasks)

    if preparer is not None and paths:
        # The detector's first call takes several seconds longer than the others, so it runs on the first image that
        # has no saved review yet, then SAM is loaded, before the browser asks for anything.
        unsaved = next((index for index, path in enumerate(paths) if store.get(path) is None), None)
        preparer.want(([lambda: detect(unsaved)] if unsaved is not None else []) + [snapper.load])

    @app.get("/api/session")
    def session():
        return {
            "model": model.name,
            "kind": model.kind,
            "categories": list(model.selectable_categories),
            "output": str(store.path),
            "images": [status(index, path) for index, path in enumerate(paths)],
        }

    @app.get("/api/images/{image_id}/file")
    def image_file(image_id: int):
        path = image_path(image_id)
        with Image.open(path) as image:
            if image.getexif().get(0x0112, 1) == 1 and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                return FileResponse(path)
            buffer = io.BytesIO()
            ImageOps.exif_transpose(image).convert("RGB").save(buffer, format="JPEG", quality=92)
        return Response(buffer.getvalue(), media_type="image/jpeg")

    @app.get("/api/images/{image_id}/annotations")
    def annotations(image_id: int):
        path = image_path(image_id)
        saved = store.get(path)
        if saved is not None:
            result = {"objects": saved["objects"], "complete": saved["complete"], "saved": True}
        else:
            result = {"objects": detect(image_id), "complete": False, "saved": False}
        prepare_around(image_id)
        return result

    @app.put("/api/images/{image_id}/annotations")
    def save(image_id: int, update: ReviewUpdate):
        path = image_path(image_id)
        objects = [item.model_dump() for item in update.objects]
        store.put(path, sizes[image_id], objects, update.complete)
        return status(image_id, path)

    @app.post("/api/images/{image_id}/snap")
    def snap(image_id: int, request: SnapRequest):
        path = image_path(image_id)
        try:
            return {"box": snapper.snap(str(path), lambda: load_image(path), request.box)}
        except Cancelled:
            raise HTTPException(409, "another image was opened while this one was being prepared")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def run_review(paths, model, find_objects, out_dir, port=8123, open_browser=True, verbose=False):
    import uvicorn

    store = ReviewStore(out_dir, model.selectable_categories, model.name)
    app = create_app(paths, model, find_objects, store, BoxSnapper(verbose))
    url = f"http://127.0.0.1:{port}/"
    print(f"Review {len(paths)} images at {url}  (corrections are saved to {store.path}, stop with Ctrl+C)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

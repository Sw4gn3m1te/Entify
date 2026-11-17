import io
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from entify.images import load_image
from entify.review.snap import BoxSnapper
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


def create_app(paths, model, find_objects, store, snapper):
    app = FastAPI(title="Entify review")
    model_lock = threading.Lock()
    proposals = {}
    sizes = {index: image_size(path) for index, path in enumerate(paths)}

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
            return {"objects": saved["objects"], "complete": saved["complete"], "saved": True}
        if image_id not in proposals:
            with model_lock:
                detections = find_objects(model, load_image(path))
            proposals[image_id] = [
                {"box": [float(v) for v in box], "category": model.categories[label], "source": "model",
                 "score": float(score)}
                for box, score, label in zip(detections.boxes, detections.scores, detections.labels)
            ]
        return {"objects": proposals[image_id], "complete": False, "saved": False}

    @app.put("/api/images/{image_id}/annotations")
    def save(image_id: int, update: ReviewUpdate):
        path = image_path(image_id)
        objects = [item.model_dump() for item in update.objects]
        store.put(path, sizes[image_id], objects, update.complete)
        return status(image_id, path)

    @app.post("/api/images/{image_id}/snap")
    def snap(image_id: int, request: SnapRequest):
        path = image_path(image_id)
        with model_lock:
            return {"box": snapper.snap(str(path), load_image(path), request.box)}

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def run_review(paths, model, find_objects, out_dir, port=8765, open_browser=True, verbose=False):
    import uvicorn

    store = ReviewStore(out_dir, model.selectable_categories, model.name)
    app = create_app(paths, model, find_objects, store, BoxSnapper(verbose))
    url = f"http://127.0.0.1:{port}/"
    print(f"Review {len(paths)} images at {url}  (corrections are saved to {store.path}, stop with Ctrl+C)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

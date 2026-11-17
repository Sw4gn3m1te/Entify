import json
from datetime import date
from pathlib import Path

ANNOTATIONS_FILE = "annotations.json"


class ReviewStore:
    """Reviewed images and their objects, saved as a COCO annotation file.

    Images use absolute paths as file_name, so `entify evaluate REVIEW_DIR` and `entify finetune REVIEW_DIR` can find
    them. Extra fields: images have "entify_complete" (every object in the image was checked), annotations have
    "entify_source" (model, adjusted, manual or snapped).
    """

    def __init__(self, folder, categories, model_name):
        self.path = Path(folder) / ANNOTATIONS_FILE
        self.model_name = model_name
        self.categories = list(categories)
        self.images = {}
        if self.path.exists():
            self.load()

    def load(self):
        coco = json.loads(self.path.read_text(encoding="utf-8"))
        names = {category["id"]: category["name"] for category in coco["categories"]}
        self.categories += [name for name in names.values() if name not in self.categories]
        by_image = {image["id"]: image for image in coco["images"]}
        for image in coco["images"]:
            self.images[image["file_name"]] = {
                "width": image["width"], "height": image["height"],
                "complete": image.get("entify_complete", False), "objects": [],
            }
        for annotation in coco["annotations"]:
            x, y, width, height = annotation["bbox"]
            self.images[by_image[annotation["image_id"]]["file_name"]]["objects"].append({
                "box": [x, y, x + width, y + height],
                "category": names[annotation["category_id"]],
                "source": annotation.get("entify_source", "manual"),
                "score": annotation.get("score"),
            })

    def get(self, image_path):
        return self.images.get(str(Path(image_path).resolve()))

    def put(self, image_path, size, objects, complete):
        width, height = size
        self.images[str(Path(image_path).resolve())] = {
            "width": width, "height": height, "complete": complete, "objects": objects,
        }
        for item in objects:
            if item["category"] not in self.categories:
                self.categories.append(item["category"])
        self.save()

    def save(self):
        category_ids = {name: index for index, name in enumerate(self.categories, 1)}
        images, annotations = [], []
        for image_id, (file_name, image) in enumerate(sorted(self.images.items()), 1):
            images.append({
                "id": image_id, "file_name": file_name, "width": image["width"], "height": image["height"],
                "entify_complete": image["complete"],
            })
            for item in image["objects"]:
                x1, y1, x2, y2 = item["box"]
                annotation = {
                    "id": len(annotations) + 1, "image_id": image_id, "category_id": category_ids[item["category"]],
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0, "entify_source": item["source"],
                }
                if item.get("score") is not None:
                    annotation["score"] = round(item["score"], 4)
                annotations.append(annotation)
        coco = {
            "info": {"description": f"Entify review with model {self.model_name}", "date_created": date.today().isoformat()},
            "images": images,
            "annotations": annotations,
            "categories": [{"id": index, "name": name} for name, index in category_ids.items()],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".part")
        partial.write_text(json.dumps(coco, indent=1), encoding="utf-8")
        partial.replace(self.path)

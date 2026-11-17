import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as file:
        shutil.copyfileobj(response, file)
    partial.replace(target)


def load_annotations(cache_dir):
    archive = cache_dir / "annotations_trainval2017.zip"
    if not archive.exists():
        print(f"Downloading {ANNOTATIONS_URL}")
        download(ANNOTATIONS_URL, archive)
    with zipfile.ZipFile(archive) as zf, zf.open(ANNOTATIONS_MEMBER) as file:
        return json.load(file)


def usable_images(coco):
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    return {
        image_id: anns
        for image_id, anns in annotations.items()
        if not any(a["iscrowd"] for a in anns)
    }


def pick_diverse(candidates, annotations, count, covered):
    picked = []
    pool = list(candidates)
    for _ in range(min(count, len(pool))):
        best = max(pool, key=lambda i: len({a["category_id"] for a in annotations[i]} - covered))
        pool.remove(best)
        picked.append(best)
        covered |= {a["category_id"] for a in annotations[best]}
    return picked


def select(annotations, dense, medium, diverse, seed):
    rng = random.Random(seed)
    ids = sorted(annotations)
    rng.shuffle(ids)
    dense_ids = [i for i in ids if len(annotations[i]) >= 15][:dense]
    medium_ids = [i for i in ids if 5 <= len(annotations[i]) < 15][:medium]
    chosen = set(dense_ids) | set(medium_ids)
    covered = {a["category_id"] for i in chosen for a in annotations[i]}
    rest = [i for i in ids if i not in chosen and len(annotations[i]) < 15]
    return dense_ids + medium_ids + pick_diverse(rest, annotations, diverse, covered)


def write_sources(path, coco, images):
    licenses = {lic["id"]: lic for lic in coco["licenses"]}
    usage = Counter(image["license"] for image in images)
    lines = [
        "# COCO val2017 subset",
        "",
        "Source: COCO 2017 validation set, https://cocodataset.org",
        "Annotations: CC BY 4.0, COCO Consortium.",
        "Images: Flickr images under the individual licences below. Used for evaluation only, not redistributed.",
        "",
        "| Licence | Images |",
        "|---|---|",
    ]
    lines += [f"| [{licenses[i]['name']}]({licenses[i]['url']}) | {n} |" for i, n in sorted(usage.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Download a reproducible subset of COCO val2017.")
    parser.add_argument("--dense", type=int, default=30, help="images with 15 or more objects")
    parser.add_argument("--medium", type=int, default=30, help="images with 5 to 14 objects")
    parser.add_argument("--diverse", type=int, default=25, help="images chosen to cover remaining categories")
    parser.add_argument("--seed", type=int, default=0, help="seed for the image selection, same seed gives the same subset (default: %(default)s)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    out_dir = args.data_dir / "coco_subset"
    image_dir = out_dir / "images"
    coco = load_annotations(args.data_dir / "cache")
    annotations = usable_images(coco)
    selected = select(annotations, args.dense, args.medium, args.diverse, args.seed)

    images_by_id = {image["id"]: image for image in coco["images"]}
    images = [images_by_id[i] for i in selected]
    for n, image in enumerate(images, 1):
        target = image_dir / image["file_name"]
        if not target.exists():
            download(image["coco_url"], target)
        print(f"[{n}/{len(images)}] {image['file_name']}")

    subset = {
        "info": coco["info"],
        "licenses": coco["licenses"],
        "categories": coco["categories"],
        "images": images,
        "annotations": [a for i in selected for a in annotations[i]],
    }
    (out_dir / "annotations.json").write_text(json.dumps(subset), encoding="utf-8")
    write_sources(out_dir / "SOURCES.md", coco, images)

    categories = {a["category_id"] for a in subset["annotations"]}
    print(f"{len(images)} images, {len(subset['annotations'])} objects, {len(categories)} of 80 categories")


if __name__ == "__main__":
    main()

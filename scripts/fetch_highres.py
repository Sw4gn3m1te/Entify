import argparse
import json
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "entify-data-fetch/0.0.1 (open-source image counting project)"
MAX_WIDTH = 3840
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

TITLES = [
    "File:A large parking lot at the Staten Island Mall.jpg",
    "File:Kippenheim vehicles logistics giant parking lot.jpg",
    "File:Driving Cars in a Traffic Jam.jpg",
    "File:Crowd at the stage of Christopher Street Day Berlin 2023-07-22 12.jpg",
    "File:Street Crowd (32418904786).jpg",
    "File:Sheep flock Niesgrau.jpg",
    "File:Flock of sheep in Vanoise National Park (2).jpg",
    "File:Cow Herd.jpg",
    "File:Bicycle parking in Deutsches Museum - Building.jpg",
    "File:Bicycle Parking in Berlin on 2026-06-01.jpg",
    "File:Feeding the pigeons, London.jpg",
    "File:Feral Pigeons Columba livia domestica, Trinity Square Toronto 2215.jpg",
    "File:Flock-of-birds-in-tree.jpg",
    "File:Fremantle Fishing Boat Harbour gnangarra 5291.JPG",
    "File:Fruit stall at Bangalore City Market.jpg",
]


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60)


def fetch_metadata(titles):
    params = {
        "action": "query",
        "format": "json",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": MAX_WIDTH,
        "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist",
    }
    with request(API_URL + "?" + urllib.parse.urlencode(params)) as response:
        pages = json.load(response)["query"]["pages"].values()
    return {page["title"]: page.get("imageinfo", [None])[0] for page in pages}


def plain(html):
    return re.sub(r"<[^>]+>", "", html).strip()


def file_name(title):
    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(title.removeprefix("File:")).stem).strip("_")
    return f"{stem}.jpg"


def main():
    parser = argparse.ArgumentParser(description="Download large, freely licensed photos from Wikimedia Commons.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    out_dir = args.data_dir / "highres"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = fetch_metadata(TITLES)
    rows = []
    for n, title in enumerate(TITLES, 1):
        info = metadata.get(title)
        if info is None:
            print(f"[{n}/{len(TITLES)}] missing on Commons: {title}")
            continue
        meta = info["extmetadata"]
        url = info.get("thumburl", info["url"])
        target = out_dir / file_name(title)
        if not target.exists():
            with request(url) as response, open(target, "wb") as file:
                shutil.copyfileobj(response, file)
        rows.append(
            f"| {target.name} | {plain(meta.get('Artist', {}).get('value', 'unknown'))} "
            f"| [{meta['LicenseShortName']['value']}]({meta.get('LicenseUrl', {}).get('value', '')}) "
            f"| {info['descriptionurl']} |"
        )
        print(f"[{n}/{len(TITLES)}] {target.name}")

    header = [
        "# High-resolution photos",
        "",
        f"Source: Wikimedia Commons, downscaled to at most {MAX_WIDTH} px width.",
        "",
        "| File | Author | Licence | Source |",
        "|---|---|---|---|",
    ]
    (out_dir / "SOURCES.md").write_text("\n".join(header + rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

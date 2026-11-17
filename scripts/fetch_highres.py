import argparse
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "entify-data-fetch/0.0.1 (open-source image counting project)"
MAX_WIDTH = 3840
MAX_RESULTS = 50
MAX_OFFSET = 10000
FALLBACK_WIDTH = 1920
DOWNLOAD_PAUSE = 2
MAX_WAIT = 30
CHUNK_SIZE = 64 * 1024
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_QUERY = 'crowd OR herd OR flock OR "parking lot" OR "traffic jam" OR "bicycle parking" OR market'

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

IMAGEINFO_PARAMS = {
    "prop": "imageinfo",
    "iiprop": "url|size|extmetadata",
    "iiurlwidth": MAX_WIDTH,
    "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist",
}


def request(url, retries=2):
    for attempt in range(retries + 1):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == retries:
                raise
            wait = min(int(error.headers.get("Retry-After") or 0) or 5 * 2**attempt, MAX_WAIT)
            print(f"rate limited, waiting {wait} s")
            time.sleep(wait)


def api(params):
    with request(API_URL + "?" + urllib.parse.urlencode({"action": "query", "format": "json", **params})) as response:
        return json.load(response)


def image_metadata(params):
    pages = api({**IMAGEINFO_PARAMS, **params}).get("query", {}).get("pages", {}).values()
    return {page["title"]: page.get("imageinfo", [None])[0] for page in pages}


def fetch_titles(titles):
    metadata = image_metadata({"titles": "|".join(titles)})
    return {title: metadata.get(title) for title in titles}


def fetch_random(search, count, min_width, seed):
    # Commons' random sort returns the same order on every call, so a random offset picks a different slice each run.
    search = f"{search} filew:>{min_width - 1} filemime:image/jpeg"
    total = api({"list": "search", "srnamespace": 6, "srsearch": search, "srinfo": "totalhits", "srlimit": 1})
    total = total["query"]["searchinfo"]["totalhits"]
    offset = random.Random(seed).randrange(max(1, min(total, MAX_OFFSET) - count + 1))
    print(f"{total} matching photos, seed {seed}, offset {offset}")
    params = {
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": search,
        "gsrsort": "random",
        "gsrlimit": count,
        "gsroffset": offset,
    }
    return image_metadata(params)


def candidate_urls(info):
    # Commons only serves standard thumbnail widths and may refuse to render a large one, so fall back to a smaller one.
    urls = [info.get("thumburl", info["url"])]
    if info["width"] > FALLBACK_WIDTH:
        original = info["url"].split("?")[0]
        base, name = original.replace("/commons/", "/commons/thumb/", 1), original.rsplit("/", 1)[1]
        urls.append(f"{base}/{FALLBACK_WIDTH}px-{name}")
    return list(dict.fromkeys(urls))


def save(urls, target, label):
    partial = target.with_suffix(".part")
    for url in urls:
        try:
            with request(url) as response, open(partial, "wb") as file:
                size = int(response.headers.get("Content-Length") or 0) or None
                with tqdm(total=size, desc=label, unit="B", unit_scale=True, unit_divisor=1024, leave=False) as bar:
                    while chunk := response.read(CHUNK_SIZE):
                        file.write(chunk)
                        bar.update(len(chunk))
        except urllib.error.HTTPError as error:
            print(f"  HTTP {error.code} for {url.split('?')[0]}")
            continue
        partial.replace(target)
        return True
    partial.unlink(missing_ok=True)
    return False


def plain(html):
    return re.sub(r"<[^>]+>", "", html).strip()


def shorten(name, length=40):
    return name if len(name) <= length else name[:length - 3] + "..."


def file_name(title):
    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(title.removeprefix("File:")).stem).strip("_")
    return f"{stem}.jpg"


def read_sources(path):
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if line.startswith("| ") and cells[0] not in ("File", ""):
            rows[cells[0]] = line
    return rows


def write_sources(path, rows):
    header = [
        "# High-resolution photos",
        "",
        f"Source: Wikimedia Commons, downscaled to at most {MAX_WIDTH} px width.",
        "",
        "| File | Author | Licence | Source |",
        "|---|---|---|---|",
    ]
    path.write_text("\n".join(header + [rows[name] for name in sorted(rows)]) + "\n", encoding="utf-8")


def download(metadata, out_dir):
    sources_path = out_dir / "SOURCES.md"
    rows = read_sources(sources_path)
    for n, (title, info) in enumerate(metadata.items(), 1):
        if info is None:
            print(f"[{n}/{len(metadata)}] missing on Commons: {title}")
            continue
        meta = info.get("extmetadata", {})
        target = out_dir / file_name(title)
        if not target.exists():
            label = f"[{n}/{len(metadata)}] {shorten(target.name)}"
            saved = save(candidate_urls(info), target, label)
            time.sleep(DOWNLOAD_PAUSE)
            if not saved:
                print(f"[{n}/{len(metadata)}] skipped: {title}")
                continue
        licence = meta.get("LicenseShortName", {}).get("value", "unknown")
        licence_url = meta.get("LicenseUrl", {}).get("value", "")
        rows[target.name] = (
            f"| {target.name} | {plain(meta.get('Artist', {}).get('value', 'unknown'))} "
            f"| {f'[{licence}]({licence_url})' if licence_url else licence} "
            f"| {info['descriptionurl']} |"
        )
        write_sources(sources_path, rows)
        print(f"[{n}/{len(metadata)}] {target.name} ({info['width']}x{info['height']})")


def main():
    parser = argparse.ArgumentParser(description="Download large, freely licensed photos from Wikimedia Commons.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    modes = parser.add_subparsers(dest="mode", metavar="MODE")
    modes.add_parser("titles", help="download the fixed list of photos in TITLES (default)")
    random_mode = modes.add_parser("random", help="download random photos that match a Commons search")
    random_mode.add_argument("-q", "--query", default=DEFAULT_QUERY, help="Commons search query (default: %(default)s)")
    random_mode.add_argument("-n", "--count", type=int, default=15, help=f"number of photos, 1 to {MAX_RESULTS} (default: %(default)s)")
    random_mode.add_argument("--min-width", type=int, default=3000, help="minimum original width in px (default: %(default)s)")
    random_mode.add_argument("--seed", type=int, help="seed for the random selection, repeats a run while Commons is unchanged (default: new seed)")
    args = parser.parse_args()

    if args.mode == "random":
        if not 1 <= args.count <= MAX_RESULTS:
            parser.error(f"--count must be between 1 and {MAX_RESULTS}")
        seed = args.seed if args.seed is not None else random.randrange(2**32)
        out_dir = args.data_dir / "highres_random"
        metadata = fetch_random(args.query, args.count, args.min_width, seed)
        if not metadata:
            print("No photos matched the search.")
            return
    else:
        out_dir = args.data_dir / "highres"
        metadata = fetch_titles(TITLES)

    out_dir.mkdir(parents=True, exist_ok=True)
    download(metadata, out_dir)


if __name__ == "__main__":
    main()

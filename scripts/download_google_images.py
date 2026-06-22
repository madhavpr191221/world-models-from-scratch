from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import os
import re
import tempfile
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageFile
from serpapi import GoogleSearch


ImageFile.LOAD_TRUNCATED_IMAGES = True


STL10_CLASSES = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)

SEARCH_QUERIES = {
    "airplane": "airplane aircraft",
    "bird": "bird",
    "car": "car",
    "cat": "cat",
    "deer": "deer",
    "dog": "dog",
    "horse": "horse",
    "monkey": "monkey",
    "ship": "ship",
    "truck": "truck",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Google Images results into STL-10-style class folders."
    )
    parser.add_argument(
        "--output-dir",
        default="data/test_images",
        help="Root output directory containing class subfolders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of final images to keep per class.",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=300,
        help="How many search results to fetch per class before filtering.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=96,
        help="Output image size in pixels (square).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep existing PNGs and only fill missing slots.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="How many classes to process at the same time.",
    )
    parser.add_argument(
        "--pages-per-class",
        type=int,
        default=4,
        help="How many Google Images pages to fetch per class.",
    )
    parser.add_argument(
        "--num-per-page",
        type=int,
        default=100,
        help="How many image results to request per page.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.5,
        help="Pause between page requests to avoid hammering the API.",
    )
    parser.add_argument(
        "--min-side",
        type=int,
        default=128,
        help="Reject decoded images whose shorter side is below this size.",
    )
    parser.add_argument(
        "--manifest",
        default="data/test_images_manifest.csv",
        help="CSV file to write download metadata into.",
    )
    return parser.parse_args()


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def ensure_api_key() -> str:
    api_key = os.getenv("SERP_API_KEY") or os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing SERP_API_KEY. Put it in .env or export it before running the script."
        )
    return api_key


def ensure_class_dirs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for class_name in STL10_CLASSES:
        (root / class_name).mkdir(parents=True, exist_ok=True)


def count_existing_pngs(directory: Path) -> int:
    return sum(1 for path in directory.glob("*.png") if path.is_file())


def safe_filename(value: str) -> str:
    value = value.split("?")[0]
    value = urllib.parse.unquote(value)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def download_bytes(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def normalize_and_save(source: Path, target: Path, size: int) -> bool:
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image = image.resize((size, size), Image.Resampling.LANCZOS)
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, format="PNG")
        return True
    except Exception:
        return False


def image_is_large_enough(path: Path, min_side: int) -> bool:
    try:
        with Image.open(path) as image:
            return min(image.size) >= min_side
    except Exception:
        return False


def fetch_image_urls(query: str, api_key: str, pages: int, num_per_page: int) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    for page in range(pages):
        start = page * num_per_page
        search = GoogleSearch(
            {
                "engine": "google",
                "q": query,
                "tbm": "isch",
                "num": str(num_per_page),
                "start": str(start),
                "api_key": api_key,
                "safe": "active",
            }
        )
        results = search.get_dict()
        if results.get("error"):
            print(f"  page {page + 1}: {results['error']}")
            break

        images = results.get("images_results", [])
        if not images:
            break

        for image in images:
            original = image.get("original")
            if not original or original in seen:
                continue
            seen.add(original)
            urls.append(original)

    return urls


def download_class(
    class_name: str,
    output_root: Path,
    *,
    api_key: str,
    limit: int,
    search_limit: int,
    size: int,
    skip_existing: bool,
    pages_per_class: int,
    num_per_page: int,
    pause_seconds: float,
    min_side: int,
    manifest_rows: list[dict[str, str]],
    manifest_lock: threading.Lock,
) -> int:
    class_dir = output_root / class_name
    existing = count_existing_pngs(class_dir) if skip_existing else 0
    if existing >= limit:
        print(f"{class_name}: already has {existing} PNGs, skipping")
        return existing

    query = SEARCH_QUERIES.get(class_name, class_name)
    print(f"{class_name}: querying Google Images for '{query}'")
    urls = fetch_image_urls(
        query,
        api_key=api_key,
        pages=max(1, pages_per_class),
        num_per_page=max(1, num_per_page),
    )

    if search_limit > 0:
        urls = urls[:search_limit]

    if not urls:
        print(f"{class_name}: no image URLs returned")
        return existing

    next_index = existing + 1
    kept = existing
    with tempfile.TemporaryDirectory(prefix=f"{class_name}_downloads_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        print(f"{class_name}: found {len(urls)} candidate URLs")
        for idx, url in enumerate(urls, start=1):
            if kept >= limit:
                break

            print(f"{class_name}: [{idx}/{len(urls)}] downloading")
            tmp_path = tmp_root / f"{idx:04d}_{safe_filename(url)}"
            if not tmp_path.suffix:
                tmp_path = tmp_path.with_suffix(".bin")

            try:
                data = download_bytes(url)
                tmp_path.write_bytes(data)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
                print(f"{class_name}: skipped {url} ({exc})")
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"{class_name}: skipped {url} ({exc})")
                continue

            target = class_dir / f"{class_name}_{next_index:03d}.png"
            if skip_existing and target.exists():
                next_index += 1
                continue

            if not image_is_large_enough(tmp_path, min_side):
                continue

            if normalize_and_save(tmp_path, target, size):
                kept += 1
                next_index += 1
                with manifest_lock:
                    manifest_rows.append(
                        {
                            "class_name": class_name,
                            "saved_name": target.name,
                            "source_url": url,
                            "source_query": query,
                            "source_file": tmp_path.name,
                        }
                    )
                if pause_seconds > 0:
                    time.sleep(pause_seconds)
            else:
                print(f"{class_name}: could not decode {url}")

    print(f"{class_name}: wrote {kept} images")
    return kept


def main() -> int:
    args = parse_args()
    load_dotenv()
    api_key = ensure_api_key()

    output_root = Path(args.output_dir).resolve()
    ensure_class_dirs(output_root)
    manifest_rows: list[dict[str, str]] = []
    manifest_lock = threading.Lock()

    totals: dict[str, int] = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        future_to_class = {
            executor.submit(
                download_class,
                class_name,
                output_root,
                api_key=api_key,
                limit=args.limit,
                search_limit=args.search_limit,
                size=args.size,
                skip_existing=args.skip_existing,
                pages_per_class=args.pages_per_class,
                num_per_page=args.num_per_page,
                pause_seconds=args.pause_seconds,
                min_side=args.min_side,
                manifest_rows=manifest_rows,
                manifest_lock=manifest_lock,
            ): class_name
            for class_name in STL10_CLASSES
        }

        for future in cf.as_completed(future_to_class):
            class_name = future_to_class[future]
            totals[class_name] = future.result()

    print("\nSummary:")
    for class_name in STL10_CLASSES:
        print(f"  {class_name}: {totals.get(class_name, 0)}")

    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_name", "saved_name", "source_url", "source_query", "source_file"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest written to {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

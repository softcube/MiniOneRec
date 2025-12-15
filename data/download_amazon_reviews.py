#!/usr/bin/env python3
"""
Utility script to fetch the raw Amazon review dumps referenced in README §2.1.

Examples
--------
Download the Industrial and Office datasets from the 2018 release:
    python data/download_amazon_reviews.py \
        --dataset-version 2018 \
        --categories Industrial_and_Scientific Office_Products \
        --output-dir data/raw

Fetch the Amazon 2023 Industrial dump without downloading (print URLs only):
    python data/download_amazon_reviews.py \
        --dataset-version 2023 \
        --categories Industrial_and_Scientific \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Base endpoints documented on the official dataset pages
AMAZON_2014_BASE = "https://snap.stanford.edu/data/amazon/productGraph"
AMAZON_2018_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"
AMAZON_2023_BASE = "https://amazon-reviews-2023.s3.amazonaws.com"

DATASET_VERSIONS = ("2014", "2018", "2023")
DATA_TYPES = ("reviews", "metadata")


def human_size(num_bytes: int) -> str:
    """Return a human readable file size string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:0.1f}{unit}"
        size /= 1024.0
    return f"{size:0.1f}PB"


def build_candidate_urls(
    dataset_version: str, data_type: str, category: str, size: str
) -> List[str]:
    """Return download URL candidates for the requested dataset."""
    category = category.strip()
    urls: List[str] = []

    if dataset_version == "2014":
        if data_type == "reviews":
            urls.append(f"{AMAZON_2014_BASE}/categoryFiles/{category}.json.gz")
        else:
            urls.append(f"{AMAZON_2014_BASE}/metaFiles/meta_{category}.json.gz")
        return urls

    if dataset_version == "2018":
        reviews_dir = "categoryFiles" if size == "full" else "categoryFilesSmall"
        if data_type == "reviews":
            urls.append(f"{AMAZON_2018_BASE}/{reviews_dir}/{category}.json.gz")
            urls.append(f"{AMAZON_2018_BASE}/{reviews_dir}/{category}_5.json.gz")
        else:
            urls.append(f"{AMAZON_2018_BASE}/metaFiles2/meta_{category}.json.gz")
        return urls

    if dataset_version == "2023":
        suffix = ".jsonl.gz"
        if data_type == "reviews":
            urls.append(f"{AMAZON_2023_BASE}/{category}{suffix}")
        else:
            urls.append(f"{AMAZON_2023_BASE}/meta_{category}{suffix}")
        return urls

    raise ValueError(f"Unsupported dataset version: {dataset_version}")


def ensure_categories(categories: Iterable[str]) -> List[str]:
    cats = [c for c in (cat.strip() for cat in categories) if c]
    if not cats:
        raise ValueError("At least one category name is required.")
    return cats


def load_registry(overrides_path: Path | None) -> dict:
    if overrides_path is None:
        return {}
    try:
        with overrides_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse registry JSON: {exc}") from exc
    return data


def resolve_override(
    registry: dict, dataset_version: str, category: str, data_type: str
) -> str | None:
    try:
        return registry[dataset_version][category][data_type]
    except KeyError:
        return None


def iter_targets(
    dataset_version: str,
    categories: Iterable[str],
    data_types: Iterable[str],
    size: str,
    registry: dict,
):
    for category in ensure_categories(categories):
        for data_type in data_types:
            override = resolve_override(registry, dataset_version, category, data_type)
            if override:
                urls = [override]
            else:
                urls = build_candidate_urls(dataset_version, data_type, category, size)
            yield category, data_type, urls


def download(
    urls: List[str],
    destination_dir: Path,
    overwrite: bool,
    *,
    timeout: float,
    retries: int,
) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)

    for url in urls:
        destination = destination_dir / Path(url.split("?")[0]).name

        if destination.exists() and not overwrite:
            print(f"[SKIP] {destination} already exists.")
            return

        request = Request(url, headers={"User-Agent": "MiniOneRec-Downloader/1.0"})
        for attempt in range(1, retries + 1):
            print(f"[GET] {url} (attempt {attempt}/{retries})")
            try:
                with urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:
                    total = response.length or 0
                    downloaded = 0
                    start = time.time()

                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        elapsed = max(time.time() - start, 1e-6)
                        speed = downloaded / elapsed
                        speed_str = f"{human_size(int(speed))}/s"
                        if total:
                            percent = downloaded / total * 100.0
                            line = (
                                f"\r    {human_size(downloaded)} of {human_size(total)} "
                                f"({percent:0.1f}%) | {speed_str} | elapsed {elapsed:0.1f}s"
                            )
                        else:
                            line = (
                                f"\r    {human_size(downloaded)} downloaded "
                                f"| {speed_str} | elapsed {elapsed:0.1f}s"
                            )
                        sys.stdout.write(line)
                        sys.stdout.flush()

                    duration = time.time() - start
                    sys.stdout.write(
                        f"\r    Finished {human_size(downloaded)} in {duration:0.1f}s{' ' * 20}\n"
                    )
                    return
            except HTTPError as exc:
                if exc.code == 404:
                    print(f"[WARN] HTTP 404 for {url}. Trying next candidate...")
                    break
                if attempt == retries:
                    raise SystemExit(f"[ERROR] HTTP {exc.code} when fetching {url}") from exc
                print(f"[WARN] HTTP error ({exc.code}). Retrying...")
                time.sleep(2 * attempt)
            except URLError as exc:
                if attempt == retries:
                    raise SystemExit(f"[ERROR] Failed to reach {url}: {exc.reason}") from exc
                print(f"[WARN] Network error ({exc.reason}). Retrying...")
                time.sleep(2 * attempt)

    raise SystemExit(f"[ERROR] Exhausted all URL candidates for {destination_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Amazon review datasets (2014/2018/2023) from the official mirrors."
    )

    parser.add_argument(
        "--dataset-version",
        "-v",
        required=True,
        choices=DATASET_VERSIONS,
        help="Amazon dataset release to use (see README §2.1).",
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        required=True,
        help="Category identifiers exactly as spelled on the official download pages "
        "(e.g., Industrial_and_Scientific, Office_Products).",
    )
    parser.add_argument(
        "--data-types",
        "-t",
        nargs="+",
        choices=DATA_TYPES,
        default=DATA_TYPES,
        help="Which files to download for each category (default: both).",
    )
    parser.add_argument(
        "--size",
        choices=("small", "full"),
        default="small",
        help="Amazon 2018 exposes small/full review dumps. Ignored for other releases.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/raw"),
        help="Where to store the downloaded archives.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even when they already exist locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the URLs that would be downloaded and exit.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help=(
            "Optional JSON manifest with overrides. Format: "
            '{"2018": {"Some_Category": {"reviews": "https://...", "metadata": "https://..."}}}'
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Socket timeout per request in seconds (default: 3600).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts per file before aborting (default: 3).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    registry = load_registry(args.registry)
    categories = ensure_categories(args.categories)
    requested_types = args.data_types or DATA_TYPES

    targets = list(
        iter_targets(
            dataset_version=args.dataset_version,
            categories=categories,
            data_types=requested_types,
            size=args.size,
            registry=registry,
        )
    )

    if args.dry_run:
        print("Planned downloads:")
        for category, data_type, urls in targets:
            print(f"  - {args.dataset_version}/{category}/{data_type}:")
            for url in urls:
                print(f"       {url}")
        return

    for category, data_type, urls in targets:
        destination_dir = args.output_dir / args.dataset_version / category
        download(
            urls,
            destination_dir,
            overwrite=args.overwrite,
            timeout=args.timeout,
            retries=args.retries,
        )


if __name__ == "__main__":
    main()

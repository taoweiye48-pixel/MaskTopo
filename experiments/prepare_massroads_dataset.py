from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image


BASE_URL = "https://www.cs.toronto.edu/~vmnih/data/mass_roads"
EXPECTED_COUNTS = {"train": 1108, "valid": 14, "test": 49}
KINDS = {"sat": ".tiff", "map": ".tif"}


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and fingerprint official Massachusetts Roads files."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("real_data/MassachusettsRoads"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(EXPECTED_COUNTS),
        default=["train", "valid"],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_url(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "TopoBridge-research-audit/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def canonical_links(index_url: str, suffix: str, timeout: float) -> list[str]:
    payload = read_url(index_url, timeout)
    parser = LinkParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    links = []
    for href in parser.links:
        url = urllib.parse.urljoin(index_url, href)
        path = urllib.parse.urlparse(url).path
        if path.lower().endswith(suffix):
            links.append(url)
    return sorted(set(links), key=lambda value: Path(urllib.parse.urlparse(value).path).name)


def file_stem(url: str) -> str:
    return Path(urllib.parse.urlparse(url).path).stem


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {
            "width": int(image.width),
            "height": int(image.height),
            "mode": image.mode,
            "format": image.format,
        }


def download_one(
    url: str,
    destination: Path,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(1, retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "TopoBridge-research-audit/1.0"},
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    with temporary.open("wb") as output:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                os.replace(temporary, destination)
                break
            except Exception as error:  # pragma: no cover - network dependent
                last_error = error
                if temporary.exists():
                    temporary.unlink()
                if attempt < retries:
                    time.sleep(float(attempt))
        else:
            raise RuntimeError(f"Failed to download {url}: {last_error}")
    image_metadata = inspect_image(destination)
    return {
        "name": destination.name,
        "url": url,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        **image_metadata,
    }


def build_split_manifest(
    root: Path,
    split: str,
    workers: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    index_payloads: dict[str, bytes] = {}
    links_by_kind: dict[str, list[str]] = {}
    for kind, suffix in KINDS.items():
        index_url = f"{BASE_URL}/{split}/{kind}/index.html"
        payload = read_url(index_url, timeout)
        index_payloads[kind] = payload
        parser = LinkParser()
        parser.feed(payload.decode("utf-8", errors="replace"))
        links = []
        for href in parser.links:
            url = urllib.parse.urljoin(index_url, href)
            if urllib.parse.urlparse(url).path.lower().endswith(suffix):
                links.append(url)
        links_by_kind[kind] = sorted(
            set(links),
            key=lambda value: Path(urllib.parse.urlparse(value).path).name,
        )
        snapshot = root / "source_metadata" / f"{split}_{kind}_index.html"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(payload)

    sat_stems = [file_stem(url) for url in links_by_kind["sat"]]
    map_stems = [file_stem(url) for url in links_by_kind["map"]]
    expected = EXPECTED_COUNTS[split]
    if sat_stems != map_stems:
        raise RuntimeError(f"Official {split} image/target names do not pair exactly.")
    if len(sat_stems) != expected:
        raise RuntimeError(
            f"Official {split} count changed: expected {expected}, got {len(sat_stems)}."
        )

    jobs: list[tuple[str, Path, str]] = []
    for kind in KINDS:
        for url in links_by_kind[kind]:
            name = Path(urllib.parse.urlparse(url).path).name
            jobs.append((url, root / split / kind / name, kind))

    records: dict[str, list[dict[str, Any]]] = {kind: [] for kind in KINDS}
    completed = 0
    total_bytes = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, url, destination, timeout, retries): kind
            for url, destination, kind in jobs
        }
        for future in as_completed(futures):
            kind = futures[future]
            record = future.result()
            records[kind].append(record)
            completed += 1
            total_bytes += int(record["bytes"])
            if completed % 50 == 0 or completed == len(jobs):
                print(
                    "MASSROADS_DOWNLOAD "
                    f"split={split} files={completed}/{len(jobs)} "
                    f"gib={total_bytes / 2**30:.2f} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    for kind in records:
        records[kind].sort(key=lambda item: item["name"])
    dimensions_ok = all(
        item["width"] == 1500 and item["height"] == 1500
        for values in records.values()
        for item in values
    )
    manifest = {
        "dataset": "Massachusetts Roads Dataset",
        "official_source": "https://www.cs.toronto.edu/~vmnih/data/",
        "license_status": "RESEARCH_USE_CAUTION_NO_STANDARD_LICENSE_TEXT_FOUND",
        "split": split,
        "expected_pair_count": expected,
        "observed_pair_count": len(sat_stems),
        "paired_stems": sat_stems,
        "dimensions_1500x1500": dimensions_ok,
        "files": records,
        "total_bytes": sum(
            int(item["bytes"])
            for values in records.values()
            for item in values
        ),
    }
    manifest_path = root / "source_metadata" / f"{split}_download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def run_self_test() -> None:
    parser = LinkParser()
    parser.feed('<a href="one.tiff">a</a><a href="two.tif">b</a>')
    assert parser.links == ["one.tiff", "two.tif"]
    assert file_stem("https://example.test/a/one.tiff") == "one"
    print("MASSROADS_PREP_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must be between 1 and 32")
    if "test" in args.splits and set(args.splits) != {"test"}:
        raise ValueError("Download test in a separate explicit invocation.")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_page = read_url("https://www.cs.toronto.edu/~vmnih/data/", args.timeout)
    metadata_dir = root / "source_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "official_data_page.html").write_bytes(base_page)
    summaries = []
    for split in args.splits:
        summaries.append(
            build_split_manifest(
                root, split, args.workers, args.timeout, args.retries
            )
        )
    print(
        json.dumps(
            [
                {
                    "split": item["split"],
                    "pairs": item["observed_pair_count"],
                    "bytes": item["total_bytes"],
                    "dimensions_ok": item["dimensions_1500x1500"],
                }
                for item in summaries
            ],
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

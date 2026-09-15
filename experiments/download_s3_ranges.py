from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel HTTP range downloader.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranges(size: int, count: int) -> list[tuple[int, int]]:
    if size <= 0 or count <= 0:
        raise ValueError("size and count must be positive")
    width = (size + count - 1) // count
    return [
        (start, min(size - 1, start + width - 1))
        for start in range(0, size, width)
    ]


def download_one(
    url: str,
    part: Path,
    start: int,
    stop: int,
    timeout: float,
) -> dict[str, object]:
    expected = stop - start + 1
    if part.exists() and part.stat().st_size == expected:
        return {"part": str(part), "bytes": expected, "reused": True}
    if part.exists():
        raise RuntimeError(
            f"Refusing to overwrite wrong-sized part {part}: {part.stat().st_size} != {expected}"
        )
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{stop}",
            "User-Agent": "TopoBridge-SpaceNet3-audit/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 206:
            raise RuntimeError(
                f"Server did not honor range {start}-{stop}: HTTP {response.status}"
            )
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{stop}/"):
            raise RuntimeError(f"Unexpected Content-Range: {content_range}")
        payload = response.read()
    if len(payload) != expected:
        raise RuntimeError(
            f"Range {start}-{stop} returned {len(payload)} bytes, expected {expected}"
        )
    with part.open("xb") as handle:
        handle.write(payload)
    return {"part": str(part), "bytes": expected, "reused": False}


def main() -> None:
    args = parse_args()
    planned = ranges(args.size, args.chunks)
    if args.self_test:
        assert ranges(10, 3) == [(0, 3), (4, 7), (8, 9)]
        print("DOWNLOAD_S3_RANGES_SELF_TEST_PASS")
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    part_dir = output.parent / f"{output.name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    futures: list[concurrent.futures.Future[dict[str, object]]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(planned)
    ) as executor:
        for index, (start, stop) in enumerate(planned):
            part = part_dir / f"part_{index:04d}_{start}_{stop}.bin"
            futures.append(
                executor.submit(
                    download_one,
                    args.url,
                    part,
                    start,
                    stop,
                    args.timeout,
                )
            )
        results = []
        for index, future in enumerate(futures, start=1):
            result = future.result()
            results.append(result)
            print(
                f"RANGE_COMPLETE {index}/{len(futures)} bytes={result['bytes']} reused={result['reused']}",
                flush=True,
            )
    with output.open("xb") as destination:
        for index, (start, stop) in enumerate(planned):
            part = part_dir / f"part_{index:04d}_{start}_{stop}.bin"
            with part.open("rb") as source:
                while block := source.read(1024 * 1024):
                    destination.write(block)
    if output.stat().st_size != args.size:
        raise RuntimeError(
            f"Assembled size mismatch: {output.stat().st_size} != {args.size}"
        )
    duration = time.perf_counter() - start_time
    report = {
        "status": "completed",
        "url": args.url,
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "chunks": len(planned),
        "duration_seconds": duration,
        "throughput_mib_per_second": output.stat().st_size / 2**20 / duration,
        "pid": os.getpid(),
        "parts_preserved": str(part_dir),
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

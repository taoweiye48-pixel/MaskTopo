from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.request
from pathlib import Path


URL = "https://ndownloader.figshare.com/files/34969398"
EXPECTED_SIZE = 1_764_974_308
EXPECTED_MD5 = "789c80dd5376a82063e27fa49192bac9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable FIVES downloader.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("real_data/FIVES/download/FIVES_dataset.rar"),
    )
    return parser.parse_args()


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main() -> None:
    output = parse_args().output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    log_path = output.parent / "download_fives.log"
    downloaded = partial.stat().st_size if partial.exists() else 0
    if output.exists() and output.stat().st_size == EXPECTED_SIZE:
        digest = hashlib.md5(output.read_bytes()).hexdigest()
        if digest == EXPECTED_MD5:
            append_log(log_path, "DOWNLOAD_ALREADY_VERIFIED")
            return
    headers = {"User-Agent": "Mozilla/5.0"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    request = urllib.request.Request(URL, headers=headers)
    started = time.time()
    append_log(
        log_path,
        f"DOWNLOAD_START offset={downloaded} expected={EXPECTED_SIZE}",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        status = int(getattr(response, "status", 200))
        if downloaded and status != 206:
            downloaded = 0
            mode = "wb"
        else:
            mode = "ab" if downloaded else "wb"
        next_report = downloaded + 64 * 1024 * 1024
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    elapsed = max(time.time() - started, 1e-6)
                    rate = (downloaded - (0 if mode == "wb" else partial.stat().st_size)) / elapsed
                    append_log(
                        log_path,
                        f"DOWNLOAD_PROGRESS bytes={downloaded} "
                        f"percent={100 * downloaded / EXPECTED_SIZE:.2f}",
                    )
                    next_report += 64 * 1024 * 1024
            handle.flush()
            os.fsync(handle.fileno())
    if partial.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError(
            f"Size mismatch: {partial.stat().st_size} != {EXPECTED_SIZE}"
        )
    digest = hashlib.md5()
    with partial.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_MD5:
        raise RuntimeError(
            f"MD5 mismatch: {digest.hexdigest()} != {EXPECTED_MD5}"
        )
    partial.replace(output)
    append_log(
        log_path,
        f"DOWNLOAD_VERIFIED bytes={EXPECTED_SIZE} md5={EXPECTED_MD5}",
    )
    print(f"FIVES_DOWNLOAD_VERIFIED {output}", flush=True)


if __name__ == "__main__":
    main()

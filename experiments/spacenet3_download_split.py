from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


BUCKET = "https://spacenet-dataset.s3.amazonaws.com/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download frozen SpaceNet 3 split objects with verification.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "heldout_test"), required=True)
    parser.add_argument("--range-downloader", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--allow-heldout", action="store_true")
    parser.add_argument("--gate-file", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    expected_size = int(item["size"])
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"Size mismatch for {path}: {path.stat().st_size} != {expected_size}")
    etag = str(item.get("etag", "")).lower()
    md5 = hash_file(path, "md5")
    if etag and "-" not in etag and len(etag) == 32 and md5 != etag:
        raise RuntimeError(f"ETag/MD5 mismatch for {path}: {md5} != {etag}")
    return {"path": str(path.resolve()), "bytes": expected_size, "sha256": hash_file(path, "sha256"), "md5": md5, "etag": etag, "etag_md5_verified": bool(etag and "-" not in etag and len(etag) == 32)}


def object_url(key: str) -> str:
    return BUCKET + urllib.parse.quote(key, safe="/")


def download_label(item: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(["curl.exe", "--fail", "--location", "--silent", "--show-error", "--retry", "0", "--connect-timeout", "20", "--max-time", "120", "--remove-on-error", "--output", str(output), object_url(str(item["key"]))], check=False)
    if completed.returncode:
        raise RuntimeError(f"curl label download failed with exit code {completed.returncode}: {output}")


def download_image(item: dict[str, Any], output: Path, downloader: Path, chunks: int) -> None:
    command = [sys.executable, str(downloader), "--url", object_url(str(item["key"])), "--output", str(output), "--size", str(item["size"]), "--chunks", str(chunks), "--timeout", "120"]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"Range downloader failed with exit code {completed.returncode}: {output}")


def verify_heldout_authority(args: argparse.Namespace, manifest_hash: str) -> None:
    if "heldout_test" not in args.splits:
        return
    if not args.allow_heldout or args.gate_file is None:
        raise PermissionError("heldout_test requires --allow-heldout and --gate-file")
    gate = json.loads(args.gate_file.resolve().read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS" or gate.get("manifest_sha256") != manifest_hash:
        raise PermissionError("Held-out gate is not PASS for this exact frozen manifest")


def main() -> None:
    args = parse_args()
    if args.self_test:
        assert object_url("a/b c.tif") == "https://spacenet-dataset.s3.amazonaws.com/a/b%20c.tif"
        print("SPACENET3_DOWNLOAD_SPLIT_SELF_TEST_PASS")
        return
    manifest_path = args.manifest.resolve()
    manifest_hash = hash_file(manifest_path, "sha256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_heldout_content_access":
        raise RuntimeError("Manifest is not in the expected frozen state")
    verify_heldout_authority(args, manifest_hash)
    downloader = args.range_downloader.resolve()
    if not downloader.is_file():
        raise FileNotFoundError(downloader)
    data_root = args.data_root.resolve()
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    total = sum(len(manifest["splits"][split]) for split in args.splits)
    current = 0
    for split in args.splits:
        for pair in manifest["splits"][split]:
            current += 1
            source_id = str(pair["source_id"])
            image_path = data_root / split / "images" / f"{source_id}.tif"
            label_path = data_root / split / "labels" / f"{source_id}.geojson"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"PAIR_START {current}/{total} split={split} source={source_id}", flush=True)
            if image_path.exists():
                image_record = validate(image_path, pair["image"])
                print(f"IMAGE_REUSED source={source_id}", flush=True)
            else:
                download_image(pair["image"], image_path, downloader, args.chunks)
                image_record = validate(image_path, pair["image"])
            if label_path.exists():
                label_record = validate(label_path, pair["label"])
                print(f"LABEL_REUSED source={source_id}", flush=True)
            else:
                download_label(pair["label"], label_path)
                label_record = validate(label_path, pair["label"])
            record = {"split": split, "source_id": source_id, "image": image_record, "label": label_record}
            record_path = data_root / "download_records" / split / f"{source_id}.json"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            if record_path.exists():
                previous = json.loads(record_path.read_text(encoding="utf-8"))
                if previous != record:
                    raise RuntimeError(f"Existing download record differs: {record_path}")
            else:
                with record_path.open("x", encoding="utf-8", newline="\n") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
            records.append(record)
            print(f"PAIR_COMPLETE {current}/{total} split={split} source={source_id}", flush=True)
    report = {
        "experiment_id": manifest["experiment_id"],
        "status": "completed_verified",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "splits": args.splits,
        "pair_count": len(records),
        "image_bytes": sum(item["image"]["bytes"] for item in records),
        "label_bytes": sum(item["label"]["bytes"] for item in records),
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    suffix = "_".join(args.splits)
    output = data_root / f"download_manifest_{suffix}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite download manifest: {output}")
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": hash_file(output, "sha256"), "pair_count": len(records), "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

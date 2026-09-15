from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import struct
import sys
import urllib.request
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


EXPERIMENT_ID = "TB-B-260803-034"
AUTHOR_URL = "https://plantimages.nottingham.ac.uk/datasets/TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip"
EXPECTED_CONTENT_LENGTH = 2_227_029_019
EXPECTED_ETAG = '"60c75959-84bdc41b"'
ID_START = 3797
ID_END = 3916
EXCLUDED_DUPLICATE_ID = "3901"
SPLIT_SEED = 20260803
SPLIT_COUNTS = {"train": 90, "dev": 14, "test": 15}
LOCAL_SIGNATURE = b"PK\x03\x04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and fetch train/dev only for TB-B-260803-034.")
    parser.add_argument(
        "--remote-index",
        type=Path,
        default=Path("实验记录/TB-B-260803-033_author_archive_remote_index.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("ROOTNAV_BRASSICA_UNTOUCHED_PROTOCOL.md"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("实验记录/TB-B-260803-034_source_manifest.json"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/RootNav2B/train_dev_raw"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--stage", choices=("manifest", "extract-train-dev", "self-test"), required=True
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_digest(source_id: str) -> str:
    return hashlib.sha256(
        f"{EXPERIMENT_ID}|split_seed={SPLIT_SEED}|{source_id}".encode("utf-8")
    ).hexdigest()


def relevant_entries(index_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = {}
    for entry in index_payload["entries"]:
        parts = PurePosixPath(entry["name"]).parts
        if not parts or not parts[0].isdigit():
            continue
        numeric_id = int(parts[0])
        if ID_START <= numeric_id <= ID_END and not entry["is_directory"]:
            entries[entry["name"]] = entry
    return entries


def build_manifest(index_path: Path, protocol_path: Path) -> dict[str, Any]:
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    http = index_payload["http"]
    if index_payload["url"] != AUTHOR_URL:
        raise ValueError("Remote archive URL changed.")
    if int(http["content_length"]) != EXPECTED_CONTENT_LENGTH or http["etag"] != EXPECTED_ETAG:
        raise ValueError("Remote archive identity changed.")
    entries = relevant_entries(index_payload)
    sources: dict[str, dict[str, Any]] = {}
    image_groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for numeric_id in range(ID_START, ID_END + 1):
        source_id = f"{numeric_id:04d}"
        image_candidates = [
            entry
            for name, entry in entries.items()
            if PurePosixPath(name).parts[0] == source_id
            and PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".tif"}
        ]
        rsml_name = f"{source_id}/image_{source_id}.rsml"
        metadata_name = f"{source_id}/metadata.json"
        if len(image_candidates) != 1 or rsml_name not in entries or metadata_name not in entries:
            raise ValueError(f"Incomplete source {source_id}")
        image = image_candidates[0]
        image_groups[(image["crc32"], int(image["uncompressed_size"]))].append(source_id)
        sources[source_id] = {
            "source_id": source_id,
            "image": image,
            "rsml": entries[rsml_name],
            "metadata": entries[metadata_name],
        }
    duplicates = [sorted(ids) for ids in image_groups.values() if len(ids) > 1]
    if duplicates != [["3797", "3901"]]:
        raise ValueError(f"Unexpected image duplicate groups: {duplicates}")
    eligible = sorted(source_id for source_id in sources if source_id != EXCLUDED_DUPLICATE_ID)
    ranked = sorted(eligible, key=lambda source_id: (split_digest(source_id), source_id))
    split_ids = {
        "train": ranked[: SPLIT_COUNTS["train"]],
        "dev": ranked[SPLIT_COUNTS["train"] : SPLIT_COUNTS["train"] + SPLIT_COUNTS["dev"]],
        "test": ranked[-SPLIT_COUNTS["test"] :],
    }
    if {key: len(value) for key, value in split_ids.items()} != SPLIT_COUNTS:
        raise AssertionError("Split counts are wrong.")
    if len(set().union(*(set(value) for value in split_ids.values()))) != len(eligible):
        raise ValueError("Split is not a disjoint partition.")
    source_records = {}
    for split, ids in split_ids.items():
        source_records[split] = []
        for source_id in ids:
            record = sources[source_id]
            source_records[split].append(
                {
                    "source_id": source_id,
                    "split_digest": split_digest(source_id),
                    "image": record["image"],
                    "rsml": record["rsml"],
                    "metadata": record["metadata"],
                }
            )
    return {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "creation_access_scope": "author archive central-directory and species-only JSON metadata; no Brassica image or RSML content",
        "remote_archive": {
            "url": AUTHOR_URL,
            "content_length": EXPECTED_CONTENT_LENGTH,
            "etag": EXPECTED_ETAG,
            "last_modified": http["last_modified"],
            "remote_index_path": str(index_path.resolve()),
            "remote_index_sha256": sha256_file(index_path),
        },
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256_file(protocol_path),
        },
        "split_origin": "project_defined_hash_split_after_prepixel_crc_dedup",
        "split_seed": SPLIT_SEED,
        "reported_author_counts": {"train": 91, "dev": 14, "test": 15},
        "prepixel_duplicate_group": ["3797", "3901"],
        "duplicate_rule": "retain smallest numeric ID",
        "excluded_source_ids": [EXCLUDED_DUPLICATE_ID],
        "retained_source_count": len(eligible),
        "counts": {key: len(value) for key, value in source_records.items()},
        "test_image_member_request_count": 0,
        "test_rsml_member_request_count": 0,
        "test_inference_marker_exists": False,
        "sources": source_records,
    }


def write_manifest(index_path: Path, protocol_path: Path, manifest_path: Path) -> None:
    payload = build_manifest(index_path, protocol_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "manifest",
                "status": "PASS",
                "counts": payload["counts"],
                "excluded": payload["excluded_source_ids"],
                "protocol_sha256": payload["protocol"]["sha256"],
                "remote_index_sha256": payload["remote_archive"]["remote_index_sha256"],
                "manifest_sha256": sha256_file(manifest_path),
                "test_image_member_request_count": 0,
                "test_rsml_member_request_count": 0,
                "test_source_ids": [item["source_id"] for item in payload["sources"]["test"]],
            },
            indent=2,
        )
    )


def range_request(start: int, end: int) -> bytes:
    request = urllib.request.Request(
        AUTHOR_URL,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": f"TopoBridge-{EXPERIMENT_ID}/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
        if response.status != 206:
            raise ValueError(f"Range request returned HTTP {response.status}")
    expected = end - start + 1
    if len(data) != expected:
        raise ValueError(f"Range returned {len(data)} bytes, expected {expected}")
    return data


def decode_member(block: bytes, range_start: int, entry: dict[str, Any]) -> bytes:
    relative = int(entry["local_header_offset"]) - range_start
    if relative < 0 or relative + 30 > len(block):
        raise ValueError(f"Local header outside source range: {entry['name']}")
    fixed = block[relative : relative + 30]
    (
        signature,
        version_needed,
        flags,
        method,
        modified_time,
        modified_date,
        crc_local,
        compressed_local,
        uncompressed_local,
        filename_length,
        extra_length,
    ) = struct.unpack("<4s5H3L2H", fixed)
    if signature != LOCAL_SIGNATURE or flags & 0x1:
        raise ValueError(f"Unsupported local header: {entry['name']}")
    name_start = relative + 30
    name_raw = block[name_start : name_start + filename_length]
    encoding = "utf-8" if flags & 0x800 else "cp437"
    if name_raw.decode(encoding) != entry["name"]:
        raise ValueError(f"Local/central filename mismatch: {entry['name']}")
    data_start = name_start + filename_length + extra_length
    data_end = data_start + int(entry["compressed_size"])
    compressed = block[data_start:data_end]
    if len(compressed) != int(entry["compressed_size"]):
        raise ValueError(f"Truncated compressed member: {entry['name']}")
    if method == 0:
        raw = compressed
    elif method == 8:
        raw = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"Unsupported compression method {method}: {entry['name']}")
    if len(raw) != int(entry["uncompressed_size"]):
        raise ValueError(f"Uncompressed size mismatch: {entry['name']}")
    crc = f"{binascii.crc32(raw) & 0xFFFFFFFF:08x}"
    if crc != entry["crc32"]:
        raise ValueError(f"CRC mismatch: {entry['name']}")
    return raw


def fetch_source(record: dict[str, Any]) -> dict[str, Any]:
    image = record["image"]
    rsml = record["rsml"]
    metadata = record["metadata"]
    start = int(image["local_header_offset"])
    end = int(metadata["local_header_offset"]) - 1
    if not (start < int(rsml["local_header_offset"]) < int(metadata["local_header_offset"])):
        raise ValueError(f"Unexpected member order for source {record['source_id']}")
    block = range_request(start, end)
    return {
        "source_id": record["source_id"],
        "range_start": start,
        "range_end": end,
        "range_bytes": len(block),
        "image_entry": image,
        "image_bytes": decode_member(block, start, image),
        "rsml_entry": rsml,
        "rsml_bytes": decode_member(block, start, rsml),
    }


def safe_output(root: Path, member: str) -> Path:
    pure = PurePosixPath(member)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe member path: {member}")
    output = (root.resolve() / pure).resolve()
    if not output.is_relative_to(root.resolve()):
        raise ValueError(f"Member escapes destination: {member}")
    return output


def write_bytes_once(path: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    if path.exists():
        if sha256_file(path) != digest:
            raise FileExistsError(f"Existing file differs: {path}")
        return "existing_sha256_verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "fetched_crc_and_sha256_verified"


def crc32_file(path: Path) -> str:
    value = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value = binascii.crc32(chunk, value)
    return f"{value & 0xFFFFFFFF:08x}"


def existing_member_verified(path: Path, entry: dict[str, Any]) -> bool:
    return bool(
        path.exists()
        and path.stat().st_size == int(entry["uncompressed_size"])
        and crc32_file(path) == entry["crc32"]
    )


def completed_existing_record(
    record: dict[str, Any], image_path: Path, rsml_path: Path
) -> dict[str, Any]:
    return {
        "split": record["split"],
        "source_id": record["source_id"],
        "range_start": None,
        "range_end": None,
        "range_bytes": 0,
        "range_request_performed_this_run": False,
        "image": {
            "member": record["image"]["name"],
            "path": str(image_path),
            "bytes": image_path.stat().st_size,
            "sha256": sha256_file(image_path),
            "status": "existing_size_crc_sha256_verified",
        },
        "rsml": {
            "member": record["rsml"]["name"],
            "path": str(rsml_path),
            "bytes": rsml_path.stat().st_size,
            "sha256": sha256_file(rsml_path),
            "status": "existing_size_crc_sha256_verified",
        },
    }


def extract_train_dev(manifest_path: Path, protocol_path: Path, destination: Path, workers: int) -> None:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be in [1,8]")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("Wrong experiment manifest.")
    if payload["protocol"]["sha256"] != sha256_file(protocol_path):
        raise ValueError("Protocol changed after source-manifest freeze.")
    marker = manifest_path.parent / f"{EXPERIMENT_ID}_TEST_INFERENCE_STARTED.json"
    if marker.exists():
        raise ValueError("Test marker exists before train/dev extraction.")
    test_ids = {item["source_id"] for item in payload["sources"]["test"]}
    records = [
        {**item, "split": split}
        for split in ("train", "dev")
        for item in payload["sources"][split]
    ]
    if any(item["source_id"] in test_ids for item in records):
        raise ValueError("Test source entered train/dev request list.")
    destination.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for record in records:
        image_path = safe_output(destination, record["image"]["name"])
        rsml_path = safe_output(destination, record["rsml"]["name"])
        if existing_member_verified(image_path, record["image"]) and existing_member_verified(
            rsml_path, record["rsml"]
        ):
            completed.append(completed_existing_record(record, image_path, rsml_path))
        else:
            pending.append(record)
    print(
        f"TRAIN_DEV_RESUME existing_verified={len(completed)} pending={len(pending)} total={len(records)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_source, record): record for record in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            fetched = future.result()
            image_path = safe_output(destination, fetched["image_entry"]["name"])
            rsml_path = safe_output(destination, fetched["rsml_entry"]["name"])
            image_status = write_bytes_once(image_path, fetched.pop("image_bytes"))
            rsml_status = write_bytes_once(rsml_path, fetched.pop("rsml_bytes"))
            completed.append(
                {
                    "split": record["split"],
                    "source_id": record["source_id"],
                    "range_start": fetched["range_start"],
                    "range_end": fetched["range_end"],
                    "range_bytes": fetched["range_bytes"],
                    "range_request_performed_this_run": True,
                    "image": {
                        "member": fetched["image_entry"]["name"],
                        "path": str(image_path),
                        "bytes": image_path.stat().st_size,
                        "sha256": sha256_file(image_path),
                        "status": image_status,
                    },
                    "rsml": {
                        "member": fetched["rsml_entry"]["name"],
                        "path": str(rsml_path),
                        "bytes": rsml_path.stat().st_size,
                        "sha256": sha256_file(rsml_path),
                        "status": rsml_status,
                    },
                }
            )
            if index % 10 == 0 or index == len(pending):
                print(f"TRAIN_DEV_FETCH progress={index}/{len(pending)}", flush=True)
    completed.sort(key=lambda item: (item["split"], item["source_id"]))
    extraction_manifest = manifest_path.with_name(f"{EXPERIMENT_ID}_train_dev_extraction_manifest.json")
    output = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": sha256_file(protocol_path),
        "scope": "frozen train/dev image and RSML members only",
        "source_counts": {
            "train": sum(item["split"] == "train" for item in completed),
            "dev": sum(item["split"] == "dev" for item in completed),
        },
        "test_image_member_request_count": 0,
        "test_rsml_member_request_count": 0,
        "range_request_count_this_run": sum(
            item["range_request_performed_this_run"] for item in completed
        ),
        "requests": completed,
    }
    extraction_manifest.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "extract-train-dev",
                "status": "PASS",
                "source_counts": output["source_counts"],
                "range_request_count_this_run": output["range_request_count_this_run"],
                "test_image_member_request_count": 0,
                "test_rsml_member_request_count": 0,
                "manifest": str(extraction_manifest.resolve()),
                "manifest_sha256": sha256_file(extraction_manifest),
            },
            indent=2,
        )
    )


def self_test() -> None:
    source_id = "3797"
    assert len(split_digest(source_id)) == 64
    raw = b"brassica-self-test"
    name = b"3797/image_3797.jpg"
    crc = binascii.crc32(raw) & 0xFFFFFFFF
    header = struct.pack(
        "<4s5H3L2H", LOCAL_SIGNATURE, 20, 0, 0, 0, 0, crc, len(raw), len(raw), len(name), 0
    )
    block = header + name + raw
    entry = {
        "name": name.decode(),
        "local_header_offset": 100,
        "compression_method": 0,
        "compressed_size": len(raw),
        "uncompressed_size": len(raw),
        "crc32": f"{crc:08x}",
    }
    assert decode_member(block, 100, entry) == raw
    print(json.dumps({"stage": "self-test", "status": "PASS"}))


def main() -> None:
    args = parse_args()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "manifest":
        write_manifest(args.remote_index.resolve(), args.protocol.resolve(), args.manifest.resolve())
    else:
        extract_train_dev(
            args.manifest.resolve(), args.protocol.resolve(), args.destination.resolve(), args.workers
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


EXPERIMENT_ID = "TB-B-260803-033"
EXPECTED_ARCHIVE_SHA256 = (
    "23c6db0a2ffaf557cc8a069f45b7af6cba39d7a0cd7742dec7f7e79550a61fbd"
)
EXPECTED_COUNTS = {"train": 200, "val": 27, "test": 50}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Name-only manifest and sealed train/dev extraction for RootNav2A."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/RootNav2A/archives/RootNav2_Arabidopsis.zip"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("ROOTNAV_ARABIDOPSIS_UNTOUCHED_PROTOCOL.md"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("实验记录/TB-B-260803-033_source_manifest.json"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/RootNav2A/train_dev_raw"),
    )
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


def normalized_member(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe ZIP member: {name!r}")
    normalized = path.as_posix()
    if normalized != name.rstrip("/"):
        raise ValueError(f"Non-canonical ZIP member: {name!r}")
    return normalized


def paired_manifest(archive: Path, protocol: Path) -> dict[str, Any]:
    archive_sha = sha256_file(archive)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            f"Archive SHA mismatch: expected {EXPECTED_ARCHIVE_SHA256}, got {archive_sha}"
        )
    with zipfile.ZipFile(archive) as handle:
        infos = {normalized_member(info.filename): info for info in handle.infolist() if not info.is_dir()}
    pairs: dict[str, list[dict[str, Any]]] = {key: [] for key in EXPECTED_COUNTS}
    used_rsml: set[str] = set()
    for split in EXPECTED_COUNTS:
        image_members = sorted(
            name
            for name in infos
            if PurePosixPath(name).parent.as_posix() == split
            and PurePosixPath(name).suffix.lower() == ".tif"
        )
        if len(image_members) != EXPECTED_COUNTS[split]:
            raise ValueError(
                f"Expected {EXPECTED_COUNTS[split]} {split} TIFF files, found {len(image_members)}"
            )
        for image_member in image_members:
            source_id = PurePosixPath(image_member).stem
            rsml_member = f"RSML/{source_id}.rsml"
            if rsml_member not in infos:
                raise ValueError(f"Missing RSML pair for {image_member}: {rsml_member}")
            if rsml_member in used_rsml:
                raise ValueError(f"RSML reused across source pairs: {rsml_member}")
            used_rsml.add(rsml_member)
            image_info = infos[image_member]
            rsml_info = infos[rsml_member]
            pairs[split].append(
                {
                    "source_id": source_id,
                    "image_member": image_member,
                    "rsml_member": rsml_member,
                    "image_archive_bytes": image_info.file_size,
                    "image_crc32": f"{image_info.CRC:08x}",
                    "rsml_archive_bytes": rsml_info.file_size,
                    "rsml_crc32": f"{rsml_info.CRC:08x}",
                }
            )
    all_rsml = {
        name
        for name in infos
        if PurePosixPath(name).parent.as_posix() == "RSML"
        and PurePosixPath(name).suffix.lower() == ".rsml"
    }
    if used_rsml != all_rsml:
        missing = sorted(all_rsml - used_rsml)
        extra = sorted(used_rsml - all_rsml)
        raise ValueError(f"RSML pairing not bijective; unused={missing[:5]}, extra={extra[:5]}")
    source_ids = [item["source_id"] for split in pairs.values() for item in split]
    if len(source_ids) != 277 or len(set(source_ids)) != 277:
        raise ValueError("Source IDs are not globally unique across archive split directories.")
    return {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "creation_access_scope": "ZIP central-directory names and metadata only; no TIFF/RSML content opened",
        "dataset": "RootNav2A_Arabidopsis",
        "split_origin": "archive_provided_train_val_test_directories",
        "split_label": "author_archive_file_level_split",
        "archive": {
            "path": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "sha256": archive_sha,
            "transport_url": "https://download.cncb.ac.cn/OPIA/RootNav2_Arabidopsis.zip",
        },
        "protocol": {
            "path": str(protocol.resolve()),
            "sha256": sha256_file(protocol),
        },
        "counts": {key: len(value) for key, value in pairs.items()},
        "test_pixel_access_count": 0,
        "test_members_remain_archive_sealed": True,
        "pairs": pairs,
    }


def write_manifest(archive: Path, protocol: Path, manifest_path: Path) -> None:
    payload = paired_manifest(archive, protocol)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "stage": "manifest",
        "status": "PASS",
        "counts": payload["counts"],
        "archive_sha256": payload["archive"]["sha256"],
        "protocol_sha256": payload["protocol"]["sha256"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "test_pixel_access_count": 0,
    }, indent=2))


def safe_output_path(destination: Path, member: str) -> Path:
    normalized_member(member)
    root = destination.resolve()
    output = (root / PurePosixPath(member)).resolve()
    if not output.is_relative_to(root):
        raise ValueError(f"ZIP member escapes destination: {member}")
    return output


def extract_train_dev(archive: Path, manifest_path: Path, destination: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Wrong experiment manifest.")
    if payload["archive"]["sha256"] != sha256_file(archive):
        raise ValueError("Archive changed after manifest freeze.")
    if payload.get("test_pixel_access_count") != 0:
        raise ValueError("Manifest says test was already accessed.")
    members: list[str] = []
    for split in ("train", "val"):
        for pair in payload["pairs"][split]:
            members.extend((pair["image_member"], pair["rsml_member"]))
    if any(member.startswith("test/") or ",testset." in member for member in members):
        raise ValueError("Sealed test member entered train/dev extraction list.")
    if len(members) != 2 * (EXPECTED_COUNTS["train"] + EXPECTED_COUNTS["val"]):
        raise ValueError("Unexpected train/dev member count.")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as handle:
        for index, member in enumerate(members, start=1):
            info = handle.getinfo(member)
            output = safe_output_path(destination, member)
            if output.exists():
                if output.stat().st_size != info.file_size:
                    raise FileExistsError(f"Existing file has wrong size: {output}")
                status = "existing_size_verified"
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info, "r") as source, output.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                status = "extracted_crc_verified_by_zipfile"
            extracted.append(
                {
                    "member": member,
                    "output": str(output),
                    "bytes": output.stat().st_size,
                    "status": status,
                    "sha256": sha256_file(output),
                }
            )
            if index % 50 == 0 or index == len(members):
                print(f"EXTRACT progress={index}/{len(members)}", flush=True)
    extraction_manifest = manifest_path.with_name(
        "TB-B-260803-033_train_dev_extraction_manifest.json"
    )
    extraction_payload = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": payload["archive"]["sha256"],
        "source_manifest_sha256": sha256_file(manifest_path),
        "scope": "train and val TIFF/RSML only",
        "test_members_extracted": 0,
        "test_pixel_access_count": 0,
        "files": extracted,
    }
    extraction_manifest.write_text(
        json.dumps(extraction_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "extract-train-dev",
        "status": "PASS",
        "files": len(extracted),
        "test_members_extracted": 0,
        "extraction_manifest": str(extraction_manifest.resolve()),
        "extraction_manifest_sha256": sha256_file(extraction_manifest),
    }, indent=2))


def self_test() -> None:
    assert normalized_member("train/a.tif") == "train/a.tif"
    for invalid in ("../a", "/a", "train/../test/a.tif"):
        try:
            normalized_member(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe path accepted: {invalid}")
    print(json.dumps({"stage": "self-test", "status": "PASS"}))


def main() -> None:
    args = parse_args()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "manifest":
        write_manifest(args.archive.resolve(), args.protocol.resolve(), args.manifest.resolve())
    else:
        extract_train_dev(args.archive.resolve(), args.manifest.resolve(), args.destination.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

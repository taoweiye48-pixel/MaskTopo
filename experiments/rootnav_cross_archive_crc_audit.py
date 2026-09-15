from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


EXPERIMENT_ID = "TB-B-260803-033"
AMBIGUOUS_OPIA = {
    "val/RN2,2,41.tif",
    "val/RN2,2,42.tif",
    "val/RN2,3,32.tif",
    "val/RN2,4,39.tif",
    "val/RN2,7,29.tif",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-map OPIA and author RootNav archives by central-directory CRC metadata.")
    parser.add_argument("--opia-archive", type=Path)
    parser.add_argument("--author-index", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def key(crc32: str, size: int) -> tuple[str, int]:
    return crc32.lower(), int(size)


def audit(opia_archive: Path, author_index_path: Path) -> dict[str, Any]:
    author_payload = json.loads(author_index_path.read_text(encoding="utf-8"))
    author_images = [
        entry
        for entry in author_payload["entries"]
        if not entry["is_directory"] and PurePosixPath(entry["name"]).suffix.lower() in {".tif", ".jpg", ".jpeg"}
    ]
    author_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in author_images:
        author_by_key[key(entry["crc32"], entry["uncompressed_size"])].append(entry)

    with zipfile.ZipFile(opia_archive) as handle:
        opia_images = [
            info
            for info in handle.infolist()
            if not info.is_dir()
            and PurePosixPath(info.filename).parent.as_posix() in {"train", "val", "test"}
            and PurePosixPath(info.filename).suffix.lower() == ".tif"
        ]
    mappings: list[dict[str, Any]] = []
    opia_by_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    for info in sorted(opia_images, key=lambda item: item.filename):
        opia_by_key[key(f"{info.CRC:08x}", info.file_size)].append(info.filename)
        matches = author_by_key.get(key(f"{info.CRC:08x}", info.file_size), [])
        mappings.append(
            {
                "opia_member": info.filename,
                "opia_split": PurePosixPath(info.filename).parts[0],
                "opia_crc32": f"{info.CRC:08x}",
                "opia_uncompressed_size": info.file_size,
                "author_image_matches": [entry["name"] for entry in matches],
                "author_rsml_candidates": [
                    f"{PurePosixPath(entry['name']).parent.as_posix()}/image_{PurePosixPath(entry['name']).parent.name}.rsml"
                    for entry in matches
                ],
                "unique_match": len(matches) == 1,
                "was_opened_in_pairing_audit": info.filename in AMBIGUOUS_OPIA
                or (
                    info.filename.startswith("train/")
                    and info.filename
                    in {
                        f"train/RN2,{plate},{index}.tif"
                        for plate, index in []
                    }
                ),
            }
        )
    by_split: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        current = [item for item in mappings if item["opia_split"] == split]
        by_split[split] = {
            "total": len(current),
            "unique_author_crc_match": sum(item["unique_match"] for item in current),
            "no_author_crc_match": sum(not item["author_image_matches"] for item in current),
            "ambiguous_author_crc_match": sum(len(item["author_image_matches"]) > 1 for item in current),
        }
    ambiguous_rows = [item for item in mappings if item["opia_member"] in AMBIGUOUS_OPIA]
    duplicate_groups = []
    for (crc32, size), members in sorted(opia_by_key.items()):
        if len(members) <= 1:
            continue
        splits = sorted({PurePosixPath(member).parts[0] for member in members})
        duplicate_groups.append(
            {
                "crc32": crc32,
                "uncompressed_size": size,
                "members": sorted(members),
                "splits": splits,
                "cross_split": len(splits) > 1,
            }
        )
    repairable = bool(
        len(ambiguous_rows) == len(AMBIGUOUS_OPIA)
        and all(item["unique_match"] for item in ambiguous_rows)
        and len({item["author_rsml_candidates"][0] for item in ambiguous_rows}) == len(AMBIGUOUS_OPIA)
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "access_scope": "ZIP central-directory CRC32 and uncompressed-size metadata only; no member content opened",
        "test_pixel_access_count": 0,
        "opia_image_count": len(opia_images),
        "author_image_count": len(author_images),
        "counts_by_opia_split": by_split,
        "opia_duplicate_image_groups": duplicate_groups,
        "opia_duplicate_image_group_count": len(duplicate_groups),
        "opia_cross_split_duplicate_group_count": sum(
            group["cross_split"] for group in duplicate_groups
        ),
        "ambiguous_opia_rows": ambiguous_rows,
        "train_dev_repairable_from_author_archive_gate": repairable,
        "mappings": mappings,
    }


def self_test() -> None:
    lookup = defaultdict(list)
    lookup[key("00AAbbCC", 123)].append("x")
    assert lookup[key("00aabbcc", 123)] == ["x"]
    print(json.dumps({"stage": "self-test", "status": "PASS"}))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.opia_archive is None or args.author_index is None or args.output is None:
        raise SystemExit("--opia-archive, --author-index and --output are required")
    payload = audit(args.opia_archive.resolve(), args.author_index.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS" if payload["train_dev_repairable_from_author_archive_gate"] else "FAIL",
        "access_scope": payload["access_scope"],
        "counts_by_opia_split": payload["counts_by_opia_split"],
        "opia_duplicate_image_group_count": payload["opia_duplicate_image_group_count"],
        "opia_cross_split_duplicate_group_count": payload["opia_cross_split_duplicate_group_count"],
        "opia_duplicate_image_groups": payload["opia_duplicate_image_groups"],
        "ambiguous_opia_rows": payload["ambiguous_opia_rows"],
        "repairable_gate": payload["train_dev_repairable_from_author_archive_gate"],
        "test_pixel_access_count": 0,
        "output": str(args.output.resolve()),
    }, indent=2))
    if not payload["train_dev_repairable_from_author_archive_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

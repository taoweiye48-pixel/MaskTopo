from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a non-Zip64 remote ZIP using HTTP ranges.")
    parser.add_argument("--url")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--members-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def request(url: str, method: str = "GET", range_header: str | None = None) -> tuple[bytes, dict[str, str], int]:
    headers = {"User-Agent": "TopoBridge-TB-B-260803-033/1.0"}
    if range_header is not None:
        headers["Range"] = range_header
    req = urllib.request.Request(url, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read(), {key.lower(): value for key, value in response.headers.items()}, response.status


def parse_eocd(tail: bytes) -> dict[str, int]:
    offset = tail.rfind(EOCD_SIGNATURE)
    if offset < 0:
        raise ValueError("ZIP EOCD signature not found in remote tail.")
    if len(tail) - offset < 22:
        raise ValueError("Truncated EOCD.")
    signature, disk, cd_disk, entries_disk, entries_total, cd_size, cd_offset, comment_length = struct.unpack_from(
        "<4s4H2LH", tail, offset
    )
    if signature != EOCD_SIGNATURE:
        raise AssertionError("EOCD parser internal error.")
    if disk != 0 or cd_disk != 0 or entries_disk != entries_total:
        raise ValueError("Multi-disk ZIP archives are unsupported.")
    if entries_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        raise ValueError("Zip64 archive detected; explicit Zip64 parser required.")
    if offset + 22 + comment_length > len(tail):
        raise ValueError("EOCD comment extends past fetched tail.")
    return {
        "entries": entries_total,
        "central_directory_size": cd_size,
        "central_directory_offset": cd_offset,
        "comment_length": comment_length,
    }


def decode_name(raw: bytes, flags: int) -> str:
    encoding = "utf-8" if flags & 0x800 else "cp437"
    return raw.decode(encoding)


def parse_central_directory(data: bytes, expected_entries: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    offset = 0
    format_string = "<4s6H3L5H2L"
    fixed_size = struct.calcsize(format_string)
    while offset < len(data):
        if len(data) - offset < fixed_size:
            raise ValueError(f"Truncated central directory at byte {offset}.")
        values = struct.unpack_from(format_string, data, offset)
        (
            signature,
            version_made,
            version_needed,
            flags,
            compression_method,
            modified_time,
            modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
            comment_length,
            disk_start,
            internal_attributes,
            external_attributes,
            local_header_offset,
        ) = values
        if signature != CENTRAL_SIGNATURE:
            raise ValueError(f"Bad central-directory signature at byte {offset}.")
        cursor = offset + fixed_size
        filename_raw = data[cursor : cursor + filename_length]
        cursor += filename_length
        extra = data[cursor : cursor + extra_length]
        cursor += extra_length
        comment = data[cursor : cursor + comment_length]
        cursor += comment_length
        if cursor > len(data):
            raise ValueError("Central-directory variable fields are truncated.")
        if compressed_size == 0xFFFFFFFF or uncompressed_size == 0xFFFFFFFF or local_header_offset == 0xFFFFFFFF:
            raise ValueError("Zip64 entry detected; explicit Zip64 extra parsing required.")
        name = decode_name(filename_raw, flags)
        entries.append(
            {
                "name": name,
                "compression_method": compression_method,
                "flags": flags,
                "crc32": f"{crc32:08x}",
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_header_offset": local_header_offset,
                "is_directory": name.endswith("/"),
            }
        )
        offset = cursor
    if len(entries) != expected_entries:
        raise ValueError(f"Expected {expected_entries} entries, parsed {len(entries)}.")
    return entries


def index_remote_zip(url: str) -> dict[str, Any]:
    _, head_headers, head_status = request(url, method="HEAD")
    if head_status != 200:
        raise ValueError(f"Unexpected HEAD status: {head_status}")
    content_length = int(head_headers["content-length"])
    if "bytes" not in head_headers.get("accept-ranges", "").lower():
        raise ValueError("Server does not advertise byte ranges.")
    tail_size = min(content_length, 131072)
    tail_start = content_length - tail_size
    tail, tail_headers, tail_status = request(url, range_header=f"bytes={tail_start}-{content_length - 1}")
    if tail_status != 206 or len(tail) != tail_size:
        raise ValueError(f"Tail range failed: status={tail_status}, bytes={len(tail)}")
    eocd = parse_eocd(tail)
    central_start = eocd["central_directory_offset"]
    central_end = central_start + eocd["central_directory_size"] - 1
    if central_start >= tail_start and central_end < content_length:
        relative = central_start - tail_start
        central = tail[relative : relative + eocd["central_directory_size"]]
        central_fetch = "reused_tail_range"
    else:
        central, _, central_status = request(url, range_header=f"bytes={central_start}-{central_end}")
        if central_status != 206 or len(central) != eocd["central_directory_size"]:
            raise ValueError(
                f"Central-directory range failed: status={central_status}, bytes={len(central)}"
            )
        central_fetch = "dedicated_range"
    entries = parse_central_directory(central, eocd["entries"])
    files = [entry for entry in entries if not entry["is_directory"]]
    top_level = Counter(PurePosixPath(entry["name"]).parts[0] for entry in files)
    extensions = Counter(PurePosixPath(entry["name"]).suffix.lower() for entry in files)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "access_scope": "HTTP HEAD plus EOCD and ZIP central-directory byte ranges only",
        "http": {
            "content_length": content_length,
            "etag": head_headers.get("etag"),
            "last_modified": head_headers.get("last-modified"),
            "accept_ranges": head_headers.get("accept-ranges"),
            "tail_content_range": tail_headers.get("content-range"),
        },
        "eocd": eocd,
        "central_fetch": central_fetch,
        "file_count": len(files),
        "top_level_file_counts": dict(sorted(top_level.items())),
        "extension_file_counts": dict(sorted(extensions.items(), key=lambda item: (-item[1], item[0]))),
        "entries": entries,
    }


def self_test() -> None:
    eocd = struct.pack("<4s4H2LH", EOCD_SIGNATURE, 0, 0, 1, 1, 46, 123, 0)
    parsed = parse_eocd(b"prefix" + eocd)
    assert parsed["entries"] == 1
    assert parsed["central_directory_offset"] == 123
    print(json.dumps({"stage": "self-test", "status": "PASS"}))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.url or args.output is None or args.members_output is None:
        raise SystemExit("--url, --output and --members-output are required outside --self-test")
    payload = index_remote_zip(args.url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.members_output.parent.mkdir(parents=True, exist_ok=True)
    args.members_output.write_text(
        "\n".join(entry["name"] for entry in payload["entries"]) + "\n", encoding="utf-8"
    )
    summary = {key: value for key, value in payload.items() if key != "entries"}
    summary["output"] = str(args.output.resolve())
    summary["members_output"] = str(args.members_output.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

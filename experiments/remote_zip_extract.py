from __future__ import annotations

import argparse
import binascii
import json
import struct
import urllib.request
import zlib
from pathlib import Path, PurePosixPath
from typing import Any


LOCAL_SIGNATURE = b"PK\x03\x04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract selected metadata members from a remote ZIP by byte range.")
    parser.add_argument("--url")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--members", nargs="+")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def range_request(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "TopoBridge-TB-B-260803-033/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
        if response.status != 206:
            raise ValueError(f"Range request returned HTTP {response.status}")
    expected = end - start + 1
    if len(data) != expected:
        raise ValueError(f"Range returned {len(data)} bytes, expected {expected}")
    return data


def member_bytes(url: str, entry: dict[str, Any]) -> bytes:
    local_offset = int(entry["local_header_offset"])
    fixed = range_request(url, local_offset, local_offset + 29)
    (
        signature,
        version_needed,
        flags,
        method,
        modified_time,
        modified_date,
        crc32,
        compressed_size_local,
        uncompressed_size_local,
        filename_length,
        extra_length,
    ) = struct.unpack("<4s5H3L2H", fixed)
    if signature != LOCAL_SIGNATURE:
        raise ValueError(f"Bad local header for {entry['name']}")
    if flags & 0x1:
        raise ValueError(f"Encrypted ZIP member unsupported: {entry['name']}")
    if method != int(entry["compression_method"]):
        raise ValueError(f"Compression method mismatch for {entry['name']}")
    data_start = local_offset + 30 + filename_length + extra_length
    compressed_size = int(entry["compressed_size"])
    compressed = range_request(url, data_start, data_start + compressed_size - 1)
    if method == 0:
        raw = compressed
    elif method == 8:
        raw = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"Unsupported compression method {method} for {entry['name']}")
    if len(raw) != int(entry["uncompressed_size"]):
        raise ValueError(f"Uncompressed size mismatch for {entry['name']}")
    crc = f"{binascii.crc32(raw) & 0xFFFFFFFF:08x}"
    if crc != entry["crc32"]:
        raise ValueError(f"CRC mismatch for {entry['name']}: {crc} != {entry['crc32']}")
    return raw


def safe_output(destination: Path, member: str) -> Path:
    pure = PurePosixPath(member)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe member path: {member}")
    root = destination.resolve()
    output = (root / pure).resolve()
    if not output.is_relative_to(root):
        raise ValueError(f"Member escapes destination: {member}")
    return output


def self_test() -> None:
    raw = b"metadata-test" * 20
    compressor = zlib.compressobj(level=6, wbits=-zlib.MAX_WBITS)
    compressed = compressor.compress(raw) + compressor.flush()
    assert zlib.decompress(compressed, -zlib.MAX_WBITS) == raw
    assert f"{binascii.crc32(raw) & 0xFFFFFFFF:08x}" == f"{binascii.crc32(raw) & 0xFFFFFFFF:08x}"
    print(json.dumps({"stage": "self-test", "status": "PASS"}))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.url or args.index is None or args.destination is None or not args.members:
        raise SystemExit("--url, --index, --destination and --members are required")
    payload = json.loads(args.index.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in payload["entries"]}
    outputs = []
    for member in args.members:
        if PurePosixPath(member).suffix.lower() not in {".json", ".txt", ".md"}:
            raise ValueError(f"This metadata-only extractor refuses non-text member: {member}")
        if member not in entries:
            raise KeyError(f"Member not found in remote index: {member}")
        raw = member_bytes(args.url, entries[member])
        output = safe_output(args.destination, member)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
        outputs.append(
            {
                "member": member,
                "output": str(output),
                "bytes": len(raw),
                "crc32": entries[member]["crc32"],
            }
        )
    print(json.dumps({"status": "PASS", "scope": "metadata text only", "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()

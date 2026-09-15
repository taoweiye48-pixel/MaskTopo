from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


BUCKET = "https://spacenet-dataset.s3.amazonaws.com/"
SELECTION_SALT = "TB-B-260802-027"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze SpaceNet 3 object-only split manifest.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_prefix(prefix: str) -> tuple[bytes, list[dict[str, object]]]:
    query = urllib.parse.urlencode({"list-type": "2", "prefix": prefix})
    request = urllib.request.Request(f"{BUCKET}?{query}", headers={"User-Agent": "TopoBridge-SpaceNet3-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    root = ET.fromstring(payload)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    if root.findtext("s3:IsTruncated", namespaces=namespace) != "false":
        raise RuntimeError(f"Unexpected truncated object listing for {prefix}")
    objects: list[dict[str, object]] = []
    for node in root.findall("s3:Contents", namespace):
        objects.append({"key": node.findtext("s3:Key", namespaces=namespace), "size": int(node.findtext("s3:Size", namespaces=namespace) or "0"), "etag": (node.findtext("s3:ETag", namespaces=namespace) or "").strip('"'), "last_modified": node.findtext("s3:LastModified", namespaces=namespace)})
    return payload, objects


def paired(city: str, images: list[dict[str, object]], labels: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    pattern = re.compile(r"_img(\d+)\.(?:tif|geojson)$")
    indexed: dict[str, dict[int, dict[str, object]]] = {}
    for kind, entries in (("image", images), ("label", labels)):
        current: dict[int, dict[str, object]] = {}
        for entry in entries:
            match = pattern.search(str(entry["key"]))
            if match:
                current[int(match.group(1))] = entry
        indexed[kind] = current
    common = sorted(set(indexed["image"]) & set(indexed["label"]))
    if not common:
        raise RuntimeError(f"No paired objects found for {city}")
    return {image_id: {"source_id": f"{city}_img{image_id}", "image": indexed["image"][image_id], "label": indexed["label"][image_id]} for image_id in common}


def rank(city: str, item: dict[str, object]) -> str:
    image_id = str(item["source_id"]).split("img")[-1]
    return sha256_bytes(f"{SELECTION_SALT}|{city}|img{image_id}".encode("utf-8"))


def main() -> None:
    args = parse_args()
    if args.self_test:
        sample = {"source_id": "paris_img100"}
        assert rank("paris", sample) == sha256_bytes(b"TB-B-260802-027|paris|img100")
        print("SPACENET3_FREEZE_MANIFEST_SELF_TEST_PASS")
        return
    output = args.output.resolve()
    metadata_dir = args.metadata_dir.resolve()
    protocol = args.protocol.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {output}")
    if not protocol.is_file():
        raise FileNotFoundError(protocol)
    planned = {
        "paris_images": "spacenet/SN3_roads/train/AOI_3_Paris/PS-RGB/",
        "paris_labels": "spacenet/SN3_roads/train/AOI_3_Paris/geojson_roads/",
        "khartoum_images": "spacenet/SN3_roads/train/AOI_5_Khartoum/PS-RGB/",
        "khartoum_labels": "spacenet/SN3_roads/train/AOI_5_Khartoum/geojson_roads/",
    }
    listings: dict[str, list[dict[str, object]]] = {}
    raw: dict[str, bytes] = {}
    for name, prefix in planned.items():
        raw[name], listings[name] = list_prefix(prefix)
    paris = paired("paris", listings["paris_images"], listings["paris_labels"])
    khartoum = paired("khartoum", listings["khartoum_images"], listings["khartoum_labels"])
    paris_ranked = sorted(paris.values(), key=lambda item: (rank("paris", item), item["source_id"]))
    khartoum_ranked = sorted(khartoum.values(), key=lambda item: (rank("khartoum", item), item["source_id"]))
    if len(paris_ranked) < 128 or len(khartoum_ranked) < 64:
        raise RuntimeError("Insufficient paired objects for frozen split sizes")
    splits = {"train": paris_ranked[:96], "dev": paris_ranked[96:128], "heldout_test": khartoum_ranked[:64]}
    for split, entries in splits.items():
        city = "paris" if split != "heldout_test" else "khartoum"
        for item in entries:
            item["selection_rank_sha256"] = rank(city, item)
    manifest = {
        "experiment_id": SELECTION_SALT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_heldout_content_access",
        "selection": {"rule": "ascending SHA256(TB-B-260802-027|<city>|img<ID>), paired objects only", "train": "Paris ranks 1-96", "dev": "Paris ranks 97-128", "heldout_test": "Khartoum ranks 1-64", "content_or_label_statistics_used": False},
        "protocol": {"path": str(protocol), "sha256": sha256_file(protocol)},
        "source_listing": {name: {"prefix": planned[name], "object_count": len(listings[name]), "total_bytes": sum(int(item["size"]) for item in listings[name]), "xml_sha256": sha256_bytes(raw[name])} for name in planned},
        "paired_counts": {"paris": len(paris), "khartoum": len(khartoum)},
        "splits": splits,
        "heldout_content_read_before_freeze": False,
        "massachusetts_test_reopened": False,
        "p1_5_disease_classification_used": False,
    }
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in raw.items():
        path = metadata_dir / f"{name}_listing.xml"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite listing snapshot: {path}")
        with path.open("xb") as handle:
            handle.write(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "frozen", "output": str(output), "sha256": sha256_file(output), "counts": {key: len(value) for key, value in splits.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


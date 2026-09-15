from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.windows import Window
from scipy.ndimage import label


EXPERIMENT_ID = "TB-B-260803-030"
OLD_EXPERIMENT_ID = "TB-B-260802-027"
OFFSETS = (0, 256, 512, 768, 1024)
CROP_SIZE = 256
SOURCE_SIZE = 1300
BUCKET = "https://spacenet-dataset.s3.amazonaws.com/"
S3_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_id(entry: Any) -> str:
    return str(entry if isinstance(entry, str) else entry["source_id"])


def selection_hash(item: dict[str, Any]) -> str:
    image_id = str(item["source_id"]).replace("paris_img", "")
    payload = f"{EXPERIMENT_ID}|paris|img{image_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_listing(path: Path) -> list[dict[str, Any]]:
    root = ET.fromstring(path.read_bytes())
    if root.findtext("s3:IsTruncated", namespaces=S3_NAMESPACE) != "false":
        raise RuntimeError(f"Frozen S3 listing is truncated: {path}")
    result: list[dict[str, Any]] = []
    for node in root.findall("s3:Contents", S3_NAMESPACE):
        result.append(
            {
                "key": node.findtext("s3:Key", namespaces=S3_NAMESPACE),
                "size": int(node.findtext("s3:Size", namespaces=S3_NAMESPACE) or "0"),
                "etag": (node.findtext("s3:ETag", namespaces=S3_NAMESPACE) or "").strip('"'),
                "last_modified": node.findtext("s3:LastModified", namespaces=S3_NAMESPACE),
            }
        )
    return result


def paired_paris(images: list[dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pattern = re.compile(r"_img(\d+)\.(?:tif|geojson)$")
    banks: dict[str, dict[int, dict[str, Any]]] = {"image": {}, "label": {}}
    for kind, entries in (("image", images), ("label", labels)):
        for entry in entries:
            match = pattern.search(str(entry["key"]))
            if match:
                banks[kind][int(match.group(1))] = entry
    common = sorted(set(banks["image"]) & set(banks["label"]))
    return {
        f"paris_img{number}": {
            "source_id": f"paris_img{number}",
            "image": banks["image"][number],
            "label": banks["label"][number],
        }
        for number in common
    }


def freeze_manifest(metadata_dir: Path, old_manifest_path: Path, output: Path, task_path: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {output}")
    old_hash = sha256_file(old_manifest_path)
    if old_hash != "8e614a8bb520101da3dcc883cc6aa509760f21c45f721e96de89620ba8db6e80":
        raise RuntimeError(f"Old split manifest hash mismatch: {old_hash}")
    old = json.loads(old_manifest_path.read_text(encoding="utf-8-sig"))
    image_listing = metadata_dir / "paris_images_listing.xml"
    label_listing = metadata_dir / "paris_labels_listing.xml"
    pairs = paired_paris(parse_listing(image_listing), parse_listing(label_listing))
    if len(pairs) != 257:
        raise RuntimeError(f"Expected 257 paired Paris tiles, found {len(pairs)}")

    train_ids = [source_id(item) for item in old["splits"]["train"]]
    old_dev_ids = [source_id(item) for item in old["splits"]["dev"]]
    heldout_ids = [source_id(item) for item in old["splits"]["heldout_test"]]
    unused = [item for sid, item in pairs.items() if sid not in set(train_ids) | set(old_dev_ids)]
    ranked = sorted(unused, key=lambda item: (selection_hash(item), item["source_id"]))
    if len(ranked) != 129:
        raise RuntimeError(f"Expected 129 unused Paris pairs, found {len(ranked)}")
    dev_v2 = ranked[:48]
    for item in dev_v2:
        item["selection_rank_sha256"] = selection_hash(item)

    train = old["splits"]["train"]
    heldout = old["splits"]["heldout_test"]
    dev_v2_ids = [source_id(item) for item in dev_v2]
    intersections = {
        "dev_v2_intersect_train": len(set(dev_v2_ids) & set(train_ids)),
        "dev_v2_intersect_old_dev32": len(set(dev_v2_ids) & set(old_dev_ids)),
        "dev_v2_intersect_heldout": len(set(dev_v2_ids) & set(heldout_ids)),
    }
    print(json.dumps(intersections, ensure_ascii=False, sort_keys=True), flush=True)
    if any(intersections.values()):
        raise AssertionError(f"Split isolation failed: {intersections}")

    listing_names = (
        "paris_images_listing.xml",
        "paris_labels_listing.xml",
        "khartoum_images_listing.xml",
        "khartoum_labels_listing.xml",
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_dev_v2_content_access",
        "selection": {
            "candidate_pool": "129 unused paired Paris tiles after frozen TB-B-260802-027 train96/dev32",
            "rule": "ascending SHA256(TB-B-260803-030|paris|img<ID>)",
            "dev_v2": "first 48 candidates",
            "content_or_label_statistics_used": False,
        },
        "task": {"path": str(task_path.resolve()), "sha256": sha256_file(task_path)},
        "parent_manifest": {"path": str(old_manifest_path.resolve()), "sha256": old_hash},
        "source_metadata": {
            name: {"path": str((metadata_dir / name).resolve()), "sha256": sha256_file(metadata_dir / name)}
            for name in listing_names
        },
        "splits": {"train": train, "dev_v2": dev_v2, "heldout_test": heldout},
        "old_dev32_source_ids": old_dev_ids,
        "counts": {"train": len(train), "dev_v2": len(dev_v2), "heldout_test": len(heldout), "unused_paris": len(ranked)},
        "intersections": intersections,
        "khartoum_content_accessed": False,
        "massachusetts_test_reopened": False,
        "p1_5_disease_classification_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    result = {"status": "frozen", "path": str(output.resolve()), "sha256": sha256_file(output), "counts": manifest["counts"], "intersections": intersections}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def object_url(key: str) -> str:
    return BUCKET + urllib.parse.quote(key, safe="/")


def validate_object(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = int(item["size"])
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"Size mismatch: {path} has {path.stat().st_size}, expected {expected_size}")
    etag = str(item.get("etag", "")).lower()
    actual_md5 = md5_file(path)
    etag_verified = bool(etag and "-" not in etag and len(etag) == 32)
    if etag_verified and actual_md5 != etag:
        raise RuntimeError(f"ETag/MD5 mismatch for {path}: {actual_md5} != {etag}")
    return {
        "path": str(path.resolve()),
        "bytes": expected_size,
        "sha256": sha256_file(path),
        "md5": actual_md5,
        "etag": etag,
        "etag_md5_verified": etag_verified,
    }


def download_dev_v2(manifest_path: Path, data_root: Path, range_downloader: Path, chunks: int) -> dict[str, Any]:
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_dev_v2_content_access":
        raise RuntimeError("dev_v2 download requires the frozen v2 manifest")
    if "dev_v2" not in manifest.get("splits", {}):
        raise RuntimeError("Frozen manifest has no dev_v2 split")
    output_manifest = data_root / "download_manifest_dev_v2.json"
    if output_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite download manifest: {output_manifest}")
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    entries = manifest["splits"]["dev_v2"]
    for index, item in enumerate(entries, start=1):
        sid = str(item["source_id"])
        image_path = data_root / "dev_v2" / "images" / f"{sid}.tif"
        label_path = data_root / "dev_v2" / "labels" / f"{sid}.geojson"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"DEV_V2_PAIR_START {index}/{len(entries)} source={sid}", flush=True)
        if not image_path.exists():
            command = [
                sys.executable,
                str(range_downloader.resolve()),
                "--url",
                object_url(str(item["image"]["key"])),
                "--output",
                str(image_path),
                "--size",
                str(item["image"]["size"]),
                "--chunks",
                str(chunks),
                "--timeout",
                "120",
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                raise RuntimeError(f"Image download failed without retry, exit={completed.returncode}, source={sid}")
        else:
            print(f"DEV_V2_IMAGE_REUSED source={sid}", flush=True)
        image_record = validate_object(image_path, item["image"])
        if not label_path.exists():
            command = [
                "curl.exe",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "0",
                "--connect-timeout",
                "20",
                "--max-time",
                "120",
                "--remove-on-error",
                "--output",
                str(label_path),
                object_url(str(item["label"]["key"])),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                raise RuntimeError(f"Label download failed without retry, exit={completed.returncode}, source={sid}")
        else:
            print(f"DEV_V2_LABEL_REUSED source={sid}", flush=True)
        label_record = validate_object(label_path, item["label"])
        records.append({"source_id": sid, "image": image_record, "label": label_record})
        print(f"DEV_V2_PAIR_COMPLETE {index}/{len(entries)} source={sid}", flush=True)
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_verified",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "split": "dev_v2",
        "pair_count": len(records),
        "image_bytes": sum(int(item["image"]["bytes"]) for item in records),
        "label_bytes": sum(int(item["label"]["bytes"]) for item in records),
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
        "khartoum_content_accessed": False,
    }
    data_root.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    summary = {"status": "completed_verified", "path": str(output_manifest.resolve()), "sha256": sha256_file(output_manifest), "pairs": len(records), "seconds": report["elapsed_seconds"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def native_mask(image_path: Path, label_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    geojson = json.loads(label_path.read_text(encoding="utf-8"))
    geometries = [feature["geometry"] for feature in geojson.get("features", []) if feature.get("geometry")]
    with rasterio.open(image_path) as src:
        if (src.count, src.height, src.width) != (3, SOURCE_SIZE, SOURCE_SIZE):
            raise RuntimeError(f"Unexpected raster profile for {image_path}: {(src.count, src.height, src.width)}")
        if src.crs is None:
            raise RuntimeError(f"Missing CRS: {image_path}")
        transformed = [transform_geom("EPSG:4326", src.crs.to_string(), geometry) for geometry in geometries]
        mask = (
            rasterize(
                ((geometry, 1) for geometry in transformed),
                out_shape=(src.height, src.width),
                transform=src.transform,
                all_touched=True,
                dtype="uint8",
            )
            if transformed
            else np.zeros((src.height, src.width), dtype=np.uint8)
        )
        metadata = {"shape": [src.count, src.height, src.width], "crs": src.crs.to_string(), "feature_count": len(geometries), "road_pixels": int(mask.sum())}
    return mask, metadata


def split_paths(split: str, sid: str, train_root: Path, dev_root: Path) -> tuple[Path, Path]:
    root = train_root if split == "train" else dev_root
    return root / split / "images" / f"{sid}.tif", root / split / "labels" / f"{sid}.geojson"


def patch_occupancy(mask: np.ndarray) -> np.ndarray:
    return mask.reshape(16, 16, 16, 16).max(axis=(1, 3))


def prepare_split(split: str, entries: list[dict[str, Any]], train_root: Path, dev_root: Path, output: Path, manifest_hash: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {output}")
    count = len(entries) * 25
    images = np.empty((count, 3, CROP_SIZE, CROP_SIZE), dtype=np.float16)
    masks = np.empty((count, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    source_ids = np.empty(count, dtype="U32")
    crop_rows = np.empty(count, dtype=np.int16)
    crop_columns = np.empty(count, dtype=np.int16)
    cursor = 0
    source_diagnostics: list[dict[str, Any]] = []
    component_counts: list[int] = []
    started = time.perf_counter()
    for source_index, entry in enumerate(entries, start=1):
        sid = str(entry["source_id"])
        image_path, label_path = split_paths(split, sid, train_root, dev_root)
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing pair for {split}/{sid}: {image_path}, {label_path}")
        road, raster_metadata = native_mask(image_path, label_path)
        source_start = cursor
        with rasterio.open(image_path) as src:
            for row in OFFSETS:
                for column in OFFSETS:
                    image = src.read(window=Window(column, row, CROP_SIZE, CROP_SIZE))
                    if image.shape != (3, CROP_SIZE, CROP_SIZE):
                        raise RuntimeError(f"Bad crop shape for {sid}@{row},{column}: {image.shape}")
                    images[cursor] = np.clip(image.astype(np.float32) / 2047.0, 0.0, 1.0).astype(np.float16)
                    masks[cursor] = road[row : row + CROP_SIZE, column : column + CROP_SIZE]
                    source_ids[cursor] = sid
                    crop_rows[cursor] = row
                    crop_columns[cursor] = column
                    cursor += 1
        current = masks[source_start:cursor]
        nonempty = current.sum(axis=(1, 2)) > 0
        current_components: list[int] = []
        for crop in current[nonempty]:
            count_components = int(label(crop.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))[1])
            current_components.append(count_components)
            component_counts.append(count_components)
        source_diagnostics.append(
            {
                "source_id": sid,
                **raster_metadata,
                "empty_crop_count": int((~nonempty).sum()),
                "nonempty_road_fraction": float(current[nonempty].mean()) if np.any(nonempty) else float("nan"),
                "patch_foreground_fraction": float(np.mean([patch_occupancy(crop).mean() for crop in current])),
                "nonempty_component_mean": float(np.mean(current_components)) if current_components else float("nan"),
            }
        )
        print(f"STAGE1_PREPARE split={split} progress={source_index}/{len(entries)} source={sid} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if cursor != count:
        raise AssertionError(f"Prepared {cursor} crops, expected {count}")
    empty = masks.sum(axis=(1, 2)) == 0
    nonempty = ~empty
    patch_fraction = float(np.mean([patch_occupancy(crop).mean() for crop in masks]))
    diagnostics = {
        "split": split,
        "source_count": len(entries),
        "crop_count": count,
        "crops_per_source": 25,
        "empty_crop_fraction": float(empty.mean()),
        "nonempty_road_pixel_fraction": float(masks[nonempty].mean()) if np.any(nonempty) else float("nan"),
        "nonempty_component_mean": float(np.mean(component_counts)) if component_counts else float("nan"),
        "nonempty_component_median": float(np.median(component_counts)) if component_counts else float("nan"),
        "patch_foreground_fraction": patch_fraction,
        "image_min": float(images.min()),
        "image_max": float(images.max()),
        "source_diagnostics": source_diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "format": "spacenet3_native_25_crops_v2",
        "manifest_sha256": manifest_hash,
        "normalization": "clip(PS-RGB/2047,0,1) stored float16; ImageNet normalization at model input",
        "crop_size": CROP_SIZE,
        "output_size": CROP_SIZE,
        "offsets": OFFSETS,
        "unused_source_border": 20,
        "label": "native rasterio all_touched centerline; no pooling",
        "diagnostics": diagnostics,
    }
    np.savez_compressed(
        output,
        images=images,
        masks=masks,
        source_ids=source_ids,
        crop_rows=crop_rows,
        crop_columns=crop_columns,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    diagnostics["cache_path"] = str(output.resolve())
    diagnostics["cache_sha256"] = sha256_file(output)
    print(json.dumps({key: diagnostics[key] for key in ("split", "source_count", "crop_count", "empty_crop_fraction", "nonempty_road_pixel_fraction", "nonempty_component_mean", "nonempty_component_median", "patch_foreground_fraction", "cache_sha256")}, ensure_ascii=False, indent=2), flush=True)
    return diagnostics


def validate_profile(diagnostics: dict[str, Any]) -> None:
    checks = {
        "empty_crop_fraction": (0.20, 0.75),
        "nonempty_road_pixel_fraction": (0.0025, 0.015),
        "patch_foreground_fraction": (0.02, 0.12),
        "nonempty_component_median": (1.0, 3.0),
    }
    failures = {
        key: {"value": float(diagnostics[key]), "allowed": bounds}
        for key, bounds in checks.items()
        if not bounds[0] <= float(diagnostics[key]) <= bounds[1]
    }
    if failures:
        raise RuntimeError(f"Frozen data-profile gate failed: {json.dumps(failures, ensure_ascii=False)}")


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    return json.loads(path.read_text(encoding="utf-8")), sha256_file(path)


def run_prepare(args: argparse.Namespace) -> None:
    manifest, manifest_hash = load_manifest(args.manifest.resolve())
    if manifest.get("status") != "frozen_before_dev_v2_content_access":
        raise RuntimeError("Unexpected manifest status")
    output_dir = args.output_dir.resolve()
    report: dict[str, Any] = {}
    for split in args.splits:
        diagnostics = prepare_split(
            split,
            manifest["splits"][split],
            args.train_root.resolve(),
            args.dev_v2_root.resolve(),
            output_dir / f"{split}.npz",
            manifest_hash,
        )
        validate_profile(diagnostics)
        report[split] = diagnostics
    report_path = output_dir / f"prepare_report_{'_'.join(args.splits)}.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump({"status": "completed_verified", "manifest_sha256": manifest_hash, "splits": report}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "completed_verified", "report": str(report_path.resolve()), "sha256": sha256_file(report_path)}, ensure_ascii=False, indent=2), flush=True)


def iter_native_crops(entries: Iterable[Any], root: Path, split: str) -> Iterable[np.ndarray]:
    for entry in entries:
        sid = source_id(entry)
        road, _ = native_mask(root / split / "images" / f"{sid}.tif", root / split / "labels" / f"{sid}.geojson")
        for row in OFFSETS:
            for column in OFFSETS:
                yield road[row : row + CROP_SIZE, column : column + CROP_SIZE]


def run_appendix(old_manifest_path: Path, old_data_root: Path) -> None:
    from spacenet3_metrics import raster_apls_proxy

    old = json.loads(old_manifest_path.read_text(encoding="utf-8-sig"))
    used = {source_id(item) for split in old["splits"].values() for item in split}
    paris_used = sorted(item for item in used if item.startswith("paris"))

    def old_rank(sid: str) -> str:
        number = sid.replace("paris_img", "")
        return hashlib.sha256(f"{OLD_EXPERIMENT_ID}|paris|img{number}".encode("utf-8")).hexdigest()

    ranked = sorted(paris_used, key=old_rank)
    train_ids = [source_id(item) for item in old["splits"]["train"]]
    dev_ids = [source_id(item) for item in old["splits"]["dev"]]
    assert ranked[:96] == sorted(train_ids, key=old_rank) and ranked[96:] == sorted(dev_ids, key=old_rank)
    print(f"APPENDIX_A1 rule_reproduces_manifest=True unused_paris={257-len(paris_used)}", flush=True)

    crops = [crop for crop in iter_native_crops(old["splits"]["dev"], old_data_root, "dev") if crop.any()]

    def blocky(mask: np.ndarray, block: int) -> np.ndarray:
        side = mask.shape[0] // block
        pooled = mask.reshape(side, block, side, block).max(axis=(1, 3))
        return np.repeat(np.repeat(pooled, block, axis=0), block, axis=1)

    oracle: dict[str, float] = {}
    for block in (64, 32, 16, 8):
        token_count = (256 // block) ** 2
        values = [float(raster_apls_proxy(crop, blocky(crop, block))["raster_apls_proxy"]) for crop in crops]
        value = float(np.mean([item for item in values if np.isfinite(item)]))
        oracle[f"K{token_count}"] = value
        print(f"APPENDIX_A3 K={token_count:<5d} block={block:>2d}px mean={value:.5f}", flush=True)

    gt = np.zeros((64, 64), np.uint8)
    gt[16, 5:30] = 1
    gt[48, 34:59] = 1
    false_link = gt.copy()
    false_link[16:49, 29:35] = 1
    metric = raster_apls_proxy(gt, false_link)
    forward = float(metric["gt_to_pred_path_score"])
    reverse = float(metric["pred_to_gt_path_score"])
    harmonic = 2 * forward * reverse / (forward + reverse) if forward + reverse else 0.0
    assert metric["path_recall"] == 1.0 and forward == 1.0 and reverse < 0.25 and harmonic < 0.5
    print(f"APPENDIX_A6 path_recall=1.0 forward={forward:.6f} reverse={reverse:.6f} bidirectional={harmonic:.6f}", flush=True)
    print(json.dumps({"status": "appendix_recalculated", "nonempty_dev32_crops": len(crops), "oracle": oracle}, ensure_ascii=False, indent=2), flush=True)


def run_self_test(train_root: Path, old_manifest: Path) -> None:
    image = np.arange(3 * CROP_SIZE * CROP_SIZE, dtype=np.uint16).reshape(3, CROP_SIZE, CROP_SIZE) % 2048
    normalized = np.clip(image.astype(np.float32) / 2047.0, 0, 1).astype(np.float16)
    assert normalized.shape == (3, CROP_SIZE, CROP_SIZE) and 0 <= float(normalized.min()) <= float(normalized.max()) <= 1
    mask = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    mask[3, 7] = 1
    occupancy = patch_occupancy(mask)
    assert occupancy.shape == (16, 16) and occupancy[0, 0] == 1 and occupancy.sum() == 1
    manifest = json.loads(old_manifest.read_text(encoding="utf-8-sig"))
    sid = source_id(manifest["splits"]["train"][0])
    road, metadata = native_mask(train_root / "train" / "images" / f"{sid}.tif", train_root / "train" / "labels" / f"{sid}.geojson")
    assert road.shape == (SOURCE_SIZE, SOURCE_SIZE) and metadata["road_pixels"] > 0
    windows = [(row, column) for row in OFFSETS for column in OFFSETS]
    assert len(windows) == 25 and windows[0] == (0, 0) and windows[-1] == (1024, 1024)
    assert max(row + CROP_SIZE for row, _ in windows) == 1280 and max(column + CROP_SIZE for _, column in windows) == 1280
    print(json.dumps({"status": "SPACENET3_STAGE1_PREPARE_SELF_TEST_PASS", "source": sid, "raster": metadata, "window_count": len(windows), "mosaic_shape": [1280, 1280], "unused_border": 20}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpaceNet 3 stage-1 frozen data preparation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-manifest")
    freeze.add_argument("--metadata-dir", type=Path, required=True)
    freeze.add_argument("--old-manifest", type=Path, required=True)
    freeze.add_argument("--task", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    download = subparsers.add_parser("download-dev-v2")
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--data-root", type=Path, required=True)
    download.add_argument("--range-downloader", type=Path, required=True)
    download.add_argument("--chunks", type=int, default=8)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--train-root", type=Path, required=True)
    prepare.add_argument("--dev-v2-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--splits", nargs="+", choices=("train", "dev_v2"), required=True)

    appendix = subparsers.add_parser("appendix")
    appendix.add_argument("--old-manifest", type=Path, required=True)
    appendix.add_argument("--old-data-root", type=Path, required=True)

    self_test = subparsers.add_parser("self-test")
    self_test.add_argument("--train-root", type=Path, required=True)
    self_test.add_argument("--old-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze-manifest":
        freeze_manifest(args.metadata_dir.resolve(), args.old_manifest.resolve(), args.output.resolve(), args.task.resolve())
    elif args.command == "download-dev-v2":
        download_dev_v2(args.manifest.resolve(), args.data_root.resolve(), args.range_downloader.resolve(), args.chunks)
    elif args.command == "prepare":
        run_prepare(args)
    elif args.command == "appendix":
        run_appendix(args.old_manifest.resolve(), args.old_data_root.resolve())
    elif args.command == "self-test":
        run_self_test(args.train_root.resolve(), args.old_manifest.resolve())
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

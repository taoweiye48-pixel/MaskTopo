from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.windows import Window


OFFSETS = (0, 256, 512, 768, 1024)
CROP = 256
OUTPUT_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen SpaceNet 3 raster crops.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "heldout_test"), required=True)
    parser.add_argument("--allow-heldout", action="store_true")
    parser.add_argument("--gate-file", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def downsample_image(image: np.ndarray) -> np.ndarray:
    if image.shape != (3, CROP, CROP):
        raise ValueError(f"Unexpected image crop shape: {image.shape}")
    pooled = image.astype(np.float32).reshape(3, OUTPUT_SIZE, 4, OUTPUT_SIZE, 4).mean(axis=(2, 4))
    return np.clip(pooled / 2047.0, 0.0, 1.0).astype(np.float16)


def downsample_mask(mask: np.ndarray) -> np.ndarray:
    if mask.shape != (CROP, CROP):
        raise ValueError(f"Unexpected mask crop shape: {mask.shape}")
    return mask.reshape(OUTPUT_SIZE, 4, OUTPUT_SIZE, 4).max(axis=(1, 3)).astype(np.uint8)


def verify_heldout(args: argparse.Namespace, manifest_hash: str) -> None:
    if "heldout_test" not in args.splits:
        return
    if not args.allow_heldout or args.gate_file is None:
        raise PermissionError("heldout_test preparation requires --allow-heldout and --gate-file")
    gate = json.loads(args.gate_file.resolve().read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS" or gate.get("manifest_sha256") != manifest_hash:
        raise PermissionError("Held-out gate is not PASS for this exact frozen manifest")


def source_paths(data_root: Path, split: str, source_id: str) -> tuple[Path, Path]:
    return data_root / split / "images" / f"{source_id}.tif", data_root / split / "labels" / f"{source_id}.geojson"


def prepare_split(split: str, entries: list[dict[str, Any]], data_root: Path, output: Path, manifest_hash: str, download_hash: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite crop cache: {output}")
    count = len(entries) * len(OFFSETS) ** 2
    images = np.empty((count, 3, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float16)
    masks = np.empty((count, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
    source_ids = np.empty(count, dtype="U32")
    crop_rows = np.empty(count, dtype=np.int16)
    crop_columns = np.empty(count, dtype=np.int16)
    cursor = 0
    source_diagnostics: list[dict[str, Any]] = []
    for source_index, entry in enumerate(entries, start=1):
        source_id = str(entry["source_id"])
        image_path, label_path = source_paths(data_root, split, source_id)
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing frozen pair for {source_id}")
        geojson = json.loads(label_path.read_text(encoding="utf-8"))
        with rasterio.open(image_path) as src:
            if (src.count, src.height, src.width) != (3, 1300, 1300):
                raise RuntimeError(f"Unexpected raster profile for {source_id}: {(src.count, src.height, src.width)}")
            geometries = [transform_geom("EPSG:4326", src.crs, feature["geometry"]) for feature in geojson.get("features", [])]
            road = rasterize(((geometry, 1) for geometry in geometries), out_shape=(src.height, src.width), transform=src.transform, all_touched=True, dtype="uint8") if geometries else np.zeros((src.height, src.width), dtype=np.uint8)
            source_start = cursor
            for row in OFFSETS:
                for column in OFFSETS:
                    window = Window(column, row, CROP, CROP)
                    images[cursor] = downsample_image(src.read(window=window))
                    masks[cursor] = downsample_mask(road[row : row + CROP, column : column + CROP])
                    source_ids[cursor] = source_id
                    crop_rows[cursor] = row
                    crop_columns[cursor] = column
                    cursor += 1
            current = masks[source_start:cursor]
            source_diagnostics.append({"source_id": source_id, "feature_count": len(geojson.get("features", [])), "road_fraction_64": float(current.mean()), "empty_crop_count": int((current.sum(axis=(1, 2)) == 0).sum())})
        print(f"PREPARE_SOURCE split={split} progress={source_index}/{len(entries)} source={source_id}", flush=True)
    if cursor != count:
        raise AssertionError(f"Prepared {cursor} crops, expected {count}")
    diagnostics = {
        "split": split,
        "source_count": len(entries),
        "crop_count": count,
        "crops_per_source": len(OFFSETS) ** 2,
        "empty_crop_count": int((masks.sum(axis=(1, 2)) == 0).sum()),
        "empty_crop_fraction": float((masks.sum(axis=(1, 2)) == 0).mean()),
        "road_pixel_fraction": float(masks.mean()),
        "image_min": float(images.min()),
        "image_max": float(images.max()),
        "source_diagnostics": source_diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"experiment_id": "TB-B-260802-027", "format": "spacenet3_fixed_25_crops_v1", "manifest_sha256": manifest_hash, "download_manifest_sha256": download_hash, "normalization": "clip(PS-RGB/2047,0,1)", "crop": CROP, "output_size": OUTPUT_SIZE, "offsets": OFFSETS, "label": "all_touched centerline then 4x4 max pool", "diagnostics": diagnostics}
    np.savez_compressed(output, images=images, masks=masks, source_ids=source_ids, crop_rows=crop_rows, crop_columns=crop_columns, metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    diagnostics["cache_path"] = str(output.resolve())
    diagnostics["cache_sha256"] = sha256_file(output)
    return diagnostics


def main() -> None:
    args = parse_args()
    if args.self_test:
        image = np.arange(3 * CROP * CROP, dtype=np.uint16).reshape(3, CROP, CROP) % 2048
        mask = np.zeros((CROP, CROP), dtype=np.uint8)
        mask[3, 7] = 1
        assert downsample_image(image).shape == (3, 64, 64)
        pooled = downsample_mask(mask)
        assert pooled.shape == (64, 64) and pooled[0, 1] == 1 and pooled.sum() == 1
        print("SPACENET3_PREPARE_SELF_TEST_PASS")
        return
    manifest_path = args.manifest.resolve()
    download_path = args.download_manifest.resolve()
    manifest_hash = sha256_file(manifest_path)
    download_hash = sha256_file(download_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    download = json.loads(download_path.read_text(encoding="utf-8"))
    if download.get("status") != "completed_verified" or download.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Download manifest does not verify against frozen split manifest")
    if not set(args.splits).issubset(set(download.get("splits", []))):
        raise RuntimeError("Requested split is absent from verified download manifest")
    verify_heldout(args, manifest_hash)
    output_dir = args.output_dir.resolve()
    results = {}
    for split in args.splits:
        results[split] = prepare_split(split, manifest["splits"][split], args.data_root.resolve(), output_dir / f"{split}.npz", manifest_hash, download_hash)
    report_path = output_dir / f"prepare_report_{'_'.join(args.splits)}.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite preparation report: {report_path}")
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump({"status": "completed_verified", "manifest_sha256": manifest_hash, "download_manifest_sha256": download_hash, "splits": results}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "completed_verified", "report": str(report_path), "sha256": sha256_file(report_path), "splits": {key: {k: value[k] for k in ("source_count", "crop_count", "empty_crop_fraction", "road_pixel_fraction", "cache_sha256")} for key, value in results.items()}}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

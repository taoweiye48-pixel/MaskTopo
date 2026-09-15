from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree

import numpy as np
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from crackforest_real_gate import (
    balanced_metrics,
    generate_arrays,
    threshold_shortcut,
)
from deepcrack_external_gate import match_shortcuts
from fives_external_gate import array_invariants


EXPERIMENT_ID = "TB-B-260803-034"
FROZEN_COUNTS = {"train": 2400, "dev": 600}
FROZEN_SOURCE_COUNTS = {"train": 90, "dev": 14, "test": 15}
FROZEN_DATA_SEED = 20260810
RSML_WIDTH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TB-B-260803-034 Brassica train/dev data gate.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("实验记录/TB-B-260803-034_source_manifest.json"),
    )
    parser.add_argument(
        "--extraction-manifest",
        type=Path,
        default=Path("实验记录/TB-B-260803-034_train_dev_extraction_manifest.json"),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/RootNav2B/train_dev_raw")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("brassica_cache"))
    parser.add_argument(
        "--output", type=Path, default=Path("实验记录/TB-B-260803-034_data_gate")
    )
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--dev-size", type=int, default=600)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_rsml_bytes(raw: bytes, shape: tuple[int, int], width: int = RSML_WIDTH) -> tuple[np.ndarray, dict[str, Any]]:
    root = ElementTree.fromstring(raw)
    height, image_width = shape
    canvas = Image.new("L", (image_width, height), 0)
    draw = ImageDraw.Draw(canvas)
    polyline_count = 0
    point_count = 0
    minimum_x = float("inf")
    minimum_y = float("inf")
    maximum_x = float("-inf")
    maximum_y = float("-inf")
    for polyline in root.findall(".//geometry/polyline"):
        points: list[tuple[float, float]] = []
        for point in polyline.findall("point"):
            x = float(point.attrib["x"])
            y = float(point.attrib["y"])
            points.append((x, y))
            minimum_x = min(minimum_x, x)
            minimum_y = min(minimum_y, y)
            maximum_x = max(maximum_x, x)
            maximum_y = max(maximum_y, y)
        if len(points) >= 2:
            draw.line(points, fill=255, width=width, joint="curve")
            polyline_count += 1
            point_count += len(points)
    if polyline_count == 0:
        raise ValueError("RSML contains no renderable polylines.")
    if minimum_x < -0.5 or minimum_y < -0.5 or maximum_x > image_width - 0.5 or maximum_y > height - 0.5:
        raise ValueError(
            f"RSML coordinate bounds {(minimum_x, minimum_y, maximum_x, maximum_y)} "
            f"exceed image shape {shape}."
        )
    mask = np.asarray(canvas, dtype=np.uint8) > 0
    return mask, {
        "polyline_count": polyline_count,
        "point_count": point_count,
        "coordinate_bounds": [minimum_x, minimum_y, maximum_x, maximum_y],
        "foreground_pixels": int(mask.sum()),
        "foreground_fraction": float(mask.mean()),
    }


def load_source_manifest(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("Wrong source manifest experiment.")
    if payload["counts"] != FROZEN_SOURCE_COUNTS:
        raise ValueError(f"Frozen source counts changed: {payload['counts']}")
    if payload["test_image_member_request_count"] != 0 or payload["test_rsml_member_request_count"] != 0:
        raise ValueError("Source manifest reports test member access.")
    split_ids = {
        split: [item["source_id"] for item in payload["sources"][split]]
        for split in ("train", "dev", "test")
    }
    if any(set(split_ids[a]) & set(split_ids[b]) for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Source splits overlap.")
    return split_ids, payload


def load_train_dev_sources(
    dataset_root: Path, split_ids: dict[str, list[str]]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    source_diagnostics: dict[str, Any] = {}
    requested_ids = split_ids["train"] + split_ids["dev"]
    test_ids = set(split_ids["test"])
    if set(requested_ids) & test_ids:
        raise ValueError("Test source entered train/dev load list.")
    for index, source_id in enumerate(requested_ids, start=1):
        directory = dataset_root / source_id
        image_path = directory / f"image_{source_id}.jpg"
        rsml_path = directory / f"image_{source_id}.rsml"
        if not image_path.exists() or not rsml_path.exists():
            raise FileNotFoundError(f"Missing train/dev pair for {source_id}")
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = np.round(
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        ).astype(np.uint8)
        mask, diagnostic = render_rsml_bytes(rsml_path.read_bytes(), gray.shape, RSML_WIDTH)
        sources[source_id] = (gray, mask)
        source_diagnostics[source_id] = {
            "image_shape": list(gray.shape),
            "image_sha256": sha256_file(image_path),
            "rsml_sha256": sha256_file(rsml_path),
            **diagnostic,
        }
        if index % 10 == 0 or index == len(requested_ids):
            print(f"BRASSICA_SOURCE_LOAD progress={index}/{len(requested_ids)}", flush=True)
    return sources, source_diagnostics


def cache_fingerprint(source_manifest: Path, extraction_manifest: Path) -> str:
    return hashlib.sha256(
        (sha256_file(source_manifest) + sha256_file(extraction_manifest)).encode("ascii")
    ).hexdigest()[:16]


def load_or_generate_split(
    cache_dir: Path,
    split: str,
    sources: dict[str, tuple[np.ndarray, np.ndarray]],
    source_ids: list[str],
    count: int,
    seed: int,
    args: argparse.Namespace,
    fingerprint: str,
) -> tuple[dict[str, np.ndarray], Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (
        f"brassica_{split}_n{count}_seed{seed}_crop64_matchedv2_{fingerprint}.npz"
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        print(f"BRASSICA_CACHE_REUSE split={split} path={path}", flush=True)
        return arrays, path
    candidates = generate_arrays(
        sources,
        source_ids,
        count * args.candidate_multiplier,
        seed,
        args,
    )
    arrays = match_shortcuts(candidates, count, seed + 77_777)
    np.savez_compressed(path, **arrays)
    return arrays, path


def marker_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    features = np.empty((arrays["images"].shape[0], 14), dtype=np.float64)
    for index, image in enumerate(arrays["images"]):
        coordinates = []
        for channel in (1, 2):
            points = np.argwhere(image[channel] > 127)
            if points.size == 0:
                raise ValueError("Missing marker in marker-only audit.")
            coordinates.append(points.mean(axis=0) / 63.0)
        first, second = coordinates
        delta = second - first
        midpoint = 0.5 * (first + second)
        features[index] = np.asarray(
            [
                first[0],
                first[1],
                second[0],
                second[1],
                delta[0],
                delta[1],
                abs(delta[0]),
                abs(delta[1]),
                float(np.linalg.norm(delta)),
                midpoint[0],
                midpoint[1],
                first[0] * first[1],
                second[0] * second[1],
                delta[0] * delta[1],
            ]
        )
    return features


def marker_only_dev_balanced_accuracy(train: dict[str, np.ndarray], dev: dict[str, np.ndarray]) -> float:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=20260803,
        ),
    )
    model.fit(marker_features(train), train["connected"])
    prediction = model.predict(marker_features(dev))
    return float(balanced_accuracy_score(dev["connected"], prediction))


def direct_structural_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    first = arrays["endpoint_patches"][:, 0].astype(np.int64)
    second = arrays["endpoint_patches"][:, 1].astype(np.int64)
    rows = np.arange(arrays["images"].shape[0])
    first_component = arrays["patch_ids"][rows, first // 16, first % 16]
    second_component = arrays["patch_ids"][rows, second // 16, second % 16]
    prediction = ((first_component > 0) & (first_component == second_component)).astype(np.uint8)
    return balanced_metrics(arrays["connected"], prediction)


def self_test() -> None:
    raw = b"""<rsml><scene><plant><root><geometry><polyline>
    <point x='4' y='4'/><point x='20' y='20'/></polyline></geometry></root>
    </plant></scene></rsml>"""
    mask, diagnostic = render_rsml_bytes(raw, (32, 32), 8)
    assert mask.shape == (32, 32) and mask[12, 12] and int(mask.sum()) > 100
    assert diagnostic["polyline_count"] == 1
    print(json.dumps({"stage": "self-test", "status": "PASS"}))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if (
        args.train_size != FROZEN_COUNTS["train"]
        or args.dev_size != FROZEN_COUNTS["dev"]
        or args.candidate_multiplier != 3
        or args.crop_size != 64
        or args.output_size != 64
        or args.min_component_pixels != 12
        or args.data_seed != FROZEN_DATA_SEED
    ):
        raise ValueError("Command differs from the frozen TB-B-260803-034 data protocol.")
    source_manifest_path = args.source_manifest.resolve()
    extraction_manifest_path = args.extraction_manifest.resolve()
    extraction = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
    if extraction["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("Wrong extraction manifest experiment.")
    if extraction["source_counts"] != {"train": 90, "dev": 14}:
        raise ValueError("Train/dev extraction counts changed.")
    if extraction["test_image_member_request_count"] != 0 or extraction["test_rsml_member_request_count"] != 0:
        raise ValueError("Extraction manifest reports test access.")
    split_ids, source_payload = load_source_manifest(source_manifest_path)
    sources, source_diagnostics = load_train_dev_sources(args.dataset_root.resolve(), split_ids)
    fingerprint = cache_fingerprint(source_manifest_path, extraction_manifest_path)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    cache_paths: dict[str, Path] = {}
    for split, count, seed in (
        ("train", args.train_size, args.data_seed),
        ("dev", args.dev_size, args.data_seed + 1_000_000),
    ):
        arrays[split], cache_paths[split] = load_or_generate_split(
            args.cache_dir.resolve(),
            split,
            sources,
            split_ids[split],
            count,
            seed,
            args,
            fingerprint,
        )
    source_masks = {key: value[1] for key, value in sources.items()}
    invariants = array_invariants(arrays, {key: split_ids[key] for key in ("train", "dev")}, source_masks)
    endpoint_shortcut = threshold_shortcut(
        arrays["train"]["endpoint_distance"],
        arrays["train"]["connected"],
        arrays["dev"]["endpoint_distance"],
        arrays["dev"]["connected"],
    )
    train_intensity = arrays["train"]["images"][:, 0].mean(axis=(1, 2))
    dev_intensity = arrays["dev"]["images"][:, 0].mean(axis=(1, 2))
    intensity_shortcut = threshold_shortcut(
        train_intensity,
        arrays["train"]["connected"],
        dev_intensity,
        arrays["dev"]["connected"],
    )
    marker_shortcut = marker_only_dev_balanced_accuracy(arrays["train"], arrays["dev"])
    direct = {split: direct_structural_metrics(current) for split, current in arrays.items()}
    diagnostics = {
        "source_counts": {key: len(value) for key, value in split_ids.items()},
        "sample_counts": {split: int(current["images"].shape[0]) for split, current in arrays.items()},
        "positive_fraction": {split: float(current["connected"].mean()) for split, current in arrays.items()},
        "unique_sources_used": {
            split: int(np.unique(current["source_id"]).size) for split, current in arrays.items()
        },
        "endpoint_distance_only_dev_balanced_accuracy": endpoint_shortcut,
        "marker_coordinate_only_logistic_dev_balanced_accuracy": marker_shortcut,
        "mean_intensity_only_dev_balanced_accuracy": intensity_shortcut,
        "direct_ground_truth_structure_metrics": direct,
        "source_mask_foreground_fraction": {
            "minimum": min(item["foreground_fraction"] for item in source_diagnostics.values()),
            "median": float(np.median([item["foreground_fraction"] for item in source_diagnostics.values()])),
            "maximum": max(item["foreground_fraction"] for item in source_diagnostics.values()),
        },
    }
    gates = {
        "source_manifest_90_14_15_and_disjoint": bool(
            {key: len(value) for key, value in split_ids.items()} == FROZEN_SOURCE_COUNTS
        ),
        "test_member_access_zero": True,
        "exact_sample_counts": diagnostics["sample_counts"] == FROZEN_COUNTS,
        "all_splits_exactly_balanced": all(
            np.isclose(value, 0.5) for value in diagnostics["positive_fraction"].values()
        ),
        "all_array_invariants_pass": all(invariants.values()),
        "endpoint_distance_only_at_most_55pct": endpoint_shortcut <= 0.55,
        "marker_coordinate_only_at_most_55pct": marker_shortcut <= 0.55,
        "direct_ground_truth_structure_at_least_65pct": direct["dev"]["balanced_accuracy"] >= 0.65,
    }
    verdict = "TRAIN_DEV_DATA_GATE_PASS" if all(gates.values()) else "TRAIN_DEV_DATA_GATE_FAIL"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": verdict,
        "test_accessed": False,
        "test_image_member_request_count": 0,
        "test_rsml_member_request_count": 0,
        "dataset_fingerprint": fingerprint,
        "data_seed": args.data_seed,
        "rsml_render_width": RSML_WIDTH,
        "marker_only_model": "StandardScaler + class-balanced logistic regression C=1 on 14 coordinate-derived features",
        "diagnostics": diagnostics,
        "array_invariants": invariants,
        "gates": gates,
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": sha256_file(source_manifest_path),
        },
        "extraction_manifest": {
            "path": str(extraction_manifest_path),
            "sha256": sha256_file(extraction_manifest_path),
        },
        "cache_files": {
            split: {"path": str(path), "sha256": sha256_file(path)}
            for split, path in cache_paths.items()
        },
        "source_diagnostics": source_diagnostics,
    }
    report_path = output / "data_gate.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "verdict": verdict,
        "diagnostics": diagnostics,
        "gates": gates,
        "cache_files": report["cache_files"],
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "test_accessed": False,
    }, indent=2), flush=True)
    if verdict != "TRAIN_DEV_DATA_GATE_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

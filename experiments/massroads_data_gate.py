from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label

from crackforest_real_gate import (
    choose_pair,
    draw_marker,
    patch_component_ids,
    threshold_shortcut,
)
from edge_topocoarsen_connector import canonicalize_images


EXPECTED_COUNTS = {"train": 1108, "dev": 14}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen train/validation data gate for Massachusetts Roads."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("real_data/MassachusettsRoads"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("massroads_cache")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results_massroads_data_audit")
    )
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--dev-size", type=int, default=600)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--min-component-pixels", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=20260802)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def paired_stems(root: Path, disk_split: str, expected: int) -> list[str]:
    sat_stems = sorted(path.stem for path in (root / disk_split / "sat").glob("*.tiff"))
    map_stems = sorted(path.stem for path in (root / disk_split / "map").glob("*.tif"))
    if sat_stems != map_stems:
        raise RuntimeError(f"{disk_split} input/target names are not paired.")
    if len(sat_stems) != expected:
        raise RuntimeError(
            f"{disk_split} count mismatch: expected {expected}, got {len(sat_stems)}."
        )
    return sat_stems


def manifest_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for split in ("train", "valid"):
        path = root / "source_metadata" / f"{split}_download_manifest.json"
        payload = path.read_bytes()
        digest.update(split.encode("utf-8"))
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()[:16]


def load_sources(
    root: Path, logical_split: str, disk_split: str, stems: list[str]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], list[int]]:
    sources: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    observed_values: set[int] = set()
    started = time.time()
    for index, stem in enumerate(stems, start=1):
        sat_path = root / disk_split / "sat" / f"{stem}.tiff"
        map_path = root / disk_split / "map" / f"{stem}.tif"
        with Image.open(sat_path) as image:
            if image.size != (1500, 1500):
                raise RuntimeError(f"Unexpected input dimensions: {sat_path}")
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
        with Image.open(map_path) as target:
            if target.size != (1500, 1500):
                raise RuntimeError(f"Unexpected target dimensions: {map_path}")
            raw_mask = np.asarray(target.convert("L"), dtype=np.uint8)
        observed_values.update(int(value) for value in np.unique(raw_mask))
        source_id = f"{logical_split}:{stem}"
        mask = raw_mask >= 128
        foreground = np.argwhere(mask).astype(np.int16)
        sources[source_id] = (gray, mask, foreground)
        if index % 100 == 0 or index == len(stems):
            print(
                f"MASSROADS_LOAD split={logical_split} "
                f"sources={index}/{len(stems)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return sources, sorted(observed_values)


def make_sample(
    sources: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    source_ids: list[str],
    index: int,
    seed: int,
    crop_size: int,
    output_size: int,
    minimum_size: int,
) -> dict[str, Any]:
    if crop_size != 128 or output_size != 64:
        raise ValueError("Frozen Massachusetts task requires crop=128, output=64.")
    rng = np.random.default_rng(seed + index * 104729)
    desired_connected = index % 2
    target_distance = [0.20, 0.34, 0.48][(index // 2) % 3]
    structure = np.ones((3, 3), dtype=np.uint8)
    for _ in range(1600):
        source_id = source_ids[int(rng.integers(0, len(source_ids)))]
        gray, mask, foreground = sources[source_id]
        if foreground.size == 0:
            continue
        anchor = foreground[int(rng.integers(0, foreground.shape[0]))]
        jitter = rng.integers(-crop_size // 4, crop_size // 4 + 1, size=2)
        top = int(
            np.clip(
                anchor[0] - crop_size // 2 + jitter[0],
                0,
                gray.shape[0] - crop_size,
            )
        )
        left = int(
            np.clip(
                anchor[1] - crop_size // 2 + jitter[1],
                0,
                gray.shape[1] - crop_size,
            )
        )
        mask_crop = mask[top : top + crop_size, left : left + crop_size]
        component_map, count = label(mask_crop, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(component_map.reshape(-1))[1:]
        pair = choose_pair(
            component_map,
            sizes,
            desired_connected,
            target_distance,
            minimum_size,
            rng,
        )
        if pair is None:
            continue
        first, second = pair
        patch_block = crop_size // 16
        first_patch = (int(first[0] // patch_block), int(first[1] // patch_block))
        second_patch = (int(second[0] // patch_block), int(second[1] // patch_block))
        if first_patch == second_patch:
            continue
        patches = patch_component_ids(component_map)
        first_component = int(component_map[tuple(first)])
        second_component = int(component_map[tuple(second)])
        patches[first_patch] = first_component
        patches[second_patch] = second_component

        gray_crop = gray[top : top + crop_size, left : left + crop_size]
        resized = np.asarray(
            Image.fromarray(gray_crop).resize(
                (output_size, output_size), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        image = np.zeros((3, output_size, output_size), dtype=np.float32)
        image[0] = 1.0 - resized / 255.0
        endpoints_out = []
        for endpoint in (first, second):
            mapped = np.clip(
                np.floor(
                    (endpoint.astype(np.float32) + 0.5)
                    * output_size
                    / crop_size
                ),
                0,
                output_size - 1,
            ).astype(int)
            endpoints_out.append(mapped)
        draw_marker(image[1], int(endpoints_out[0][0]), int(endpoints_out[0][1]))
        draw_marker(image[2], int(endpoints_out[1][0]), int(endpoints_out[1][1]))
        distance = float(
            np.linalg.norm(first.astype(np.float32) - second)
            / (math.sqrt(2.0) * (crop_size - 1))
        )
        if not mask_crop[tuple(first)] or not mask_crop[tuple(second)]:
            raise RuntimeError("Endpoint does not overlap the target road mask.")
        return {
            "image": np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8),
            "connected": desired_connected,
            "patch_ids": patches,
            "endpoint_patches": np.asarray(
                [
                    first_patch[0] * 16 + first_patch[1],
                    second_patch[0] * 16 + second_patch[1],
                ],
                dtype=np.int16,
            ),
            "endpoint_native": np.asarray([first, second], dtype=np.int16),
            "endpoint_distance": distance,
            "road_fraction": float(mask_crop.mean()),
            "source_id": source_id,
            "crop_box": np.asarray(
                [top, left, top + crop_size, left + crop_size], dtype=np.int16
            ),
        }
    raise RuntimeError(
        f"Could not generate sample index={index}, label={desired_connected}."
    )


def generate_candidates(
    sources: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    source_ids: list[str],
    count: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    arrays = {
        "images": np.empty((count, 3, 64, 64), dtype=np.uint8),
        "connected": np.empty(count, dtype=np.int64),
        "patch_ids": np.empty((count, 16, 16), dtype=np.int16),
        "endpoint_patches": np.empty((count, 2), dtype=np.int16),
        "endpoint_native": np.empty((count, 2, 2), dtype=np.int16),
        "endpoint_distance": np.empty(count, dtype=np.float32),
        "road_fraction": np.empty(count, dtype=np.float32),
        "source_id": np.empty(count, dtype="<U64"),
        "crop_box": np.empty((count, 4), dtype=np.int16),
    }
    started = time.time()
    for index in range(count):
        sample = make_sample(
            sources,
            source_ids,
            index,
            seed,
            args.crop_size,
            args.output_size,
            args.min_component_pixels,
        )
        arrays["images"][index] = sample["image"]
        for key in arrays:
            if key != "images":
                arrays[key][index] = sample[key]
        if (index + 1) % 200 == 0 or index + 1 == count:
            print(
                f"MASSROADS_CANDIDATES {index + 1}/{count} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    arrays["images"] = canonicalize_images(arrays["images"])
    return arrays


def matching_edges(train: dict[str, np.ndarray]) -> dict[str, list[float]]:
    features = {
        "mean_luminance": train["images"][:, 0].mean(axis=(1, 2)),
        "endpoint_distance": train["endpoint_distance"].astype(np.float64),
        "road_fraction": train["road_fraction"].astype(np.float64),
    }
    bin_counts = {"mean_luminance": 4, "endpoint_distance": 3, "road_fraction": 3}
    output = {}
    for name, values in features.items():
        edges = np.unique(
            np.quantile(values, np.linspace(0, 1, bin_counts[name] + 1))
        )
        if edges.size != bin_counts[name] + 1:
            raise RuntimeError(
                f"Frozen bin grid collapsed for {name}: "
                f"expected {bin_counts[name] + 1} edges, got {edges.size}."
            )
        output[name] = [float(value) for value in edges]
    return output


def match_candidates(
    candidates: dict[str, np.ndarray],
    count: int,
    seed: int,
    edges: dict[str, list[float]],
) -> tuple[dict[str, np.ndarray], int]:
    if count % 2:
        raise ValueError("Final sample count must be even.")
    labels = candidates["connected"].astype(np.uint8)
    feature_values = {
        "mean_luminance": candidates["images"][:, 0].mean(axis=(1, 2)),
        "endpoint_distance": candidates["endpoint_distance"].astype(np.float64),
        "road_fraction": candidates["road_fraction"].astype(np.float64),
    }
    feature_bins = {
        name: np.digitize(values, np.asarray(edges[name])[1:-1], right=True)
        for name, values in feature_values.items()
    }
    rng = np.random.default_rng(seed)
    pair_bank: list[tuple[int, int]] = []
    for intensity_bin in range(len(edges["mean_luminance"]) - 1):
        for distance_bin in range(len(edges["endpoint_distance"]) - 1):
            for road_bin in range(len(edges["road_fraction"]) - 1):
                member = (
                    (feature_bins["mean_luminance"] == intensity_bin)
                    & (feature_bins["endpoint_distance"] == distance_bin)
                    & (feature_bins["road_fraction"] == road_bin)
                )
                negative = np.flatnonzero(member & (labels == 0))
                positive = np.flatnonzero(member & (labels == 1))
                rng.shuffle(negative)
                rng.shuffle(positive)
                matched = min(negative.size, positive.size)
                pair_bank.extend(
                    (int(negative[index]), int(positive[index]))
                    for index in range(matched)
                )
    if len(pair_bank) < count // 2:
        raise RuntimeError(
            f"Only {len(pair_bank)} fixed-grid matched pairs for {count // 2} required."
        )
    chosen = rng.choice(len(pair_bank), size=count // 2, replace=False)
    selected = np.asarray(
        [item for pair_index in chosen for item in pair_bank[int(pair_index)]],
        dtype=np.int64,
    )
    rng.shuffle(selected)
    output = {key: value[selected] for key, value in candidates.items()}
    if not np.isclose(output["connected"].mean(), 0.5):
        raise RuntimeError("Matching changed the frozen 50/50 balance.")
    return output, len(pair_bank)


def sample_invariants(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    first = arrays["endpoint_patches"][:, 0]
    second = arrays["endpoint_patches"][:, 1]
    flat = arrays["patch_ids"].reshape(arrays["patch_ids"].shape[0], -1)
    rows = np.arange(flat.shape[0])
    first_id = flat[rows, first]
    second_id = flat[rows, second]
    expected = arrays["connected"].astype(bool)
    observed = first_id == second_id
    return {
        "image_shape_ok": arrays["images"].shape[1:] == (3, 64, 64),
        "patch_shape_ok": arrays["patch_ids"].shape[1:] == (16, 16),
        "endpoint_patches_distinct": bool(np.all(first != second)),
        "endpoint_components_positive": bool(np.all((first_id > 0) & (second_id > 0))),
        "component_ids_match_labels": bool(np.all(observed == expected)),
        "finite_features": bool(
            np.isfinite(arrays["endpoint_distance"]).all()
            and np.isfinite(arrays["road_fraction"]).all()
        ),
    }


def render_examples(arrays: dict[str, np.ndarray], path: Path) -> None:
    indices = list(range(min(6, arrays["images"].shape[0])))
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    for axis, index in zip(axes.reshape(-1), indices):
        image = arrays["images"][index].astype(np.float32) / 255.0
        display = np.stack(
            (image[0], np.maximum(image[0], image[1]), np.maximum(image[0], image[2])),
            axis=-1,
        )
        axis.imshow(display)
        axis.set_title(
            f"label={int(arrays['connected'][index])}, "
            f"source={arrays['source_id'][index].split(':', 1)[1]}"
        )
        axis.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_self_test() -> None:
    count = 72
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), count // 2)
    rng = np.random.default_rng(123)
    candidates = {
        "images": rng.integers(0, 255, size=(count, 3, 64, 64), dtype=np.uint8),
        "connected": labels,
        "patch_ids": np.ones((count, 16, 16), dtype=np.int16),
        "endpoint_patches": np.tile(np.asarray([[0, 255]], dtype=np.int16), (count, 1)),
        "endpoint_native": np.zeros((count, 2, 2), dtype=np.int16),
        "endpoint_distance": np.tile(np.asarray([0.2, 0.2]), count // 2).astype(np.float32),
        "road_fraction": np.tile(np.asarray([0.1, 0.1]), count // 2).astype(np.float32),
        "source_id": np.asarray([f"train:{i}" for i in range(count)]),
        "crop_box": np.zeros((count, 4), dtype=np.int16),
    }
    edges = {
        "mean_luminance": [0.0, 255.0],
        "endpoint_distance": [0.0, 1.0],
        "road_fraction": [0.0, 1.0],
    }
    matched, available = match_candidates(candidates, 20, 456, edges)
    assert matched["images"].shape == (20, 3, 64, 64)
    assert np.isclose(matched["connected"].mean(), 0.5)
    assert available == count // 2
    print("MASSROADS_DATA_GATE_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if (
        args.crop_size != 128
        or args.output_size != 64
        or args.min_component_pixels != 32
        or args.candidate_multiplier != 3
    ):
        raise ValueError("Command differs from the frozen Massachusetts protocol.")
    root = args.dataset_root.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "train": paired_stems(root, "train", EXPECTED_COUNTS["train"]),
        "dev": paired_stems(root, "valid", EXPECTED_COUNTS["dev"]),
    }
    if set(manifests["train"]) & set(manifests["dev"]):
        raise RuntimeError("Official train/validation source identifiers overlap.")
    train_sources, train_values = load_sources(root, "train", "train", manifests["train"])
    dev_sources, dev_values = load_sources(root, "dev", "valid", manifests["dev"])
    if set(train_values) - {0, 255} or set(dev_values) - {0, 255}:
        raise RuntimeError(
            f"Official target maps are not binary: train={train_values}, dev={dev_values}."
        )
    fingerprint = manifest_fingerprint(root)

    candidate_specs = {
        "train": (
            train_sources,
            sorted(train_sources),
            args.train_size * args.candidate_multiplier,
            args.data_seed,
        ),
        "dev": (
            dev_sources,
            sorted(dev_sources),
            args.dev_size * args.candidate_multiplier,
            args.data_seed + 1_000_000,
        ),
    }
    candidates = {}
    for split, (sources, source_ids, count, seed) in candidate_specs.items():
        candidate_path = cache_dir / (
            f"massroads_{split}_candidates_n{count}_seed{seed}_{fingerprint}.npz"
        )
        if candidate_path.exists():
            with np.load(candidate_path) as archive:
                candidates[split] = {key: archive[key] for key in archive.files}
        else:
            candidates[split] = generate_candidates(
                sources, source_ids, count, seed, args
            )
            np.savez_compressed(candidate_path, **candidates[split])

    edges = matching_edges(candidates["train"])
    final = {}
    available_pairs = {}
    for split, count, seed in (
        ("train", args.train_size, args.data_seed + 77_777),
        ("dev", args.dev_size, args.data_seed + 1_077_777),
    ):
        final[split], available_pairs[split] = match_candidates(
            candidates[split], count, seed, edges
        )
        final_path = cache_dir / (
            f"massroads_{split}_n{count}_seed{seed}_matchedv1_{fingerprint}.npz"
        )
        np.savez_compressed(final_path, **final[split])

    intensity_train = final["train"]["images"][:, 0].mean(axis=(1, 2))
    intensity_dev = final["dev"]["images"][:, 0].mean(axis=(1, 2))
    shortcut_accuracy = {
        "endpoint_distance": threshold_shortcut(
            final["train"]["endpoint_distance"],
            final["train"]["connected"],
            final["dev"]["endpoint_distance"],
            final["dev"]["connected"],
        ),
        "mean_luminance": threshold_shortcut(
            intensity_train,
            final["train"]["connected"],
            intensity_dev,
            final["dev"]["connected"],
        ),
        "road_fraction": threshold_shortcut(
            final["train"]["road_fraction"],
            final["train"]["connected"],
            final["dev"]["road_fraction"],
            final["dev"]["connected"],
        ),
    }
    invariants = {split: sample_invariants(arrays) for split, arrays in final.items()}
    unique_sources = {
        split: int(np.unique(arrays["source_id"]).size)
        for split, arrays in final.items()
    }
    gates = {
        "official_train_count_and_pairing": len(manifests["train"]) == 1108,
        "official_validation_count_and_pairing": len(manifests["dev"]) == 14,
        "source_disjoint": not bool(set(manifests["train"]) & set(manifests["dev"])),
        "dimensions_and_binary_targets": train_values == [0, 255]
        and dev_values == [0, 255],
        "exact_class_balance": all(
            np.isclose(arrays["connected"].mean(), 0.5)
            for arrays in final.values()
        ),
        "source_coverage": unique_sources["train"] >= 800
        and unique_sources["dev"] == 14,
        "shortcut_gate": all(value <= 0.55 for value in shortcut_accuracy.values()),
        "sample_invariants": all(
            all(checks.values()) for checks in invariants.values()
        ),
        "projected_test_pair_feasibility": available_pairs["dev"] * 2 >= 600,
    }
    report = {
        "experiment_id": "TB-B-260802-021_massroads_train_dev_data_gate",
        "status": "completed",
        "protocol": str(
            (Path(__file__).resolve().parent / "MASSACHUSETTS_ROADS_EXTERNAL_CONFIRMATION_PROTOCOL.md")
        ),
        "test_accessed": False,
        "dataset_fingerprint": fingerprint,
        "official_source_counts": {"train": 1108, "validation": 14, "test": 49},
        "target_values": {"train": train_values, "dev": dev_values},
        "candidate_counts": {
            split: int(arrays["images"].shape[0]) for split, arrays in candidates.items()
        },
        "final_counts": {
            split: int(arrays["images"].shape[0]) for split, arrays in final.items()
        },
        "positive_fraction": {
            split: float(arrays["connected"].mean()) for split, arrays in final.items()
        },
        "unique_sources_used": unique_sources,
        "matching_edges_from_train": edges,
        "available_matched_pairs": available_pairs,
        "validation_shortcut_accuracy": shortcut_accuracy,
        "sample_invariants": invariants,
        "gates": gates,
        "verdict": "TRAIN_DEV_DATA_GATE_PASS"
        if all(gates.values())
        else "TRAIN_DEV_DATA_GATE_FAIL_STOP_BEFORE_TEST",
    }
    (output_dir / "train_dev_split_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "Massachusetts Roads Dataset",
                "license": "RESEARCH_USE_CAUTION_NO_STANDARD_LICENSE_TEXT_FOUND",
                "official_split": manifests,
                "dataset_fingerprint": fingerprint,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "data_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_examples(final["dev"], output_dir / "validation_examples.png")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

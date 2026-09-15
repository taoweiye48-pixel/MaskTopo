from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from brassica_data_gate import (
    FROZEN_DATA_SEED,
    FROZEN_SOURCE_COUNTS,
    array_invariants,
    cache_fingerprint,
    direct_structural_metrics,
    load_source_manifest,
    load_train_dev_sources,
    marker_features,
    marker_only_dev_balanced_accuracy,
    sha256_file,
)
from crackforest_real_gate import generate_arrays, threshold_shortcut


EXPERIMENT_ID = "TB-B-260803-035"
FROZEN_COUNTS = {"train": 2400, "dev": 600}
K_NEIGHBORS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TB035 marker-covariate matchedv3 train/dev gate.")
    parser.add_argument(
        "--source-manifest", type=Path, default=Path("实验记录/TB-B-260803-034_source_manifest.json")
    )
    parser.add_argument(
        "--extraction-manifest",
        type=Path,
        default=Path("实验记录/TB-B-260803-034_train_dev_extraction_manifest.json"),
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("ROOTNAV_BRASSICA_UNTOUCHED_PROTOCOL_V2.md")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/RootNav2B/train_dev_raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("brassica_cache_v3"))
    parser.add_argument(
        "--output", type=Path, default=Path("实验记录/TB-B-260803-035_data_gate")
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


def covariates(candidates: dict[str, np.ndarray]) -> np.ndarray:
    coordinate = marker_features(candidates)
    precise_distance = candidates["endpoint_distance"].astype(np.float64)[:, None]
    intensity = candidates["images"][:, 0].mean(axis=(1, 2), dtype=np.float64)[:, None]
    result = np.concatenate((coordinate, precise_distance, intensity), axis=1)
    if result.shape[1] != 16 or not np.isfinite(result).all():
        raise ValueError(f"Invalid matchedv3 covariates: {result.shape}")
    return result


def greedy_unique_match_indices(
    features: np.ndarray,
    labels: np.ndarray,
    pair_count: int,
    k_neighbors: int = K_NEIGHBORS,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = labels.astype(np.uint8)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if negative.size < pair_count or positive.size < pair_count:
        raise ValueError("Candidate pool is too small for balanced unique matching.")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (features - mean) / scale
    k = min(k_neighbors, positive.size)
    neighbors = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean", n_jobs=1)
    neighbors.fit(standardized[positive])
    distances, local_positive = neighbors.kneighbors(standardized[negative], return_distance=True)
    negative_edges = np.repeat(negative, k)
    positive_edges = positive[local_positive.reshape(-1)]
    distance_edges = distances.reshape(-1)
    order = np.lexsort((positive_edges, negative_edges, distance_edges))
    used_negative: set[int] = set()
    used_positive: set[int] = set()
    pairs: list[tuple[int, int]] = []
    accepted_distances: list[float] = []
    for edge_index in order:
        neg = int(negative_edges[edge_index])
        pos = int(positive_edges[edge_index])
        if neg in used_negative or pos in used_positive:
            continue
        used_negative.add(neg)
        used_positive.add(pos)
        pairs.append((neg, pos))
        accepted_distances.append(float(distance_edges[edge_index]))
        if len(pairs) == pair_count:
            break
    if len(pairs) != pair_count:
        raise RuntimeError(
            f"matchedv3 found only {len(pairs)} unique pairs; required {pair_count}."
        )
    selected = np.asarray([index for pair in pairs for index in pair], dtype=np.int64)
    accepted = np.asarray(accepted_distances, dtype=np.float64)
    return selected, {
        "feature_dimension": int(features.shape[1]),
        "candidate_count": int(features.shape[0]),
        "negative_candidates": int(negative.size),
        "positive_candidates": int(positive.size),
        "k_neighbors": int(k),
        "available_edge_count": int(distance_edges.size),
        "accepted_pair_count": int(len(pairs)),
        "standardized_pair_distance_min": float(accepted.min()),
        "standardized_pair_distance_median": float(np.median(accepted)),
        "standardized_pair_distance_p95": float(np.quantile(accepted, 0.95)),
        "standardized_pair_distance_max": float(accepted.max()),
    }


def match_covariates(
    candidates: dict[str, np.ndarray], count: int, shuffle_seed: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if count % 2:
        raise ValueError("Reportable sample count must be even.")
    selected, diagnostics = greedy_unique_match_indices(
        covariates(candidates), candidates["connected"], count // 2, K_NEIGHBORS
    )
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(selected)
    output = {key: value[selected] for key, value in candidates.items()}
    if output["images"].shape[0] != count or not np.isclose(output["connected"].mean(), 0.5):
        raise RuntimeError("matchedv3 violated exact count or balance.")
    output["matching_v3_k"] = np.full(count, K_NEIGHBORS, dtype=np.int16)
    output["matching_v3_feature_dimension"] = np.full(count, 16, dtype=np.int16)
    return output, diagnostics


def load_or_generate_candidates(
    cache_dir: Path,
    split: str,
    sources: dict[str, tuple[np.ndarray, np.ndarray]],
    source_ids: list[str],
    candidate_count: int,
    seed: int,
    args: argparse.Namespace,
    fingerprint: str,
) -> tuple[dict[str, np.ndarray], Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (
        f"brassica_{split}_candidates_n{candidate_count}_seed{seed}_crop64_{fingerprint}.npz"
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            current = {key: archive[key] for key in archive.files}
        print(f"MATCHEDV3_CANDIDATE_CACHE_REUSE split={split} path={path}", flush=True)
        return current, path
    current = generate_arrays(sources, source_ids, candidate_count, seed, args)
    np.savez_compressed(path, **current)
    print(f"MATCHEDV3_CANDIDATE_CACHE_WRITE split={split} path={path}", flush=True)
    return current, path


def self_test() -> None:
    negative = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    positive = negative + 0.01
    features = np.concatenate((negative, positive), axis=0)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    selected, diagnostics = greedy_unique_match_indices(features, labels, 4, 4)
    assert selected.size == 8 and np.unique(selected).size == 8
    assert diagnostics["accepted_pair_count"] == 4
    print(json.dumps({"stage": "self-test", "status": "PASS"}))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if (
        args.train_size != 2400
        or args.dev_size != 600
        or args.candidate_multiplier != 3
        or args.crop_size != 64
        or args.output_size != 64
        or args.min_component_pixels != 12
        or args.data_seed != FROZEN_DATA_SEED
    ):
        raise ValueError("Command differs from the frozen TB035 matchedv3 protocol.")
    source_manifest_path = args.source_manifest.resolve()
    extraction_manifest_path = args.extraction_manifest.resolve()
    protocol_path = args.protocol.resolve()
    if sha256_file(source_manifest_path) != "102ad36ce884045141cf5b71919667fa81a09e2d56829ebbc60022be0ae84de0":
        raise ValueError("Frozen source manifest changed.")
    extraction = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
    if extraction["test_image_member_request_count"] != 0 or extraction["test_rsml_member_request_count"] != 0:
        raise ValueError("Extraction manifest reports test access.")
    split_ids, _ = load_source_manifest(source_manifest_path)
    sources, _ = load_train_dev_sources(args.dataset_root.resolve(), split_ids)
    fingerprint = cache_fingerprint(source_manifest_path, extraction_manifest_path)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    candidate_paths: dict[str, Path] = {}
    matched_paths: dict[str, Path] = {}
    matching_diagnostics: dict[str, Any] = {}
    split_spec = (
        ("train", args.train_size, args.data_seed, args.data_seed + 77_777),
        ("dev", args.dev_size, args.data_seed + 1_000_000, args.data_seed + 1_077_777),
    )
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split, count, generation_seed, shuffle_seed in split_spec:
        candidates, candidate_paths[split] = load_or_generate_candidates(
            cache_dir,
            split,
            sources,
            split_ids[split],
            count * args.candidate_multiplier,
            generation_seed,
            args,
            fingerprint,
        )
        arrays[split], matching_diagnostics[split] = match_covariates(
            candidates, count, shuffle_seed
        )
        matched_path = cache_dir / (
            f"brassica_{split}_n{count}_seed{generation_seed}_crop64_matchedv3_{fingerprint}.npz"
        )
        np.savez_compressed(matched_path, **arrays[split])
        matched_paths[split] = matched_path
    source_masks = {key: value[1] for key, value in sources.items()}
    invariants = array_invariants(
        arrays, {key: split_ids[key] for key in ("train", "dev")}, source_masks
    )
    endpoint_shortcut = threshold_shortcut(
        arrays["train"]["endpoint_distance"], arrays["train"]["connected"],
        arrays["dev"]["endpoint_distance"], arrays["dev"]["connected"],
    )
    train_intensity = arrays["train"]["images"][:, 0].mean(axis=(1, 2))
    dev_intensity = arrays["dev"]["images"][:, 0].mean(axis=(1, 2))
    intensity_shortcut = threshold_shortcut(
        train_intensity, arrays["train"]["connected"],
        dev_intensity, arrays["dev"]["connected"],
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
        "matching": matching_diagnostics,
    }
    gates = {
        "source_manifest_90_14_15_and_disjoint": {key: len(value) for key, value in split_ids.items()} == FROZEN_SOURCE_COUNTS,
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
    verdict = "TRAIN_DEV_MATCHEDV3_GATE_PASS" if all(gates.values()) else "TRAIN_DEV_MATCHEDV3_GATE_FAIL"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": verdict,
        "test_accessed": False,
        "test_image_member_request_count": 0,
        "test_rsml_member_request_count": 0,
        "data_seed": args.data_seed,
        "matcher": "16-covariate standardized exact-brute kNN k=256 greedy unique pairs",
        "diagnostics": diagnostics,
        "array_invariants": invariants,
        "gates": gates,
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "source_manifest": {"path": str(source_manifest_path), "sha256": sha256_file(source_manifest_path)},
        "extraction_manifest": {"path": str(extraction_manifest_path), "sha256": sha256_file(extraction_manifest_path)},
        "candidate_caches": {
            split: {"path": str(path), "sha256": sha256_file(path)} for split, path in candidate_paths.items()
        },
        "matched_caches": {
            split: {"path": str(path), "sha256": sha256_file(path)} for split, path in matched_paths.items()
        },
    }
    report_path = output / "data_gate.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "verdict": verdict,
        "diagnostics": diagnostics,
        "gates": gates,
        "candidate_caches": report["candidate_caches"],
        "matched_caches": report["matched_caches"],
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "test_accessed": False,
    }, indent=2), flush=True)
    if verdict != "TRAIN_DEV_MATCHEDV3_GATE_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

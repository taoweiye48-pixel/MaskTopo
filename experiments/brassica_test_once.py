from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import platform
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path, PurePosixPath
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Any, Callable

import gudhi
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from brassica_train_dev import (
    evaluate_generic,
    forward_topology,
    render_rsml_bytes,
    sha256_file,
)
from crackforest_mask_topo import CrackUNet, predict_masks
from crackforest_mechanism_ablation import build_artifacts
from crackforest_real_gate import RealDataset, generate_arrays
from mask_aware_equal_supervision import (
    MODELS as MASK_AWARE_MODELS,
    forward_batch as forward_mask_aware,
    make_model as make_mask_aware_model,
)
from ph_token_baselines import (
    DESCRIPTOR_DIM,
    PHDescriptorDataset,
    aggregate_descriptor_diagnostics,
    forward_batch as forward_ph,
    make_model as make_ph_model,
    ph_descriptors,
)
from strong_reducer_baselines import ReducerDataset
from topocoarsen_oracle import TopoCoarsenModel


EXPERIMENT_ID = "TB-B-260803-035"
AUTHOR_URL = "https://plantimages.nottingham.ac.uk/datasets/TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip"
EXPECTED_CONTENT_LENGTH = 2_227_029_019
EXPECTED_ETAG = '"60c75959-84bdc41b"'
SOURCE_MANIFEST_SHA256 = "102ad36ce884045141cf5b71919667fa81a09e2d56829ebbc60022be0ae84de0"
FREEZE_MANIFEST_SHA256 = "9a3b1354969ebcdeada396ec7162d612a4b2ab21156e4d84e6d3dcb2fec40eb6"
SEEDS = (20260810, 20260811, 20260812)
SELECTED_MODELS = (
    "mask_topo_external",
    "mask_conditioned_perceiver_strong",
    "mask_conditioned_slot_param_matched",
    "ph_only",
)
TEST_IDS = (
    "3911", "3868", "3899", "3860", "3848", "3797", "3798", "3887",
    "3808", "3852", "3905", "3863", "3829", "3886", "3889",
)
LOCAL_SIGNATURE = b"PK\x03\x04"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260803


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TB035 preflight or one-shot sealed-test evaluation.")
    parser.add_argument("--stage", choices=("self-test", "preflight", "run-once"), required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("实验记录/TB-B-260803-035_train_dev_freeze_manifest.json"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("实验记录/TB-B-260803-034_source_manifest.json"),
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=Path("实验记录/TB-B-260803-035_test_entry_preflight.json"),
    )
    parser.add_argument(
        "--destination", type=Path, default=Path("data/RootNav2B/test_once_raw")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("实验记录/TB-B-260803-035_test_once")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def checked_file(path: Path, expected: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Hash mismatch for {path}: {actual} != {expected}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}


def walk_hashed_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            records.append(value)
        for child in value.values():
            records.extend(walk_hashed_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(walk_hashed_records(child))
    return records


def verify_frozen_state(root: Path, freeze_path: Path, source_path: Path) -> dict[str, Any]:
    checked_file(freeze_path, FREEZE_MANIFEST_SHA256)
    checked_file(source_path, SOURCE_MANIFEST_SHA256)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if freeze["experiment_id"] != EXPERIMENT_ID or freeze["status"] != "TRAIN_DEV_FROZEN_TEST_STILL_SEALED":
        raise ValueError("Invalid train/dev freeze manifest.")
    if freeze["test_accessed"] or freeze["test_marker_exists"]:
        raise ValueError("Freeze manifest reports test access.")
    if tuple(freeze["optimization_seeds"]) != SEEDS:
        raise ValueError("Frozen optimization seeds changed.")
    if tuple(freeze["selected_test_models"]) != SELECTED_MODELS or freeze["selected_best_ph"] != "ph_only":
        raise ValueError("Frozen test model identities changed.")
    if tuple(freeze["test_ids"]) != TEST_IDS:
        raise ValueError("Frozen test IDs changed in freeze manifest.")
    if source["test_image_member_request_count"] != 0 or source["test_rsml_member_request_count"] != 0:
        raise ValueError("Source manifest reports test member access.")
    source_ids = tuple(item["source_id"] for item in source["sources"]["test"])
    if source_ids != TEST_IDS:
        raise ValueError(f"Source manifest test ordering changed: {source_ids}")
    verified: dict[str, dict[str, Any]] = {}
    for record in walk_hashed_records(freeze):
        key = str(Path(record["path"]).resolve())
        if key not in verified:
            verified[key] = checked_file(Path(key), record["sha256"])
    return {"freeze": freeze, "source": source, "verified_file_count": len(verified)}


def marker_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    features = np.empty((arrays["images"].shape[0], 14), dtype=np.float64)
    for index, image in enumerate(arrays["images"]):
        coordinates = []
        for channel in (1, 2):
            points = np.argwhere(image[channel] > 127)
            if points.size == 0:
                raise ValueError("Missing marker in matchedv3 test candidate.")
            coordinates.append(points.mean(axis=0) / 63.0)
        first, second = coordinates
        delta = second - first
        midpoint = 0.5 * (first + second)
        features[index] = np.asarray(
            [
                first[0], first[1], second[0], second[1], delta[0], delta[1],
                abs(delta[0]), abs(delta[1]), float(np.linalg.norm(delta)),
                midpoint[0], midpoint[1], first[0] * first[1],
                second[0] * second[1], delta[0] * delta[1],
            ]
        )
    return features


def covariates(candidates: dict[str, np.ndarray]) -> np.ndarray:
    coordinate = marker_features(candidates)
    precise_distance = candidates["endpoint_distance"].astype(np.float64)[:, None]
    intensity = candidates["images"][:, 0].mean(axis=(1, 2), dtype=np.float64)[:, None]
    result = np.concatenate((coordinate, precise_distance, intensity), axis=1)
    if result.shape[1] != 16 or not np.isfinite(result).all():
        raise ValueError(f"Invalid matchedv3 test covariates: {result.shape}")
    return result


def greedy_unique_match_indices(
    features: np.ndarray, labels: np.ndarray, pair_count: int, k_neighbors: int = 256
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = labels.astype(np.uint8)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if negative.size < pair_count or positive.size < pair_count:
        raise ValueError("Test candidate pool is too small for matchedv3.")
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (features - center) / scale
    neg = standardized[negative]
    pos = standardized[positive]
    squared = (
        np.sum(neg * neg, axis=1)[:, None]
        + np.sum(pos * pos, axis=1)[None, :]
        - 2.0 * neg @ pos.T
    )
    np.maximum(squared, 0.0, out=squared)
    k = min(k_neighbors, positive.size)
    local_order = np.argsort(squared, axis=1, kind="stable")[:, :k]
    distances = np.sqrt(np.take_along_axis(squared, local_order, axis=1))
    negative_edges = np.repeat(negative, k)
    positive_edges = positive[local_order.reshape(-1)]
    distance_edges = distances.reshape(-1)
    order = np.lexsort((positive_edges, negative_edges, distance_edges))
    used_negative: set[int] = set()
    used_positive: set[int] = set()
    pairs: list[tuple[int, int]] = []
    accepted_distances: list[float] = []
    for edge_index in order:
        negative_index = int(negative_edges[edge_index])
        positive_index = int(positive_edges[edge_index])
        if negative_index in used_negative or positive_index in used_positive:
            continue
        used_negative.add(negative_index)
        used_positive.add(positive_index)
        pairs.append((negative_index, positive_index))
        accepted_distances.append(float(distance_edges[edge_index]))
        if len(pairs) == pair_count:
            break
    if len(pairs) != pair_count:
        raise RuntimeError(f"matchedv3 found only {len(pairs)} test pairs; required {pair_count}.")
    selected = np.asarray([index for pair in pairs for index in pair], dtype=np.int64)
    accepted = np.asarray(accepted_distances, dtype=np.float64)
    return selected, {
        "feature_dimension": int(features.shape[1]),
        "candidate_count": int(features.shape[0]),
        "negative_candidates": int(negative.size),
        "positive_candidates": int(positive.size),
        "k_neighbors": int(k),
        "accepted_pair_count": len(pairs),
        "standardized_pair_distance_min": float(accepted.min()),
        "standardized_pair_distance_median": float(np.median(accepted)),
        "standardized_pair_distance_p95": float(np.quantile(accepted, 0.95)),
        "standardized_pair_distance_max": float(accepted.max()),
    }


def matchedv3_test(candidates: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected, diagnostics = greedy_unique_match_indices(
        covariates(candidates), candidates["connected"], 600, 256
    )
    rng = np.random.default_rng(20260810 + 2_077_777)
    rng.shuffle(selected)
    arrays = {key: value[selected] for key, value in candidates.items()}
    arrays["matching_v3_k"] = np.full(1200, 256, dtype=np.int16)
    arrays["matching_v3_feature_dimension"] = np.full(1200, 16, dtype=np.int16)
    if arrays["images"].shape != (1200, 3, 64, 64):
        raise ValueError("Test matchedv3 image shape mismatch.")
    if not np.isclose(arrays["connected"].mean(), 0.5):
        raise ValueError("Test matchedv3 is not exactly balanced.")
    if tuple(sorted(np.unique(arrays["source_id"]))) != tuple(sorted(TEST_IDS)):
        raise ValueError("Test matchedv3 did not retain all 15 frozen sources.")
    return arrays, diagnostics


def decode_source_block(block: bytes, range_start: int, entry: dict[str, Any]) -> bytes:
    relative = int(entry["local_header_offset"]) - range_start
    fixed = block[relative : relative + 30]
    if len(fixed) != 30:
        raise ValueError(f"Local header outside range: {entry['name']}")
    (
        signature, _version, flags, method, _time, _date, _crc, _compressed,
        _uncompressed, filename_length, extra_length,
    ) = struct.unpack("<4s5H3L2H", fixed)
    if signature != LOCAL_SIGNATURE or flags & 0x1:
        raise ValueError(f"Unsupported local header: {entry['name']}")
    name_start = relative + 30
    encoding = "utf-8" if flags & 0x800 else "cp437"
    name = block[name_start : name_start + filename_length].decode(encoding)
    if name != entry["name"]:
        raise ValueError(f"Local/central filename mismatch: {entry['name']}")
    data_start = name_start + filename_length + extra_length
    data_end = data_start + int(entry["compressed_size"])
    compressed = block[data_start:data_end]
    if method == 0:
        raw = compressed
    elif method == 8:
        raw = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"Unsupported compression method {method}: {entry['name']}")
    if len(raw) != int(entry["uncompressed_size"]):
        raise ValueError(f"Uncompressed size mismatch: {entry['name']}")
    crc = f"{binascii.crc32(raw) & 0xFFFFFFFF:08x}"
    if crc != entry["crc32"]:
        raise ValueError(f"CRC mismatch: {entry['name']}")
    return raw


def range_request(start: int, end: int, attempts: int = 5) -> bytes:
    expected = end - start + 1
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                AUTHOR_URL,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": f"TopoBridge-{EXPERIMENT_ID}-TEST-ONCE/1.0",
                    "If-Range": EXPECTED_ETAG,
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
                status = response.status
                etag = response.headers.get("ETag")
                content_range = response.headers.get("Content-Range", "")
            if status != 206 or len(data) != expected:
                raise ValueError(f"HTTP/range length mismatch: status={status}, bytes={len(data)}")
            if etag is not None and etag != EXPECTED_ETAG:
                raise ValueError(f"Remote ETag changed: {etag}")
            if content_range and not content_range.endswith(f"/{EXPECTED_CONTENT_LENGTH}"):
                raise ValueError(f"Remote archive length changed: {content_range}")
            return data
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError("; ".join(errors))


def safe_output(root: Path, member: str) -> Path:
    pure = PurePosixPath(member)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe ZIP member: {member}")
    output = (root.resolve() / pure).resolve()
    if not output.is_relative_to(root.resolve()):
        raise ValueError(f"ZIP member escapes destination: {member}")
    return output


def fetch_test_source(record: dict[str, Any], destination: Path) -> dict[str, Any]:
    image = record["image"]
    rsml = record["rsml"]
    metadata = record["metadata"]
    start = int(image["local_header_offset"])
    end = int(metadata["local_header_offset"]) - 1
    if not (start < int(rsml["local_header_offset"]) < int(metadata["local_header_offset"])):
        raise ValueError(f"Unexpected archive member order for {record['source_id']}")
    block = range_request(start, end)
    image_raw = decode_source_block(block, start, image)
    rsml_raw = decode_source_block(block, start, rsml)
    image_path = safe_output(destination, image["name"])
    rsml_path = safe_output(destination, rsml["name"])
    if image_path.exists() or rsml_path.exists():
        raise FileExistsError(f"Refusing existing one-shot test member: {record['source_id']}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_raw)
    rsml_path.write_bytes(rsml_raw)
    return {
        "source_id": record["source_id"],
        "range_start": start,
        "range_end": end,
        "range_bytes": len(block),
        "image": checked_file(image_path),
        "image_member": image["name"],
        "image_crc32": image["crc32"],
        "rsml": checked_file(rsml_path),
        "rsml_member": rsml["name"],
        "rsml_crc32": rsml["crc32"],
    }


def create_marker_once(marker: Path, payload: dict[str, Any]) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_test_sources(destination: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    diagnostics = {}
    for source_id in TEST_IDS:
        directory = destination / source_id
        image_path = directory / f"image_{source_id}.jpg"
        rsml_path = directory / f"image_{source_id}.rsml"
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = np.round(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
        mask, diagnostic = render_rsml_bytes(rsml_path.read_bytes(), gray.shape, 8)
        sources[source_id] = (gray, mask)
        diagnostics[source_id] = {"shape": list(gray.shape), **diagnostic}
    return sources, diagnostics


def precompute_ph(probabilities: np.ndarray, output_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    descriptors = np.empty((probabilities.shape[0], 64, DESCRIPTOR_DIM), dtype=np.float32)
    started = time.perf_counter()
    for index in range(probabilities.shape[0]):
        descriptors[index], _ = ph_descriptors(probabilities[index], 64)
        if (index + 1) % 250 == 0 or index + 1 == probabilities.shape[0]:
            print(f"BRASSICA_TEST_PH progress={index + 1}/{probabilities.shape[0]}", flush=True)
    metadata = {
        "gudhi_version": gudhi.__version__,
        "seconds": time.perf_counter() - started,
        "diagnostics": aggregate_descriptor_diagnostics(descriptors),
    }
    np.savez_compressed(output_path, test=descriptors, metadata_json=np.asarray(json.dumps(metadata)))
    metadata["file"] = checked_file(output_path)
    return descriptors, metadata


def load_model_checkpoint(
    model: nn.Module, checkpoint_record: dict[str, Any], device: torch.device
) -> nn.Module:
    checked_file(Path(checkpoint_record["path"]), checkpoint_record["sha256"])
    checkpoint = torch.load(checkpoint_record["path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = truth.astype(np.uint8)
    prediction = prediction.astype(np.uint8)
    if not np.any(truth == 1) or not np.any(truth == 0):
        raise ValueError("Balanced accuracy requires both classes.")
    sensitivity = np.mean(prediction[truth == 1] == 1)
    specificity = np.mean(prediction[truth == 0] == 0)
    return float(0.5 * (sensitivity + specificity))


def aggregate_predictions(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    values = [balanced_accuracy(truth, current) for current in predictions]
    return {
        "balanced_accuracy_by_seed": values,
        "mean_balanced_accuracy": mean(values),
        "sample_sd_balanced_accuracy": stdev(values),
    }


def cluster_bootstrap(
    truth: np.ndarray, sources: np.ndarray, first: np.ndarray, second: np.ndarray
) -> dict[str, Any]:
    unique_sources = np.unique(sources)
    if unique_sources.size != 15:
        raise ValueError(f"Expected 15 source clusters, got {unique_sources.size}.")
    source_indices = {source: np.flatnonzero(sources == source) for source in unique_sources}
    gains = [balanced_accuracy(truth, first[i]) - balanced_accuracy(truth, second[i]) for i in range(3)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for repetition in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique_sources, size=unique_sources.size, replace=True)
        indices = np.concatenate([source_indices[source] for source in sampled])
        bootstrap[repetition] = mean(
            balanced_accuracy(truth[indices], first[i, indices])
            - balanced_accuracy(truth[indices], second[i, indices])
            for i in range(3)
        )
    return {
        "paired_gains_by_seed": gains,
        "mean_gain": mean(gains),
        "sample_sd_gain": stdev(gains),
        "source_cluster_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "source_image_count": 15,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def model_factories() -> dict[str, tuple[Callable[[], nn.Module], Callable]]:
    return {
        "mask_topo_external": (
            lambda: TopoCoarsenModel("mask_topo_external", 64, 16), forward_topology,
        ),
        "mask_conditioned_perceiver_strong": (
            lambda: make_mask_aware_model("mask_conditioned_perceiver_strong", 64, 16),
            forward_mask_aware,
        ),
        "mask_conditioned_slot_param_matched": (
            lambda: make_mask_aware_model("mask_conditioned_slot_param_matched", 64, 16),
            forward_mask_aware,
        ),
        "ph_only": (lambda: make_ph_model("ph_only", 64, 16), forward_ph),
    }


def synthetic_self_test(device: torch.device) -> dict[str, bool]:
    features = np.vstack((np.arange(8)[:, None], np.arange(8)[:, None] + 0.01)).astype(np.float64)
    features = np.hstack((features, features * 0.5))
    labels = np.asarray([0] * 8 + [1] * 8, dtype=np.uint8)
    selected, diagnostic = greedy_unique_match_indices(features, labels, 8, 8)
    checks = {
        "matcher_exact_unique_pairs": selected.size == 16 and np.unique(selected).size == 16,
        "matcher_pair_count": diagnostic["accepted_pair_count"] == 8,
    }
    image = torch.rand(2, 3, 64, 64, device=device)
    mask = torch.rand(2, 1, 64, 64, device=device)
    for name in MASK_AWARE_MODELS:
        model = make_mask_aware_model(name, 64, 16).to(device)
        tokens, metadata = model.reduce_tokens(image, mask)
        checks[f"{name}_exact_k"] = tuple(tokens.shape) == (2, 16, 64)
        checks[f"{name}_firewall"] = bool(
            not model.uses_connected_components
            and not model.uses_graph_construction
            and not model.uses_persistent_homology
        )
        checks[f"{name}_metadata"] = tuple(metadata.shape) == (2, 16, 3)
    descriptor = torch.rand(2, 16, DESCRIPTOR_DIM, device=device)
    checks["ph_only_forward"] = tuple(make_ph_model("ph_only", 64, 16).to(device)(image, mask, descriptor).shape) == (2,)
    return checks


def preflight(root: Path, args: argparse.Namespace) -> None:
    freeze_path = (root / args.freeze_manifest).resolve()
    source_path = (root / args.source_manifest).resolve()
    report_path = (root / args.preflight_report).resolve()
    destination = (root / args.destination).resolve()
    output = (root / args.output).resolve()
    marker = root / "实验记录" / f"{EXPERIMENT_ID}_TEST_INFERENCE_STARTED.json"
    if report_path.exists() or marker.exists() or destination.exists() or output.exists():
        raise FileExistsError("Preflight requires absent report, marker, test destination and test output.")
    frozen = verify_frozen_state(root, freeze_path, source_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks = synthetic_self_test(device)
    if not all(checks.values()):
        raise RuntimeError(f"Synthetic test-entry check failed: {checks}")
    factories = model_factories()
    checkpoint_checks = {}
    for seed in SEEDS:
        seed_key = str(seed)
        for name in SELECTED_MODELS:
            checkpoint_record = frozen["freeze"]["seed_artifacts"][seed_key]["model_checkpoints"][name]
            model, _forward = factories[name]
            loaded = load_model_checkpoint(model(), checkpoint_record, device)
            checkpoint_checks[f"{seed}/{name}"] = sum(parameter.numel() for parameter in loaded.parameters())
    script_path = Path(__file__).resolve()
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "TEST_ENTRY_PREFLIGHT_PASS_TEST_STILL_SEALED",
        "created_at": datetime.now().astimezone().isoformat(),
        "test_accessed": False,
        "test_image_member_request_count": 0,
        "test_rsml_member_request_count": 0,
        "marker_exists": False,
        "freeze_manifest": checked_file(freeze_path, FREEZE_MANIFEST_SHA256),
        "source_manifest": checked_file(source_path, SOURCE_MANIFEST_SHA256),
        "test_entry_script": checked_file(script_path),
        "verified_frozen_file_count": frozen["verified_file_count"],
        "selected_models": list(SELECTED_MODELS),
        "seeds": list(SEEDS),
        "test_ids": list(TEST_IDS),
        "synthetic_checks": checks,
        "checkpoint_parameter_counts": checkpoint_checks,
        "device": str(device),
        "python": sys.version,
        "torch": torch.__version__,
        "gudhi": gudhi.__version__,
        "network_request_performed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "preflight_report_sha256": sha256_file(report_path)}, ensure_ascii=False, indent=2))


def run_once(root: Path, args: argparse.Namespace) -> None:
    if args.workers < 1 or args.workers > 8 or args.batch_size != 32:
        raise ValueError("Frozen run-once command requires workers in [1,8] and batch size 32.")
    freeze_path = (root / args.freeze_manifest).resolve()
    source_path = (root / args.source_manifest).resolve()
    preflight_path = (root / args.preflight_report).resolve()
    destination = (root / args.destination).resolve()
    output = (root / args.output).resolve()
    marker = root / "实验记录" / f"{EXPERIMENT_ID}_TEST_INFERENCE_STARTED.json"
    if marker.exists() or destination.exists() or output.exists():
        raise FileExistsError("One-shot marker, destination or output already exists; refusing rerun.")
    frozen = verify_frozen_state(root, freeze_path, source_path)
    preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_report["status"] != "TEST_ENTRY_PREFLIGHT_PASS_TEST_STILL_SEALED":
        raise ValueError("Test entry preflight did not pass.")
    checked_file(Path(preflight_report["test_entry_script"]["path"]), preflight_report["test_entry_script"]["sha256"])
    if preflight_report["network_request_performed"] or preflight_report["test_accessed"]:
        raise ValueError("Preflight report has invalid access status.")

    marker_payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ONE_SHOT_TEST_INFERENCE_STARTED",
        "created_at": datetime.now().astimezone().isoformat(),
        "freeze_manifest_sha256": FREEZE_MANIFEST_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "preflight_report_sha256": sha256_file(preflight_path),
        "test_entry_script_sha256": sha256_file(Path(__file__).resolve()),
        "test_ids": list(TEST_IDS),
        "selected_models": list(SELECTED_MODELS),
        "seeds": list(SEEDS),
        "notice": "This marker is immutable. Any later failure is reportable and does not authorize a second test run.",
    }
    create_marker_once(marker, marker_payload)
    print(f"BRASSICA_TEST_MARKER_CREATED path={marker} sha256={sha256_file(marker)}", flush=True)

    output.mkdir(parents=True, exist_ok=False)
    destination.mkdir(parents=True, exist_ok=False)
    records = frozen["source"]["sources"]["test"]
    if tuple(record["source_id"] for record in records) != TEST_IDS:
        raise ValueError("Test request list changed after marker.")
    requests = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_test_source, record, destination): record for record in records}
        for index, future in enumerate(as_completed(futures), start=1):
            request = future.result()
            requests.append(request)
            print(f"BRASSICA_TEST_FETCH progress={index}/15 source={request['source_id']}", flush=True)
    requests.sort(key=lambda item: TEST_IDS.index(item["source_id"]))
    extraction = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ONE_SHOT_TEST_MEMBERS_FETCHED_AND_VERIFIED",
        "created_at": datetime.now().astimezone().isoformat(),
        "marker": checked_file(marker),
        "remote_archive": {
            "url": AUTHOR_URL,
            "expected_content_length": EXPECTED_CONTENT_LENGTH,
            "expected_etag": EXPECTED_ETAG,
        },
        "source_count": 15,
        "image_member_request_count": 15,
        "rsml_member_request_count": 15,
        "range_request_count": 15,
        "requests": requests,
    }
    extraction_path = root / "实验记录" / f"{EXPERIMENT_ID}_test_extraction_manifest.json"
    extraction_path.write_text(json.dumps(extraction, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sources, source_diagnostics = load_test_sources(destination)
    generation_args = SimpleNamespace(crop_size=64, output_size=64, min_component_pixels=12)
    candidates = generate_arrays(sources, list(TEST_IDS), 3600, 20260810 + 2_000_000, generation_args)
    candidate_path = output / "brassica_test_candidates_n3600_seed22260810.npz"
    np.savez_compressed(candidate_path, **candidates)
    arrays, matching_diagnostics = matchedv3_test(candidates)
    matched_path = output / "brassica_test_n1200_seed22260810_matchedv3.npz"
    np.savez_compressed(matched_path, **arrays)
    truth_reference = arrays["connected"].astype(np.uint8)
    sources_reference = arrays["source_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factories = model_factories()
    prediction_bank = {name: [] for name in SELECTED_MODELS}
    probability_bank = {name: [] for name in SELECTED_MODELS}
    seed_results: dict[str, Any] = {}

    for seed_index, seed in enumerate(SEEDS):
        seed_key = str(seed)
        seed_output = output / f"seed{seed}"
        seed_output.mkdir()
        seed_train = frozen["freeze"]["seed_artifacts"][seed_key]
        train_result = json.loads(Path(seed_train["result"]["path"]).read_text(encoding="utf-8"))
        mask_checkpoint = seed_train["mask_checkpoint"]
        mask_model = load_model_checkpoint(CrackUNet(), mask_checkpoint, device)
        native_probability = predict_masks(mask_model, arrays["images"], args.batch_size, device)
        probability_path = seed_output / "mask_probabilities_test.npz"
        np.savez_compressed(probability_path, test=native_probability.astype(np.float16))
        with np.load(probability_path, allow_pickle=False) as archive:
            if archive["test"].dtype != np.float16 or archive["test"].shape != (1200, 64, 64):
                raise ValueError(f"Seed {seed} test probability storage mismatch.")
            probabilities = archive["test"].astype(np.float32)
        selected = train_result["selected_on_dev_mask_calibration"]
        artifacts, artifact_diagnostics = build_artifacts(
            arrays,
            probabilities,
            float(selected["mask_threshold"]),
            int(selected["closing_iterations"]),
            16,
            "mask_topo_recheck",
            20260803 + 200_000,
        )
        if artifacts["assignment"].shape != (1200, 16, 16):
            raise ValueError(f"Seed {seed} MaskTopo assignment shape mismatch.")
        ph_path = seed_output / "ph_descriptors_test.npz"
        descriptors, ph_metadata = precompute_ph(probabilities, ph_path)
        datasets = {
            "mask_topo_external": RealDataset(arrays, artifacts, augment=False),
            "mask_conditioned_perceiver_strong": ReducerDataset(arrays, probabilities, augment=False),
            "mask_conditioned_slot_param_matched": ReducerDataset(arrays, probabilities, augment=False),
            "ph_only": PHDescriptorDataset(arrays, probabilities, descriptors, 16, augment=False),
        }
        current_seed = {
            "seed": seed,
            "mask_calibration_frozen_from_dev": selected,
            "mask_probabilities": checked_file(probability_path),
            "mask_probability_storage_dtype": "float16",
            "mask_probability_compute_dtype": "float32",
            "ph_cache": ph_metadata,
            "artifact_diagnostics": artifact_diagnostics,
            "models": {},
        }
        for name in SELECTED_MODELS:
            model_factory, forward = factories[name]
            checkpoint_record = seed_train["model_checkpoints"][name]
            model = load_model_checkpoint(model_factory(), checkpoint_record, device)
            loader = DataLoader(datasets[name], batch_size=32, shuffle=False, num_workers=0)
            metrics, prediction, probability, truth = evaluate_generic(model, loader, device, forward)
            if not np.array_equal(truth, truth_reference):
                raise ValueError(f"Truth changed for seed/model {seed}/{name}.")
            prediction_bank[name].append(prediction)
            probability_bank[name].append(probability)
            current_seed["models"][name] = {
                "metrics": metrics,
                "checkpoint": checkpoint_record,
                "finite_probability": bool(np.isfinite(probability).all()),
            }
            print(
                f"BRASSICA_TEST_EVAL seed={seed} model={name} balanced_accuracy={metrics['balanced_accuracy']:.6f}",
                flush=True,
            )
        seed_prediction_path = seed_output / "test_predictions.npz"
        np.savez_compressed(
            seed_prediction_path,
            truth=truth_reference,
            source_id=sources_reference,
            **{f"prediction_{name}": prediction_bank[name][-1] for name in SELECTED_MODELS},
            **{f"probability_{name}": probability_bank[name][-1] for name in SELECTED_MODELS},
        )
        current_seed["predictions"] = checked_file(seed_prediction_path)
        seed_results[seed_key] = current_seed

    predictions = {name: np.stack(values) for name, values in prediction_bank.items()}
    probabilities_all = {name: np.stack(values) for name, values in probability_bank.items()}
    combined_path = output / "test_predictions_all_seeds.npz"
    np.savez_compressed(
        combined_path,
        truth=truth_reference,
        source_id=sources_reference,
        seeds=np.asarray(SEEDS, dtype=np.int64),
        **{f"prediction_{name}": value for name, value in predictions.items()},
        **{f"probability_{name}": value for name, value in probabilities_all.items()},
    )
    aggregates = {name: aggregate_predictions(truth_reference, predictions[name]) for name in SELECTED_MODELS}
    contrasts = {
        "masktopo_minus_mask_perceiver_strong": cluster_bootstrap(
            truth_reference, sources_reference, predictions["mask_topo_external"],
            predictions["mask_conditioned_perceiver_strong"],
        ),
        "masktopo_minus_mask_slot_param_matched": cluster_bootstrap(
            truth_reference, sources_reference, predictions["mask_topo_external"],
            predictions["mask_conditioned_slot_param_matched"],
        ),
        "masktopo_minus_best_ph_dev_ph_only": cluster_bootstrap(
            truth_reference, sources_reference, predictions["mask_topo_external"], predictions["ph_only"],
        ),
    }
    non_topology_keys = (
        "masktopo_minus_mask_perceiver_strong",
        "masktopo_minus_mask_slot_param_matched",
    )
    fairness_support = all(contrasts[key]["source_cluster_bootstrap_95_ci"][0] > 0 for key in non_topology_keys)
    topology_support = fairness_support and contrasts["masktopo_minus_best_ph_dev_ph_only"]["source_cluster_bootstrap_95_ci"][0] > 0
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ONE_SHOT_TEST_COMPLETED",
        "completed_at": datetime.now().astimezone().isoformat(),
        "claim_boundary": "untouched-test external validation after disclosed train/dev task-shortcut repair",
        "endpoint_disease_firewall": "Endpoint-connectivity effects are separate from P1-P5 disease classification +1.00 pp.",
        "one_shot_marker": checked_file(marker),
        "freeze_manifest": checked_file(freeze_path, FREEZE_MANIFEST_SHA256),
        "preflight_report": checked_file(preflight_path),
        "test_entry_script": checked_file(Path(__file__).resolve()),
        "test_source_count": int(np.unique(sources_reference).size),
        "test_sample_count": int(truth_reference.size),
        "positive_fraction": float(truth_reference.mean()),
        "test_ids": list(TEST_IDS),
        "test_extraction_manifest": checked_file(extraction_path),
        "candidate_cache": checked_file(candidate_path),
        "matchedv3_cache": checked_file(matched_path),
        "matching_diagnostics": matching_diagnostics,
        "source_diagnostics": source_diagnostics,
        "selected_models": list(SELECTED_MODELS),
        "best_ph_selected_on_dev": "ph_only",
        "seed_results": seed_results,
        "aggregates": aggregates,
        "primary_contrasts": contrasts,
        "strong_external_fairness_support": fairness_support,
        "strong_external_topology_representation_support": topology_support,
        "combined_predictions": checked_file(combined_path),
        "statistics": {
            "unit": "source image cluster",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "source_count": 15,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gudhi": gudhi.__version__,
        },
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# TB-B-260803-035 Brassica one-shot sealed-test report",
        "",
        "本实验是披露 train/dev task-shortcut repair 后的 untouched-test 外部验证；不是原作者文件级划分复现。",
        "端点连通任务效应与 P1–P5 疾病分类 `+1.00 pp` 严格分开。",
        "",
        "## Test balanced accuracy",
        "",
        "| 模型 | 三种子 BA | 均值±样本SD |",
        "|---|---|---|",
    ]
    for name in SELECTED_MODELS:
        item = aggregates[name]
        seeds_text = "/".join(f"{100 * value:.2f}%" for value in item["balanced_accuracy_by_seed"])
        lines.append(f"| {name} | {seeds_text} | {100 * item['mean_balanced_accuracy']:.2f}±{100 * item['sample_sd_balanced_accuracy']:.2f}% |")
    lines.extend(["", "## Source-level paired cluster bootstrap", ""])
    for name, item in contrasts.items():
        lower, upper = item["source_cluster_bootstrap_95_ci"]
        lines.append(f"- {name}: {100 * item['mean_gain']:+.2f} pp, 95% CI [{100 * lower:+.2f}, {100 * upper:+.2f}] pp.")
    lines.extend(
        [
            "",
            f"- strong external fairness support: `{str(fairness_support).lower()}`",
            f"- strong external topology-representation support: `{str(topology_support).lower()}`",
            "- 统计单位：15 个 source images；20,000 次配对 source-level cluster bootstrap，seed 20260803。",
            "- test 在冻结后只评估一次；任何阴性、反向或 CI 跨零结果均原样报告。",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    completion = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ONE_SHOT_TEST_COMPLETED",
        "result": checked_file(result_path),
        "report": checked_file(output / "REPORT.md"),
        "strong_external_fairness_support": fairness_support,
        "strong_external_topology_representation_support": topology_support,
    }
    completion_path = root / "实验记录" / f"{EXPERIMENT_ID}_TEST_COMPLETED.json"
    completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.stage == "self-test":
        checks = synthetic_self_test(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if not all(checks.values()):
            raise RuntimeError(checks)
        print(json.dumps({"stage": "self-test", "status": "PASS", "checks": checks}, indent=2))
    elif args.stage == "preflight":
        preflight(root, args)
    else:
        run_once(root, args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

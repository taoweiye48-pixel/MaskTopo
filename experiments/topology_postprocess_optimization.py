from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from crackforest_mechanism_ablation import limit_groups
from crackforest_mask_topo import mask_groups
from crackforest_real_gate import coarse_graph8, patch_component_ids
from standard_vit_efficiency import StandardViTTokenBenchmark
from topology_construction_benchmark import (
    artifacts_to_device,
    coarse_graph8_batch,
    dynamic_masktopo_forward,
)
from topocoarsen_oracle import (
    allocate_cluster_counts,
    assign_group,
    farthest_seeds,
    oracle_assignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exactness and speed gate for TopoBridge post-processing."
    )
    parser.add_argument(
        "--mask-run-dir",
        type=Path,
        default=Path("results_fives_seed20260810"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("fives_cache_v2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_topology_postprocess_optimization"),
    )
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--correctness-count", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--synthetic-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def summary(milliseconds: list[float], batch_size: int) -> dict[str, float | int]:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "batch_size": int(batch_size),
        "measurements": int(values.size),
        "mean_ms_per_batch": float(values.mean()),
        "median_ms_per_batch": float(np.median(values)),
        "std_ms_per_batch": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p95_ms_per_batch": float(np.percentile(values, 95)),
        "mean_ms_per_sample": float(values.mean() / batch_size),
        "median_ms_per_sample": float(np.median(values) / batch_size),
        "p95_ms_per_sample": float(np.percentile(values, 95) / batch_size),
    }


def benchmark(
    function: Callable[[], Any],
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        function()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        timings.append(1000.0 * (time.perf_counter() - started))
    return summary(timings, batch_size)


def vectorized_patch_component_ids(component_maps: np.ndarray) -> np.ndarray:
    if component_maps.ndim != 3 or component_maps.shape[1:] != (64, 64):
        raise ValueError("Expected Bx64x64 component maps.")
    batch_size = component_maps.shape[0]
    blocks = component_maps.reshape(
        batch_size, 16, 4, 16, 4
    ).transpose(0, 1, 3, 2, 4).reshape(batch_size, 16, 16, 16)
    sorted_blocks = np.sort(blocks, axis=-1)
    positive = sorted_blocks > 0
    same = sorted_blocks[..., :, None] == sorted_blocks[..., None, :]
    counts = (same & positive[..., :, None]).sum(axis=-1, dtype=np.int16)
    counts = np.where(positive, counts, -1)
    best_index = np.argmax(counts, axis=-1)
    patches = np.take_along_axis(
        sorted_blocks, best_index[..., None], axis=-1
    )[..., 0]
    patches[counts.max(axis=-1) < 0] = 0
    return patches.astype(np.int16)


def vectorized_limit_groups(
    patches: np.ndarray, maximum_foreground: int
) -> np.ndarray:
    output = np.zeros_like(patches, dtype=np.int16)
    for index, patch in enumerate(patches):
        labels, counts = np.unique(patch, return_counts=True)
        positive = labels > 0
        labels = labels[positive].astype(np.int16)
        counts = counts[positive]
        if labels.size > maximum_foreground:
            order = np.lexsort((labels, -counts))
            kept = np.sort(labels[order[:maximum_foreground]])
        else:
            kept = labels
        if kept.size == 0:
            continue
        positive_pixels = patch > 0
        retained = positive_pixels & np.isin(patch, kept)
        output[index][retained] = (
            np.searchsorted(kept, patch[retained]).astype(np.int16) + 1
        )
    return output


def vectorized_groups_from_components(
    component_maps: np.ndarray, maximum_foreground: int
) -> np.ndarray:
    patches = vectorized_patch_component_ids(component_maps)
    return vectorized_limit_groups(patches, maximum_foreground)


def fast_farthest_seeds(
    nodes: np.ndarray, count: int
) -> list[int]:
    coordinates = nodes.astype(np.float64)
    centroid = coordinates.mean(axis=0)
    selected_indices = [int(np.argmin(np.sum((coordinates - centroid) ** 2, axis=1)))]
    while len(selected_indices) < count:
        selected = coordinates[np.asarray(selected_indices)]
        distances = np.min(
            np.sum(
                (coordinates[:, None, :] - selected[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        distances[np.asarray(selected_indices)] = -1.0
        selected_indices.append(int(np.argmax(distances)))
    return [int(nodes[index, 0] * 16 + nodes[index, 1]) for index in selected_indices]


def fast_assign_group(nodes: np.ndarray, seeds: list[int]) -> np.ndarray:
    mask = np.zeros((16, 16), dtype=bool)
    mask[nodes[:, 0], nodes[:, 1]] = True
    seed_coordinates = np.asarray(
        [(seed // 16, seed % 16) for seed in seeds], dtype=np.int32
    )
    infinity = np.int32(1_000_000_000)
    distances = np.full((len(seeds), 16, 16), infinity, dtype=np.int32)
    for label, (row, col) in enumerate(seed_coordinates):
        distances[label, row, col] = 0
    for _ in range(256):
        candidate = np.full_like(distances, infinity)
        candidate[:, 1:, :] = np.minimum(
            candidate[:, 1:, :], distances[:, :-1, :] + 1
        )
        candidate[:, :-1, :] = np.minimum(
            candidate[:, :-1, :], distances[:, 1:, :] + 1
        )
        candidate[:, :, 1:] = np.minimum(
            candidate[:, :, 1:], distances[:, :, :-1] + 1
        )
        candidate[:, :, :-1] = np.minimum(
            candidate[:, :, :-1], distances[:, :, 1:] + 1
        )
        candidate[:, ~mask] = infinity
        updated = np.minimum(distances, candidate)
        if np.array_equal(updated, distances):
            distances = updated
            break
        distances = updated
    labels = np.argmin(distances, axis=0).astype(np.int16)
    unreachable = distances.min(axis=0) >= infinity
    if np.any(unreachable):
        coordinates = np.indices((16, 16)).transpose(1, 2, 0)
        squared = np.sum(
            (coordinates[unreachable, None, :] - seed_coordinates[None, :, :])
            ** 2,
            axis=2,
        )
        labels[unreachable] = np.argmin(squared, axis=1).astype(np.int16)
    return labels


def fast_canonicalize_assignment(
    assignment: np.ndarray, token_count: int
) -> np.ndarray:
    rows, cols = np.indices(assignment.shape)
    flat = assignment.reshape(-1)
    row_sum = np.bincount(
        flat, weights=rows.reshape(-1), minlength=token_count
    )
    col_sum = np.bincount(
        flat, weights=cols.reshape(-1), minlength=token_count
    )
    counts = np.bincount(flat, minlength=token_count)
    row_mean = row_sum / counts
    col_mean = col_sum / counts
    order = np.lexsort((np.arange(token_count), col_mean, row_mean))
    remap = np.empty(token_count, dtype=np.int16)
    remap[order] = np.arange(token_count, dtype=np.int16)
    return remap[assignment]


def fast_oracle_assignment(
    patch_ids: np.ndarray, token_count: int
) -> np.ndarray:
    counts = allocate_cluster_counts(patch_ids, token_count, None)
    assignment = np.full((16, 16), -1, dtype=np.int16)
    next_cluster = 0
    for group in sorted(counts):
        nodes = np.argwhere(patch_ids == group)
        seeds = fast_farthest_seeds(nodes, counts[group])
        local = fast_assign_group(nodes, seeds)
        assignment[nodes[:, 0], nodes[:, 1]] = (
            next_cluster + local[nodes[:, 0], nodes[:, 1]]
        )
        next_cluster += counts[group]
    if np.any(assignment < 0) or next_cluster != token_count:
        raise AssertionError("Fast assignment did not cover the full patch grid.")
    return fast_canonicalize_assignment(assignment, token_count)


def reference_groups_from_components(component_maps: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            limit_groups(
                patch_component_ids(component_map), maximum_foreground=7
            )
            for component_map in component_maps
        ]
    ).astype(np.int16)


def reference_assignments(groups: np.ndarray, token_count: int) -> np.ndarray:
    return np.stack(
        [oracle_assignment(group, token_count, bridge_aware=False) for group in groups]
    ).astype(np.int16)


def fast_assignments(groups: np.ndarray, token_count: int) -> np.ndarray:
    return np.stack(
        [fast_oracle_assignment(group, token_count) for group in groups]
    ).astype(np.int16)


def build_reference(groups: np.ndarray, token_count: int) -> dict[str, np.ndarray]:
    assignments = reference_assignments(groups, token_count)
    adjacency = []
    reachability = []
    for assignment, group in zip(assignments, groups):
        graph, closure = coarse_graph8(
            assignment, group, token_count, topology_aware=True
        )
        adjacency.append(graph)
        reachability.append(closure)
    return {
        "assignment": assignments,
        "adjacency": np.stack(adjacency).astype(np.uint8),
        "reachability": np.stack(reachability).astype(np.uint8),
    }


def build_reference_from_components(
    component_maps: np.ndarray, token_count: int
) -> dict[str, np.ndarray]:
    groups = reference_groups_from_components(component_maps)
    return build_reference(groups, token_count)


def build_hybrid(component_maps: np.ndarray, token_count: int) -> dict[str, np.ndarray]:
    groups = vectorized_groups_from_components(component_maps, 7)
    assignments = reference_assignments(groups, token_count)
    adjacency, reachability = coarse_graph8_batch(
        assignments, groups, token_count
    )
    return {
        "assignment": assignments,
        "adjacency": adjacency,
        "reachability": reachability,
    }


def build_fast(component_maps: np.ndarray, token_count: int) -> dict[str, np.ndarray]:
    groups = vectorized_groups_from_components(component_maps, 7)
    assignments = fast_assignments(groups, token_count)
    adjacency, reachability = coarse_graph8_batch(
        assignments, groups, token_count
    )
    return {
        "assignment": assignments,
        "adjacency": adjacency,
        "reachability": reachability,
    }


def artifact_equal(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> dict[str, bool]:
    return {
        key: bool(np.array_equal(first[key], second[key]))
        for key in ("assignment", "adjacency", "reachability")
    }


def load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(args.mask_run_dir / "mask_probabilities.npz") as archive:
        probabilities = archive["test"][: args.sample_count].astype(np.float32)
    frozen_result = json.loads(
        (args.mask_run_dir / "result.json").read_text(encoding="utf-8")
    )
    selected = frozen_result["selected_on_dev"]
    threshold = float(selected["mask_threshold"])
    if int(selected["closing_iterations"]) != 0:
        raise ValueError("This exact post-processing gate is frozen at closing=0.")
    cache_files = sorted(args.cache_dir.glob("fives_test_n1200_*.npz"))
    if len(cache_files) != 1:
        raise FileNotFoundError(
            f"Expected one cached FIVES test file, found {cache_files}."
        )
    with np.load(cache_files[0]) as archive:
        images = archive["images"][: args.sample_count].astype(np.float32) / 255.0
    return probabilities, images, threshold


def run_vit_output_gate(
    images: np.ndarray,
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, float | bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The downstream output gate requires CUDA.")
    image_tensor = torch.from_numpy(images[:batch_size]).to(device)
    image_tensor = F.interpolate(
        image_tensor, size=(256, 256), mode="bilinear", align_corners=False
    )
    reference_current = {
        key: value[:batch_size] for key, value in reference.items()
    }
    candidate_current = {
        key: value[:batch_size] for key, value in candidate.items()
    }
    model = StandardViTTokenBenchmark(
        8,
        2,
        "masktopo_cached",
        reference_current["assignment"],
        reference_current["adjacency"],
        reference_current["reachability"],
    ).to(device).eval()
    ref_tensors = artifacts_to_device(reference_current, device)
    candidate_tensors = artifacts_to_device(candidate_current, device)
    with torch.no_grad():
        reference_output = dynamic_masktopo_forward(
            model, image_tensor, *ref_tensors
        )
        candidate_output = dynamic_masktopo_forward(
            model, image_tensor, *candidate_tensors
        )
    difference = (reference_output - candidate_output).abs()
    return {
        "allclose": bool(torch.allclose(reference_output, candidate_output, atol=1e-6, rtol=1e-5)),
        "max_abs_difference": float(difference.max().item()),
    }


def make_synthetic_groups(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    patterns = []
    for _ in range(count):
        groups = np.zeros((16, 16), dtype=np.int16)
        group_count = int(rng.integers(1, 5))
        groups[:] = rng.integers(0, group_count + 1, size=(16, 16))
        patterns.append(groups)
    return np.stack(patterns)


def run_self_test() -> None:
    rng = np.random.default_rng(11)
    component_maps = np.zeros((8, 64, 64), dtype=np.int32)
    component_maps[:, 8:56, 8:56] = rng.integers(
        1, 9, size=(8, 48, 48), dtype=np.int32
    )
    reference_groups = reference_groups_from_components(component_maps)
    fast_groups = vectorized_groups_from_components(component_maps, 7)
    assert np.array_equal(reference_groups, fast_groups)
    reference = build_reference(reference_groups, 8)
    fast = build_fast(component_maps, 8)
    assert all(artifact_equal(reference, fast).values())
    synthetic = make_synthetic_groups(32, 13)
    assert all(
        np.array_equal(
            reference_assignments(synthetic[index : index + 1], 8)[0],
            fast_assignments(synthetic[index : index + 1], 8)[0],
        )
        for index in range(synthetic.shape[0])
    )
    print("TOPOLOGY_POSTPROCESS_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.token_count != 8:
        raise ValueError("This gate is frozen at K=8.")
    if args.sample_count < max(args.batch_sizes):
        raise ValueError("sample-count must cover the largest batch.")
    if args.correctness_count > args.sample_count:
        raise ValueError("correctness-count cannot exceed sample-count.")
    if args.self_test:
        run_self_test()
        return
    probabilities, images, threshold = load_inputs(args)
    component_maps = np.stack(
        [mask_groups(probability, threshold, 0)[0] for probability in probabilities]
    ).astype(np.int32)
    correctness_maps = component_maps[: args.correctness_count]
    print(
        f"CORRECTNESS_START samples={args.correctness_count} threshold={threshold}",
        flush=True,
    )
    reference_groups = reference_groups_from_components(correctness_maps)
    fast_groups = vectorized_groups_from_components(correctness_maps, 7)
    groups_equal = bool(np.array_equal(reference_groups, fast_groups))
    reference_artifacts = build_reference(reference_groups, args.token_count)
    hybrid_artifacts = build_hybrid(correctness_maps, args.token_count)
    fast_artifacts = build_fast(correctness_maps, args.token_count)
    hybrid_artifact_checks = artifact_equal(reference_artifacts, hybrid_artifacts)
    fast_artifact_checks = artifact_equal(reference_artifacts, fast_artifacts)
    synthetic = make_synthetic_groups(args.synthetic_count, args.seed)
    synthetic_assignment_equal = True
    synthetic_mismatch_indices = []
    for index in range(synthetic.shape[0]):
        reference_assignment = reference_assignments(
            synthetic[index : index + 1], args.token_count
        )[0]
        fast_assignment = fast_assignments(
            synthetic[index : index + 1], args.token_count
        )[0]
        if not np.array_equal(reference_assignment, fast_assignment):
            synthetic_assignment_equal = False
            synthetic_mismatch_indices.append(index)
    hybrid_vit_gate = run_vit_output_gate(
        images[: args.correctness_count],
        reference_artifacts,
        hybrid_artifacts,
        min(8, args.correctness_count),
    )
    fast_vit_gate = run_vit_output_gate(
        images[: args.correctness_count],
        reference_artifacts,
        fast_artifacts,
        min(8, args.correctness_count),
    )
    correctness = {
        "sample_count": args.correctness_count,
        "binary_component_maps_fixed": True,
        "patch_groups_equal": groups_equal,
        "hybrid_artifact_equality": hybrid_artifact_checks,
        "fully_vectorized_artifact_equality": fast_artifact_checks,
        "synthetic_count": args.synthetic_count,
        "synthetic_assignment_equal": synthetic_assignment_equal,
        "synthetic_mismatch_indices": synthetic_mismatch_indices,
        "hybrid_vit_output_gate": hybrid_vit_gate,
        "fully_vectorized_vit_output_gate": fast_vit_gate,
    }
    correctness_pass = bool(
        groups_equal
        and all(hybrid_artifact_checks.values())
        and all(fast_artifact_checks.values())
        and synthetic_assignment_equal
        and hybrid_vit_gate["allclose"]
        and fast_vit_gate["allclose"]
    )
    correctness["gate_pass"] = correctness_pass
    if not correctness_pass:
        raise RuntimeError(
            "POSTPROCESS_CORRECTNESS_GATE_FAILED "
            + json.dumps(correctness, ensure_ascii=False)
        )
    print("CORRECTNESS_PASS", flush=True)

    patch_results: dict[str, Any] = {}
    assignment_results: dict[str, Any] = {}
    full_results: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        key = str(batch_size)
        batch_maps = component_maps[:batch_size]
        reference_group_batch = reference_groups_from_components(batch_maps)
        fast_group_batch = vectorized_groups_from_components(batch_maps, 7)
        patch_results[key] = {
            "reference": benchmark(
                lambda: reference_groups_from_components(batch_maps),
                batch_size,
                args.warmup,
                args.repeats,
            ),
            "vectorized": benchmark(
                lambda: vectorized_groups_from_components(batch_maps, 7),
                batch_size,
                args.warmup,
                args.repeats,
            ),
        }
        assignment_results[key] = {
            "reference": benchmark(
                lambda: reference_assignments(reference_group_batch, args.token_count),
                batch_size,
                args.warmup,
                args.repeats,
            ),
            "vectorized": benchmark(
                lambda: fast_assignments(fast_group_batch, args.token_count),
                batch_size,
                args.warmup,
                args.repeats,
            ),
        }
        full_results[key] = {
            "reference_postprocess": benchmark(
                lambda: build_reference_from_components(batch_maps, args.token_count),
                batch_size,
                args.warmup,
                args.repeats,
            ),
            "hybrid_postprocess": benchmark(
                lambda: build_hybrid(batch_maps, args.token_count),
                batch_size,
                args.warmup,
                args.repeats,
            ),
            "fully_vectorized_postprocess": benchmark(
                lambda: build_fast(batch_maps, args.token_count),
                batch_size,
                args.warmup,
                args.repeats,
            ),
        }
        print(f"BATCH_DONE batch={batch_size}", flush=True)

    speedups = {}
    for batch_size in args.batch_sizes:
        key = str(batch_size)
        reference_ms = full_results[key]["reference_postprocess"][
            "mean_ms_per_sample"
        ]
        hybrid_ms = full_results[key]["hybrid_postprocess"][
            "mean_ms_per_sample"
        ]
        fully_vectorized_ms = full_results[key]["fully_vectorized_postprocess"][
            "mean_ms_per_sample"
        ]
        speedups[key] = {
            "reference_ms_per_sample": reference_ms,
            "hybrid_ms_per_sample": hybrid_ms,
            "fully_vectorized_ms_per_sample": fully_vectorized_ms,
            "hybrid_reduction_percent": 100.0
            * (reference_ms - hybrid_ms)
            / reference_ms,
            "fully_vectorized_reduction_percent": 100.0
            * (reference_ms - fully_vectorized_ms)
            / reference_ms,
            "patch_reduction_percent": 100.0
            * (
                patch_results[key]["reference"]["mean_ms_per_sample"]
                - patch_results[key]["vectorized"]["mean_ms_per_sample"]
            )
            / patch_results[key]["reference"]["mean_ms_per_sample"],
            "assignment_reduction_percent": 100.0
            * (
                assignment_results[key]["reference"]["mean_ms_per_sample"]
                - assignment_results[key]["vectorized"]["mean_ms_per_sample"]
            )
            / assignment_results[key]["reference"]["mean_ms_per_sample"],
        }

    report = {
        "experiment_id": "TB-B-260731-005",
        "status": "completed",
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu",
        },
        "dataset": {
            "name": "FIVES mask probabilities",
            "mask_run_dir": str(args.mask_run_dir.resolve()),
            "sample_count": args.sample_count,
            "correctness_count": args.correctness_count,
            "mask_threshold": threshold,
            "closing_iterations": 0,
        },
        "configuration": {
            "patch_grid": [16, 16],
            "token_count": args.token_count,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "synthetic_count": args.synthetic_count,
        },
        "correctness": correctness,
        "patch_grouping": patch_results,
        "assignment": assignment_results,
        "full_postprocess": full_results,
        "speedups": speedups,
        "limitations": [
            "This gate uses frozen mask probabilities and excludes mask predictor latency.",
            "The vectorized assignment uses the same K=8 non-bridge-aware oracle policy; bridge-aware mode is not changed.",
            "The ViT output gate uses random initialization and supports only numerical equivalence, not accuracy.",
        ],
        "protocol": str(
            (Path(__file__).parent / "TOPOLOGY_POSTPROCESS_OPTIMIZATION_PROTOCOL.md").resolve()
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# TopoBridge post-processing optimization",
        "",
        "The hybrid and fully vectorized paths passed real-data, synthetic-pattern, artifact, and downstream-output exactness gates before timing.",
        "",
        "| Batch | Reference postprocess | Hybrid selected | Fully vectorized | Hybrid reduction | Fully vectorized reduction | Patch reduction | Assignment reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for batch_size in args.batch_sizes:
        key = str(batch_size)
        item = speedups[key]
        lines.append(
            f"| {batch_size} | {item['reference_ms_per_sample']:.3f} | "
            f"{item['hybrid_ms_per_sample']:.3f} | "
            f"{item['fully_vectorized_ms_per_sample']:.3f} | "
            f"{item['hybrid_reduction_percent']:.2f}% | "
            f"{item['fully_vectorized_reduction_percent']:.2f}% | "
            f"{item['patch_reduction_percent']:.2f}% | "
            f"{item['assignment_reduction_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "All means are CPU wall-clock milliseconds per sample; median and P95 are in `result.json`.",
            "",
            f"Correctness gate: {correctness_pass}",
            f"Real patch groups equal: {groups_equal}",
            f"Synthetic assignments equal: {synthetic_assignment_equal}",
            f"Hybrid downstream ViT allclose: {hybrid_vit_gate['allclose']}",
            f"Fully vectorized downstream ViT allclose: {fast_vit_gate['allclose']}",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "completed", "speedups": speedups}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

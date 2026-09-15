from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_closing, label as scipy_label

from crackforest_mask_topo import CrackUNet, mask_groups
from crackforest_mechanism_ablation import limit_groups, predicted_groups
from crackforest_real_gate import coarse_graph8, patch_component_ids
from fives_external_gate import load_all_arrays
from standard_vit_efficiency import (
    StandardViTTokenBenchmark,
    rectangular_assignment,
)
from topocoarsen_oracle import oracle_assignment


PROTOCOL_NAME = "TOPOLOGY_CONSTRUCTION_BENCHMARK_PROTOCOL.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile and optimize TopoBridge topology construction."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("real_data/FIVES/dataset/preprocessed512_v2"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("fives_cache_v2"))
    parser.add_argument(
        "--mask-run-dir",
        type=Path,
        default=Path("results_fives_seed20260810"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_topology_construction_benchmark"),
    )
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--correctness-count", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--insert-layer", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--gpu-check-interval", type=int, default=16)
    parser.add_argument("--gpu-max-iterations", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def latency_summary(milliseconds: list[float], batch_size: int) -> dict[str, float | int]:
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    function: Callable[[], Any],
    batch_size: int,
    warmup: int,
    repeats: int,
    device: torch.device | None = None,
) -> dict[str, float | int | None]:
    for _ in range(warmup):
        function()
    if device is not None:
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    for _ in range(repeats):
        if device is not None:
            synchronize(device)
        started = time.perf_counter()
        function()
        if device is not None:
            synchronize(device)
        timings.append(1000.0 * (time.perf_counter() - started))
    result: dict[str, float | int | None] = latency_summary(timings, batch_size)
    result["peak_cuda_memory_mb"] = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device is not None and device.type == "cuda"
        else None
    )
    return result


def build_cpu_reference(
    probabilities: np.ndarray,
    threshold: float,
    closing_iterations: int,
    token_count: int,
) -> dict[str, np.ndarray]:
    assignments = []
    adjacency = []
    reachability = []
    for probability in probabilities:
        groups = predicted_groups(
            probability,
            threshold,
            closing_iterations,
            maximum_foreground=token_count - 1,
        )
        assignment = oracle_assignment(groups, token_count, bridge_aware=False)
        graph, closure = coarse_graph8(
            assignment, groups, token_count, topology_aware=True
        )
        assignments.append(assignment)
        adjacency.append(graph)
        reachability.append(closure)
    return {
        "assignment": np.stack(assignments).astype(np.int16),
        "adjacency": np.stack(adjacency).astype(np.uint8),
        "reachability": np.stack(reachability).astype(np.uint8),
    }


def coarse_graph8_batch(
    assignments: np.ndarray,
    patch_ids: np.ndarray,
    token_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = assignments.shape[0]
    adjacency = np.broadcast_to(
        np.eye(token_count, dtype=np.uint8),
        (batch_size, token_count, token_count),
    ).copy()
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for delta_row, delta_col in directions:
        row_source = slice(max(0, -delta_row), min(16, 16 - delta_row))
        col_source = slice(max(0, -delta_col), min(16, 16 - delta_col))
        row_target = slice(max(0, delta_row), min(16, 16 + delta_row))
        col_target = slice(max(0, delta_col), min(16, 16 + delta_col))
        first_group = patch_ids[:, row_source, col_source]
        second_group = patch_ids[:, row_target, col_target]
        valid = ((first_group == 0) & (second_group == 0)) | (
            (first_group > 0) & (first_group == second_group)
        )
        batch_index, row_index, col_index = np.nonzero(valid)
        if batch_index.size == 0:
            continue
        first_cluster = assignments[:, row_source, col_source][
            batch_index, row_index, col_index
        ]
        second_cluster = assignments[:, row_target, col_target][
            batch_index, row_index, col_index
        ]
        adjacency[batch_index, first_cluster, second_cluster] = 1
        adjacency[batch_index, second_cluster, first_cluster] = 1
    reachability = adjacency.astype(bool)
    for pivot in range(token_count):
        reachability |= (
            reachability[:, :, pivot, None]
            & reachability[:, pivot, None, :]
        )
    return adjacency, reachability.astype(np.uint8)


def build_cpu_batch(
    probabilities: np.ndarray,
    threshold: float,
    closing_iterations: int,
    token_count: int,
) -> dict[str, np.ndarray]:
    groups = np.stack(
        [
            predicted_groups(
                probability,
                threshold,
                closing_iterations,
                maximum_foreground=token_count - 1,
            )
            for probability in probabilities
        ]
    ).astype(np.int16)
    assignments = np.stack(
        [
            oracle_assignment(group, token_count, bridge_aware=False)
            for group in groups
        ]
    ).astype(np.int16)
    adjacency, reachability = coarse_graph8_batch(
        assignments, groups, token_count
    )
    return {
        "assignment": assignments,
        "adjacency": adjacency,
        "reachability": reachability,
    }


@torch.no_grad()
def gpu_connected_components(
    probabilities: torch.Tensor,
    threshold: float,
    check_interval: int,
    max_iterations: int,
) -> tuple[np.ndarray, int]:
    if probabilities.ndim != 3 or probabilities.shape[1:] != (64, 64):
        raise ValueError("Expected Bx64x64 mask probabilities.")
    foreground = probabilities >= threshold
    base = torch.arange(
        1,
        64 * 64 + 1,
        device=probabilities.device,
        dtype=torch.float32,
    ).reshape(1, 64, 64)
    infinity = torch.tensor(1_000_000.0, device=probabilities.device)
    labels = torch.where(foreground, base, infinity)
    iterations = 0
    while iterations < max_iterations:
        checkpoint = labels.clone()
        steps = min(check_interval, max_iterations - iterations)
        for _ in range(steps):
            neighbor_minimum = -F.max_pool2d(
                -labels.unsqueeze(1), kernel_size=3, stride=1, padding=1
            ).squeeze(1)
            labels = torch.where(
                foreground, torch.minimum(labels, neighbor_minimum), infinity
            )
        iterations += steps
        if torch.equal(labels, checkpoint):
            break
    else:
        raise RuntimeError(
            f"GPU label propagation did not converge in {max_iterations} iterations."
        )
    labels = torch.where(foreground, labels, torch.zeros_like(labels))
    return labels.to(torch.int32).cpu().numpy(), iterations


def groups_from_component_maps(
    component_maps: np.ndarray,
    maximum_foreground: int,
) -> np.ndarray:
    return np.stack(
        [
            limit_groups(
                patch_component_ids(component_map),
                maximum_foreground=maximum_foreground,
            )
            for component_map in component_maps
        ]
    ).astype(np.int16)


@torch.no_grad()
def build_gpu_hybrid(
    probabilities: torch.Tensor,
    threshold: float,
    closing_iterations: int,
    token_count: int,
    check_interval: int,
    max_iterations: int,
) -> tuple[dict[str, np.ndarray], int]:
    if closing_iterations != 0:
        raise ValueError(
            "The exact GPU gate currently supports the frozen zero-closing setting only."
        )
    component_maps, iterations = gpu_connected_components(
        probabilities, threshold, check_interval, max_iterations
    )
    groups = groups_from_component_maps(
        component_maps, maximum_foreground=token_count - 1
    )
    assignments = np.stack(
        [
            oracle_assignment(group, token_count, bridge_aware=False)
            for group in groups
        ]
    ).astype(np.int16)
    adjacency, reachability = coarse_graph8_batch(
        assignments, groups, token_count
    )
    return {
        "assignment": assignments,
        "adjacency": adjacency,
        "reachability": reachability,
    }, iterations


def canonicalize_component_map(component_map: np.ndarray) -> np.ndarray:
    flat = component_map.reshape(-1)
    output = np.zeros(flat.shape, dtype=np.int32)
    positive = flat > 0
    values = np.unique(flat[positive])
    first_occurrence = sorted(
        ((int(np.flatnonzero(flat == value)[0]), int(value)) for value in values)
    )
    for new_label, (_, old_label) in enumerate(first_occurrence, start=1):
        output[flat == old_label] = new_label
    return output.reshape(component_map.shape)


def compare_artifacts(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, bool]:
    return {
        key: bool(np.array_equal(reference[key], candidate[key]))
        for key in ("assignment", "adjacency", "reachability")
    }


def artifacts_to_device(
    artifacts: dict[str, np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(artifacts["assignment"].astype(np.int64)).to(device),
        torch.from_numpy(artifacts["adjacency"].astype(np.float32)).to(device),
        torch.from_numpy(artifacts["reachability"].astype(np.float32)).to(device),
    )


@torch.no_grad()
def dynamic_masktopo_forward(
    model: StandardViTTokenBenchmark,
    images: torch.Tensor,
    assignment: torch.Tensor,
    adjacency: torch.Tensor,
    reachability: torch.Tensor,
) -> torch.Tensor:
    x = model.vit._process_input(images)
    batch_size = x.shape[0]
    cls = model.vit.class_token.expand(batch_size, -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = x + model.vit.encoder.pos_embedding
    x = model.vit.encoder.dropout(x)
    for layer_index, block in enumerate(model.vit.encoder.layers):
        x = block(x)
        if layer_index + 1 == model.insert_layer:
            cls, patch_tokens = x[:, :1], x[:, 1:]
            patch_tokens = model.pool(
                patch_tokens, assignment.reshape(batch_size, -1)
            )
            if model.graph is None:
                raise AssertionError("Dynamic MaskTopo model is missing GraphMixLayer.")
            patch_tokens = model.graph(patch_tokens, adjacency, reachability)
            x = torch.cat((cls, patch_tokens), dim=1)
    return model.vit.encoder.ln(x)[:, 0]


def profile_construction_stages(
    probabilities: np.ndarray,
    threshold: float,
    closing_iterations: int,
    token_count: int,
    warmup: int,
    repeats: int,
) -> dict[str, dict[str, float | int]]:
    structure = np.ones((3, 3), dtype=np.uint8)

    def component_maps_once() -> list[np.ndarray]:
        component_maps = []
        for probability in probabilities:
            binary = probability >= threshold
            if closing_iterations:
                binary = binary_closing(
                    binary,
                    structure=structure.astype(bool),
                    iterations=closing_iterations,
                )
            component_map, _ = scipy_label(binary, structure=structure)
            component_maps.append(component_map)
        return component_maps

    def groups_once(component_maps: list[np.ndarray]) -> list[np.ndarray]:
        return [
            limit_groups(
                patch_component_ids(component_map),
                maximum_foreground=token_count - 1,
            )
            for component_map in component_maps
        ]

    def assignments_once(groups: list[np.ndarray]) -> list[np.ndarray]:
        return [
            oracle_assignment(group, token_count, bridge_aware=False)
            for group in groups
        ]

    def run_once() -> tuple[list[np.ndarray], list[np.ndarray]]:
        component_maps = component_maps_once()
        groups = groups_once(component_maps)
        assignments = [
            oracle_assignment(group, token_count, bridge_aware=False)
            for group in groups
        ]
        return groups, assignments

    for _ in range(warmup):
        groups, assignments = run_once()
        for assignment, group in zip(assignments, groups):
            coarse_graph8(assignment, group, token_count, topology_aware=True)
        coarse_graph8_batch(
            np.stack(assignments), np.stack(groups), token_count
        )
    component_times: list[float] = []
    group_times: list[float] = []
    assignment_times: list[float] = []
    graph_reference_times: list[float] = []
    graph_batch_times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        component_maps = component_maps_once()
        component_times.append(1000.0 * (time.perf_counter() - started))
        started = time.perf_counter()
        groups = groups_once(component_maps)
        group_times.append(1000.0 * (time.perf_counter() - started))
        started = time.perf_counter()
        assignments = assignments_once(groups)
        assignment_times.append(1000.0 * (time.perf_counter() - started))
        started = time.perf_counter()
        for assignment, group in zip(assignments, groups):
            coarse_graph8(assignment, group, token_count, topology_aware=True)
        graph_reference_times.append(1000.0 * (time.perf_counter() - started))
        started = time.perf_counter()
        coarse_graph8_batch(
            np.stack(assignments), np.stack(groups), token_count
        )
        graph_batch_times.append(1000.0 * (time.perf_counter() - started))
    return {
        "threshold_and_connected_components": latency_summary(
            component_times, probabilities.shape[0]
        ),
        "patch_grouping_and_limit": latency_summary(
            group_times, probabilities.shape[0]
        ),
        "assignment": latency_summary(assignment_times, probabilities.shape[0]),
        "graph_and_reachability_reference": latency_summary(
            graph_reference_times, probabilities.shape[0]
        ),
        "graph_and_reachability_batch": latency_summary(
            graph_batch_times, probabilities.shape[0]
        ),
    }


def reduction_percent(full_ms: float, candidate_ms: float) -> float:
    return 100.0 * (full_ms - candidate_ms) / full_ms


def claim_gate(reduction: float) -> str:
    if reduction >= 30.0:
        return "SOLID_SUPPORTING_EFFICIENCY_CONTRIBUTION"
    if reduction >= 20.0:
        return "REPORT_EFFICIENCY_NOT_CORE"
    return "DO_NOT_MAKE_EFFICIENCY_CORE"


def run_self_test(device: torch.device) -> None:
    rng = np.random.default_rng(7)
    probabilities = rng.random((4, 64, 64), dtype=np.float32)
    probabilities[:, 20:44, 30:34] = 0.99
    threshold = 0.5
    reference = build_cpu_reference(probabilities, threshold, 0, 8)
    batched = build_cpu_batch(probabilities, threshold, 0, 8)
    assert all(compare_artifacts(reference, batched).values())
    gpu, _ = build_gpu_hybrid(
        torch.from_numpy(probabilities).to(device), threshold, 0, 8, 8, 4096
    )
    assert all(compare_artifacts(reference, gpu).values())
    print("TOPOLOGY_CONSTRUCTION_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.token_count != 8:
        raise ValueError("This frozen gate requires K=8.")
    if args.sample_count < max(args.batch_sizes):
        raise ValueError("sample-count must be at least the largest batch size.")
    if args.correctness_count > args.sample_count:
        raise ValueError("correctness-count cannot exceed sample-count.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.self_test:
        run_self_test(device)
        return
    if device.type != "cuda":
        raise RuntimeError("The frozen efficiency gate requires CUDA.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset_args = argparse.Namespace(
        dataset="fives",
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        train_size=2400,
        dev_size=600,
        test_size=1200,
        candidate_multiplier=3,
        crop_size=64,
        output_size=64,
        split_seed=20260731,
        data_seed=20260810,
    )
    arrays, manifest, _, fingerprint = load_all_arrays(dataset_args)
    current = {
        key: value[: args.sample_count] for key, value in arrays["test"].items()
    }
    with np.load(args.mask_run_dir / "mask_probabilities.npz") as archive:
        probabilities = archive["test"][: args.sample_count].astype(np.float32)
    frozen_result = json.loads(
        (args.mask_run_dir / "result.json").read_text(encoding="utf-8")
    )
    selected = frozen_result["selected_on_dev"]
    threshold = float(selected["mask_threshold"])
    closing_iterations = int(selected["closing_iterations"])
    if closing_iterations != 0:
        raise RuntimeError(
            "Frozen selection changed: the implemented exact GPU path assumes zero closing."
        )

    correctness_probabilities = probabilities[: args.correctness_count]
    print(
        f"CORRECTNESS_START samples={args.correctness_count} "
        f"threshold={threshold} closing={closing_iterations}",
        flush=True,
    )
    reference_correctness = build_cpu_reference(
        correctness_probabilities,
        threshold,
        closing_iterations,
        args.token_count,
    )
    batch_correctness = build_cpu_batch(
        correctness_probabilities,
        threshold,
        closing_iterations,
        args.token_count,
    )
    gpu_probability_tensor = torch.from_numpy(correctness_probabilities).to(device)
    gpu_component_maps, gpu_iterations = gpu_connected_components(
        gpu_probability_tensor,
        threshold,
        args.gpu_check_interval,
        args.gpu_max_iterations,
    )
    gpu_groups = groups_from_component_maps(
        gpu_component_maps, args.token_count - 1
    )
    gpu_correctness, _ = build_gpu_hybrid(
        gpu_probability_tensor,
        threshold,
        closing_iterations,
        args.token_count,
        args.gpu_check_interval,
        args.gpu_max_iterations,
    )
    cpu_component_maps = []
    cpu_groups = []
    binary_equal = True
    partition_equal = True
    for index, probability in enumerate(correctness_probabilities):
        component_map, _ = mask_groups(probability, threshold, closing_iterations)
        cpu_component_maps.append(component_map)
        cpu_group = predicted_groups(
            probability,
            threshold,
            closing_iterations,
            maximum_foreground=args.token_count - 1,
        )
        cpu_groups.append(cpu_group)
        binary_equal &= bool(
            np.array_equal(
                probability >= threshold,
                gpu_component_maps[index] > 0,
            )
        )
        partition_equal &= bool(
            np.array_equal(
                canonicalize_component_map(component_map),
                canonicalize_component_map(gpu_component_maps[index]),
            )
        )
    cpu_groups_array = np.stack(cpu_groups)
    correctness = {
        "sample_count": args.correctness_count,
        "binary_masks_equal": bool(binary_equal),
        "component_partitions_equal_after_canonicalization": bool(partition_equal),
        "patch_groups_equal": bool(np.array_equal(cpu_groups_array, gpu_groups)),
        "cpu_batch_artifacts": compare_artifacts(
            reference_correctness, batch_correctness
        ),
        "gpu_hybrid_artifacts": compare_artifacts(
            reference_correctness, gpu_correctness
        ),
        "gpu_label_propagation_iterations": int(gpu_iterations),
    }
    artifact_checks = [
        *correctness["cpu_batch_artifacts"].values(),
        *correctness["gpu_hybrid_artifacts"].values(),
    ]
    correctness_pass = bool(
        correctness["binary_masks_equal"]
        and correctness["component_partitions_equal_after_canonicalization"]
        and correctness["patch_groups_equal"]
        and all(artifact_checks)
    )
    correctness["artifact_gate_pass"] = correctness_pass
    if not correctness_pass:
        raise RuntimeError(
            "Correctness gate failed; timing was intentionally not started. "
            + json.dumps(correctness, ensure_ascii=False)
        )
    print(
        f"CORRECTNESS_PASS gpu_iterations={gpu_iterations}", flush=True
    )

    raw_images = torch.from_numpy(
        current["images"].astype(np.float32) / 255.0
    ).to(device)
    vit_images = F.interpolate(
        raw_images,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )
    mask_predictor = CrackUNet().to(device).eval()
    checkpoint = torch.load(
        args.mask_run_dir / "mask_predictor_seed20260810.pt",
        map_location=device,
        weights_only=False,
    )
    mask_predictor.load_state_dict(checkpoint["model"])

    print("CONSTRUCTION_STAGE_PROFILE_START", flush=True)
    stage_profiles = {
        str(batch_size): profile_construction_stages(
            probabilities[:batch_size],
            threshold,
            closing_iterations,
            args.token_count,
            args.warmup,
            args.repeats,
        )
        for batch_size in args.batch_sizes
    }
    print("CONSTRUCTION_STAGE_PROFILE_DONE", flush=True)

    construction_results: dict[str, Any] = {}
    component_results: dict[str, Any] = {}
    end_to_end_results: dict[str, Any] = {}
    downstream_checks: dict[str, Any] = {}

    for batch_size in args.batch_sizes:
        key = str(batch_size)
        print(f"BATCH_START batch={batch_size}", flush=True)
        batch_probabilities = probabilities[:batch_size]
        batch_probability_gpu = torch.from_numpy(batch_probabilities).to(device)
        batch_raw = raw_images[:batch_size]
        batch_vit = vit_images[:batch_size]
        cached_artifacts = build_cpu_reference(
            batch_probabilities,
            threshold,
            closing_iterations,
            args.token_count,
        )
        cached_assignment, cached_adjacency, cached_reachability = (
            artifacts_to_device(cached_artifacts, device)
        )

        construction_results[key] = {
            "cpu_reference": benchmark_callable(
                lambda: build_cpu_reference(
                    batch_probabilities,
                    threshold,
                    closing_iterations,
                    args.token_count,
                ),
                batch_size,
                args.warmup,
                args.repeats,
            ),
            "cpu_batch": benchmark_callable(
                lambda: build_cpu_batch(
                    batch_probabilities,
                    threshold,
                    closing_iterations,
                    args.token_count,
                ),
                batch_size,
                args.warmup,
                args.repeats,
            ),
            "gpu_hybrid": benchmark_callable(
                lambda: build_gpu_hybrid(
                    batch_probability_gpu,
                    threshold,
                    closing_iterations,
                    args.token_count,
                    args.gpu_check_interval,
                    args.gpu_max_iterations,
                ),
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "precomputed_lookup": benchmark_callable(
                lambda: (
                    cached_assignment,
                    cached_adjacency,
                    cached_reachability,
                ),
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
        }

        def mask_predictor_call() -> torch.Tensor:
            return torch.sigmoid(mask_predictor(batch_raw[:, :1]))

        component_results[key] = {
            "mask_predictor": benchmark_callable(
                mask_predictor_call,
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "token_pooling": None,
            "artifact_transfer": None,
            "gpu_connected_components_including_map_transfer": benchmark_callable(
                lambda: gpu_connected_components(
                    batch_probability_gpu,
                    threshold,
                    args.gpu_check_interval,
                    args.gpu_max_iterations,
                ),
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "vit_full": None,
            "vit_grid_compressed": None,
            "vit_masktopo_compressed": None,
        }

        model_masktopo = StandardViTTokenBenchmark(
            args.token_count,
            args.insert_layer,
            "masktopo_cached",
            cached_artifacts["assignment"],
            cached_artifacts["adjacency"],
            cached_artifacts["reachability"],
        ).to(device).eval()
        model_full = StandardViTTokenBenchmark(
            args.token_count,
            args.insert_layer,
            "full",
            None,
            None,
            None,
        ).to(device).eval()
        model_grid = StandardViTTokenBenchmark(
            args.token_count,
            args.insert_layer,
            "grid",
            None,
            None,
            None,
        ).to(device).eval()
        patch_tokens = torch.randn(batch_size, 256, 768, device=device)
        component_results[key]["token_pooling"] = benchmark_callable(
            lambda: model_masktopo.pool(
                patch_tokens, cached_assignment.reshape(batch_size, -1)
            ),
            batch_size,
            args.warmup,
            args.repeats,
            device,
        )
        component_results[key]["artifact_transfer"] = benchmark_callable(
            lambda: artifacts_to_device(cached_artifacts, device),
            batch_size,
            args.warmup,
            args.repeats,
            device,
        )
        component_results[key]["vit_full"] = benchmark_callable(
            lambda: model_full(batch_vit),
            batch_size,
            args.warmup,
            args.repeats,
            device,
        )
        component_results[key]["vit_grid_compressed"] = benchmark_callable(
            lambda: model_grid(batch_vit),
            batch_size,
            args.warmup,
            args.repeats,
            device,
        )
        component_results[key]["vit_masktopo_compressed"] = benchmark_callable(
            lambda: dynamic_masktopo_forward(
                model_masktopo,
                batch_vit,
                cached_assignment,
                cached_adjacency,
                cached_reachability,
            ),
            batch_size,
            args.warmup,
            args.repeats,
            device,
        )

        with torch.no_grad():
            reference_output = dynamic_masktopo_forward(
                model_masktopo,
                batch_vit,
                cached_assignment,
                cached_adjacency,
                cached_reachability,
            )
            batch_artifacts_for_check = build_cpu_batch(
                batch_probabilities,
                threshold,
                closing_iterations,
                args.token_count,
            )
            batch_tensors_for_check = artifacts_to_device(
                batch_artifacts_for_check, device
            )
            batch_output = dynamic_masktopo_forward(
                model_masktopo, batch_vit, *batch_tensors_for_check
            )
            gpu_artifacts_for_check, _ = build_gpu_hybrid(
                batch_probability_gpu,
                threshold,
                closing_iterations,
                args.token_count,
                args.gpu_check_interval,
                args.gpu_max_iterations,
            )
            gpu_tensors_for_check = artifacts_to_device(
                gpu_artifacts_for_check, device
            )
            gpu_output = dynamic_masktopo_forward(
                model_masktopo, batch_vit, *gpu_tensors_for_check
            )
        downstream_checks[key] = {
            "cpu_batch_allclose": bool(
                torch.allclose(reference_output, batch_output, atol=1e-6, rtol=1e-5)
            ),
            "gpu_hybrid_allclose": bool(
                torch.allclose(reference_output, gpu_output, atol=1e-6, rtol=1e-5)
            ),
            "cpu_batch_max_abs_difference": float(
                (reference_output - batch_output).abs().max().item()
            ),
            "gpu_hybrid_max_abs_difference": float(
                (reference_output - gpu_output).abs().max().item()
            ),
        }

        def e2e_cpu(builder: Callable[..., dict[str, np.ndarray]]) -> torch.Tensor:
            predicted = torch.sigmoid(mask_predictor(batch_raw[:, :1]))
            synchronize(device)
            predicted_numpy = predicted.detach().cpu().numpy()
            artifacts = builder(
                predicted_numpy,
                threshold,
                closing_iterations,
                args.token_count,
            )
            tensors = artifacts_to_device(artifacts, device)
            return dynamic_masktopo_forward(model_masktopo, batch_vit, *tensors)

        def e2e_gpu_hybrid() -> torch.Tensor:
            predicted = torch.sigmoid(mask_predictor(batch_raw[:, :1])).squeeze(1)
            artifacts, _ = build_gpu_hybrid(
                predicted,
                threshold,
                closing_iterations,
                args.token_count,
                args.gpu_check_interval,
                args.gpu_max_iterations,
            )
            tensors = artifacts_to_device(artifacts, device)
            return dynamic_masktopo_forward(model_masktopo, batch_vit, *tensors)

        def e2e_precomputed() -> torch.Tensor:
            return dynamic_masktopo_forward(
                model_masktopo,
                batch_vit,
                cached_assignment,
                cached_adjacency,
                cached_reachability,
            )

        def e2e_mask_only() -> torch.Tensor:
            _ = torch.sigmoid(mask_predictor(batch_raw[:, :1]))
            return model_grid(batch_vit)

        end_to_end_results[key] = {
            "full_vit": benchmark_callable(
                lambda: model_full(batch_vit),
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "masktopo_cpu_reference": benchmark_callable(
                lambda: e2e_cpu(build_cpu_reference),
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "masktopo_cpu_batch": benchmark_callable(
                lambda: e2e_cpu(build_cpu_batch),
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "masktopo_gpu_hybrid": benchmark_callable(
                e2e_gpu_hybrid,
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "precomputed_upper_bound": benchmark_callable(
                e2e_precomputed,
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "mask_only_no_graph": benchmark_callable(
                e2e_mask_only,
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
        }
        del model_masktopo, model_full, model_grid, patch_tokens
        torch.cuda.empty_cache()
        print(f"BATCH_DONE batch={batch_size}", flush=True)

    downstream_gate = all(
        item["cpu_batch_allclose"] and item["gpu_hybrid_allclose"]
        for item in downstream_checks.values()
    )
    correctness["downstream_output_gate_pass"] = bool(downstream_gate)
    if not downstream_gate:
        raise RuntimeError("Downstream-output equivalence gate failed.")

    gates: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        key = str(batch_size)
        full_ms = float(
            end_to_end_results[key]["full_vit"]["mean_ms_per_sample"]
        )
        online_modes = (
            "masktopo_cpu_reference",
            "masktopo_cpu_batch",
            "masktopo_gpu_hybrid",
        )
        reductions = {
            mode: reduction_percent(
                full_ms,
                float(end_to_end_results[key][mode]["mean_ms_per_sample"]),
            )
            for mode in online_modes
        }
        best_mode = max(reductions, key=reductions.__getitem__)
        gates[key] = {
            "full_vit_mean_ms_per_sample": full_ms,
            "online_reduction_percent": reductions,
            "best_online_mode": best_mode,
            "best_online_reduction_percent": float(reductions[best_mode]),
            "claim_gate": claim_gate(float(reductions[best_mode])),
            "gpu_beats_cpu_batch": bool(
                end_to_end_results[key]["masktopo_gpu_hybrid"][
                    "mean_ms_per_sample"
                ]
                < end_to_end_results[key]["masktopo_cpu_batch"][
                    "mean_ms_per_sample"
                ]
            ),
        }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_id": "TB-B-260731-004",
        "status": "completed",
        "purpose": "topology-construction optimization and end-to-end efficiency gate",
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "dataset": {
            "name": "FIVES",
            "fingerprint": fingerprint,
            "official_source_counts": {
                split: len(values) for split, values in manifest.items()
            },
            "sample_count": args.sample_count,
            "correctness_count": args.correctness_count,
        },
        "frozen_configuration": {
            "mask_run_dir": str(args.mask_run_dir.resolve()),
            "mask_threshold": threshold,
            "closing_iterations": closing_iterations,
            "backbone": "torchvision ViT-B/16, random initialization",
            "input_size": 256,
            "insert_layer": args.insert_layer,
            "fine_patch_tokens": 256,
            "reduced_patch_tokens": args.token_count,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "correctness": correctness,
        "downstream_checks": downstream_checks,
        "construction_stage_profiles": stage_profiles,
        "construction": construction_results,
        "components": component_results,
        "end_to_end": end_to_end_results,
        "gates": gates,
        "limitations": [
            "The ViT backbone is randomly initialized; efficiency only, not accuracy.",
            "The precomputed path is an offline upper bound, not online inference.",
            "The GPU path accelerates only connected-component propagation; fixed-K assignment remains on CPU.",
            "Microbenchmarks repeat fixed held-out batches and exclude dataset I/O and process start-up.",
        ],
        "protocol": str((Path(__file__).parent / PROTOCOL_NAME).resolve()),
    }
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# TopoBridge topology-construction benchmark",
        "",
        "All optimized paths passed exact artifact parity and downstream-output parity before timing.",
        "The ViT backbone is randomly initialized, so this report supports efficiency only.",
        "",
        "## End-to-end latency",
        "",
        "| Batch | Full ViT | CPU reference | CPU batch | GPU hybrid | Precomputed upper bound | Mask-only | Best online reduction | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for batch_size in args.batch_sizes:
        key = str(batch_size)
        values = end_to_end_results[key]
        gate = gates[key]
        lines.append(
            f"| {batch_size} | "
            f"{values['full_vit']['mean_ms_per_sample']:.3f} | "
            f"{values['masktopo_cpu_reference']['mean_ms_per_sample']:.3f} | "
            f"{values['masktopo_cpu_batch']['mean_ms_per_sample']:.3f} | "
            f"{values['masktopo_gpu_hybrid']['mean_ms_per_sample']:.3f} | "
            f"{values['precomputed_upper_bound']['mean_ms_per_sample']:.3f} | "
            f"{values['mask_only_no_graph']['mean_ms_per_sample']:.3f} | "
            f"{gate['best_online_reduction_percent']:.2f}% | "
            f"{gate['claim_gate']} |"
        )
    lines.extend(
        [
            "",
            "All latency cells are synchronized mean milliseconds per sample. Median and P95 values are stored in `result.json`.",
            "",
            "## Reference CPU stage profile (batch 32)",
            "",
            "| Stage | Mean ms/sample | Median ms/sample | P95 ms/sample |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in stage_profiles[str(max(args.batch_sizes))].items():
        lines.append(
            f"| {name} | {values['mean_ms_per_sample']:.3f} | "
            f"{values['median_ms_per_sample']:.3f} | "
            f"{values['p95_ms_per_sample']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Correctness",
            "",
            f"- Samples: {correctness['sample_count']}",
            f"- Binary masks equal: {correctness['binary_masks_equal']}",
            f"- Component partitions equal: {correctness['component_partitions_equal_after_canonicalization']}",
            f"- Patch groups equal: {correctness['patch_groups_equal']}",
            f"- Artifact gate: {correctness['artifact_gate_pass']}",
            f"- Downstream output gate: {correctness['downstream_output_gate_pass']}",
            f"- GPU propagation iterations: {correctness['gpu_label_propagation_iterations']}",
            "",
            "## Interpretation constraints",
            "",
            "- Precomputed assignment is reported only as an offline upper bound.",
            "- Mask-only is an efficiency ablation and needs separately trained accuracy evidence.",
            "- GPU hybrid still performs fixed-K assignment on CPU.",
            "- Random initialization prohibits any task-accuracy conclusion.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "completed", "gates": gates}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

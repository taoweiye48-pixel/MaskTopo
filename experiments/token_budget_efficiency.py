from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from crackforest_mask_topo import CrackUNet, find_cached_split
from crackforest_mechanism_ablation import build_artifacts
from deepcrack_external_gate import load_all_arrays
from strong_reducer_baselines import REDUCERS, make_model
from topocoarsen_oracle import TopoCoarsenModel


BUDGETS = (8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Architecture-level token-budget efficiency audit."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_token_budget_efficiency"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark(
    call: Callable[[], torch.Tensor],
    batch_size: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | None]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        call()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        call()
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "batch_size": batch_size,
        "latency_ms_per_batch": 1000.0 * elapsed / repeats,
        "latency_ms_per_sample": (
            1000.0 * elapsed / repeats / batch_size
        ),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else None
        ),
    }


def counted_flops(call: Callable[[], torch.Tensor]) -> int:
    with profile(
        activities=[ProfilerActivity.CPU],
        with_flops=True,
        record_shapes=False,
    ) as profiler:
        with torch.no_grad():
            call()
    return int(
        sum(int(event.flops or 0) for event in profiler.key_averages())
    )


def synthetic_assignment(token_count: int, batch_size: int) -> torch.Tensor:
    flat = torch.div(
        torch.arange(256) * token_count,
        256,
        rounding_mode="floor",
    )
    if torch.unique(flat).numel() != token_count:
        raise AssertionError("Synthetic assignment has empty tokens.")
    return flat.reshape(1, 16, 16).expand(batch_size, -1, -1)


def reducer_efficiency(
    name: str,
    token_count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model = make_model(name, args.dim, token_count).to(device).eval()
    image = torch.rand(args.batch_size, 3, 64, 64, device=device)
    mask = torch.rand(args.batch_size, 1, 64, 64, device=device)
    runtime = benchmark(
        lambda: model(image, mask),
        args.batch_size,
        device,
        args.warmup,
        args.repeats,
    )
    cpu_model = make_model(name, args.dim, token_count).cpu().eval()
    cpu_image = torch.rand(1, 3, 64, 64)
    cpu_mask = torch.rand(1, 1, 64, 64)
    flops = counted_flops(lambda: cpu_model(cpu_image, cpu_mask))
    return {
        "model": name,
        "token_count": token_count,
        "parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "torch_profiler_counted_flops_batch1": flops,
        "profiler_flop_scope": (
            "partial operator count; unsupported operators may report zero"
        ),
        "connector_runtime": runtime,
        "requires_mask_predictor": name == "mask_guided_queries",
        "requires_cpu_topology_construction": False,
    }


def masktopo_efficiency(
    token_count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    name = f"mask_topo_budget_k{token_count}"
    model = TopoCoarsenModel(name, args.dim, token_count).to(device).eval()
    image = torch.rand(args.batch_size, 3, 64, 64, device=device)
    assignment = synthetic_assignment(
        token_count, args.batch_size
    ).to(device)
    identity = torch.eye(
        token_count, dtype=torch.int64, device=device
    ).unsqueeze(0).expand(args.batch_size, -1, -1)
    runtime = benchmark(
        lambda: model(image, assignment, identity, identity),
        args.batch_size,
        device,
        args.warmup,
        args.repeats,
    )
    cpu_model = TopoCoarsenModel(name, args.dim, token_count).cpu().eval()
    cpu_image = torch.rand(1, 3, 64, 64)
    cpu_assignment = synthetic_assignment(token_count, 1)
    cpu_identity = torch.eye(
        token_count, dtype=torch.int64
    ).unsqueeze(0)
    flops = counted_flops(
        lambda: cpu_model(
            cpu_image, cpu_assignment, cpu_identity, cpu_identity
        )
    )
    return {
        "model": "mask_topo",
        "token_count": token_count,
        "parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "torch_profiler_counted_flops_batch1": flops,
        "profiler_flop_scope": (
            "partial operator count; unsupported operators may report zero"
        ),
        "connector_runtime": runtime,
        "requires_mask_predictor": True,
        "requires_cpu_topology_construction": True,
    }


def mask_predictor_efficiency(
    args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    model = CrackUNet().to(device).eval()
    image = torch.rand(args.batch_size, 1, 64, 64, device=device)
    runtime = benchmark(
        lambda: model(image),
        args.batch_size,
        device,
        args.warmup,
        args.repeats,
    )
    cpu_model = CrackUNet().cpu().eval()
    cpu_image = torch.rand(1, 1, 64, 64)
    flops = counted_flops(lambda: cpu_model(cpu_image))
    return {
        "model": "crack_unet_mask_predictor",
        "parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "torch_profiler_counted_flops_batch1": flops,
        "profiler_flop_scope": (
            "partial operator count; unsupported operators may report zero"
        ),
        "runtime": runtime,
    }


def real_topology_preprocessing(root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in ("crackforest", "deepcrack"):
        if dataset == "crackforest":
            arrays = find_cached_split(
                (root / "real_cache").resolve(),
                "test",
                320,
                22260810,
                64,
            )
            prior_root = (
                root / "results_crackforest_mechanism_seed20260810"
            )
            selected_key = "selected_on_prior_dev"
        else:
            namespace = SimpleNamespace(
                dataset_root=(
                    root / "real_data/DeepCrack/dataset/extracted"
                ),
                cache_dir=root / "deepcrack_cache",
                train_size=2400,
                dev_size=600,
                test_size=1200,
                candidate_multiplier=3,
                crop_size=64,
                output_size=64,
                min_component_pixels=12,
                data_seed=20260810,
                split_seed=20260730,
            )
            arrays = load_all_arrays(namespace)[0]["test"]
            prior_root = root / "results_deepcrack_seed20260810"
            selected_key = "selected_on_dev"
        prior = json.loads(
            (prior_root / "result.json").read_text(encoding="utf-8")
        )
        selected = prior[selected_key]
        with np.load(prior_root / "mask_probabilities.npz") as archive:
            probabilities = archive["test"].astype(np.float32)
        for token_count in BUDGETS:
            started = time.perf_counter()
            _, diagnostics = build_artifacts(
                arrays,
                probabilities,
                float(selected["mask_threshold"]),
                int(selected["closing_iterations"]),
                token_count,
                "mask_topo_recheck",
                20260731,
            )
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "dataset": dataset,
                    "token_count": token_count,
                    "sample_count": int(arrays["images"].shape[0]),
                    "seconds": elapsed,
                    "milliseconds_per_sample": (
                        1000.0 * elapsed / arrays["images"].shape[0]
                    ),
                    "direct_structural_balanced_accuracy": diagnostics[
                        "direct_structural_metrics"
                    ]["balanced_accuracy"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for token_count in BUDGETS:
        for name in REDUCERS:
            print(
                f"EFFICIENCY_START model={name} k={token_count}",
                flush=True,
            )
            rows.append(
                reducer_efficiency(name, token_count, args, device)
            )
        print(
            f"EFFICIENCY_START model=mask_topo k={token_count}",
            flush=True,
        )
        rows.append(
            masktopo_efficiency(token_count, args, device)
        )
    print("EFFICIENCY_START model=mask_predictor", flush=True)
    mask_predictor = mask_predictor_efficiency(args, device)
    print("EFFICIENCY_START real_topology_preprocessing", flush=True)
    topology_preprocessing = real_topology_preprocessing(
        args.root.resolve()
    )
    result = {
        "experiment_id": "topobridge_token_budget_efficiency",
        "status": "completed",
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": rows,
        "mask_predictor": mask_predictor,
        "topology_preprocessing": topology_preprocessing,
        "latency_scope": (
            "architecture-level synthetic-input benchmark; connector runtime "
            "excludes data loading and, unless separately added, mask prediction "
            "and CPU topology construction"
        ),
        "protocol": str(
            (Path(__file__).parent / "TOKEN_BUDGET_PROTOCOL.md").resolve()
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

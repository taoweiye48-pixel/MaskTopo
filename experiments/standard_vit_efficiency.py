from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16

from crackforest_mask_topo import CrackUNet
from fives_external_gate import grid_artifacts, load_all_arrays
from fives_g2tm_collision import (
    grid_edges,
    maximum_spanning_forest_assignments,
)
from topocoarsen_oracle import GraphMixLayer


MODES = ("full", "grid", "g2tm_fixedk", "masktopo_cached")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standard ViT-B/16 token compression efficiency benchmark."
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
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--insert-layer", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def rectangular_assignment(token_count: int) -> np.ndarray:
    pairs = [
        (rows, token_count // rows)
        for rows in range(1, int(np.sqrt(token_count)) + 1)
        if token_count % rows == 0
        and 16 % rows == 0
        and 16 % (token_count // rows) == 0
    ]
    if not pairs:
        raise ValueError("K must have factors dividing the 16x16 patch grid.")
    rows, columns = max(pairs, key=lambda item: item[0])
    base = np.arange(token_count, dtype=np.int64).reshape(rows, columns)
    return base.repeat(16 // rows, axis=0).repeat(16 // columns, axis=1)


class StandardViTTokenBenchmark(nn.Module):
    def __init__(
        self,
        token_count: int,
        insert_layer: int,
        mode: str,
        mask_assignment: np.ndarray | None,
        mask_adjacency: np.ndarray | None,
        mask_reachability: np.ndarray | None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.token_count = token_count
        self.insert_layer = insert_layer
        self.vit = vit_b_16(weights=None, image_size=256)
        self.vit.eval()
        self.graph = (
            GraphMixLayer(768) if mode == "masktopo_cached" else None
        )
        self.register_buffer(
            "grid_assignment",
            torch.from_numpy(rectangular_assignment(token_count)),
            persistent=False,
        )
        source, target = grid_edges()
        self.register_buffer("edge_source", source, persistent=False)
        self.register_buffer("edge_target", target, persistent=False)
        if mask_assignment is not None:
            self.register_buffer(
                "mask_assignment",
                torch.from_numpy(mask_assignment.astype(np.int64)),
                persistent=False,
            )
            self.register_buffer(
                "mask_adjacency",
                torch.from_numpy(mask_adjacency.astype(np.float32)),
                persistent=False,
            )
            self.register_buffer(
                "mask_reachability",
                torch.from_numpy(mask_reachability.astype(np.float32)),
                persistent=False,
            )
        else:
            self.mask_assignment = None
            self.mask_adjacency = None
            self.mask_reachability = None

    def pool(
        self,
        patch_tokens: torch.Tensor,
        assignment: torch.Tensor,
    ) -> torch.Tensor:
        one_hot = F.one_hot(
            assignment,
            num_classes=self.token_count,
        ).to(patch_tokens.dtype)
        counts = one_hot.sum(dim=1).clamp_min(1.0)
        return torch.bmm(one_hot.transpose(1, 2), patch_tokens) / counts.unsqueeze(-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.vit._process_input(images)
        batch_size = x.shape[0]
        cls = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        for layer_index, block in enumerate(self.vit.encoder.layers):
            x = block(x)
            if (
                self.mode != "full"
                and layer_index + 1 == self.insert_layer
            ):
                cls, patch_tokens = x[:, :1], x[:, 1:]
                if self.mode == "grid":
                    assignment = self.grid_assignment.reshape(-1).unsqueeze(0).expand(
                        batch_size, -1
                    )
                elif self.mode == "g2tm_fixedk":
                    assignment = maximum_spanning_forest_assignments(
                        patch_tokens,
                        self.token_count,
                        self.edge_source,
                        self.edge_target,
                    )
                elif self.mode == "masktopo_cached":
                    if self.mask_assignment is None:
                        raise AssertionError("Missing cached MaskTopo assignment.")
                    assignment = self.mask_assignment[:batch_size].reshape(
                        batch_size, -1
                    ).to(patch_tokens.device)
                else:
                    raise ValueError(self.mode)
                patch_tokens = self.pool(patch_tokens, assignment)
                if self.graph is not None:
                    adjacency = self.mask_adjacency[:batch_size].to(
                        patch_tokens.device
                    )
                    reachability = self.mask_reachability[:batch_size].to(
                        patch_tokens.device
                    )
                    patch_tokens = self.graph(
                        patch_tokens, adjacency, reachability
                    )
                x = torch.cat((cls, patch_tokens), dim=1)
        return self.vit.encoder.ln(x)[:, 0]


@torch.no_grad()
def benchmark(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | None]:
    model.eval()
    images = images.to(device)
    for _ in range(warmup):
        model(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(repeats):
        model(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "batch_size": int(images.shape[0]),
        "repeats": repeats,
        "latency_ms_per_batch": 1000 * elapsed / repeats,
        "latency_ms_per_sample": 1000 * elapsed / repeats / images.shape[0],
        "throughput_samples_per_second": repeats * images.shape[0] / elapsed,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else None
        ),
    }


@torch.no_grad()
def benchmark_mask_predictor(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | None]:
    model.eval()
    image = images[:, :1].to(device)
    for _ in range(warmup):
        model(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        model(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "batch_size": int(image.shape[0]),
        "repeats": repeats,
        "latency_ms_per_batch": 1000 * elapsed / repeats,
        "latency_ms_per_sample": 1000 * elapsed / repeats / image.shape[0],
        "throughput_samples_per_second": repeats * image.shape[0] / elapsed,
    }


def run_self_test() -> None:
    torch.manual_seed(1)
    assignment = rectangular_assignment(8)
    assert assignment.shape == (16, 16)
    model = StandardViTTokenBenchmark(
        8,
        2,
        "grid",
        None,
        None,
        None,
    )
    images = torch.rand(1, 3, 256, 256)
    output = model(images)
    assert output.shape == (1, 768)
    print("STANDARD_VIT_EFFICIENCY_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.output is None:
        raise ValueError("--output is required.")
    if args.token_count != 8:
        raise ValueError("The first standard-ViT gate is frozen at K=8.")
    if not 1 <= args.insert_layer <= 12:
        raise ValueError("insert-layer must be between 1 and 12.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.dataset = "fives"
    arrays, manifest, _, fingerprint = load_all_arrays(
        argparse.Namespace(
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
    )
    current = {
        key: value[: args.sample_count]
        for key, value in arrays["test"].items()
    }
    with np.load(args.mask_run_dir / "mask_probabilities.npz") as archive:
        probabilities = archive["test"][: args.sample_count].astype(np.float32)
    result_source = json.loads(
        (args.mask_run_dir / "result.json").read_text(encoding="utf-8")
    )
    selected = result_source["selected_on_dev"]
    # Replace the rectangular grid artifacts with predicted-mask topology.
    from crackforest_mechanism_ablation import build_artifacts

    mask_build_started = time.perf_counter()
    mask_artifacts, mask_diagnostics = build_artifacts(
        current,
        probabilities,
        float(selected["mask_threshold"]),
        int(selected["closing_iterations"]),
        args.token_count,
        "mask_topo_recheck",
        20260731,
    )
    mask_build_seconds = time.perf_counter() - mask_build_started
    raw_images = torch.from_numpy(
        current["images"].astype(np.float32) / 255.0
    )
    images = raw_images
    images = F.interpolate(
        images,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_predictor = CrackUNet().to(device)
    mask_checkpoint = torch.load(
        args.mask_run_dir / "mask_predictor_seed20260810.pt",
        map_location=device,
        weights_only=False,
    )
    mask_predictor.load_state_dict(mask_checkpoint["model"])
    mask_predictor_stats = benchmark_mask_predictor(
        mask_predictor,
        raw_images[: args.batch_size],
        device,
        args.warmup,
        args.repeats,
    )
    del mask_predictor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    summaries: dict[str, Any] = {}
    for mode in MODES:
        print(f"STANDARD_VIT_START mode={mode}", flush=True)
        model = StandardViTTokenBenchmark(
            args.token_count,
            args.insert_layer,
            mode,
            mask_artifacts["assignment"] if mode == "masktopo_cached" else None,
            mask_artifacts["adjacency"] if mode == "masktopo_cached" else None,
            mask_artifacts["reachability"] if mode == "masktopo_cached" else None,
        ).to(device)
        summaries[mode] = benchmark(
            model,
            images[: args.batch_size],
            device,
            args.warmup,
            args.repeats,
        )
        summaries[mode]["fine_tokens"] = 256
        summaries[mode]["reduced_tokens"] = (
            256 if mode == "full" else args.token_count
        )
        summaries[mode]["insert_layer"] = args.insert_layer
        print(
            f"STANDARD_VIT_DONE mode={mode} "
            f"latency_ms_sample={summaries[mode]['latency_ms_per_sample']:.3f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_id": "standard_vit_b16_token_efficiency",
        "status": "completed",
        "dataset": "FIVES",
        "dataset_fingerprint": fingerprint,
        "official_source_counts": {
            split: len(values) for split, values in manifest.items()
        },
        "sample_count": args.sample_count,
        "input_size": 256,
        "backbone": "torchvision ViT-B/16, random initialization",
        "insert_layer": args.insert_layer,
        "token_count": args.token_count,
        "mask_topo_artifact_diagnostics": mask_diagnostics,
        "mask_predictor_benchmark": mask_predictor_stats,
        "mask_artifact_build": {
            "sample_count": args.sample_count,
            "total_seconds": mask_build_seconds,
            "ms_per_sample": 1000 * mask_build_seconds / args.sample_count,
        },
        "modes": summaries,
        "masktopo_estimated_end_to_end_ms_per_sample": (
            summaries["masktopo_cached"]["latency_ms_per_sample"]
            + mask_predictor_stats["latency_ms_per_sample"]
            + 1000 * mask_build_seconds / args.sample_count
        ),
        "interpretation": (
            "This is an efficiency-only standard-backbone benchmark. The "
            "backbone is randomly initialized, so no accuracy claim is made. "
            "A later trained ViT experiment is required for the main paper."
        ),
        "protocol": str(
            (Path(__file__).parent / "STANDARD_VIT_EFFICIENCY_PROTOCOL.md")
            .resolve()
        ),
    }
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Standard ViT-B/16 token efficiency benchmark",
        "",
        "Randomly initialized backbone; this report supports efficiency only, "
        "not accuracy.",
        "",
        "| Mode | Tokens after layer 2 | Latency (ms/sample) | Throughput (sample/s) | Peak memory (MB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, values in summaries.items():
        lines.append(
            f"| {mode} | {values['reduced_tokens']} | "
            f"{values['latency_ms_per_sample']:.3f} | "
            f"{values['throughput_samples_per_second']:.2f} | "
            f"{values['peak_cuda_memory_mb'] if values['peak_cuda_memory_mb'] is not None else 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## MaskTopo overhead excluded from cached token benchmark",
            "",
            f"- Mask predictor: "
            f"{mask_predictor_stats['latency_ms_per_sample']:.3f} ms/sample",
            f"- CPU mask-to-artifact construction: "
            f"{1000 * mask_build_seconds / args.sample_count:.3f} ms/sample",
            f"- Estimated end-to-end MaskTopo: "
            f"{report['masktopo_estimated_end_to_end_ms_per_sample']:.3f} ms/sample",
        ]
    )
    lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

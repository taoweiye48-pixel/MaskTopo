from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label

from crackforest_mask_topo import CrackUNet
from standard_vit_efficiency import StandardViTTokenBenchmark
from topology_construction_benchmark import (
    artifacts_to_device,
    benchmark_callable,
    dynamic_masktopo_forward,
)
from topology_postprocess_optimization import (
    artifact_equal,
    build_hybrid,
    build_reference_from_components,
    load_inputs,
    run_vit_output_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end gate for optimized TopoBridge post-processing."
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
        default=Path("results_topology_postprocess_e2e"),
    )
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--correctness-count", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--insert-layer", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def components_from_probabilities(
    probabilities: np.ndarray, threshold: float
) -> np.ndarray:
    structure = np.ones((3, 3), dtype=np.uint8)
    return np.stack(
        [
            scipy_label(probability >= threshold, structure=structure)[0]
            for probability in probabilities
        ]
    ).astype(np.int32)


def build_reference_from_probabilities(
    probabilities: np.ndarray, threshold: float, token_count: int
) -> dict[str, np.ndarray]:
    return build_reference_from_components(
        components_from_probabilities(probabilities, threshold), token_count
    )


def build_hybrid_from_probabilities(
    probabilities: np.ndarray, threshold: float, token_count: int
) -> dict[str, np.ndarray]:
    return build_hybrid(
        components_from_probabilities(probabilities, threshold), token_count
    )


def reduction_percent(full_ms: float, candidate_ms: float) -> float:
    return 100.0 * (full_ms - candidate_ms) / full_ms


def claim_gate(reduction: float) -> str:
    if reduction >= 30.0:
        return "SOLID_SUPPORTING_EFFICIENCY_CONTRIBUTION"
    if reduction >= 20.0:
        return "REPORT_EFFICIENCY_NOT_CORE"
    return "DO_NOT_MAKE_EFFICIENCY_CORE"


def main() -> None:
    args = parse_args()
    if args.token_count != 8:
        raise ValueError("This end-to-end gate is frozen at K=8.")
    if args.sample_count < max(args.batch_sizes):
        raise ValueError("sample-count must cover the largest batch.")
    if args.correctness_count > args.sample_count:
        raise ValueError("correctness-count cannot exceed sample-count.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The frozen end-to-end gate requires CUDA.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    probabilities, images, threshold = load_inputs(args)

    print(f"CORRECTNESS_START samples={args.correctness_count}", flush=True)
    correctness_probabilities = probabilities[: args.correctness_count]
    reference_correctness = build_reference_from_probabilities(
        correctness_probabilities, threshold, args.token_count
    )
    hybrid_correctness = build_hybrid_from_probabilities(
        correctness_probabilities, threshold, args.token_count
    )
    artifact_checks = artifact_equal(reference_correctness, hybrid_correctness)
    vit_gate = run_vit_output_gate(
        images[: args.correctness_count],
        reference_correctness,
        hybrid_correctness,
        min(8, args.correctness_count),
    )
    correctness = {
        "sample_count": args.correctness_count,
        "artifact_equality": artifact_checks,
        "vit_output_gate": vit_gate,
        "gate_pass": bool(all(artifact_checks.values()) and vit_gate["allclose"]),
    }
    if not correctness["gate_pass"]:
        raise RuntimeError(
            "E2E_CORRECTNESS_GATE_FAILED "
            + json.dumps(correctness, ensure_ascii=False)
        )
    print("CORRECTNESS_PASS", flush=True)

    raw_images = torch.from_numpy(images).to(device)
    vit_images = F.interpolate(
        raw_images, size=(256, 256), mode="bilinear", align_corners=False
    )
    mask_predictor = CrackUNet().to(device).eval()
    checkpoint = torch.load(
        args.mask_run_dir / "mask_predictor_seed20260810.pt",
        map_location=device,
        weights_only=False,
    )
    mask_predictor.load_state_dict(checkpoint["model"])

    results: dict[str, Any] = {}
    components: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        key = str(batch_size)
        batch_probabilities = probabilities[:batch_size]
        batch_raw = raw_images[:batch_size]
        batch_vit = vit_images[:batch_size]
        cached_hybrid = build_hybrid_from_probabilities(
            batch_probabilities, threshold, args.token_count
        )
        assignment, adjacency, reachability = artifacts_to_device(
            cached_hybrid, device
        )
        model_masktopo = StandardViTTokenBenchmark(
            args.token_count,
            args.insert_layer,
            "masktopo_cached",
            cached_hybrid["assignment"],
            cached_hybrid["adjacency"],
            cached_hybrid["reachability"],
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

        def mask_predictor_call() -> torch.Tensor:
            return torch.sigmoid(mask_predictor(batch_raw[:, :1]))

        def e2e_builder(builder: Any) -> torch.Tensor:
            predicted = torch.sigmoid(mask_predictor(batch_raw[:, :1]))
            predicted_numpy = predicted.detach().cpu().numpy()
            artifacts = builder(
                predicted_numpy, threshold, args.token_count
            )
            tensors = artifacts_to_device(artifacts, device)
            return dynamic_masktopo_forward(
                model_masktopo, batch_vit, *tensors
            )

        def precomputed_call() -> torch.Tensor:
            return dynamic_masktopo_forward(
                model_masktopo,
                batch_vit,
                assignment,
                adjacency,
                reachability,
            )

        def mask_only_call() -> torch.Tensor:
            _ = torch.sigmoid(mask_predictor(batch_raw[:, :1]))
            return model_grid(batch_vit)

        components[key] = {
            "mask_predictor": benchmark_callable(
                mask_predictor_call,
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "reference_construction": benchmark_callable(
                lambda: build_reference_from_probabilities(
                    batch_probabilities, threshold, args.token_count
                ),
                batch_size,
                args.warmup,
                args.repeats,
            ),
            "hybrid_construction": benchmark_callable(
                lambda: build_hybrid_from_probabilities(
                    batch_probabilities, threshold, args.token_count
                ),
                batch_size,
                args.warmup,
                args.repeats,
            ),
            "compressed_vit_with_cached_artifacts": benchmark_callable(
                precomputed_call,
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
        }
        results[key] = {
            "full_vit": benchmark_callable(
                lambda: model_full(batch_vit),
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "masktopo_cpu_reference": benchmark_callable(
                lambda: e2e_builder(build_reference_from_probabilities),
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "masktopo_hybrid": benchmark_callable(
                lambda: e2e_builder(build_hybrid_from_probabilities),
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "precomputed_upper_bound": benchmark_callable(
                precomputed_call,
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
            "mask_only_no_graph": benchmark_callable(
                mask_only_call,
                batch_size,
                args.warmup,
                args.repeats,
                device,
            ),
        }
        full_ms = float(results[key]["full_vit"]["mean_ms_per_sample"])
        reference_ms = float(
            results[key]["masktopo_cpu_reference"]["mean_ms_per_sample"]
        )
        hybrid_ms = float(results[key]["masktopo_hybrid"]["mean_ms_per_sample"])
        hybrid_reduction = reduction_percent(full_ms, hybrid_ms)
        gates[key] = {
            "reference_reduction_percent": reduction_percent(full_ms, reference_ms),
            "hybrid_reduction_percent": hybrid_reduction,
            "hybrid_improvement_over_reference_percent": reduction_percent(
                reference_ms, hybrid_ms
            ),
            "claim_gate": claim_gate(hybrid_reduction),
        }
        del model_masktopo, model_full, model_grid
        torch.cuda.empty_cache()
        print(f"BATCH_DONE batch={batch_size}", flush=True)

    prior_path = Path("results_topology_construction_benchmark_v2/result.json")
    prior_summary = None
    if prior_path.exists():
        prior_report = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_summary = {
            key: prior_report["gates"].get(key) for key in ("1", "8", "32")
        }
    report = {
        "experiment_id": "TB-B-260731-006",
        "status": "completed",
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "configuration": {
            "dataset": "FIVES held-out cached crops",
            "sample_count": args.sample_count,
            "correctness_count": args.correctness_count,
            "mask_threshold": threshold,
            "closing_iterations": 0,
            "backbone": "torchvision ViT-B/16, random initialization",
            "input_size": 256,
            "insert_layer": args.insert_layer,
            "token_count": args.token_count,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "correctness": correctness,
        "components": components,
        "end_to_end": results,
        "gates": gates,
        "prior_benchmark_gate_summary": prior_summary,
        "limitations": [
            "Random initialization supports latency only, not task accuracy.",
            "The precomputed path is an offline upper bound.",
            "The mask-only path is an untrained efficiency ablation in this run.",
            "Microbenchmarks repeat fixed cached batches and exclude dataset I/O and model loading.",
        ],
        "protocol": str(
            (Path(__file__).parent / "TOPOLOGY_POSTPROCESS_E2E_PROTOCOL.md").resolve()
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Optimized TopoBridge end-to-end efficiency",
        "",
        "The hybrid passed exact artifact and downstream-output gates before timing.",
        "",
        "| Batch | Full ViT | CPU reference | Optimized hybrid | Precomputed upper bound | Mask-only | Hybrid reduction vs Full | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for batch_size in args.batch_sizes:
        key = str(batch_size)
        item = results[key]
        gate = gates[key]
        lines.append(
            f"| {batch_size} | {item['full_vit']['mean_ms_per_sample']:.3f} | "
            f"{item['masktopo_cpu_reference']['mean_ms_per_sample']:.3f} | "
            f"{item['masktopo_hybrid']['mean_ms_per_sample']:.3f} | "
            f"{item['precomputed_upper_bound']['mean_ms_per_sample']:.3f} | "
            f"{item['mask_only_no_graph']['mean_ms_per_sample']:.3f} | "
            f"{gate['hybrid_reduction_percent']:.2f}% | "
            f"{gate['claim_gate']} |"
        )
    lines.extend(
        [
            "",
            "All values are synchronized mean milliseconds per sample. Median, P95, and peak CUDA memory are in `result.json`.",
            "",
            "Random initialization: efficiency only, no accuracy claim.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "completed", "gates": gates}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

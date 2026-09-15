from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from crackforest_mask_topo import CrackUNet
from crackforest_mechanism_ablation import build_artifacts
from ph_token_baselines import make_model as make_ph_model
from ph_token_baselines import ph_descriptors
from strong_reducer_baselines import REDUCERS, load_arrays, make_model as make_reducer
from topocoarsen_oracle import TopoCoarsenModel


DATASETS = ("crackforest", "deepcrack")
BUDGETS = (8, 16, 32, 64)
BATCH_SIZES = (1, 8, 32)
PH_MODELS = ("ph_only", "ph_guided")
MODELS = ("mask_topo", *REDUCERS, *PH_MODELS)
MASK_REQUIRED = {"mask_topo", "mask_guided_queries", *PH_MODELS}
SEED = 20260810


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TB-B-260802-023E frozen-checkpoint efficiency audit."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("results_ph_pareto_efficiency")
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        default=Path("实验记录/TB-B-260802-023E_runtime.log"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cpu-repeats", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def dataset_args(root: Path, dataset: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset,
        cache_dir=root / ("real_cache" if dataset == "crackforest" else "deepcrack_cache"),
        dataset_root=root / "real_data/DeepCrack/dataset/extracted",
        train_size=None,
        dev_size=None,
        test_size=None,
        candidate_multiplier=3,
        crop_size=64,
        output_size=64,
        min_component_pixels=12,
        data_seed=SEED,
        split_seed=20260730,
    )


def load_frozen_inputs(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        arrays = load_arrays(dataset_args(root, dataset))["test"]
        prior_root = (
            root / f"results_crackforest_mechanism_seed{SEED}"
            if dataset == "crackforest"
            else root / f"results_deepcrack_seed{SEED}"
        )
        with np.load(prior_root / "mask_probabilities.npz") as archive:
            probabilities = archive["test"].astype(np.float32)
        if probabilities.shape != (arrays["images"].shape[0], 64, 64):
            raise RuntimeError(f"Probability/input mismatch for {dataset}.")
        result = json.loads((prior_root / "result.json").read_text(encoding="utf-8"))
        selected = result[
            "selected_on_prior_dev" if dataset == "crackforest" else "selected_on_dev"
        ]
        output[dataset] = {
            "arrays": arrays,
            "probabilities": probabilities,
            "threshold": float(selected["mask_threshold"]),
            "closing": int(selected["closing_iterations"]),
            "prior_root": prior_root,
        }
    with np.load(root / f"ph_cache/deepcrack_seed{SEED}_k64.npz") as archive:
        output["deepcrack"]["ph_descriptors"] = archive["test"].astype(np.float32)
    return output


def checkpoint_spec(root: Path, token_count: int, model: str) -> dict[str, Any]:
    if model == "mask_topo":
        if token_count == 16:
            run_root = root / f"results_deepcrack_seed{SEED}"
            checkpoint = run_root / f"mask_topo_external_seed{SEED}.pt"
            prediction_key = "mask_topo_external"
        else:
            run_root = root / f"results_token_budget_masktopo_deepcrack_k{token_count}_seed{SEED}"
            checkpoint = run_root / f"mask_topo_budget_k{token_count}_seed{SEED}.pt"
            prediction_key = "mask_topo"
    elif model in REDUCERS:
        run_root = (
            root / f"results_strong_reducer_deepcrack_seed{SEED}"
            if token_count == 16
            else root / f"results_token_budget_deepcrack_k{token_count}_seed{SEED}"
        )
        checkpoint = run_root / f"{model}_seed{SEED}.pt"
        prediction_key = model
    elif model in PH_MODELS:
        run_root = root / f"results_ph_deepcrack_k{token_count}_seed{SEED}"
        checkpoint = run_root / f"{model}_seed{SEED}.pt"
        prediction_key = model
    else:
        raise ValueError(model)
    return {
        "run_root": run_root,
        "checkpoint": checkpoint,
        "prediction_key": prediction_key,
    }


def make_frozen_model(
    root: Path, token_count: int, model_name: str
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    spec = checkpoint_spec(root, token_count, model_name)
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    if model_name == "mask_topo":
        architecture_name = str(checkpoint["model_name"])
        model = TopoCoarsenModel(architecture_name, 64, token_count)
    elif model_name in REDUCERS:
        model = make_reducer(model_name, 64, token_count)
    else:
        model = make_ph_model(model_name, 64, token_count)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    checkpoint_args = checkpoint.get("args", {})
    if int(checkpoint_args.get("seed", SEED)) != SEED:
        raise RuntimeError(f"Checkpoint seed mismatch: K={token_count}, {model_name}.")
    if int(checkpoint_args.get("token_count", token_count)) != token_count:
        raise RuntimeError(f"Checkpoint K mismatch: K={token_count}, {model_name}.")
    return model, checkpoint, spec


def batch_inputs(
    frozen: dict[str, Any],
    artifacts: dict[int, dict[str, np.ndarray]],
    token_count: int,
    model_name: str,
    start: int,
    stop: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    image = torch.from_numpy(
        frozen["arrays"]["images"][start:stop].astype(np.float32) / 255.0
    ).to(device)
    if model_name == "mask_topo":
        current = artifacts[token_count]
        return (
            image,
            torch.from_numpy(current["assignment"][start:stop].astype(np.int64)).to(device),
            torch.from_numpy(current["adjacency"][start:stop].astype(np.int64)).to(device),
            torch.from_numpy(current["reachability"][start:stop].astype(np.int64)).to(device),
        )
    mask = torch.from_numpy(
        frozen["probabilities"][start:stop].astype(np.float32)
    ).unsqueeze(1).to(device)
    if model_name in PH_MODELS:
        descriptor = torch.from_numpy(
            frozen["ph_descriptors"][start:stop, :token_count].astype(np.float32)
        ).to(device)
        return image, mask, descriptor
    return image, mask


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def verify_predictions(
    model: nn.Module,
    model_name: str,
    token_count: int,
    spec: dict[str, Any],
    frozen: dict[str, Any],
    artifacts: dict[int, dict[str, np.ndarray]],
    device: torch.device,
) -> int:
    with np.load(spec["run_root"] / "heldout_predictions.npz") as archive:
        expected = archive[spec["prediction_key"]].astype(np.uint8)
    parts: list[np.ndarray] = []
    for start in range(0, expected.size, 32):
        stop = min(start + 32, expected.size)
        inputs = batch_inputs(
            frozen, artifacts, token_count, model_name, start, stop, device
        )
        logits = model(*inputs)
        parts.append((torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.uint8))
    actual = np.concatenate(parts)
    mismatch = int(np.count_nonzero(actual != expected))
    if mismatch:
        raise RuntimeError(
            f"Frozen checkpoint prediction mismatch: K={token_count}, "
            f"model={model_name}, count={mismatch}."
        )
    return mismatch


@torch.inference_mode()
def benchmark_model(
    model: nn.Module,
    input_builder: Callable[[int, torch.device], tuple[torch.Tensor, ...]],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for batch in BATCH_SIZES:
        inputs = input_builder(batch, device)
        for _ in range(warmup):
            model(*inputs)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(*inputs)
        synchronize(device)
        elapsed = time.perf_counter() - started
        output[str(batch)] = {
            "batch_size": batch,
            "latency_ms_per_batch": 1000.0 * elapsed / repeats,
            "latency_ms_per_sample": 1000.0 * elapsed / repeats / batch,
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else None
            ),
        }
        del inputs
    return output


def counted_flops(
    model: nn.Module, inputs: tuple[torch.Tensor, ...]
) -> dict[str, Any]:
    with profile(
        activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=False
    ) as profiler:
        with torch.inference_mode():
            model(*inputs)
    events = profiler.key_averages()
    nonzero = [event for event in events if int(event.flops or 0) > 0]
    return {
        "torch_profiler_counted_flops_batch1": int(
            sum(int(event.flops or 0) for event in events)
        ),
        "operator_groups_total": len(events),
        "operator_groups_with_nonzero_flops": len(nonzero),
        "scope": "partial supported-operator count; unsupported operators report zero",
    }


@torch.inference_mode()
def verify_mask_predictor(
    model: CrackUNet,
    frozen: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    expected = frozen["probabilities"]
    parts: list[np.ndarray] = []
    images = frozen["arrays"]["images"]
    for start in range(0, images.shape[0], 32):
        image = torch.from_numpy(
            images[start : start + 32, 0].astype(np.float32) / 255.0
        ).unsqueeze(1).to(device)
        parts.append(torch.sigmoid(model(image)).cpu().numpy().astype(np.float32))
    actual = np.concatenate(parts)
    difference = np.abs(actual - expected)
    maximum = float(difference.max())
    mean = float(difference.mean())
    if maximum > 1e-3:
        raise RuntimeError(
            f"Mask predictor does not reproduce cached float16 probabilities: {maximum}."
        )
    return {"maximum_absolute_difference": maximum, "mean_absolute_difference": mean}


def benchmark_mask_predictor(
    root: Path,
    frozen: dict[str, Any],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    checkpoint_path = root / f"results_deepcrack_seed{SEED}/mask_predictor_seed{SEED}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CrackUNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    cpu_input = torch.from_numpy(
        frozen["arrays"]["images"][:1, 0].astype(np.float32) / 255.0
    ).unsqueeze(1)
    flops = counted_flops(model, (cpu_input,))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model = model.to(device)
    verification = verify_mask_predictor(model, frozen, device)

    def builder(batch: int, current_device: torch.device) -> tuple[torch.Tensor, ...]:
        image = torch.from_numpy(
            frozen["arrays"]["images"][:batch, 0].astype(np.float32) / 255.0
        ).unsqueeze(1).to(current_device)
        return (image,)

    runtime = benchmark_model(model, builder, device, warmup, repeats)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "parameters": parameters,
        "flops": flops,
        "prediction_cache_verification": verification,
        "runtime": runtime,
    }


def subset_arrays(arrays: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    return {
        key: value[:count]
        for key, value in arrays.items()
        if isinstance(value, np.ndarray) and value.shape[:1] == (arrays["images"].shape[0],)
    }


def benchmark_cpu_preprocessing(
    frozen_inputs: dict[str, dict[str, Any]], repeats: int, logger: RunLogger
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, np.ndarray]]]]:
    report: dict[str, Any] = {}
    full_artifacts: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for dataset in DATASETS:
        frozen = frozen_inputs[dataset]
        report[dataset] = {"mask_topo": {}, "ph": {}}
        full_artifacts[dataset] = {}
        for token_count in BUDGETS:
            logger.write(f"CPU_ARTIFACT_BUILD dataset={dataset} K={token_count}")
            full_artifacts[dataset][token_count] = build_artifacts(
                frozen["arrays"],
                frozen["probabilities"],
                frozen["threshold"],
                frozen["closing"],
                token_count,
                "mask_topo_recheck",
                20260731,
            )[0]
            report[dataset]["mask_topo"][str(token_count)] = {}
            for batch in BATCH_SIZES:
                current_arrays = subset_arrays(frozen["arrays"], batch)
                current_probabilities = frozen["probabilities"][:batch]
                durations = []
                for repetition in range(repeats + 1):
                    started = time.perf_counter()
                    build_artifacts(
                        current_arrays,
                        current_probabilities,
                        frozen["threshold"],
                        frozen["closing"],
                        token_count,
                        "mask_topo_recheck",
                        20260731,
                    )
                    elapsed = time.perf_counter() - started
                    if repetition:
                        durations.append(1000.0 * elapsed)
                report[dataset]["mask_topo"][str(token_count)][str(batch)] = {
                    "batch_size": batch,
                    "repeats": repeats,
                    "latency_ms_per_batch_values": durations,
                    "latency_ms_per_batch": float(np.mean(durations)),
                    "latency_ms_per_sample": float(np.mean(durations) / batch),
                }
        logger.write(f"CPU_PH_BENCHMARK dataset={dataset}")
        for batch in BATCH_SIZES:
            durations = []
            for repetition in range(repeats + 1):
                started = time.perf_counter()
                for index in range(batch):
                    ph_descriptors(frozen["probabilities"][index], 64)
                elapsed = time.perf_counter() - started
                if repetition:
                    durations.append(1000.0 * elapsed)
            report[dataset]["ph"][str(batch)] = {
                "batch_size": batch,
                "repeats": repeats,
                "latency_ms_per_batch_values": durations,
                "latency_ms_per_batch": float(np.mean(durations)),
                "latency_ms_per_sample": float(np.mean(durations) / batch),
                "max_ph_tokens": 64,
            }
    return report, full_artifacts


def benchmark_connectors(
    root: Path,
    frozen: dict[str, Any],
    artifacts: dict[int, dict[str, np.ndarray]],
    device: torch.device,
    warmup: int,
    repeats: int,
    logger: RunLogger,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for token_count in BUDGETS:
        output[str(token_count)] = {}
        for model_name in MODELS:
            logger.write(f"CONNECTOR_START K={token_count} model={model_name}")
            model, checkpoint, spec = make_frozen_model(root, token_count, model_name)
            parameters = sum(parameter.numel() for parameter in model.parameters())
            cpu_inputs = batch_inputs(
                frozen, artifacts, token_count, model_name, 0, 1, torch.device("cpu")
            )
            flops = counted_flops(model, cpu_inputs)
            del cpu_inputs
            model = model.to(device)
            mismatch_count = verify_predictions(
                model,
                model_name,
                token_count,
                spec,
                frozen,
                artifacts,
                device,
            )

            def builder(
                batch: int, current_device: torch.device
            ) -> tuple[torch.Tensor, ...]:
                return batch_inputs(
                    frozen,
                    artifacts,
                    token_count,
                    model_name,
                    0,
                    batch,
                    current_device,
                )

            runtime = benchmark_model(model, builder, device, warmup, repeats)
            output[str(token_count)][model_name] = {
                "checkpoint": str(spec["checkpoint"].resolve()),
                "checkpoint_sha256": file_sha256(spec["checkpoint"]),
                "checkpoint_model_name": checkpoint.get("model_name"),
                "parameters": parameters,
                "saved_prediction_mismatch_count": mismatch_count,
                "flops": flops,
                "runtime": runtime,
            }
            logger.write(
                f"CONNECTOR_DONE K={token_count} model={model_name} prediction_match=true"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return output


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front = []
    for row in rows:
        dominated = any(
            other["mean_balanced_accuracy"] >= row["mean_balanced_accuracy"]
            and other["conservative_end_to_end_ms_per_sample"]
            <= row["conservative_end_to_end_ms_per_sample"]
            and (
                other["mean_balanced_accuracy"] > row["mean_balanced_accuracy"]
                or other["conservative_end_to_end_ms_per_sample"]
                < row["conservative_end_to_end_ms_per_sample"]
            )
            for other in rows
        )
        if not dominated:
            front.append(
                {
                    key: row[key]
                    for key in (
                        "dataset",
                        "batch_size",
                        "token_count",
                        "model",
                        "mean_balanced_accuracy",
                        "conservative_end_to_end_ms_per_sample",
                    )
                }
            )
    return sorted(
        front,
        key=lambda item: (
            item["conservative_end_to_end_ms_per_sample"],
            -item["mean_balanced_accuracy"],
        ),
    )


def build_unified_rows(
    accuracy: dict[str, Any],
    connectors: dict[str, Any],
    predictor: dict[str, Any],
    preprocessing: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for token_count in BUDGETS:
            accuracy_budget = accuracy["datasets"][dataset]["budgets"][str(token_count)]
            for model_name in MODELS:
                aggregate = accuracy_budget["aggregates"][model_name]
                connector = connectors[str(token_count)][model_name]
                uses_mask = model_name in MASK_REQUIRED
                if model_name == "mask_topo":
                    preprocessor = "connected_components_assignment_graph"
                elif model_name in PH_MODELS:
                    preprocessor = "cubical_persistent_homology"
                else:
                    preprocessor = "none"
                for batch in BATCH_SIZES:
                    key = str(batch)
                    connector_runtime = connector["runtime"][key]
                    predictor_runtime = predictor["runtime"][key] if uses_mask else None
                    if model_name == "mask_topo":
                        preprocess_ms = preprocessing[dataset]["mask_topo"][
                            str(token_count)
                        ][key]["latency_ms_per_sample"]
                    elif model_name in PH_MODELS:
                        preprocess_ms = preprocessing[dataset]["ph"][key][
                            "latency_ms_per_sample"
                        ]
                    else:
                        preprocess_ms = 0.0
                    mask_ms = (
                        float(predictor_runtime["latency_ms_per_sample"])
                        if predictor_runtime is not None
                        else 0.0
                    )
                    connector_ms = float(connector_runtime["latency_ms_per_sample"])
                    connector_flops = int(
                        connector["flops"]["torch_profiler_counted_flops_batch1"]
                    )
                    predictor_flops = int(
                        predictor["flops"]["torch_profiler_counted_flops_batch1"]
                    ) if uses_mask else 0
                    connector_memory = connector_runtime["peak_cuda_memory_mb"]
                    predictor_memory = (
                        predictor_runtime["peak_cuda_memory_mb"]
                        if predictor_runtime is not None
                        else None
                    )
                    memories = [
                        float(value)
                        for value in (connector_memory, predictor_memory)
                        if value is not None
                    ]
                    rows.append(
                        {
                            "dataset": dataset,
                            "batch_size": batch,
                            "token_count": token_count,
                            "model": model_name,
                            "mean_balanced_accuracy": aggregate[
                                "mean_balanced_accuracy"
                            ],
                            "sample_std_balanced_accuracy": aggregate[
                                "sample_std_balanced_accuracy"
                            ],
                            "connector_parameters": connector["parameters"],
                            "pipeline_parameters": connector["parameters"]
                            + (predictor["parameters"] if uses_mask else 0),
                            "connector_partial_flops_batch1": connector_flops,
                            "pipeline_gpu_partial_flops_batch1": connector_flops
                            + predictor_flops,
                            "cpu_preprocessor": preprocessor,
                            "cpu_preprocessor_flops_status": (
                                "unsupported_not_counted"
                                if preprocessor != "none"
                                else "not_applicable"
                            ),
                            "connector_ms_per_sample": connector_ms,
                            "mask_predictor_ms_per_sample": mask_ms,
                            "cpu_preprocessing_ms_per_sample": preprocess_ms,
                            "conservative_end_to_end_ms_per_sample": connector_ms
                            + mask_ms
                            + preprocess_ms,
                            "serial_pipeline_peak_cuda_memory_mb": (
                                max(memories) if memories else None
                            ),
                            "connector_benchmark_input_dataset": "deepcrack",
                        }
                    )
    fronts: dict[str, Any] = {}
    for dataset in DATASETS:
        fronts[dataset] = {}
        for batch in BATCH_SIZES:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["batch_size"] == batch
            ]
            fronts[dataset][str(batch)] = pareto_front(selected)
    return rows, fronts


def report_markdown(result: dict[str, Any]) -> str:
    rows = result["unified_rows"]
    lines = [
        "# TB-B-260802-023E frozen-checkpoint efficiency audit",
        "",
        "The table below uses endpoint-connectivity balanced accuracy, not the P1-5 disease-classification result.",
        "All connector checkpoints reproduced their saved seed-20260810 held-out predictions exactly before timing.",
        "",
        "## Representative K=16 latency",
        "",
        "| Dataset | Model | batch 1 end-to-end ms/sample | batch 32 end-to-end ms/sample | partial pipeline GFLOPs | peak CUDA MB (batch 32) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for model_name in MODELS:
            batch1 = next(
                row
                for row in rows
                if row["dataset"] == dataset
                and row["token_count"] == 16
                and row["model"] == model_name
                and row["batch_size"] == 1
            )
            batch32 = next(
                row
                for row in rows
                if row["dataset"] == dataset
                and row["token_count"] == 16
                and row["model"] == model_name
                and row["batch_size"] == 32
            )
            lines.append(
                f"| {dataset} | {model_name} | "
                f"{batch1['conservative_end_to_end_ms_per_sample']:.3f} | "
                f"{batch32['conservative_end_to_end_ms_per_sample']:.3f} | "
                f"{batch1['pipeline_gpu_partial_flops_batch1'] / 1e9:.3f} | "
                f"{batch32['serial_pipeline_peak_cuda_memory_mb']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- End-to-end latency is a conservative serial sum; it does not claim CPU/GPU overlap.",
            "- FLOPs cover only operators counted by PyTorch profiler. CPU connected-components and cubical-PH algorithms are marked unsupported and excluded from FLOPs.",
            "- Connector latency uses real DeepCrack inputs and frozen checkpoints; dataset-specific CPU preprocessing is measured separately for both datasets.",
            "- Grid and mechanism ablations remain K=16 evidence and are not fabricated into a variable-K curve.",
            "- Massachusetts Roads was not reopened.",
            "",
        ]
    )
    return "\n".join(lines)


def run_self_test() -> None:
    rows = [
        {
            "dataset": "x",
            "batch_size": 1,
            "token_count": 8,
            "model": "a",
            "mean_balanced_accuracy": 0.7,
            "conservative_end_to_end_ms_per_sample": 2.0,
        },
        {
            "dataset": "x",
            "batch_size": 1,
            "token_count": 16,
            "model": "b",
            "mean_balanced_accuracy": 0.6,
            "conservative_end_to_end_ms_per_sample": 3.0,
        },
        {
            "dataset": "x",
            "batch_size": 1,
            "token_count": 32,
            "model": "c",
            "mean_balanced_accuracy": 0.8,
            "conservative_end_to_end_ms_per_sample": 4.0,
        },
    ]
    front = pareto_front(rows)
    assert [item["model"] for item in front] == ["a", "c"]
    print("BENCHMARK_PH_INCLUSIVE_EFFICIENCY_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = args.root.resolve()
    output = args.output.resolve()
    run_log = (root / args.run_log).resolve() if not args.run_log.is_absolute() else args.run_log
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if run_log.exists():
        raise FileExistsError(f"Refusing to overwrite existing run log: {run_log}")
    output.mkdir(parents=True)
    run_log.parent.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_log)
    logger.write("START TB-B-260802-023E")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen efficiency protocol requires the available CUDA GPU.")
    torch.backends.cudnn.benchmark = False
    # Match the frozen checkpoint inference environment. Disabling cuDNN TF32
    # changes the cached mask probabilities by up to 0.00313 on this GPU.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    frozen_inputs = load_frozen_inputs(root)
    logger.write("INPUTS_LOADED massachusetts_test_reopened=false")
    preprocessing, full_artifacts = benchmark_cpu_preprocessing(
        frozen_inputs, args.cpu_repeats, logger
    )
    logger.write("MASK_PREDICTOR_START")
    predictor = benchmark_mask_predictor(
        root, frozen_inputs["deepcrack"], device, args.warmup, args.repeats
    )
    logger.write("MASK_PREDICTOR_DONE prediction_cache_match=true")
    connectors = benchmark_connectors(
        root,
        frozen_inputs["deepcrack"],
        full_artifacts["deepcrack"],
        device,
        args.warmup,
        args.repeats,
        logger,
    )
    accuracy_path = root / "results_ph_pareto_summary/result.json"
    accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
    if accuracy["status"] != "completed_verified_accuracy_and_ph_efficiency":
        raise RuntimeError("Verified PH Pareto accuracy input is missing.")
    unified_rows, fronts = build_unified_rows(
        accuracy, connectors, predictor, preprocessing
    )
    result = {
        "experiment_id": "TB-B-260802-023E",
        "status": "completed_verified",
        "parent_experiment": "TB-B-260802-023",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "optimization_seed_checkpoint": SEED,
        "batch_sizes": list(BATCH_SIZES),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cpu_repeats": args.cpu_repeats,
        "mask_predictor": predictor,
        "cpu_preprocessing": preprocessing,
        "connectors": connectors,
        "unified_rows": unified_rows,
        "latency_accuracy_pareto_fronts": fronts,
        "all_checkpoint_predictions_reproduced": all(
            item["saved_prediction_mismatch_count"] == 0
            for budget in connectors.values()
            for item in budget.values()
        ),
        "accuracy_summary": str(accuracy_path.resolve()),
        "accuracy_summary_sha256": file_sha256(accuracy_path),
        "protocol": str((root / "PH_PARETO_EFFICIENCY_PROTOCOL.md").resolve()),
        "protocol_sha256": file_sha256(root / "PH_PARETO_EFFICIENCY_PROTOCOL.md"),
        "run_log": str(run_log),
        "massachusetts_test_reopened": False,
        "disease_classification_p1_5_used": False,
        "flop_scope": "partial PyTorch-supported GPU-model operators; CPU topology/PH excluded",
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(report_markdown(result), encoding="utf-8")
    with (output / "UNIFIED_PARETO.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(unified_rows[0]))
        writer.writeheader()
        writer.writerows(unified_rows)
    logger.write("COMPLETED status=completed_verified")
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "gpu_name": result["gpu_name"],
                "checkpoints_verified": result[
                    "all_checkpoint_predictions_reproduced"
                ],
                "unified_rows": len(unified_rows),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

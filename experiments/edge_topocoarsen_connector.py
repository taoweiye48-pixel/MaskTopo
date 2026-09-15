from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from edge_graph_diagnostic import (
    image_endpoints,
    image_foreground,
    metrics as edge_graph_metrics,
    predict_edges,
)
from topobridge_mvp import load_or_generate, set_seed
from topocoarsen_learned import LearnableTopoCoarsen, slice_arrays
from topocoarsen_oracle import (
    TopoCoarsenModel,
    coarse_graph,
    oracle_assignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only EdgeTopoCoarsen: learned local edges followed by "
            "deterministic 256-to-16 graph coarsening."
        )
    )
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--dev-size", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=20260810,
        help="Fixed split seed; vary --seed to isolate optimization variance.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99],
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument(
        "--edge-checkpoint",
        type=Path,
        default=Path(
            "results_learned_topocoarsen/"
            "learned_topocoarsen_seed20260810.pt"
        ),
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("results_learned_topocoarsen_v2/summary.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results_edge_topocoarsen")
    )
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Apply per-image/channel robust contrast canonicalization.",
    )
    parser.add_argument(
        "--train-grid-control",
        action="store_true",
        help="Retrain the grid control under the identical preprocessing.",
    )
    parser.add_argument(
        "--photometric-augmentation",
        action="store_true",
        help="Train with random contrast, brightness, and sensor noise.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


class EdgeCoarsenDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        artifacts: dict[str, np.ndarray],
        augment: bool = False,
    ) -> None:
        self.images = arrays["images"]
        self.labels = arrays["connected"]
        self.artifacts = artifacts
        self.augment = augment

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(
            self.images[index].astype(np.float32) / 255.0
        )
        if self.augment:
            contrast = 0.72 + 0.38 * torch.rand(())
            brightness = -0.04 + 0.18 * torch.rand(())
            noise_scale = 0.055 * torch.rand(())
            image = torch.clamp(
                brightness
                + contrast * image
                + noise_scale * torch.randn_like(image),
                0.0,
                1.0,
            )
        return {
            "image": image,
            "connected": torch.tensor(
                self.labels[index], dtype=torch.float32
            ),
            "assignment": torch.from_numpy(
                self.artifacts["assignment"][index].astype(np.int64)
            ),
            "adjacency": torch.from_numpy(
                self.artifacts["adjacency"][index].astype(np.int64)
            ),
            "reachability": torch.from_numpy(
                self.artifacts["reachability"][index].astype(np.int64)
            ),
        }


def canonicalize_images(images: np.ndarray) -> np.ndarray:
    """Remove global per-channel intensity shifts without using labels."""
    normalized = images.astype(np.float32) / 255.0
    flat = normalized.reshape(normalized.shape[0], normalized.shape[1], -1)
    low = np.quantile(flat, 0.10, axis=-1, keepdims=True)
    high = np.quantile(flat, 0.999, axis=-1, keepdims=True)
    scale = np.maximum(high - low, 0.15)
    canonical = np.clip((flat - low) / scale, 0.0, 1.0)
    return np.round(
        canonical.reshape(normalized.shape) * 255.0
    ).astype(np.uint8)


def fixed_grid_artifacts(
    sample_count: int, token_count: int
) -> dict[str, np.ndarray]:
    assignment = np.arange(token_count, dtype=np.int16).reshape(4, 4)
    assignment = assignment.repeat(4, axis=0).repeat(4, axis=1)
    adjacency, reachability = coarse_graph(
        assignment,
        np.zeros((16, 16), dtype=np.int16),
        token_count,
        topology_aware=False,
    )
    return {
        "assignment": np.repeat(
            assignment[None], sample_count, axis=0
        ),
        "adjacency": np.repeat(adjacency[None], sample_count, axis=0),
        "reachability": np.repeat(
            reachability[None], sample_count, axis=0
        ),
    }


def predicted_component_ids(
    image: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Infer semantic groups from image pixels and learned edge probabilities."""
    foreground = image_foreground(image)
    groups = np.zeros((16, 16), dtype=np.int16)
    next_group = 1
    for start_row in range(16):
        for start_col in range(16):
            if not foreground[start_row, start_col]:
                continue
            if groups[start_row, start_col] != 0:
                continue
            groups[start_row, start_col] = next_group
            queue: deque[tuple[int, int]] = deque(
                [(start_row, start_col)]
            )
            while queue:
                row, col = queue.popleft()
                neighbors = (
                    (
                        row,
                        col - 1,
                        horizontal[row, col - 1] if col > 0 else -1.0,
                    ),
                    (
                        row,
                        col + 1,
                        horizontal[row, col] if col < 15 else -1.0,
                    ),
                    (
                        row - 1,
                        col,
                        vertical[row - 1, col] if row > 0 else -1.0,
                    ),
                    (
                        row + 1,
                        col,
                        vertical[row, col] if row < 15 else -1.0,
                    ),
                )
                for new_row, new_col, probability in neighbors:
                    if not (0 <= new_row < 16 and 0 <= new_col < 16):
                        continue
                    if (
                        foreground[new_row, new_col]
                        and groups[new_row, new_col] == 0
                        and probability >= threshold
                    ):
                        groups[new_row, new_col] = next_group
                        queue.append((new_row, new_col))
            next_group += 1
    return groups


def make_artifacts(
    images: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    threshold: float,
    token_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample_count = images.shape[0]
    assignments = np.empty((sample_count, 16, 16), dtype=np.int16)
    adjacencies = np.empty(
        (sample_count, token_count, token_count), dtype=np.uint8
    )
    reachabilities = np.empty_like(adjacencies)
    component_counts: list[int] = []
    structural_predictions = np.empty(sample_count, dtype=np.uint8)

    for index in range(sample_count):
        groups = predicted_component_ids(
            images[index],
            horizontal[index],
            vertical[index],
            threshold,
        )
        group_count = int(np.unique(groups).size)
        if group_count > token_count:
            raise RuntimeError(
                f"sample {index} has {group_count} groups for "
                f"{token_count} tokens"
            )
        assignment = oracle_assignment(groups, token_count, False)
        adjacency, reachability = coarse_graph(
            assignment, groups, token_count, topology_aware=True
        )
        assignments[index] = assignment
        adjacencies[index] = adjacency
        reachabilities[index] = reachability
        component_counts.append(group_count - 1)
        first, second = image_endpoints(images[index])
        first_cluster = int(assignment.reshape(-1)[first])
        second_cluster = int(assignment.reshape(-1)[second])
        structural_predictions[index] = reachability[
            first_cluster, second_cluster
        ]

    counts = np.asarray(component_counts)
    diagnostics = {
        "sample_count": int(sample_count),
        "all_256_patches_assigned": bool(
            np.all((assignments >= 0) & (assignments < token_count))
        ),
        "all_16_tokens_nonempty": bool(
            all(
                np.unique(assignments[index]).size == token_count
                for index in range(sample_count)
            )
        ),
        "foreground_component_count": {
            "minimum": int(counts.min()),
            "median": float(np.median(counts)),
            "p95": float(np.quantile(counts, 0.95)),
            "maximum": int(counts.max()),
        },
        "structural_predictions": structural_predictions,
    }
    return {
        "assignment": assignments,
        "adjacency": adjacencies,
        "reachability": reachabilities,
    }, diagnostics


def balanced_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    positive = float(np.mean(prediction[truth] == truth[truth]))
    negative = float(np.mean(prediction[~truth] == truth[~truth]))
    return {
        "balanced_accuracy": 0.5 * (positive + negative),
        "positive_accuracy": positive,
        "negative_accuracy": negative,
        "accuracy": float(np.mean(prediction == truth)),
    }


@torch.no_grad()
def evaluate(
    model: TopoCoarsenModel,
    loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False,
) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None]:
    model.eval()
    truths = []
    predictions = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        logits = model(
            image,
            batch["assignment"].to(device, non_blocking=True),
            batch["adjacency"].to(device, non_blocking=True),
            batch["reachability"].to(device, non_blocking=True),
        )
        truths.append(batch["connected"].cpu().numpy().astype(np.uint8))
        predictions.append(
            (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.uint8)
        )
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    return (
        balanced_metrics(truth, prediction),
        prediction if return_predictions else None,
        truth if return_predictions else None,
    )


def bootstrap_balanced_accuracy(
    truth: np.ndarray,
    prediction: np.ndarray,
    seed: int,
    replicates: int = 10_000,
) -> list[float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(truth == 1)
    negative = np.flatnonzero(truth == 0)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_positive = rng.choice(
            positive, size=positive.size, replace=True
        )
        sampled_negative = rng.choice(
            negative, size=negative.size, replace=True
        )
        positive_accuracy = np.mean(prediction[sampled_positive] == 1)
        negative_accuracy = np.mean(prediction[sampled_negative] == 0)
        values[index] = 0.5 * (positive_accuracy + negative_accuracy)
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def train_connector(
    args: argparse.Namespace,
    train_dataset: EdgeCoarsenDataset,
    dev_dataset: EdgeCoarsenDataset,
    test_dataset: EdgeCoarsenDataset,
    output: Path,
    device: torch.device,
    model_name: str = "edge_topocoarsen",
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    set_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = TopoCoarsenModel(model_name, args.dim, args.token_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = output / f"{model_name}_seed{args.seed}.pt"
    best_dev = -1.0
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            image = batch["image"].to(device, non_blocking=True)
            target = batch["connected"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    image,
                    batch["assignment"].to(device, non_blocking=True),
                    batch["adjacency"].to(device, non_blocking=True),
                    batch["reachability"].to(device, non_blocking=True),
                )
                loss = F.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batches += 1

        dev_metrics, _, _ = evaluate(model, dev_loader, device)
        record = {
            "seed": args.seed,
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            "dev_balanced_accuracy": dev_metrics["balanced_accuracy"],
            "dev_positive_accuracy": dev_metrics["positive_accuracy"],
            "dev_negative_accuracy": dev_metrics["negative_accuracy"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"{model_name.upper()} epoch={epoch}/{args.epochs} "
            f"loss={record['train_loss']:.4f} "
            f"dev_bal={record['dev_balanced_accuracy']:.4f}",
            flush=True,
        )
        if dev_metrics["balanced_accuracy"] > best_dev:
            best_dev = dev_metrics["balanced_accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "dev_metrics": dev_metrics,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    test_metrics, prediction, truth = evaluate(
        model, test_loader, device, return_predictions=True
    )
    if prediction is None or truth is None:
        raise AssertionError("Missing held-out predictions.")
    summary = {
        "model": model_name,
        "seed": args.seed,
        "parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "best_dev_balanced_accuracy": best_dev,
        "test_metrics": test_metrics,
        "training_seconds": time.time() - started,
        "test_inference_uses_images_only": True,
    }
    return summary, history, prediction, truth


def read_baselines(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["model"]: float(row["test_balanced_accuracy"]) for row in rows
    }


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    image = np.zeros((3, 64, 64), dtype=np.float32)
    image[0, 8:24, 8:24] = 1.0
    image[1, 9:12, 9:12] = 1.0
    image[2, 20:23, 20:23] = 1.0
    horizontal = np.ones((16, 15), dtype=np.float32)
    vertical = np.ones((15, 16), dtype=np.float32)
    groups = predicted_component_ids(image, horizontal, vertical, 0.9)
    assert groups.shape == (16, 16)
    assert np.unique(groups).size == 2
    images = image[None]
    artifacts, diagnostics = make_artifacts(
        images,
        horizontal[None],
        vertical[None],
        0.9,
        16,
    )
    assert artifacts["assignment"].shape == (1, 16, 16)
    assert diagnostics["all_256_patches_assigned"]
    assert diagnostics["all_16_tokens_nonempty"]
    shifted = np.round(
        np.clip(0.12 + 0.76 * image[None], 0.0, 1.0) * 255
    ).astype(np.uint8)
    canonical = canonicalize_images(shifted)
    assert canonical.shape == shifted.shape
    assert canonical.dtype == np.uint8
    print("EDGE_TOPOCOARSEN_SELF_TEST_PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if not (0 < args.dev_size < args.val_size):
        raise ValueError("dev-size must be strictly between 0 and val-size.")
    if args.token_count != 16:
        raise ValueError("This controlled gate requires exactly 16 tokens.")

    set_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "torch": torch.__version__,
                "args": vars(args),
            },
            default=str,
        ),
        flush=True,
    )

    train_arrays = load_or_generate(
        cache_dir, "train", args.train_size, args.data_seed
    )
    validation_arrays = load_or_generate(
        cache_dir, "val", args.val_size, args.data_seed + 1_000_000
    )
    dev_arrays = slice_arrays(validation_arrays, 0, args.dev_size)
    test_arrays = slice_arrays(
        validation_arrays, args.dev_size, args.val_size
    )
    if args.canonicalize:
        train_arrays["images"] = canonicalize_images(
            train_arrays["images"]
        )
        dev_arrays["images"] = canonicalize_images(dev_arrays["images"])
        test_arrays["images"] = canonicalize_images(test_arrays["images"])
    normalized = {
        "train": train_arrays["images"].astype(np.float32) / 255.0,
        "dev": dev_arrays["images"].astype(np.float32) / 255.0,
        "test": test_arrays["images"].astype(np.float32) / 255.0,
    }

    edge_model = LearnableTopoCoarsen(
        args.dim, args.token_count, args.temperature
    ).to(device)
    edge_checkpoint = torch.load(
        args.edge_checkpoint.resolve(),
        map_location=device,
        weights_only=False,
    )
    edge_model.load_state_dict(edge_checkpoint["model"])
    predicted_edges = {
        split: predict_edges(edge_model, images, device, args.batch_size)
        for split, images in normalized.items()
    }
    del edge_model

    dev_foreground = np.stack(
        [image_foreground(image) for image in normalized["dev"]]
    )
    dev_endpoints = [
        image_endpoints(image) for image in normalized["dev"]
    ]
    threshold_rows = []
    dev_indices = np.arange(args.dev_size)
    for threshold in args.thresholds:
        current = edge_graph_metrics(
            dev_indices,
            dev_arrays["connected"],
            dev_foreground,
            dev_endpoints,
            predicted_edges["dev"][0],
            predicted_edges["dev"][1],
            threshold,
        )
        threshold_rows.append({"threshold": threshold, **current})
    selected = max(
        threshold_rows,
        key=lambda row: (row["balanced_accuracy"], row["threshold"]),
    )
    threshold = float(selected["threshold"])

    all_artifacts: dict[str, dict[str, np.ndarray]] = {}
    artifact_diagnostics: dict[str, Any] = {}
    for split, arrays in (
        ("train", train_arrays),
        ("dev", dev_arrays),
        ("test", test_arrays),
    ):
        artifacts, diagnostics = make_artifacts(
            normalized[split],
            predicted_edges[split][0],
            predicted_edges[split][1],
            threshold,
            args.token_count,
        )
        structural = balanced_metrics(
            arrays["connected"],
            diagnostics.pop("structural_predictions"),
        )
        diagnostics["structural_endpoint_metrics"] = structural
        all_artifacts[split] = artifacts
        artifact_diagnostics[split] = diagnostics
        print(
            f"EDGE_ARTIFACT split={split} "
            f"structural_bal={structural['balanced_accuracy']:.4f} "
            f"components_max="
            f"{diagnostics['foreground_component_count']['maximum']}",
            flush=True,
        )

    train_dataset = EdgeCoarsenDataset(
        train_arrays,
        all_artifacts["train"],
        augment=args.photometric_augmentation,
    )
    dev_dataset = EdgeCoarsenDataset(dev_arrays, all_artifacts["dev"])
    test_dataset = EdgeCoarsenDataset(test_arrays, all_artifacts["test"])
    summary, history, prediction, truth = train_connector(
        args,
        train_dataset,
        dev_dataset,
        test_dataset,
        output,
        device,
    )
    baselines = read_baselines(args.baseline_summary.resolve())
    grid_control_summary = None
    if args.train_grid_control:
        grid_train_dataset = EdgeCoarsenDataset(
            train_arrays,
            fixed_grid_artifacts(args.train_size, args.token_count),
            augment=args.photometric_augmentation,
        )
        grid_dev_dataset = EdgeCoarsenDataset(
            dev_arrays,
            fixed_grid_artifacts(args.dev_size, args.token_count),
        )
        grid_test_dataset = EdgeCoarsenDataset(
            test_arrays,
            fixed_grid_artifacts(
                args.val_size - args.dev_size, args.token_count
            ),
        )
        (
            grid_control_summary,
            grid_history,
            _,
            _,
        ) = train_connector(
            args,
            grid_train_dataset,
            grid_dev_dataset,
            grid_test_dataset,
            output,
            device,
            model_name="grid",
        )
        history.extend(grid_history)
        baselines["grid"] = grid_control_summary["test_metrics"][
            "balanced_accuracy"
        ]
    strongest_baseline_name = max(
        ("grid", "conv"), key=lambda name: baselines[name]
    )
    strongest_baseline = baselines[strongest_baseline_name]
    connector_score = summary["test_metrics"]["balanced_accuracy"]
    gain = connector_score - strongest_baseline
    confidence_interval = bootstrap_balanced_accuracy(
        truth, prediction, args.seed
    )
    np.savez_compressed(
        output / "heldout_predictions.npz",
        truth=truth,
        prediction=prediction,
    )
    write_history(output / "history.csv", history)
    with (output / "threshold_selection.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(threshold_rows[0])
        )
        writer.writeheader()
        writer.writerows(threshold_rows)

    result = {
        "experiment_id": "edge_topocoarsen_connector_gate",
        "status": "completed",
        "data_seed": args.data_seed,
        "optimization_seed": args.seed,
        "per_image_channel_contrast_canonicalization": args.canonicalize,
        "training_photometric_augmentation": (
            args.photometric_augmentation
        ),
        "method": (
            "learned image-only local edges -> foreground union-find -> "
            "deterministic component-preserving 256-to-16 coarsening"
        ),
        "selected_edge_threshold_on_dev": threshold,
        "selected_edge_dev_metrics": {
            key: value for key, value in selected.items() if key != "threshold"
        },
        "artifact_diagnostics": artifact_diagnostics,
        "connector": summary,
        "retrained_grid_control": grid_control_summary,
        "connector_test_balanced_accuracy_bootstrap_95_ci": (
            confidence_interval
        ),
        "reference_scores_same_split_seed_training_protocol": baselines,
        "strongest_image_only_baseline": strongest_baseline_name,
        "gain_over_strongest_baseline": gain,
        "oracle_gap_remaining": (
            baselines["topology_oracle"] - connector_score
        ),
        "inference_contract": {
            "edge_and_coarsening_inputs": ["image"],
            "downstream_connector_inputs": [
                "image",
                "image-derived assignment",
                "image-derived coarse adjacency",
                "image-derived reachability",
            ],
            "uses_ground_truth_patch_ids_at_dev_or_test": False,
            "uses_endpoint_metadata_at_dev_or_test": False,
            "all_256_patches_aggregated": True,
            "exact_output_token_count": args.token_count,
        },
        "gates": {
            "structural_test_accuracy_at_least_90pct": (
                artifact_diagnostics["test"][
                    "structural_endpoint_metrics"
                ]["balanced_accuracy"]
                >= 0.90
            ),
            "connector_gain_over_baseline_at_least_3pp": gain >= 0.03,
            "connector_recovers_at_least_half_oracle_gap": (
                connector_score - strongest_baseline
                >= 0.5 * (baselines["topology_oracle"] - strongest_baseline)
            ),
        },
    }
    result["verdict"] = (
        "PROCEED_TO_MULTISEED_OOD"
        if all(result["gates"].values())
        else (
            "MECHANISM_SUPPORTED_CONNECTOR_NEEDS_REDESIGN"
            if result["gates"]["structural_test_accuracy_at_least_90pct"]
            else "STOP_EDGE_TOPOCOARSEN"
        )
    )
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    names = [strongest_baseline_name, "edge_topocoarsen", "topology_oracle"]
    values = [
        strongest_baseline,
        connector_score,
        baselines["topology_oracle"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(names, values, color=["#4C78A8", "#E45756", "#72B7B2"])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("held-out test balanced accuracy")
    ax.set_title("EdgeTopoCoarsen connector gate (256 patches to 16 tokens)")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(output / "connector_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

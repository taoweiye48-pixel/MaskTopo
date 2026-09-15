from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from topobridge_mvp import (
    PatchStem,
    coordinate_shortcut_accuracy,
    load_or_generate,
    set_seed,
)
from topobridge_oracle import endpoint_patch_indices
from topocoarsen_oracle import (
    CoarseGraphHead,
    TopoCoarsenModel,
    build_artifacts,
    fine_coordinates,
)


MODEL_NAMES = ["grid", "conv", "learned_topocoarsen", "topology_oracle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learned TopoCoarsen validation with image-only inference."
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
    parser.add_argument("--assignment-weight", type=float, default=0.3)
    parser.add_argument("--edge-weight", type=float, default=0.2)
    parser.add_argument("--graph-weight", type=float, default=0.5)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--output", type=Path, default=Path("results_learned"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def slice_arrays(
    arrays: dict[str, np.ndarray], start: int, stop: int
) -> dict[str, np.ndarray]:
    return {key: value[start:stop].copy() for key, value in arrays.items()}


class LearnedDataset(Dataset):
    def __init__(
        self, arrays: dict[str, np.ndarray], artifacts: dict[str, np.ndarray]
    ) -> None:
        self.arrays = arrays
        self.artifacts = artifacts

    def __len__(self) -> int:
        return int(self.arrays["images"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "image": torch.from_numpy(
                self.arrays["images"][index].astype(np.float32) / 255.0
            ),
            "connected": torch.tensor(
                self.arrays["connected"][index], dtype=torch.float32
            ),
            "patch_ids": torch.from_numpy(
                self.arrays["patch_ids"][index].astype(np.int64)
            ),
            "endpoint_features": torch.from_numpy(
                self.arrays["endpoint_features"][index].astype(np.float32)
            ),
        }
        for key, value in self.artifacts.items():
            item[key] = torch.from_numpy(value[index].astype(np.int64))
        return item


def grid_anchors(token_count: int) -> torch.Tensor:
    side = int(math.isqrt(token_count))
    if side * side != token_count:
        raise ValueError("token-count must be a perfect square.")
    axis = (torch.arange(side, dtype=torch.float32) + 0.5) / side
    rows, cols = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((2 * rows - 1, 2 * cols - 1), dim=-1).reshape(-1, 2)


class LearnableTopoCoarsen(nn.Module):
    """Training may use topology targets; forward inference accepts images only."""

    def __init__(
        self, dim: int, token_count: int, temperature: float
    ) -> None:
        super().__init__()
        self.token_count = token_count
        self.temperature = temperature
        self.stem = PatchStem(dim)
        hidden = max(32, dim)
        self.assignment_net = nn.Sequential(
            nn.Conv2d(dim + 2, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, token_count, 1),
        )
        edge_hidden = max(16, dim // 2)
        self.edge_net = nn.Sequential(
            nn.Conv2d(3 * dim, edge_hidden, 1),
            nn.GELU(),
            nn.Conv2d(edge_hidden, 1, 1),
        )
        self.project = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.head = CoarseGraphHead(dim, token_count)
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)
        self.register_buffer("anchors", grid_anchors(token_count), persistent=False)

    def edge_logits(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        return self.edge_net(
            torch.cat((first, second, torch.abs(first - second)), dim=1)
        ).squeeze(1)

    def predicted_coarse_graph(
        self,
        assignment: torch.Tensor,
        horizontal: torch.Tensor,
        vertical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = assignment.shape[0]
        assignment_map = assignment.reshape(batch, 16, 16, self.token_count)
        left = assignment_map[:, :, :-1, :].reshape(
            batch, -1, self.token_count
        )
        right = assignment_map[:, :, 1:, :].reshape(
            batch, -1, self.token_count
        )
        top = assignment_map[:, :-1, :, :].reshape(
            batch, -1, self.token_count
        )
        bottom = assignment_map[:, 1:, :, :].reshape(
            batch, -1, self.token_count
        )
        horizontal_weight = torch.sigmoid(horizontal).reshape(batch, -1, 1)
        vertical_weight = torch.sigmoid(vertical).reshape(batch, -1, 1)
        coarse = torch.bmm(
            (left * horizontal_weight).transpose(1, 2), right
        )
        coarse = coarse + torch.bmm(
            (right * horizontal_weight).transpose(1, 2), left
        )
        coarse = coarse + torch.bmm(
            (top * vertical_weight).transpose(1, 2), bottom
        )
        coarse = coarse + torch.bmm(
            (bottom * vertical_weight).transpose(1, 2), top
        )
        identity = torch.eye(
            self.token_count, device=coarse.device, dtype=coarse.dtype
        ).unsqueeze(0)
        coarse = coarse + identity
        transition = coarse / coarse.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        reachability = identity.expand(batch, -1, -1)
        power = identity.expand(batch, -1, -1)
        for _ in range(6):
            power = torch.bmm(power, transition)
            reachability = reachability + power
        return coarse, reachability

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.stem(image)
        batch = features.shape[0]
        coordinate_map = self.coordinates.transpose(0, 1).reshape(1, 2, 16, 16)
        coordinate_map = coordinate_map.expand(batch, -1, -1, -1)
        assignment_logits = self.assignment_net(
            torch.cat((features, coordinate_map.to(features.dtype)), dim=1)
        )
        fine_coords = self.coordinates.unsqueeze(1)
        anchor_coords = self.anchors.unsqueeze(0)
        anchor_bias = -4.0 * torch.sum(
            (fine_coords - anchor_coords) ** 2, dim=-1
        )
        assignment_logits = assignment_logits + anchor_bias.transpose(
            0, 1
        ).reshape(1, self.token_count, 16, 16)
        soft_assignment = torch.softmax(
            assignment_logits.flatten(2).transpose(1, 2) / self.temperature,
            dim=-1,
        )
        hard_assignment = F.one_hot(
            torch.argmax(soft_assignment, dim=-1),
            num_classes=self.token_count,
        ).to(soft_assignment.dtype)
        # Straight-through hard coarsening: the forward graph is sparse, while
        # gradients still flow through the soft assignment probabilities.
        assignment = (
            hard_assignment
            + soft_assignment
            - soft_assignment.detach()
        )

        horizontal_logits = self.edge_logits(
            features[:, :, :, :-1], features[:, :, :, 1:]
        )
        vertical_logits = self.edge_logits(
            features[:, :, :-1, :], features[:, :, 1:, :]
        )
        adjacency, reachability = self.predicted_coarse_graph(
            assignment, horizontal_logits, vertical_logits
        )

        flat_features = features.flatten(2).transpose(1, 2)
        mass = assignment.sum(dim=1).clamp_min(1e-6)
        tokens = torch.bmm(assignment.transpose(1, 2), flat_features)
        tokens = tokens / mass.unsqueeze(-1)
        coordinate_bank = self.coordinates.unsqueeze(0).expand(batch, -1, -1)
        centroids = torch.bmm(assignment.transpose(1, 2), coordinate_bank)
        centroids = centroids / mass.unsqueeze(-1)
        metadata = torch.cat(
            (centroids, (mass / 256.0).unsqueeze(-1)), dim=-1
        )
        connected = self.head(
            self.project(tokens),
            metadata,
            adjacency,
            reachability,
        )
        return {
            "connected": connected,
            "assignment_logits": assignment_logits,
            "assignment": assignment,
            "horizontal_logits": horizontal_logits,
            "vertical_logits": vertical_logits,
            "adjacency": adjacency,
            "reachability": reachability,
            "mass": mass,
        }


def balanced_binary_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    positive = target
    negative = ~target
    losses = []
    if torch.any(positive):
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits[positive], torch.ones_like(logits[positive])
            )
        )
    if torch.any(negative):
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits[negative], torch.zeros_like(logits[negative])
            )
        )
    return torch.stack(losses).mean()


def fine_edge_targets(
    patch_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal = patch_ids[:, :, :-1] == patch_ids[:, :, 1:]
    vertical = patch_ids[:, :-1, :] == patch_ids[:, 1:, :]
    return horizontal, vertical


def row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def learned_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    task = F.binary_cross_entropy_with_logits(
        output["connected"], batch["connected"]
    )
    assignment_target = batch["assignment_topology"]
    assignment = F.cross_entropy(
        output["assignment_logits"], assignment_target
    )
    horizontal_target, vertical_target = fine_edge_targets(batch["patch_ids"])
    edge = 0.5 * (
        balanced_binary_loss(output["horizontal_logits"], horizontal_target)
        + balanced_binary_loss(output["vertical_logits"], vertical_target)
    )
    target_adjacency = batch["adjacency_topology"].to(
        output["adjacency"].dtype
    )
    target_reachability = batch["reachability_topology"].to(
        output["reachability"].dtype
    )
    graph = 0.5 * (
        F.mse_loss(
            row_normalize(output["adjacency"]),
            row_normalize(target_adjacency),
        )
        + F.mse_loss(
            row_normalize(output["reachability"]),
            row_normalize(target_reachability),
        )
    )
    desired_mass = torch.full_like(
        output["mass"], 256.0 / args.token_count
    )
    balance = F.mse_loss(
        output["mass"] / 256.0, desired_mass / 256.0
    )
    total = (
        task
        + args.assignment_weight * assignment
        + args.edge_weight * edge
        + args.graph_weight * graph
        + args.balance_weight * balance
    )
    return total, {
        "task_loss": float(task.detach()),
        "assignment_loss": float(assignment.detach()),
        "edge_loss": float(edge.detach()),
        "graph_loss": float(graph.detach()),
        "balance_loss": float(balance.detach()),
    }


def artifact_name(model_name: str) -> str:
    if model_name in {"grid", "conv"}:
        return "grid"
    if model_name == "topology_oracle":
        return "topology"
    raise ValueError(f"No fixed artifact for {model_name}.")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    truths = []
    predictions = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        if model_name == "learned_topocoarsen":
            # Strict inference contract: no patch_ids or Oracle artifacts are passed.
            logits = model(image)["connected"]
        else:
            name = artifact_name(model_name)
            logits = model(
                image,
                batch[f"assignment_{name}"].to(device, non_blocking=True),
                batch[f"adjacency_{name}"].to(device, non_blocking=True),
                batch[f"reachability_{name}"].to(device, non_blocking=True),
            )
        truths.append(batch["connected"].bool())
        predictions.append((torch.sigmoid(logits) >= 0.5).cpu())
    truth = torch.cat(truths)
    prediction = torch.cat(predictions)
    positive = float((prediction[truth] == truth[truth]).float().mean())
    negative = float((prediction[~truth] == truth[~truth]).float().mean())
    return {
        "accuracy": float((prediction == truth).float().mean()),
        "balanced_accuracy": 0.5 * (positive + negative),
        "positive_accuracy": positive,
        "negative_accuracy": negative,
    }


def hard_reachability(
    hard_assignment: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    token_count: int,
) -> np.ndarray:
    graph = np.eye(token_count, dtype=bool)
    assignment = hard_assignment.reshape(16, 16)
    for row in range(16):
        for col in range(15):
            if horizontal[row, col]:
                first = assignment[row, col]
                second = assignment[row, col + 1]
                graph[first, second] = True
                graph[second, first] = True
    for row in range(15):
        for col in range(16):
            if vertical[row, col]:
                first = assignment[row, col]
                second = assignment[row + 1, col]
                graph[first, second] = True
                graph[second, first] = True
    for pivot in range(token_count):
        graph |= graph[:, pivot, None] & graph[pivot, None, :]
    return graph


@torch.no_grad()
def learned_diagnostics(
    model: LearnableTopoCoarsen,
    loader: DataLoader,
    device: torch.device,
    token_count: int,
) -> dict[str, float]:
    model.eval()
    assignment_correct = 0
    assignment_total = 0
    edge_positive_correct = 0
    edge_positive_total = 0
    edge_negative_correct = 0
    edge_negative_total = 0
    structural_correct = 0
    samples = 0
    empty_clusters = 0
    for batch in loader:
        image = batch["image"].to(device)
        output = model(image)
        predicted_assignment = torch.argmax(
            output["assignment_logits"], dim=1
        ).cpu()
        target_assignment = batch["assignment_topology"]
        assignment_correct += int(
            (predicted_assignment == target_assignment).sum()
        )
        assignment_total += int(target_assignment.numel())

        horizontal_target, vertical_target = fine_edge_targets(
            batch["patch_ids"]
        )
        horizontal_prediction = (
            torch.sigmoid(output["horizontal_logits"]).cpu() >= 0.5
        )
        vertical_prediction = (
            torch.sigmoid(output["vertical_logits"]).cpu() >= 0.5
        )
        for prediction, target in (
            (horizontal_prediction, horizontal_target),
            (vertical_prediction, vertical_target),
        ):
            edge_positive_correct += int((prediction[target] == target[target]).sum())
            edge_positive_total += int(target.sum())
            edge_negative_correct += int(
                (prediction[~target] == target[~target]).sum()
            )
            edge_negative_total += int((~target).sum())

        for index in range(image.shape[0]):
            hard = predicted_assignment[index].numpy().reshape(-1)
            empty_clusters += token_count - len(np.unique(hard))
            closure = hard_reachability(
                hard,
                horizontal_prediction[index].numpy(),
                vertical_prediction[index].numpy(),
                token_count,
            )
            endpoints = endpoint_patch_indices(
                batch["endpoint_features"][index].numpy()
            )
            first_cluster = int(hard[endpoints[0]])
            second_cluster = int(hard[endpoints[-1]])
            prediction = int(closure[first_cluster, second_cluster])
            structural_correct += int(
                prediction == int(batch["connected"][index])
            )
            samples += 1
    positive_accuracy = edge_positive_correct / max(edge_positive_total, 1)
    negative_accuracy = edge_negative_correct / max(edge_negative_total, 1)
    return {
        "assignment_pixel_accuracy": assignment_correct
        / max(assignment_total, 1),
        "edge_balanced_accuracy": 0.5
        * (positive_accuracy + negative_accuracy),
        "hard_graph_endpoint_reachability_accuracy": structural_correct
        / max(samples, 1),
        "mean_empty_hard_clusters": empty_clusters / max(samples, 1),
        "inference_uses_patch_ids": False,
    }


def train_one(
    model_name: str,
    args: argparse.Namespace,
    train_dataset: LearnedDataset,
    dev_dataset: LearnedDataset,
    test_dataset: LearnedDataset,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float] | None]:
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
    if model_name == "learned_topocoarsen":
        model: nn.Module = LearnableTopoCoarsen(
            args.dim, args.token_count, args.temperature
        )
    else:
        model = TopoCoarsenModel(model_name, args.dim, args.token_count)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_dev = -1.0
    best_path = output_dir / f"{model_name}_seed{args.seed}.pt"
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        auxiliary_sums: dict[str, float] = {}
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["connected"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                if model_name == "learned_topocoarsen":
                    prediction = model(image)
                    device_batch = {
                        key: value.to(device, non_blocking=True)
                        for key, value in batch.items()
                        if key
                        in {
                            "connected",
                            "patch_ids",
                            "assignment_topology",
                            "adjacency_topology",
                            "reachability_topology",
                        }
                    }
                    loss, parts = learned_loss(prediction, device_batch, args)
                else:
                    name = artifact_name(model_name)
                    logits = model(
                        image,
                        batch[f"assignment_{name}"].to(
                            device, non_blocking=True
                        ),
                        batch[f"adjacency_{name}"].to(
                            device, non_blocking=True
                        ),
                        batch[f"reachability_{name}"].to(
                            device, non_blocking=True
                        ),
                    )
                    loss = F.binary_cross_entropy_with_logits(logits, target)
                    parts = {"task_loss": float(loss.detach())}
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batch_count += 1
            for key, value in parts.items():
                auxiliary_sums[key] = auxiliary_sums.get(key, 0.0) + value

        dev_metrics = evaluate(model, model_name, dev_loader, device)
        record = {
            "seed": args.seed,
            "model": model_name,
            "epoch": epoch,
            "train_loss": loss_sum / max(batch_count, 1),
            "dev_balanced_accuracy": dev_metrics["balanced_accuracy"],
            "dev_positive_accuracy": dev_metrics["positive_accuracy"],
            "dev_negative_accuracy": dev_metrics["negative_accuracy"],
            **{
                key: value / max(batch_count, 1)
                for key, value in auxiliary_sums.items()
            },
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"LEARNED_TRAIN seed={args.seed} model={model_name} "
            f"epoch={epoch}/{args.epochs} loss={record['train_loss']:.4f} "
            f"dev_bal={dev_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        if dev_metrics["balanced_accuracy"] > best_dev:
            best_dev = dev_metrics["balanced_accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "model_name": model_name,
                    "dev_metrics": dev_metrics,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, model_name, test_loader, device)
    summary = {
        "seed": args.seed,
        "model": model_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_dev_balanced_accuracy": best_dev,
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_positive_accuracy": test_metrics["positive_accuracy"],
        "test_negative_accuracy": test_metrics["negative_accuracy"],
        "training_seconds": time.time() - started,
        "test_inference_uses_patch_ids": model_name == "topology_oracle",
    }
    diagnostic = (
        learned_diagnostics(
            model, test_loader, device, args.token_count
        )
        if model_name == "learned_topocoarsen"
        else None
    )
    return summary, history, diagnostic


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_result(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    shortcut_accuracy: float,
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    by_model = {row["model"]: row for row in summaries}
    baseline_name = max(
        ("grid", "conv"),
        key=lambda name: by_model[name]["test_balanced_accuracy"],
    )
    baseline = by_model[baseline_name]["test_balanced_accuracy"]
    learned = by_model["learned_topocoarsen"]["test_balanced_accuracy"]
    oracle = by_model["topology_oracle"]["test_balanced_accuracy"]
    gain = learned - baseline
    result = {
        "experiment_id": "learned_topocoarsen_image_only",
        "status": "completed",
        "split_manifest": split_manifest,
        "coordinate_shortcut_accuracy": shortcut_accuracy,
        "test_scores": {
            name: row["test_balanced_accuracy"] for name, row in by_model.items()
        },
        "strongest_image_only_baseline": baseline_name,
        "learned_gain_over_baseline": gain,
        "oracle_gap_remaining": oracle - learned,
        "learned_diagnostics": diagnostics,
        "gates": {
            "dev_test_separated": True,
            "learned_test_inference_image_only": not diagnostics[
                "inference_uses_patch_ids"
            ],
            "learned_gain_at_least_3pp": gain >= 0.03,
            "hard_graph_reachability_at_least_70pct": diagnostics[
                "hard_graph_endpoint_reachability_accuracy"
            ]
            >= 0.70,
        },
        "verdict": (
            "PROCEED_TO_MULTISEED_OOD"
            if gain >= 0.03
            and diagnostics["hard_graph_endpoint_reachability_accuracy"] >= 0.70
            else (
                "INCONCLUSIVE_REDESIGN_REQUIRED"
                if gain > 0
                else "DO_NOT_SCALE_DIRECT_ASSIGNMENT"
            )
        ),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    names = [row["model"] for row in summaries]
    values = [row["test_balanced_accuracy"] for row in summaries]
    colors = [
        "#E45756"
        if name == "learned_topocoarsen"
        else ("#72B7B2" if name == "topology_oracle" else "#4C78A8")
        for name in names
    ]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(names, values, color=colors)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("held-out test connected balanced accuracy")
    ax.set_title("Learned TopoCoarsen: image-only inference")
    ax.tick_params(axis="x", labelrotation=16)
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(output_dir / "learned_topocoarsen_comparison.png", dpi=180)
    plt.close(fig)
    return result


def run_self_test() -> None:
    model = LearnableTopoCoarsen(32, 16, 0.6)
    image = torch.rand(2, 3, 64, 64)
    output = model(image)
    assert output["connected"].shape == (2,)
    assert output["assignment"].shape == (2, 256, 16)
    assert output["adjacency"].shape == (2, 16, 16)
    assert torch.allclose(
        output["assignment"].sum(dim=-1),
        torch.ones(2, 256),
        atol=1e-5,
    )
    patch_ids = torch.zeros(2, 16, 16, dtype=torch.long)
    target_assignment = torch.arange(16).reshape(1, 4, 4)
    target_assignment = target_assignment.repeat_interleave(4, 1)
    target_assignment = target_assignment.repeat_interleave(4, 2)
    batch = {
        "connected": torch.tensor([0.0, 1.0]),
        "patch_ids": patch_ids,
        "assignment_topology": target_assignment.repeat(2, 1, 1),
        "adjacency_topology": torch.eye(16).repeat(2, 1, 1),
        "reachability_topology": torch.eye(16).repeat(2, 1, 1),
    }
    args = argparse.Namespace(
        assignment_weight=0.3,
        edge_weight=0.2,
        graph_weight=0.5,
        balance_weight=0.05,
        token_count=16,
    )
    loss, parts = learned_loss(output, batch, args)
    assert torch.isfinite(loss)
    assert set(parts) == {
        "task_loss",
        "assignment_loss",
        "edge_loss",
        "graph_loss",
        "balance_loss",
    }
    print("LEARNED_TOPOCOARSEN_SELF_TEST_PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    unknown = sorted(set(args.models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if set(args.models) != set(MODEL_NAMES):
        raise ValueError(f"Gate run requires exactly {MODEL_NAMES}.")
    if not (0 < args.dev_size < args.val_size):
        raise ValueError("dev-size must be strictly between 0 and val-size.")
    if args.dev_size % 2 or (args.val_size - args.dev_size) % 2:
        raise ValueError("dev and test sizes must be even for balanced labels.")
    grid_anchors(args.token_count)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "args": vars(args),
            },
            default=str,
        ),
        flush=True,
    )

    train_arrays = load_or_generate(
        cache_dir, "train", args.train_size, args.seed
    )
    validation_arrays = load_or_generate(
        cache_dir, "val", args.val_size, args.seed + 1_000_000
    )
    dev_arrays = slice_arrays(validation_arrays, 0, args.dev_size)
    test_arrays = slice_arrays(
        validation_arrays, args.dev_size, args.val_size
    )
    shortcut = coordinate_shortcut_accuracy(train_arrays, test_arrays)
    train_artifacts, _ = build_artifacts(train_arrays, args.token_count)
    dev_artifacts, _ = build_artifacts(dev_arrays, args.token_count)
    test_artifacts, _ = build_artifacts(test_arrays, args.token_count)
    split_manifest = {
        "train": {
            "count": args.train_size,
            "source": f"train_n{args.train_size}_seed{args.seed}.npz",
        },
        "dev": {
            "count": args.dev_size,
            "source": "validation indices [0, dev_size)",
        },
        "test": {
            "count": args.val_size - args.dev_size,
            "source": "validation indices [dev_size, val_size)",
            "used_for_checkpoint_selection": False,
        },
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"LEARNED_SHORTCUT test_distance_accuracy={shortcut:.4f}",
        flush=True,
    )

    train_dataset = LearnedDataset(train_arrays, train_artifacts)
    dev_dataset = LearnedDataset(dev_arrays, dev_artifacts)
    test_dataset = LearnedDataset(test_arrays, test_artifacts)
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    learned_diagnostic: dict[str, float] | None = None
    for model_name in args.models:
        summary, records, diagnostic = train_one(
            model_name,
            args,
            train_dataset,
            dev_dataset,
            test_dataset,
            output_dir,
            device,
        )
        summaries.append(summary)
        history.extend(records)
        if diagnostic is not None:
            learned_diagnostic = diagnostic
    if learned_diagnostic is None:
        raise AssertionError("Missing learned model diagnostics.")
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "history.csv", history)
    result = render_result(
        output_dir,
        summaries,
        learned_diagnostic,
        shortcut,
        split_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

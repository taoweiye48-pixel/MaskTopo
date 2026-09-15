from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
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
    ConvConnector,
    PatchStem,
    coordinate_shortcut_accuracy,
    load_or_generate,
    set_seed,
)
from topobridge_oracle import endpoint_patch_indices, topology_critical_scores


MODEL_NAMES = [
    "grid",
    "conv",
    "component_oracle",
    "topology_oracle",
    "topology_coords_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle validation for topology-preserving graph coarsening."
    )
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--output", type=Path, default=Path("results_topocoarsen"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def grid_assignment(token_count: int) -> np.ndarray:
    side = int(math.isqrt(token_count))
    if side * side != token_count or 16 % side != 0:
        raise ValueError("token-count must be a square whose side divides 16.")
    block = 16 // side
    rows, cols = np.indices((16, 16))
    return ((rows // block) * side + cols // block).astype(np.int16)


def allocate_cluster_counts(
    patch_ids: np.ndarray,
    token_count: int,
    critical_scores: np.ndarray | None,
) -> dict[int, int]:
    groups = [int(item) for item in np.unique(patch_ids)]
    if token_count < len(groups):
        raise ValueError("Token budget is smaller than the number of semantic groups.")
    sizes = {group: int(np.sum(patch_ids == group)) for group in groups}
    counts = {group: 1 for group in groups}
    weights = {}
    for group in groups:
        weight = float(sizes[group])
        if critical_scores is not None and group > 0:
            mask = patch_ids.reshape(-1) == group
            weight += 2.0 * float(np.count_nonzero(critical_scores[mask] > 0))
        weights[group] = weight
    while sum(counts.values()) < token_count:
        candidates = [
            group for group in groups if counts[group] < sizes[group]
        ]
        if not candidates:
            raise RuntimeError("Could not allocate all coarse tokens.")
        chosen = max(
            candidates,
            key=lambda group: (
                weights[group] / counts[group],
                sizes[group],
                -group,
            ),
        )
        counts[chosen] += 1
    return counts


def farthest_seeds(
    nodes: np.ndarray,
    count: int,
    critical_scores: np.ndarray | None,
) -> list[int]:
    flat_nodes = nodes[:, 0] * 16 + nodes[:, 1]
    selected: list[int] = []
    if critical_scores is not None:
        ranked = sorted(
            (int(item) for item in flat_nodes if critical_scores[int(item)] > 0),
            key=lambda item: (-float(critical_scores[item]), item),
        )
        critical_quota = min(len(ranked), max(1, count // 2))
        selected.extend(ranked[:critical_quota])
    if not selected:
        centroid = np.mean(nodes, axis=0)
        distances = np.sum((nodes - centroid) ** 2, axis=1)
        selected.append(int(flat_nodes[int(np.argmin(distances))]))
    candidate_rc = nodes.astype(np.float64)
    while len(selected) < count:
        selected_rc = np.array(
            [(item // 16, item % 16) for item in selected], dtype=np.float64
        )
        distances = np.min(
            np.sum(
                (candidate_rc[:, None, :] - selected_rc[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        for item in selected:
            position = np.flatnonzero(flat_nodes == item)
            if position.size:
                distances[position[0]] = -1.0
        selected.append(int(flat_nodes[int(np.argmax(distances))]))
    return selected


def assign_group(nodes: np.ndarray, seeds: list[int]) -> dict[int, int]:
    node_set = {int(row) * 16 + int(col) for row, col in nodes}
    labels = {node: -1 for node in node_set}
    distances = {node: 10**9 for node in node_set}
    queue: deque[int] = deque()
    for label, seed in enumerate(seeds):
        labels[seed] = label
        distances[seed] = 0
        queue.append(seed)
    while queue:
        node = queue.popleft()
        row, col = divmod(node, 16)
        for nr, nc in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            neighbor = nr * 16 + nc
            if 0 <= nr < 16 and 0 <= nc < 16 and neighbor in node_set:
                proposed = distances[node] + 1
                if proposed < distances[neighbor]:
                    distances[neighbor] = proposed
                    labels[neighbor] = labels[node]
                    queue.append(neighbor)
    seed_rc = np.array([(item // 16, item % 16) for item in seeds])
    for node, label in list(labels.items()):
        if label >= 0:
            continue
        point = np.array([node // 16, node % 16])
        labels[node] = int(np.argmin(np.sum((seed_rc - point) ** 2, axis=1)))
    return labels


def canonicalize_assignment(assignment: np.ndarray, token_count: int) -> np.ndarray:
    centroids = []
    for cluster in range(token_count):
        coordinates = np.argwhere(assignment == cluster)
        if coordinates.size == 0:
            raise AssertionError("Empty coarse cluster.")
        centroid = np.mean(coordinates, axis=0)
        centroids.append((float(centroid[0]), float(centroid[1]), cluster))
    order = [item[2] for item in sorted(centroids)]
    remap = {old: new for new, old in enumerate(order)}
    return np.vectorize(remap.__getitem__, otypes=[np.int16])(assignment)


def oracle_assignment(
    patch_ids: np.ndarray, token_count: int, bridge_aware: bool
) -> np.ndarray:
    scores = topology_critical_scores(patch_ids) if bridge_aware else None
    counts = allocate_cluster_counts(patch_ids, token_count, scores)
    assignment = np.full((16, 16), -1, dtype=np.int16)
    next_cluster = 0
    for group in sorted(counts):
        nodes = np.argwhere(patch_ids == group)
        group_scores = scores if bridge_aware and group > 0 else None
        seeds = farthest_seeds(nodes, counts[group], group_scores)
        local = assign_group(nodes, seeds)
        for flat, local_label in local.items():
            assignment[flat // 16, flat % 16] = next_cluster + local_label
        next_cluster += counts[group]
    if np.any(assignment < 0) or next_cluster != token_count:
        raise AssertionError("Oracle coarsening did not cover the full grid.")
    return canonicalize_assignment(assignment, token_count)


def coarse_graph(
    assignment: np.ndarray,
    patch_ids: np.ndarray,
    token_count: int,
    topology_aware: bool,
) -> tuple[np.ndarray, np.ndarray]:
    adjacency = np.eye(token_count, dtype=np.uint8)
    for row in range(16):
        for col in range(16):
            for nr, nc in ((row + 1, col), (row, col + 1)):
                if nr >= 16 or nc >= 16:
                    continue
                valid = True
                if topology_aware:
                    first = int(patch_ids[row, col])
                    second = int(patch_ids[nr, nc])
                    valid = (first == 0 and second == 0) or (
                        first > 0 and first == second
                    )
                if not valid:
                    continue
                left = int(assignment[row, col])
                right = int(assignment[nr, nc])
                adjacency[left, right] = 1
                adjacency[right, left] = 1
    reachability = adjacency.astype(bool)
    for pivot in range(token_count):
        reachability |= (
            reachability[:, pivot, None] & reachability[pivot, None, :]
        )
    return adjacency, reachability.astype(np.uint8)


def endpoint_reachability(
    assignment: np.ndarray,
    reachability: np.ndarray,
    endpoint_features: np.ndarray,
) -> int:
    endpoints = endpoint_patch_indices(endpoint_features)
    if len(endpoints) == 1:
        return 1
    first = int(assignment[endpoints[0] // 16, endpoints[0] % 16])
    second = int(assignment[endpoints[1] // 16, endpoints[1] % 16])
    return int(reachability[first, second])


def build_artifacts(
    arrays: dict[str, np.ndarray], token_count: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample_count = arrays["images"].shape[0]
    names = ("grid", "component", "topology")
    assignments = {
        name: np.empty((sample_count, 16, 16), dtype=np.int16) for name in names
    }
    adjacency = {
        name: np.empty(
            (sample_count, token_count, token_count), dtype=np.uint8
        )
        for name in names
    }
    reachability = {
        name: np.empty(
            (sample_count, token_count, token_count), dtype=np.uint8
        )
        for name in names
    }
    grid = grid_assignment(token_count)
    structural_correct = {name: 0 for name in names}
    critical_cluster_sizes = {"component": [], "topology": []}
    mixed_cluster_count = {"component": 0, "topology": 0}

    for index in range(sample_count):
        patch_ids = arrays["patch_ids"][index]
        current = {
            "grid": grid,
            "component": oracle_assignment(patch_ids, token_count, False),
            "topology": oracle_assignment(patch_ids, token_count, True),
        }
        for name in names:
            assignments[name][index] = current[name]
            graph, closure = coarse_graph(
                current[name],
                patch_ids,
                token_count,
                topology_aware=name != "grid",
            )
            adjacency[name][index] = graph
            reachability[name][index] = closure
            prediction = endpoint_reachability(
                current[name], closure, arrays["endpoint_features"][index]
            )
            structural_correct[name] += int(
                prediction == int(arrays["connected"][index])
            )

        critical = np.flatnonzero(topology_critical_scores(patch_ids) > 0)
        for name in ("component", "topology"):
            cluster_sizes = np.bincount(
                current[name].reshape(-1), minlength=token_count
            )
            if critical.size:
                critical_clusters = current[name].reshape(-1)[critical]
                critical_cluster_sizes[name].extend(
                    cluster_sizes[critical_clusters].tolist()
                )
            for cluster in range(token_count):
                values = np.unique(patch_ids[current[name] == cluster])
                if values.size != 1:
                    mixed_cluster_count[name] += 1

    artifacts: dict[str, np.ndarray] = {}
    for name in names:
        artifacts[f"assignment_{name}"] = assignments[name]
        artifacts[f"adjacency_{name}"] = adjacency[name]
        artifacts[f"reachability_{name}"] = reachability[name]
    diagnostics = {
        "samples": int(sample_count),
        "token_count": int(token_count),
        "all_patch_coverage": 1.0,
        "structural_endpoint_reachability_accuracy": {
            name: structural_correct[name] / sample_count for name in names
        },
        "mixed_semantic_clusters": mixed_cluster_count,
        "mean_cluster_size_at_critical_nodes": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in critical_cluster_sizes.items()
        },
    }
    return artifacts, diagnostics


class CoarsenDataset(Dataset):
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
        }
        for key, values in self.artifacts.items():
            item[key] = torch.from_numpy(values[index].astype(np.int64))
        return item


def fine_coordinates() -> torch.Tensor:
    axis = (torch.arange(16, dtype=torch.float32) + 0.5) / 16
    rows, cols = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((2 * rows - 1, 2 * cols - 1), dim=-1).reshape(-1, 2)


class GraphMixLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.local_message = nn.Linear(dim, dim)
        self.component_message = nn.Linear(dim, dim)
        self.local_gate = nn.Parameter(torch.zeros(()))
        self.component_gate = nn.Parameter(torch.zeros(()))
        # Forward-only controls used by parameter-matched branch ablations.
        # They are ordinary Python attributes, so the historical state-dict
        # schema and parameter count remain unchanged.
        self.use_local = True
        self.use_reachability = True

    def forward(
        self,
        tokens: torch.Tensor,
        adjacency: torch.Tensor,
        reachability: torch.Tensor,
    ) -> torch.Tensor:
        local_weights = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1)
        global_weights = reachability / reachability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1)
        mixed = tokens
        if self.use_local:
            local = torch.bmm(local_weights, tokens)
            mixed = mixed + torch.tanh(self.local_gate) * self.local_message(local)
        if self.use_reachability:
            component = torch.bmm(global_weights, tokens)
            mixed = mixed + (
                torch.tanh(self.component_gate)
                * self.component_message(component)
            )
        return mixed


class CoarseGraphHead(nn.Module):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__()
        self.metadata = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.graph_layers = nn.ModuleList([GraphMixLayer(dim)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=4 * dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 1)
        self.token_count = token_count

    def forward(
        self,
        tokens: torch.Tensor,
        metadata: torch.Tensor,
        adjacency: torch.Tensor,
        reachability: torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens + self.metadata(metadata)
        for layer in self.graph_layers:
            tokens = layer(tokens, adjacency, reachability)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1))
        return self.classifier(self.norm(encoded[:, 0])).squeeze(-1)


class TopoCoarsenModel(nn.Module):
    def __init__(self, name: str, dim: int, token_count: int) -> None:
        super().__init__()
        self.name = name
        self.token_count = token_count
        self.stem = PatchStem(dim)
        side = int(math.isqrt(token_count))
        self.conv = ConvConnector(dim, side) if name == "conv" else None
        self.project = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.head = CoarseGraphHead(dim, token_count)
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def pool(
        self, features: torch.Tensor, assignment: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = features.flatten(2).transpose(1, 2)
        one_hot = F.one_hot(
            assignment.reshape(assignment.shape[0], -1),
            num_classes=self.token_count,
        ).to(flat.dtype)
        counts = one_hot.sum(dim=1).clamp_min(1)
        tokens = torch.bmm(one_hot.transpose(1, 2), flat) / counts.unsqueeze(-1)
        coordinate_bank = self.coordinates.unsqueeze(0).expand(
            features.shape[0], -1, -1
        )
        centroids = (
            torch.bmm(one_hot.transpose(1, 2), coordinate_bank)
            / counts.unsqueeze(-1)
        )
        mass = (counts / 256.0).unsqueeze(-1)
        return tokens, torch.cat((centroids, mass), dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        assignment: torch.Tensor,
        adjacency: torch.Tensor,
        reachability: torch.Tensor,
    ) -> torch.Tensor:
        features = self.stem(image)
        if self.name == "conv":
            if self.conv is None:
                raise AssertionError("Missing convolutional connector.")
            compressed, _ = self.conv(features)
            tokens = compressed.flatten(2).transpose(1, 2)
            side = int(math.isqrt(self.token_count))
            axis = (torch.arange(side, device=image.device, dtype=tokens.dtype) + 0.5) / side
            rows, cols = torch.meshgrid(axis, axis, indexing="ij")
            centroids = torch.stack(
                (2 * rows - 1, 2 * cols - 1), dim=-1
            ).reshape(1, self.token_count, 2)
            centroids = centroids.expand(image.shape[0], -1, -1)
            mass = torch.full(
                (image.shape[0], self.token_count, 1),
                1.0 / self.token_count,
                device=image.device,
                dtype=tokens.dtype,
            )
            metadata = torch.cat((centroids, mass), dim=-1)
        else:
            tokens, metadata = self.pool(features, assignment)
            if self.name == "topology_coords_only":
                tokens = torch.zeros_like(tokens)
        return self.head(
            self.project(tokens),
            metadata,
            adjacency.to(tokens.dtype),
            reachability.to(tokens.dtype),
        )


def artifact_name(model_name: str) -> str:
    if model_name in {"grid", "conv"}:
        return "grid"
    if model_name == "component_oracle":
        return "component"
    return "topology"


@torch.no_grad()
def evaluate(
    model: TopoCoarsenModel,
    loader: DataLoader,
    device: torch.device,
    override_artifact: str | None = None,
) -> dict[str, float]:
    model.eval()
    truths = []
    predictions = []
    for batch in loader:
        name = override_artifact or artifact_name(model.name)
        image = batch["image"].to(device, non_blocking=True)
        assignment = batch[f"assignment_{name}"].to(device, non_blocking=True)
        adjacency = batch[f"adjacency_{name}"].to(device, non_blocking=True)
        reachability = batch[f"reachability_{name}"].to(
            device, non_blocking=True
        )
        logits = model(image, assignment, adjacency, reachability)
        truths.append(batch["connected"].bool())
        predictions.append((torch.sigmoid(logits) >= 0.5).cpu())
    truth = torch.cat(truths)
    prediction = torch.cat(predictions)
    positive = float((prediction[truth] == truth[truth]).float().mean())
    negative = float((prediction[~truth] == truth[~truth]).float().mean())
    return {
        "connected_accuracy": float((prediction == truth).float().mean()),
        "connected_balanced_accuracy": 0.5 * (positive + negative),
        "positive_accuracy": positive,
        "negative_accuracy": negative,
    }


def train_one(
    model_name: str,
    args: argparse.Namespace,
    train_dataset: CoarsenDataset,
    val_dataset: CoarsenDataset,
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_dataset,
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
    best_metric = -1.0
    best_path = output / f"{model_name}_seed{args.seed}.pt"
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for batch in train_loader:
            name = artifact_name(model_name)
            image = batch["image"].to(device, non_blocking=True)
            target = batch["connected"].to(device, non_blocking=True)
            assignment = batch[f"assignment_{name}"].to(
                device, non_blocking=True
            )
            adjacency = batch[f"adjacency_{name}"].to(device, non_blocking=True)
            reachability = batch[f"reachability_{name}"].to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(image, assignment, adjacency, reachability)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batch_count += 1
        metrics = evaluate(model, val_loader, device)
        record = {
            "seed": args.seed,
            "model": model_name,
            "epoch": epoch,
            "train_loss": loss_sum / max(batch_count, 1),
            **metrics,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"COARSEN_TRAIN seed={args.seed} model={model_name} "
            f"epoch={epoch}/{args.epochs} loss={record['train_loss']:.4f} "
            f"conn_bal={metrics['connected_balanced_accuracy']:.4f}",
            flush=True,
        )
        if metrics["connected_balanced_accuracy"] > best_metric:
            best_metric = metrics["connected_balanced_accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "model_name": model_name,
                    "metrics": metrics,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, val_loader, device)
    summary: dict[str, Any] = {
        "seed": args.seed,
        "model": model_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch_connected_balanced_accuracy": best_metric,
        **final_metrics,
        "training_seconds": time.time() - started,
    }
    if model_name == "topology_oracle":
        for override in ("component", "grid"):
            metrics = evaluate(
                model, val_loader, device, override_artifact=override
            )
            summary[f"counterfactual_{override}_balanced_accuracy"] = metrics[
                "connected_balanced_accuracy"
            ]
    return summary, history


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
    output: Path,
    summaries: list[dict[str, Any]],
    shortcut_accuracy: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    by_model = {row["model"]: row for row in summaries}
    baseline_names = [name for name in ("grid", "conv") if name in by_model]
    strongest_baseline = max(
        baseline_names,
        key=lambda name: by_model[name]["connected_balanced_accuracy"],
    )
    baseline = by_model[strongest_baseline]["connected_balanced_accuracy"]
    component = by_model["component_oracle"]["connected_balanced_accuracy"]
    topology = by_model["topology_oracle"]["connected_balanced_accuracy"]
    coordinate_only = by_model["topology_coords_only"][
        "connected_balanced_accuracy"
    ]
    component_gain = component - baseline
    topology_gain = topology - baseline
    bridge_gain = topology - component
    structural = diagnostics["validation"][
        "structural_endpoint_reachability_accuracy"
    ]["topology"]

    leakage_pass = coordinate_only <= 0.60
    structural_pass = structural >= 0.99
    gain_pass = topology_gain >= 0.03
    if not leakage_pass:
        verdict = "INVALID_ORACLE_STRUCTURE_LEAKAGE"
    elif not structural_pass:
        verdict = "INVALID_TOPOLOGY_COARSENING"
    elif gain_pass and bridge_gain >= 0.01:
        verdict = "BRIDGE_AWARE_TOPOLOGY_COARSENING_SUPPORTED"
    elif gain_pass:
        verdict = "COMPONENT_COARSENING_SUPPORTED_BRIDGE_BONUS_UNPROVEN"
    elif max(component_gain, topology_gain) <= 0:
        verdict = "NO_GO_FOR_ORACLE_TOPOLOGY_COARSENING"
    else:
        verdict = "INCONCLUSIVE_REDESIGN_REQUIRED"

    result = {
        "experiment_id": "topocoarsen_oracle_upper_bound",
        "status": "completed",
        "coordinate_shortcut_accuracy": shortcut_accuracy,
        "diagnostics": diagnostics,
        "scores": {
            name: row["connected_balanced_accuracy"] for name, row in by_model.items()
        },
        "strongest_non_topology_baseline": strongest_baseline,
        "component_oracle_gain": component_gain,
        "topology_oracle_gain": topology_gain,
        "bridge_allocation_gain_over_component": bridge_gain,
        "topology_coordinate_only_accuracy": coordinate_only,
        "gates": {
            "dataset_shortcut_pass": shortcut_accuracy <= 0.60,
            "topology_structural_reachability_at_least_99pct": structural_pass,
            "coordinate_structure_leakage_pass": leakage_pass,
            "topology_gain_at_least_3pp": gain_pass,
        },
        "verdict": verdict,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    names = [row["model"] for row in summaries]
    values = [row["connected_balanced_accuracy"] for row in summaries]
    colors = ["#4C78A8" if "oracle" not in name else "#E45756" for name in names]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(names, values, color=colors)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("validation connected balanced accuracy")
    ax.set_title("TopoCoarsen Oracle: all 256 patches aggregated into 16 tokens")
    ax.tick_params(axis="x", labelrotation=17)
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(output / "topocoarsen_comparison.png", dpi=180)
    plt.close(fig)
    return result


def run_self_test() -> None:
    patch_ids = np.zeros((16, 16), dtype=np.int16)
    patch_ids[2:6, 2:7] = 1
    patch_ids[10:14, 9:14] = 2
    endpoint_same = np.array(
        [4 / 31, 4 / 31, 10 / 31, 10 / 31, 0, 0], dtype=np.float32
    )
    endpoint_different = np.array(
        [4 / 31, 4 / 31, 24 / 31, 24 / 31, 0, 0], dtype=np.float32
    )
    arrays = {
        "images": np.zeros((2, 3, 64, 64), dtype=np.uint8),
        "connected": np.array([1, 0], dtype=np.int64),
        "patch_ids": np.stack((patch_ids, patch_ids)),
        "endpoint_features": np.stack((endpoint_same, endpoint_different)),
    }
    artifacts, diagnostics = build_artifacts(arrays, 16)
    for name in ("grid", "component", "topology"):
        selected = artifacts[f"assignment_{name}"]
        assert np.all(np.sort(np.unique(selected)) == np.arange(16))
    assert diagnostics["all_patch_coverage"] == 1.0
    assert diagnostics["structural_endpoint_reachability_accuracy"][
        "component"
    ] == 1.0
    dataset = CoarsenDataset(arrays, artifacts)
    batch = {
        key: torch.stack([dataset[0][key], dataset[1][key]])
        for key in dataset[0]
    }
    model = TopoCoarsenModel("topology_oracle", 32, 16)
    logits = model(
        batch["image"],
        batch["assignment_topology"],
        batch["adjacency_topology"],
        batch["reachability_topology"],
    )
    assert logits.shape == (2,)
    print("TOPOCOARSEN_SELF_TEST_PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    unknown = sorted(set(args.models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    required = {"component_oracle", "topology_oracle", "topology_coords_only"}
    if not required.issubset(args.models):
        raise ValueError(f"Models must include {sorted(required)}.")
    grid_assignment(args.token_count)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
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
    val_arrays = load_or_generate(
        cache_dir, "val", args.val_size, args.seed + 1_000_000
    )
    shortcut = coordinate_shortcut_accuracy(train_arrays, val_arrays)
    print(f"COARSEN_SHORTCUT distance_accuracy={shortcut:.4f}", flush=True)
    train_artifacts, train_diagnostics = build_artifacts(
        train_arrays, args.token_count
    )
    val_artifacts, val_diagnostics = build_artifacts(
        val_arrays, args.token_count
    )
    diagnostics = {
        "train": train_diagnostics,
        "validation": val_diagnostics,
    }
    (output / "coarsening_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False), flush=True)

    train_dataset = CoarsenDataset(train_arrays, train_artifacts)
    val_dataset = CoarsenDataset(val_arrays, val_artifacts)
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for model_name in args.models:
        summary, records = train_one(
            model_name, args, train_dataset, val_dataset, output, device
        )
        summaries.append(summary)
        history.extend(records)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "history.csv", history)
    result = render_result(
        output, summaries, shortcut, diagnostics
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

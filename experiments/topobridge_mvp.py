from __future__ import annotations

import argparse
import csv
import json
import math
import random
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


MODEL_NAMES = [
    "mlp",
    "spatial",
    "conv",
    "topobridge_noaux",
    "topobridge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TopoBridge minimal validation.")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--tokens-per-side", type=int, default=4)
    parser.add_argument("--beta0-loss-weight", type=float, default=0.3)
    parser.add_argument("--beta1-loss-weight", type=float, default=0.3)
    parser.add_argument("--edge-loss-weight", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260810])
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results_smoke"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def smooth_field(field: np.ndarray, rounds: int) -> np.ndarray:
    result = field.astype(np.float64, copy=True)
    for _ in range(rounds):
        padded = np.pad(result, 1, mode="reflect")
        result = (
            4.0 * padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        ) / 12.0
    return result


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int16)
    sizes: list[int] = []
    current = 0
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col] != 0:
                continue
            current += 1
            queue = deque([(row, col)])
            labels[row, col] = current
            size = 0
            while queue:
                r, c = queue.popleft()
                size += 1
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and labels[nr, nc] == 0
                    ):
                        labels[nr, nc] = current
                        queue.append((nr, nc))
            sizes.append(size)
    return labels, sizes


def remove_small_components(mask: np.ndarray, minimum_size: int = 6) -> np.ndarray:
    labels, sizes = connected_components(mask)
    keep = np.zeros(len(sizes) + 1, dtype=bool)
    for index, size in enumerate(sizes, start=1):
        keep[index] = size >= minimum_size
    return keep[labels]


def count_holes(mask: np.ndarray, minimum_size: int = 3) -> int:
    background = ~mask
    labels, sizes = connected_components(background)
    if not sizes:
        return 0
    border_labels = set(labels[0, :]) | set(labels[-1, :])
    border_labels |= set(labels[:, 0]) | set(labels[:, -1])
    return sum(
        size >= minimum_size and index not in border_labels
        for index, size in enumerate(sizes, start=1)
    )


def choose_endpoint_pair(
    labels: np.ndarray,
    desired_connected: int,
    target_distance: float,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    coords = np.argwhere(labels > 0)
    if coords.shape[0] < 2:
        return None
    best: tuple[tuple[int, int], tuple[int, int]] | None = None
    best_gap = float("inf")
    normalizer = math.sqrt(2.0) * (labels.shape[0] - 1)
    for _ in range(1200):
        first_index, second_index = rng.integers(0, coords.shape[0], size=2)
        if first_index == second_index:
            continue
        first = coords[first_index]
        second = coords[second_index]
        connected = int(labels[tuple(first)] == labels[tuple(second)])
        if connected != desired_connected:
            continue
        distance = float(np.linalg.norm(first - second) / normalizer)
        gap = abs(distance - target_distance)
        if gap < best_gap:
            best_gap = gap
            best = ((int(first[0]), int(first[1])), (int(second[0]), int(second[1])))
    return best if best_gap <= 0.055 else None


def patch_component_ids(labels: np.ndarray) -> np.ndarray:
    if labels.shape != (32, 32):
        raise ValueError("Expected a 32x32 component map.")
    patches = np.zeros((16, 16), dtype=np.int16)
    for row in range(16):
        for col in range(16):
            block = labels[2 * row : 2 * row + 2, 2 * col : 2 * col + 2]
            positive = block[block > 0]
            if positive.size:
                values, counts = np.unique(positive, return_counts=True)
                patches[row, col] = values[np.argmax(counts)]
    return patches


def render_sample(
    mask: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    high = np.repeat(np.repeat(mask.astype(np.float32), 2, axis=0), 2, axis=1)
    image = np.zeros((3, 64, 64), dtype=np.float32)
    contrast = rng.uniform(0.75, 1.0)
    background = rng.uniform(0.0, 0.08)
    image[0] = background + contrast * high
    for channel, point in ((1, first), (2, second)):
        center_row = 2 * point[0] + 1
        center_col = 2 * point[1] + 1
        rr, cc = np.ogrid[:64, :64]
        marker = (rr - center_row) ** 2 + (cc - center_col) ** 2 <= 6
        image[channel, marker] = 1.0
    noise = rng.normal(0.0, 0.015, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def generate_one(index: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + index * 7919)
    desired_connected = index % 2
    distance_targets = [0.18, 0.30, 0.42]
    target_distance = distance_targets[(index // 2) % len(distance_targets)]
    for _ in range(500):
        field = smooth_field(rng.normal(size=(32, 32)), int(rng.integers(2, 5)))
        quantile = rng.uniform(0.57, 0.68)
        mask = field > np.quantile(field, quantile)
        mask = remove_small_components(mask, minimum_size=6)
        labels, sizes = connected_components(mask)
        component_count = len(sizes)
        foreground_fraction = float(np.mean(mask))
        if not (2 <= component_count <= 4 and 0.16 <= foreground_fraction <= 0.48):
            continue
        pair = choose_endpoint_pair(
            labels, desired_connected, target_distance, rng
        )
        if pair is None:
            continue
        first, second = pair
        holes = min(count_holes(mask), 3)
        patch_ids = patch_component_ids(labels)
        return {
            "image": render_sample(mask, first, second, rng),
            "connected": desired_connected,
            "beta0": component_count - 1,
            "beta1": holes,
            "patch_ids": patch_ids,
            "endpoint_features": np.array(
                [
                    first[0] / 31.0,
                    first[1] / 31.0,
                    second[0] / 31.0,
                    second[1] / 31.0,
                    np.linalg.norm(np.array(first) - np.array(second))
                    / (math.sqrt(2.0) * 31.0),
                    foreground_fraction,
                ],
                dtype=np.float32,
            ),
        }
    raise RuntimeError(f"Could not generate sample index={index}, seed={seed}.")


def generate_split(count: int, seed: int) -> dict[str, np.ndarray]:
    images = np.empty((count, 3, 64, 64), dtype=np.uint8)
    connected = np.empty(count, dtype=np.int64)
    beta0 = np.empty(count, dtype=np.int64)
    beta1 = np.empty(count, dtype=np.int64)
    patch_ids = np.empty((count, 16, 16), dtype=np.int16)
    endpoint_features = np.empty((count, 6), dtype=np.float32)
    start = time.time()
    for index in range(count):
        sample = generate_one(index, seed)
        images[index] = np.round(sample["image"] * 255.0).astype(np.uint8)
        connected[index] = sample["connected"]
        beta0[index] = sample["beta0"]
        beta1[index] = sample["beta1"]
        patch_ids[index] = sample["patch_ids"]
        endpoint_features[index] = sample["endpoint_features"]
        if (index + 1) % 250 == 0 or index + 1 == count:
            print(
                f"DATA seed={seed} {index + 1}/{count} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    return {
        "images": images,
        "connected": connected,
        "beta0": beta0,
        "beta1": beta1,
        "patch_ids": patch_ids,
        "endpoint_features": endpoint_features,
    }


def load_or_generate(
    cache_dir: Path, split: str, count: int, seed: int
) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}_n{count}_seed{seed}.npz"
    if path.exists():
        with np.load(path) as archive:
            return {key: archive[key] for key in archive.files}
    arrays = generate_split(count, seed)
    np.savez_compressed(path, **arrays)
    return arrays


def coordinate_shortcut_accuracy(
    train: dict[str, np.ndarray], val: dict[str, np.ndarray]
) -> float:
    train_distance = train["endpoint_features"][:, 4]
    train_label = train["connected"]
    val_distance = val["endpoint_features"][:, 4]
    val_label = val["connected"]
    best_threshold = 0.5
    best_direction = 1
    best_accuracy = -1.0
    for threshold in np.linspace(0.05, 0.75, 141):
        for direction in (-1, 1):
            prediction = (
                train_distance <= threshold
                if direction == 1
                else train_distance > threshold
            )
            accuracy = float(np.mean(prediction == train_label))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = float(threshold)
                best_direction = direction
    val_prediction = (
        val_distance <= best_threshold
        if best_direction == 1
        else val_distance > best_threshold
    )
    return float(np.mean(val_prediction == val_label))


class TopoDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return int(self.arrays["images"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(
                self.arrays["images"][index].astype(np.float32) / 255.0
            ),
            "connected": torch.tensor(
                self.arrays["connected"][index], dtype=torch.float32
            ),
            "beta0": torch.tensor(self.arrays["beta0"][index], dtype=torch.long),
            "beta1": torch.tensor(self.arrays["beta1"][index], dtype=torch.long),
            "patch_ids": torch.from_numpy(
                self.arrays["patch_ids"][index].astype(np.int64)
            ),
        }


class PatchStem(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class MLPConnector(nn.Module):
    def __init__(self, dim: int, side: int) -> None:
        super().__init__()
        self.side = side
        self.project = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 1),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, 1),
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = F.adaptive_avg_pool2d(self.project(features), (self.side, self.side))
        return output, {}


class SpatialConnector(nn.Module):
    def __init__(self, dim: int, side: int) -> None:
        super().__init__()
        self.side = side
        self.project = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled = F.adaptive_avg_pool2d(features, (self.side, self.side))
        return self.project(pooled), {}


class ConvConnector(nn.Module):
    def __init__(self, dim: int, side: int) -> None:
        super().__init__()
        self.side = side
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        local = features + self.local(features)
        return F.adaptive_avg_pool2d(local, (self.side, self.side)), {}


class TopoBridgeConnector(nn.Module):
    def __init__(self, dim: int, side: int) -> None:
        super().__init__()
        self.side = side
        hidden = max(16, dim // 2)
        self.edge_net = nn.Sequential(
            nn.Conv2d(3 * dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.message = nn.Conv2d(dim, dim, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 1),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, 1),
        )

    def edge_logits(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return self.edge_net(torch.cat([left, right, torch.abs(left - right)], dim=1))

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        horizontal_logits = self.edge_logits(
            features[:, :, :, :-1], features[:, :, :, 1:]
        )
        vertical_logits = self.edge_logits(
            features[:, :, :-1, :], features[:, :, 1:, :]
        )
        horizontal = torch.sigmoid(horizontal_logits)
        vertical = torch.sigmoid(vertical_logits)
        aggregate = torch.zeros_like(features)
        normalizer = torch.zeros_like(features[:, :1])
        aggregate[:, :, :, :-1] += horizontal * features[:, :, :, 1:]
        aggregate[:, :, :, 1:] += horizontal * features[:, :, :, :-1]
        normalizer[:, :, :, :-1] += horizontal
        normalizer[:, :, :, 1:] += horizontal
        aggregate[:, :, :-1, :] += vertical * features[:, :, 1:, :]
        aggregate[:, :, 1:, :] += vertical * features[:, :, :-1, :]
        normalizer[:, :, :-1, :] += vertical
        normalizer[:, :, 1:, :] += vertical
        mixed = features + self.message(aggregate / normalizer.clamp_min(1e-4))
        mixed = mixed + self.refine(mixed)
        output = F.adaptive_avg_pool2d(mixed, (self.side, self.side))
        return output, {
            "horizontal_logits": horizontal_logits.squeeze(1),
            "vertical_logits": vertical_logits.squeeze(1),
        }


class MultiTaskHead(nn.Module):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, token_count + 1, dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)
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
        self.connected = nn.Linear(dim, 1)
        self.beta0 = nn.Linear(dim, 4)
        self.beta1 = nn.Linear(dim, 4)

    def forward(self, feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = feature_map.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(sequence + self.position)
        pooled = self.norm(encoded[:, 0])
        return {
            "connected": self.connected(pooled).squeeze(-1),
            "beta0": self.beta0(pooled),
            "beta1": self.beta1(pooled),
        }


class TopoModel(nn.Module):
    def __init__(self, name: str, dim: int, side: int) -> None:
        super().__init__()
        self.name = name
        self.stem = PatchStem(dim)
        if name == "mlp":
            self.connector = MLPConnector(dim, side)
        elif name == "spatial":
            self.connector = SpatialConnector(dim, side)
        elif name == "conv":
            self.connector = ConvConnector(dim, side)
        elif name in {"topobridge_noaux", "topobridge"}:
            self.connector = TopoBridgeConnector(dim, side)
        else:
            raise ValueError(f"Unknown model: {name}")
        self.head = MultiTaskHead(dim, side * side)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.stem(image)
        compressed, auxiliary = self.connector(features)
        output = self.head(compressed)
        output.update(auxiliary)
        return output


def balanced_edge_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    positive = valid & target
    negative = valid & ~target
    losses = []
    if torch.any(positive):
        losses.append(F.binary_cross_entropy_with_logits(logits[positive], torch.ones_like(logits[positive])))
    if torch.any(negative):
        losses.append(F.binary_cross_entropy_with_logits(logits[negative], torch.zeros_like(logits[negative])))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def edge_targets(
    patch_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left = patch_ids[:, :, :-1]
    right = patch_ids[:, :, 1:]
    top = patch_ids[:, :-1, :]
    bottom = patch_ids[:, 1:, :]
    horizontal_target = (left > 0) & (left == right)
    vertical_target = (top > 0) & (top == bottom)
    horizontal_valid = (left > 0) | (right > 0)
    vertical_valid = (top > 0) | (bottom > 0)
    return (
        horizontal_target,
        horizontal_valid,
        vertical_target,
        vertical_valid,
    )


def compute_loss(
    model_name: str,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    beta0_weight: float,
    beta1_weight: float,
    edge_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    connected_loss = F.binary_cross_entropy_with_logits(
        output["connected"], batch["connected"]
    )
    beta0_loss = F.cross_entropy(output["beta0"], batch["beta0"])
    beta1_loss = F.cross_entropy(output["beta1"], batch["beta1"])
    total = (
        connected_loss
        + beta0_weight * beta0_loss
        + beta1_weight * beta1_loss
    )
    edge_loss = output["connected"].sum() * 0.0
    if model_name == "topobridge":
        ht, hv, vt, vv = edge_targets(batch["patch_ids"])
        edge_loss = 0.5 * (
            balanced_edge_loss(output["horizontal_logits"], ht, hv)
            + balanced_edge_loss(output["vertical_logits"], vt, vv)
        )
        total = total + edge_weight * edge_loss
    return total, {
        "connected_loss": float(connected_loss.detach()),
        "beta0_loss": float(beta0_loss.detach()),
        "beta1_loss": float(beta1_loss.detach()),
        "edge_loss": float(edge_loss.detach()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    connected_true = []
    connected_pred = []
    beta0_true = []
    beta0_pred = []
    beta1_true = []
    beta1_pred = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(batch["image"])
        connected_true.append(batch["connected"].cpu())
        connected_pred.append((torch.sigmoid(output["connected"]) >= 0.5).cpu())
        beta0_true.append(batch["beta0"].cpu())
        beta0_pred.append(torch.argmax(output["beta0"], dim=1).cpu())
        beta1_true.append(batch["beta1"].cpu())
        beta1_pred.append(torch.argmax(output["beta1"], dim=1).cpu())
    y = torch.cat(connected_true).bool()
    pred = torch.cat(connected_pred).bool()
    positive_accuracy = float((pred[y] == y[y]).float().mean()) if torch.any(y) else 0.0
    negative_accuracy = float((pred[~y] == y[~y]).float().mean()) if torch.any(~y) else 0.0
    beta0_y = torch.cat(beta0_true)
    beta0_p = torch.cat(beta0_pred)
    beta1_y = torch.cat(beta1_true)
    beta1_p = torch.cat(beta1_pred)
    return {
        "connected_accuracy": float((pred == y).float().mean()),
        "connected_balanced_accuracy": 0.5 * (positive_accuracy + negative_accuracy),
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "beta0_accuracy": float((beta0_p == beta0_y).float().mean()),
        "beta1_accuracy": float((beta1_p == beta1_y).float().mean()),
    }


def train_one(
    model_name: str,
    seed: int,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(seed)
    train_loader = DataLoader(
        TopoDataset(train_arrays),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TopoDataset(val_arrays),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = TopoModel(model_name, args.dim, args.tokens_per_side).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_metric = -1.0
    best_path = output / f"{model_name}_seed{seed}.pt"
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                prediction = model(batch["image"])
                loss, _ = compute_loss(
                    model_name,
                    prediction,
                    batch,
                    args.beta0_loss_weight,
                    args.beta1_loss_weight,
                    args.edge_loss_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            batches += 1
        metrics = evaluate(model, val_loader, device)
        record = {
            "seed": seed,
            "model": model_name,
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            **metrics,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            "TRAIN "
            f"seed={seed} model={model_name} epoch={epoch}/{args.epochs} "
            f"loss={record['train_loss']:.4f} "
            f"conn_bal={metrics['connected_balanced_accuracy']:.4f} "
            f"b0={metrics['beta0_accuracy']:.4f} "
            f"b1={metrics['beta1_accuracy']:.4f}",
            flush=True,
        )
        if metrics["connected_balanced_accuracy"] > best_metric:
            best_metric = metrics["connected_balanced_accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "seed": seed,
                    "model_name": model_name,
                    "metrics": metrics,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, val_loader, device)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    summary = {
        "seed": seed,
        "model": model_name,
        "parameters": parameter_count,
        "best_epoch_connected_balanced_accuracy": best_metric,
        **final_metrics,
        "training_seconds": time.time() - started,
        "peak_cuda_memory_mb": peak_memory_mb,
    }
    return summary, history


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_results(
    output: Path, summaries: list[dict[str, Any]], shortcut_accuracy: float
) -> dict[str, Any]:
    model_scores: dict[str, list[float]] = {}
    for row in summaries:
        model_scores.setdefault(row["model"], []).append(
            row["connected_balanced_accuracy"]
        )
    aggregate = {
        model: {
            "mean_connected_balanced_accuracy": float(np.mean(values)),
            "std_connected_balanced_accuracy": float(np.std(values)),
            "seeds": len(values),
        }
        for model, values in model_scores.items()
    }
    baseline_models = [
        model
        for model in ("mlp", "spatial", "conv", "topobridge_noaux")
        if model in aggregate
    ]
    strongest_baseline = max(
        baseline_models,
        key=lambda model: aggregate[model]["mean_connected_balanced_accuracy"],
    )
    topobridge_score = aggregate.get(
        "topobridge", {"mean_connected_balanced_accuracy": float("nan")}
    )["mean_connected_balanced_accuracy"]
    baseline_score = aggregate[strongest_baseline][
        "mean_connected_balanced_accuracy"
    ]
    gain = topobridge_score - baseline_score
    result = {
        "experiment_id": "topobridge_mvp_smoke",
        "status": "completed",
        "coordinate_shortcut_accuracy": shortcut_accuracy,
        "aggregate": aggregate,
        "strongest_baseline": strongest_baseline,
        "topobridge_gain_over_strongest_baseline": gain,
        "smoke_gate": {
            "dataset_shortcut_pass": shortcut_accuracy <= 0.60,
            "topobridge_gain_at_least_2pp": bool(gain >= 0.02),
        },
        "verdict": (
            "PROCEED_TO_FORMAL"
            if shortcut_accuracy <= 0.60 and gain >= 0.02
            else "DO_NOT_SCALE_YET"
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    models = list(aggregate)
    values = [
        aggregate[model]["mean_connected_balanced_accuracy"] for model in models
    ]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(models, values, color="#4C78A8")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("validation connected balanced accuracy")
    ax.set_title("TopoBridge MVP connector comparison")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(output / "connector_comparison.png", dpi=180)
    plt.close(fig)
    return result


def run_self_test() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True
    labels, sizes = connected_components(mask)
    assert len(sizes) == 2
    assert labels[1, 1] != labels[5, 5]
    ring = np.zeros((8, 8), dtype=bool)
    ring[1:7, 1] = True
    ring[1:7, 6] = True
    ring[1, 1:7] = True
    ring[6, 1:7] = True
    assert count_holes(ring) == 1
    sample = generate_one(0, 12345)
    assert sample["image"].shape == (3, 64, 64)
    assert sample["patch_ids"].shape == (16, 16)
    model = TopoModel("topobridge", 32, 4)
    batch = torch.from_numpy(sample["image"][None])
    output = model(batch)
    assert output["connected"].shape == (1,)
    assert output["horizontal_logits"].shape == (1, 16, 15)
    assert output["vertical_logits"].shape == (1, 15, 16)
    print("SELF_TEST_PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    unknown = sorted(set(args.models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if args.tokens_per_side < 2 or 16 % args.tokens_per_side != 0:
        raise ValueError("tokens-per-side must divide 16 and be at least 2.")
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
                "cuda": torch.version.cuda,
                "args": vars(args),
            },
            default=str,
        ),
        flush=True,
    )

    summaries = []
    history = []
    shortcut_scores = []
    for seed in args.seeds:
        train_arrays = load_or_generate(
            cache_dir, "train", args.train_size, seed
        )
        val_arrays = load_or_generate(
            cache_dir, "val", args.val_size, seed + 1_000_000
        )
        shortcut = coordinate_shortcut_accuracy(train_arrays, val_arrays)
        shortcut_scores.append(shortcut)
        print(f"SHORTCUT seed={seed} distance_accuracy={shortcut:.4f}", flush=True)
        for model_name in args.models:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            summary, records = train_one(
                model_name,
                seed,
                train_arrays,
                val_arrays,
                args,
                output,
                device,
            )
            summaries.append(summary)
            history.extend(records)

    write_csv(output / "summary.csv", summaries)
    write_csv(output / "history.csv", history)
    result = render_results(output, summaries, float(np.mean(shortcut_scores)))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

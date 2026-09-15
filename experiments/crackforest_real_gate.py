from __future__ import annotations

import argparse
import csv
import hashlib
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
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import label
from torch.utils.data import DataLoader, Dataset

from edge_topocoarsen_connector import (
    balanced_metrics,
    bootstrap_balanced_accuracy,
    canonicalize_images,
)
from topobridge_mvp import PatchStem, set_seed
from topocoarsen_oracle import (
    CoarseGraphHead,
    TopoCoarsenModel,
    fine_coordinates,
    oracle_assignment,
)


MODEL_NAMES = [
    "grid",
    "conv",
    "attention",
    "edge_topocoarsen",
    "topology_oracle",
]
EDGE_NAMES = (
    "horizontal",
    "vertical",
    "diagonal_right",
    "diagonal_left",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-data EdgeTopoCoarsen gate on CrackForest."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("real_data/CrackForest-dataset"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("real_cache"))
    parser.add_argument(
        "--output", type=Path, default=Path("results_crackforest_real")
    )
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--dev-size", type=int, default=320)
    parser.add_argument("--test-size", type=int, default=320)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--edge-epochs", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument(
        "--node-thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    parser.add_argument(
        "--edge-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    return parser.parse_args()


def dataset_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "groundTruth").glob("*.mat")):
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:16]


def source_split(
    root: Path, split_seed: int
) -> dict[str, list[str]]:
    stems = sorted(
        path.stem for path in (root / "groundTruth").glob("*.mat")
    )
    if len(stems) != 118:
        raise ValueError(f"Expected 118 annotated images, found {len(stems)}.")
    rng = np.random.default_rng(split_seed)
    shuffled = np.asarray(stems)
    rng.shuffle(shuffled)
    return {
        "train": sorted(shuffled[:82].tolist()),
        "dev": sorted(shuffled[82:100].tolist()),
        "test": sorted(shuffled[100:].tolist()),
    }


def load_sources(
    root: Path, stems: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for stem in stems:
        rgb = np.asarray(
            Image.open(root / "image" / f"{stem}.jpg").convert("RGB")
        )
        gray = np.round(
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        ).astype(np.uint8)
        ground_truth = loadmat(
            root / "groundTruth" / f"{stem}.mat",
            squeeze_me=True,
            struct_as_record=False,
        )["groundTruth"]
        mask = np.asarray(ground_truth.Boundaries).astype(bool)
        if gray.shape != (320, 480) or mask.shape != gray.shape:
            raise ValueError(f"Unexpected source shape for {stem}.")
        sources[stem] = (gray, mask)
    return sources


def choose_pair(
    component_map: np.ndarray,
    sizes: np.ndarray,
    desired_connected: int,
    target_distance: float,
    minimum_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    eligible = np.flatnonzero(sizes >= minimum_size) + 1
    if desired_connected == 0 and eligible.size < 2:
        return None
    if desired_connected == 1 and eligible.size < 1:
        return None
    coordinate_bank = {
        int(component): np.argwhere(component_map == component)
        for component in eligible
    }
    best_pair = None
    best_gap = float("inf")
    normalizer = math.sqrt(2.0) * (component_map.shape[0] - 1)
    for _ in range(500):
        if desired_connected:
            component = int(rng.choice(eligible))
            coordinates = coordinate_bank[component]
            if coordinates.shape[0] < 2:
                continue
            indices = rng.integers(0, coordinates.shape[0], size=2)
            if indices[0] == indices[1]:
                continue
            first, second = coordinates[indices]
        else:
            components = rng.choice(eligible, size=2, replace=False)
            first_bank = coordinate_bank[int(components[0])]
            second_bank = coordinate_bank[int(components[1])]
            first = first_bank[int(rng.integers(0, first_bank.shape[0]))]
            second = second_bank[int(rng.integers(0, second_bank.shape[0]))]
        distance = float(np.linalg.norm(first - second) / normalizer)
        gap = abs(distance - target_distance)
        if gap < best_gap:
            best_gap = gap
            best_pair = (first.copy(), second.copy())
    return best_pair if best_gap <= 0.075 else None


def patch_component_ids(component_map: np.ndarray) -> np.ndarray:
    if (
        component_map.ndim != 2
        or component_map.shape[0] != component_map.shape[1]
        or component_map.shape[0] % 16 != 0
    ):
        raise ValueError("Component crop must be square and divisible by 16.")
    block_size = component_map.shape[0] // 16
    patches = np.zeros((16, 16), dtype=np.int16)
    for row in range(16):
        for col in range(16):
            block = component_map[
                block_size * row : block_size * (row + 1),
                block_size * col : block_size * (col + 1),
            ]
            positive = block[block > 0]
            if positive.size:
                values, counts = np.unique(positive, return_counts=True)
                patches[row, col] = int(values[np.argmax(counts)])
    return patches


def draw_marker(channel: np.ndarray, row: int, col: int) -> None:
    rows, cols = np.ogrid[: channel.shape[0], : channel.shape[1]]
    channel[(rows - row) ** 2 + (cols - col) ** 2 <= 9] = 1.0


def make_sample(
    sources: dict[str, tuple[np.ndarray, np.ndarray]],
    source_ids: list[str],
    index: int,
    seed: int,
    crop_size: int,
    output_size: int,
    minimum_size: int,
) -> dict[str, Any]:
    if crop_size % 16 != 0 or output_size != 64:
        raise ValueError(
            "crop-size must be divisible by 16 and output-size must be 64."
        )
    rng = np.random.default_rng(seed + index * 104729)
    desired_connected = index % 2
    target_distance = [0.20, 0.34, 0.48][(index // 2) % 3]
    structure = np.ones((3, 3), dtype=np.uint8)
    for _ in range(1600):
        source_id = source_ids[int(rng.integers(0, len(source_ids)))]
        gray, mask = sources[source_id]
        foreground = np.argwhere(mask)
        if foreground.size == 0:
            continue
        anchor = foreground[int(rng.integers(0, foreground.shape[0]))]
        jitter = rng.integers(-crop_size // 4, crop_size // 4 + 1, size=2)
        top = int(
            np.clip(
                anchor[0] - crop_size // 2 + jitter[0],
                0,
                gray.shape[0] - crop_size,
            )
        )
        left = int(
            np.clip(
                anchor[1] - crop_size // 2 + jitter[1],
                0,
                gray.shape[1] - crop_size,
            )
        )
        mask_crop = mask[top : top + crop_size, left : left + crop_size]
        component_map, count = label(mask_crop, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(component_map.reshape(-1))[1:]
        pair = choose_pair(
            component_map,
            sizes,
            desired_connected,
            target_distance,
            minimum_size,
            rng,
        )
        if pair is None:
            continue
        first, second = pair
        patch_block = crop_size // 16
        first_patch = (
            int(first[0] // patch_block),
            int(first[1] // patch_block),
        )
        second_patch = (
            int(second[0] // patch_block),
            int(second[1] // patch_block),
        )
        if first_patch == second_patch:
            continue
        patches = patch_component_ids(component_map)
        first_component = int(component_map[tuple(first)])
        second_component = int(component_map[tuple(second)])
        patches[first_patch] = first_component
        patches[second_patch] = second_component

        gray_crop = gray[top : top + crop_size, left : left + crop_size]
        resized = np.asarray(
            Image.fromarray(gray_crop).resize(
                (output_size, output_size), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        image = np.zeros((3, output_size, output_size), dtype=np.float32)
        image[0] = 1.0 - resized / 255.0
        first_out = np.clip(
            np.floor(
                (first.astype(np.float32) + 0.5)
                * output_size
                / crop_size
            ),
            0,
            output_size - 1,
        ).astype(int)
        second_out = np.clip(
            np.floor(
                (second.astype(np.float32) + 0.5)
                * output_size
                / crop_size
            ),
            0,
            output_size - 1,
        ).astype(int)
        draw_marker(image[1], int(first_out[0]), int(first_out[1]))
        draw_marker(image[2], int(second_out[0]), int(second_out[1]))
        distance = float(
            np.linalg.norm(first.astype(np.float32) - second)
            / (math.sqrt(2.0) * (crop_size - 1))
        )
        return {
            "image": np.round(np.clip(image, 0.0, 1.0) * 255).astype(
                np.uint8
            ),
            "connected": desired_connected,
            "patch_ids": patches,
            "endpoint_patches": np.asarray(
                [
                    first_patch[0] * 16 + first_patch[1],
                    second_patch[0] * 16 + second_patch[1],
                ],
                dtype=np.int16,
            ),
            "endpoint_distance": distance,
            "source_id": source_id,
            "crop_box": np.asarray(
                [top, left, top + crop_size, left + crop_size],
                dtype=np.int16,
            ),
        }
    raise RuntimeError(
        f"Could not generate sample index={index}, label={desired_connected}."
    )


def generate_arrays(
    sources: dict[str, tuple[np.ndarray, np.ndarray]],
    source_ids: list[str],
    count: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    arrays = {
        "images": np.empty(
            (count, 3, args.output_size, args.output_size), dtype=np.uint8
        ),
        "connected": np.empty(count, dtype=np.int64),
        "patch_ids": np.empty((count, 16, 16), dtype=np.int16),
        "endpoint_patches": np.empty((count, 2), dtype=np.int16),
        "endpoint_distance": np.empty(count, dtype=np.float32),
        "source_id": np.empty(count, dtype="<U64"),
        "crop_box": np.empty((count, 4), dtype=np.int16),
    }
    started = time.time()
    for index in range(count):
        sample = make_sample(
            sources,
            source_ids,
            index,
            seed,
            args.crop_size,
            args.output_size,
            args.min_component_pixels,
        )
        arrays["images"][index] = sample["image"]
        for key in (
            "connected",
            "patch_ids",
            "endpoint_patches",
            "endpoint_distance",
            "source_id",
            "crop_box",
        ):
            arrays[key][index] = sample[key]
        if (index + 1) % 200 == 0 or index + 1 == count:
            print(
                f"REAL_DATA {index + 1}/{count} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    arrays["images"] = canonicalize_images(arrays["images"])
    return arrays


def load_or_generate_split(
    cache_dir: Path,
    split: str,
    sources: dict[str, tuple[np.ndarray, np.ndarray]],
    source_ids: list[str],
    count: int,
    seed: int,
    args: argparse.Namespace,
    fingerprint: str,
) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (
        f"crackforest_{split}_n{count}_seed{seed}_"
        f"crop{args.crop_size}_{fingerprint}.npz"
    )
    if path.exists():
        with np.load(path) as archive:
            return {key: archive[key] for key in archive.files}
    arrays = generate_arrays(sources, source_ids, count, seed, args)
    np.savez_compressed(path, **arrays)
    return arrays


def coarse_graph8(
    assignment: np.ndarray,
    patch_ids: np.ndarray,
    token_count: int,
    topology_aware: bool,
) -> tuple[np.ndarray, np.ndarray]:
    adjacency = np.eye(token_count, dtype=np.uint8)
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(16):
        for col in range(16):
            for delta_row, delta_col in directions:
                new_row = row + delta_row
                new_col = col + delta_col
                if not (0 <= new_row < 16 and 0 <= new_col < 16):
                    continue
                if topology_aware:
                    first_id = int(patch_ids[row, col])
                    second_id = int(patch_ids[new_row, new_col])
                    valid = (first_id == 0 and second_id == 0) or (
                        first_id > 0 and first_id == second_id
                    )
                    if not valid:
                        continue
                first = int(assignment[row, col])
                second = int(assignment[new_row, new_col])
                adjacency[first, second] = 1
                adjacency[second, first] = 1
    reachability = adjacency.astype(bool)
    for pivot in range(token_count):
        reachability |= (
            reachability[:, pivot, None] & reachability[pivot, None, :]
        )
    return adjacency, reachability.astype(np.uint8)


def structural_prediction(
    assignment: np.ndarray,
    reachability: np.ndarray,
    endpoints: np.ndarray,
    patch_ids: np.ndarray,
) -> int:
    first = int(endpoints[0])
    second = int(endpoints[1])
    first_id = int(patch_ids[first // 16, first % 16])
    second_id = int(patch_ids[second // 16, second % 16])
    if first_id <= 0 or second_id <= 0:
        return 0
    first_cluster = int(assignment[first // 16, first % 16])
    second_cluster = int(assignment[second // 16, second % 16])
    return int(reachability[first_cluster, second_cluster])


def fixed_artifacts(
    arrays: dict[str, np.ndarray],
    token_count: int,
    name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample_count = arrays["images"].shape[0]
    assignments = np.empty((sample_count, 16, 16), dtype=np.int16)
    adjacency = np.empty(
        (sample_count, token_count, token_count), dtype=np.uint8
    )
    reachability = np.empty_like(adjacency)
    predictions = np.empty(sample_count, dtype=np.uint8)
    if name == "grid":
        base = np.arange(token_count, dtype=np.int16).reshape(4, 4)
        base = base.repeat(4, 0).repeat(4, 1)
    elif name != "topology":
        raise ValueError(name)
    for index in range(sample_count):
        patch_ids = arrays["patch_ids"][index]
        assignment = (
            base
            if name == "grid"
            else oracle_assignment(patch_ids, token_count, False)
        )
        graph, closure = coarse_graph8(
            assignment,
            patch_ids,
            token_count,
            topology_aware=name == "topology",
        )
        assignments[index] = assignment
        adjacency[index] = graph
        reachability[index] = closure
        predictions[index] = structural_prediction(
            assignment,
            closure,
            arrays["endpoint_patches"][index],
            patch_ids,
        )
    return {
        "assignment": assignments,
        "adjacency": adjacency,
        "reachability": reachability,
    }, {
        "structural_metrics": balanced_metrics(
            arrays["connected"], predictions
        ),
        "all_tokens_nonempty": bool(
            all(
                np.unique(assignments[index]).size == token_count
                for index in range(sample_count)
            )
        ),
    }


class RealDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        artifacts: dict[str, np.ndarray] | None = None,
        augment: bool = False,
    ) -> None:
        self.arrays = arrays
        self.artifacts = artifacts
        self.augment = augment

    def __len__(self) -> int:
        return int(self.arrays["images"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(
            self.arrays["images"][index].astype(np.float32) / 255.0
        )
        if self.augment:
            contrast = 0.82 + 0.28 * torch.rand(())
            brightness = -0.04 + 0.08 * torch.rand(())
            noise = 0.025 * torch.rand(()) * torch.randn_like(image)
            image[0] = torch.clamp(
                brightness + contrast * image[0] + noise[0], 0.0, 1.0
            )
        item = {
            "image": image,
            "connected": torch.tensor(
                self.arrays["connected"][index], dtype=torch.float32
            ),
            "patch_ids": torch.from_numpy(
                self.arrays["patch_ids"][index].astype(np.int64)
            ),
            "endpoint_patches": torch.from_numpy(
                self.arrays["endpoint_patches"][index].astype(np.int64)
            ),
        }
        if self.artifacts is not None:
            for key, value in self.artifacts.items():
                item[key] = torch.from_numpy(
                    value[index].astype(np.int64)
                )
        return item


class RealEdgePredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.stem = PatchStem(dim)
        hidden = max(16, dim // 2)
        self.node_head = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(3 * dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def edge_logits(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        return self.edge_head(
            torch.cat((first, second, torch.abs(first - second)), dim=1)
        ).squeeze(1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.stem(image)
        return {
            "node": self.node_head(features).squeeze(1),
            "horizontal": self.edge_logits(
                features[:, :, :, :-1], features[:, :, :, 1:]
            ),
            "vertical": self.edge_logits(
                features[:, :, :-1, :], features[:, :, 1:, :]
            ),
            "diagonal_right": self.edge_logits(
                features[:, :, :-1, :-1], features[:, :, 1:, 1:]
            ),
            "diagonal_left": self.edge_logits(
                features[:, :, :-1, 1:], features[:, :, 1:, :-1]
            ),
        }


def balanced_binary_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    target = target.bool()
    parts = []
    if torch.any(target):
        parts.append(
            F.binary_cross_entropy_with_logits(
                logits[target], torch.ones_like(logits[target])
            )
        )
    if torch.any(~target):
        parts.append(
            F.binary_cross_entropy_with_logits(
                logits[~target], torch.zeros_like(logits[~target])
            )
        )
    return torch.stack(parts).mean()


def edge_targets(
    patch_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    def same(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return (first > 0) & (first == second)

    return {
        "horizontal": same(patch_ids[:, :, :-1], patch_ids[:, :, 1:]),
        "vertical": same(patch_ids[:, :-1, :], patch_ids[:, 1:, :]),
        "diagonal_right": same(
            patch_ids[:, :-1, :-1], patch_ids[:, 1:, 1:]
        ),
        "diagonal_left": same(
            patch_ids[:, :-1, 1:], patch_ids[:, 1:, :-1]
        ),
    }


def edge_loss(
    output: dict[str, torch.Tensor], patch_ids: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    node_target = patch_ids > 0
    node = balanced_binary_loss(output["node"], node_target)
    targets = edge_targets(patch_ids)
    edge_parts = [
        balanced_binary_loss(output[name], targets[name])
        for name in EDGE_NAMES
    ]
    edge = torch.stack(edge_parts).mean()
    total = node + edge
    return total, {
        "node_loss": float(node.detach()),
        "edge_loss": float(edge.detach()),
    }


@torch.no_grad()
def edge_validation_metrics(
    model: RealEdgePredictor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    node_positive = []
    node_negative = []
    edge_positive = []
    edge_negative = []
    for batch in loader:
        image = batch["image"].to(device)
        patch_ids = batch["patch_ids"].to(device)
        output = model(image)
        node_target = patch_ids > 0
        node_prediction = output["node"] >= 0
        node_positive.append(
            (node_prediction[node_target] == node_target[node_target])
            .float()
            .cpu()
        )
        node_negative.append(
            (node_prediction[~node_target] == node_target[~node_target])
            .float()
            .cpu()
        )
        targets = edge_targets(patch_ids)
        for name in EDGE_NAMES:
            prediction = output[name] >= 0
            target = targets[name]
            edge_positive.append(
                (prediction[target] == target[target]).float().cpu()
            )
            edge_negative.append(
                (prediction[~target] == target[~target]).float().cpu()
            )
    node_positive_accuracy = float(torch.cat(node_positive).mean())
    node_negative_accuracy = float(torch.cat(node_negative).mean())
    edge_positive_accuracy = float(torch.cat(edge_positive).mean())
    edge_negative_accuracy = float(torch.cat(edge_negative).mean())
    return {
        "node_balanced_accuracy": 0.5
        * (node_positive_accuracy + node_negative_accuracy),
        "edge_balanced_accuracy": 0.5
        * (edge_positive_accuracy + edge_negative_accuracy),
        "selection_score": 0.5
        * (
            0.5 * (node_positive_accuracy + node_negative_accuracy)
            + 0.5 * (edge_positive_accuracy + edge_negative_accuracy)
        ),
    }


def train_edge_predictor(
    args: argparse.Namespace,
    train_dataset: RealDataset,
    dev_dataset: RealDataset,
    output_dir: Path,
    device: torch.device,
) -> tuple[RealEdgePredictor, list[dict[str, Any]]]:
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
    )
    model = RealEdgePredictor(args.dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    checkpoint_path = output_dir / f"real_edge_seed{args.seed}.pt"
    history = []
    started = time.time()
    for epoch in range(1, args.edge_epochs + 1):
        model.train()
        sums = {"loss": 0.0, "node_loss": 0.0, "edge_loss": 0.0}
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            image = batch["image"].to(device, non_blocking=True)
            patch_ids = batch["patch_ids"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                prediction = model(image)
                loss, parts = edge_loss(prediction, patch_ids)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.detach())
            for key, value in parts.items():
                sums[key] += value
            batches += 1
        metrics = edge_validation_metrics(model, dev_loader, device)
        record = {
            "stage": "edge_predictor",
            "seed": args.seed,
            "epoch": epoch,
            **{key: value / batches for key, value in sums.items()},
            **{f"dev_{key}": value for key, value in metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"REAL_EDGE epoch={epoch}/{args.edge_epochs} "
            f"loss={record['loss']:.4f} "
            f"dev_node={metrics['node_balanced_accuracy']:.4f} "
            f"dev_edge={metrics['edge_balanced_accuracy']:.4f}",
            flush=True,
        )
        if metrics["selection_score"] > best:
            best = metrics["selection_score"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "dev_metrics": metrics,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    return model, history


@torch.no_grad()
def predict_probabilities(
    model: RealEdgePredictor,
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {
        "node": [],
        **{name: [] for name in EDGE_NAMES},
    }
    model.eval()
    for start in range(0, images.shape[0], batch_size):
        image = torch.from_numpy(
            images[start : start + batch_size].astype(np.float32) / 255.0
        ).to(device)
        output = model(image)
        for name in result:
            result[name].append(torch.sigmoid(output[name]).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in result.items()}


def predicted_groups(
    probabilities: dict[str, np.ndarray],
    index: int,
    node_threshold: float,
    edge_threshold: float,
    maximum_components: int = 15,
) -> np.ndarray:
    foreground = probabilities["node"][index] >= node_threshold
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
                neighbors: list[tuple[int, int, float]] = []
                if col > 0:
                    neighbors.append(
                        (
                            row,
                            col - 1,
                            probabilities["horizontal"][index, row, col - 1],
                        )
                    )
                if col < 15:
                    neighbors.append(
                        (
                            row,
                            col + 1,
                            probabilities["horizontal"][index, row, col],
                        )
                    )
                if row > 0:
                    neighbors.append(
                        (
                            row - 1,
                            col,
                            probabilities["vertical"][index, row - 1, col],
                        )
                    )
                if row < 15:
                    neighbors.append(
                        (
                            row + 1,
                            col,
                            probabilities["vertical"][index, row, col],
                        )
                    )
                if row > 0 and col > 0:
                    neighbors.append(
                        (
                            row - 1,
                            col - 1,
                            probabilities["diagonal_right"][
                                index, row - 1, col - 1
                            ],
                        )
                    )
                if row < 15 and col < 15:
                    neighbors.append(
                        (
                            row + 1,
                            col + 1,
                            probabilities["diagonal_right"][index, row, col],
                        )
                    )
                if row > 0 and col < 15:
                    neighbors.append(
                        (
                            row - 1,
                            col + 1,
                            probabilities["diagonal_left"][
                                index, row - 1, col
                            ],
                        )
                    )
                if row < 15 and col > 0:
                    neighbors.append(
                        (
                            row + 1,
                            col - 1,
                            probabilities["diagonal_left"][
                                index, row, col - 1
                            ],
                        )
                    )
                for new_row, new_col, probability in neighbors:
                    if (
                        foreground[new_row, new_col]
                        and groups[new_row, new_col] == 0
                        and probability >= edge_threshold
                    ):
                        groups[new_row, new_col] = next_group
                        queue.append((new_row, new_col))
            next_group += 1
    component_count = next_group - 1
    if component_count > maximum_components:
        sizes = np.bincount(groups.reshape(-1), minlength=component_count + 1)
        keep = set(
            (
                np.argsort(sizes[1:])[-maximum_components:] + 1
            ).tolist()
        )
        for component in range(1, component_count + 1):
            if component not in keep:
                groups[groups == component] = 0
        remap = {
            old: new
            for new, old in enumerate(
                [0, *sorted(int(item) for item in np.unique(groups) if item > 0)]
            )
        }
        groups = np.vectorize(remap.__getitem__, otypes=[np.int16])(groups)
    return groups


def group_structural_prediction(
    groups: np.ndarray, endpoints: np.ndarray
) -> int:
    first = int(endpoints[0])
    second = int(endpoints[1])
    first_group = int(groups[first // 16, first % 16])
    second_group = int(groups[second // 16, second % 16])
    return int(first_group > 0 and first_group == second_group)


def calibrate_thresholds(
    arrays: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    node_thresholds: list[float],
    edge_thresholds: list[float],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = []
    for node_threshold in node_thresholds:
        for edge_threshold in edge_thresholds:
            predictions = np.asarray(
                [
                    group_structural_prediction(
                        predicted_groups(
                            probabilities,
                            index,
                            node_threshold,
                            edge_threshold,
                        ),
                        arrays["endpoint_patches"][index],
                    )
                    for index in range(arrays["images"].shape[0])
                ],
                dtype=np.uint8,
            )
            metrics = balanced_metrics(arrays["connected"], predictions)
            rows.append(
                {
                    "node_threshold": node_threshold,
                    "edge_threshold": edge_threshold,
                    **metrics,
                }
            )
    selected = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["node_threshold"] + row["edge_threshold"],
        ),
    )
    return selected, rows


def predicted_artifacts(
    arrays: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    node_threshold: float,
    edge_threshold: float,
    token_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample_count = arrays["images"].shape[0]
    assignments = np.empty((sample_count, 16, 16), dtype=np.int16)
    adjacency = np.empty(
        (sample_count, token_count, token_count), dtype=np.uint8
    )
    reachability = np.empty_like(adjacency)
    structural = np.empty(sample_count, dtype=np.uint8)
    component_counts = []
    for index in range(sample_count):
        groups = predicted_groups(
            probabilities, index, node_threshold, edge_threshold
        )
        component_counts.append(int(np.unique(groups[groups > 0]).size))
        assignment = oracle_assignment(groups, token_count, False)
        graph, closure = coarse_graph8(
            assignment, groups, token_count, topology_aware=True
        )
        assignments[index] = assignment
        adjacency[index] = graph
        reachability[index] = closure
        structural[index] = group_structural_prediction(
            groups, arrays["endpoint_patches"][index]
        )
    counts = np.asarray(component_counts)
    return {
        "assignment": assignments,
        "adjacency": adjacency,
        "reachability": reachability,
    }, {
        "structural_metrics": balanced_metrics(
            arrays["connected"], structural
        ),
        "predicted_foreground_components": {
            "minimum": int(counts.min()),
            "median": float(np.median(counts)),
            "p95": float(np.quantile(counts, 0.95)),
            "maximum": int(counts.max()),
        },
        "all_tokens_nonempty": bool(
            all(
                np.unique(assignments[index]).size == token_count
                for index in range(sample_count)
            )
        ),
    }


class AttentionCoarsenModel(nn.Module):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__()
        self.stem = PatchStem(dim)
        self.queries = nn.Parameter(torch.randn(token_count, dim) * 0.02)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.project = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.head = CoarseGraphHead(dim, token_count)
        self.token_count = token_count
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.stem(image).flatten(2).transpose(1, 2)
        keys = self.key(features)
        scores = torch.einsum("kd,bnd->bkn", self.queries, keys)
        weights = torch.softmax(scores / math.sqrt(keys.shape[-1]), dim=-1)
        tokens = torch.bmm(weights, self.value(features))
        coordinates = self.coordinates.unsqueeze(0).expand(
            image.shape[0], -1, -1
        )
        centroids = torch.bmm(weights, coordinates)
        mass = torch.full(
            (image.shape[0], self.token_count, 1),
            1.0 / self.token_count,
            device=image.device,
            dtype=tokens.dtype,
        )
        metadata = torch.cat((centroids, mass), dim=-1)
        identity = torch.eye(
            self.token_count, device=image.device, dtype=tokens.dtype
        ).unsqueeze(0).expand(image.shape[0], -1, -1)
        return self.head(
            self.project(tokens), metadata, identity, identity
        )


def model_forward(
    model: nn.Module,
    model_name: str,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    image = batch["image"].to(device, non_blocking=True)
    if model_name == "attention":
        return model(image)
    return model(
        image,
        batch["assignment"].to(device, non_blocking=True),
        batch["adjacency"].to(device, non_blocking=True),
        batch["reachability"].to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False,
) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None]:
    model.eval()
    truth_parts = []
    prediction_parts = []
    for batch in loader:
        logits = model_forward(model, model_name, batch, device)
        truth_parts.append(batch["connected"].numpy().astype(np.uint8))
        prediction_parts.append(
            (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.uint8)
        )
    truth = np.concatenate(truth_parts)
    prediction = np.concatenate(prediction_parts)
    return (
        balanced_metrics(truth, prediction),
        prediction if return_predictions else None,
        truth if return_predictions else None,
    )


def train_classifier(
    model_name: str,
    args: argparse.Namespace,
    train_dataset: RealDataset,
    dev_dataset: RealDataset,
    test_dataset: RealDataset,
    output_dir: Path,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
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
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    branch_controls = {
        "mask_topo_branch_none": (False, False),
        "mask_topo_a_only": (True, False),
        "mask_topo_r_only": (False, True),
    }
    if model_name == "attention":
        model: nn.Module = AttentionCoarsenModel(
            args.dim, args.token_count
        )
    else:
        model = TopoCoarsenModel(
            model_name, args.dim, args.token_count
        )
        if model_name in branch_controls:
            use_local, use_reachability = branch_controls[model_name]
            for graph_layer in model.head.graph_layers:
                graph_layer.use_local = use_local
                graph_layer.use_reachability = use_reachability
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = output_dir / f"{model_name}_seed{args.seed}.pt"
    best = -1.0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = batch["connected"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model_forward(model, model_name, batch, device)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            batches += 1
        dev_metrics, _, _ = evaluate_classifier(
            model, model_name, dev_loader, device
        )
        record = {
            "stage": "classifier",
            "seed": args.seed,
            "model": model_name,
            "epoch": epoch,
            "train_loss": total_loss / batches,
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"REAL_CLASSIFIER model={model_name} "
            f"epoch={epoch}/{args.epochs} "
            f"loss={record['train_loss']:.4f} "
            f"dev_bal={dev_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        if dev_metrics["balanced_accuracy"] > best:
            best = dev_metrics["balanced_accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "model_name": model_name,
                    "dev_metrics": dev_metrics,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    test_metrics, prediction, truth = evaluate_classifier(
        model, model_name, test_loader, device, return_predictions=True
    )
    if prediction is None or truth is None:
        raise AssertionError("Missing test predictions.")
    graph_branch_controls = None
    learned_graph_gates = None
    if isinstance(model, TopoCoarsenModel):
        graph_layer = model.head.graph_layers[0]
        graph_branch_controls = {
            "use_local_adjacency_A": bool(graph_layer.use_local),
            "use_component_reachability_R": bool(
                graph_layer.use_reachability
            ),
        }
        learned_graph_gates = {
            "tanh_alpha": float(torch.tanh(graph_layer.local_gate).detach()),
            "tanh_beta": float(
                torch.tanh(graph_layer.component_gate).detach()
            ),
        }
    return {
        "model": model_name,
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_dev_balanced_accuracy": best,
        "test_metrics": test_metrics,
        "training_seconds": time.time() - started,
        "test_inference_uses_ground_truth_topology": (
            model_name == "topology_oracle"
        ),
        "graph_branch_controls": graph_branch_controls,
        "learned_graph_gates": learned_graph_gates,
    }, history, prediction, truth


def threshold_shortcut(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    candidates = np.unique(
        np.quantile(train_values, np.linspace(0.01, 0.99, 99))
    )
    best = (-1.0, 0.0, 1)
    for threshold in candidates:
        for direction in (-1, 1):
            prediction = (
                train_values <= threshold
                if direction == 1
                else train_values > threshold
            )
            accuracy = float(np.mean(prediction == train_labels))
            if accuracy > best[0]:
                best = (accuracy, float(threshold), direction)
    test_prediction = (
        test_values <= best[1] if best[2] == 1 else test_values > best[1]
    )
    return float(np.mean(test_prediction == test_labels))


def data_diagnostics(
    arrays_by_split: dict[str, dict[str, np.ndarray]],
    source_manifest: dict[str, list[str]],
) -> dict[str, Any]:
    train = arrays_by_split["train"]
    test = arrays_by_split["test"]
    overlaps = {}
    for first in ("train", "dev", "test"):
        for second in ("train", "dev", "test"):
            if first < second:
                overlaps[f"{first}_{second}"] = sorted(
                    set(source_manifest[first]) & set(source_manifest[second])
                )
    mean_intensity_train = train["images"][:, 0].mean(axis=(1, 2))
    mean_intensity_test = test["images"][:, 0].mean(axis=(1, 2))
    return {
        "source_counts": {
            split: len(ids) for split, ids in source_manifest.items()
        },
        "source_overlap": overlaps,
        "sample_counts": {
            split: int(arrays["images"].shape[0])
            for split, arrays in arrays_by_split.items()
        },
        "positive_fraction": {
            split: float(arrays["connected"].mean())
            for split, arrays in arrays_by_split.items()
        },
        "unique_sources_used": {
            split: int(np.unique(arrays["source_id"]).size)
            for split, arrays in arrays_by_split.items()
        },
        "endpoint_distance_shortcut_test_accuracy": threshold_shortcut(
            train["endpoint_distance"],
            train["connected"],
            test["endpoint_distance"],
            test["connected"],
        ),
        "mean_intensity_shortcut_test_accuracy": threshold_shortcut(
            mean_intensity_train,
            train["connected"],
            mean_intensity_test,
            test["connected"],
        ),
    }


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


def render_examples(
    arrays: dict[str, np.ndarray], output_path: Path
) -> None:
    indices = [0, 1, 2, 3, 4, 5]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    for axis, index in zip(axes.reshape(-1), indices):
        image = arrays["images"][index].astype(np.float32) / 255.0
        display = np.stack(
            (
                image[0],
                np.maximum(image[0], image[1]),
                np.maximum(image[0], image[2]),
            ),
            axis=-1,
        )
        axis.imshow(display)
        axis.set_title(
            f"label={int(arrays['connected'][index])}, "
            f"source={arrays['source_id'][index]}"
        )
        axis.axis("off")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_self_test(dataset_root: Path) -> None:
    manifest = source_split(dataset_root, 123)
    assert sum(len(value) for value in manifest.values()) == 118
    assert not (set(manifest["train"]) & set(manifest["test"]))
    stems = manifest["train"][:8]
    sources = load_sources(dataset_root, stems)
    namespace = argparse.Namespace(
        crop_size=128,
        output_size=64,
        min_component_pixels=12,
    )
    positive = make_sample(sources, stems, 1, 1234, 128, 64, 12)
    negative = make_sample(sources, stems, 0, 1234, 128, 64, 12)
    assert positive["connected"] == 1
    assert negative["connected"] == 0
    assert positive["image"].shape == (3, 64, 64)
    assert positive["patch_ids"].shape == (16, 16)
    generated = generate_arrays(sources, stems, 2, 1234, namespace)
    assert generated["images"].shape == (2, 3, 64, 64)
    assert generated["connected"].tolist() == [0, 1]
    model = RealEdgePredictor(32)
    output = model(torch.rand(2, 3, 64, 64))
    assert output["node"].shape == (2, 16, 16)
    attention = AttentionCoarsenModel(32, 16)
    assert attention(torch.rand(2, 3, 64, 64)).shape == (2,)
    print("CRACKFOREST_REAL_SELF_TEST_PASS")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if args.self_test:
        run_self_test(dataset_root)
        return
    if args.token_count != 16:
        raise ValueError("The controlled gate requires exactly 16 tokens.")
    unknown = sorted(set(args.models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    fingerprint = dataset_fingerprint(dataset_root)
    manifest = source_split(dataset_root, args.split_seed)
    all_stems = sorted(
        set().union(*[set(values) for values in manifest.values()])
    )
    sources = load_sources(dataset_root, all_stems)
    split_config = {
        "train": (args.train_size, args.seed),
        "dev": (args.dev_size, args.seed + 1_000_000),
        "test": (args.test_size, args.seed + 2_000_000),
    }
    arrays_by_split = {
        split: load_or_generate_split(
            cache_dir,
            split,
            sources,
            manifest[split],
            count,
            seed,
            args,
            fingerprint,
        )
        for split, (count, seed) in split_config.items()
    }
    diagnostics = data_diagnostics(arrays_by_split, manifest)
    split_manifest = {
        "dataset": "CrackForest Dataset",
        "dataset_fingerprint": fingerprint,
        "license": "non-commercial research purposes only",
        "source_image_split": manifest,
        "sample_generation": {
            split: {
                "count": split_config[split][0],
                "seed": split_config[split][1],
            }
            for split in split_config
        },
        "no_source_image_overlap": all(
            not values for values in diagnostics["source_overlap"].values()
        ),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "data_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    render_examples(
        arrays_by_split["test"], output_dir / "test_examples.png"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)
    if args.data_only:
        return

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
    train_edge_dataset = RealDataset(
        arrays_by_split["train"], augment=True
    )
    dev_edge_dataset = RealDataset(arrays_by_split["dev"])
    edge_model, history = train_edge_predictor(
        args,
        train_edge_dataset,
        dev_edge_dataset,
        output_dir,
        device,
    )
    probabilities = {
        split: predict_probabilities(
            edge_model,
            arrays["images"],
            args.batch_size,
            device,
        )
        for split, arrays in arrays_by_split.items()
    }
    selected_thresholds, threshold_rows = calibrate_thresholds(
        arrays_by_split["dev"],
        probabilities["dev"],
        args.node_thresholds,
        args.edge_thresholds,
    )
    write_csv(output_dir / "threshold_calibration.csv", threshold_rows)

    artifact_bank: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    artifact_diagnostics: dict[str, Any] = {}
    for split, arrays in arrays_by_split.items():
        grid_artifacts, grid_diagnostic = fixed_artifacts(
            arrays, args.token_count, "grid"
        )
        oracle_artifacts, oracle_diagnostic = fixed_artifacts(
            arrays, args.token_count, "topology"
        )
        edge_artifacts, edge_diagnostic = predicted_artifacts(
            arrays,
            probabilities[split],
            float(selected_thresholds["node_threshold"]),
            float(selected_thresholds["edge_threshold"]),
            args.token_count,
        )
        artifact_bank[split] = {
            "grid": grid_artifacts,
            "topology": oracle_artifacts,
            "edge": edge_artifacts,
        }
        artifact_diagnostics[split] = {
            "grid": grid_diagnostic,
            "topology_oracle": oracle_diagnostic,
            "edge_topocoarsen": edge_diagnostic,
        }
        print(
            f"REAL_ARTIFACT split={split} "
            f"edge_structural="
            f"{edge_diagnostic['structural_metrics']['balanced_accuracy']:.4f} "
            f"oracle_structural="
            f"{oracle_diagnostic['structural_metrics']['balanced_accuracy']:.4f}",
            flush=True,
        )

    summaries = []
    predictions_by_model: dict[str, np.ndarray] = {}
    truth = None
    for model_name in args.models:
        if model_name in {"grid", "conv"}:
            artifact_name = "grid"
        elif model_name == "topology_oracle":
            artifact_name = "topology"
        else:
            artifact_name = "edge"
        datasets = {
            split: RealDataset(
                arrays_by_split[split],
                artifact_bank[split][artifact_name],
                augment=split == "train",
            )
            for split in ("train", "dev", "test")
        }
        summary, model_history, prediction, current_truth = train_classifier(
            model_name,
            args,
            datasets["train"],
            datasets["dev"],
            datasets["test"],
            output_dir,
            device,
        )
        summaries.append(summary)
        history.extend(model_history)
        predictions_by_model[model_name] = prediction
        truth = current_truth
    if truth is None:
        raise AssertionError("No classifier results.")
    write_csv(output_dir / "history.csv", history)
    write_csv(
        output_dir / "summary.csv",
        [
            {
                "model": row["model"],
                "seed": row["seed"],
                "parameters": row["parameters"],
                "best_dev_balanced_accuracy": row[
                    "best_dev_balanced_accuracy"
                ],
                **{
                    f"test_{key}": value
                    for key, value in row["test_metrics"].items()
                },
                "training_seconds": row["training_seconds"],
            }
            for row in summaries
        ],
    )
    np.savez_compressed(
        output_dir / "heldout_predictions.npz",
        truth=truth,
        **predictions_by_model,
    )
    by_model = {row["model"]: row for row in summaries}
    baseline_names = [
        name for name in ("grid", "conv", "attention") if name in by_model
    ]
    strongest_baseline_name = max(
        baseline_names,
        key=lambda name: by_model[name]["test_metrics"][
            "balanced_accuracy"
        ],
    )
    baseline = by_model[strongest_baseline_name]["test_metrics"][
        "balanced_accuracy"
    ]
    edge_score = by_model["edge_topocoarsen"]["test_metrics"][
        "balanced_accuracy"
    ]
    gain = edge_score - baseline
    result = {
        "experiment_id": "crackforest_real_edge_topocoarsen_gate",
        "status": "completed",
        "dataset": {
            "name": "CrackForest Dataset",
            "annotated_source_images": 118,
            "license": "non-commercial research purposes only",
            "source_split_counts": diagnostics["source_counts"],
            "source_image_overlap": diagnostics["source_overlap"],
        },
        "task": (
            "Given a real pavement crop and two visible endpoint markers, "
            "predict whether the points are connected by the annotated crack."
        ),
        "data_diagnostics": diagnostics,
        "selected_thresholds_on_dev": selected_thresholds,
        "artifact_diagnostics": artifact_diagnostics,
        "model_results": by_model,
        "strongest_image_only_baseline": strongest_baseline_name,
        "edge_gain_over_strongest_baseline": gain,
        "edge_test_bootstrap_95_ci": bootstrap_balanced_accuracy(
            truth,
            predictions_by_model["edge_topocoarsen"],
            args.seed,
        ),
        "inference_contract": {
            "edge_topocoarsen_test_inputs": ["real crop", "visible markers"],
            "uses_ground_truth_patch_ids_at_test": False,
            "all_256_patches_aggregated": True,
            "exact_output_token_count": args.token_count,
            "topology_oracle_is_upper_bound_only": True,
        },
        "gates": {
            "source_split_leakage_absent": all(
                not values
                for values in diagnostics["source_overlap"].values()
            ),
            "distance_shortcut_at_most_55pct": diagnostics[
                "endpoint_distance_shortcut_test_accuracy"
            ]
            <= 0.55,
            "predicted_graph_structural_accuracy_at_least_70pct": (
                artifact_diagnostics["test"]["edge_topocoarsen"][
                    "structural_metrics"
                ]["balanced_accuracy"]
                >= 0.70
            ),
            "edge_gain_over_strongest_baseline_at_least_3pp": gain >= 0.03,
        },
    }
    result["verdict"] = (
        "PROCEED_TO_MULTISEED_SECOND_DATASET"
        if all(result["gates"].values())
        else (
            "REAL_GRAPH_LEARNABLE_CONNECTOR_REDESIGN"
            if result["gates"][
                "predicted_graph_structural_accuracy_at_least_70pct"
            ]
            else "STOP_GENERAL_CONNECTOR_CLAIM"
        )
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    names = [row["model"] for row in summaries]
    values = [
        row["test_metrics"]["balanced_accuracy"] for row in summaries
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.bar(
        names,
        values,
        color=[
            "#E45756" if name == "edge_topocoarsen" else "#4C78A8"
            for name in names
        ],
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("held-out balanced accuracy")
    ax.set_title("CrackForest real-data 256-to-16 token gate")
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

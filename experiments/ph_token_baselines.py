from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import gudhi
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from crackforest_real_gate import balanced_metrics, write_csv
from strong_reducer_baselines import (
    CommonTokenClassifier,
    ResamplerBlock,
    load_arrays as load_existing_arrays,
)
from topobridge_mvp import PatchStem, set_seed
from topocoarsen_oracle import fine_coordinates


MODELS = ("ph_only", "ph_guided")
SPLITS = ("train", "dev", "test")
DESCRIPTOR_DIM = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent-homology fixed-K token baselines for TopoBridge."
    )
    parser.add_argument(
        "--dataset", choices=("fives", "deepcrack", "crackforest"), required=True
    )
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--mask-probabilities", type=Path, required=False)
    parser.add_argument("--ph-cache", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("real_data/DeepCrack/dataset/extracted"),
    )
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--dev-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, required=False, default=16)
    parser.add_argument("--max-ph-tokens", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument("--ph-latency-repeats", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_sizes(dataset: str) -> tuple[int, int, int]:
    if dataset == "crackforest":
        return 1600, 320, 320
    return 2400, 600, 1200


def find_fives_cache(
    cache_dir: Path,
    split: str,
    count: int,
    seed: int,
    crop_size: int,
) -> Path:
    candidates = sorted(
        cache_dir.glob(
            f"fives_{split}_n{count}_seed{seed}_crop{crop_size}_matchedv1_*.npz"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one FIVES cache for {split}, found {len(candidates)}: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def load_arrays(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    defaults = default_sizes(args.dataset)
    args.train_size = args.train_size or defaults[0]
    args.dev_size = args.dev_size or defaults[1]
    args.test_size = args.test_size or defaults[2]
    if args.dataset != "fives":
        arrays = load_existing_arrays(args)
        return arrays, {split: "loaded_by_frozen_dataset_loader" for split in SPLITS}
    cache_dir = (args.cache_dir or Path("fives_cache_v2")).resolve()
    counts = {
        "train": args.train_size,
        "dev": args.dev_size,
        "test": args.test_size,
    }
    seeds = {
        "train": args.data_seed,
        "dev": args.data_seed + 1_000_000,
        "test": args.data_seed + 2_000_000,
    }
    paths = {
        split: find_fives_cache(
            cache_dir, split, counts[split], seeds[split], args.crop_size
        )
        for split in SPLITS
    }
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split, path in paths.items():
        with np.load(path) as archive:
            arrays[split] = {key: archive[key] for key in archive.files}
    return arrays, {split: str(path) for split, path in paths.items()}


def normalized_cell_coordinate(index: int, shape: tuple[int, int]) -> tuple[float, float]:
    row, column = np.unravel_index(index, shape, order="F")
    return (
        2.0 * (float(row) + 0.5) / shape[0] - 1.0,
        2.0 * (float(column) + 0.5) / shape[1] - 1.0,
    )


def ph_descriptors(
    mask_probability: np.ndarray,
    max_tokens: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mask_probability.ndim != 2:
        raise ValueError(f"Expected a 2-D mask probability, got {mask_probability.shape}.")
    probability = np.clip(mask_probability.astype(np.float64), 0.0, 1.0)
    filtration = 1.0 - probability
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=filtration)
    complex_.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    regular, essential = complex_.cofaces_of_persistence_pairs()
    flat = filtration.reshape(-1, order="F")
    records: list[tuple[float, int, int, int, list[float]]] = []
    for dimension, pairs in enumerate(regular):
        if dimension > 1:
            continue
        for birth_index, death_index in pairs:
            birth_index = int(birth_index)
            death_index = int(death_index)
            birth = float(flat[birth_index])
            death = float(flat[death_index])
            persistence = max(0.0, death - birth)
            birth_row, birth_column = normalized_cell_coordinate(
                birth_index, filtration.shape
            )
            death_row, death_column = normalized_cell_coordinate(
                death_index, filtration.shape
            )
            descriptor = [
                birth,
                death,
                persistence,
                float(dimension == 0),
                float(dimension == 1),
                0.0,
                1.0,
                birth_row,
                birth_column,
                death_row,
                death_column,
            ]
            records.append(
                (persistence, dimension, birth_index, death_index, descriptor)
            )
    for dimension, births in enumerate(essential):
        if dimension > 1:
            continue
        for birth_index in births:
            birth_index = int(birth_index)
            birth = float(flat[birth_index])
            death = 1.0
            persistence = max(0.0, death - birth)
            birth_row, birth_column = normalized_cell_coordinate(
                birth_index, filtration.shape
            )
            descriptor = [
                birth,
                death,
                persistence,
                float(dimension == 0),
                float(dimension == 1),
                1.0,
                1.0,
                birth_row,
                birth_column,
                birth_row,
                birth_column,
            ]
            records.append((persistence, dimension, birth_index, -1, descriptor))
    records.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    output = np.zeros((max_tokens, DESCRIPTOR_DIM), dtype=np.float32)
    selected = records[:max_tokens]
    for index, record in enumerate(selected):
        output[index] = np.asarray(record[-1], dtype=np.float32)
    persistence_values = np.asarray([record[0] for record in selected], dtype=np.float64)
    diagnostics = {
        "available_pairs": len(records),
        "selected_pairs": len(selected),
        "padding_count": max_tokens - len(selected),
        "selected_h0": sum(int(record[1] == 0) for record in selected),
        "selected_h1": sum(int(record[1] == 1) for record in selected),
        "selected_essential": sum(int(record[3] == -1) for record in selected),
        "persistence_min": (
            float(persistence_values.min()) if persistence_values.size else 0.0
        ),
        "persistence_median": (
            float(np.median(persistence_values)) if persistence_values.size else 0.0
        ),
        "persistence_max": (
            float(persistence_values.max()) if persistence_values.size else 0.0
        ),
    }
    return output, diagnostics


def aggregate_descriptor_diagnostics(
    descriptors: np.ndarray,
) -> dict[str, float | int]:
    valid = descriptors[..., 6] > 0.5
    h0 = descriptors[..., 3] > 0.5
    h1 = descriptors[..., 4] > 0.5
    essential = descriptors[..., 5] > 0.5
    persistence = descriptors[..., 2][valid]
    return {
        "sample_count": int(descriptors.shape[0]),
        "max_tokens": int(descriptors.shape[1]),
        "valid_pair_mean": float(valid.sum(axis=1).mean()),
        "padding_rate": float(1.0 - valid.mean()),
        "selected_h0_mean": float((valid & h0).sum(axis=1).mean()),
        "selected_h1_mean": float((valid & h1).sum(axis=1).mean()),
        "selected_essential_mean": float((valid & essential).sum(axis=1).mean()),
        "persistence_q05": (
            float(np.quantile(persistence, 0.05)) if persistence.size else 0.0
        ),
        "persistence_median": (
            float(np.median(persistence)) if persistence.size else 0.0
        ),
        "persistence_q95": (
            float(np.quantile(persistence, 0.95)) if persistence.size else 0.0
        ),
    }


def build_or_load_ph_cache(
    probabilities: dict[str, np.ndarray],
    mask_path: Path,
    cache_path: Path,
    max_tokens: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    mask_hash = file_sha256(mask_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata["mask_sha256"] != mask_hash:
                raise ValueError("PH cache mask SHA256 does not match the requested mask file.")
            if int(metadata["max_ph_tokens"]) != max_tokens:
                raise ValueError("PH cache max token count does not match.")
            if metadata["gudhi_version"] != gudhi.__version__:
                raise ValueError("PH cache GUDHI version does not match.")
            descriptors = {split: archive[split] for split in SPLITS}
        return descriptors, metadata
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptors: dict[str, np.ndarray] = {}
    split_metadata: dict[str, Any] = {}
    for split in SPLITS:
        count = probabilities[split].shape[0]
        current = np.empty((count, max_tokens, DESCRIPTOR_DIM), dtype=np.float32)
        started = time.perf_counter()
        for index in range(count):
            current[index], _ = ph_descriptors(probabilities[split][index], max_tokens)
            if (index + 1) % 250 == 0 or index + 1 == count:
                print(
                    f"PH_PRECOMPUTE split={split} progress={index + 1}/{count}",
                    flush=True,
                )
        elapsed = time.perf_counter() - started
        descriptors[split] = current
        split_metadata[split] = {
            "seconds": elapsed,
            "milliseconds_per_sample": 1000.0 * elapsed / count,
            "diagnostics": aggregate_descriptor_diagnostics(current),
        }
    metadata = {
        "format_version": "topobridge_ph_descriptor_v1",
        "gudhi_version": gudhi.__version__,
        "mask_path": str(mask_path),
        "mask_sha256": mask_hash,
        "max_ph_tokens": max_tokens,
        "descriptor_dimension": DESCRIPTOR_DIM,
        "filtration": "1_minus_predicted_foreground_probability",
        "homology_dimensions": [0, 1],
        "critical_cell_index_order": "numpy_fortran",
        "split_metadata": split_metadata,
    }
    np.savez_compressed(
        cache_path,
        **descriptors,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    metadata["cache_sha256"] = file_sha256(cache_path)
    return descriptors, metadata


class PHDescriptorDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        mask_probabilities: np.ndarray,
        descriptors: np.ndarray,
        token_count: int,
        augment: bool = False,
    ) -> None:
        count = int(arrays["images"].shape[0])
        if mask_probabilities.shape != (count, 64, 64):
            raise ValueError("Mask probability shape does not align with images.")
        if descriptors.shape[0] != count or descriptors.shape[2] != DESCRIPTOR_DIM:
            raise ValueError("PH descriptor shape does not align with images.")
        if token_count > descriptors.shape[1]:
            raise ValueError("Requested token count exceeds cached PH descriptors.")
        self.images = arrays["images"]
        self.labels = arrays["connected"]
        self.mask_probabilities = mask_probabilities.astype(np.float32)
        self.descriptors = descriptors[:, :token_count].astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(self.images[index].astype(np.float32) / 255.0)
        if self.augment:
            contrast = 0.82 + 0.28 * torch.rand(())
            brightness = -0.04 + 0.08 * torch.rand(())
            noise = 0.025 * torch.rand(()) * torch.randn_like(image)
            image[0] = torch.clamp(
                brightness + contrast * image[0] + noise[0], 0.0, 1.0
            )
        return {
            "image": image,
            "mask_probability": torch.from_numpy(
                self.mask_probabilities[index]
            ).unsqueeze(0),
            "ph_descriptor": torch.from_numpy(self.descriptors[index]),
            "connected": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class PHOnlyModel(CommonTokenClassifier):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__(dim, token_count)
        self.descriptor = nn.Sequential(
            nn.Linear(DESCRIPTOR_DIM, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(
        self,
        image: torch.Tensor,
        mask_probability: torch.Tensor,
        descriptor: torch.Tensor,
    ) -> torch.Tensor:
        del image, mask_probability
        tokens = self.descriptor(descriptor)
        persistence = descriptor[..., 2:3] * descriptor[..., 6:7]
        mass = persistence / persistence.sum(dim=1, keepdim=True).clamp_min(1e-6)
        metadata = torch.cat((descriptor[..., 7:9], mass), dim=-1)
        return self.classify(tokens, metadata)


class PHGuidedModel(CommonTokenClassifier):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__(dim, token_count)
        self.stem = PatchStem(dim)
        self.descriptor = nn.Sequential(
            nn.Linear(DESCRIPTOR_DIM, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.position = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList([ResamplerBlock(dim) for _ in range(2)])
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def forward(
        self,
        image: torch.Tensor,
        mask_probability: torch.Tensor,
        descriptor: torch.Tensor,
    ) -> torch.Tensor:
        del mask_probability
        context = self.stem(image).flatten(2).transpose(1, 2)
        coordinates = self.coordinates.to(context.dtype).unsqueeze(0).expand(
            image.shape[0], -1, -1
        )
        context = context + self.position(coordinates)
        latents = self.descriptor(descriptor)
        attention_weights = None
        for block in self.blocks:
            latents, attention_weights = block(latents, context)
        if attention_weights is None:
            raise AssertionError("PH-guided resampler returned no attention weights.")
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        centroids = torch.bmm(attention_weights, coordinates)
        persistence = descriptor[..., 2:3] * descriptor[..., 6:7]
        mass = persistence / persistence.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return self.classify(latents, torch.cat((centroids, mass), dim=-1))


def make_model(name: str, dim: int, token_count: int) -> nn.Module:
    if name == "ph_only":
        return PHOnlyModel(dim, token_count)
    if name == "ph_guided":
        return PHGuidedModel(dim, token_count)
    raise ValueError(name)


def forward_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    return model(
        batch["image"].to(device, non_blocking=True),
        batch["mask_probability"].to(device, non_blocking=True),
        batch["ph_descriptor"].to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False,
) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    model.eval()
    truth_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    for batch in loader:
        logits = forward_batch(model, batch, device)
        truth_parts.append(batch["connected"].numpy().astype(np.uint8))
        probability_parts.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
    truth = np.concatenate(truth_parts)
    probability = np.concatenate(probability_parts)
    prediction = (probability >= 0.5).astype(np.uint8)
    return (
        balanced_metrics(truth, prediction),
        prediction if return_predictions else None,
        probability if return_predictions else None,
        truth if return_predictions else None,
    )


@torch.no_grad()
def benchmark_model(
    model: nn.Module,
    dataset: PHDescriptorDataset,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, dict[str, float | int | None]]:
    model.eval()
    output: dict[str, dict[str, float | int | None]] = {}
    for requested_batch in (1, 8, 32):
        batch_size = min(requested_batch, len(dataset))
        items = [dataset[index] for index in range(batch_size)]
        image = torch.stack([item["image"] for item in items]).to(device)
        mask = torch.stack([item["mask_probability"] for item in items]).to(device)
        descriptor = torch.stack([item["ph_descriptor"] for item in items]).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            model(image, mask, descriptor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(image, mask, descriptor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        output[str(requested_batch)] = {
            "batch_size": batch_size,
            "latency_ms_per_batch": 1000.0 * elapsed / repeats,
            "latency_ms_per_sample": 1000.0 * elapsed / repeats / batch_size,
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else None
            ),
        }
    return output


def benchmark_ph_preprocessing(
    mask_probabilities: np.ndarray,
    max_tokens: int,
    repeats: int,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for requested_batch in (1, 8, 32):
        batch_size = min(requested_batch, mask_probabilities.shape[0])
        started = time.perf_counter()
        for _ in range(repeats):
            for index in range(batch_size):
                ph_descriptors(mask_probabilities[index], max_tokens)
        elapsed = time.perf_counter() - started
        output[str(requested_batch)] = {
            "batch_size": batch_size,
            "repeats": repeats,
            "latency_ms_per_batch": 1000.0 * elapsed / repeats,
            "latency_ms_per_sample": 1000.0 * elapsed / repeats / batch_size,
        }
    return output


def train_one(
    name: str,
    args: argparse.Namespace,
    datasets: dict[str, PHDescriptorDataset],
    output: Path,
    device: torch.device,
    ph_latency: dict[str, dict[str, float | int]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    set_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed),
        ),
        "dev": DataLoader(
            datasets["dev"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }
    model = make_model(name, args.dim, args.token_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = output / f"{name}_seed{args.seed}.pt"
    best = -1.0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            target = batch["connected"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = forward_batch(model, batch, device)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batches += 1
        dev_metrics, _, _, _ = evaluate(model, loaders["dev"], device)
        record = {
            "stage": "ph_token_baseline",
            "dataset": args.dataset,
            "seed": args.seed,
            "model": name,
            "token_count": args.token_count,
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"PH_BASELINE dataset={args.dataset} K={args.token_count} "
            f"model={name} epoch={epoch}/{args.epochs} "
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
                    "model_name": name,
                    "dev_metrics": dev_metrics,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, prediction, probability, truth = evaluate(
        model, loaders["test"], device, return_predictions=True
    )
    if prediction is None or probability is None or truth is None:
        raise AssertionError("Test predictions were not returned.")
    model_latency = benchmark_model(
        model,
        datasets["test"],
        device,
        args.latency_warmup,
        args.latency_repeats,
    )
    end_to_end: dict[str, dict[str, float | int | None]] = {}
    for batch in ("1", "8", "32"):
        model_item = model_latency[batch]
        ph_item = ph_latency[batch]
        total_batch = float(model_item["latency_ms_per_batch"]) + float(
            ph_item["latency_ms_per_batch"]
        )
        end_to_end[batch] = {
            "batch_size": int(model_item["batch_size"]),
            "ph_preprocessing_ms_per_batch": ph_item["latency_ms_per_batch"],
            "model_ms_per_batch": model_item["latency_ms_per_batch"],
            "conservative_total_ms_per_batch": total_batch,
            "conservative_total_ms_per_sample": total_batch
            / int(model_item["batch_size"]),
            "peak_cuda_memory_mb": model_item["peak_cuda_memory_mb"],
        }
    return (
        {
            "model": name,
            "seed": args.seed,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_dev_balanced_accuracy": best,
            "test_metrics": test_metrics,
            "training_seconds": time.time() - started,
            "model_latency": model_latency,
            "end_to_end_latency": end_to_end,
            "uses_predicted_mask": True,
            "uses_ground_truth_mask_or_topology": False,
            "ph_representation": "cubical_H0_H1_fixed_K",
        },
        history,
        prediction,
        probability,
        truth,
    )


def run_self_test() -> None:
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[5:30, 5:30] = 0.9
    probability[35:60, 35:60] = 0.8
    probability[12:23, 12:23] = 0.1
    descriptors, diagnostics = ph_descriptors(probability, 16)
    assert descriptors.shape == (16, DESCRIPTOR_DIM)
    assert diagnostics["selected_h0"] >= 1
    valid_persistence = descriptors[descriptors[:, 6] > 0.5, 2]
    assert np.all(valid_persistence[:-1] >= valid_persistence[1:])
    arrays = {
        "images": np.zeros((4, 3, 64, 64), dtype=np.uint8),
        "connected": np.asarray([0, 1, 0, 1], dtype=np.uint8),
    }
    all_descriptors = np.repeat(descriptors[None], 4, axis=0)
    dataset = PHDescriptorDataset(
        arrays, np.repeat(probability[None], 4, axis=0), all_descriptors, 8, True
    )
    batch = {
        key: torch.stack([dataset[0][key], dataset[1][key]])
        for key in ("image", "mask_probability", "ph_descriptor", "connected")
    }
    for name in MODELS:
        model = make_model(name, 32, 8)
        logits = model(
            batch["image"], batch["mask_probability"], batch["ph_descriptor"]
        )
        assert logits.shape == (2,)
        logits.square().mean().backward()
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    print("PH_TOKEN_BASELINES_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.output is None or args.mask_probabilities is None:
        raise ValueError("--output and --mask-probabilities are required.")
    if args.crop_size != 64 or args.output_size != 64:
        raise ValueError("The frozen PH protocol requires 64x64 inputs.")
    if args.token_count not in {8, 16, 32, 64}:
        raise ValueError("token-count must be one of 8, 16, 32, or 64.")
    if args.max_ph_tokens < args.token_count:
        raise ValueError("max-ph-tokens must be at least token-count.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays, array_sources = load_arrays(args)
    mask_path = args.mask_probabilities.resolve()
    with np.load(mask_path) as archive:
        probabilities = {
            split: archive[split][: arrays[split]["images"].shape[0]].astype(
                np.float32
            )
            for split in SPLITS
        }
    for split in SPLITS:
        expected = (arrays[split]["images"].shape[0], 64, 64)
        if probabilities[split].shape != expected:
            raise ValueError(
                f"Mask probability mismatch for {split}: "
                f"{probabilities[split].shape} != {expected}."
            )
    ph_cache = (
        args.ph_cache.resolve()
        if args.ph_cache is not None
        else output / "ph_descriptors_k64.npz"
    )
    descriptors, ph_metadata = build_or_load_ph_cache(
        probabilities, mask_path, ph_cache, args.max_ph_tokens
    )
    if "cache_sha256" not in ph_metadata:
        ph_metadata["cache_sha256"] = file_sha256(ph_cache)
    datasets = {
        split: PHDescriptorDataset(
            arrays[split],
            probabilities[split],
            descriptors[split],
            args.token_count,
            augment=split == "train",
        )
        for split in SPLITS
    }
    ph_latency = benchmark_ph_preprocessing(
        probabilities["test"], args.max_ph_tokens, args.ph_latency_repeats
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    prediction_probabilities: dict[str, np.ndarray] = {}
    truth_reference: np.ndarray | None = None
    for name in args.models:
        print(
            f"PH_BASELINE_START dataset={args.dataset} K={args.token_count} "
            f"model={name} seed={args.seed}",
            flush=True,
        )
        summary, history, prediction, probability, truth = train_one(
            name, args, datasets, output, device, ph_latency
        )
        if truth_reference is None:
            truth_reference = truth
        elif not np.array_equal(truth_reference, truth):
            raise RuntimeError("Held-out truth changed between PH models.")
        summaries[name] = summary
        histories.extend(history)
        predictions[name] = prediction
        prediction_probabilities[f"{name}_probability"] = probability
        print(
            f"PH_BASELINE_DONE dataset={args.dataset} K={args.token_count} "
            f"model={name} seed={args.seed} "
            f"test_bal={summary['test_metrics']['balanced_accuracy']:.4f}",
            flush=True,
        )
    if truth_reference is None:
        raise RuntimeError("No PH model was run.")
    write_csv(output / "history.csv", histories)
    prediction_payload: dict[str, np.ndarray] = {
        "truth": truth_reference,
        **predictions,
        **prediction_probabilities,
    }
    if "source_id" in arrays["test"]:
        prediction_payload["source_id"] = arrays["test"]["source_id"]
    np.savez_compressed(output / "heldout_predictions.npz", **prediction_payload)
    result = {
        "experiment_id": "TB-B-260802-022",
        "status": "completed",
        "confirmatory_status": "retrospective_collision",
        "dataset": args.dataset,
        "optimization_seed": args.seed,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "sample_counts": {split: len(datasets[split]) for split in SPLITS},
        "fine_token_count": 256,
        "reduced_token_count": args.token_count,
        "models": list(args.models),
        "model_results": summaries,
        "ph_preprocessing_latency": ph_latency,
        "ph_metadata": ph_metadata,
        "ph_diagnostics_at_k": {
            split: aggregate_descriptor_diagnostics(
                descriptors[split][:, : args.token_count]
            )
            for split in SPLITS
        },
        "mask_probabilities": {
            "path": str(mask_path),
            "sha256": file_sha256(mask_path),
        },
        "array_sources": array_sources,
        "protocol": str((Path(__file__).parent / "PH_TOKEN_BASELINE_PROTOCOL.md").resolve()),
        "protocol_sha256": file_sha256(
            Path(__file__).parent / "PH_TOKEN_BASELINE_PROTOCOL.md"
        ),
        "code_sha256": file_sha256(Path(__file__)),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gudhi": gudhi.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "uses_ground_truth_mask_or_topology_at_inference": False,
        "test_was_previously_observed": True,
        "selection_uses_development_only": True,
        "massachusetts_test_reopened": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

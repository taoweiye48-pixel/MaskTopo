from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from crackforest_mask_topo import find_cached_split
from crackforest_real_gate import balanced_metrics, write_csv
from deepcrack_external_gate import load_all_arrays
from topobridge_mvp import PatchStem, set_seed
from topocoarsen_oracle import CoarseGraphHead, fine_coordinates


REDUCERS = (
    "tokenlearner",
    "perceiver_resampler",
    "tome_style",
    "mask_guided_queries",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen strong token-reducer baselines for TopoBridge."
    )
    parser.add_argument(
        "--dataset", choices=("crackforest", "deepcrack"), required=True
    )
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--mask-probabilities", type=Path, required=False)
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
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--reducers", nargs="+", choices=REDUCERS, default=list(REDUCERS))
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


class ReducerDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        mask_probabilities: np.ndarray,
        augment: bool = False,
    ) -> None:
        count = int(arrays["images"].shape[0])
        if mask_probabilities.shape != (count, 64, 64):
            raise ValueError(
                "Mask probabilities must have shape "
                f"({count}, 64, 64), got {mask_probabilities.shape}."
            )
        self.images = arrays["images"]
        self.labels = arrays["connected"]
        self.mask_probabilities = mask_probabilities.astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(
            self.images[index].astype(np.float32) / 255.0
        )
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
            "connected": torch.tensor(
                self.labels[index], dtype=torch.float32
            ),
        }


class CommonTokenClassifier(nn.Module):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__()
        self.token_count = token_count
        self.project = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.head = CoarseGraphHead(dim, token_count)

    def classify(
        self, tokens: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        identity = torch.eye(
            self.token_count,
            device=tokens.device,
            dtype=tokens.dtype,
        ).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        return self.head(
            self.project(tokens), metadata, identity, identity
        )


class TokenLearnerModel(CommonTokenClassifier):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__(dim, token_count)
        self.stem = PatchStem(dim)
        self.score = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, token_count, kernel_size=1),
        )
        self.value = nn.Conv2d(dim, dim, kernel_size=1)
        self.register_buffer(
            "coordinates", fine_coordinates(), persistent=False
        )

    def forward(
        self,
        image: torch.Tensor,
        mask_probability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask_probability
        features = self.stem(image)
        values = self.value(features).flatten(2).transpose(1, 2)
        raw_attention = torch.sigmoid(self.score(features)).flatten(2)
        mass_raw = raw_attention.sum(dim=-1).clamp_min(1e-6)
        weights = raw_attention / mass_raw.unsqueeze(-1)
        tokens = torch.bmm(weights, values)
        coordinate_bank = self.coordinates.to(values.dtype).unsqueeze(0).expand(
            image.shape[0], -1, -1
        )
        centroids = torch.bmm(weights, coordinate_bank)
        mass = (mass_raw / mass_raw.sum(dim=-1, keepdim=True)).unsqueeze(-1)
        return self.classify(tokens, torch.cat((centroids, mass), dim=-1))


class ResamplerBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * dim, dim),
        )

    def forward(
        self, latents: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update, weights = self.cross_attention(
            self.query_norm(latents),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=True,
            average_attn_weights=True,
        )
        latents = latents + update
        latents = latents + self.feed_forward(latents)
        return latents, weights


class PerceiverResamplerModel(CommonTokenClassifier):
    def __init__(
        self, dim: int, token_count: int, mask_guided: bool
    ) -> None:
        super().__init__(dim, token_count)
        self.stem = PatchStem(dim)
        self.queries = nn.Parameter(torch.randn(token_count, dim) * 0.02)
        self.position = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.mask_guided = mask_guided
        self.mask_embedding = (
            nn.Sequential(
                nn.Linear(1, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            if mask_guided
            else None
        )
        self.blocks = nn.ModuleList([ResamplerBlock(dim) for _ in range(2)])
        self.register_buffer(
            "coordinates", fine_coordinates(), persistent=False
        )

    def forward(
        self,
        image: torch.Tensor,
        mask_probability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.stem(image).flatten(2).transpose(1, 2)
        coordinates = self.coordinates.to(context.dtype).unsqueeze(0).expand(
            image.shape[0], -1, -1
        )
        context = context + self.position(coordinates)
        if self.mask_guided:
            if mask_probability is None or self.mask_embedding is None:
                raise ValueError("mask_guided_queries requires a predicted mask.")
            mask_tokens = F.adaptive_avg_pool2d(
                mask_probability, (16, 16)
            ).flatten(2).transpose(1, 2)
            context = context + self.mask_embedding(mask_tokens)
        latents = self.queries.unsqueeze(0).expand(image.shape[0], -1, -1)
        weights = None
        for block in self.blocks:
            latents, weights = block(latents, context)
        if weights is None:
            raise AssertionError("Resampler did not return attention weights.")
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        centroids = torch.bmm(weights, coordinates)
        mass = torch.full(
            (image.shape[0], self.token_count, 1),
            1.0 / self.token_count,
            device=image.device,
            dtype=latents.dtype,
        )
        return self.classify(
            latents, torch.cat((centroids, mass), dim=-1)
        )


class ToMeStyleModel(CommonTokenClassifier):
    def __init__(
        self,
        dim: int,
        token_count: int,
        sinkhorn_iterations: int = 3,
        temperature: float = 0.10,
    ) -> None:
        super().__init__(dim, token_count)
        if token_count <= 0 or 256 % token_count:
            raise ValueError("token_count must divide 256.")
        ratio = 256 // token_count
        if ratio & (ratio - 1):
            raise ValueError("256/token_count must be a power of two.")
        self.stem = PatchStem(dim)
        self.position = nn.Linear(2, dim)
        self.sinkhorn_iterations = sinkhorn_iterations
        self.temperature = temperature
        self.merge_rounds = int(math.log2(ratio))
        self.register_buffer(
            "coordinates", fine_coordinates(), persistent=False
        )

    def soft_pairing(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_coordinates: torch.Tensor,
        right_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        left_key = F.normalize(
            left + self.position(left_coordinates), dim=-1
        )
        right_key = F.normalize(
            right + self.position(right_coordinates), dim=-1
        )
        log_pairing = torch.bmm(
            left_key, right_key.transpose(1, 2)
        ) / self.temperature
        for _ in range(self.sinkhorn_iterations):
            log_pairing = log_pairing - torch.logsumexp(
                log_pairing, dim=-1, keepdim=True
            )
            log_pairing = log_pairing - torch.logsumexp(
                log_pairing, dim=-2, keepdim=True
            )
        return torch.exp(log_pairing)

    def merge_once(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, right = tokens[:, 0::2], tokens[:, 1::2]
        left_coordinates = coordinates[:, 0::2]
        right_coordinates = coordinates[:, 1::2]
        left_mass, right_mass = mass[:, 0::2], mass[:, 1::2]
        pairing = self.soft_pairing(
            left, right, left_coordinates, right_coordinates
        )
        transported_mass = torch.bmm(
            pairing.transpose(1, 2), left_mass
        )
        new_mass = (right_mass + transported_mass).clamp_min(1e-6)
        transported_tokens = torch.bmm(
            pairing.transpose(1, 2), left * left_mass
        )
        transported_coordinates = torch.bmm(
            pairing.transpose(1, 2), left_coordinates * left_mass
        )
        new_tokens = (
            right * right_mass + transported_tokens
        ) / new_mass
        new_coordinates = (
            right_coordinates * right_mass + transported_coordinates
        ) / new_mass
        return new_tokens, new_coordinates, new_mass

    def forward(
        self,
        image: torch.Tensor,
        mask_probability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask_probability
        tokens = self.stem(image).flatten(2).transpose(1, 2)
        coordinates = self.coordinates.to(tokens.dtype).unsqueeze(0).expand(
            image.shape[0], -1, -1
        )
        mass = torch.full(
            (image.shape[0], 256, 1),
            1.0 / 256,
            device=image.device,
            dtype=tokens.dtype,
        )
        for _ in range(self.merge_rounds):
            tokens, coordinates, mass = self.merge_once(
                tokens, coordinates, mass
            )
        if tokens.shape[1] != self.token_count:
            raise AssertionError("ToMe-style merging produced the wrong token count.")
        mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return self.classify(
            tokens, torch.cat((coordinates, mass), dim=-1)
        )


def make_model(name: str, dim: int, token_count: int) -> nn.Module:
    if name == "tokenlearner":
        return TokenLearnerModel(dim, token_count)
    if name == "perceiver_resampler":
        return PerceiverResamplerModel(dim, token_count, mask_guided=False)
    if name == "tome_style":
        return ToMeStyleModel(dim, token_count)
    if name == "mask_guided_queries":
        return PerceiverResamplerModel(dim, token_count, mask_guided=True)
    raise ValueError(name)


def forward_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    return model(
        batch["image"].to(device, non_blocking=True),
        batch["mask_probability"].to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False,
) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None]:
    model.eval()
    truth_parts = []
    prediction_parts = []
    for batch in loader:
        logits = forward_batch(model, batch, device)
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


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | None]:
    model.eval()
    batch = next(iter(loader))
    image = batch["image"].to(device)
    mask = batch["mask_probability"].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        model(image, mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        model(image, mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "batch_size": int(image.shape[0]),
        "latency_ms_per_batch": 1000.0 * elapsed / repeats,
        "latency_ms_per_sample": 1000.0 * elapsed / repeats / image.shape[0],
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else None
        ),
    }


def train_one(
    name: str,
    args: argparse.Namespace,
    datasets: dict[str, ReducerDataset],
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
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
        dev_metrics, _, _ = evaluate(
            model, loaders["dev"], device
        )
        record = {
            "stage": "strong_reducer",
            "dataset": args.dataset,
            "seed": args.seed,
            "model": name,
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"STRONG_REDUCER dataset={args.dataset} model={name} "
            f"epoch={epoch}/{args.epochs} loss={record['train_loss']:.4f} "
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
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    test_metrics, prediction, truth = evaluate(
        model,
        loaders["test"],
        device,
        return_predictions=True,
    )
    if prediction is None or truth is None:
        raise AssertionError("Test predictions were not returned.")
    inference = benchmark_inference(
        model,
        loaders["test"],
        device,
        args.latency_warmup,
        args.latency_repeats,
    )
    return {
        "model": name,
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_dev_balanced_accuracy": best,
        "test_metrics": test_metrics,
        "training_seconds": time.time() - started,
        "inference": inference,
        "uses_predicted_mask": name == "mask_guided_queries",
        "uses_ground_truth_mask_or_topology": False,
    }, history, prediction, truth


def default_sizes(dataset: str) -> tuple[int, int, int]:
    if dataset == "crackforest":
        return 1600, 320, 320
    return 2400, 600, 1200


def load_arrays(args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    defaults = default_sizes(args.dataset)
    args.train_size = args.train_size or defaults[0]
    args.dev_size = args.dev_size or defaults[1]
    args.test_size = args.test_size or defaults[2]
    if args.dataset == "crackforest":
        cache = (args.cache_dir or Path("real_cache")).resolve()
        return {
            "train": find_cached_split(
                cache, "train", args.train_size, args.data_seed, args.crop_size
            ),
            "dev": find_cached_split(
                cache,
                "dev",
                args.dev_size,
                args.data_seed + 1_000_000,
                args.crop_size,
            ),
            "test": find_cached_split(
                cache,
                "test",
                args.test_size,
                args.data_seed + 2_000_000,
                args.crop_size,
            ),
        }
    args.cache_dir = args.cache_dir or Path("deepcrack_cache")
    arrays, _, _, _ = load_all_arrays(args)
    return arrays


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_self_test() -> None:
    set_seed(123)
    image = torch.rand(2, 3, 64, 64)
    mask = torch.rand(2, 1, 64, 64)
    for token_count in (8, 16, 32, 64):
        for name in REDUCERS:
            model = make_model(name, 32, token_count)
            logits = model(image, mask)
            assert logits.shape == (2,), (name, token_count, logits.shape)
            loss = logits.square().mean()
            loss.backward()
            assert all(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            ), (name, token_count)
    arrays = {
        "images": np.zeros((2, 3, 64, 64), dtype=np.uint8),
        "connected": np.array([0, 1], dtype=np.uint8),
    }
    dataset = ReducerDataset(
        arrays, np.zeros((2, 64, 64), dtype=np.float32), augment=True
    )
    assert dataset[0]["image"].shape == (3, 64, 64)
    assert dataset[0]["mask_probability"].shape == (1, 64, 64)
    print("STRONG_REDUCER_BASELINES_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.output is None or args.mask_probabilities is None:
        raise ValueError("--output and --mask-probabilities are required.")
    if args.crop_size != 64 or args.output_size != 64:
        raise ValueError("This frozen protocol requires 64x64 inputs.")
    if args.token_count not in {8, 16, 32, 64}:
        raise ValueError("token-count must be one of 8, 16, 32, or 64.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(args)
    mask_path = args.mask_probabilities.resolve()
    with np.load(mask_path) as archive:
        probabilities = {
            split: archive[split][
                : arrays[split]["images"].shape[0]
            ].astype(np.float32)
            for split in ("train", "dev", "test")
        }
    datasets = {
        split: ReducerDataset(
            arrays[split],
            probabilities[split],
            augment=split == "train",
        )
        for split in ("train", "dev", "test")
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    histories: list[dict[str, Any]] = []
    truth_reference: np.ndarray | None = None
    for name in args.reducers:
        print(
            f"STRONG_REDUCER_START dataset={args.dataset} "
            f"model={name} seed={args.seed}",
            flush=True,
        )
        summary, history, prediction, truth = train_one(
            name, args, datasets, output, device
        )
        if truth_reference is None:
            truth_reference = truth
        elif not np.array_equal(truth_reference, truth):
            raise RuntimeError("Held-out truth changed between reducers.")
        summaries[name] = summary
        predictions[name] = prediction
        histories.extend(history)
        print(
            f"STRONG_REDUCER_DONE dataset={args.dataset} "
            f"model={name} seed={args.seed} "
            f"test_bal={summary['test_metrics']['balanced_accuracy']:.4f}",
            flush=True,
        )
    if truth_reference is None:
        raise RuntimeError("No reducer was run.")
    write_csv(output / "history.csv", histories)
    prediction_payload: dict[str, np.ndarray] = {
        "truth": truth_reference,
        **predictions,
    }
    if args.dataset == "deepcrack":
        prediction_payload["source_id"] = arrays["test"]["source_id"]
    np.savez_compressed(
        output / "heldout_predictions.npz", **prediction_payload
    )
    result = {
        "experiment_id": "topobridge_token_budget_reducer_baselines",
        "status": "completed",
        "dataset": args.dataset,
        "optimization_seed": args.seed,
        "data_seed": args.data_seed,
        "sample_counts": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        "fine_token_count": 256,
        "reduced_token_count": args.token_count,
        "mask_probabilities": {
            "path": str(mask_path),
            "sha256": file_sha256(mask_path),
        },
        "reducers": list(args.reducers),
        "model_results": summaries,
        "test_was_previously_observed": True,
        "selection_uses_development_only": True,
        "protocol": str(
            (
                Path(__file__).parent
                / (
                    "STRONG_REDUCER_BASELINE_PROTOCOL.md"
                    if args.token_count == 16
                    else "TOKEN_BUDGET_PROTOCOL.md"
                )
            ).resolve()
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

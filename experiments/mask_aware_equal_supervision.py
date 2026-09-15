from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crackforest_real_gate import balanced_metrics, write_csv
from strong_reducer_baselines import CommonTokenClassifier, ReducerDataset, load_arrays
from topobridge_mvp import PatchStem, set_seed
from topocoarsen_oracle import fine_coordinates


EXPERIMENT_ID = "TB-B-260803-032"
MODELS = (
    "mask_conditioned_perceiver_strong",
    "mask_conditioned_slot_param_matched",
)
MASKTOPO_PARAMETER_ANCHOR = 149_123
MATCH_TOLERANCE = 0.01
FROZEN_DATA_SEED = 20260810
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepCrack K16 equal-supervision mask-aware non-topological baselines."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mask-probabilities", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("deepcrack_cache"))
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
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_mask(mask_probability: torch.Tensor) -> torch.Tensor:
    if mask_probability.ndim != 4 or mask_probability.shape[1] != 1:
        raise ValueError(f"Expected Bx1xHxW mask, got {tuple(mask_probability.shape)}")
    return F.adaptive_avg_pool2d(mask_probability, (16, 16)).flatten(2).squeeze(1)


def foreground_log_bias(mask_tokens: torch.Tensor) -> torch.Tensor:
    return torch.log(mask_tokens.clamp(min=1e-4, max=1.0))


class MaskBiasedResamplerBlock(nn.Module):
    def __init__(self, dim: int, token_count: int, num_heads: int = 4) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * dim, dim),
        )
        self.mask_bias_scale = nn.Parameter(torch.zeros(token_count))
        self.num_heads = num_heads

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        mask_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = F.softplus(self.mask_bias_scale).view(1, -1, 1)
        bias = scale * foreground_log_bias(mask_tokens).unsqueeze(1)
        attention_mask = bias.repeat_interleave(self.num_heads, dim=0).to(
            dtype=latents.dtype
        )
        normalized_context = self.context_norm(context)
        update, weights = self.cross_attention(
            self.query_norm(latents),
            normalized_context,
            normalized_context,
            attn_mask=attention_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        latents = latents + update
        latents = latents + self.feed_forward(latents)
        return latents, weights


class MaskConditionedPerceiverStrong(CommonTokenClassifier):
    uses_connected_components = False
    uses_graph_construction = False
    uses_persistent_homology = False

    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__(dim, token_count)
        self.stem = PatchStem(dim)
        self.queries = nn.Parameter(torch.randn(token_count, dim) * 0.02)
        self.position = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.mask_embedding = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList(
            [MaskBiasedResamplerBlock(dim, token_count) for _ in range(2)]
        )
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def reduce_tokens(
        self, image: torch.Tensor, mask_probability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.stem(image).flatten(2).transpose(1, 2)
        coordinates = self.coordinates.to(context.dtype).unsqueeze(0).expand(
            image.shape[0], -1, -1
        )
        mask_tokens = patch_mask(mask_probability).to(context.dtype)
        context = (
            context
            + self.position(coordinates)
            + self.mask_embedding(mask_tokens.unsqueeze(-1))
        )
        latents = self.queries.unsqueeze(0).expand(image.shape[0], -1, -1)
        weights = None
        for block in self.blocks:
            latents, weights = block(latents, context, mask_tokens)
        if weights is None:
            raise AssertionError("Strong mask-conditioned resampler returned no weights.")
        if latents.shape[1] != self.token_count:
            raise AssertionError("Strong mask-conditioned resampler violated exact K.")
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        centroids = torch.bmm(weights, coordinates)
        foreground_mass = torch.bmm(weights, mask_tokens.unsqueeze(-1)).clamp_min(
            1e-6
        )
        mass = foreground_mass / foreground_mass.sum(dim=1, keepdim=True)
        return latents, torch.cat((centroids, mass), dim=-1)

    def forward(
        self, image: torch.Tensor, mask_probability: torch.Tensor
    ) -> torch.Tensor:
        tokens, metadata = self.reduce_tokens(image, mask_probability)
        return self.classify(tokens, metadata)


class MaskConditionedSlotParamMatched(CommonTokenClassifier):
    uses_connected_components = False
    uses_graph_construction = False
    uses_persistent_homology = False

    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__(dim, token_count)
        self.stem = PatchStem(dim)
        self.slots = nn.Parameter(torch.randn(token_count, dim) * 0.02)
        self.mask_bias_scale = nn.Parameter(torch.zeros(token_count))
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def reduce_tokens(
        self, image: torch.Tensor, mask_probability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.stem(image).flatten(2).transpose(1, 2)
        coordinates = self.coordinates.to(context.dtype).unsqueeze(0).expand(
            image.shape[0], -1, -1
        )
        context = context.clone()
        context[..., :2] = context[..., :2] + coordinates
        mask_tokens = patch_mask(mask_probability).to(context.dtype)
        log_mask = foreground_log_bias(mask_tokens).unsqueeze(1)
        slots = self.slots.unsqueeze(0).expand(image.shape[0], -1, -1)
        weights = None
        for _ in range(2):
            queries = F.normalize(slots, dim=-1)
            keys = F.normalize(context, dim=-1)
            logits = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(
                context.shape[-1]
            )
            logits = logits + F.softplus(self.mask_bias_scale).view(
                1, -1, 1
            ) * log_mask
            weights = torch.softmax(logits, dim=-1)
            pooled = torch.bmm(weights, context)
            slots = 0.5 * slots + 0.5 * pooled
        if weights is None or slots.shape[1] != self.token_count:
            raise AssertionError("Matched mask-conditioned slot model violated exact K.")
        centroids = torch.bmm(weights, coordinates)
        foreground_mass = torch.bmm(weights, mask_tokens.unsqueeze(-1)).clamp_min(
            1e-6
        )
        mass = foreground_mass / foreground_mass.sum(dim=1, keepdim=True)
        return slots, torch.cat((centroids, mass), dim=-1)

    def forward(
        self, image: torch.Tensor, mask_probability: torch.Tensor
    ) -> torch.Tensor:
        tokens, metadata = self.reduce_tokens(image, mask_probability)
        return self.classify(tokens, metadata)


def make_model(name: str, dim: int, token_count: int) -> nn.Module:
    if name == "mask_conditioned_perceiver_strong":
        return MaskConditionedPerceiverStrong(dim, token_count)
    if name == "mask_conditioned_slot_param_matched":
        return MaskConditionedSlotParamMatched(dim, token_count)
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
    dataset: ReducerDataset,
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


def train_one(
    name: str,
    args: argparse.Namespace,
    datasets: dict[str, ReducerDataset],
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
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
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if name == "mask_conditioned_slot_param_matched":
        relative_error = abs(parameter_count - MASKTOPO_PARAMETER_ANCHOR) / float(
            MASKTOPO_PARAMETER_ANCHOR
        )
        if relative_error > MATCH_TOLERANCE:
            raise RuntimeError(
                f"Parameter-matched model error {relative_error:.4%} exceeds 1%."
            )
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
            "stage": "equal_supervision_mask_aware_non_topological",
            "dataset": "deepcrack",
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
            f"MASK_AWARE dataset=deepcrack K={args.token_count} model={name} "
            f"seed={args.seed} epoch={epoch}/{args.epochs} "
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
    latency = benchmark_model(
        model,
        datasets["test"],
        device,
        args.latency_warmup,
        args.latency_repeats,
    )
    return (
        {
            "model": name,
            "seed": args.seed,
            "parameters": parameter_count,
            "parameter_ratio_to_masktopo": parameter_count
            / float(MASKTOPO_PARAMETER_ANCHOR),
            "best_dev_balanced_accuracy": best,
            "test_metrics": test_metrics,
            "training_seconds": time.time() - started,
            "model_latency": latency,
            "uses_identical_predicted_mask": True,
            "uses_ground_truth_mask_or_topology": False,
            "uses_connected_components": False,
            "uses_graph_construction": False,
            "uses_persistent_homology": False,
            "exact_output_tokens": args.token_count,
        },
        history,
        prediction,
        probability,
        truth,
    )


def run_self_test() -> None:
    set_seed(20260803)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.rand(2, 3, 64, 64, device=device)
    mask = torch.rand(2, 1, 64, 64, device=device)
    rows: list[dict[str, Any]] = []
    for name in MODELS:
        model = make_model(name, 64, 16).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        captured: dict[str, bool] = {"identity": False}

        def identity_hook(
            _module: nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> None:
            adjacency = inputs[2]
            reachability = inputs[3]
            identity = torch.eye(16, device=adjacency.device, dtype=adjacency.dtype)
            identity = identity.unsqueeze(0).expand(adjacency.shape[0], -1, -1)
            if not torch.equal(adjacency, identity) or not torch.equal(
                reachability, identity
            ):
                raise AssertionError("Non-identity classifier graph reached TB032 head.")
            captured["identity"] = True

        handle = model.head.register_forward_pre_hook(identity_hook)
        tokens, metadata = model.reduce_tokens(image, mask)
        if tokens.shape != (2, 16, 64) or metadata.shape != (2, 16, 3):
            raise AssertionError(
                f"Unexpected exact-K shapes for {name}: {tokens.shape}, {metadata.shape}"
            )
        logits = model(image, mask)
        handle.remove()
        if logits.shape != (2,) or not captured["identity"]:
            raise AssertionError(f"Classifier self-test failed for {name}.")
        logits.square().mean().backward()
        missing = [
            parameter_name
            for parameter_name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing:
            raise AssertionError(f"Active parameters without gradients for {name}: {missing}")
        source = inspect.getsource(type(model)).lower()
        prohibited = (
            "cv2.connectedcomponents",
            "measure.label(",
            "gudhi.",
            "cubicalcomplex(",
            "persistence_pairs(",
            "skeletonize(",
        )
        hits = [term for term in prohibited if term in source]
        if hits:
            raise AssertionError(f"Prohibited topology API in {name}: {hits}")
        rows.append(
            {
                "model": name,
                "parameters": parameter_count,
                "ratio_to_masktopo": parameter_count
                / float(MASKTOPO_PARAMETER_ANCHOR),
                "token_shape": list(tokens.shape),
                "metadata_shape": list(metadata.shape),
                "identity_head_graph_verified": captured["identity"],
                "prohibited_topology_api_hits": hits,
            }
        )
    matched = next(
        row for row in rows if row["model"] == "mask_conditioned_slot_param_matched"
    )
    if abs(matched["ratio_to_masktopo"] - 1.0) > MATCH_TOLERANCE:
        raise AssertionError("Parameter-matched self-test failed the ±1% gate.")
    print(
        json.dumps(
            {
                "status": "MASK_AWARE_EQUAL_SUPERVISION_SELF_TEST_PASS",
                "masktopo_parameter_anchor": MASKTOPO_PARAMETER_ANCHOR,
                "models": rows,
                "cuda": torch.cuda.is_available(),
                "device": str(device),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.output is None or args.mask_probabilities is None:
        raise ValueError("--output and --mask-probabilities are required.")
    if args.crop_size != 64 or args.output_size != 64:
        raise ValueError("The frozen DeepCrack protocol requires 64x64 inputs.")
    if args.token_count != 16 or args.dim != 64:
        raise ValueError("TB032 is frozen to K=16 and dim=64.")
    if args.epochs != 22 or args.batch_size != 32:
        raise ValueError("Training budget changed from 22 epochs / batch 32.")
    if args.data_seed != FROZEN_DATA_SEED:
        raise ValueError(
            f"TB032 requires frozen data seed {FROZEN_DATA_SEED}; "
            f"received {args.data_seed}."
        )
    if args.lr != 1e-3 or args.weight_decay != 1e-4:
        raise ValueError("Optimizer settings changed from the frozen protocol.")
    if list(args.models) != list(MODELS):
        raise ValueError("Both frozen TB032 models must be run and reported.")

    args.dataset = "deepcrack"
    arrays = load_arrays(args)
    mask_path = args.mask_probabilities.resolve()
    with np.load(mask_path, allow_pickle=False) as archive:
        storage_dtypes = {split: str(archive[split].dtype) for split in SPLITS}
        if any(archive[split].dtype != np.float16 for split in SPLITS):
            raise RuntimeError(f"Mask archives are not all stored float16: {storage_dtypes}")
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
    datasets = {
        split: ReducerDataset(
            arrays[split], probabilities[split], augment=split == "train"
        )
        for split in SPLITS
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    probabilities_out: dict[str, np.ndarray] = {}
    truth_reference: np.ndarray | None = None
    for name in MODELS:
        print(
            f"MASK_AWARE_START dataset=deepcrack K=16 model={name} seed={args.seed}",
            flush=True,
        )
        summary, history, prediction, probability, truth = train_one(
            name, args, datasets, output, device
        )
        if truth_reference is None:
            truth_reference = truth
        elif not np.array_equal(truth_reference, truth):
            raise RuntimeError("Held-out truth changed between TB032 models.")
        summaries[name] = summary
        histories.extend(history)
        predictions[name] = prediction
        probabilities_out[f"{name}_probability"] = probability
        print(
            f"MASK_AWARE_DONE dataset=deepcrack K=16 model={name} "
            f"seed={args.seed} test_bal={summary['test_metrics']['balanced_accuracy']:.4f}",
            flush=True,
        )
    if truth_reference is None:
        raise RuntimeError("No TB032 model ran.")
    write_csv(output / "history.csv", histories)
    prediction_payload: dict[str, np.ndarray] = {
        "truth": truth_reference,
        **predictions,
        **probabilities_out,
    }
    if "source_id" in arrays["test"]:
        prediction_payload["source_id"] = arrays["test"]["source_id"]
    np.savez_compressed(output / "heldout_predictions.npz", **prediction_payload)

    protocol_path = Path(__file__).parent / "MASK_AWARE_EQUAL_SUPERVISION_PROTOCOL.md"
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "confirmatory_status": "retrospective_collision",
        "dataset": "deepcrack",
        "task": "endpoint_connectivity",
        "optimization_seed": args.seed,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "sample_counts": {split: len(datasets[split]) for split in SPLITS},
        "fine_token_count": 256,
        "reduced_token_count": args.token_count,
        "common_token_dimension": args.dim,
        "models": list(MODELS),
        "model_results": summaries,
        "masktopo_parameter_anchor": MASKTOPO_PARAMETER_ANCHOR,
        "parameter_match_tolerance": MATCH_TOLERANCE,
        "mask_probabilities": {
            "path": str(mask_path),
            "sha256": file_sha256(mask_path),
            "storage_dtype_by_split": storage_dtypes,
            "loaded_compute_dtype": "float32",
        },
        "equal_supervision": {
            "same_saved_predicted_mask_as_masktopo": True,
            "same_dense_mask_predictor_product": True,
            "ground_truth_mask_used_at_inference": False,
        },
        "non_topological_firewall": {
            "connected_components": False,
            "component_assignment": False,
            "adjacency_graph_construction": False,
            "reachability_graph_construction": False,
            "persistent_homology": False,
            "skeletonization": False,
            "classifier_graph_inputs": "identity_only_common_head",
        },
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": file_sha256(protocol_path),
        "code_sha256": file_sha256(Path(__file__)),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            ),
        },
        "test_was_previously_observed": True,
        "selection_uses_development_only": True,
        "massachusetts_test_reopened": False,
        "p1_to_p5_result_used": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gudhi
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crackforest_real_gate import balanced_metrics, write_csv
from ph_token_baselines import (
    DESCRIPTOR_DIM,
    SPLITS,
    PHDescriptorDataset,
    aggregate_descriptor_diagnostics,
    benchmark_ph_preprocessing,
    build_or_load_ph_cache,
    load_arrays,
)
from strong_reducer_baselines import CommonTokenClassifier, ResamplerBlock
from topobridge_mvp import PatchStem, set_seed
from topocoarsen_oracle import fine_coordinates


EXPERIMENT_ID = "TB-B-260803-031"
MODEL_NAME = "phgnet_author_adaptation"
MODEL_LABEL = "PHG-Net author-code-based adaptation"
EXPECTED_AUTHOR_COMMIT = "6daa5f7dba556e9882611eb4e2e1c89a67f0d2c5"
EXPECTED_AUTHOR_MODULE_SHA256 = (
    "88535E17B99B6DE7BAD8CA4202698CBDBA1DD056CA1F0ADB67BD005729312B20"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepCrack K16 PHG-Net author-code-based adaptation."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mask-probabilities", type=Path)
    parser.add_argument("--ph-cache", type=Path)
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
    parser.add_argument("--max-ph-tokens", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument("--ph-latency-repeats", type=int, default=3)
    parser.add_argument(
        "--author-repo",
        type=Path,
        default=Path("third_party/TopoClassification"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True, encoding="utf-8"
    ).strip()


def verify_author_source(author_repo: Path) -> dict[str, Any]:
    repo = author_repo.resolve()
    module_path = repo / "models/pointnet/pointnet_utils.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Author PointNet module not found: {module_path}")
    commit = git_output(repo, "rev-parse", "HEAD")
    if commit != EXPECTED_AUTHOR_COMMIT:
        raise RuntimeError(f"Author commit changed: {commit}")
    status = git_output(repo, "status", "--porcelain")
    if status:
        raise RuntimeError(f"Author clone is not clean:\n{status}")
    module_hash = file_sha256(module_path).upper()
    if module_hash != EXPECTED_AUTHOR_MODULE_SHA256:
        raise RuntimeError(f"Author module hash changed: {module_hash}")
    return {
        "repository": "https://github.com/yaoppeng/TopoClassification",
        "local_path": str(repo),
        "commit": commit,
        "branch": git_output(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "module_path": str(module_path),
        "module_sha256": module_hash,
        "license_file_present": any(
            (repo / name).is_file()
            for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
        ),
        "reuse_scope": "PointNetEncoder class; local research adaptation only",
    }


def load_author_encoder_class(author_repo: Path) -> type[nn.Module]:
    module_path = author_repo.resolve() / "models/pointnet/pointnet_utils.py"
    spec = importlib.util.spec_from_file_location(
        "topoclassification_author_pointnet_utils", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load author module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PointNetEncoder


class PHGNetAuthorAdaptation(CommonTokenClassifier):
    """Fixed-K adapter around the PHG-Net authors' PointNet-like PD encoder."""

    def __init__(
        self,
        dim: int,
        token_count: int,
        pointnet_encoder_class: type[nn.Module],
    ) -> None:
        super().__init__(dim, token_count)
        self.author_encoder = pointnet_encoder_class(
            global_feat=False,
            feature_transform=False,
            channel=4,
        )
        self.author_projection = nn.Sequential(
            nn.Linear(2112, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.stem = PatchStem(dim)
        self.topology_gate = nn.Linear(2048, dim)
        self.position = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList([ResamplerBlock(dim) for _ in range(2)])
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    @staticmethod
    def author_point_cloud(descriptor: torch.Tensor) -> torch.Tensor:
        valid = descriptor[..., 6:7]
        points = torch.cat(
            (
                descriptor[..., 0:2],
                descriptor[..., 3:5],
            ),
            dim=-1,
        ) * valid
        centroid = points.mean(dim=1, keepdim=True)
        centered = points - centroid
        radius = torch.linalg.vector_norm(centered, dim=-1).amax(
            dim=1, keepdim=True
        )
        return centered / radius.unsqueeze(-1).clamp_min(1e-6)

    def forward(
        self,
        image: torch.Tensor,
        mask_probability: torch.Tensor,
        descriptor: torch.Tensor,
    ) -> torch.Tensor:
        del mask_probability
        point_cloud = self.author_point_cloud(descriptor).transpose(1, 2)
        author_features, transform, feature_transform = self.author_encoder(
            point_cloud
        )
        if transform is not None or feature_transform is not None:
            raise AssertionError("Frozen author encoder unexpectedly enabled transforms.")
        if author_features.shape != (
            image.shape[0],
            2112,
            self.token_count,
        ):
            raise AssertionError(
                f"Author encoder returned {tuple(author_features.shape)}, expected "
                f"({image.shape[0]}, 2112, {self.token_count})."
            )
        global_topology = author_features[:, :2048, 0]
        tokens = self.author_projection(author_features.transpose(1, 2))
        if tokens.shape[1] != self.token_count:
            raise AssertionError("Author adapter did not emit exact K tokens.")

        context = self.stem(image).flatten(2).transpose(1, 2)
        coordinates = self.coordinates.to(context.dtype).unsqueeze(0).expand(
            image.shape[0], -1, -1
        )
        gate = torch.sigmoid(self.topology_gate(global_topology)).unsqueeze(1)
        context = context + context * gate
        context = context + self.position(coordinates)

        attention_weights = None
        for block in self.blocks:
            tokens, attention_weights = block(tokens, context)
        if attention_weights is None:
            raise AssertionError("PHG-Net adapter returned no attention weights.")
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        centroids = torch.bmm(attention_weights, coordinates)
        persistence = descriptor[..., 2:3] * descriptor[..., 6:7]
        mass = persistence / persistence.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return self.classify(tokens, torch.cat((centroids, mass), dim=-1))


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


def run_self_test(author_repo: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen author module constructs CUDA parameters.")
    provenance = verify_author_source(author_repo)
    encoder_class = load_author_encoder_class(author_repo)
    set_seed(20260803)
    device = torch.device("cuda")
    model = PHGNetAuthorAdaptation(32, 16, encoder_class).to(device)
    image = torch.rand(2, 3, 64, 64, device=device)
    mask = torch.rand(2, 1, 64, 64, device=device)
    descriptor = torch.zeros(2, 16, DESCRIPTOR_DIM, device=device)
    descriptor[..., 0] = torch.rand(2, 16, device=device) * 0.7
    descriptor[..., 1] = descriptor[..., 0] + torch.rand(2, 16, device=device) * 0.3
    descriptor[..., 2] = descriptor[..., 1] - descriptor[..., 0]
    descriptor[..., 3] = 1.0
    descriptor[:, -4:, 3] = 0.0
    descriptor[:, -4:, 4] = 1.0
    descriptor[..., 6] = 1.0
    descriptor[..., 7:11] = torch.rand(2, 16, 4, device=device) * 2.0 - 1.0
    logits = model(image, mask, descriptor)
    if logits.shape != (2,):
        raise AssertionError(f"Unexpected self-test logits: {tuple(logits.shape)}")
    logits.square().mean().backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is None
        and not name.startswith("author_encoder.stn.")
    ]
    if missing:
        raise AssertionError(f"Active parameters without gradients: {missing}")
    print(
        json.dumps(
            {
                "status": "PHGNET_AUTHOR_ADAPTATION_SELF_TEST_PASS",
                "author": provenance,
                "input_shape": [2, 4, 16],
                "output_token_shape": [2, 16, 32],
                "logit_shape": list(logits.shape),
                "author_stn_is_present_but_disabled_in_frozen_forward": True,
            },
            indent=2,
        ),
        flush=True,
    )


def train_one(
    args: argparse.Namespace,
    datasets: dict[str, PHDescriptorDataset],
    output: Path,
    device: torch.device,
    encoder_class: type[nn.Module],
    ph_latency: dict[str, dict[str, float | int]],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    set_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
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
    model = PHGNetAuthorAdaptation(
        args.dim, args.token_count, encoder_class
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    checkpoint_path = output / f"{MODEL_NAME}_seed{args.seed}.pt"
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
                device_type="cuda", dtype=torch.float16, enabled=True
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
            "stage": "phgnet_author_code_based_adaptation",
            "dataset": "deepcrack",
            "seed": args.seed,
            "model": MODEL_NAME,
            "token_count": args.token_count,
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"PHG_AUTHOR dataset=deepcrack K={args.token_count} "
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
                    "model_name": MODEL_NAME,
                    "model_label": MODEL_LABEL,
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
            "model": MODEL_NAME,
            "label": MODEL_LABEL,
            "seed": args.seed,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_dev_balanced_accuracy": best,
            "test_metrics": test_metrics,
            "training_seconds": time.time() - started,
            "model_latency": model_latency,
            "end_to_end_latency": end_to_end,
            "uses_predicted_mask": True,
            "uses_ground_truth_mask_or_topology": False,
            "input_pd": "cubical_H0_H1_fixed_K_from_saved_float16_mask_probability",
            "exact_output_tokens": args.token_count,
        },
        history,
        prediction,
        probability,
        truth,
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args.author_repo)
        return
    if args.output is None or args.mask_probabilities is None or args.ph_cache is None:
        raise ValueError("--output, --mask-probabilities and --ph-cache are required.")
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen PHG-Net author module requires CUDA here.")
    if args.crop_size != 64 or args.output_size != 64:
        raise ValueError("The frozen DeepCrack protocol requires 64x64 inputs.")
    if args.token_count != 16:
        raise ValueError("This protocol is frozen to DeepCrack K=16.")
    if args.max_ph_tokens != 64:
        raise ValueError("This protocol reuses the frozen K64 descriptor cache.")
    if args.epochs != 22 or args.batch_size != 32:
        raise ValueError("Training budget changed from the frozen 22-epoch/batch-32 protocol.")
    if args.lr != 1e-3 or args.weight_decay != 1e-4:
        raise ValueError("Optimizer settings changed from the frozen protocol.")

    author_provenance = verify_author_source(args.author_repo)
    encoder_class = load_author_encoder_class(args.author_repo)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.dataset = "deepcrack"
    arrays, array_sources = load_arrays(args)

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

    ph_cache = args.ph_cache.resolve()
    descriptors, ph_metadata = build_or_load_ph_cache(
        probabilities, mask_path, ph_cache, args.max_ph_tokens
    )
    if "cache_sha256" not in ph_metadata:
        ph_metadata["cache_sha256"] = file_sha256(ph_cache)
    diagnostics = {
        split: aggregate_descriptor_diagnostics(
            descriptors[split][:, : args.token_count]
        )
        for split in SPLITS
    }
    if any(float(diagnostics[split]["padding_rate"]) != 0.0 for split in SPLITS):
        raise RuntimeError("Frozen DeepCrack K16 unexpectedly contains PD padding.")
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
    device = torch.device("cuda")
    print(
        f"PHG_AUTHOR_START dataset=deepcrack K=16 seed={args.seed} "
        f"author_commit={author_provenance['commit']}",
        flush=True,
    )
    summary, history, prediction, probability, truth = train_one(
        args, datasets, output, device, encoder_class, ph_latency
    )
    write_csv(output / "history.csv", history)
    prediction_payload: dict[str, np.ndarray] = {
        "truth": truth,
        MODEL_NAME: prediction,
        f"{MODEL_NAME}_probability": probability,
    }
    if "source_id" in arrays["test"]:
        prediction_payload["source_id"] = arrays["test"]["source_id"]
    np.savez_compressed(output / "heldout_predictions.npz", **prediction_payload)

    protocol_path = Path(__file__).parent / "PHGNET_AUTHOR_ADAPTATION_PROTOCOL.md"
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "method_label": MODEL_LABEL,
        "original_protocol_reproduction": False,
        "confirmatory_status": "retrospective_collision",
        "dataset": "deepcrack",
        "task": "endpoint_connectivity",
        "optimization_seed": args.seed,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "sample_counts": {split: len(datasets[split]) for split in SPLITS},
        "fine_token_count": 256,
        "reduced_token_count": args.token_count,
        "exact_k_interface_verified": True,
        "model_results": {MODEL_NAME: summary},
        "ph_preprocessing_latency": ph_latency,
        "ph_metadata": ph_metadata,
        "ph_diagnostics_at_k": diagnostics,
        "mask_probabilities": {
            "path": str(mask_path),
            "sha256": file_sha256(mask_path),
            "storage_dtype_by_split": storage_dtypes,
            "loaded_compute_dtype": "float32",
        },
        "array_sources": array_sources,
        "author_source": author_provenance,
        "adaptation": {
            "pd_point_channels": ["birth", "death", "H0", "H1"],
            "author_normalization": "center_then_divide_by_max_point_radius",
            "author_encoder_call": (
                "PointNetEncoder(global_feat=False, feature_transform=False, channel=4)"
            ),
            "author_encoder_output_per_point": 2112,
            "common_token_dimension": args.dim,
            "visual_guidance": "author_repository_style_residual_channel_gate",
            "visual_interaction": "same_two_ResamplerBlocks_as_existing_PH_guided",
            "classification_head": "unchanged_CommonTokenClassifier_CoarseGraphHead",
        },
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": file_sha256(protocol_path),
        "code_sha256": file_sha256(Path(__file__)),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gudhi": gudhi.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0),
        },
        "uses_ground_truth_mask_or_topology_at_inference": False,
        "test_was_previously_observed": True,
        "selection_uses_development_only": True,
        "massachusetts_test_reopened": False,
        "p1_to_p5_result_used": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"PHG_AUTHOR_DONE dataset=deepcrack K=16 seed={args.seed} "
        f"test_bal={summary['test_metrics']['balanced_accuracy']:.4f}",
        flush=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


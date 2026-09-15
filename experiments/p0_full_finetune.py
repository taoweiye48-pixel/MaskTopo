from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16

from fives_g2tm_collision import grid_edges, maximum_spanning_forest_assignments
from standard_vit_accuracy_pilot import (
    IMAGE_SIZE,
    MODES as PILOT_MODES,
    PRETRAINED_ID,
    FivesViTDataset,
    classification_metrics,
    file_sha256,
    load_inputs,
    load_pretrained_state,
    make_loader,
    move_batch,
    rectangular_assignment,
    set_seed,
)
from topology_construction_benchmark import coarse_graph8_batch


MODES = (
    "full",
    "grid",
    "g2tm_fixedk",
    "assignment_only",
    "local_only",
    "reachability_only",
    "hybrid",
    "shuffled_global",
    "shuffled_within_class",
    "random_connected",
)
GRAPH_MODES = {
    "local_only",
    "reachability_only",
    "hybrid",
    "shuffled_global",
    "shuffled_within_class",
    "random_connected",
}
ARTIFACT_MODES = GRAPH_MODES | {"assignment_only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P0 full-fine-tuning, mechanism and topology controls."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("fives_cache_v2"))
    parser.add_argument(
        "--mask-run-dir", type=Path, default=Path("results_fives_seed20260810")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--dev-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--insert-layer", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--initialization", choices=("imagenet", "random"), default="imagenet"
    )
    parser.add_argument("--models", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("A derangement requires at least two samples.")
    order = rng.permutation(size)
    mapping = np.empty(size, dtype=np.int64)
    mapping[order] = np.roll(order, 1)
    if np.any(mapping == np.arange(size)):
        raise AssertionError("Derangement contains a fixed point.")
    return mapping


def within_class_derangement(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    mapping = np.empty(len(labels), dtype=np.int64)
    for value in np.unique(labels):
        indices = np.flatnonzero(labels == value)
        local = derangement(len(indices), rng)
        mapping[indices] = indices[local]
    if np.any(mapping == np.arange(len(labels))):
        raise AssertionError("Within-class derangement contains a fixed point.")
    if not np.array_equal(labels, labels[mapping]):
        raise AssertionError("Within-class derangement changed class identity.")
    return mapping


def random_connected_artifacts(
    sample_count: int, token_count: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    coordinates = np.indices((16, 16)).reshape(2, -1).T.astype(np.int16)
    assignments = np.empty((sample_count, 16, 16), dtype=np.int16)
    for index in range(sample_count):
        seed_indices = np.sort(rng.choice(256, token_count, replace=False))
        seeds = coordinates[seed_indices]
        distances = np.abs(
            coordinates[:, None, :].astype(np.int32)
            - seeds[None, :, :].astype(np.int32)
        ).sum(axis=2)
        assignment = np.argmin(distances, axis=1).astype(np.int16).reshape(16, 16)
        if np.unique(assignment).size != token_count:
            raise AssertionError("Random connected partition lost a seed region.")
        assignments[index] = assignment
    full_background = np.zeros_like(assignments, dtype=np.int16)
    adjacency, reachability = coarse_graph8_batch(
        assignments, full_background, token_count
    )
    return {
        "assignment": assignments,
        "adjacency": adjacency.astype(np.uint8),
        "reachability": reachability.astype(np.uint8),
    }


def artifact_variant(
    arrays: dict[str, dict[str, np.ndarray]], mode: str, seed: int, token_count: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    offsets = {"train": 401, "dev": 402, "test": 403}
    for split, original in arrays.items():
        current = dict(original)
        mapping: np.ndarray | None = None
        if mode == "shuffled_global":
            mapping = derangement(
                len(original["labels"]), np.random.default_rng(seed + offsets[split])
            )
        elif mode == "shuffled_within_class":
            mapping = within_class_derangement(
                original["labels"], np.random.default_rng(seed + offsets[split])
            )
        if mapping is not None:
            for key in ("assignment", "adjacency", "reachability"):
                current[key] = original[key][mapping]
        elif mode == "random_connected":
            random_artifacts = random_connected_artifacts(
                len(original["labels"]), token_count, seed + offsets[split]
            )
            current.update(random_artifacts)
        output[split] = current
        digest = hashlib.sha256()
        for key in ("assignment", "adjacency", "reachability"):
            digest.update(np.ascontiguousarray(current[key]).tobytes())
        diagnostics[split] = {
            "artifact_sha256": digest.hexdigest(),
            "mean_adjacency_density": float(current["adjacency"].mean()),
            "mean_reachability_density": float(current["reachability"].mean()),
            "mapping_sha256": (
                hashlib.sha256(mapping.tobytes()).hexdigest()
                if mapping is not None
                else None
            ),
            "fixed_points": (
                int(np.sum(mapping == np.arange(len(mapping))))
                if mapping is not None
                else None
            ),
        }
    return output, diagnostics


class ControlledGraphMixLayer(nn.Module):
    def __init__(self, dim: int, use_local: bool, use_reachability: bool) -> None:
        super().__init__()
        self.use_local = use_local
        self.use_reachability = use_reachability
        self.local_message = nn.Linear(dim, dim)
        self.component_message = nn.Linear(dim, dim)
        self.local_gate = nn.Parameter(torch.zeros(()))
        self.component_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        tokens: torch.Tensor,
        adjacency: torch.Tensor,
        reachability: torch.Tensor,
    ) -> torch.Tensor:
        output = tokens
        if self.use_local:
            local_weights = adjacency / adjacency.sum(
                dim=-1, keepdim=True
            ).clamp_min(1)
            local = torch.bmm(local_weights, tokens)
            output = output + torch.tanh(self.local_gate) * self.local_message(local)
        if self.use_reachability:
            global_weights = reachability / reachability.sum(
                dim=-1, keepdim=True
            ).clamp_min(1)
            component = torch.bmm(global_weights, tokens)
            output = output + torch.tanh(
                self.component_gate
            ) * self.component_message(component)
        return output


class P0ViT(nn.Module):
    def __init__(
        self,
        mode: str,
        token_count: int,
        insert_layer: int,
        pretrained_state: OrderedDict[str, torch.Tensor] | None,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.token_count = token_count
        self.insert_layer = insert_layer
        self.vit = vit_b_16(weights=None, image_size=IMAGE_SIZE)
        if pretrained_state is not None:
            self.vit.load_state_dict(pretrained_state)
        self.classifier = nn.Linear(768, 1)
        if mode in GRAPH_MODES:
            self.graph = ControlledGraphMixLayer(
                768,
                use_local=mode != "reachability_only",
                use_reachability=mode != "local_only",
            )
        else:
            self.graph = None
        self.register_buffer(
            "grid_assignment",
            torch.from_numpy(rectangular_assignment(token_count).reshape(-1)),
            persistent=False,
        )
        source, target = grid_edges()
        self.register_buffer("edge_source", source, persistent=False)
        self.register_buffer("edge_target", target, persistent=False)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        # Full fine tuning: train patch embedding, class/position embeddings,
        # every transformer block and final LayerNorm. The original ImageNet
        # classification head is unused and remains frozen.
        for parameter in self.vit.parameters():
            parameter.requires_grad = True
        for parameter in self.vit.heads.parameters():
            parameter.requires_grad = False

    def pool(self, tokens: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(assignment, num_classes=self.token_count).to(tokens.dtype)
        counts = one_hot.sum(dim=1).clamp_min(1.0)
        return torch.bmm(one_hot.transpose(1, 2), tokens) / counts.unsqueeze(-1)

    def forward(
        self,
        images: torch.Tensor,
        assignment: torch.Tensor,
        adjacency: torch.Tensor,
        reachability: torch.Tensor,
    ) -> torch.Tensor:
        images = F.interpolate(
            images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )
        images = (images - self.image_mean) / self.image_std
        x = self.vit._process_input(images)
        batch_size = x.shape[0]
        cls = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.vit.encoder.dropout(x + self.vit.encoder.pos_embedding)
        for layer_index, block in enumerate(self.vit.encoder.layers):
            x = block(x)
            if self.mode != "full" and layer_index + 1 == self.insert_layer:
                cls, patch_tokens = x[:, :1], x[:, 1:]
                if self.mode == "grid":
                    current_assignment = self.grid_assignment.unsqueeze(0).expand(
                        batch_size, -1
                    )
                elif self.mode == "g2tm_fixedk":
                    current_assignment = maximum_spanning_forest_assignments(
                        patch_tokens,
                        self.token_count,
                        self.edge_source,
                        self.edge_target,
                    )
                elif self.mode in ARTIFACT_MODES:
                    current_assignment = assignment.reshape(batch_size, -1)
                else:
                    raise ValueError(self.mode)
                patch_tokens = self.pool(patch_tokens, current_assignment)
                if self.graph is not None:
                    patch_tokens = self.graph(
                        patch_tokens,
                        adjacency.to(patch_tokens.dtype),
                        reachability.to(patch_tokens.dtype),
                    )
                x = torch.cat((cls, patch_tokens), dim=1)
        cls_embedding = self.vit.encoder.ln(x)[:, 0]
        return self.classifier(cls_embedding).squeeze(-1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    source_ids: list[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(
                batch["image"],
                batch["assignment"],
                batch["adjacency"],
                batch["reachability"],
            )
            loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
        losses.append(float(loss.item()) * len(logits))
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        truths.append(batch["label"].long().cpu().numpy())
        source_ids.extend(batch["source_id"])
    probability = np.concatenate(probabilities)
    truth = np.concatenate(truths)
    prediction = (probability >= 0.5).astype(np.int64)
    metrics = classification_metrics(truth, prediction)
    metrics["loss"] = float(sum(losses) / len(truth))
    return metrics, {
        "truth": truth,
        "probability": probability,
        "prediction": prediction,
        "source_id": np.asarray(source_ids),
    }


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def train_model(
    mode: str,
    args: argparse.Namespace,
    arrays: dict[str, dict[str, np.ndarray]],
    pretrained_state: OrderedDict[str, torch.Tensor] | None,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    set_seed(args.seed)
    model = P0ViT(
        mode, args.token_count, args.insert_layer, pretrained_state
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    backbone_parameters = [p for p in model.vit.parameters() if p.requires_grad]
    new_parameters = list(model.classifier.parameters())
    if model.graph is not None:
        new_parameters.extend(model.graph.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": new_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(
        FivesViTDataset(arrays["train"], augment=True),
        args.batch_size,
        True,
        args.seed + 100,
        args.num_workers,
    )
    dev_loader = make_loader(
        FivesViTDataset(arrays["dev"], augment=False),
        args.eval_batch_size,
        False,
        args.seed + 200,
        args.num_workers,
    )
    test_loader = make_loader(
        FivesViTDataset(arrays["test"], augment=False),
        args.eval_batch_size,
        False,
        args.seed + 300,
        args.num_workers,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = max(1, args.epochs * updates_per_epoch)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_dev = -math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    peak_memory_mb = 0.0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        sample_count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    batch["image"],
                    batch["assignment"],
                    batch["adjacency"],
                    batch["reachability"],
                )
                raw_loss = F.binary_cross_entropy_with_logits(
                    logits, batch["label"]
                )
                loss = raw_loss / args.gradient_accumulation
            if not bool(torch.isfinite(raw_loss)):
                raise FloatingPointError(f"Non-finite loss in {mode} epoch {epoch}.")
            scaler.scale(loss).backward()
            update = (
                batch_index % args.gradient_accumulation == 0
                or batch_index == len(train_loader)
            )
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(raw_loss.item()) * len(logits)
            sample_count += len(logits)
        dev_metrics, _ = evaluate(model, dev_loader, device)
        if device.type == "cuda":
            peak_memory_mb = max(
                peak_memory_mb, torch.cuda.max_memory_allocated() / 1024**2
            )
        record: dict[str, Any] = {
            "model": mode,
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        print(
            f"EPOCH model={mode} epoch={epoch}/{args.epochs} "
            f"train_loss={record['train_loss']:.5f} "
            f"dev_bacc={dev_metrics['balanced_accuracy']:.5f} "
            f"peak_mb={peak_memory_mb:.0f}",
            flush=True,
        )
        if dev_metrics["balanced_accuracy"] > best_dev:
            best_dev = dev_metrics["balanced_accuracy"]
            best_epoch = epoch
            best_state = trainable_state(model)
    training_seconds = time.perf_counter() - start
    if best_state is None:
        raise AssertionError("No best checkpoint was selected.")
    missing, unexpected = model.load_state_dict(best_state, strict=False)
    if unexpected:
        raise AssertionError(f"Unexpected checkpoint keys: {unexpected}")
    if set(missing) != set(model.state_dict()) - set(best_state):
        raise AssertionError("Checkpoint reload mismatch.")
    test_metrics, predictions = evaluate(model, test_loader, device)
    checkpoint_path: Path | None = None
    checkpoint_hash: str | None = None
    if not args.no_checkpoints:
        checkpoint_path = args.output / f"{mode}_trainable_seed{args.seed}.pt"
        torch.save(
            {
                "model": mode,
                "seed": args.seed,
                "pretrained_id": (
                    PRETRAINED_ID if args.initialization == "imagenet" else "random"
                ),
                "best_epoch": best_epoch,
                "trainable_state": best_state,
            },
            checkpoint_path,
        )
        checkpoint_hash = file_sha256(checkpoint_path)
    result = {
        "model": mode,
        "best_epoch": best_epoch,
        "best_dev_balanced_accuracy": best_dev,
        "test": test_metrics,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "training_seconds": training_seconds,
        "peak_cuda_memory_mb": peak_memory_mb,
        "checkpoint": str(checkpoint_path.resolve()) if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_hash,
        "history": history,
    }
    del model, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, predictions


def write_history(path: Path, results: dict[str, dict[str, Any]]) -> None:
    rows = [row for result in results.values() for row in result["history"]]
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_self_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.rand(2, 3, 64, 64, device=device)
    assignment = torch.from_numpy(
        np.stack([rectangular_assignment(args.token_count)] * 2)
    ).to(device)
    eye = torch.eye(args.token_count, device=device).unsqueeze(0).repeat(2, 1, 1)
    for mode in MODES:
        set_seed(args.seed)
        model = P0ViT(mode, args.token_count, args.insert_layer, None).to(device)
        logits = model(images, assignment, eye, eye)
        logits.square().mean().backward()
        nonzero_gradients = sum(
            int(
                p.grad is not None
                and torch.isfinite(p.grad).all()
                and float(p.grad.abs().sum()) > 0
            )
            for p in model.parameters()
            if p.requires_grad
        )
        if logits.shape != (2,) or not torch.isfinite(logits).all():
            raise AssertionError(f"Forward self-test failed for {mode}.")
        if nonzero_gradients == 0:
            raise AssertionError(f"No nonzero gradient for {mode}.")
        print(
            f"SELF_TEST mode={mode} nonzero_gradients={nonzero_gradients}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    labels = np.asarray([0, 0, 1, 1])
    rng = np.random.default_rng(args.seed)
    global_map = derangement(4, rng)
    class_map = within_class_derangement(labels, rng)
    if np.any(global_map == np.arange(4)) or np.any(class_map == np.arange(4)):
        raise AssertionError("Counterfactual self-test failed.")
    random_artifacts = random_connected_artifacts(8, args.token_count, args.seed)
    if not all(
        np.unique(assignment).size == args.token_count
        for assignment in random_artifacts["assignment"]
    ):
        raise AssertionError("Random connected self-test failed.")
    print("SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.token_count != 8 or args.insert_layer != 2:
        raise ValueError("The P0 protocol is frozen at K=8 and insert-layer=2.")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive.")
    if args.self_test:
        run_self_test(args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The formal P0 protocol requires CUDA.")
    args.output.mkdir(parents=True, exist_ok=True)
    arrays, data_provenance = load_inputs(args)
    pretrained_state, weight_provenance = load_pretrained_state(args.initialization)
    results: dict[str, dict[str, Any]] = {}
    predictions_by_mode: dict[str, dict[str, np.ndarray]] = {}
    artifact_diagnostics: dict[str, Any] = {}
    for mode in args.models:
        model_result_path = args.output / f"model_result_{mode}.json"
        model_prediction_path = args.output / f"prediction_{mode}.npz"
        if args.resume and model_result_path.exists() and model_prediction_path.exists():
            results[mode] = json.loads(model_result_path.read_text(encoding="utf-8"))
            with np.load(model_prediction_path) as archive:
                predictions_by_mode[mode] = {
                    key: archive[key] for key in archive.files
                }
            artifact_diagnostics[mode] = results[mode]["artifact_diagnostics"]
            print(f"MODEL_RESUME_SKIP model={mode}", flush=True)
            continue
        current_arrays, diagnostics = artifact_variant(
            arrays, mode, args.seed, args.token_count
        )
        print(f"MODEL_START model={mode} seed={args.seed}", flush=True)
        result, predictions = train_model(
            mode, args, current_arrays, pretrained_state, device
        )
        result["artifact_diagnostics"] = diagnostics
        results[mode] = result
        predictions_by_mode[mode] = predictions
        artifact_diagnostics[mode] = diagnostics
        model_result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(model_prediction_path, **predictions)
        print(
            f"MODEL_DONE model={mode} test_bacc={result['test']['balanced_accuracy']:.5f}",
            flush=True,
        )
    truth_reference: np.ndarray | None = None
    source_reference: np.ndarray | None = None
    combined: dict[str, np.ndarray] = {}
    for mode in args.models:
        predictions = predictions_by_mode[mode]
        if truth_reference is None:
            truth_reference = predictions["truth"]
            source_reference = predictions["source_id"]
            combined["truth"] = truth_reference
            combined["source_id"] = source_reference
        elif not np.array_equal(truth_reference, predictions["truth"]) or not np.array_equal(
            source_reference, predictions["source_id"]
        ):
            raise AssertionError("Test identity changed between P0 conditions.")
        combined[f"{mode}_probability"] = predictions["probability"]
        combined[f"{mode}_prediction"] = predictions["prediction"]
    np.savez_compressed(args.output / "heldout_predictions.npz", **combined)
    write_history(args.output / "history.csv", results)
    is_formal = (
        args.initialization == "imagenet"
        and args.train_size == 2400
        and args.dev_size == 600
        and args.test_size == 1200
        and args.epochs == 8
        and args.batch_size * args.gradient_accumulation == 8
        and set(args.models) == set(MODES)
    )
    final = {
        "experiment_id": f"TB-B-260801-P0-seed{args.seed}",
        "status": "completed",
        "experiment_class": (
            "formal_p0_full_finetuning" if is_formal else "non_paper_smoke_or_ablation"
        ),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "configuration": {
            "dataset": "FIVES matched crops v2",
            "train_size": args.train_size,
            "dev_size": args.dev_size,
            "test_size": args.test_size,
            "epochs": args.epochs,
            "micro_batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "eval_batch_size": args.eval_batch_size,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "token_count": args.token_count,
            "insert_layer": args.insert_layer,
            "seed": args.seed,
            "models": args.models,
            "initialization": args.initialization,
            "backbone_training": "all ViT parameters except unused ImageNet head",
            "test_evaluations_per_model": 1,
            "classification_threshold": 0.5,
        },
        "pretrained_weights": weight_provenance,
        "data_provenance": data_provenance,
        "artifact_diagnostics": artifact_diagnostics,
        "models": results,
        "protocol": str(
            (Path(__file__).parent / "P0_FULL_FINETUNE_PROTOCOL.md").resolve()
        ),
    }
    (args.output / "result.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_PATH {args.output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()


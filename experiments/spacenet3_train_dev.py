from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spacenet3_metrics import sample_metrics
from spacenet3_models import MODEL_NAMES, RoadUNet, build_artifacts, build_ph, make_model, segmentation_loss
from topobridge_mvp import set_seed


GENERIC = ("grid", "tokenlearner", "perceiver_resampler", "tome_style")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen SpaceNet 3 Paris train/dev mechanism gate.")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--mask-epochs", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("images", "masks", "source_ids", "crop_rows", "crop_columns")}
        metadata = json.loads(str(archive["metadata_json"].item()))
    if arrays["images"].shape[1:] != (3, 64, 64) or arrays["masks"].shape[1:] != (64, 64):
        raise RuntimeError(f"Unexpected cache shapes in {path}")
    return arrays, metadata


class MaskDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], augment: bool) -> None:
        self.images = arrays["images"]
        self.masks = arrays["masks"]
        self.augment = augment

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(self.images[index].astype(np.float32))
        mask = torch.from_numpy(self.masks[index].astype(np.float32))
        if self.augment:
            rotations = int(torch.randint(0, 4, ()).item())
            image = torch.rot90(image, rotations, dims=(1, 2))
            mask = torch.rot90(mask, rotations, dims=(0, 1))
            if bool(torch.rand(()) < 0.5):
                image = torch.flip(image, (2,))
                mask = torch.flip(mask, (1,))
            if bool(torch.rand(()) < 0.5):
                image = torch.flip(image, (1,))
                mask = torch.flip(mask, (0,))
            contrast = 0.9 + 0.2 * torch.rand(())
            brightness = -0.05 + 0.1 * torch.rand(())
            image = torch.clamp(brightness + contrast * image, 0.0, 1.0)
        return {"image": image, "mask": mask}


class TokenDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], probabilities: np.ndarray, descriptors: np.ndarray, artifact: dict[str, np.ndarray], augment: bool) -> None:
        self.arrays = arrays
        self.probabilities = probabilities.astype(np.float32)
        self.descriptors = descriptors.astype(np.float32)
        self.artifact = artifact
        self.augment = augment

    def __len__(self) -> int:
        return int(self.arrays["images"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(self.arrays["images"][index].astype(np.float32))
        if self.augment:
            contrast = 0.9 + 0.2 * torch.rand(())
            brightness = -0.05 + 0.1 * torch.rand(())
            image = torch.clamp(brightness + contrast * image, 0.0, 1.0)
        return {
            "image": image,
            "mask": torch.from_numpy(self.arrays["masks"][index].astype(np.float32)),
            "mask_probability": torch.from_numpy(self.probabilities[index]),
            "ph_descriptor": torch.from_numpy(self.descriptors[index]),
            "assignment": torch.from_numpy(self.artifact["assignment"][index]),
            "adjacency": torch.from_numpy(self.artifact["adjacency"][index]),
            "reachability": torch.from_numpy(self.artifact["reachability"][index]),
        }


def move_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if key != "mask"}


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device, mask_model: bool) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        target = batch["mask"].to(device, non_blocking=True)
        logits = model(batch["image"].to(device, non_blocking=True)) if mask_model else model(**move_inputs(batch, device))
        loss, _ = segmentation_loss(logits, target)
        total += float(loss) * target.shape[0]
        count += target.shape[0]
    return total / count


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, mask_model: bool) -> np.ndarray:
    model.eval()
    parts = []
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True)) if mask_model else model(**move_inputs(batch, device))
        parts.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def train_mask(args: argparse.Namespace, train: dict[str, np.ndarray], dev: dict[str, np.ndarray], output: Path, device: torch.device) -> tuple[RoadUNet, list[dict[str, Any]]]:
    set_seed(args.seed)
    train_loader = DataLoader(MaskDataset(train, True), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(MaskDataset(dev, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = RoadUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = output / f"mask_predictor_seed{args.seed}.pt"
    best = float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.mask_epochs + 1):
        model.train()
        sums = {"loss": 0.0, "balanced_bce": 0.0, "dice_loss": 0.0}
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss, parts = segmentation_loss(model(image), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.detach())
            sums["balanced_bce"] += parts["balanced_bce"]
            sums["dice_loss"] += parts["dice_loss"]
            batches += 1
        dev_loss = evaluate_loss(model, dev_loader, device, True)
        row = {"stage": "mask_predictor", "epoch": epoch, "train_loss": sums["loss"] / batches, "train_balanced_bce": sums["balanced_bce"] / batches, "train_dice_loss": sums["dice_loss"] / batches, "dev_loss": dev_loss, "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        print(f"MASK epoch={epoch}/{args.mask_epochs} train_loss={row['train_loss']:.4f} dev_loss={dev_loss:.4f}", flush=True)
        if dev_loss < best:
            best = dev_loss
            torch.save({"model": model.state_dict(), "seed": args.seed, "best_dev_loss": best, "epoch": epoch}, checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model, history


def train_token_model(args: argparse.Namespace, name: str, train_dataset: TokenDataset, dev_dataset: TokenDataset, output: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], list[dict[str, Any]], np.ndarray]:
    set_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = make_model(name, args.dim, args.token_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = output / f"{name}_seed{args.seed}.pt"
    best = float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss, _ = segmentation_loss(model(**move_inputs(batch, device)), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            batches += 1
        dev_loss = evaluate_loss(model, dev_loader, device, False)
        row = {"stage": "token_model", "model": name, "epoch": epoch, "train_loss": total / batches, "dev_loss": dev_loss, "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        print(f"TOKEN model={name} epoch={epoch}/{args.epochs} train_loss={row['train_loss']:.4f} dev_loss={dev_loss:.4f}", flush=True)
        if dev_loss < best:
            best = dev_loss
            torch.save({"model": model.state_dict(), "model_name": name, "seed": args.seed, "best_dev_loss": best, "epoch": epoch, "dim": args.dim, "token_count": args.token_count}, checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    probabilities = predict(model, dev_loader, device, False)
    summary = {"model": name, "seed": args.seed, "parameters": sum(parameter.numel() for parameter in model.parameters()), "best_dev_loss": best, "best_epoch": state["epoch"], "training_seconds": time.perf_counter() - started, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint)}
    return model, summary, history, probabilities


def metric_rows(targets: np.ndarray, probabilities: np.ndarray, source_ids: np.ndarray, crop_rows: np.ndarray, crop_columns: np.ndarray, model: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for index in range(targets.shape[0]):
        metrics = sample_metrics(targets[index], probabilities[index])
        rows.append({"model": model, "index": index, "source_id": str(source_ids[index]), "crop_row": int(crop_rows[index]), "crop_column": int(crop_columns[index]), **metrics})
        if (index + 1) % 200 == 0 or index + 1 == targets.shape[0]:
            print(f"METRICS model={model} progress={index + 1}/{targets.shape[0]}", flush=True)
    metric_names = ("road_f1", "soft_dice", "cldice", "raster_apls_proxy", "gt_to_pred_path_score", "pred_to_gt_path_score", "path_recall", "disconnection_rate")
    source_means: dict[str, dict[str, float]] = {}
    for source_id in sorted(set(str(item) for item in source_ids)):
        selected = [row for row in rows if row["source_id"] == source_id]
        source_means[source_id] = {name: float(np.nanmean([float(row[name]) for row in selected])) if np.any(np.isfinite([float(row[name]) for row in selected])) else float("nan") for name in metric_names}
    summary = {name: float(np.nanmean([value[name] for value in source_means.values()])) for name in metric_names}
    summary["source_count"] = float(len(source_means))
    summary["crop_count"] = float(len(rows))
    summary["valid_apls_crop_count"] = float(sum(np.isfinite(float(row["raster_apls_proxy"])) for row in rows))
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_probability(path: Path, probability: np.ndarray, source_ids: np.ndarray, crop_rows: np.ndarray, crop_columns: np.ndarray) -> np.ndarray:
    if path.exists():
        raise FileExistsError(path)
    stored = probability.astype(np.float16)
    np.savez_compressed(path, probability=stored, source_ids=source_ids, crop_rows=crop_rows, crop_columns=crop_columns)
    with np.load(path, allow_pickle=False) as archive:
        return archive["probability"].astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.self_test:
        set_seed(9)
        image = torch.rand(2, 3, 64, 64)
        target = torch.zeros(2, 64, 64)
        target[:, 30:34, 6:58] = 1
        predictor = RoadUNet(base=8)
        loss, _ = segmentation_loss(predictor(image), target)
        loss.backward()
        assert torch.isfinite(loss)
        print("SPACENET3_TRAIN_DEV_SELF_TEST_PASS")
        return
    if args.seed not in (20260830, 20260831, 20260832) or args.token_count != 16 or args.dim != 64 or args.mask_epochs != 12 or args.epochs != 12:
        raise ValueError("Arguments differ from frozen protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen run requires CUDA")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {output}")
    output.mkdir(parents=True)
    train, train_metadata = load_cache(args.train_cache.resolve())
    dev, dev_metadata = load_cache(args.dev_cache.resolve())
    if train_metadata["manifest_sha256"] != dev_metadata["manifest_sha256"]:
        raise RuntimeError("Train/dev manifest hashes differ")
    code_paths = [Path(__file__).resolve(), (Path(__file__).parent / "spacenet3_models.py").resolve(), (Path(__file__).parent / "spacenet3_metrics.py").resolve()]
    protocol_path = args.protocol.resolve()
    freeze = {
        "experiment_id": "TB-B-260802-027",
        "status": "frozen_before_training",
        "seed": args.seed,
        "models": args.models,
        "hyperparameters": {"mask_epochs": args.mask_epochs, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay, "dim": args.dim, "token_count": args.token_count},
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "code": [{"path": str(path), "sha256": sha256_file(path)} for path in code_paths],
        "train_cache": {"path": str(args.train_cache.resolve()), "sha256": sha256_file(args.train_cache.resolve())},
        "dev_cache": {"path": str(args.dev_cache.resolve()), "sha256": sha256_file(args.dev_cache.resolve())},
        "manifest_sha256": train_metadata["manifest_sha256"],
        "heldout_content_accessed": False,
    }
    freeze_path = output / "run_freeze.json"
    with freeze_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(freeze, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    device = torch.device("cuda")
    started = time.perf_counter()
    mask_model, mask_history = train_mask(args, train, dev, output, device)
    train_mask_loader = DataLoader(MaskDataset(train, False), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    dev_mask_loader = DataLoader(MaskDataset(dev, False), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    raw_train_probability = predict(mask_model, train_mask_loader, device, True)
    raw_dev_probability = predict(mask_model, dev_mask_loader, device, True)
    mask_cache = output / f"mask_probabilities_seed{args.seed}.npz"
    np.savez_compressed(mask_cache, train=raw_train_probability.astype(np.float16), dev=raw_dev_probability.astype(np.float16), metadata_json=np.asarray(json.dumps({"dtype": "float16", "threshold": 0.5, "manifest_sha256": train_metadata["manifest_sha256"]})))
    with np.load(mask_cache, allow_pickle=False) as archive:
        probabilities = {"train": archive["train"].astype(np.float32), "dev": archive["dev"].astype(np.float32)}
    artifacts = {}
    artifact_diagnostics = {}
    descriptors = {}
    ph_diagnostics = {}
    for split in ("train", "dev"):
        artifacts[split], artifact_diagnostics[split] = build_artifacts(probabilities[split], args.token_count, 0.5)
        descriptors[split], ph_diagnostics[split] = build_ph(probabilities[split], args.token_count)
        print(f"ARTIFACTS_COMPLETE split={split}", flush=True)
    artifact_path = output / f"artifacts_seed{args.seed}.npz"
    artifact_arrays = {}
    for split in ("train", "dev"):
        for mode, values in artifacts[split].items():
            for key, value in values.items():
                artifact_arrays[f"{split}_{mode}_{key}"] = value
        artifact_arrays[f"{split}_ph_descriptor"] = descriptors[split]
    np.savez_compressed(artifact_path, **artifact_arrays, metadata_json=np.asarray(json.dumps({"artifact_diagnostics": artifact_diagnostics, "ph_diagnostics": ph_diagnostics, "mask_cache_sha256": sha256_file(mask_cache)}, ensure_ascii=False)))
    mask_dev_stored = save_probability(output / f"dev_predictions_mask_only_seed{args.seed}.npz", probabilities["dev"], dev["source_ids"], dev["crop_rows"], dev["crop_columns"])
    all_rows, mask_metrics = metric_rows(dev["masks"], mask_dev_stored, dev["source_ids"], dev["crop_rows"], dev["crop_columns"], "mask_only")
    model_results = {}
    histories = {"mask_predictor": mask_history}
    for name in args.models:
        artifact_name = name if name in artifacts["train"] else "mask_topo"
        train_dataset = TokenDataset(train, probabilities["train"], descriptors["train"], artifacts["train"][artifact_name], True)
        dev_dataset = TokenDataset(dev, probabilities["dev"], descriptors["dev"], artifacts["dev"][artifact_name], False)
        model, model_summary, history, raw_dev = train_token_model(args, name, train_dataset, dev_dataset, output, device)
        del model
        torch.cuda.empty_cache()
        stored_dev = save_probability(output / f"dev_predictions_{name}_seed{args.seed}.npz", raw_dev, dev["source_ids"], dev["crop_rows"], dev["crop_columns"])
        rows, metrics = metric_rows(dev["masks"], stored_dev, dev["source_ids"], dev["crop_rows"], dev["crop_columns"], name)
        all_rows.extend(rows)
        model_results[name] = {**model_summary, "dev_metrics": metrics, "prediction_sha256": sha256_file(output / f"dev_predictions_{name}_seed{args.seed}.npz")}
        histories[name] = history
        print(f"MODEL_COMPLETE name={name} apls={metrics['raster_apls_proxy']:.4f} cldice={metrics['cldice']:.4f}", flush=True)
    write_csv(output / "dev_crop_metrics.csv", all_rows)
    with (output / "history.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(histories, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    required = set(MODEL_NAMES)
    gate_ready = required.issubset(model_results)
    strongest = max(GENERIC, key=lambda name: model_results[name]["dev_metrics"]["raster_apls_proxy"]) if gate_ready else None
    if gate_ready and strongest is not None:
        topo = model_results["mask_topo"]["dev_metrics"]["raster_apls_proxy"]
        gain = topo - model_results[strongest]["dev_metrics"]["raster_apls_proxy"]
        verdict = "PASS" if gain >= 0.03 and topo > model_results["mask_assignment_identity"]["dev_metrics"]["raster_apls_proxy"] and topo > model_results["shuffled_topology"]["dev_metrics"]["raster_apls_proxy"] else "FAIL"
    else:
        gain = float("nan")
        verdict = "INCOMPLETE"
    gate = {"experiment_id": "TB-B-260802-027", "verdict": verdict, "manifest_sha256": train_metadata["manifest_sha256"], "seed": args.seed, "strongest_generic": strongest, "mask_topo_minus_strongest_generic": gain, "requirements": {"gain_at_least_0_03": bool(gate_ready and gain >= 0.03), "above_assignment": bool(gate_ready and model_results["mask_topo"]["dev_metrics"]["raster_apls_proxy"] > model_results["mask_assignment_identity"]["dev_metrics"]["raster_apls_proxy"]), "above_shuffled": bool(gate_ready and model_results["mask_topo"]["dev_metrics"]["raster_apls_proxy"] > model_results["shuffled_topology"]["dev_metrics"]["raster_apls_proxy"])}, "metric": "source-mean symmetric raster_apls_proxy", "heldout_content_access_authorized": verdict == "PASS"}
    gate_path = output / "dev_gate.json"
    with gate_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(gate, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    result = {"experiment_id": "TB-B-260802-027", "status": "completed_dev_gate", "seed": args.seed, "device": str(device), "manifest_sha256": train_metadata["manifest_sha256"], "run_freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)}, "train_cache": {"path": str(args.train_cache.resolve()), "sha256": sha256_file(args.train_cache.resolve())}, "dev_cache": {"path": str(args.dev_cache.resolve()), "sha256": sha256_file(args.dev_cache.resolve())}, "mask_cache": {"path": str(mask_cache), "sha256": sha256_file(mask_cache), "dtype": "float16"}, "artifact_cache": {"path": str(artifact_path), "sha256": sha256_file(artifact_path)}, "mask_only_dev_metrics": mask_metrics, "models": model_results, "gate": gate, "total_seconds": time.perf_counter() - started, "massachusetts_test_reopened": False, "p1_5_disease_classification_used": False}
    result_path = output / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "result": str(result_path), "sha256": sha256_file(result_path), "gate": gate, "total_seconds": result["total_seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

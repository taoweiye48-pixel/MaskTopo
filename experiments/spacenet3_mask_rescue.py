from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from spacenet3_metrics import binary_f1, sample_metrics
from spacenet3_models import RoadUNet
from topobridge_mvp import set_seed


CANDIDATES = ("pos8_dice", "focal_dice", "pos4_dice_cldice")
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse-road mask predictor rescue gate on Paris train/dev.")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260840)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    return arrays, metadata


class RoadDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], augment: bool) -> None:
        self.images = arrays["images"]
        self.masks = arrays["masks"]
        self.augment = augment

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index].astype(np.float32))
        mask = torch.from_numpy(self.masks[index].astype(np.float32))
        if self.augment:
            rotations = int(torch.randint(0, 4, ()).item())
            image = torch.rot90(image, rotations, (1, 2))
            mask = torch.rot90(mask, rotations, (0, 1))
            if bool(torch.rand(()) < 0.5):
                image, mask = torch.flip(image, (2,)), torch.flip(mask, (1,))
            if bool(torch.rand(()) < 0.5):
                image, mask = torch.flip(image, (1,)), torch.flip(mask, (0,))
            image = torch.clamp((-0.05 + 0.1 * torch.rand(())) + (0.9 + 0.2 * torch.rand(())) * image, 0.0, 1.0)
        return image, mask


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = torch.sum(probability * target, dim=(1, 2))
    denominator = torch.sum(probability + target, dim=(1, 2))
    return torch.mean(1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-image, 3, stride=1, padding=1)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, 3, stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeleton(image: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    opened = soft_open(image)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = soft_open(image)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits).unsqueeze(1)
    target = target.unsqueeze(1)
    prediction_skeleton = soft_skeleton(probability)
    target_skeleton = soft_skeleton(target)
    topology_precision = (prediction_skeleton * target).sum(dim=(1, 2, 3)) / prediction_skeleton.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    topology_sensitivity = (target_skeleton * probability).sum(dim=(1, 2, 3)) / target_skeleton.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    return torch.mean(1.0 - (2.0 * topology_precision * topology_sensitivity + 1e-6) / (topology_precision + topology_sensitivity + 1e-6))


def candidate_loss(name: str, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    dice = soft_dice_loss(logits, target)
    if name == "pos8_dice":
        primary = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(8.0, device=logits.device))
        topology = torch.zeros((), device=logits.device)
    elif name == "focal_dice":
        probability = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        point_probability = probability * target + (1.0 - probability) * (1.0 - target)
        alpha = 0.75 * target + 0.25 * (1.0 - target)
        primary = torch.mean(alpha * (1.0 - point_probability).pow(2) * bce)
        topology = torch.zeros((), device=logits.device)
    elif name == "pos4_dice_cldice":
        primary = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(4.0, device=logits.device))
        topology = soft_cldice_loss(logits, target)
    else:
        raise ValueError(name)
    total = primary + dice + 0.5 * topology
    return total, {"primary": float(primary.detach()), "dice": float(dice.detach()), "cldice": float(topology.detach())}


@torch.no_grad()
def evaluate_loss(model: RoadUNet, loader: DataLoader, device: torch.device, name: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    for image, target in loader:
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        loss, _ = candidate_loss(name, model(image), target)
        total += float(loss) * image.shape[0]
        count += image.shape[0]
    return total / count


@torch.no_grad()
def predict(model: RoadUNet, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.sigmoid(model(image.to(device, non_blocking=True))).cpu().numpy() for image, _ in loader]).astype(np.float32)


def source_mean(values: np.ndarray, source_ids: np.ndarray) -> float:
    means = []
    source_text = source_ids.astype(str)
    for source in sorted(set(source_text)):
        selected = values[source_text == source]
        selected = selected[np.isfinite(selected)]
        means.append(float(np.mean(selected)) if selected.size else float("nan"))
    return float(np.nanmean(means))


def select_threshold(target: np.ndarray, probability: np.ndarray, source_ids: np.ndarray) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = []
    for threshold in THRESHOLDS:
        f1 = np.asarray([binary_f1(target[index], probability[index] >= threshold) for index in range(target.shape[0])], dtype=np.float64)
        rows.append({"threshold": threshold, "road_f1": source_mean(f1, source_ids), "foreground_fraction": float((probability >= threshold).mean()), "valid_f1_crop_count": float(np.isfinite(f1).sum())})
    return max(rows, key=lambda row: (row["road_f1"], row["threshold"])), rows


def graph_summary(target: np.ndarray, probability: np.ndarray, source_ids: np.ndarray, threshold: float, name: str) -> dict[str, float]:
    rows = []
    for index in range(target.shape[0]):
        rows.append(sample_metrics(target[index], probability[index], threshold))
        if (index + 1) % 200 == 0 or index + 1 == target.shape[0]:
            print(f"MASK_RESCUE_METRICS candidate={name} progress={index + 1}/{target.shape[0]}", flush=True)
    summary = {}
    for metric in ("road_f1", "soft_dice", "cldice", "raster_apls_proxy", "path_recall", "disconnection_rate"):
        summary[metric] = source_mean(np.asarray([float(row[metric]) for row in rows]), source_ids)
    summary["valid_apls_crop_count"] = float(sum(np.isfinite(float(row["raster_apls_proxy"])) for row in rows))
    summary["foreground_fraction"] = float((probability >= threshold).mean())
    summary["threshold"] = threshold
    return summary


def train_candidate(args: argparse.Namespace, name: str, train: dict[str, np.ndarray], dev: dict[str, np.ndarray], output: Path, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(args.seed)
    train_loader = DataLoader(RoadDataset(train, True), batch_size=args.batch_size, shuffle=True, pin_memory=True, generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(RoadDataset(dev, False), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    model = RoadUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    checkpoint = output / f"{name}_seed{args.seed}.pt"
    best = float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for image, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, _ = candidate_loss(name, model(image), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            batches += 1
        dev_loss = evaluate_loss(model, dev_loader, device, name)
        row = {"candidate": name, "epoch": epoch, "train_loss": total / batches, "dev_loss": dev_loss, "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        print(f"MASK_RESCUE candidate={name} epoch={epoch}/{args.epochs} train_loss={row['train_loss']:.4f} dev_loss={dev_loss:.4f}", flush=True)
        if dev_loss < best:
            best = dev_loss
            torch.save({"model": model.state_dict(), "candidate": name, "seed": args.seed, "best_dev_loss": best, "epoch": epoch}, checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    raw = predict(model, dev_loader, device)
    prediction_path = output / f"dev_predictions_{name}_seed{args.seed}.npz"
    np.savez_compressed(prediction_path, probability=raw.astype(np.float16), source_ids=dev["source_ids"], crop_rows=dev["crop_rows"], crop_columns=dev["crop_columns"])
    with np.load(prediction_path, allow_pickle=False) as archive:
        probability = archive["probability"].astype(np.float32)
    selected, sweep = select_threshold(dev["masks"], probability, dev["source_ids"])
    metrics = graph_summary(dev["masks"], probability, dev["source_ids"], selected["threshold"], name)
    return {"candidate": name, "best_dev_loss": best, "best_epoch": state["epoch"], "training_seconds": time.perf_counter() - started, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "prediction": str(prediction_path), "prediction_sha256": sha256_file(prediction_path), "threshold_sweep": sweep, "selected_metrics": metrics}, history


def main() -> None:
    args = parse_args()
    if args.self_test:
        image = torch.rand(2, 3, 64, 64)
        target = torch.zeros(2, 64, 64)
        target[:, 31:33, 8:56] = 1
        for name in CANDIDATES:
            model = RoadUNet(base=8)
            loss, _ = candidate_loss(name, model(image), target)
            loss.backward()
            assert torch.isfinite(loss)
        print("SPACENET3_MASK_RESCUE_SELF_TEST_PASS")
        return
    if args.seed != 20260840 or args.epochs != 20 or args.batch_size != 64 or not torch.cuda.is_available():
        raise ValueError("Arguments/device differ from frozen protocol")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    train, train_metadata = load_cache(args.train_cache.resolve())
    dev, dev_metadata = load_cache(args.dev_cache.resolve())
    if train_metadata["manifest_sha256"] != dev_metadata["manifest_sha256"]:
        raise RuntimeError("Cache manifest mismatch")
    freeze = {"experiment_id": "TB-B-260803-029", "status": "frozen_before_training", "protocol_sha256": sha256_file(args.protocol.resolve()), "code_sha256": sha256_file(Path(__file__).resolve()), "model_code_sha256": sha256_file((Path(__file__).parent / "spacenet3_models.py").resolve()), "train_cache_sha256": sha256_file(args.train_cache.resolve()), "dev_cache_sha256": sha256_file(args.dev_cache.resolve()), "manifest_sha256": train_metadata["manifest_sha256"], "candidates": CANDIDATES, "thresholds": THRESHOLDS, "seed": args.seed, "epochs": args.epochs, "heldout_content_accessed": False}
    with (output / "run_freeze.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(freeze, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    results = {}
    histories = {}
    for name in CANDIDATES:
        results[name], histories[name] = train_candidate(args, name, train, dev, output, torch.device("cuda"))
        torch.cuda.empty_cache()
    best = max(CANDIDATES, key=lambda name: (results[name]["selected_metrics"]["road_f1"], -results[name]["selected_metrics"]["foreground_fraction"], -CANDIDATES.index(name)))
    metrics = results[best]["selected_metrics"]
    requirements = {"road_f1_at_least_0_15": metrics["road_f1"] >= 0.15, "apls_at_least_0_20": metrics["raster_apls_proxy"] >= 0.20, "foreground_fraction_at_most_0_05": metrics["foreground_fraction"] <= 0.05}
    verdict = "PASS_PARIS_DECODER_DEVELOPMENT_AUTHORIZED" if all(requirements.values()) else "FAIL_NATURAL_GRAPH_EXTENSION_PAUSED"
    with (output / "history.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(histories, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    result = {"experiment_id": "TB-B-260803-029", "status": "completed_mask_rescue_gate", "verdict": verdict, "best_candidate": best, "requirements": requirements, "candidates": results, "manifest_sha256": train_metadata["manifest_sha256"], "heldout_content_accessed": False, "parent_tb027_gate_remains": "FAIL"}
    result_path = output / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "verdict": verdict, "best_candidate": best, "best_metrics": metrics, "requirements": requirements, "result": str(result_path), "sha256": sha256_file(result_path)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


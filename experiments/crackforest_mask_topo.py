from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.ndimage import binary_closing, label
from torch.utils.data import DataLoader, Dataset

from crackforest_real_gate import (
    RealDataset,
    balanced_binary_loss,
    balanced_metrics,
    coarse_graph8,
    group_structural_prediction,
    load_sources,
    patch_component_ids,
    predicted_artifacts,
    source_split,
    train_classifier,
    write_csv,
)
from edge_topocoarsen_connector import bootstrap_balanced_accuracy
from topobridge_mvp import set_seed
from topocoarsen_oracle import oracle_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Global mask-consistent TopoCoarsen diagnostic on CrackForest."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("real_data/CrackForest-dataset"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("real_cache"))
    parser.add_argument(
        "--output", type=Path, default=Path("results_crackforest_mask_topo")
    )
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--dev-size", type=int, default=320)
    parser.add_argument("--test-size", type=int, default=320)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--mask-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument(
        "--mask-thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument(
        "--closing-iterations", type=int, nargs="+", default=[0, 1, 2]
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def find_cached_split(
    cache_dir: Path, split: str, count: int, seed: int, crop_size: int
) -> dict[str, np.ndarray]:
    paths = list(
        cache_dir.glob(
            f"crackforest_{split}_n{count}_seed{seed}_"
            f"crop{crop_size}_*.npz"
        )
    )
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one cached {split} split, found {len(paths)}."
        )
    with np.load(paths[0]) as archive:
        return {key: archive[key] for key in archive.files}


def build_pixel_masks(
    arrays: dict[str, np.ndarray],
    source_masks: dict[str, np.ndarray],
) -> np.ndarray:
    masks = np.empty(
        (arrays["images"].shape[0], 64, 64), dtype=np.uint8
    )
    for index, (source_id, box) in enumerate(
        zip(arrays["source_id"], arrays["crop_box"])
    ):
        top, left, bottom, right = (int(item) for item in box)
        crop = source_masks[str(source_id)][top:bottom, left:right]
        if crop.shape != (64, 64):
            raise ValueError("MaskTopo requires native 64x64 crops.")
        masks[index] = crop.astype(np.uint8)
    return masks


class MaskDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        augment: bool,
    ) -> None:
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(
            self.images[index, 0].astype(np.float32) / 255.0
        ).unsqueeze(0)
        if self.augment:
            contrast = 0.78 + 0.42 * torch.rand(())
            brightness = -0.06 + 0.12 * torch.rand(())
            noise = 0.035 * torch.rand(()) * torch.randn_like(image)
            image = torch.clamp(
                brightness + contrast * image + noise, 0.0, 1.0
            )
        return {
            "image": image,
            "mask": torch.from_numpy(
                self.masks[index].astype(np.float32)
            ),
        }


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class CrackUNet(nn.Module):
    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(1, base)
        self.encoder2 = ConvBlock(base, 2 * base)
        self.encoder3 = ConvBlock(2 * base, 4 * base)
        self.bottleneck = ConvBlock(4 * base, 8 * base)
        self.decoder3 = ConvBlock(12 * base, 4 * base)
        self.decoder2 = ConvBlock(6 * base, 2 * base)
        self.decoder1 = ConvBlock(3 * base, base)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        first = self.encoder1(image)
        second = self.encoder2(F.max_pool2d(first, 2))
        third = self.encoder3(F.max_pool2d(second, 2))
        bottleneck = self.bottleneck(F.max_pool2d(third, 2))
        value = F.interpolate(
            bottleneck, size=third.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.decoder3(torch.cat((value, third), dim=1))
        value = F.interpolate(
            value, size=second.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.decoder2(torch.cat((value, second), dim=1))
        value = F.interpolate(
            value, size=first.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.decoder1(torch.cat((value, first), dim=1))
        return self.output(value).squeeze(1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = torch.sum(probability * target, dim=(1, 2))
    denominator = torch.sum(probability + target, dim=(1, 2))
    return torch.mean(1.0 - (2 * intersection + 1) / (denominator + 1))


def mask_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = balanced_binary_loss(logits, target > 0.5)
    dice = dice_loss(logits, target)
    total = bce + dice
    return total, {
        "balanced_bce": float(bce.detach()),
        "dice_loss": float(dice.detach()),
    }


@torch.no_grad()
def segmentation_metrics(
    model: CrackUNet,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["mask"].to(device).bool()
        prediction = torch.sigmoid(model(image)) >= threshold
        true_positive += int((prediction & target).sum())
        false_positive += int((prediction & ~target).sum())
        false_negative += int((~prediction & target).sum())
        true_negative += int((~prediction & ~target).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    return {
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def train_mask_model(
    args: argparse.Namespace,
    train_dataset: MaskDataset,
    dev_dataset: MaskDataset,
    output: Path,
    device: torch.device,
) -> tuple[CrackUNet, list[dict[str, Any]]]:
    set_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False
    )
    model = CrackUNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0
    checkpoint_path = output / f"mask_predictor_seed{args.seed}.pt"
    history = []
    started = time.time()
    for epoch in range(1, args.mask_epochs + 1):
        model.train()
        sums = {"loss": 0.0, "balanced_bce": 0.0, "dice_loss": 0.0}
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(image)
                loss, parts = mask_loss(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.detach())
            for key, value in parts.items():
                sums[key] += value
            batches += 1
        metrics = segmentation_metrics(model, dev_loader, device)
        record = {
            "stage": "mask_predictor",
            "epoch": epoch,
            **{key: value / batches for key, value in sums.items()},
            **{f"dev_{key}": value for key, value in metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"MASK_PREDICTOR epoch={epoch}/{args.mask_epochs} "
            f"loss={record['loss']:.4f} "
            f"dev_f1={metrics['f1']:.4f} "
            f"dev_recall={metrics['recall']:.4f}",
            flush=True,
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
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
def predict_masks(
    model: CrackUNet,
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, images.shape[0], batch_size):
        image = torch.from_numpy(
            images[start : start + batch_size, 0].astype(np.float32) / 255.0
        ).unsqueeze(1).to(device)
        parts.append(torch.sigmoid(model(image)).cpu().numpy())
    return np.concatenate(parts)


def endpoint_pixels(image: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    endpoints = []
    for channel in (1, 2):
        coordinates = np.argwhere(image[channel] > 127)
        if coordinates.size == 0:
            raise ValueError("Missing endpoint marker.")
        center = np.mean(coordinates, axis=0)
        endpoints.append(
            (
                int(np.clip(round(center[0]), 0, 63)),
                int(np.clip(round(center[1]), 0, 63)),
            )
        )
    return endpoints[0], endpoints[1]


def mask_groups(
    probability: np.ndarray,
    threshold: float,
    closing_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    binary = probability >= threshold
    if closing_iterations:
        binary = binary_closing(
            binary,
            structure=np.ones((3, 3), dtype=bool),
            iterations=closing_iterations,
        )
    component_map, _ = label(
        binary, structure=np.ones((3, 3), dtype=np.uint8)
    )
    patches = patch_component_ids(component_map)
    return component_map, patches


def mask_structural_prediction(
    component_map: np.ndarray,
    endpoints: tuple[tuple[int, int], tuple[int, int]],
) -> int:
    first = int(component_map[endpoints[0]])
    second = int(component_map[endpoints[1]])
    return int(first > 0 and first == second)


def calibrate_mask(
    arrays: dict[str, np.ndarray],
    probabilities: np.ndarray,
    thresholds: list[float],
    closing_iterations: list[int],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = []
    for threshold in thresholds:
        for closing in closing_iterations:
            predictions = []
            for index in range(probabilities.shape[0]):
                _, groups = mask_groups(
                    probabilities[index], threshold, closing
                )
                predictions.append(
                    group_structural_prediction(
                        groups, arrays["endpoint_patches"][index]
                    )
                )
            metrics = balanced_metrics(
                arrays["connected"], np.asarray(predictions)
            )
            rows.append(
                {
                    "mask_threshold": threshold,
                    "closing_iterations": closing,
                    **metrics,
                }
            )
    selected = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["mask_threshold"],
            -row["closing_iterations"],
        ),
    )
    return selected, rows


def mask_artifacts(
    arrays: dict[str, np.ndarray],
    probabilities: np.ndarray,
    threshold: float,
    closing_iterations: int,
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
        component_map, groups = mask_groups(
            probabilities[index], threshold, closing_iterations
        )
        components = [
            int(item) for item in np.unique(groups) if int(item) > 0
        ]
        if len(components) > 15:
            sizes = {
                component: int(np.sum(groups == component))
                for component in components
            }
            keep = set(
                sorted(components, key=sizes.__getitem__, reverse=True)[:15]
            )
            for component in components:
                if component not in keep:
                    groups[groups == component] = 0
            remap = {
                old: new
                for new, old in enumerate(
                    [
                        0,
                        *sorted(
                            int(item)
                            for item in np.unique(groups)
                            if item > 0
                        ),
                    ]
                )
            }
            groups = np.vectorize(
                remap.__getitem__, otypes=[np.int16]
            )(groups)
        component_counts.append(
            int(np.unique(groups[groups > 0]).size)
        )
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
        "patch_component_counts": {
            "median": float(np.median(counts)),
            "p95": float(np.quantile(counts, 0.95)),
            "maximum": int(counts.max()),
        },
    }


def save_prediction_examples(
    images: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5), constrained_layout=True)
    for row, index in enumerate((0, 1, 2)):
        axes[row, 0].imshow(images[index, 0], cmap="gray")
        axes[row, 0].set_title("input")
        axes[row, 1].imshow(targets[index], cmap="gray")
        axes[row, 1].set_title("annotation")
        axes[row, 2].imshow(probabilities[index], cmap="magma", vmin=0, vmax=1)
        axes[row, 2].set_title("probability")
        axes[row, 3].imshow(probabilities[index] >= 0.5, cmap="gray")
        axes[row, 3].set_title("threshold 0.5")
        for axis in axes[row]:
            axis.axis("off")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_self_test() -> None:
    model = CrackUNet(base=8)
    logits = model(torch.rand(2, 1, 64, 64))
    target = torch.zeros(2, 64, 64)
    target[:, 20:22, 10:50] = 1
    loss, parts = mask_loss(logits, target)
    assert logits.shape == target.shape
    assert torch.isfinite(loss)
    assert set(parts) == {"balanced_bce", "dice_loss"}
    component_map, patches = mask_groups(
        target[0].numpy(), 0.5, 0
    )
    assert component_map.shape == (64, 64)
    assert patches.shape == (16, 16)
    print("CRACKFOREST_MASK_TOPO_SELF_TEST_PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.crop_size != 64 or args.token_count != 16:
        raise ValueError("This diagnostic requires crop-size=64 and 16 tokens.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    arrays = {
        "train": find_cached_split(
            cache_dir,
            "train",
            args.train_size,
            args.data_seed,
            args.crop_size,
        ),
        "dev": find_cached_split(
            cache_dir,
            "dev",
            args.dev_size,
            args.data_seed + 1_000_000,
            args.crop_size,
        ),
        "test": find_cached_split(
            cache_dir,
            "test",
            args.test_size,
            args.data_seed + 2_000_000,
            args.crop_size,
        ),
    }
    manifest = source_split(args.dataset_root.resolve(), args.split_seed)
    sources = load_sources(
        args.dataset_root.resolve(),
        sorted(set().union(*[set(items) for items in manifest.values()])),
    )
    source_masks = {key: value[1] for key, value in sources.items()}
    pixel_masks = {
        split: build_pixel_masks(current, source_masks)
        for split, current in arrays.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_model, history = train_mask_model(
        args,
        MaskDataset(arrays["train"]["images"], pixel_masks["train"], True),
        MaskDataset(arrays["dev"]["images"], pixel_masks["dev"], False),
        output,
        device,
    )
    probabilities = {
        split: predict_masks(
            mask_model,
            current["images"],
            args.batch_size,
            device,
        )
        for split, current in arrays.items()
    }
    selected, calibration_rows = calibrate_mask(
        arrays["dev"],
        probabilities["dev"],
        args.mask_thresholds,
        args.closing_iterations,
    )
    write_csv(output / "mask_calibration.csv", calibration_rows)
    artifacts = {}
    artifact_diagnostics = {}
    for split in ("train", "dev", "test"):
        artifacts[split], artifact_diagnostics[split] = mask_artifacts(
            arrays[split],
            probabilities[split],
            float(selected["mask_threshold"]),
            int(selected["closing_iterations"]),
            args.token_count,
        )
        print(
            f"MASK_ARTIFACT split={split} "
            f"structural="
            f"{artifact_diagnostics[split]['structural_metrics']['balanced_accuracy']:.4f}",
            flush=True,
        )
    save_prediction_examples(
        arrays["test"]["images"],
        pixel_masks["test"],
        probabilities["test"],
        output / "mask_predictions.png",
    )

    classifier_args = argparse.Namespace(**vars(args))
    classifier_args.num_workers = 0
    train_dataset = RealDataset(
        arrays["train"], artifacts["train"], augment=True
    )
    dev_dataset = RealDataset(arrays["dev"], artifacts["dev"])
    test_dataset = RealDataset(arrays["test"], artifacts["test"])
    summary, classifier_history, prediction, truth = train_classifier(
        "edge_topocoarsen",
        classifier_args,
        train_dataset,
        dev_dataset,
        test_dataset,
        output,
        device,
    )
    history.extend(classifier_history)
    write_csv(output / "history.csv", history)
    np.savez_compressed(
        output / "heldout_predictions.npz",
        truth=truth,
        mask_topocoarsen=prediction,
    )

    prior_result_path = (
        args.output.resolve().parent
        / "results_crackforest_crop64_seed20260810"
        / "result.json"
    )
    prior = json.loads(prior_result_path.read_text(encoding="utf-8"))
    baseline_name = prior["strongest_image_only_baseline"]
    baseline_score = prior["model_results"][baseline_name][
        "test_metrics"
    ]["balanced_accuracy"]
    score = summary["test_metrics"]["balanced_accuracy"]
    gain = score - baseline_score
    result = {
        "experiment_id": "crackforest_global_mask_topocoarsen",
        "status": "completed",
        "method": (
            "global crack probability mask -> connected components -> "
            "deterministic 256-to-16 topology coarsening"
        ),
        "source_image_split_unchanged": True,
        "data_seed": args.data_seed,
        "optimization_seed": args.seed,
        "selected_on_dev": selected,
        "artifact_diagnostics": artifact_diagnostics,
        "classifier": summary,
        "strongest_prior_image_only_baseline": {
            "name": baseline_name,
            "balanced_accuracy": baseline_score,
        },
        "gain_over_baseline": gain,
        "bootstrap_95_ci": bootstrap_balanced_accuracy(
            truth, prediction, args.seed
        ),
        "gates": {
            "test_structural_accuracy_at_least_70pct": (
                artifact_diagnostics["test"]["structural_metrics"][
                    "balanced_accuracy"
                ]
                >= 0.70
            ),
            "classifier_gain_at_least_3pp": gain >= 0.03,
        },
    }
    result["verdict"] = (
        "PROCEED_TO_MULTISEED"
        if all(result["gates"].values())
        else (
            "GLOBAL_MASK_LEARNABLE_CONNECTOR_HEAD_FAILED"
            if result["gates"]["test_structural_accuracy_at_least_70pct"]
            else "STOP_CURRENT_REAL_DATA_ROUTE"
        )
    )
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

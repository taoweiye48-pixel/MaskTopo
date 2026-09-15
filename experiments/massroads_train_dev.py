from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from crackforest_mask_topo import (
    MaskDataset,
    calibrate_mask,
    predict_masks,
    save_prediction_examples,
    train_mask_model,
)
from crackforest_mechanism_ablation import build_artifacts
from crackforest_real_gate import (
    RealDataset,
    evaluate_classifier,
    model_forward,
    write_csv,
)
from fives_external_gate import (
    grid_artifacts as fives_grid_artifacts,
    rectangular_grid_assignment,
)
from fives_g2tm_collision import make_collision_model
from strong_reducer_baselines import (
    ReducerDataset,
    evaluate as evaluate_reducer,
    forward_batch as forward_reducer_batch,
    make_model as make_reducer_model,
)
from topobridge_mvp import set_seed
from topocoarsen_oracle import TopoCoarsenModel


TOPOLOGY_MODELS = (
    "grid_external",
    "mask_assignment_identity",
    "shuffled_topology",
    "mask_topo_external",
)
REDUCER_MODELS = (
    "tokenlearner",
    "perceiver_resampler",
    "tome_style",
    "g2tm_fixedk_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and freeze Massachusetts Roads models using train/dev only."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("real_data/MassachusettsRoads"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("massroads_cache"))
    parser.add_argument(
        "--data-audit", type=Path, default=Path("results_massroads_data_audit")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--dev-size", type=int, default=600)
    parser.add_argument("--mask-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-seed", type=int, default=20260802)
    parser.add_argument("--shuffle-seed", type=int, default=20260802)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_final_split(
    cache_dir: Path, split: str, count: int, fingerprint: str
) -> tuple[dict[str, np.ndarray], Path]:
    paths = list(
        cache_dir.glob(
            f"massroads_{split}_n{count}_seed*_matchedv1_{fingerprint}.npz"
        )
    )
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one frozen {split} cache, found {len(paths)}: {paths}"
        )
    with np.load(paths[0]) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, paths[0]


def load_used_source_masks(
    root: Path, arrays: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    source_ids = sorted(
        set(str(item) for current in arrays.values() for item in current["source_id"])
    )
    masks = {}
    for index, source_id in enumerate(source_ids, start=1):
        logical_split, stem = source_id.split(":", maxsplit=1)
        disk_split = "train" if logical_split == "train" else "valid"
        path = root / disk_split / "map" / f"{stem}.tif"
        with Image.open(path) as image:
            masks[source_id] = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
        if index % 100 == 0 or index == len(source_ids):
            print(
                f"MASSROADS_MASK_LOAD sources={index}/{len(source_ids)}",
                flush=True,
            )
    return masks


def build_pixel_masks_2x2(
    arrays: dict[str, np.ndarray], source_masks: dict[str, np.ndarray]
) -> np.ndarray:
    output = np.empty((arrays["images"].shape[0], 64, 64), dtype=np.uint8)
    for index, (source_id, box) in enumerate(
        zip(arrays["source_id"], arrays["crop_box"])
    ):
        top, left, bottom, right = (int(item) for item in box)
        crop = source_masks[str(source_id)][top:bottom, left:right]
        if crop.shape != (128, 128):
            raise ValueError(f"Unexpected target crop shape {crop.shape}.")
        output[index] = crop.reshape(64, 2, 64, 2).max(axis=(1, 3)).astype(np.uint8)
    return output


def train_topology_dev_only(
    model_name: str,
    args: argparse.Namespace,
    train_dataset: RealDataset,
    dev_dataset: RealDataset,
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    model = TopoCoarsenModel(model_name, args.dim, args.token_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = output / f"{model_name}_seed{args.seed}.pt"
    best = -1.0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
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
            loss_sum += float(loss.detach())
            batches += 1
        dev_metrics, _, _ = evaluate_classifier(
            model, model_name, dev_loader, device
        )
        record = {
            "stage": "classifier_train_dev",
            "seed": args.seed,
            "model": model_name,
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"MASSROADS_TOPO model={model_name} epoch={epoch}/{args.epochs} "
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
                    "model_family": "topology",
                    "dev_metrics": dev_metrics,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return (
        {
            "model": model_name,
            "model_family": "topology",
            "seed": args.seed,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_dev_balanced_accuracy": best,
            "best_dev_metrics": checkpoint["dev_metrics"],
            "training_seconds": time.time() - started,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        history,
    )


def train_reducer_dev_only(
    model_name: str,
    factory: Callable[[str, int, int], nn.Module],
    args: argparse.Namespace,
    datasets: dict[str, ReducerDataset],
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    }
    model = factory(model_name, args.dim, args.token_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = output / f"{model_name}_seed{args.seed}.pt"
    best = -1.0
    history = []
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
                logits = forward_reducer_batch(model, batch, device)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batches += 1
        dev_metrics, _, _ = evaluate_reducer(model, loaders["dev"], device)
        record = {
            "stage": "reducer_train_dev",
            "seed": args.seed,
            "model": model_name,
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"MASSROADS_REDUCER model={model_name} epoch={epoch}/{args.epochs} "
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
                    "model_family": "g2tm"
                    if model_name == "g2tm_fixedk_mask"
                    else "reducer",
                    "dev_metrics": dev_metrics,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return (
        {
            "model": model_name,
            "model_family": checkpoint["model_family"],
            "seed": args.seed,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_dev_balanced_accuracy": best,
            "best_dev_metrics": checkpoint["dev_metrics"],
            "training_seconds": time.time() - started,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        history,
    )


def run_self_test() -> None:
    grid = rectangular_grid_assignment(8)
    assert grid.shape == (16, 16)
    assert np.array_equal(np.unique(grid), np.arange(8))
    assert all(np.count_nonzero(grid == token) == 32 for token in range(8))
    array = np.zeros((2, 3, 64, 64), dtype=np.uint8)
    masks = np.zeros((2, 64, 64), dtype=np.float32)
    dataset = ReducerDataset(
        {"images": array, "connected": np.asarray([0, 1])}, masks
    )
    assert len(dataset) == 2
    for name in ("tokenlearner", "perceiver_resampler", "tome_style"):
        model = make_reducer_model(name, 16, 8)
        logits = model(torch.rand(2, 3, 64, 64), torch.rand(2, 1, 64, 64))
        assert logits.shape == (2,)
    model = make_collision_model("g2tm_fixedk_mask", 16, 8)
    assert model(torch.rand(2, 3, 64, 64), torch.rand(2, 1, 64, 64)).shape == (2,)
    print("MASSROADS_TRAIN_DEV_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.output is None or args.seed is None:
        raise ValueError("Formal training requires both --output and --seed.")
    if args.seed not in {20260820, 20260821, 20260822}:
        raise ValueError("Seed is outside the frozen Massachusetts protocol.")
    if (
        args.train_size != 4000
        or args.dev_size != 600
        or args.mask_epochs != 30
        or args.epochs != 22
        or args.batch_size != 32
        or args.token_count != 8
    ):
        raise ValueError("Command differs from the frozen Massachusetts protocol.")
    root = args.dataset_root.resolve()
    if (root / "test").exists():
        raise RuntimeError("Train/dev freezing must run before any local test directory exists.")
    audit_path = args.data_audit.resolve() / "data_gate.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["verdict"] != "TRAIN_DEV_DATA_GATE_PASS" or audit["test_accessed"]:
        raise RuntimeError("Frozen train/dev data gate is not valid.")
    fingerprint = str(audit["dataset_fingerprint"])
    arrays = {}
    cache_paths = {}
    for split, count in (("train", args.train_size), ("dev", args.dev_size)):
        arrays[split], cache_paths[split] = load_final_split(
            args.cache_dir.resolve(), split, count, fingerprint
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    source_masks = load_used_source_masks(root, arrays)
    pixel_masks = {
        split: build_pixel_masks_2x2(current, source_masks)
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
        split: predict_masks(mask_model, current["images"], args.batch_size, device)
        for split, current in arrays.items()
    }
    probability_path = output / "mask_probabilities_train_dev.npz"
    np.savez_compressed(
        probability_path,
        **{split: values.astype(np.float16) for split, values in probabilities.items()},
    )
    selected, calibration_rows = calibrate_mask(
        arrays["dev"],
        probabilities["dev"],
        args.mask_thresholds,
        args.closing_iterations,
    )
    write_csv(output / "mask_calibration.csv", calibration_rows)
    save_prediction_examples(
        arrays["dev"]["images"],
        pixel_masks["dev"],
        probabilities["dev"],
        output / "validation_mask_predictions.png",
    )
    threshold = float(selected["mask_threshold"])
    closing = int(selected["closing_iterations"])

    grid_artifacts = {
        split: fives_grid_artifacts(current, args.token_count)[0]
        for split, current in arrays.items()
    }
    artifact_modes = {
        "mask_assignment_identity": "mask_assignment_identity",
        "shuffled_topology": "shuffled_topology",
        "mask_topo_external": "mask_topo_recheck",
    }
    artifact_bank: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    artifact_diagnostics: dict[str, Any] = {}
    for result_name, builder_name in artifact_modes.items():
        artifact_bank[result_name] = {}
        artifact_diagnostics[result_name] = {}
        for split_index, split in enumerate(("train", "dev")):
            (
                artifact_bank[result_name][split],
                artifact_diagnostics[result_name][split],
            ) = build_artifacts(
                arrays[split],
                probabilities[split],
                threshold,
                closing,
                args.token_count,
                builder_name,
                args.shuffle_seed + 100_000 * split_index,
            )

    summaries: dict[str, Any] = {}
    for model_name in TOPOLOGY_MODELS:
        artifacts = grid_artifacts if model_name == "grid_external" else artifact_bank[model_name]
        summary, rows = train_topology_dev_only(
            model_name,
            args,
            RealDataset(arrays["train"], artifacts["train"], augment=True),
            RealDataset(arrays["dev"], artifacts["dev"]),
            output,
            device,
        )
        summaries[model_name] = summary
        history.extend(rows)

    reducer_datasets = {
        split: ReducerDataset(
            arrays[split], probabilities[split], augment=split == "train"
        )
        for split in ("train", "dev")
    }
    for model_name in REDUCER_MODELS:
        factory = (
            make_collision_model
            if model_name == "g2tm_fixedk_mask"
            else make_reducer_model
        )
        summary, rows = train_reducer_dev_only(
            model_name, factory, args, reducer_datasets, output, device
        )
        summary["uses_predicted_mask"] = model_name == "g2tm_fixedk_mask"
        summary["method_status"] = (
            "matched_fixed_budget_adaptation_of_official_g2tm"
            if model_name == "g2tm_fixedk_mask"
            else "controlled_in_framework_implementation"
        )
        summaries[model_name] = summary
        history.extend(rows)

    write_csv(output / "history.csv", history)
    mask_checkpoint = output / f"mask_predictor_seed{args.seed}.pt"
    result = {
        "experiment_id": f"TB-B-260802-021_massroads_train_dev_seed{args.seed}",
        "status": "completed_train_dev_frozen",
        "test_accessed": False,
        "dataset_fingerprint": fingerprint,
        "optimization_seed": args.seed,
        "data_seed": args.data_seed,
        "token_count": args.token_count,
        "selected_on_validation": selected,
        "artifact_diagnostics": artifact_diagnostics,
        "model_results": summaries,
        "mask_checkpoint": {
            "path": str(mask_checkpoint),
            "sha256": sha256_file(mask_checkpoint),
        },
        "mask_probabilities_train_dev": {
            "path": str(probability_path),
            "sha256": sha256_file(probability_path),
        },
        "cache_files": {
            split: {"path": str(path), "sha256": sha256_file(path)}
            for split, path in cache_paths.items()
        },
        "protocol": str(
            Path(__file__).resolve().parent
            / "MASSACHUSETTS_ROADS_EXTERNAL_CONFIRMATION_PROTOCOL.md"
        ),
    }
    (output / "train_dev_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spacenet3_metrics import binary_f1, cldice, raster_apls_proxy
from spacenet3_stage1_metrics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    THRESHOLDS,
    bootstrap_summary,
    gate_a_metrics,
    select_threshold,
    source_mosaic_metrics,
    stitch_mosaic,
    write_csv,
)
from spacenet3_stage1_models import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    METHODS,
    ResNet34UNet,
    artifact_inputs,
    build_artifacts,
    function_sha256,
    imagenet_weight_record,
    make_model,
    parameter_count,
    segmentation_loss,
    soft_cldice_loss,
    structure_hash,
)
from spacenet3_stage1_prepare import freeze_manifest, sha256_file
from topobridge_mvp import set_seed


EXPERIMENT_ID = "TB-B-260803-030"
OPTIMIZATION_SEED = 20260850
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HARD_TIMEOUT_GATE_A_SECONDS = 2 * 60 * 60
HARD_TIMEOUT_GATE_B_SECONDS = 6 * 60 * 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def load_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("images", "masks", "source_ids", "crop_rows", "crop_columns")}
        metadata = json.loads(str(archive["metadata_json"].item()))
    return arrays, metadata


def configure_determinism(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class RoadDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], augment: bool) -> None:
        self.images = arrays["images"]
        self.masks = arrays["masks"]
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]

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
            brightness = 0.90 + 0.20 * torch.rand(())
            contrast = 0.90 + 0.20 * torch.rand(())
            image = image * brightness
            channel_mean = image.mean(dim=(1, 2), keepdim=True)
            image = torch.clamp(channel_mean + contrast * (image - channel_mean), 0.0, 1.0)
        return (image - self.mean) / self.std, mask


class ArtifactDataset(RoadDataset):
    def __init__(self, arrays: dict[str, np.ndarray], artifacts: dict[str, np.ndarray], augment: bool) -> None:
        super().__init__(arrays, augment)
        self.artifacts = artifacts

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, mask = super().__getitem__(index)
        return {
            "image": image,
            "target": mask,
            "assignment": torch.from_numpy(self.artifacts["assignment"][index].astype(np.int64)),
            "adjacency_identity": torch.from_numpy(self.artifacts["adjacency_identity"][index]),
            "reachability_identity": torch.from_numpy(self.artifacts["reachability_identity"][index]),
            "adjacency_real": torch.from_numpy(self.artifacts["adjacency_real"][index]),
            "reachability_real": torch.from_numpy(self.artifacts["reachability_real"][index]),
            "adjacency_shuffled": torch.from_numpy(self.artifacts["adjacency_shuffled"][index]),
            "reachability_shuffled": torch.from_numpy(self.artifacts["reachability_shuffled"][index]),
        }


def make_loader(dataset: Dataset, shuffle: bool, seed: int, batch_size: int = BATCH_SIZE) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def amp_context(device: torch.device) -> torch.amp.autocast:
    return torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device, method: str | None = None) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        if method is None:
            image, target = batch
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            inputs: dict[str, torch.Tensor] = {}
        else:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key not in {"image", "target"}}
            inputs = artifact_inputs(method, moved)
        with amp_context(device):
            logits = model(image, **inputs)
            loss, _ = segmentation_loss(logits, target)
        total += float(loss) * image.shape[0]
        count += image.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, method: str | None = None) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for batch in loader:
        if method is None:
            image, _ = batch
            inputs: dict[str, torch.Tensor] = {}
        else:
            image = batch["image"]
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key not in {"image", "target"}}
            inputs = artifact_inputs(method, moved)
        image = image.to(device, non_blocking=True)
        with amp_context(device):
            logits = model(image, **inputs)
        outputs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(outputs)


def train_one(
    model: torch.nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    checkpoint: Path,
    method: str | None,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        if time.perf_counter() - started > timeout_seconds:
            raise TimeoutError(f"Hard timeout reached during training at epoch {epoch}")
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            if method is None:
                image, target = batch
                image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
                inputs: dict[str, torch.Tensor] = {}
            else:
                image = batch["image"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                moved = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key not in {"image", "target"}}
                inputs = artifact_inputs(method, moved)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = model(image, **inputs)
                loss, components = segmentation_loss(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * image.shape[0]
            count += image.shape[0]
        dev_loss = evaluate_loss(model, dev_loader, device, method)
        train_loss = total / max(count, 1)
        if not np.isfinite(train_loss) or not np.isfinite(dev_loss):
            raise FloatingPointError(f"Non-finite epoch result at epoch {epoch}: train={train_loss}, dev={dev_loss}")
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "dev_loss": dev_loss}, checkpoint)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm_last_batch": gradient_norm,
            "loss_components_last_batch": components,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(f"TRAIN method={method or 'mask_predictor'} epoch={epoch:02d}/{EPOCHS} train_loss={train_loss:.6f} dev_loss={dev_loss:.6f} best_epoch={best_epoch} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
        scheduler.step()
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint_data["model"])
    convergence = convergence_check(history)
    return history, {"best_epoch": best_epoch, "best_dev_loss": best_loss, "convergence": convergence, "training_seconds": time.perf_counter() - started}


def convergence_check(history: list[dict[str, Any]]) -> dict[str, Any]:
    train_values = np.asarray([float(row["train_loss"]) for row in history])
    dev_values = np.asarray([float(row["dev_loss"]) for row in history])
    first_mean = float(dev_values[:10].mean())
    last_mean = float(dev_values[-10:].mean())
    last_std = float(dev_values[-10:].std(ddof=0))
    requirements = {
        "finite_train_and_dev": bool(np.all(np.isfinite(train_values)) and np.all(np.isfinite(dev_values))),
        "first10_dev_loss_gt_last10": bool(first_mean > last_mean),
        "last10_std_lt_first_minus_last": bool(last_std < first_mean - last_mean),
    }
    return {"first10_dev_loss_mean": first_mean, "last10_dev_loss_mean": last_mean, "last10_dev_loss_std": last_std, "requirements": requirements, "pass": all(requirements.values())}


def environment_record(python: Path) -> tuple[str, dict[str, Any]]:
    import pandas
    import rasterio
    import scipy
    import shapely
    import skimage
    import torchvision

    pip_freeze = subprocess.run([str(python), "-m", "pip", "freeze"], check=True, capture_output=True, text=True, encoding="utf-8").stdout
    nvidia = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    record = {
        "python_executable": str(python.resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "rasterio": rasterio.__version__,
        "shapely": shapely.__version__,
        "scikit_image": skimage.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "nvidia_smi": nvidia,
        "pip_freeze_sha256": sha256_text(pip_freeze),
    }
    text = json.dumps(record, ensure_ascii=False, indent=2) + "\n\n[pip freeze]\n" + pip_freeze
    return text, record


def formal_commands(args: argparse.Namespace) -> dict[str, str]:
    python = str(args.python.resolve())
    root = str(Path(__file__).resolve().parent)
    output = str(args.output.resolve())
    manifest = str(args.manifest.resolve())
    task = str(args.task.resolve())
    return {
        "preflight": f'"{python}" "{root}\\spacenet3_stage1_run.py" preflight --python "{python}" --task "{task}" --manifest "{manifest}" --old-manifest "{args.old_manifest.resolve()}" --metadata-dir "{args.metadata_dir.resolve()}" --output "{output}"',
        "prepare_train": f'"{python}" "{root}\\spacenet3_stage1_prepare.py" prepare --manifest "{manifest}" --train-root "{args.train_root.resolve()}" --dev-v2-root "{args.dev_v2_root.resolve()}" --output-dir "{Path(output) / "cache"}" --splits train',
        "download_dev_v2": f'"{python}" "{root}\\spacenet3_stage1_prepare.py" download-dev-v2 --manifest "{manifest}" --data-root "{args.dev_v2_root.resolve()}" --range-downloader "{Path(root) / "download_s3_ranges.py"}" --chunks 8',
        "prepare_dev_v2": f'"{python}" "{root}\\spacenet3_stage1_prepare.py" prepare --manifest "{manifest}" --train-root "{args.train_root.resolve()}" --dev-v2-root "{args.dev_v2_root.resolve()}" --output-dir "{Path(output) / "cache"}" --splits dev_v2',
        "gate_a": f'"{python}" "{root}\\spacenet3_stage1_run.py" gate-a --train-cache "{Path(output) / "cache" / "train.npz"}" --dev-cache "{Path(output) / "cache" / "dev_v2.npz"}" --output "{output}" --seed {OPTIMIZATION_SEED}',
        "gate_b_conditional": f'"{python}" "{root}\\spacenet3_stage1_run.py" gate-b --gate-a "{Path(output) / "result.json"}" --train-cache "{Path(output) / "cache" / "train.npz"}" --dev-cache "{Path(output) / "cache" / "dev_v2.npz"}" --output "{Path(output).parent / "results_spacenet3_stage1_gateB_seed20260850"}" --seed {OPTIMIZATION_SEED}',
    }


def run_preflight(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    write_json(output / "process_start_preflight.json", {"pid": os.getpid(), "started_utc": started, "process_alive": True, "hard_timeout_seconds": 30 * 60})
    if args.python.resolve() != Path(sys.executable).resolve():
        raise RuntimeError(f"Preflight must run under the formal interpreter: expected {args.python.resolve()}, got {sys.executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the formal environment")
    configure_determinism(OPTIMIZATION_SEED)

    from spacenet3_mask_rescue import soft_cldice_loss as tb029_soft_cldice_loss
    from spacenet3_stage1_metrics import run_self_test as metrics_self_test
    from spacenet3_stage1_models import run_self_test as models_self_test
    from spacenet3_stage1_prepare import run_self_test as prepare_self_test

    prepare_self_test(args.train_root.resolve(), args.old_manifest.resolve())
    metrics_self_test()
    models_self_test()
    if inspect.getsource(tb029_soft_cldice_loss) != inspect.getsource(soft_cldice_loss):
        raise RuntimeError("soft_cldice_loss is not a verbatim reuse of TB029")

    weight = imagenet_weight_record()
    environment_text, environment = environment_record(args.python.resolve())
    write_text(output / "environment.txt", environment_text)
    commands = formal_commands(args)
    write_json(output / "formal_commands.json", commands)
    write_text(output / "formal_command.txt", commands["gate_a"] + "\n")
    code_paths = [
        Path(__file__).resolve().parent / "spacenet3_stage1_prepare.py",
        Path(__file__).resolve().parent / "spacenet3_stage1_models.py",
        Path(__file__).resolve().parent / "spacenet3_stage1_metrics.py",
        Path(__file__).resolve(),
    ]
    code_hashes = {path.name: sha256_file(path) for path in code_paths}
    parent_hashes = {
        name: sha256_file(Path(__file__).resolve().parent / name)
        for name in ("spacenet3_models.py", "spacenet3_metrics.py", "crackforest_real_gate.py", "topocoarsen_oracle.py", "spacenet3_mask_rescue.py")
    }

    manifest_result = freeze_manifest(
        args.metadata_dir.resolve(),
        args.old_manifest.resolve(),
        args.manifest.resolve(),
        args.task.resolve(),
    )
    freeze = {
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_before_dev_v2_content_access",
        "created_utc": utc_now(),
        "task": {"path": str(args.task.resolve()), "sha256": sha256_file(args.task.resolve())},
        "code_sha256": code_hashes,
        "reused_source_sha256": parent_hashes,
        "soft_cldice": {
            "source_file": str((Path(__file__).resolve().parent / "spacenet3_mask_rescue.py").resolve()),
            "source_function_sha256": function_sha256(tb029_soft_cldice_loss),
            "stage1_function_sha256": function_sha256(soft_cldice_loss),
            "verbatim_match": True,
        },
        "environment": environment,
        "environment_file_sha256": sha256_file(output / "environment.txt"),
        "imagenet_weights": weight,
        "imagenet_normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "threshold_candidates": THRESHOLDS,
        "optimization_seed": OPTIMIZATION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": {"name": "AdamW", "lr": LEARNING_RATE, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": WEIGHT_DECAY},
        "scheduler": {"name": "CosineAnnealingLR", "T_max": EPOCHS, "eta_min": 1e-5},
        "determinism": {"num_workers": 0, "cudnn_benchmark": False, "cudnn_deterministic": True, "tf32": False, "deterministic_algorithms": True, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        "formal_commands": commands,
        "manifest": manifest_result,
        "hard_timeouts_seconds": {"gate_a": HARD_TIMEOUT_GATE_A_SECONDS, "gate_b": HARD_TIMEOUT_GATE_B_SECONDS, "source_mosaic_metric": 60 * 60},
        "claim_scope": "topology_conditioned_dense_reconstruction_not_strict_k_token_compression",
        "khartoum_content_accessed": False,
        "massachusetts_test_reopened": False,
        "p1_5_disease_classification_used": False,
        "p1_checkpoint_loaded": False,
    }
    write_json(output / "run_freeze.json", freeze)
    write_json(output / "process_end_preflight.json", {"pid": os.getpid(), "started_utc": started, "ended_utc": utc_now(), "exit_code": 0, "process_alive": False})
    print(json.dumps({"status": "PREFLIGHT_FROZEN", "output": str(output), "run_freeze_sha256": sha256_file(output / "run_freeze.json"), "manifest": manifest_result, "code_sha256": code_hashes, "weight_sha256": weight["sha256"]}, ensure_ascii=False, indent=2), flush=True)


def train_mask_predictor(train: dict[str, np.ndarray], dev: dict[str, np.ndarray], output: Path, seed: int, device: torch.device) -> tuple[ResNet34UNet, list[dict[str, Any]], dict[str, Any], np.ndarray]:
    configure_determinism(seed)
    model = ResNet34UNet(pretrained=True)
    train_loader = make_loader(RoadDataset(train, augment=True), shuffle=True, seed=seed)
    dev_loader = make_loader(RoadDataset(dev, augment=False), shuffle=False, seed=seed)
    checkpoint = output / f"mask_predictor_seed{seed}.pt"
    history, training = train_one(model, train_loader, dev_loader, device, checkpoint, method=None, timeout_seconds=HARD_TIMEOUT_GATE_A_SECONDS)
    probabilities = predict(model, dev_loader, device)
    return model, history, training, probabilities


def checksum_manifest(output: Path, name: str = "all_sha256.json") -> None:
    rows = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != name):
        rows[str(path.relative_to(output))] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    write_json(output / name, rows)


def gate_a_report(result: dict[str, Any]) -> str:
    gate = result["gate"]
    metric = gate["metrics"]
    a1 = metric["A1_road_f1"]["mean"]
    a2 = metric["A2_fixed_spatial_edge_iou"]["mean"]
    a3 = metric["A3_patch_kappa"]["mean"]
    a4 = metric["A4_empty_crop_predicted_pixel_fraction"]["mean"]
    conclusion = (
        "门 A 全部通过；门 B 获得条件执行授权。"
        if gate["verdict"] == "PASS"
        else f"我们量化了 mask 质量到拓扑构造质量的传导：在 SpaceNet 3 Paris 256×256 上，road F1 = {a1:.4f} 对应固定空间边 IoU = {a2:.4f}、patch κ = {a3:.4f}，未达到预注册的拓扑条件化重建底座门。"
    )
    rows = []
    for name in ("A1_road_f1", "A2_fixed_spatial_edge_iou", "A3_patch_kappa", "A4_empty_crop_predicted_pixel_fraction"):
        item = metric[name]
        rows.append(f"| {name} | {item['mean']:.6f} | [{item['ci_low']:.6f}, {item['ci_high']:.6f}] | {item['valid_sources']} |")
    return f"""## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: {utc_now()}
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: {EXPERIMENT_ID}
- **Type**: training
- **Status**: completed
- **Command**: `{result['command']}`
- **Working Directory**: `{result['working_directory']}`
- **Duration**: {result['duration_seconds']:.1f} seconds
- **Exit Code**: 0
- **Gate A Verdict**: {gate['verdict']}
- **Selected mask threshold**: {result['selected_threshold']:.2f}

### Gate A

| Gate metric | Source mean | Bootstrap 95% CI | Effective sources |
|---|---:|---:|---:|
{chr(10).join(rows)}

### Conclusion

{conclusion}

### Claim boundary

This is topology-conditioned dense road reconstruction over a shared stride-4 spatial substrate. It is not strict K-only token reduction evidence. Khartoum content and the Massachusetts Roads official test were not accessed.

### Anomalies Detected

None during the completed formal process. Any pre-run/download anomaly is recorded separately in `实验记录`.
"""


def run_gate_a(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if not (output / "run_freeze.json").is_file():
        raise FileNotFoundError(output / "run_freeze.json")
    for required in ("result.json", "REPORT.md", "history.json", "dev_crop_metrics.csv", "dev_source_metrics.csv"):
        if (output / required).exists():
            raise FileExistsError(f"Refusing to overwrite formal Gate A artifact: {output / required}")
    started_wall = utc_now()
    started = time.perf_counter()
    write_json(output / "process_start_gate_a.json", {"pid": os.getpid(), "started_utc": started_wall, "process_alive": True, "hard_timeout_seconds": HARD_TIMEOUT_GATE_A_SECONDS})
    configure_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Gate A requires the frozen CUDA environment")
    torch.cuda.reset_peak_memory_stats(device)
    train, train_metadata = load_cache(args.train_cache.resolve())
    dev, dev_metadata = load_cache(args.dev_cache.resolve())
    if train_metadata["manifest_sha256"] != dev_metadata["manifest_sha256"]:
        raise RuntimeError("Train/dev_v2 cache manifest mismatch")
    if train_metadata["format"] != "spacenet3_native_25_crops_v2" or dev_metadata["format"] != train_metadata["format"]:
        raise RuntimeError("Gate A received a non-v2 cache")
    model, history, training, probabilities = train_mask_predictor(train, dev, output, args.seed, device)
    threshold, threshold_rows = select_threshold(dev["masks"], probabilities, dev["source_ids"])
    if threshold not in THRESHOLDS:
        raise AssertionError("Selected threshold escaped the frozen candidates")
    write_csv(output / "threshold_sweep.csv", threshold_rows)
    crop_rows, source_rows, gate = gate_a_metrics(dev["masks"], probabilities, dev["source_ids"], dev["crop_rows"], dev["crop_columns"], threshold)
    write_csv(output / "dev_crop_metrics.csv", crop_rows)
    write_csv(output / "dev_source_metrics.csv", source_rows)
    write_json(output / "history.json", {"mask_predictor": history})
    np.savez_compressed(
        output / f"dev_predictions_mask_predictor_seed{args.seed}.npz",
        probability=probabilities.astype(np.float16),
        source_ids=dev["source_ids"],
        crop_rows=dev["crop_rows"],
        crop_columns=dev["crop_columns"],
        threshold=np.asarray(threshold, dtype=np.float32),
        manifest_sha256=np.asarray(dev_metadata["manifest_sha256"]),
    )
    duration = time.perf_counter() - started
    command = json.loads((output / "formal_commands.json").read_text(encoding="utf-8"))["gate_a"]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_gate_a",
        "seed": args.seed,
        "device": str(device),
        "command": command,
        "working_directory": str(Path.cwd().resolve()),
        "duration_seconds": duration,
        "exit_code": 0,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "hard_timeout_seconds": HARD_TIMEOUT_GATE_A_SECONDS,
        "selected_threshold": threshold,
        "threshold_candidates": THRESHOLDS,
        "training": training,
        "gate": gate,
        "manifest_sha256": dev_metadata["manifest_sha256"],
        "train_cache": {"path": str(args.train_cache.resolve()), "sha256": sha256_file(args.train_cache.resolve())},
        "dev_cache": {"path": str(args.dev_cache.resolve()), "sha256": sha256_file(args.dev_cache.resolve())},
        "run_freeze": {"path": str((output / "run_freeze.json").resolve()), "sha256": sha256_file(output / "run_freeze.json")},
        "checkpoint": {"path": str((output / f"mask_predictor_seed{args.seed}.pt").resolve()), "sha256": sha256_file(output / f"mask_predictor_seed{args.seed}.pt")},
        "claim_scope": "topology_conditioned_dense_reconstruction_not_strict_k_token_compression",
        "gate_b_authorized": gate["verdict"] == "PASS",
        "khartoum_content_accessed": False,
        "massachusetts_test_reopened": False,
        "p1_checkpoint_loaded": False,
        "p1_5_disease_classification_used": False,
    }
    write_json(output / "result.json", result)
    write_text(output / "REPORT.md", gate_a_report(result))
    write_json(output / "process_end_gate_a.json", {"pid": os.getpid(), "started_utc": started_wall, "ended_utc": utc_now(), "exit_code": 0, "process_alive": False, "duration_seconds": duration, "gpu_peak_memory_bytes": result["gpu_peak_memory_bytes"]})
    checksum_manifest(output)
    print(json.dumps({"status": "GATE_A_COMPLETE", "verdict": gate["verdict"], "threshold": threshold, "metrics": gate["metrics"], "requirements": gate["requirements"], "duration_seconds": duration, "gpu_peak_memory_bytes": result["gpu_peak_memory_bytes"]}, ensure_ascii=False, indent=2), flush=True)


def infer_mask_probabilities(checkpoint: Path, arrays: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    model = ResNet34UNet(pretrained=False).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    loader = make_loader(RoadDataset(arrays, augment=False), shuffle=False, seed=OPTIMIZATION_SEED)
    return predict(model, loader, device)


def save_artifact_cache(path: Path, artifacts: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **artifacts, metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))


def load_artifact_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        artifacts = {key: archive[key] for key in archive.files if key != "metadata_json"}
    return artifacts, metadata


def paired_bootstrap(left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, float | int]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    difference = difference[np.isfinite(difference)]
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    chunk = 1000
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        stop = min(BOOTSTRAP_REPLICATES, start + chunk)
        indices = rng.integers(0, difference.size, size=(stop - start, difference.size))
        means[start:stop] = difference[indices].mean(axis=1)
    paired_sd = float(difference.std(ddof=1)) if difference.size > 1 else float("nan")
    point = float(difference.mean())
    mde = float((1.96 + 0.84) * paired_sd / math.sqrt(difference.size)) if difference.size else float("nan")
    required_sources = int(math.ceil(((1.96 + 0.84) * paired_sd / point) ** 2)) if point > 0 and paired_sd > 0 else None
    return {
        "point_estimate": point,
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "ci_width": float(np.quantile(means, 0.975) - np.quantile(means, 0.025)),
        "paired_source_sd": paired_sd,
        "conditional_mde_n_current": mde,
        "valid_sources": int(difference.size),
        "sources_required_for_point_difference_at_80pct_power": required_sources,
    }


def model_efficiency(model: torch.nn.Module, device: torch.device, sample_batch: dict[str, torch.Tensor], method: str) -> dict[str, Any]:
    model.eval().to(device)
    output: dict[str, Any] = {"parameters": parameter_count(model), "structure_hash": structure_hash(model)}
    with torch.no_grad():
        s4, t16 = model.encoder(sample_batch["image"][:1].to(device))
    output.update(
        {
            "S4_shape": list(s4.shape),
            "T16_shape": list(t16.shape),
            "skip_elements_per_sample": int(s4[0].numel()),
            "skip_branch_parameters": sum(parameter.numel() for parameter in model.encoder.s4.parameters()),
            "skip_branch_structure_hash": structure_hash(model.encoder.s4),
        }
    )
    latencies: dict[str, float] = {}
    for batch_size in (1, 8, 32):
        image = sample_batch["image"][:1].to(device).expand(batch_size, -1, -1, -1).contiguous()
        moved = {key: value[:1].to(device).expand(batch_size, *value.shape[1:]).contiguous() for key, value in sample_batch.items() if key not in {"image", "target"}}
        inputs = artifact_inputs(method, moved)
        with torch.no_grad():
            for _ in range(3):
                model(image, **inputs)
            torch.cuda.synchronize(device)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(10):
                model(image, **inputs)
            end_event.record()
            torch.cuda.synchronize(device)
            latencies[f"batch_{batch_size}_milliseconds"] = float(start_event.elapsed_time(end_event) / 10)
    output["latency"] = latencies
    try:
        from torch.profiler import ProfilerActivity, profile

        image = sample_batch["image"][:1].to(device)
        moved = {key: value[:1].to(device) for key, value in sample_batch.items() if key not in {"image", "target"}}
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as profiler:
            with torch.no_grad():
                model(image, **artifact_inputs(method, moved))
        output["partial_flops_batch1"] = int(sum(int(event.flops or 0) for event in profiler.key_averages()))
        output["flops_scope"] = "torch.profiler supported operations; partial lower bound"
    except Exception as error:
        output["partial_flops_batch1"] = None
        output["flops_scope"] = f"unavailable: {type(error).__name__}: {error}"
    output["cross_attention_update_absmax"] = model.decoder.last_cross_update_absmax
    return output


def token_crop_rows(targets: np.ndarray, probabilities: np.ndarray, source_ids: np.ndarray, rows: np.ndarray, columns: np.ndarray, model: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(targets.shape[0]):
        target = targets[index]
        prediction = probabilities[index] >= 0.5
        proxy = raster_apls_proxy(target, prediction)
        result.append(
            {
                "model": model,
                "source_id": str(source_ids[index]),
                "crop_row": int(rows[index]),
                "crop_column": int(columns[index]),
                "road_f1": binary_f1(target, prediction),
                "cldice": cldice(target, prediction),
                "predicted_foreground_fraction": float(prediction.mean()),
                "target_foreground_fraction": float(target.mean()),
                **proxy,
            }
        )
    return result


def token_source_rows(targets: np.ndarray, probabilities: np.ndarray, source_ids: np.ndarray, rows: np.ndarray, columns: np.ndarray, model: str, metric_timeout: float = 60 * 60) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    started = time.perf_counter()
    unique_sources = np.unique(source_ids)
    for position, source in enumerate(unique_sources, start=1):
        if time.perf_counter() - started > metric_timeout:
            raise TimeoutError(f"Source-mosaic metric hard timeout for {model}")
        indices = np.flatnonzero(source_ids == source)
        target = stitch_mosaic(targets[indices], rows[indices], columns[indices])
        prediction = stitch_mosaic((probabilities[indices] >= 0.5).astype(np.uint8), rows[indices], columns[indices])
        metric = source_mosaic_metrics(target, prediction)
        output.append(
            {
                "model": model,
                "source_id": str(source),
                "bidirectional_raster_path_quality": metric["bidirectional_raster_path_quality"],
                "gt_to_pred_path_score": metric["gt_to_pred_path_score"],
                "pred_to_gt_path_score": metric["pred_to_gt_path_score"],
                "arithmetic_mean_directional_path_score": (float(metric["gt_to_pred_path_score"]) + float(metric["pred_to_gt_path_score"])) / 2,
                "path_recall": metric["path_recall"],
                "disconnection_rate": metric["disconnection_rate"],
                "gt_path_pair_count": metric["gt_path_pair_count"],
                "pred_path_pair_count": metric["pred_path_pair_count"],
                "road_f1": binary_f1(target, prediction),
                "cldice": cldice(target, prediction),
                "predicted_foreground_fraction": float(prediction.mean()),
                "target_foreground_fraction": float(target.mean()),
                "crop_count": len(indices),
                "mosaic_height": 1280,
                "mosaic_width": 1280,
                "unused_right_bottom_border_pixels": 20,
            }
        )
        print(f"SOURCE_METRIC model={model} progress={position}/{len(unique_sources)} source={source} bidirectional={metric['bidirectional_raster_path_quality']:.6f}", flush=True)
    if len(output) != 48:
        raise RuntimeError(f"Gate B requires exactly 48 source records per method, found {len(output)} for {model}")
    return output


def gate_b_report(result: dict[str, Any]) -> str:
    comparisons = result["comparisons"]
    rows = [f"| {name} | {value['point_estimate']:.6f} | [{value['ci_low']:.6f}, {value['ci_high']:.6f}] | {value['paired_source_sd']:.6f} | {value['conditional_mde_n_current']:.6f} |" for name, value in comparisons.items()]
    return f"""## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: {utc_now()}
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: {EXPERIMENT_ID}-GateB
- **Type**: training
- **Status**: completed
- **Command**: `{result['command']}`
- **Working Directory**: `{result['working_directory']}`
- **Duration**: {result['duration_seconds']:.1f} seconds
- **Exit Code**: 0
- **Gate B Verdict**: {result['verdict']}

| Paired source comparison | Difference | Bootstrap 95% CI | Paired SD | Conditional MDE (n=48) |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

The inference unit is the 1280×1280 source-tile mosaic. The primary metric is `bidirectional_raster_path_quality`, a raster proxy and not official SpaceNet APLS.

The spatial skip transmits 4096 positions. This experiment is topology-conditioned dense reconstruction over a shared high-bandwidth spatial substrate, not strict K-only token reduction, and must not enter a fixed-K efficiency main table.
"""


def run_gate_b(args: argparse.Namespace) -> None:
    gate_a_path = args.gate_a.resolve()
    gate_a = json.loads(gate_a_path.read_text(encoding="utf-8"))
    if gate_a.get("gate", {}).get("verdict") != "PASS" or not gate_a.get("gate_b_authorized"):
        raise PermissionError("Gate B is not authorized because Gate A did not pass all preregistered requirements")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started_wall = utc_now()
    started = time.perf_counter()
    write_json(output / "process_start_gate_b.json", {"pid": os.getpid(), "started_utc": started_wall, "process_alive": True, "hard_timeout_seconds": HARD_TIMEOUT_GATE_B_SECONDS})
    write_text(output / "environment.txt", (gate_a_path.parent / "environment.txt").read_text(encoding="utf-8"))
    commands = json.loads((gate_a_path.parent / "formal_commands.json").read_text(encoding="utf-8"))
    write_text(output / "formal_command.txt", commands["gate_b_conditional"] + "\n")
    configure_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Gate B requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    train, train_metadata = load_cache(args.train_cache.resolve())
    dev, dev_metadata = load_cache(args.dev_cache.resolve())
    if not (
        train_metadata["manifest_sha256"]
        == dev_metadata["manifest_sha256"]
        == gate_a["manifest_sha256"]
    ):
        raise RuntimeError("Gate B manifest mismatch")
    threshold = float(gate_a["selected_threshold"])
    checkpoint = gate_a_path.parent / f"mask_predictor_seed{args.seed}.pt"
    train_probabilities = infer_mask_probabilities(checkpoint, train, device)
    with np.load(gate_a_path.parent / f"dev_predictions_mask_predictor_seed{args.seed}.npz", allow_pickle=False) as archive:
        dev_probabilities = archive["probability"].astype(np.float32)
    train_artifacts, train_artifact_metadata = build_artifacts(train_probabilities, threshold, shuffle_seed=args.seed)
    dev_artifacts, dev_artifact_metadata = build_artifacts(dev_probabilities, threshold, shuffle_seed=args.seed)
    if train_artifact_metadata["threshold"] != threshold or dev_artifact_metadata["threshold"] != threshold:
        raise AssertionError("Artifact threshold does not equal the Gate A frozen threshold")
    train_artifact_path = output / "train_mask_artifacts.npz"
    dev_artifact_path = output / "dev_mask_artifacts.npz"
    save_artifact_cache(train_artifact_path, train_artifacts, train_artifact_metadata)
    save_artifact_cache(dev_artifact_path, dev_artifacts, dev_artifact_metadata)
    input_freeze = {
        "experiment_id": EXPERIMENT_ID,
        "status": "gate_b_inputs_frozen_before_token_training",
        "gate_a_result": {"path": str(gate_a_path), "sha256": sha256_file(gate_a_path)},
        "mask_checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "selected_threshold": threshold,
        "train_artifact": {"path": str(train_artifact_path.resolve()), "sha256": sha256_file(train_artifact_path), "metadata": train_artifact_metadata},
        "dev_artifact": {"path": str(dev_artifact_path.resolve()), "sha256": sha256_file(dev_artifact_path), "metadata": dev_artifact_metadata},
        "methods": METHODS,
        "seed": args.seed,
        "K": 16,
        "decoder_threshold": 0.5,
        "khartoum_content_accessed": False,
    }
    write_json(output / "gate_b_input_freeze.json", input_freeze)
    print(json.dumps({"status": "GATE_B_INPUTS_FROZEN", "threshold": threshold, "train_artifact_sha256": input_freeze["train_artifact"]["sha256"], "dev_artifact_sha256": input_freeze["dev_artifact"]["sha256"]}, ensure_ascii=False, indent=2), flush=True)

    train_dataset = ArtifactDataset(train, train_artifacts, augment=True)
    dev_dataset = ArtifactDataset(dev, dev_artifacts, augment=False)
    histories: dict[str, Any] = {}
    model_results: dict[str, Any] = {}
    crop_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    dev_predictions: dict[str, np.ndarray] = {}
    sample_loader = make_loader(dev_dataset, shuffle=False, seed=args.seed, batch_size=32)
    sample_batch = next(iter(sample_loader))
    efficiency: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        if time.perf_counter() - started > HARD_TIMEOUT_GATE_B_SECONDS:
            raise TimeoutError(f"Gate B hard timeout before {method}")
        configure_determinism(args.seed)
        model = make_model(method)
        train_loader = make_loader(train_dataset, shuffle=True, seed=args.seed)
        dev_loader = make_loader(dev_dataset, shuffle=False, seed=args.seed)
        history, training = train_one(model, train_loader, dev_loader, device, output / f"{method}_seed{args.seed}.pt", method=method, timeout_seconds=HARD_TIMEOUT_GATE_B_SECONDS - (time.perf_counter() - started))
        if not training["convergence"]["pass"]:
            write_json(output / f"{method}_convergence_failure.json", training)
            raise RuntimeError(f"Frozen convergence gate failed for {method}: {training['convergence']}")
        probabilities = predict(model, dev_loader, device, method)
        dev_predictions[method] = probabilities
        np.savez_compressed(output / f"dev_predictions_{method}_seed{args.seed}.npz", probability=probabilities.astype(np.float16), source_ids=dev["source_ids"], crop_rows=dev["crop_rows"], crop_columns=dev["crop_columns"], threshold=np.asarray(0.5, dtype=np.float32))
        histories[method] = history
        current_crop = token_crop_rows(dev["masks"], probabilities, dev["source_ids"], dev["crop_rows"], dev["crop_columns"], method)
        current_source = token_source_rows(dev["masks"], probabilities, dev["source_ids"], dev["crop_rows"], dev["crop_columns"], method)
        crop_rows.extend(current_crop)
        source_rows.extend(current_source)
        efficiency[method] = model_efficiency(model, device, sample_batch, method)
        model_results[method] = {
            "training": training,
            "checkpoint_sha256": sha256_file(output / f"{method}_seed{args.seed}.pt"),
            "source_mean_bidirectional_raster_path_quality": float(np.mean([float(row["bidirectional_raster_path_quality"]) for row in current_source])),
            "source_mean_road_f1": float(np.mean([float(row["road_f1"]) for row in current_source])),
            "source_mean_cldice": float(np.mean([float(row["cldice"]) for row in current_source])),
        }
        print(json.dumps({"status": "GATE_B_METHOD_COMPLETE", "method": method, **model_results[method]}, ensure_ascii=False, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(output / "history.json", histories)
    write_csv(output / "dev_crop_metrics.csv", crop_rows)
    write_csv(output / "dev_source_metrics.csv", source_rows)
    write_json(output / "efficiency.json", efficiency)

    parameter_values = [int(item["parameters"]) for item in efficiency.values()]
    if max(parameter_values) / min(parameter_values) > 1.10:
        raise RuntimeError(f"Parameter count spread exceeds 10%: {parameter_values}")
    if len({item["skip_branch_structure_hash"] for item in efficiency.values()}) != 1:
        raise RuntimeError("Skip branch structures differ across methods")
    if efficiency["skip_only"]["cross_attention_update_absmax"] != 0.0:
        raise RuntimeError("skip_only cross-attention update is not exactly zero")
    if not all(efficiency[name]["cross_attention_update_absmax"] > 0 for name in METHODS if name != "skip_only"):
        raise RuntimeError("A non-skip method has an identically zero cross-attention update")

    values_by_method = {
        method: np.asarray([float(row["bidirectional_raster_path_quality"]) for row in source_rows if row["model"] == method], dtype=np.float64)
        for method in METHODS
    }
    comparison_pairs = {
        "mask_topo_minus_grid": ("mask_topo", "grid"),
        "mask_topo_minus_mask_assignment_identity": ("mask_topo", "mask_assignment_identity"),
        "mask_topo_minus_shuffled_topology": ("mask_topo", "shuffled_topology"),
        "mask_topo_minus_skip_only": ("mask_topo", "skip_only"),
        "mask_assignment_identity_minus_grid": ("mask_assignment_identity", "grid"),
    }
    comparisons = {name: paired_bootstrap(values_by_method[left], values_by_method[right], BOOTSTRAP_SEED + index) for index, (name, (left, right)) in enumerate(comparison_pairs.items())}
    core = [comparisons[name] for name in list(comparison_pairs)[:4]]
    if all(float(item["point_estimate"]) > 0 and float(item["ci_low"]) > 0 for item in core):
        verdict = "B_STRONG"
    elif all(float(item["point_estimate"]) > 0 for item in core):
        verdict = "B_WEAK_N48_NOT_DISTINGUISHABLE"
    else:
        verdict = "B_NEGATIVE"
    resolution = {
        method: {**bootstrap_summary(values.tolist(), seed=BOOTSTRAP_SEED + 100 + index), "ci_width": None}
        for index, (method, values) in enumerate(values_by_method.items())
    }
    for item in resolution.values():
        item["ci_width"] = float(item["ci_high"] - item["ci_low"])
    duration = time.perf_counter() - started
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_gate_b",
        "verdict": verdict,
        "seed": args.seed,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "command": commands["gate_b_conditional"],
        "working_directory": str(Path.cwd().resolve()),
        "duration_seconds": duration,
        "exit_code": 0,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "hard_timeout_seconds": HARD_TIMEOUT_GATE_B_SECONDS,
        "metric_hard_timeout_seconds": 60 * 60,
        "metric": "source-mosaic bidirectional_raster_path_quality (raster proxy, not official SpaceNet APLS)",
        "models": model_results,
        "efficiency": efficiency,
        "comparisons": comparisons,
        "measurement_resolution": resolution,
        "gate_b_input_freeze": {"path": str((output / "gate_b_input_freeze.json").resolve()), "sha256": sha256_file(output / "gate_b_input_freeze.json")},
        "claim_firewall": "S4 skip transmits 4096 positions; this is not strict K-only token reduction and must remain outside the fixed-K efficiency main table",
        "khartoum_content_accessed": False,
        "massachusetts_test_reopened": False,
        "p1_5_disease_classification_used": False,
    }
    write_json(output / "result.json", result)
    write_text(output / "REPORT.md", gate_b_report(result))
    write_json(output / "process_end_gate_b.json", {"pid": os.getpid(), "started_utc": started_wall, "ended_utc": utc_now(), "exit_code": 0, "process_alive": False, "duration_seconds": duration, "gpu_peak_memory_bytes": result["gpu_peak_memory_bytes"]})
    checksum_manifest(output)
    print(json.dumps({"status": "GATE_B_COMPLETE", "verdict": verdict, "comparisons": comparisons, "duration_seconds": duration}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TB-B-260803-030 stage-1 runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--python", type=Path, required=True)
    preflight.add_argument("--task", type=Path, required=True)
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--old-manifest", type=Path, required=True)
    preflight.add_argument("--metadata-dir", type=Path, required=True)
    preflight.add_argument("--train-root", type=Path, default=Path("real_data/SpaceNet3/frozen_pilot"))
    preflight.add_argument("--dev-v2-root", type=Path, default=Path("real_data/SpaceNet3/stage1_dev_v2"))
    preflight.add_argument("--output", type=Path, required=True)

    gate_a = subparsers.add_parser("gate-a")
    gate_a.add_argument("--train-cache", type=Path, required=True)
    gate_a.add_argument("--dev-cache", type=Path, required=True)
    gate_a.add_argument("--output", type=Path, required=True)
    gate_a.add_argument("--seed", type=int, default=OPTIMIZATION_SEED)

    gate_b = subparsers.add_parser("gate-b")
    gate_b.add_argument("--gate-a", type=Path, required=True)
    gate_b.add_argument("--train-cache", type=Path, required=True)
    gate_b.add_argument("--dev-cache", type=Path, required=True)
    gate_b.add_argument("--output", type=Path, required=True)
    gate_b.add_argument("--seed", type=int, default=OPTIMIZATION_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        run_preflight(args)
    elif args.command == "gate-a":
        run_gate_a(args)
    elif args.command == "gate-b":
        run_gate_b(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

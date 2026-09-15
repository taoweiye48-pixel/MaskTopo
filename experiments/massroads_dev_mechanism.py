"""TB-B-260803-040 Massachusetts Roads dev-only mechanism completion.

Purpose
-------
The four-domain mechanism master table (TB-B-260803-039, Table 2) is missing the
``Grid + predicted-mask graph`` (graph-only) cell for Massachusetts Roads. That
missing cell is the single reason the cross-domain graph-messaging interaction
claim (C-002 / focus area 2) could only be scored ``partial`` in the
result-to-claim adjudication.

This experiment fills exactly that cell as a **post-hoc, development-split-only**
mechanism diagnostic, mirroring the frozen TB-B-260803-038 Brassica design:

* It reuses the frozen TB-B-260802-021 train/dev caches, mask predictor
  probabilities and the four already-trained topology checkpoints
  (``grid_external``, ``mask_assignment_identity``, ``shuffled_topology``,
  ``mask_topo_external``). Those four checkpoints are only *reproduced* on the dev
  split, never retrained.
* It trains the one genuinely missing condition, ``grid_mask_graph`` (spatial
  grid pooling on the predicted-mask topology graph), on train and evaluates on
  dev, for the three frozen optimization seeds.
* It never downloads, reads, renders or runs inference on the sealed
  Massachusetts official test split. A hard path firewall rejects any test path.

The reported ``Grid + mask graph - Grid`` and difference-in-differences contrasts
are descriptive dev-only diagnostics. They do not modify the sealed-test main
numbers in TB-B-260802-021 and cannot be used to numerically decompose the
sealed-test ``+7.81 pp`` gain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crackforest_mechanism_ablation import build_artifacts
from crackforest_real_gate import RealDataset, balanced_metrics, model_forward
from fives_external_gate import grid_artifacts, grid_mask_graph_artifacts
from topobridge_mvp import set_seed
from topocoarsen_oracle import TopoCoarsenModel

EXPERIMENT_ID = "TB-B-260803-040"
UPSTREAM_EXPERIMENT_ID = "TB-B-260802-021"
SEEDS = (20260820, 20260821, 20260822)

# The one genuinely missing mechanism cell for Massachusetts.
NEW_MODELS = ("grid_mask_graph",)
# Conditions reused (only reproduced on dev, never retrained).
REUSED_MODELS = (
    "grid_external",
    "mask_assignment_identity",
    "shuffled_topology",
    "mask_topo_external",
)
REFERENCE_MODEL = "mask_topo_external"
ALL_MODELS = (*REUSED_MODELS, *NEW_MODELS)

CONTRASTS: dict[str, dict[str, float]] = {
    "masktopo_minus_grid": {REFERENCE_MODEL: 1.0, "grid_external": -1.0},
    "component_pooling_contribution": {
        REFERENCE_MODEL: 1.0,
        "grid_mask_graph": -1.0,
    },
    "graph_message_contribution": {
        REFERENCE_MODEL: 1.0,
        "mask_assignment_identity": -1.0,
    },
    "topology_correspondence_control": {
        REFERENCE_MODEL: 1.0,
        "shuffled_topology": -1.0,
    },
    "assignment_identity_minus_grid": {
        "mask_assignment_identity": 1.0,
        "grid_external": -1.0,
    },
    "grid_mask_graph_minus_grid": {
        "grid_mask_graph": 1.0,
        "grid_external": -1.0,
    },
    "operational_difference_in_differences": {
        REFERENCE_MODEL: 1.0,
        "mask_assignment_identity": -1.0,
        "grid_mask_graph": -1.0,
        "grid_external": 1.0,
    },
}

# Frozen TB-B-260802-021 protocol constants.
TRAIN_SIZE = 4000
DEV_SIZE = 600
TOKEN_COUNT = 8
TOKEN_DIMENSION = 64
EPOCHS = 22
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DATA_SEED = 20260802
SHUFFLE_SEED = 20260802
STATISTICS_SEED = 20260804
BOOTSTRAP_REPETITIONS = 20_000
EXPECTED_PARAMETERS = 149_123
DATASET_FINGERPRINT = "d62012f5905a7c17"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    record_root = project_root / "实验记录"
    parser = argparse.ArgumentParser(
        description="TB040 Massachusetts Roads train/dev-only mechanism diagnostic."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--seed", type=int)
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--record-root", type=Path, default=record_root)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=record_root / "TB-B-260803-040_Massachusetts_dev机制补齐协议.md",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=record_root / "TB-B-260803-040_PRESTART_FREEZE.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=record_root / "TB-B-260803-040_summary",
    )
    parser.add_argument(
        "--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS
    )
    parser.add_argument("--statistics-seed", type=int, default=STATISTICS_SEED)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def csv_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def assert_no_test_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    prohibited = (
        "test_once",
        "test_predictions",
        "test_cache",
        "test_once_raw",
        "massroads_external_summary",
        "massroads_test",
    )
    if any(item in lowered for item in prohibited):
        raise RuntimeError(f"TB040 test firewall rejected path: {path}")


def upstream_root(seed: int) -> Path:
    """The frozen TB-021 train/dev directory for a seed.

    Seed 20260820 was completed in the ``_retry1`` directory (the initial run
    stopped before endpoint results). Seeds 20260821/20260822 completed in the
    base directory. This resolver mirrors that recorded provenance.
    """

    project_root = Path(__file__).resolve().parent
    if seed == 20260820:
        return project_root / "results_massroads_seed20260820_train_dev_retry1"
    return project_root / f"results_massroads_seed{seed}_train_dev"


def validate_upstream_result(seed: int) -> tuple[dict[str, Any], Path]:
    root = upstream_root(seed).resolve()
    assert_no_test_path(root)
    result_path = root / "train_dev_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    required = {
        "experiment_id": f"TB-B-260802-021_massroads_train_dev_seed{seed}",
        "status": "completed_train_dev_frozen",
        "optimization_seed": seed,
        "data_seed": DATA_SEED,
        "token_count": TOKEN_COUNT,
        "test_accessed": False,
        "dataset_fingerprint": DATASET_FINGERPRINT,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise RuntimeError(
                f"Upstream TB021 field changed for seed {seed}: "
                f"{key}={result.get(key)!r}, expected={expected!r}."
            )
    for name in REUSED_MODELS:
        if result["model_results"][name]["parameters"] != EXPECTED_PARAMETERS:
            raise RuntimeError(
                f"Upstream {name} parameter count changed for seed {seed}."
            )
    return result, result_path


def receipt(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    assert_no_test_path(path)
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"Hash mismatch: {path}")
    return {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}


def freeze_manifest(args: argparse.Namespace) -> None:
    protocol = args.protocol.resolve()
    manifest_path = args.freeze_manifest.resolve()
    if manifest_path.exists():
        raise FileExistsError(f"Freeze manifest already exists: {manifest_path}")
    seed_receipts: dict[str, Any] = {}
    for seed in SEEDS:
        result, result_path = validate_upstream_result(seed)
        root = upstream_root(seed).resolve()
        selected = result["selected_on_validation"]
        probability_path = root / "mask_probabilities_train_dev.npz"
        caches = result["cache_files"]
        checkpoints = {
            name: result["model_results"][name] for name in REUSED_MODELS
        }
        seed_receipts[str(seed)] = {
            "train_dev_result": receipt(result_path),
            "mask_probabilities_train_dev": receipt(
                probability_path,
                result["mask_probabilities_train_dev"]["sha256"],
            ),
            "reused_checkpoints": {
                name: receipt(
                    Path(checkpoints[name]["checkpoint"]),
                    checkpoints[name]["checkpoint_sha256"],
                )
                for name in REUSED_MODELS
            },
            "recorded_dev_balanced_accuracy": {
                name: float(checkpoints[name]["best_dev_balanced_accuracy"])
                for name in REUSED_MODELS
            },
            "train_cache": receipt(
                Path(caches["train"]["path"]), caches["train"]["sha256"]
            ),
            "dev_cache": receipt(
                Path(caches["dev"]["path"]), caches["dev"]["sha256"]
            ),
            "mask_threshold": float(selected["mask_threshold"]),
            "closing_iterations": int(selected["closing_iterations"]),
        }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_BEFORE_TB040_TRAINING",
        "created_at": datetime.now().astimezone().isoformat(),
        "code": receipt(Path(__file__)),
        "protocol": receipt(protocol),
        "upstream_experiment": UPSTREAM_EXPERIMENT_ID,
        "seeds": list(SEEDS),
        "new_models": list(NEW_MODELS),
        "reused_models": list(REUSED_MODELS),
        "reference_model": REFERENCE_MODEL,
        "frozen_training": {
            "train_size": TRAIN_SIZE,
            "dev_size": DEV_SIZE,
            "dev_source_count": 14,
            "token_count": TOKEN_COUNT,
            "token_dimension": TOKEN_DIMENSION,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "data_seed": DATA_SEED,
            "shuffle_seed": SHUFFLE_SEED,
        },
        "statistics": {
            "unit": "source_image",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "statistics_seed": STATISTICS_SEED,
            "interpretation": "descriptive_posthoc_dev_only",
        },
        "test_firewall": {
            "massachusetts_test_access_authorized": False,
            "test_forward_authorized": False,
            "test_predictions_authorized": False,
            "allowed_splits": ["train", "dev"],
        },
        "seed_receipts": seed_receipts,
    }
    json_dump(manifest_path, payload, exclusive=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def load_and_validate_freeze(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.freeze_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Foreign TB040 freeze manifest.")
    if manifest.get("status") != "FROZEN_BEFORE_TB040_TRAINING":
        raise RuntimeError("TB040 freeze manifest is not in the frozen state.")
    if manifest.get("seeds") != list(SEEDS):
        raise RuntimeError("TB040 frozen seeds changed.")
    if manifest.get("new_models") != list(NEW_MODELS):
        raise RuntimeError("TB040 frozen new-model list changed.")
    if manifest.get("reused_models") != list(REUSED_MODELS):
        raise RuntimeError("TB040 frozen reused-model list changed.")
    if manifest["test_firewall"]["massachusetts_test_access_authorized"]:
        raise RuntimeError("TB040 freeze manifest incorrectly authorizes test access.")
    if sha256_file(Path(__file__).resolve()) != manifest["code"]["sha256"]:
        raise RuntimeError("TB040 code changed after the prestart freeze.")
    if sha256_file(args.protocol.resolve()) != manifest["protocol"]["sha256"]:
        raise RuntimeError("TB040 protocol changed after the prestart freeze.")
    return manifest


def load_split_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def load_seed_inputs(
    args: argparse.Namespace, seed: int, manifest: dict[str, Any]
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
]:
    result, result_path = validate_upstream_result(seed)
    frozen = manifest["seed_receipts"][str(seed)]
    if sha256_file(result_path) != frozen["train_dev_result"]["sha256"]:
        raise RuntimeError(f"TB021 train/dev result changed after freeze: {seed}.")

    caches = result["cache_files"]
    cache_paths = {
        "train": Path(caches["train"]["path"]).resolve(),
        "dev": Path(caches["dev"]["path"]).resolve(),
    }
    for split in ("train", "dev"):
        expected = frozen[f"{split}_cache"]["sha256"]
        if sha256_file(cache_paths[split]) != expected:
            raise RuntimeError(f"TB021 matched {split} cache changed: {seed}.")
    arrays = {split: load_split_arrays(path) for split, path in cache_paths.items()}

    probability_path = Path(
        result["mask_probabilities_train_dev"]["path"]
    ).resolve()
    if (
        sha256_file(probability_path)
        != frozen["mask_probabilities_train_dev"]["sha256"]
    ):
        raise RuntimeError(f"TB021 mask archive changed after freeze: {seed}.")
    assert_no_test_path(probability_path)
    with np.load(probability_path, allow_pickle=False) as archive:
        if set(archive.files) != {"train", "dev"}:
            raise RuntimeError(f"Mask archive has non-train/dev keys: {archive.files}")
        if any(archive[split].dtype != np.float16 for split in ("train", "dev")):
            raise RuntimeError(f"Mask archive is not uniformly float16: {seed}.")
        probabilities = {
            split: archive[split].astype(np.float32) for split in ("train", "dev")
        }

    reused_checkpoints: dict[str, dict[str, Any]] = {}
    for name in REUSED_MODELS:
        record = result["model_results"][name]
        checkpoint_path = Path(record["checkpoint"]).resolve()
        if (
            sha256_file(checkpoint_path)
            != frozen["reused_checkpoints"][name]["sha256"]
        ):
            raise RuntimeError(f"TB021 {name} checkpoint changed after freeze: {seed}.")
        reused_checkpoints[name] = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": record["checkpoint_sha256"],
            "recorded_dev_balanced_accuracy": float(
                record["best_dev_balanced_accuracy"]
            ),
        }

    reference = {
        "threshold": float(result["selected_on_validation"]["mask_threshold"]),
        "closing": int(result["selected_on_validation"]["closing_iterations"]),
        "upstream_result_path": result_path,
        "upstream_result_sha256": sha256_file(result_path),
        "mask_probability_path": probability_path,
        "mask_probability_sha256": sha256_file(probability_path),
        "reused_checkpoints": reused_checkpoints,
    }
    return arrays, probabilities, result, reference


def build_artifact_bank(
    arrays: dict[str, dict[str, np.ndarray]],
    probabilities: dict[str, np.ndarray],
    threshold: float,
    closing: int,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, Any]],
    dict[str, bool],
]:
    bank: dict[str, dict[str, dict[str, np.ndarray]]] = {
        name: {} for name in ALL_MODELS
    }
    diagnostics: dict[str, dict[str, Any]] = {name: {} for name in ALL_MODELS}
    checks: dict[str, bool] = {}
    for split_index, split in enumerate(("train", "dev")):
        bank["grid_external"][split], diagnostics["grid_external"][split] = (
            grid_artifacts(arrays[split], TOKEN_COUNT)
        )
        bank["grid_mask_graph"][split], diagnostics["grid_mask_graph"][split] = (
            grid_mask_graph_artifacts(
                arrays[split], probabilities[split], threshold, closing, TOKEN_COUNT
            )
        )
        for name, build_name in (
            ("mask_assignment_identity", "mask_assignment_identity"),
            ("shuffled_topology", "shuffled_topology"),
            (REFERENCE_MODEL, "mask_topo_recheck"),
        ):
            bank[name][split], diagnostics[name][split] = build_artifacts(
                arrays[split],
                probabilities[split],
                threshold,
                closing,
                TOKEN_COUNT,
                build_name,
                SHUFFLE_SEED + 100_000 * split_index,
            )
        for name in ALL_MODELS:
            artifact = bank[name][split]
            count = arrays[split]["images"].shape[0]
            checks[f"{name}_{split}_shape"] = bool(
                artifact["assignment"].shape == (count, 16, 16)
                and artifact["adjacency"].shape == (count, TOKEN_COUNT, TOKEN_COUNT)
                and artifact["reachability"].shape
                == (count, TOKEN_COUNT, TOKEN_COUNT)
            )
            checks[f"{name}_{split}_exact_k"] = bool(
                all(
                    np.unique(item).size == TOKEN_COUNT
                    for item in artifact["assignment"]
                )
            )
            checks[f"{name}_{split}_finite"] = bool(
                np.isfinite(artifact["assignment"]).all()
                and np.isfinite(artifact["adjacency"]).all()
                and np.isfinite(artifact["reachability"]).all()
            )
    identity = np.eye(TOKEN_COUNT, dtype=np.uint8)
    checks["assignment_identity_graph_is_identity_train"] = bool(
        np.all(bank["mask_assignment_identity"]["train"]["adjacency"] == identity)
        and np.all(
            bank["mask_assignment_identity"]["train"]["reachability"] == identity
        )
    )
    checks["assignment_identity_graph_is_identity_dev"] = bool(
        np.all(bank["mask_assignment_identity"]["dev"]["adjacency"] == identity)
        and np.all(
            bank["mask_assignment_identity"]["dev"]["reachability"] == identity
        )
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"TB040 artifact self-check failed: {failed}")
    return bank, diagnostics, checks


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        logits = model_forward(model, model_name, batch, device)
        truths.append(batch["connected"].numpy().astype(np.uint8))
        probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
    truth = np.concatenate(truths)
    probability = np.concatenate(probabilities)
    prediction = (probability >= 0.5).astype(np.uint8)
    return balanced_metrics(truth, prediction), prediction, probability, truth


def train_model(
    seed: int,
    name: str,
    train_dataset: RealDataset,
    dev_dataset: RealDataset,
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    set_seed(seed)
    model = TopoCoarsenModel(name, TOKEN_DIMENSION, TOKEN_COUNT).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Parameter count changed for {name}: {parameter_count}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = output / f"{name}_seed{seed}.pt"
    best = -1.0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
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
                logits = model_forward(model, name, batch, device)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batches += 1
        metrics, _, _, _ = evaluate(model, name, dev_loader, device)
        row = {
            "stage": "classifier_train_dev",
            "experiment_id": EXPERIMENT_ID,
            "dataset": "Massachusetts_Roads",
            "seed": seed,
            "model": name,
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            **{f"dev_{key}": value for key, value in metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"TB040_TRAIN model={name} seed={seed} epoch={epoch}/{EPOCHS} "
            f"loss={row['train_loss']:.4f} dev_bal={metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        if metrics["balanced_accuracy"] > best:
            best = float(metrics["balanced_accuracy"])
            torch.save(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "upstream_experiment_id": UPSTREAM_EXPERIMENT_ID,
                    "model": model.state_dict(),
                    "model_name": name,
                    "optimization_seed": seed,
                    "dev_metrics": metrics,
                    "test_accessed": False,
                    "frozen_training": {
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "token_count": TOKEN_COUNT,
                        "token_dimension": TOKEN_DIMENSION,
                    },
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics, prediction, probability, truth = evaluate(model, name, dev_loader, device)
    return (
        {
            "model": name,
            "seed": seed,
            "parameters": parameter_count,
            "best_dev_balanced_accuracy": best,
            "best_dev_metrics": metrics,
            "training_seconds": time.time() - started,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        history,
        {"truth": truth, "prediction": prediction, "probability": probability},
    )


def reproduce_reused(
    seed: int,
    name: str,
    dev_dataset: RealDataset,
    checkpoint_meta: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model = TopoCoarsenModel(name, TOKEN_DIMENSION, TOKEN_COUNT).to(device)
    checkpoint = torch.load(
        checkpoint_meta["checkpoint_path"], map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    metrics, prediction, probability, truth = evaluate(model, name, loader, device)
    recorded = checkpoint_meta["recorded_dev_balanced_accuracy"]
    reproduced = float(metrics["balanced_accuracy"])
    checks = {
        "balanced_accuracy_exact": bool(
            np.isclose(reproduced, recorded, rtol=0.0, atol=1e-9)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"TB021 {name} dev reproduction failed for seed {seed}: "
            f"reproduced={reproduced}, recorded={recorded}."
        )
    return (
        {
            "model": name,
            "seed": seed,
            "reproduced_dev_balanced_accuracy": reproduced,
            "recorded_dev_balanced_accuracy": recorded,
            "balanced_accuracy_exact": checks["balanced_accuracy_exact"],
            "checkpoint": str(checkpoint_meta["checkpoint_path"]),
            "checkpoint_sha256": checkpoint_meta["checkpoint_sha256"],
        },
        {"truth": truth, "prediction": prediction, "probability": probability},
    )


def run_seed(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    if seed not in SEEDS:
        raise ValueError("Formal TB040 training requires a frozen seed.")
    manifest = load_and_validate_freeze(args)
    record_root = args.record_root.resolve()
    expected_output = record_root / f"TB-B-260803-040_seed{seed}"
    # The output path always lives under the (Unicode) experiment-record root.
    # It is derived internally from --record-root to avoid shell code-page
    # corruption of Chinese command-line arguments; an explicit --output, if
    # provided, must match this frozen location.
    output = args.output.resolve() if args.output is not None else expected_output
    if output != expected_output:
        raise RuntimeError(
            f"Output must be the frozen experiment-record path: {expected_output}"
        )
    assert_no_test_path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    arrays, probabilities, upstream, reference = load_seed_inputs(
        args, seed, manifest
    )
    bank, diagnostics, artifact_checks = build_artifact_bank(
        arrays, probabilities, reference["threshold"], reference["closing"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TB040_DEVICE seed={seed} device={device}", flush=True)

    outputs: dict[str, dict[str, np.ndarray]] = {}
    reused_reproductions: dict[str, Any] = {}
    for name in REUSED_MODELS:
        summary, current_output = reproduce_reused(
            seed,
            name,
            RealDataset(arrays["dev"], bank[name]["dev"], augment=False),
            reference["reused_checkpoints"][name],
            device,
        )
        reused_reproductions[name] = summary
        outputs[name] = current_output
        print(
            f"TB040_REUSE_REPRODUCTION model={name} seed={seed} status=PASS "
            f"dev_bal={summary['reproduced_dev_balanced_accuracy']:.4f}",
            flush=True,
        )

    summaries: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    for name in NEW_MODELS:
        summary, rows, current_output = train_model(
            seed,
            name,
            RealDataset(arrays["train"], bank[name]["train"], augment=True),
            RealDataset(arrays["dev"], bank[name]["dev"], augment=False),
            output,
            device,
        )
        summaries[name] = summary
        outputs[name] = current_output
        history.extend(rows)

    truth_reference = outputs[REFERENCE_MODEL]["truth"]
    for name, values in outputs.items():
        if not np.array_equal(values["truth"], truth_reference):
            raise RuntimeError(f"Dev truth changed for {name}.")
    prediction_path = output / "dev_predictions.npz"
    np.savez_compressed(
        prediction_path,
        truth=truth_reference,
        source_id=arrays["dev"]["source_id"].astype(str),
        **{f"prediction_{name}": outputs[name]["prediction"] for name in ALL_MODELS},
        **{f"probability_{name}": outputs[name]["probability"] for name in ALL_MODELS},
    )
    csv_dump(output / "history.csv", history)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_train_dev_seed_posthoc_diagnostic",
        "dataset": "Massachusetts_Roads",
        "task": "derived_endpoint_connectivity",
        "confirmatory_status": "posthoc_dev_only_after_tb021_test_observed",
        "test_accessed": False,
        "test_forward_count": 0,
        "test_prediction_file_reads": 0,
        "optimization_seed": seed,
        "data_seed": DATA_SEED,
        "sample_counts": {"train": TRAIN_SIZE, "dev": DEV_SIZE},
        "source_counts": {"train": 1027, "dev": 14},
        "token_count": TOKEN_COUNT,
        "token_dimension": TOKEN_DIMENSION,
        "device": str(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest.resolve()),
        "frozen_training": manifest["frozen_training"],
        "mask_probability_storage_dtype": "float16",
        "mask_probability_loaded_compute_dtype": "float32",
        "selected_on_validation_mask_calibration": {
            "mask_threshold": reference["threshold"],
            "closing_iterations": reference["closing"],
        },
        "upstream_receipts": {
            "train_dev_result": {
                "path": str(reference["upstream_result_path"]),
                "sha256": reference["upstream_result_sha256"],
            },
            "mask_probabilities_train_dev": {
                "path": str(reference["mask_probability_path"]),
                "sha256": reference["mask_probability_sha256"],
            },
        },
        "artifact_diagnostics": diagnostics,
        "artifact_self_checks": artifact_checks,
        "reused_dev_reproductions": reused_reproductions,
        "reused_models_not_retrained": list(REUSED_MODELS),
        "new_model_results": summaries,
        "dev_predictions": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
        },
        "duration_seconds": time.time() - started,
    }
    result_path = output / "train_dev_result.json"
    json_dump(result_path, result)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": result["status"],
                "seed": seed,
                "reused_dev_balanced_accuracy": {
                    name: value["reproduced_dev_balanced_accuracy"]
                    for name, value in reused_reproductions.items()
                },
                "new_model_dev_balanced_accuracy": {
                    name: value["best_dev_balanced_accuracy"]
                    for name, value in summaries.items()
                },
                "test_accessed": False,
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(balanced_metrics(truth, prediction)["balanced_accuracy"])


def linear_effect(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    weights: dict[str, float],
    indices: np.ndarray | None = None,
) -> np.ndarray:
    if indices is None:
        indices = np.arange(truth.size)
    effects = np.zeros(predictions[REFERENCE_MODEL].shape[0], dtype=np.float64)
    for name, weight in weights.items():
        for seed_index in range(effects.size):
            effects[seed_index] += weight * balanced_accuracy(
                truth[indices], predictions[name][seed_index, indices]
            )
    return effects


def cluster_bootstrap_contrast(
    truth: np.ndarray,
    sources: np.ndarray,
    predictions: dict[str, np.ndarray],
    weights: dict[str, float],
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    unique_sources = np.unique(sources)
    source_indices = {
        source: np.flatnonzero(sources == source) for source in unique_sources
    }
    seed_effects = linear_effect(truth, predictions, weights)
    bootstrapped = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(unique_sources, size=unique_sources.size, replace=True)
        indices = np.concatenate([source_indices[source] for source in sampled])
        bootstrapped[repetition] = float(
            linear_effect(truth, predictions, weights, indices).mean()
        )
    source_effects: dict[str, float] = {}
    for source in unique_sources:
        indices = source_indices[source]
        source_effects[str(source)] = float(
            linear_effect(truth, predictions, weights, indices).mean()
        )
    values = np.asarray(list(source_effects.values()))
    return {
        "weights": weights,
        "paired_effects_by_seed": seed_effects.tolist(),
        "mean_effect": float(seed_effects.mean()),
        "sample_std_effect": float(seed_effects.std(ddof=1)),
        "source_cluster_bootstrap_95_ci": [
            float(np.quantile(bootstrapped, 0.025)),
            float(np.quantile(bootstrapped, 0.975)),
        ],
        "source_count": int(unique_sources.size),
        "sources_with_positive_mean_effect": int(np.sum(values > 0)),
        "sources_with_zero_mean_effect": int(np.sum(values == 0)),
        "sources_with_negative_mean_effect": int(np.sum(values < 0)),
        "source_effects": source_effects,
    }


def build_report(report: dict[str, Any]) -> str:
    model_labels = {
        "grid_external": "Grid",
        "grid_mask_graph": "Grid + predicted-mask graph",
        "mask_assignment_identity": "Component assignment + identity graph",
        "shuffled_topology": "Shuffled topology",
        REFERENCE_MODEL: "MaskTopo (reused TB021 dev checkpoint)",
    }
    lines = [
        "# TB-B-260803-040 Massachusetts Roads dev-only mechanism diagnostic",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-08-04",
        "- Verification Status: ANALYZED_AND_UPSTREAM_DEV_REPRODUCED",
        "- Version Label: TB040_result_v1",
        "",
        "This is a post-hoc diagnostic on the 14-source Massachusetts development "
        "split. It fills the previously missing Grid + predicted-mask graph "
        "mechanism cell. It did not load or re-evaluate Massachusetts test data "
        "and does not alter the sealed TB-B-260802-021 official-test numbers.",
        "",
        "## Development balanced accuracy",
        "",
        "| Condition | Three seeds (%) | Mean +/- sample SD (%) |",
        "|---|---:|---:|",
    ]
    for name in ALL_MODELS:
        aggregate = report["aggregates"][name]
        scores = "/".join(
            f"{100.0 * value:.2f}" for value in aggregate["scores_by_seed"]
        )
        lines.append(
            f"| {model_labels[name]} | {scores} | "
            f"{100.0 * aggregate['mean_balanced_accuracy']:.2f} +/- "
            f"{100.0 * aggregate['sample_std_balanced_accuracy']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Source-cluster descriptive contrasts",
            "",
            "Twenty thousand paired source-image cluster-bootstrap repetitions; "
            "the unit is 14 dev source images.",
            "",
            "| Contrast | Mean effect (pp) | 95% CI (pp) | Source directions +/0/- |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, result in report["contrasts"].items():
        lower, upper = result["source_cluster_bootstrap_95_ci"]
        lines.append(
            f"| {name} | {100.0 * result['mean_effect']:+.2f} | "
            f"[{100.0 * lower:+.2f}, {100.0 * upper:+.2f}] | "
            f"{result['sources_with_positive_mean_effect']}/"
            f"{result['sources_with_zero_mean_effect']}/"
            f"{result['sources_with_negative_mean_effect']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The newly added Grid + mask graph condition applies the predicted-"
            "mask topology graph on top of spatial-grid pooling, isolating the "
            "graph-messaging pathway from component-aligned pooling.",
            "- The component-pooling contrast compares MaskTopo with Grid + the same "
            "predicted-mask graph construction.",
            "- The graph-message contrast compares MaskTopo with the same component "
            "assignment and identity graph.",
            "- The shuffled contrast tests topology correspondence, not generic "
            "regularization.",
            "- Dev selected checkpoints and mask calibration and was then reused for "
            "this post-hoc diagnostic; the confidence intervals are descriptive, not "
            "a new confirmatory gate.",
            "- These values cannot be used to numerically decompose the TB021 sealed-"
            "test +7.81 pp gain, nor rewritten as natural-road APLS utility.",
            "- No value here concerns FIVES P1--P5 disease classification.",
            "",
            "## Reproducibility and firewall",
            "",
            f"- Reused-checkpoint dev reproduction: {report['all_reused_reproductions_pass']}.",
            "- New checkpoints: 3 (one condition x three seeds).",
            "- Massachusetts test reads/forwards: 0/0.",
            "- No automatic retry was performed.",
            "",
            f"Verdict: `{report['verdict']}`",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> None:
    if (
        args.bootstrap_repetitions != BOOTSTRAP_REPETITIONS
        or args.statistics_seed != STATISTICS_SEED
    ):
        raise RuntimeError("TB040 summary command differs from the frozen statistics.")
    manifest = load_and_validate_freeze(args)
    output = args.summary_output.resolve()
    assert_no_test_path(output)
    output.mkdir(parents=True, exist_ok=False)
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in ALL_MODELS}
    scores: dict[str, list[float]] = {name: [] for name in ALL_MODELS}
    truth_reference: np.ndarray | None = None
    source_reference: np.ndarray | None = None
    seed_rows: list[dict[str, Any]] = []
    run_receipts: dict[str, Any] = {}
    reused_passes: list[bool] = []
    for seed in SEEDS:
        root = args.record_root.resolve() / f"TB-B-260803-040_seed{seed}"
        result_path = root / "train_dev_result.json"
        prediction_path = root / "dev_predictions.npz"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("experiment_id") != EXPERIMENT_ID
            or result.get("status") != "completed_train_dev_seed_posthoc_diagnostic"
            or result.get("optimization_seed") != seed
            or result.get("test_accessed")
            or result.get("test_forward_count") != 0
            or result.get("test_prediction_file_reads") != 0
        ):
            raise RuntimeError(f"Incomplete or invalid TB040 seed result: {seed}.")
        if result["code_sha256"] != manifest["code"]["sha256"]:
            raise RuntimeError(f"TB040 code receipt mismatch: {seed}.")
        reused_passes.append(
            all(
                bool(value["balanced_accuracy_exact"])
                for value in result["reused_dev_reproductions"].values()
            )
        )
        with np.load(prediction_path, allow_pickle=False) as archive:
            truth = archive["truth"].astype(np.uint8)
            sources = archive["source_id"].astype(str)
            if truth_reference is None:
                truth_reference = truth
                source_reference = sources
            elif not np.array_equal(truth_reference, truth) or not np.array_equal(
                source_reference, sources
            ):
                raise RuntimeError("TB040 dev truth/source order changed across seeds.")
            for name in ALL_MODELS:
                prediction = archive[f"prediction_{name}"].astype(np.uint8)
                score = balanced_accuracy(truth, prediction)
                predictions[name].append(prediction)
                scores[name].append(score)
                if name in REUSED_MODELS:
                    recorded = result["reused_dev_reproductions"][name][
                        "reproduced_dev_balanced_accuracy"
                    ]
                else:
                    recorded = result["new_model_results"][name][
                        "best_dev_balanced_accuracy"
                    ]
                if not np.isclose(score, recorded, rtol=0.0, atol=1e-9):
                    raise RuntimeError(f"Stored TB040 metric mismatch: {seed}, {name}.")
                seed_rows.append(
                    {"seed": seed, "model": name, "dev_balanced_accuracy": score}
                )
        run_receipts[str(seed)] = {
            "result": receipt(result_path),
            "predictions": receipt(prediction_path),
            "history": receipt(root / "history.csv"),
            "checkpoints": {
                name: receipt(Path(result["new_model_results"][name]["checkpoint"]))
                for name in NEW_MODELS
            },
        }
    if truth_reference is None or source_reference is None:
        raise RuntimeError("No TB040 predictions were loaded.")
    prediction_arrays = {
        name: np.stack(values, axis=0) for name, values in predictions.items()
    }
    aggregates = {}
    for name, values in scores.items():
        current = np.asarray(values, dtype=np.float64)
        aggregates[name] = {
            "scores_by_seed": current.tolist(),
            "mean_balanced_accuracy": float(current.mean()),
            "sample_std_balanced_accuracy": float(current.std(ddof=1)),
        }
    rng = np.random.default_rng(args.statistics_seed)
    contrast_results = {
        name: cluster_bootstrap_contrast(
            truth_reference,
            source_reference,
            prediction_arrays,
            weights,
            args.bootstrap_repetitions,
            rng,
        )
        for name, weights in CONTRASTS.items()
    }
    source_rows: list[dict[str, Any]] = []
    for source in np.unique(source_reference):
        indices = np.flatnonzero(source_reference == source)
        row: dict[str, Any] = {
            "source_id": str(source),
            "sample_count": int(indices.size),
            "positive_count": int(truth_reference[indices].sum()),
            "negative_count": int(indices.size - truth_reference[indices].sum()),
        }
        for name, weights in CONTRASTS.items():
            row[name] = float(
                linear_effect(
                    truth_reference, prediction_arrays, weights, indices
                ).mean()
            )
        source_rows.append(row)
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_analyzed_posthoc_dev_only",
        "dataset": "Massachusetts_Roads",
        "task": "derived_endpoint_connectivity",
        "confirmatory_status": "posthoc_dev_only_after_tb021_test_observed",
        "test_accessed": False,
        "test_forward_count": 0,
        "test_prediction_file_reads": 0,
        "optimization_seeds": list(SEEDS),
        "sample_count": int(truth_reference.size),
        "source_count": int(np.unique(source_reference).size),
        "token_count": TOKEN_COUNT,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "statistics_seed": args.statistics_seed,
        "aggregates": aggregates,
        "contrasts": contrast_results,
        "all_reused_reproductions_pass": bool(all(reused_passes)),
        "run_receipts": run_receipts,
        "freeze_manifest": receipt(args.freeze_manifest.resolve()),
        "protocol": receipt(args.protocol.resolve()),
        "limitations": [
            "The same dev set selected checkpoints/mask calibration and supports "
            "this post-hoc diagnostic.",
            "Only 14 Massachusetts dev source images are available.",
            "No Massachusetts test condition was added or re-evaluated.",
            "The result cannot numerically decompose the TB021 sealed-test gain.",
            "The operational four-cell table is not a classical randomized factorial "
            "design.",
        ],
        "verdict": "MASSROADS_DEV_MECHANISM_DIAGNOSTIC_COMPLETE_NO_TEST_CHANGE",
    }
    result_path = output / "RESULT.json"
    json_dump(result_path, report)
    (output / "REPORT.md").write_text(build_report(report), encoding="utf-8", newline="\n")
    csv_dump(output / "seed_scores.csv", seed_rows)
    csv_dump(output / "source_effects.csv", source_rows)
    json_dump(output / "artifact_hashes.json", run_receipts)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def run_self_test(args: argparse.Namespace) -> None:
    count = 8
    arrays = {
        "images": np.zeros((count, 3, 64, 64), dtype=np.uint8),
        "connected": np.asarray([0, 1] * (count // 2), dtype=np.uint8),
        "patch_ids": np.zeros((count, 16, 16), dtype=np.int16),
        "endpoint_patches": np.asarray(
            [[0, 255], [17, 34], [1, 254], [33, 34]] * (count // 4),
            dtype=np.int16,
        ),
        "source_id": np.asarray([f"synthetic_{index // 2}" for index in range(count)]),
    }
    probabilities = np.zeros((count, 64, 64), dtype=np.float32)
    probabilities[:, 4:60, 30:34] = 0.99
    split_arrays = {"train": arrays, "dev": arrays}
    split_probabilities = {"train": probabilities, "dev": probabilities}
    bank, _, checks = build_artifact_bank(split_arrays, split_probabilities, 0.2, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_checks = {}
    for name in ALL_MODELS:
        dataset = RealDataset(arrays, bank[name]["dev"], augment=False)
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        model = TopoCoarsenModel(name, TOKEN_DIMENSION, TOKEN_COUNT).to(device)
        logits = model_forward(model, name, batch, device)
        forward_checks[f"{name}_shape"] = tuple(logits.shape) == (2,)
        forward_checks[f"{name}_finite"] = bool(torch.isfinite(logits).all())
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "self_test",
        "status": "PASS"
        if all(checks.values()) and all(forward_checks.values())
        else "FAIL",
        "device": str(device),
        "artifact_checks": checks,
        "forward_checks": forward_checks,
        "test_accessed": False,
    }
    if args.output_json is not None:
        json_dump(args.output_json.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if payload["status"] != "PASS":
        raise RuntimeError("TB040 self-test failed.")


def main() -> None:
    args = parse_args()
    if args.freeze:
        freeze_manifest(args)
    elif args.self_test:
        run_self_test(args)
    elif args.seed is not None:
        run_seed(args)
    elif args.summarize:
        summarize(args)
    else:
        raise AssertionError("No TB040 action selected.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

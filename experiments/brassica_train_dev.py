from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree

import gudhi
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from crackforest_mask_topo import (
    MaskDataset,
    build_pixel_masks,
    calibrate_mask,
    predict_masks,
    save_prediction_examples,
    segmentation_metrics,
    train_mask_model,
)
from crackforest_mechanism_ablation import build_artifacts
from crackforest_real_gate import RealDataset, balanced_metrics, model_forward, write_csv
from mask_aware_equal_supervision import (
    MASKTOPO_PARAMETER_ANCHOR,
    MATCH_TOLERANCE,
    MODELS as MASK_AWARE_MODELS,
    forward_batch as forward_mask_aware,
    make_model as make_mask_aware_model,
)
from ph_token_baselines import (
    DESCRIPTOR_DIM,
    MODELS as PH_MODELS,
    PHDescriptorDataset,
    aggregate_descriptor_diagnostics,
    forward_batch as forward_ph,
    make_model as make_ph_model,
    ph_descriptors,
)
from strong_reducer_baselines import ReducerDataset
from topobridge_mvp import set_seed
from topocoarsen_oracle import TopoCoarsenModel


EXPERIMENT_ID = "TB-B-260803-035"
FROZEN_SEEDS = {20260810, 20260811, 20260812}
FROZEN_PROTOCOL_SHA256 = "db40091a6d78d1ef980a3d93ca01f5467fb7d45ed7410b97ac233f0c030af1fb"
FROZEN_SOURCE_MANIFEST_SHA256 = "102ad36ce884045141cf5b71919667fa81a09e2d56829ebbc60022be0ae84de0"
FROZEN_EXTRACTION_MANIFEST_SHA256 = "9735c85ae3186839b6a56b9091478a654ce9a82af377cf654e401ee756dfb457"
FROZEN_DATA_GATE_SHA256 = "aefa91fb24a9ae596de55ca1d143d6006914e1e24b9bd89fe9eefbe8137a6e39"
FROZEN_SOURCE_COUNTS = {"train": 90, "dev": 14, "test": 15}
RSML_WIDTH = 8
REPORTABLE_FAMILIES = ("mask_topo_external", *MASK_AWARE_MODELS, *PH_MODELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TB035 Brassica train/dev-only neural training.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--data-gate", type=Path, default=Path("实验记录/TB-B-260803-035_data_gate/data_gate.json")
    )
    parser.add_argument(
        "--source-manifest", type=Path, default=Path("实验记录/TB-B-260803-034_source_manifest.json")
    )
    parser.add_argument(
        "--extraction-manifest",
        type=Path,
        default=Path("实验记录/TB-B-260803-034_train_dev_extraction_manifest.json"),
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("ROOTNAV_BRASSICA_UNTOUCHED_PROTOCOL_V2.md")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/RootNav2B/train_dev_raw"))
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--dev-size", type=int, default=600)
    parser.add_argument("--mask-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-ph-tokens", type=int, default=64)
    parser.add_argument("--shuffle-seed", type=int, default=20260803)
    parser.add_argument(
        "--mask-thresholds", type=float, nargs="+", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    )
    parser.add_argument("--closing-iterations", type=int, nargs="+", default=[0,1,2])
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_rsml_bytes(
    raw: bytes, shape: tuple[int, int], width: int = RSML_WIDTH
) -> tuple[np.ndarray, dict[str, Any]]:
    root = ElementTree.fromstring(raw)
    height, image_width = shape
    canvas = Image.new("L", (image_width, height), 0)
    draw = ImageDraw.Draw(canvas)
    polyline_count = 0
    point_count = 0
    minimum_x = float("inf")
    minimum_y = float("inf")
    maximum_x = float("-inf")
    maximum_y = float("-inf")
    for polyline in root.findall(".//geometry/polyline"):
        points: list[tuple[float, float]] = []
        for point in polyline.findall("point"):
            x = float(point.attrib["x"])
            y = float(point.attrib["y"])
            points.append((x, y))
            minimum_x = min(minimum_x, x)
            minimum_y = min(minimum_y, y)
            maximum_x = max(maximum_x, x)
            maximum_y = max(maximum_y, y)
        if len(points) >= 2:
            draw.line(points, fill=255, width=width, joint="curve")
            polyline_count += 1
            point_count += len(points)
    if polyline_count == 0:
        raise ValueError("RSML contains no renderable polylines.")
    if (
        minimum_x < -0.5
        or minimum_y < -0.5
        or maximum_x > image_width - 0.5
        or maximum_y > height - 0.5
    ):
        raise ValueError(
            f"RSML coordinate bounds {(minimum_x, minimum_y, maximum_x, maximum_y)} "
            f"exceed image shape {shape}."
        )
    mask = np.asarray(canvas, dtype=np.uint8) > 0
    return mask, {
        "polyline_count": polyline_count,
        "point_count": point_count,
        "coordinate_bounds": [minimum_x, minimum_y, maximum_x, maximum_y],
        "foreground_pixels": int(mask.sum()),
        "foreground_fraction": float(mask.mean()),
    }


def load_source_manifest(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "TB-B-260803-034":
        raise ValueError("Wrong frozen source manifest experiment.")
    if payload["counts"] != FROZEN_SOURCE_COUNTS:
        raise ValueError(f"Frozen source counts changed: {payload['counts']}")
    if (
        payload["test_image_member_request_count"] != 0
        or payload["test_rsml_member_request_count"] != 0
    ):
        raise ValueError("Source manifest reports test member access.")
    split_ids = {
        split: [item["source_id"] for item in payload["sources"][split]]
        for split in ("train", "dev", "test")
    }
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    if any(set(split_ids[a]) & set(split_ids[b]) for a, b in pairs):
        raise ValueError("Source splits overlap.")
    return split_ids, payload


def load_train_dev_sources(
    dataset_root: Path, split_ids: dict[str, list[str]]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    source_diagnostics: dict[str, Any] = {}
    requested_ids = split_ids["train"] + split_ids["dev"]
    test_ids = set(split_ids["test"])
    if set(requested_ids) & test_ids:
        raise ValueError("Test source entered train/dev load list.")
    for index, source_id in enumerate(requested_ids, start=1):
        directory = dataset_root / source_id
        image_path = directory / f"image_{source_id}.jpg"
        rsml_path = directory / f"image_{source_id}.rsml"
        if not image_path.exists() or not rsml_path.exists():
            raise FileNotFoundError(f"Missing train/dev pair for {source_id}")
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = np.round(
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        ).astype(np.uint8)
        mask, diagnostic = render_rsml_bytes(rsml_path.read_bytes(), gray.shape, RSML_WIDTH)
        sources[source_id] = (gray, mask)
        source_diagnostics[source_id] = {
            "image_shape": list(gray.shape),
            "image_sha256": sha256_file(image_path),
            "rsml_sha256": sha256_file(rsml_path),
            **diagnostic,
        }
        if index % 10 == 0 or index == len(requested_ids):
            print(f"BRASSICA_SOURCE_LOAD progress={index}/{len(requested_ids)}", flush=True)
    return sources, source_diagnostics


def load_arrays(data_gate: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Path]]:
    arrays = {}
    paths = {}
    for split in ("train", "dev"):
        item = data_gate["matched_caches"][split]
        path = Path(item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Matched cache hash changed for {split}.")
        with np.load(path, allow_pickle=False) as archive:
            arrays[split] = {key: archive[key] for key in archive.files}
        paths[split] = path
    return arrays, paths


def precompute_ph(
    probabilities: dict[str, np.ndarray], output_path: Path, max_tokens: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    descriptors: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "gudhi_version": gudhi.__version__,
        "max_ph_tokens": max_tokens,
        "descriptor_dimension": DESCRIPTOR_DIM,
        "filtration": "1_minus_predicted_foreground_probability",
        "homology_dimensions": [0, 1],
        "splits": {},
    }
    for split in ("train", "dev"):
        count = probabilities[split].shape[0]
        current = np.empty((count, max_tokens, DESCRIPTOR_DIM), dtype=np.float32)
        started = time.perf_counter()
        for index in range(count):
            current[index], _ = ph_descriptors(probabilities[split][index], max_tokens)
            if (index + 1) % 250 == 0 or index + 1 == count:
                print(f"BRASSICA_PH_PRECOMPUTE split={split} progress={index + 1}/{count}", flush=True)
        descriptors[split] = current
        metadata["splits"][split] = {
            "seconds": time.perf_counter() - started,
            "diagnostics": aggregate_descriptor_diagnostics(current),
        }
    np.savez_compressed(
        output_path,
        **descriptors,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    metadata["path"] = str(output_path)
    metadata["sha256"] = sha256_file(output_path)
    return descriptors, metadata


@torch.no_grad()
def evaluate_generic(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward: Callable[[nn.Module, dict[str, torch.Tensor], torch.device], torch.Tensor],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    truth_parts = []
    probability_parts = []
    for batch in loader:
        logits = forward(model, batch, device)
        truth_parts.append(batch["connected"].numpy().astype(np.uint8))
        probability_parts.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
    truth = np.concatenate(truth_parts)
    probability = np.concatenate(probability_parts)
    prediction = (probability >= 0.5).astype(np.uint8)
    return balanced_metrics(truth, prediction), prediction, probability, truth


def train_dev_model(
    name: str,
    family: str,
    args: argparse.Namespace,
    model_factory: Callable[[], nn.Module],
    train_dataset: torch.utils.data.Dataset,
    dev_dataset: torch.utils.data.Dataset,
    forward: Callable[[nn.Module, dict[str, torch.Tensor], torch.device], torch.Tensor],
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    set_seed(args.seed)
    model = model_factory().to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = output / f"{name}_seed{args.seed}.pt"
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
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = forward(model, batch, device)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batches += 1
        dev_metrics, _, _, _ = evaluate_generic(model, dev_loader, device, forward)
        row = {
            "stage": "classifier_train_dev",
            "dataset": "RootNav2_Brassica_napus",
            "seed": args.seed,
            "model": name,
            "model_family": family,
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"BRASSICA_TRAIN model={name} seed={args.seed} epoch={epoch}/{args.epochs} "
            f"loss={row['train_loss']:.4f} dev_bal={dev_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        if dev_metrics["balanced_accuracy"] > best:
            best = dev_metrics["balanced_accuracy"]
            torch.save(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "model_name": name,
                    "model_family": family,
                    "dev_metrics": dev_metrics,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    dev_metrics, prediction, probability, truth = evaluate_generic(model, dev_loader, device, forward)
    return (
        {
            "model": name,
            "model_family": family,
            "seed": args.seed,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_dev_balanced_accuracy": best,
            "best_dev_metrics": dev_metrics,
            "training_seconds": time.time() - started,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        history,
        {"truth": truth, "prediction": prediction, "probability": probability},
    )


def forward_topology(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return model_forward(model, "mask_topo_external", batch, device)


def structural_self_tests(
    args: argparse.Namespace,
    arrays: dict[str, dict[str, np.ndarray]],
    artifacts: dict[str, dict[str, np.ndarray]],
    probabilities: dict[str, np.ndarray],
    ph_descriptors_by_split: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for split in ("train", "dev"):
        checks[f"{split}_assignment_exact_k"] = bool(
            artifacts[split]["assignment"].shape == (arrays[split]["images"].shape[0], 16, 16)
            and all(np.unique(item).size == args.token_count for item in artifacts[split]["assignment"])
        )
        checks[f"{split}_float16_reloaded_finite"] = bool(np.isfinite(probabilities[split]).all())
        checks[f"{split}_ph_exact_k_available"] = bool(
            ph_descriptors_by_split[split].shape[1] >= args.token_count
        )
    image = torch.rand(2, 3, 64, 64, device=device)
    mask = torch.rand(2, 1, 64, 64, device=device)
    for name in MASK_AWARE_MODELS:
        model = make_mask_aware_model(name, args.dim, args.token_count).to(device)
        tokens, metadata = model.reduce_tokens(image, mask)
        checks[f"{name}_exact_token_shape"] = tuple(tokens.shape) == (2, 16, 64)
        checks[f"{name}_metadata_shape"] = tuple(metadata.shape) == (2, 16, 3)
        checks[f"{name}_finite"] = bool(torch.isfinite(tokens).all() and torch.isfinite(metadata).all())
        checks[f"{name}_non_topological_firewall"] = bool(
            not model.uses_connected_components
            and not model.uses_graph_construction
            and not model.uses_persistent_homology
        )
    return checks


def run_self_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.rand(2, 3, 64, 64, device=device)
    mask = torch.rand(2, 1, 64, 64, device=device)
    for name in MASK_AWARE_MODELS:
        model = make_mask_aware_model(name, 64, 16).to(device)
        tokens, metadata = model.reduce_tokens(image, mask)
        assert tokens.shape == (2, 16, 64) and metadata.shape == (2, 16, 3)
    descriptors = torch.rand(2, 16, DESCRIPTOR_DIM, device=device)
    for name in PH_MODELS:
        model = make_ph_model(name, 64, 16).to(device)
        assert model(image, mask, descriptors).shape == (2,)
    print(json.dumps({"stage": "self-test", "status": "PASS", "device": str(device)}))


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.seed not in FROZEN_SEEDS or args.output is None:
        raise ValueError("Formal training requires a frozen seed and output.")
    if (
        args.train_size != 2400
        or args.dev_size != 600
        or args.mask_epochs != 30
        or args.epochs != 22
        or args.batch_size != 32
        or args.lr != 1e-3
        or args.weight_decay != 1e-4
        or args.dim != 64
        or args.token_count != 16
        or args.data_seed != 20260810
        or args.max_ph_tokens != 64
    ):
        raise ValueError("Command differs from the frozen TB035 training protocol.")
    protocol_path = args.protocol.resolve()
    source_manifest_path = args.source_manifest.resolve()
    extraction_manifest_path = args.extraction_manifest.resolve()
    if sha256_file(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("Frozen TB035 V2 protocol changed.")
    if sha256_file(source_manifest_path) != FROZEN_SOURCE_MANIFEST_SHA256:
        raise ValueError("Frozen source manifest changed.")
    if sha256_file(extraction_manifest_path) != FROZEN_EXTRACTION_MANIFEST_SHA256:
        raise ValueError("Frozen train/dev extraction manifest changed.")
    test_marker = source_manifest_path.parent / f"{EXPERIMENT_ID}_TEST_INFERENCE_STARTED.json"
    if test_marker.exists():
        raise ValueError("Test marker exists before train/dev training.")
    data_gate_path = args.data_gate.resolve()
    if sha256_file(data_gate_path) != FROZEN_DATA_GATE_SHA256:
        raise ValueError("Frozen TB035 data-gate report changed.")
    data_gate = json.loads(data_gate_path.read_text(encoding="utf-8"))
    if data_gate["experiment_id"] != EXPERIMENT_ID or data_gate["status"] != "TRAIN_DEV_MATCHEDV3_GATE_PASS":
        raise ValueError("TB035 data gate is not a valid PASS.")
    if data_gate["test_accessed"] or not all(data_gate["gates"].values()):
        raise ValueError("Data gate reports test access or failed checks.")
    arrays, cache_paths = load_arrays(data_gate)
    split_ids, _ = load_source_manifest(source_manifest_path)
    sources, _ = load_train_dev_sources(args.dataset_root.resolve(), split_ids)
    source_masks = {key: value[1] for key, value in sources.items()}
    pixel_masks = {
        split: build_pixel_masks(current, source_masks) for split, current in arrays.items()
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"BRASSICA_DEVICE seed={args.seed} device={device}", flush=True)
    mask_model, history = train_mask_model(
        args,
        MaskDataset(arrays["train"]["images"], pixel_masks["train"], True),
        MaskDataset(arrays["dev"]["images"], pixel_masks["dev"], False),
        output,
        device,
    )
    dev_mask_metrics = segmentation_metrics(
        mask_model,
        DataLoader(
            MaskDataset(arrays["dev"]["images"], pixel_masks["dev"], False),
            batch_size=args.batch_size,
            shuffle=False,
        ),
        device,
    )
    if dev_mask_metrics["f1"] < 0.30:
        raise RuntimeError(f"Frozen dev mask F1 gate failed: {dev_mask_metrics['f1']:.4f}")
    native_probabilities = {
        split: predict_masks(mask_model, current["images"], args.batch_size, device)
        for split, current in arrays.items()
    }
    probability_path = output / "mask_probabilities_train_dev.npz"
    np.savez_compressed(
        probability_path,
        **{split: value.astype(np.float16) for split, value in native_probabilities.items()},
    )
    with np.load(probability_path, allow_pickle=False) as archive:
        storage_dtypes = {split: str(archive[split].dtype) for split in ("train", "dev")}
        if any(archive[split].dtype != np.float16 for split in ("train", "dev")):
            raise RuntimeError("Mask probability storage is not uniformly float16.")
        probabilities = {split: archive[split].astype(np.float32) for split in ("train", "dev")}
    selected, calibration_rows = calibrate_mask(
        arrays["dev"], probabilities["dev"], args.mask_thresholds, args.closing_iterations
    )
    write_csv(output / "mask_calibration.csv", calibration_rows)
    save_prediction_examples(
        arrays["dev"]["images"], pixel_masks["dev"], probabilities["dev"],
        output / "dev_mask_predictions.png",
    )
    threshold = float(selected["mask_threshold"])
    closing = int(selected["closing_iterations"])
    artifact_bank: dict[str, dict[str, np.ndarray]] = {}
    artifact_diagnostics = {}
    for split_index, split in enumerate(("train", "dev")):
        artifact_bank[split], artifact_diagnostics[split] = build_artifacts(
            arrays[split], probabilities[split], threshold, closing, args.token_count,
            "mask_topo_recheck", args.shuffle_seed + 100_000 * split_index,
        )
    ph_cache_path = output / "ph_descriptors_train_dev.npz"
    descriptors, ph_metadata = precompute_ph(probabilities, ph_cache_path, args.max_ph_tokens)
    self_test_checks = structural_self_tests(
        args, arrays, artifact_bank, probabilities, descriptors, device
    )
    if not all(self_test_checks.values()):
        raise RuntimeError(f"Reducer self-test failed: {self_test_checks}")
    summaries: dict[str, Any] = {}
    dev_outputs: dict[str, dict[str, np.ndarray]] = {}
    topology_datasets = {
        split: RealDataset(arrays[split], artifact_bank[split], augment=split == "train")
        for split in ("train", "dev")
    }
    summary, rows, dev_output = train_dev_model(
        "mask_topo_external", "topology", args,
        lambda: TopoCoarsenModel("mask_topo_external", args.dim, args.token_count),
        topology_datasets["train"], topology_datasets["dev"], forward_topology,
        output, device,
    )
    summaries["mask_topo_external"] = summary
    history.extend(rows)
    dev_outputs["mask_topo_external"] = dev_output
    reducer_datasets = {
        split: ReducerDataset(arrays[split], probabilities[split], augment=split == "train")
        for split in ("train", "dev")
    }
    for name in MASK_AWARE_MODELS:
        summary, rows, dev_output = train_dev_model(
            name, "mask_aware_non_topological", args,
            lambda name=name: make_mask_aware_model(name, args.dim, args.token_count),
            reducer_datasets["train"], reducer_datasets["dev"], forward_mask_aware,
            output, device,
        )
        summaries[name] = summary
        history.extend(rows)
        dev_outputs[name] = dev_output
    actual_anchor = summaries["mask_topo_external"]["parameters"]
    slot_parameters = summaries["mask_conditioned_slot_param_matched"]["parameters"]
    parameter_error = abs(slot_parameters - actual_anchor) / float(actual_anchor)
    if actual_anchor != MASKTOPO_PARAMETER_ANCHOR or parameter_error > MATCH_TOLERANCE:
        raise RuntimeError(
            f"Parameter gate failed: actual_anchor={actual_anchor}, frozen_anchor={MASKTOPO_PARAMETER_ANCHOR}, "
            f"slot={slot_parameters}, error={parameter_error:.4%}"
        )
    ph_datasets = {
        split: PHDescriptorDataset(
            arrays[split], probabilities[split], descriptors[split], args.token_count,
            augment=split == "train",
        )
        for split in ("train", "dev")
    }
    for name in PH_MODELS:
        summary, rows, dev_output = train_dev_model(
            name, "persistent_homology", args,
            lambda name=name: make_ph_model(name, args.dim, args.token_count),
            ph_datasets["train"], ph_datasets["dev"], forward_ph,
            output, device,
        )
        summaries[name] = summary
        history.extend(rows)
        dev_outputs[name] = dev_output
    truth_reference = dev_outputs["mask_topo_external"]["truth"]
    for name, values in dev_outputs.items():
        if not np.array_equal(values["truth"], truth_reference):
            raise RuntimeError(f"Dev truth changed for {name}.")
    np.savez_compressed(
        output / "dev_predictions.npz",
        truth=truth_reference,
        source_id=arrays["dev"]["source_id"],
        **{f"prediction_{name}": values["prediction"] for name, values in dev_outputs.items()},
        **{f"probability_{name}": values["probability"] for name, values in dev_outputs.items()},
    )
    write_csv(output / "history.csv", history)
    mask_checkpoint = output / f"mask_predictor_seed{args.seed}.pt"
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_train_dev_seed_frozen",
        "dataset": "RootNav2_Brassica_napus",
        "test_accessed": False,
        "test_image_member_request_count": 0,
        "test_rsml_member_request_count": 0,
        "optimization_seed": args.seed,
        "data_seed": args.data_seed,
        "sample_counts": {split: int(current["images"].shape[0]) for split, current in arrays.items()},
        "token_count": args.token_count,
        "token_dimension": args.dim,
        "device": str(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "data_gate": {"path": str(data_gate_path), "sha256": sha256_file(data_gate_path)},
        "selected_on_dev_mask_calibration": selected,
        "dev_mask_metrics": dev_mask_metrics,
        "mask_probability_storage_dtypes": storage_dtypes,
        "mask_probability_loaded_compute_dtype": "float32",
        "mask_checkpoint": {"path": str(mask_checkpoint), "sha256": sha256_file(mask_checkpoint)},
        "mask_probabilities": {"path": str(probability_path), "sha256": sha256_file(probability_path)},
        "ph_cache": ph_metadata,
        "artifact_diagnostics": artifact_diagnostics,
        "self_test_checks": self_test_checks,
        "parameter_gate": {
            "masktopo_parameters": actual_anchor,
            "slot_parameters": slot_parameters,
            "relative_error": parameter_error,
            "tolerance": MATCH_TOLERANCE,
            "pass": True,
        },
        "model_results": summaries,
        "matched_cache_files": {
            split: {"path": str(path), "sha256": sha256_file(path)}
            for split, path in cache_paths.items()
        },
        "dev_predictions": {
            "path": str(output / "dev_predictions.npz"),
            "sha256": sha256_file(output / "dev_predictions.npz"),
        },
    }
    result_path = output / "train_dev_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "status": result["status"],
        "seed": args.seed,
        "dev_mask_metrics": dev_mask_metrics,
        "selected_on_dev_mask_calibration": selected,
        "model_dev_balanced_accuracy": {
            name: item["best_dev_balanced_accuracy"] for name, item in summaries.items()
        },
        "parameter_gate": result["parameter_gate"],
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "test_accessed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

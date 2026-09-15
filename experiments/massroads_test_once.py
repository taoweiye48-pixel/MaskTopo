from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crackforest_mask_topo import CrackUNet, predict_masks
from crackforest_mechanism_ablation import build_artifacts
from crackforest_real_gate import (
    RealDataset,
    balanced_metrics,
    evaluate_classifier,
    threshold_shortcut,
)
from fives_external_gate import grid_artifacts
from fives_g2tm_collision import make_collision_model
from massroads_data_gate import (
    generate_candidates,
    load_sources,
    match_candidates,
    paired_stems,
    sample_invariants,
)
from massroads_train_dev import load_final_split
from strong_reducer_baselines import (
    ReducerDataset,
    evaluate as evaluate_reducer,
    make_model as make_reducer_model,
)
from summarize_crackforest_mechanism_ablation import balanced_accuracy
from summarize_strong_reducer_baselines import cluster_bootstrap
from topobridge_mvp import set_seed
from topocoarsen_oracle import TopoCoarsenModel


SEEDS = (20260820, 20260821, 20260822)
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
MODELS = (*TOPOLOGY_MODELS, *REDUCER_MODELS)
STRONG_REDUCERS = (
    "grid_external",
    "tokenlearner",
    "perceiver_resampler",
    "tome_style",
    "g2tm_fixedk_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot Massachusetts Roads official-test evaluation."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("real_data/MassachusettsRoads")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("massroads_cache"))
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("实验记录/TB-B-260802-021_train_dev_freeze_manifest.json"),
    )
    parser.add_argument(
        "--data-audit", type=Path, default=Path("results_massroads_data_audit")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results_massroads_external_summary")
    )
    parser.add_argument(
        "--marker",
        type=Path,
        default=Path("实验记录/TB-B-260802-021_TEST_INFERENCE_STARTED.json"),
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=Path("实验记录/TB-B-260802-021_test_preflight.json"),
    )
    parser.add_argument("--test-size", type=int, default=1200)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--min-component-pixels", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=20260802)
    parser.add_argument("--shuffle-seed", type=int, default=20260802)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--statistics-seed", type=int, default=20260802)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def verify_freeze_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["status"] != "TRAIN_DEV_CHECKPOINTS_FROZEN_BEFORE_TEST_ACCESS":
        raise RuntimeError("Train/dev freeze manifest has the wrong status.")
    if manifest["test_accessed"]:
        raise RuntimeError("Freeze manifest records prior test access.")
    if tuple(manifest["optimization_seeds"]) != SEEDS:
        raise RuntimeError("Optimization seed set changed.")
    if tuple(manifest["models"]) != MODELS or int(manifest["token_count"]) != 8:
        raise RuntimeError("Frozen model list or token budget changed.")
    if len(manifest["runs"]) != len(SEEDS):
        raise RuntimeError("Frozen run count changed.")
    for run, seed in zip(manifest["runs"], SEEDS):
        if int(run["seed"]) != seed:
            raise RuntimeError("Frozen run order/seed mismatch.")
        result_entry = run["result_json"]
        if sha256_file(Path(result_entry["path"])) != result_entry["sha256"]:
            raise RuntimeError(f"Frozen result JSON changed for seed {seed}.")
        if sha256_file(Path(run["mask_checkpoint"]["path"])) != run[
            "mask_checkpoint"
        ]["sha256"]:
            raise RuntimeError(f"Frozen mask checkpoint changed for seed {seed}.")
        if set(run["checkpoints"]) != set(MODELS):
            raise RuntimeError(f"Frozen model set changed for seed {seed}.")
        for model in MODELS:
            entry = run["checkpoints"][model]
            if sha256_file(Path(entry["path"])) != entry["sha256"]:
                raise RuntimeError(f"Frozen checkpoint changed: {seed}, {model}.")
    return manifest


def load_state(path: Path, expected_name: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != expected_name:
        raise RuntimeError(
            f"Checkpoint model mismatch: expected {expected_name}, "
            f"got {checkpoint.get('model_name')}."
        )
    return checkpoint


def make_reducer(name: str, dim: int, token_count: int) -> nn.Module:
    factory: Callable[[str, int, int], nn.Module] = (
        make_collision_model if name == "g2tm_fixedk_mask" else make_reducer_model
    )
    return factory(name, dim, token_count)


@torch.no_grad()
def checkpoint_preflight(
    manifest: dict[str, Any], dim: int, token_count: int
) -> dict[str, Any]:
    image = torch.zeros(2, 3, 64, 64)
    assignment = torch.arange(256).reshape(1, 16, 16) % token_count
    assignment = assignment.repeat(2, 1, 1)
    graph = torch.eye(token_count, dtype=torch.int64).unsqueeze(0).repeat(2, 1, 1)
    mask = torch.zeros(2, 1, 64, 64)
    checked = []
    for run in manifest["runs"]:
        seed = int(run["seed"])
        mask_checkpoint = torch.load(
            Path(run["mask_checkpoint"]["path"]),
            map_location="cpu",
            weights_only=False,
        )
        mask_model = CrackUNet()
        mask_model.load_state_dict(mask_checkpoint["model"])
        if mask_model(image[:, :1]).shape != (2, 64, 64):
            raise RuntimeError(f"Mask preflight shape failed for seed {seed}.")
        checked.append({"seed": seed, "model": "mask_predictor"})
        del mask_model, mask_checkpoint
        for name in MODELS:
            checkpoint = load_state(Path(run["checkpoints"][name]["path"]), name)
            if name in TOPOLOGY_MODELS:
                model = TopoCoarsenModel(name, dim, token_count)
                model.load_state_dict(checkpoint["model"])
                logits = model(image, assignment, graph, graph)
            else:
                model = make_reducer(name, dim, token_count)
                model.load_state_dict(checkpoint["model"])
                logits = model(image, mask)
            if logits.shape != (2,):
                raise RuntimeError(f"Preflight shape failed: {seed}, {name}.")
            checked.append({"seed": seed, "model": name})
            del model, checkpoint
        gc.collect()
    return {"checked_checkpoint_count": len(checked), "checks": checked}


def update_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pixel_masks_2x2(
    arrays: dict[str, np.ndarray],
    sources: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    masks = np.empty((arrays["images"].shape[0], 64, 64), dtype=np.uint8)
    for index, (source_id, box) in enumerate(
        zip(arrays["source_id"], arrays["crop_box"])
    ):
        top, left, bottom, right = (int(item) for item in box)
        crop = sources[str(source_id)][1][top:bottom, left:right]
        if crop.shape != (128, 128):
            raise RuntimeError("Unexpected native test target crop shape.")
        masks[index] = crop.reshape(64, 2, 64, 2).max(axis=(1, 3))
    return masks


def mask_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    truth = target.astype(bool)
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(tn / max(tn + fp, 1)),
        "iou": float(tp / max(tp + fp + fn, 1)),
    }


def score_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "scores_by_seed": array.tolist(),
        "mean_balanced_accuracy": float(array.mean()),
        "sample_std_balanced_accuracy": float(array.std(ddof=1)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Massachusetts Roads 一次性外部确认",
        "",
        "指标为 official test 端点连通 balanced accuracy；均值 ± 样本标准差来自三个冻结优化种子。",
        "",
        "| 方法 | 三种子 BA (%) | 均值 ± SD (%) |",
        "|---|---:|---:|",
    ]
    for name, values in report["aggregates"].items():
        seeds = ", ".join(f"{100 * value:.2f}" for value in values["scores_by_seed"])
        lines.append(
            f"| {name} | {seeds} | "
            f"{100 * values['mean_balanced_accuracy']:.2f} ± "
            f"{100 * values['sample_std_balanced_accuracy']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"最强预设 reducer：`{report['strongest_reducer']}`。",
            "",
        ]
    )
    for label, key in (
        ("最强 reducer", "strongest_reducer_contrast"),
        ("assignment-only", "assignment_identity_contrast"),
        ("shuffled topology", "shuffled_topology_contrast"),
    ):
        contrast = report[key]
        ci = contrast["source_cluster_bootstrap_95_ci"]
        lines.append(
            f"- MaskTopo 相对 {label}：{100 * contrast['mean_gain']:+.2f} pp，"
            f"source-tile cluster 95% CI [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] pp。"
        )
    direct = report["direct_structural_accuracy"]
    mask = report["heldout_mask_f1"]
    lines.extend(
        [
            "",
            f"直接结构 BA：{100 * direct['mean']:.2f} ± {100 * direct['sample_std']:.2f}%；"
            f"test mask F1：{100 * mask['mean']:.2f} ± {100 * mask['sample_std']:.2f}%。",
            "",
            "## 冻结门",
            "",
            f"结论：**{report['verdict']}**",
            "",
        ]
    )
    for name, value in report["frozen_gates"].items():
        lines.append(f"- {name}: {value}")
    lines.extend(
        [
            "",
            "本结果属于端点连通任务，不得与 P1-5 疾病四分类约 +1 pp 的结果合并表述。",
            "",
        ]
    )
    return "\n".join(lines)


def run_self_test() -> None:
    truth = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    first = np.asarray([[0, 0, 1, 1]] * 3, dtype=np.uint8)
    second = np.asarray([[0, 1, 1, 0]] * 3, dtype=np.uint8)
    sources = np.asarray(["a", "a", "b", "b"])
    contrast = cluster_bootstrap(
        truth, sources, first, second, 100, np.random.default_rng(7)
    )
    assert np.isclose(contrast["mean_gain"], 0.5)
    assert balanced_metrics(truth, first[0])["balanced_accuracy"] == 1.0
    target = np.asarray([[[0, 1], [0, 1]]], dtype=np.uint8)
    probability = target.astype(np.float32)
    assert mask_metrics(probability, target)["f1"] == 1.0
    print("MASSROADS_TEST_ONCE_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if (
        args.test_size != 1200
        or args.candidate_multiplier != 3
        or args.crop_size != 128
        or args.output_size != 64
        or args.min_component_pixels != 32
        or args.data_seed != 20260802
        or args.shuffle_seed != 20260802
        or args.token_count != 8
        or args.dim != 64
        or args.batch_size != 32
        or args.bootstrap_repetitions != 20000
    ):
        raise ValueError("Command differs from the frozen Massachusetts protocol.")
    root = args.root.resolve()
    dataset_root = resolve(root, args.dataset_root)
    cache_dir = resolve(root, args.cache_dir)
    freeze_path = resolve(root, args.freeze_manifest)
    output = resolve(root, args.output)
    marker_path = resolve(root, args.marker)
    preflight_output = resolve(root, args.preflight_output)
    audit_path = resolve(root, args.data_audit) / "data_gate.json"
    manifest = verify_freeze_manifest(freeze_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["verdict"] != "TRAIN_DEV_DATA_GATE_PASS" or audit["test_accessed"]:
        raise RuntimeError("Train/dev data audit is not clean.")

    if args.preflight:
        if (dataset_root / "test").exists():
            raise RuntimeError("Preflight must run before the test download.")
        if output.exists() or marker_path.exists() or preflight_output.exists():
            raise FileExistsError("Preflight/test output already exists.")
        checks = checkpoint_preflight(manifest, args.dim, args.token_count)
        report = {
            "experiment_id": "TB-B-260802-021_massroads_test_preflight",
            "status": "TEST_SCRIPT_PREFLIGHT_PASS_BEFORE_TEST_DOWNLOAD",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "test_accessed": False,
            "freeze_manifest": {
                "path": str(freeze_path),
                "sha256": sha256_file(freeze_path),
            },
            "test_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            **checks,
        }
        preflight_output.parent.mkdir(parents=True, exist_ok=True)
        preflight_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    if output.exists():
        raise FileExistsError("One-shot output already exists; refusing rerun.")
    if marker_path.exists():
        raise RuntimeError("One-shot test inference marker already exists; refusing rerun.")
    preflight = json.loads(preflight_output.read_text(encoding="utf-8"))
    if preflight["status"] != "TEST_SCRIPT_PREFLIGHT_PASS_BEFORE_TEST_DOWNLOAD":
        raise RuntimeError("Test preflight did not pass.")
    if preflight["test_script"]["sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Test script changed after preflight.")
    output.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stems = paired_stems(dataset_root, "test", 49)
    download_manifest_path = dataset_root / "source_metadata/test_download_manifest.json"
    download_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    if (
        download_manifest["observed_pair_count"] != 49
        or not download_manifest["dimensions_1500x1500"]
        or download_manifest["paired_stems"] != stems
    ):
        raise RuntimeError("Official test download manifest failed integrity checks.")
    split_manifest = json.loads(
        (resolve(root, args.data_audit) / "train_dev_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    train_dev_stems = set(split_manifest["official_split"]["train"]) | set(
        split_manifest["official_split"]["dev"]
    )
    source_disjoint = not bool(train_dev_stems & set(stems))
    sources, target_values = load_sources(dataset_root, "test", "test", stems)
    test_manifest_hash = sha256_file(download_manifest_path)
    test_fingerprint = test_manifest_hash[:16]
    candidate_count = args.test_size * args.candidate_multiplier
    candidate_seed = args.data_seed + 2_000_000
    candidate_path = cache_dir / (
        f"massroads_test_candidates_n{candidate_count}_seed{candidate_seed}_"
        f"{test_fingerprint}.npz"
    )
    if candidate_path.exists():
        with np.load(candidate_path) as archive:
            candidates = {key: archive[key] for key in archive.files}
    else:
        candidates = generate_candidates(
            sources, sorted(sources), candidate_count, candidate_seed, args
        )
        np.savez_compressed(candidate_path, **candidates)
    final_seed = args.data_seed + 2_077_777
    final, available_pairs = match_candidates(
        candidates,
        args.test_size,
        final_seed,
        audit["matching_edges_from_train"],
    )
    final_path = cache_dir / (
        f"massroads_test_n{args.test_size}_seed{final_seed}_matchedv1_"
        f"{test_fingerprint}.npz"
    )
    if final_path.exists():
        with np.load(final_path) as archive:
            frozen_final = {key: archive[key] for key in archive.files}
        if set(frozen_final) != set(final) or not all(
            np.array_equal(frozen_final[key], final[key]) for key in final
        ):
            raise RuntimeError("Existing deterministic test cache changed.")
        final = frozen_final
    else:
        np.savez_compressed(final_path, **final)
    invariants = sample_invariants(final)
    unique_sources = int(np.unique(final["source_id"]).size)
    data_gates = {
        "official_pair_count_and_names": len(stems) == 49,
        "dimensions_and_binary_targets": target_values == [0, 255],
        "source_disjoint_from_train_validation": source_disjoint,
        "candidate_count_exact": int(candidates["images"].shape[0]) == 3600,
        "final_count_and_balance": int(final["images"].shape[0]) == 1200
        and np.isclose(final["connected"].mean(), 0.5),
        "at_least_45_test_tiles_used": unique_sources >= 45,
        "sample_invariants": all(invariants.values()),
        "matched_pair_feasibility": available_pairs >= 600,
    }
    data_gate = {
        "experiment_id": "TB-B-260802-021_massroads_test_data_integrity",
        "status": "completed_before_model_test_inference",
        "test_accessed": True,
        "model_test_inference_started": False,
        "test_download_manifest": {
            "path": str(download_manifest_path),
            "sha256": test_manifest_hash,
        },
        "test_fingerprint": test_fingerprint,
        "target_values": target_values,
        "candidate_count": int(candidates["images"].shape[0]),
        "final_count": int(final["images"].shape[0]),
        "positive_fraction": float(final["connected"].mean()),
        "available_matched_pairs": int(available_pairs),
        "unique_sources_used": unique_sources,
        "sample_invariants": invariants,
        "gates": {name: bool(value) for name, value in data_gates.items()},
        "verdict": "TEST_DATA_INTEGRITY_PASS"
        if all(data_gates.values())
        else "TEST_DATA_INTEGRITY_FAIL_STOP_BEFORE_INFERENCE",
        "shortcut_accuracies_withheld_until_after_inference": True,
    }
    test_gate_path = output / "test_data_gate.json"
    test_gate_path.write_text(
        json.dumps(data_gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not all(data_gates.values()):
        print(json.dumps(data_gate, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(2)

    pixel_masks = pixel_masks_2x2(final, sources)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in MODELS}
    scores: dict[str, list[float]] = {name: [] for name in MODELS}
    direct_structural: list[float] = []
    heldout_mask_f1: list[float] = []
    per_seed_results: list[dict[str, Any]] = []
    probability_parts: list[np.ndarray] = []
    marker = {
        "experiment_id": "TB-B-260802-021_massroads_one_shot_test",
        "status": "TEST_INFERENCE_STARTED_DO_NOT_RERUN",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "test_fingerprint": test_fingerprint,
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "test_script_sha256": sha256_file(Path(__file__).resolve()),
        "completed_checkpoints": [],
    }
    update_marker(marker_path, marker)

    truth_reference = final["connected"].astype(np.uint8)
    for run in manifest["runs"]:
        seed = int(run["seed"])
        set_seed(seed)
        mask_checkpoint = torch.load(
            Path(run["mask_checkpoint"]["path"]),
            map_location=device,
            weights_only=False,
        )
        mask_model = CrackUNet().to(device)
        mask_model.load_state_dict(mask_checkpoint["model"])
        probability = predict_masks(mask_model, final["images"], args.batch_size, device)
        probability_parts.append(probability.astype(np.float16))
        current_mask_metrics = mask_metrics(probability, pixel_masks)
        heldout_mask_f1.append(current_mask_metrics["f1"])
        marker["completed_checkpoints"].append(
            {"seed": seed, "model": "mask_predictor"}
        )
        update_marker(marker_path, marker)
        del mask_model, mask_checkpoint

        grid, grid_diagnostics = grid_artifacts(final, args.token_count)
        artifacts: dict[str, dict[str, np.ndarray]] = {"grid_external": grid}
        diagnostics: dict[str, Any] = {"grid_external": grid_diagnostics}
        threshold = float(run["selected_on_validation"]["mask_threshold"])
        closing = int(run["selected_on_validation"]["closing_iterations"])
        for result_name, builder_name in (
            ("mask_assignment_identity", "mask_assignment_identity"),
            ("shuffled_topology", "shuffled_topology"),
            ("mask_topo_external", "mask_topo_recheck"),
        ):
            artifacts[result_name], diagnostics[result_name] = build_artifacts(
                final,
                probability,
                threshold,
                closing,
                args.token_count,
                builder_name,
                args.shuffle_seed + 200_000,
            )
        direct_structural.append(
            float(
                diagnostics["mask_topo_external"]["direct_structural_metrics"][
                    "balanced_accuracy"
                ]
            )
        )
        seed_metrics: dict[str, Any] = {}
        seed_predictions: dict[str, np.ndarray] = {}
        for name in TOPOLOGY_MODELS:
            checkpoint = load_state(Path(run["checkpoints"][name]["path"]), name)
            model = TopoCoarsenModel(name, args.dim, args.token_count).to(device)
            model.load_state_dict(checkpoint["model"])
            loader = DataLoader(
                RealDataset(final, artifacts[name]),
                batch_size=args.batch_size,
                shuffle=False,
            )
            metrics, prediction, truth = evaluate_classifier(
                model, name, loader, device, return_predictions=True
            )
            if prediction is None or truth is None or not np.array_equal(
                truth, truth_reference
            ):
                raise RuntimeError(f"Test truth/prediction mismatch: {seed}, {name}.")
            seed_metrics[name] = metrics
            seed_predictions[name] = prediction
            predictions[name].append(prediction)
            scores[name].append(float(metrics["balanced_accuracy"]))
            marker["completed_checkpoints"].append({"seed": seed, "model": name})
            update_marker(marker_path, marker)
            del model, checkpoint, loader

        reducer_dataset = ReducerDataset(final, probability, augment=False)
        for name in REDUCER_MODELS:
            checkpoint = load_state(Path(run["checkpoints"][name]["path"]), name)
            model = make_reducer(name, args.dim, args.token_count).to(device)
            model.load_state_dict(checkpoint["model"])
            loader = DataLoader(
                reducer_dataset, batch_size=args.batch_size, shuffle=False
            )
            metrics, prediction, truth = evaluate_reducer(
                model, loader, device, return_predictions=True
            )
            if prediction is None or truth is None or not np.array_equal(
                truth, truth_reference
            ):
                raise RuntimeError(f"Test truth/prediction mismatch: {seed}, {name}.")
            seed_metrics[name] = metrics
            seed_predictions[name] = prediction
            predictions[name].append(prediction)
            scores[name].append(float(metrics["balanced_accuracy"]))
            marker["completed_checkpoints"].append({"seed": seed, "model": name})
            update_marker(marker_path, marker)
            del model, checkpoint, loader

        seed_result = {
            "seed": seed,
            "mask_threshold": threshold,
            "closing_iterations": closing,
            "heldout_mask_metrics": current_mask_metrics,
            "artifact_diagnostics": diagnostics,
            "model_test_metrics": seed_metrics,
        }
        per_seed_results.append(seed_result)
        (output / f"seed{seed}_result.json").write_text(
            json.dumps(seed_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            output / f"seed{seed}_predictions.npz",
            truth=truth_reference,
            source_id=final["source_id"],
            **seed_predictions,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    np.savez_compressed(
        output / "mask_probabilities_test.npz",
        seeds=np.asarray(SEEDS),
        probabilities=np.stack(probability_parts),
    )
    stacked = {name: np.stack(values) for name, values in predictions.items()}
    aggregates = {name: score_summary(values) for name, values in scores.items()}
    strongest = max(
        STRONG_REDUCERS,
        key=lambda name: aggregates[name]["mean_balanced_accuracy"],
    )
    rng = np.random.default_rng(args.statistics_seed)
    strongest_contrast = cluster_bootstrap(
        truth_reference,
        final["source_id"].astype(str),
        stacked["mask_topo_external"],
        stacked[strongest],
        args.bootstrap_repetitions,
        rng,
    )
    identity_contrast = cluster_bootstrap(
        truth_reference,
        final["source_id"].astype(str),
        stacked["mask_topo_external"],
        stacked["mask_assignment_identity"],
        args.bootstrap_repetitions,
        rng,
    )
    shuffled_contrast = cluster_bootstrap(
        truth_reference,
        final["source_id"].astype(str),
        stacked["mask_topo_external"],
        stacked["shuffled_topology"],
        args.bootstrap_repetitions,
        rng,
    )

    train, _ = load_final_split(
        cache_dir, "train", 4000, str(audit["dataset_fingerprint"])
    )
    shortcut_accuracy = {
        "endpoint_distance": threshold_shortcut(
            train["endpoint_distance"],
            train["connected"],
            final["endpoint_distance"],
            final["connected"],
        ),
        "mean_luminance": threshold_shortcut(
            train["images"][:, 0].mean(axis=(1, 2)),
            train["connected"],
            final["images"][:, 0].mean(axis=(1, 2)),
            final["connected"],
        ),
        "road_fraction": threshold_shortcut(
            train["road_fraction"],
            train["connected"],
            final["road_fraction"],
            final["connected"],
        ),
    }
    direct = np.asarray(direct_structural, dtype=np.float64)
    mask_f1 = np.asarray(heldout_mask_f1, dtype=np.float64)
    gates = {
        "mean_gain_over_strongest_reducer_at_least_2pp": strongest_contrast[
            "mean_gain"
        ]
        >= 0.02,
        "all_seed_gains_over_strongest_reducer_positive": all(
            value > 0 for value in strongest_contrast["paired_gains_by_seed"]
        ),
        "strongest_reducer_cluster_ci_lower_above_zero": strongest_contrast[
            "source_cluster_bootstrap_95_ci"
        ][0]
        > 0,
        "mean_gain_over_assignment_only_at_least_1pp": identity_contrast["mean_gain"]
        >= 0.01,
        "assignment_only_cluster_ci_lower_above_zero": identity_contrast[
            "source_cluster_bootstrap_95_ci"
        ][0]
        > 0,
        "mean_gain_over_shuffled_topology_at_least_2pp": shuffled_contrast[
            "mean_gain"
        ]
        >= 0.02,
        "shuffled_topology_cluster_ci_lower_above_zero": shuffled_contrast[
            "source_cluster_bootstrap_95_ci"
        ][0]
        > 0,
        "mean_direct_structural_accuracy_at_least_70pct": direct.mean() >= 0.70,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    report = {
        "experiment_id": "TB-B-260802-021_massroads_external_confirmation",
        "status": "completed_one_shot_test",
        "dataset": "Massachusetts Roads Dataset",
        "task": "derived_endpoint_connectivity",
        "test_fingerprint": test_fingerprint,
        "test_crop_count": int(truth_reference.size),
        "test_source_tile_count": int(np.unique(final["source_id"]).size),
        "optimization_seeds": list(SEEDS),
        "token_count": args.token_count,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "official_test_used_for_model_selection": False,
        "inference_uses_ground_truth_test_mask": False,
        "test_model_performance_previously_observed_before_this_run": False,
        "aggregates": aggregates,
        "strongest_reducer_definition": (
            "highest frozen three-seed mean official-test balanced accuracy among "
            "Grid, TokenLearner, Perceiver, ToMe-style, and matched G2TM"
        ),
        "strongest_reducer": strongest,
        "strongest_reducer_contrast": strongest_contrast,
        "assignment_identity_contrast": identity_contrast,
        "shuffled_topology_contrast": shuffled_contrast,
        "direct_structural_accuracy": {
            "scores_by_seed": direct.tolist(),
            "mean": float(direct.mean()),
            "sample_std": float(direct.std(ddof=1)),
        },
        "heldout_mask_f1": {
            "scores_by_seed": mask_f1.tolist(),
            "mean": float(mask_f1.mean()),
            "sample_std": float(mask_f1.std(ddof=1)),
        },
        "test_shortcut_accuracy_reported_after_inference": shortcut_accuracy,
        "test_data_gate": data_gate,
        "per_seed": per_seed_results,
        "frozen_gates": gates,
        "verdict": "MASSROADS_EXTERNAL_CONFIRMATION_PASS"
        if all(gates.values())
        else "MASSROADS_EXTERNAL_CONFIRMATION_FAIL",
        "claim_boundary": (
            "This is an endpoint-connectivity result and must not be merged with "
            "the P1-5 four-class disease result of approximately +1 pp."
        ),
        "limitations": [
            "Endpoint labels are derived from dense raster road masks.",
            "Rasterized crossings can merge overpasses and crop boundaries can split roads.",
            "The matched 50/50 class prevalence is not deployment prevalence.",
            "G2TM is a matched fixed-budget adaptation, not an official reproduction.",
        ],
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(markdown_report(report), encoding="utf-8")
    with (output / "source_data.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "model", "balanced_accuracy", "parameters"),
        )
        writer.writeheader()
        for run in manifest["runs"]:
            seed = int(run["seed"])
            seed_index = SEEDS.index(seed)
            for name in MODELS:
                writer.writerow(
                    {
                        "seed": seed,
                        "model": name,
                        "balanced_accuracy": scores[name][seed_index],
                        "parameters": run["checkpoints"][name]["parameters"],
                    }
                )
    marker["status"] = "TEST_INFERENCE_COMPLETED_DO_NOT_RERUN"
    marker["completed_utc"] = datetime.now(timezone.utc).isoformat()
    marker["result"] = {"path": str(result_path), "sha256": sha256_file(result_path)}
    update_marker(marker_path, marker)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from crackforest_mask_topo import (
    MaskDataset,
    build_pixel_masks,
    calibrate_mask,
    predict_masks,
    save_prediction_examples,
    train_mask_model,
)
from crackforest_mechanism_ablation import (
    MaskFeatureDataset,
    build_artifacts,
    train_mask_feature_grid,
)
from crackforest_real_gate import (
    RealDataset,
    data_diagnostics,
    fixed_artifacts,
    generate_arrays,
    make_sample,
    render_examples,
    train_classifier,
    write_csv,
)


MODEL_KEYS = (
    "grid_external",
    "mask_feature_grid",
    "mask_assignment_identity",
    "grid_mask_graph",
    "shuffled_topology",
    "mask_topo_external",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot external MaskTopo confirmation on DeepCrack."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("real_data/DeepCrack/dataset/extracted"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("deepcrack_cache"))
    parser.add_argument("--output", type=Path, default=Path("results_deepcrack"))
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--dev-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=1200)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--mask-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--shuffle-seed", type=int, default=20260731)
    parser.add_argument(
        "--mask-thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument(
        "--closing-iterations", type=int, nargs="+", default=[0, 1, 2]
    )
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def dataset_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for directory in ("train_img", "train_lab", "test_img", "test_lab"):
        for path in sorted((root / directory).iterdir()):
            if path.is_file():
                digest.update(directory.encode())
                digest.update(path.name.encode())
                digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:16]


def source_manifest(root: Path, split_seed: int) -> dict[str, list[str]]:
    train_images = sorted(path.stem for path in (root / "train_img").glob("*.jpg"))
    train_labels = sorted(path.stem for path in (root / "train_lab").glob("*.png"))
    test_images = sorted(path.stem for path in (root / "test_img").glob("*.jpg"))
    test_labels = sorted(path.stem for path in (root / "test_lab").glob("*.png"))
    if train_images != train_labels or len(train_images) != 300:
        raise ValueError("DeepCrack train image/label pairs are incomplete.")
    if test_images != test_labels or len(test_images) != 237:
        raise ValueError("DeepCrack test image/label pairs are incomplete.")
    rng = np.random.default_rng(split_seed)
    shuffled = np.asarray(train_images)
    rng.shuffle(shuffled)
    return {
        "train": sorted(f"train:{stem}" for stem in shuffled[:240]),
        "dev": sorted(f"train:{stem}" for stem in shuffled[240:]),
        "test": sorted(f"test:{stem}" for stem in test_images),
    }


def load_sources(
    root: Path, source_ids: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sources = {}
    for source_id in source_ids:
        split, stem = source_id.split(":", maxsplit=1)
        image_path = root / f"{split}_img" / f"{stem}.jpg"
        label_path = root / f"{split}_lab" / f"{stem}.png"
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        gray = np.round(
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        ).astype(np.uint8)
        mask = np.asarray(Image.open(label_path).convert("L")) >= 128
        if gray.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch for {source_id}.")
        sources[source_id] = (gray, mask)
    return sources


def load_or_generate_split(
    cache_dir: Path,
    split: str,
    sources: dict[str, tuple[np.ndarray, np.ndarray]],
    source_ids: list[str],
    count: int,
    seed: int,
    args: argparse.Namespace,
    fingerprint: str,
) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (
        f"deepcrack_{split}_n{count}_seed{seed}_"
        f"crop{args.crop_size}_matchedv2_{fingerprint}.npz"
    )
    if path.exists():
        with np.load(path) as archive:
            return {key: archive[key] for key in archive.files}
    candidate_count = count * args.candidate_multiplier
    candidates = generate_arrays(
        sources, source_ids, candidate_count, seed, args
    )
    arrays = match_shortcuts(candidates, count, seed + 77_777)
    np.savez_compressed(path, **arrays)
    return arrays


def match_shortcuts(
    candidates: dict[str, np.ndarray],
    count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if count % 2:
        raise ValueError("Matched sample count must be even.")
    labels = candidates["connected"].astype(np.uint8)
    if int(np.sum(labels == 0)) < count // 2 or int(np.sum(labels == 1)) < count // 2:
        raise ValueError("Candidate pool is not large enough for matching.")
    intensity = candidates["images"][:, 0].mean(axis=(1, 2))
    distance = candidates["endpoint_distance"].astype(np.float64)
    rng = np.random.default_rng(seed)
    pair_bank: list[tuple[int, int]] = []
    selected_grid = None
    for intensity_bins, distance_bins in ((12, 6), (8, 5), (6, 4), (4, 3)):
        intensity_edges = np.unique(
            np.quantile(intensity, np.linspace(0, 1, intensity_bins + 1))
        )
        distance_edges = np.unique(
            np.quantile(distance, np.linspace(0, 1, distance_bins + 1))
        )
        intensity_id = np.digitize(
            intensity, intensity_edges[1:-1], right=True
        )
        distance_id = np.digitize(distance, distance_edges[1:-1], right=True)
        pair_bank = []
        for intensity_bin in np.unique(intensity_id):
            for distance_bin in np.unique(distance_id):
                member = (intensity_id == intensity_bin) & (
                    distance_id == distance_bin
                )
                negative = np.flatnonzero(member & (labels == 0))
                positive = np.flatnonzero(member & (labels == 1))
                rng.shuffle(negative)
                rng.shuffle(positive)
                matched = min(negative.size, positive.size)
                pair_bank.extend(
                    (int(negative[index]), int(positive[index]))
                    for index in range(matched)
                )
        if len(pair_bank) >= count // 2:
            selected_grid = {
                "intensity_bins": intensity_bins,
                "distance_bins": distance_bins,
                "available_pairs": len(pair_bank),
            }
            break
    if selected_grid is None:
        raise RuntimeError("Could not construct enough shortcut-matched pairs.")
    chosen_pairs = rng.choice(
        len(pair_bank), size=count // 2, replace=False
    )
    selected = np.asarray(
        [
            item
            for pair_index in chosen_pairs
            for item in pair_bank[int(pair_index)]
        ],
        dtype=np.int64,
    )
    rng.shuffle(selected)
    output = {key: value[selected] for key, value in candidates.items()}
    output["matching_intensity_bins"] = np.full(
        count, selected_grid["intensity_bins"], dtype=np.int16
    )
    output["matching_distance_bins"] = np.full(
        count, selected_grid["distance_bins"], dtype=np.int16
    )
    output["matching_available_pairs"] = np.full(
        count, selected_grid["available_pairs"], dtype=np.int32
    )
    if not np.isclose(output["connected"].mean(), 0.5):
        raise RuntimeError("Matching changed class balance.")
    return output


def load_all_arrays(
    args: argparse.Namespace,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, list[str]],
    dict[str, tuple[np.ndarray, np.ndarray]],
    str,
]:
    root = args.dataset_root.resolve()
    manifest = source_manifest(root, args.split_seed)
    source_ids = sorted(set().union(*[set(items) for items in manifest.values()]))
    sources = load_sources(root, source_ids)
    fingerprint = dataset_fingerprint(root)
    arrays = {
        "train": load_or_generate_split(
            args.cache_dir.resolve(),
            "train",
            sources,
            manifest["train"],
            args.train_size,
            args.data_seed,
            args,
            fingerprint,
        ),
        "dev": load_or_generate_split(
            args.cache_dir.resolve(),
            "dev",
            sources,
            manifest["dev"],
            args.dev_size,
            args.data_seed + 1_000_000,
            args,
            fingerprint,
        ),
        "test": load_or_generate_split(
            args.cache_dir.resolve(),
            "test",
            sources,
            manifest["test"],
            args.test_size,
            args.data_seed + 2_000_000,
            args,
            fingerprint,
        ),
    }
    return arrays, manifest, sources, fingerprint


def run_self_test(root: Path) -> None:
    manifest = source_manifest(root, 123)
    assert [len(manifest[key]) for key in ("train", "dev", "test")] == [
        240,
        60,
        237,
    ]
    assert not (set(manifest["train"]) & set(manifest["dev"]))
    assert not (set(manifest["train"]) & set(manifest["test"]))
    ids = manifest["train"][:10]
    sources = load_sources(root, ids)
    positive = make_sample(sources, ids, 1, 1234, 64, 64, 12)
    negative = make_sample(sources, ids, 0, 1234, 64, 64, 12)
    assert positive["connected"] == 1
    assert negative["connected"] == 0
    assert positive["image"].shape == (3, 64, 64)
    namespace = argparse.Namespace(
        crop_size=64,
        output_size=64,
        min_component_pixels=12,
    )
    candidates = generate_arrays(sources, ids, 30, 4321, namespace)
    matched = match_shortcuts(candidates, 10, 9876)
    assert matched["images"].shape == (10, 3, 64, 64)
    assert np.isclose(matched["connected"].mean(), 0.5)
    print("DEEPCRACK_EXTERNAL_GATE_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args.dataset_root.resolve())
        return
    if args.crop_size != 64 or args.output_size != 64 or args.token_count != 16:
        raise ValueError("External gate requires native 64x64 crops and 16 tokens.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays, manifest, sources, fingerprint = load_all_arrays(args)
    diagnostics = data_diagnostics(arrays, manifest)
    diagnostic_gates = {
        "source_split_overlap_absent": all(
            not values for values in diagnostics["source_overlap"].values()
        ),
        "endpoint_distance_shortcut_at_most_55pct": (
            diagnostics["endpoint_distance_shortcut_test_accuracy"] <= 0.55
        ),
        "mean_intensity_shortcut_at_most_55pct": (
            diagnostics["mean_intensity_shortcut_test_accuracy"] <= 0.55
        ),
        "at_least_90pct_official_test_sources_used": (
            diagnostics["unique_sources_used"]["test"] >= 214
        ),
    }
    data_report = {
        "dataset": "DeepCrack",
        "dataset_fingerprint": fingerprint,
        "official_source_counts": {
            split: len(items) for split, items in manifest.items()
        },
        "diagnostics": diagnostics,
        "diagnostic_gates": diagnostic_gates,
    }
    (output / "data_audit.json").write_text(
        json.dumps(data_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    render_examples(arrays["test"], output / "test_examples.png")
    if args.data_only:
        print(json.dumps(data_report, ensure_ascii=False, indent=2), flush=True)
        return
    if not all(diagnostic_gates.values()):
        raise RuntimeError("DeepCrack data gate failed; training was not started.")

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
            mask_model, current["images"], args.batch_size, device
        )
        for split, current in arrays.items()
    }
    np.savez_compressed(
        output / "mask_probabilities.npz",
        **{
            split: probability.astype(np.float16)
            for split, probability in probabilities.items()
        },
    )
    selected, calibration_rows = calibrate_mask(
        arrays["dev"],
        probabilities["dev"],
        args.mask_thresholds,
        args.closing_iterations,
    )
    write_csv(output / "mask_calibration.csv", calibration_rows)
    threshold = float(selected["mask_threshold"])
    closing = int(selected["closing_iterations"])
    save_prediction_examples(
        arrays["test"]["images"],
        pixel_masks["test"],
        probabilities["test"],
        output / "mask_predictions.png",
    )

    grid_artifacts = {
        split: fixed_artifacts(current, args.token_count, "grid")[0]
        for split, current in arrays.items()
    }
    artifact_modes = {
        "mask_assignment_identity": "mask_assignment_identity",
        "grid_mask_graph": "grid_mask_graph",
        "shuffled_topology": "shuffled_topology",
        "mask_topo_external": "mask_topo_recheck",
    }
    artifact_bank: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    artifact_diagnostics: dict[str, Any] = {}
    for result_name, builder_name in artifact_modes.items():
        artifact_bank[result_name] = {}
        artifact_diagnostics[result_name] = {}
        for split_index, split in enumerate(("train", "dev", "test")):
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

    summaries = {}
    predictions = {}
    truth_reference: np.ndarray | None = None
    for model_key in MODEL_KEYS:
        print(f"EXTERNAL_START model={model_key}", flush=True)
        current_artifacts = (
            grid_artifacts
            if model_key in {"grid_external", "mask_feature_grid"}
            else artifact_bank[model_key]
        )
        if model_key == "mask_feature_grid":
            summary, current_history, prediction, truth = (
                train_mask_feature_grid(
                    args,
                    MaskFeatureDataset(
                        arrays["train"],
                        current_artifacts["train"],
                        probabilities["train"],
                        True,
                    ),
                    MaskFeatureDataset(
                        arrays["dev"],
                        current_artifacts["dev"],
                        probabilities["dev"],
                        False,
                    ),
                    MaskFeatureDataset(
                        arrays["test"],
                        current_artifacts["test"],
                        probabilities["test"],
                        False,
                    ),
                    output,
                    device,
                )
            )
        else:
            summary, current_history, prediction, truth = train_classifier(
                model_key,
                args,
                RealDataset(
                    arrays["train"], current_artifacts["train"], augment=True
                ),
                RealDataset(arrays["dev"], current_artifacts["dev"]),
                RealDataset(arrays["test"], current_artifacts["test"]),
                output,
                device,
            )
        if truth_reference is None:
            truth_reference = truth
        elif not np.array_equal(truth_reference, truth):
            raise RuntimeError("Held-out truth changed between external models.")
        summaries[model_key] = summary
        predictions[model_key] = prediction
        history.extend(current_history)
        print(
            f"EXTERNAL_DONE model={model_key} "
            f"test_bal={summary['test_metrics']['balanced_accuracy']:.4f}",
            flush=True,
        )
    if truth_reference is None:
        raise RuntimeError("No external models were run.")
    write_csv(output / "history.csv", history)
    np.savez_compressed(
        output / "heldout_predictions.npz",
        truth=truth_reference,
        source_id=arrays["test"]["source_id"],
        **predictions,
    )
    result = {
        "experiment_id": "deepcrack_masktopo_external_confirmation",
        "status": "completed",
        "dataset": {
            "name": "DeepCrack",
            "fingerprint": fingerprint,
            "license": "non-commercial research and educational purposes only",
            "official_source_counts": {
                split: len(items) for split, items in manifest.items()
            },
        },
        "optimization_seed": args.seed,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "data_diagnostics": diagnostics,
        "selected_on_dev": selected,
        "model_results": summaries,
        "artifact_diagnostics": artifact_diagnostics,
        "official_test_used_for_model_selection": False,
        "inference_uses_ground_truth_mask": False,
        "test_model_performance_previously_observed_before_this_run": False,
        "test_aggregate_shortcut_audit_performed_before_this_run": True,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

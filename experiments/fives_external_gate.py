from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
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
from crackforest_mechanism_ablation import (
    MaskFeatureDataset,
    build_artifacts,
    predicted_groups,
    train_mask_feature_grid,
)
from crackforest_real_gate import (
    RealDataset,
    balanced_metrics,
    coarse_graph8,
    data_diagnostics,
    generate_arrays,
    render_examples,
    structural_prediction,
    train_classifier,
    write_csv,
)
from deepcrack_external_gate import match_shortcuts
from strong_reducer_baselines import REDUCERS, ReducerDataset, train_one


MECHANISM_MODELS = (
    "grid_external",
    "mask_feature_grid",
    "mask_assignment_identity",
    "grid_mask_graph",
    "shuffled_topology",
    "mask_topo_external",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen third-domain MaskTopo confirmation on FIVES."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("real_data/FIVES/dataset/preprocessed512_v2"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("fives_cache"))
    parser.add_argument("--output", type=Path, default=Path("results_fives"))
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
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--split-seed", type=int, default=20260731)
    parser.add_argument("--shuffle-seed", type=int, default=20260731)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=30)
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
        paths = sorted((root / directory).glob("*.png"))
        for path in paths:
            digest.update(directory.encode())
            digest.update(path.name.encode())
            digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:16]


def disease_suffix(stem: str) -> str:
    suffix = stem.rsplit("_", maxsplit=1)[-1].upper()
    if suffix not in {"A", "D", "G", "N"}:
        raise ValueError(f"Unrecognized FIVES disease suffix: {stem}")
    return suffix


def source_manifest(root: Path, split_seed: int) -> dict[str, list[str]]:
    train_images = sorted(path.stem for path in (root / "train_img").glob("*.png"))
    train_labels = sorted(path.stem for path in (root / "train_lab").glob("*.png"))
    test_images = sorted(path.stem for path in (root / "test_img").glob("*.png"))
    test_labels = sorted(path.stem for path in (root / "test_lab").glob("*.png"))
    if train_images != train_labels or len(train_images) != 600:
        raise ValueError("FIVES train image/label pairs are incomplete.")
    if test_images != test_labels or len(test_images) != 200:
        raise ValueError("FIVES test image/label pairs are incomplete.")
    rng = np.random.default_rng(split_seed)
    train: list[str] = []
    dev: list[str] = []
    for suffix in ("A", "D", "G", "N"):
        members = np.asarray(
            [stem for stem in train_images if disease_suffix(stem) == suffix]
        )
        if members.size != 150:
            raise ValueError(
                f"Expected 150 official-train images for {suffix}, "
                f"found {members.size}."
            )
        rng.shuffle(members)
        train.extend(f"train:{stem}" for stem in members[:120])
        dev.extend(f"train:{stem}" for stem in members[120:])
    return {
        "train": sorted(train),
        "dev": sorted(dev),
        "test": sorted(f"test:{stem}" for stem in test_images),
    }


def load_sources(
    root: Path, source_ids: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, source_id in enumerate(source_ids, start=1):
        split, stem = source_id.split(":", maxsplit=1)
        image = np.asarray(
            Image.open(root / f"{split}_img" / f"{stem}.png").convert("L")
        )
        mask = (
            np.asarray(
                Image.open(root / f"{split}_lab" / f"{stem}.png").convert("L")
            )
            > 0
        )
        if image.shape != (512, 512) or mask.shape != image.shape:
            raise ValueError(f"Unexpected prepared shape for {source_id}.")
        sources[source_id] = (image, mask)
        if index % 200 == 0:
            print(
                f"FIVES_SOURCE_LOAD progress={index}/{len(source_ids)}",
                flush=True,
            )
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
        f"fives_{split}_n{count}_seed{seed}_crop{args.crop_size}_"
        f"matchedv1_{fingerprint}.npz"
    )
    if path.exists():
        with np.load(path) as archive:
            return {key: archive[key] for key in archive.files}
    candidates = generate_arrays(
        sources,
        source_ids,
        count * args.candidate_multiplier,
        seed,
        args,
    )
    arrays = match_shortcuts(candidates, count, seed + 77_777)
    np.savez_compressed(path, **arrays)
    return arrays


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
    source_ids = sorted(set().union(*(set(value) for value in manifest.values())))
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


def rectangular_grid_assignment(token_count: int) -> np.ndarray:
    factor_pairs = [
        (rows, token_count // rows)
        for rows in range(1, int(np.sqrt(token_count)) + 1)
        if token_count % rows == 0
        and 16 % rows == 0
        and 16 % (token_count // rows) == 0
    ]
    if not factor_pairs:
        raise ValueError(
            "Token count needs rectangular factors that both divide 16."
        )
    rows, columns = max(factor_pairs, key=lambda pair: pair[0])
    base = np.arange(token_count, dtype=np.int16).reshape(rows, columns)
    return base.repeat(16 // rows, axis=0).repeat(16 // columns, axis=1)


def grid_artifacts(
    arrays: dict[str, np.ndarray], token_count: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    assignment = rectangular_grid_assignment(token_count)
    graph, closure = coarse_graph8(
        assignment,
        np.zeros((16, 16), dtype=np.int16),
        token_count,
        topology_aware=False,
    )
    sample_count = arrays["images"].shape[0]
    assignments = np.repeat(assignment[None], sample_count, axis=0)
    adjacency = np.repeat(graph[None], sample_count, axis=0)
    reachability = np.repeat(closure[None], sample_count, axis=0)
    prediction = np.asarray(
        [
            structural_prediction(
                assignment,
                closure,
                arrays["endpoint_patches"][index],
                arrays["patch_ids"][index],
            )
            for index in range(sample_count)
        ],
        dtype=np.uint8,
    )
    return {
        "assignment": assignments,
        "adjacency": adjacency,
        "reachability": reachability,
    }, {
        "direct_structural_metrics": balanced_metrics(
            arrays["connected"], prediction
        ),
        "all_tokens_nonempty": True,
    }


def grid_mask_graph_artifacts(
    arrays: dict[str, np.ndarray],
    probabilities: np.ndarray,
    threshold: float,
    closing: int,
    token_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    assignment = rectangular_grid_assignment(token_count)
    sample_count = arrays["images"].shape[0]
    assignments = np.repeat(assignment[None], sample_count, axis=0)
    adjacency = np.empty((sample_count, token_count, token_count), dtype=np.uint8)
    reachability = np.empty_like(adjacency)
    structural = np.empty(sample_count, dtype=np.uint8)
    densities = np.empty(sample_count, dtype=np.float64)
    for index in range(sample_count):
        groups = predicted_groups(
            probabilities[index],
            threshold,
            closing,
            maximum_foreground=token_count - 1,
        )
        graph, closure = coarse_graph8(
            assignment, groups, token_count, topology_aware=True
        )
        adjacency[index] = graph
        reachability[index] = closure
        structural[index] = structural_prediction(
            assignment,
            closure,
            arrays["endpoint_patches"][index],
            groups,
        )
        densities[index] = float(graph.mean())
    return {
        "assignment": assignments,
        "adjacency": adjacency,
        "reachability": reachability,
    }, {
        "direct_structural_metrics": balanced_metrics(
            arrays["connected"], structural
        ),
        "mean_adjacency_density": float(densities.mean()),
        "all_tokens_nonempty": True,
    }


def array_invariants(
    arrays: dict[str, dict[str, np.ndarray]],
    manifest: dict[str, list[str]],
    source_masks: dict[str, np.ndarray],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for split, current in arrays.items():
        count = current["images"].shape[0]
        checks[f"{split}_shape"] = (
            current["images"].shape == (count, 3, 64, 64)
            and current["patch_ids"].shape == (count, 16, 16)
            and current["endpoint_patches"].shape == (count, 2)
        )
        checks[f"{split}_balanced"] = bool(
            np.isclose(current["connected"].mean(), 0.5)
        )
        checks[f"{split}_source_membership"] = set(
            current["source_id"].tolist()
        ).issubset(set(manifest[split]))
        reconstructed = build_pixel_masks(current, source_masks)
        marker_overlap = []
        for channel in (1, 2):
            marker_overlap.append(
                np.all(
                    np.any(
                        (current["images"][:, channel] > 127)
                        & reconstructed.astype(bool),
                        axis=(1, 2),
                    )
                )
            )
        checks[f"{split}_marker_mask_overlap"] = all(marker_overlap)
        first_flat = current["endpoint_patches"][:, 0].astype(np.int64)
        second_flat = current["endpoint_patches"][:, 1].astype(np.int64)
        sample_index = np.arange(count)
        first_id = current["patch_ids"][
            sample_index, first_flat // 16, first_flat % 16
        ]
        second_id = current["patch_ids"][
            sample_index, second_flat // 16, second_flat % 16
        ]
        expected = current["connected"].astype(bool)
        checks[f"{split}_endpoint_component_label_consistency"] = bool(
            np.all(first_id > 0)
            and np.all(second_id > 0)
            and np.all((first_id == second_id) == expected)
        )
    return checks


def run_self_test() -> None:
    assignment = rectangular_grid_assignment(8)
    assert assignment.shape == (16, 16)
    assert np.unique(assignment).tolist() == list(range(8))
    arrays = {
        "images": np.zeros((2, 3, 64, 64), dtype=np.uint8),
        "connected": np.asarray([0, 1], dtype=np.int64),
        "patch_ids": np.zeros((2, 16, 16), dtype=np.int16),
        "endpoint_patches": np.asarray([[0, 255], [16, 239]], dtype=np.int16),
    }
    artifacts, diagnostics = grid_artifacts(arrays, 8)
    assert artifacts["assignment"].shape == (2, 16, 16)
    assert diagnostics["all_tokens_nonempty"]
    print("FIVES_EXTERNAL_GATE_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    args.dataset = "fives"
    if args.self_test:
        run_self_test()
        return
    if (
        args.crop_size != 64
        or args.output_size != 64
        or args.token_count != 8
    ):
        raise ValueError(
            "The frozen FIVES gate requires 64x64 working crops and K=8."
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays, manifest, sources, fingerprint = load_all_arrays(args)
    source_masks = {key: value[1] for key, value in sources.items()}
    diagnostics = data_diagnostics(arrays, manifest)
    invariants = array_invariants(arrays, manifest, source_masks)
    diagnostic_gates = {
        "source_split_overlap_absent": all(
            not values for values in diagnostics["source_overlap"].values()
        ),
        "all_splits_exactly_balanced": all(
            np.isclose(value, 0.5)
            for value in diagnostics["positive_fraction"].values()
        ),
        "endpoint_distance_shortcut_at_most_55pct": (
            diagnostics["endpoint_distance_shortcut_test_accuracy"] <= 0.55
        ),
        "mean_intensity_shortcut_at_most_55pct": (
            diagnostics["mean_intensity_shortcut_test_accuracy"] <= 0.55
        ),
        "at_least_90pct_official_test_sources_used": (
            diagnostics["unique_sources_used"]["test"] >= 180
        ),
        "all_array_invariants_pass": all(invariants.values()),
    }
    data_report = {
        "dataset": "FIVES",
        "dataset_fingerprint": fingerprint,
        "official_source_counts": {
            split: len(values) for split, values in manifest.items()
        },
        "diagnostics": diagnostics,
        "array_invariants": invariants,
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
        raise RuntimeError("FIVES data gate failed; training was not started.")

    pixel_masks = {
        split: build_pixel_masks(current, source_masks)
        for split, current in arrays.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_model, histories = train_mask_model(
        args,
        MaskDataset(arrays["train"]["images"], pixel_masks["train"], True),
        MaskDataset(arrays["dev"]["images"], pixel_masks["dev"], False),
        output,
        device,
    )
    heldout_mask_metrics = segmentation_metrics(
        mask_model,
        DataLoader(
            MaskDataset(
                arrays["test"]["images"], pixel_masks["test"], False
            ),
            batch_size=args.batch_size,
            shuffle=False,
        ),
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

    grid_bank: dict[str, dict[str, np.ndarray]] = {}
    artifact_bank: dict[str, dict[str, dict[str, np.ndarray]]] = {
        name: {} for name in MECHANISM_MODELS[2:]
    }
    artifact_diagnostics: dict[str, dict[str, Any]] = {
        name: {} for name in MECHANISM_MODELS
    }
    for split_index, split in enumerate(("train", "dev", "test")):
        grid_bank[split], artifact_diagnostics["grid_external"][split] = (
            grid_artifacts(arrays[split], args.token_count)
        )
        artifact_diagnostics["mask_feature_grid"][split] = (
            artifact_diagnostics["grid_external"][split]
        )
        for result_name, builder_name in (
            ("mask_assignment_identity", "mask_assignment_identity"),
            ("shuffled_topology", "shuffled_topology"),
            ("mask_topo_external", "mask_topo_recheck"),
        ):
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
        (
            artifact_bank["grid_mask_graph"][split],
            artifact_diagnostics["grid_mask_graph"][split],
        ) = grid_mask_graph_artifacts(
            arrays[split],
            probabilities[split],
            threshold,
            closing,
            args.token_count,
        )

    summaries: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    truth_reference: np.ndarray | None = None
    for model_name in MECHANISM_MODELS:
        print(
            f"FIVES_MECHANISM_START model={model_name} seed={args.seed}",
            flush=True,
        )
        current_artifacts = (
            grid_bank
            if model_name in {"grid_external", "mask_feature_grid"}
            else artifact_bank[model_name]
        )
        if model_name == "mask_feature_grid":
            summary, history, prediction, truth = train_mask_feature_grid(
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
        else:
            summary, history, prediction, truth = train_classifier(
                model_name,
                args,
                RealDataset(
                    arrays["train"],
                    current_artifacts["train"],
                    augment=True,
                ),
                RealDataset(
                    arrays["dev"], current_artifacts["dev"]
                ),
                RealDataset(
                    arrays["test"], current_artifacts["test"]
                ),
                output,
                device,
            )
        if truth_reference is None:
            truth_reference = truth
        elif not np.array_equal(truth_reference, truth):
            raise RuntimeError("Held-out truth changed between mechanisms.")
        summaries[model_name] = summary
        predictions[model_name] = prediction
        histories.extend(history)
        print(
            f"FIVES_MECHANISM_DONE model={model_name} seed={args.seed} "
            f"test_bal={summary['test_metrics']['balanced_accuracy']:.4f}",
            flush=True,
        )

    reducer_datasets = {
        split: ReducerDataset(
            arrays[split],
            probabilities[split],
            augment=split == "train",
        )
        for split in ("train", "dev", "test")
    }
    for model_name in REDUCERS:
        print(
            f"FIVES_REDUCER_START model={model_name} seed={args.seed}",
            flush=True,
        )
        summary, history, prediction, truth = train_one(
            model_name, args, reducer_datasets, output, device
        )
        if truth_reference is None or not np.array_equal(truth_reference, truth):
            raise RuntimeError("Held-out truth changed between model families.")
        summaries[model_name] = summary
        predictions[model_name] = prediction
        histories.extend(history)
        print(
            f"FIVES_REDUCER_DONE model={model_name} seed={args.seed} "
            f"test_bal={summary['test_metrics']['balanced_accuracy']:.4f}",
            flush=True,
        )
    if truth_reference is None:
        raise RuntimeError("No FIVES model was run.")
    write_csv(output / "history.csv", histories)
    np.savez_compressed(
        output / "heldout_predictions.npz",
        truth=truth_reference,
        source_id=arrays["test"]["source_id"],
        **predictions,
    )
    result = {
        "experiment_id": "fives_masktopo_external_confirmation",
        "status": "completed",
        "dataset": {
            "name": "FIVES",
            "figshare_doi": "10.6084/m9.figshare.19688169.v1",
            "fingerprint": fingerprint,
            "license": "CC BY 4.0",
            "official_source_counts": {
                split: len(values) for split, values in manifest.items()
            },
        },
        "optimization_seed": args.seed,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "fine_token_count": 256,
        "reduced_token_count": args.token_count,
        "data_diagnostics": diagnostics,
        "array_invariants": invariants,
        "selected_on_dev": selected,
        "heldout_mask_metrics": heldout_mask_metrics,
        "model_results": summaries,
        "artifact_diagnostics": artifact_diagnostics,
        "official_test_used_for_model_selection": False,
        "inference_uses_ground_truth_mask": False,
        "test_model_performance_previously_observed_before_this_run": False,
        "protocol": str(
            (
                Path(__file__).parent
                / "FIVES_EXTERNAL_CONFIRMATION_PROTOCOL.md"
            ).resolve()
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

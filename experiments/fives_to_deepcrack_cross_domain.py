from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import strong_reducer_baselines as reducer_baselines
from crackforest_mask_topo import (
    CrackUNet,
    build_pixel_masks,
    predict_masks,
)
from crackforest_mechanism_ablation import build_artifacts
from crackforest_real_gate import (
    RealDataset,
    balanced_metrics,
    evaluate_classifier,
)
from deepcrack_external_gate import load_all_arrays
from fives_external_gate import grid_artifacts as fives_grid_artifacts
from fives_external_gate import load_all_arrays as load_fives_arrays
from fives_g2tm_collision import G2TMFixedKModel, make_collision_model
from strong_reducer_baselines import (
    ReducerDataset,
    evaluate,
)
from topocoarsen_oracle import TopoCoarsenModel


SEEDS = (20260810, 20260811, 20260812)
TOPO_MODELS = ("grid_external", "mask_topo_external")
REDUCER_MODELS = (
    "tokenlearner",
    "tome_style",
    "mask_guided_queries",
)
COLLISION_MODELS = ("g2tm_fixedk_feature", "g2tm_fixedk_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen FIVES-to-DeepCrack cross-domain transfer."
    )
    parser.add_argument(
        "--fives-root",
        type=Path,
        default=Path("real_data/FIVES/dataset/preprocessed512_v2"),
    )
    parser.add_argument(
        "--deepcrack-root",
        type=Path,
        default=Path("real_data/DeepCrack/dataset/extracted"),
    )
    parser.add_argument("--fives-cache", type=Path, default=Path("fives_cache_v2"))
    parser.add_argument(
        "--deepcrack-cache", type=Path, default=Path("deepcrack_cache")
    )
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--dev-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=1200)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def pixel_f1(
    probabilities: np.ndarray,
    target_masks: np.ndarray,
    threshold: float = 0.5,
) -> float:
    prediction = probabilities >= threshold
    target = target_masks.astype(bool)
    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def load_model(
    checkpoint: Path,
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    if model_name in TOPO_MODELS:
        model = TopoCoarsenModel(model_name, args.dim, args.token_count)
    elif model_name in REDUCER_MODELS:
        model = reducer_baselines.make_model(
            model_name, args.dim, args.token_count
        )
    elif model_name in COLLISION_MODELS:
        model = make_collision_model(model_name, args.dim, args.token_count)
    else:
        raise ValueError(model_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def evaluate_topology_model(
    model: torch.nn.Module,
    model_name: str,
    arrays: dict[str, np.ndarray],
    artifacts: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    loader = DataLoader(
        RealDataset(arrays, artifacts, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    metrics, prediction, truth = evaluate_classifier(
        model, model_name, loader, device, return_predictions=True
    )
    if prediction is None or truth is None:
        raise AssertionError("Missing topology predictions.")
    return float(metrics["balanced_accuracy"]), prediction, truth


def evaluate_reducer_model(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    probabilities: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    loader = DataLoader(
        ReducerDataset(arrays, probabilities, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    metrics, prediction, truth = evaluate(
        model, loader, device, return_predictions=True
    )
    if prediction is None or truth is None:
        raise AssertionError("Missing reducer predictions.")
    return float(metrics["balanced_accuracy"]), prediction, truth


def main() -> None:
    started = time.time()
    args = parse_args()
    if args.self_test:
        print("FIVES_TO_DEEPCRACK_CROSS_DOMAIN_SELF_TEST_PASS", flush=True)
        return
    if args.output is None:
        raise ValueError("--output is required.")
    if args.token_count != 8 or args.crop_size != 64:
        raise ValueError("This frozen transfer gate requires crop=64 and K=8.")
    args.dataset = "deepcrack"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deep_arrays, deep_manifest, deep_sources, deep_fingerprint = load_all_arrays(
        argparse.Namespace(
            dataset_root=args.deepcrack_root,
            cache_dir=args.deepcrack_cache,
            train_size=args.train_size,
            dev_size=args.dev_size,
            test_size=args.test_size,
            candidate_multiplier=args.candidate_multiplier,
            crop_size=args.crop_size,
            output_size=args.output_size,
            split_seed=20260730,
            data_seed=20260810,
        )
    )
    target_test = deep_arrays["test"]
    target_masks = build_pixel_masks(
        target_test,
        {key: value[1] for key, value in deep_sources.items()},
    )

    per_seed: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"CROSS_DOMAIN_START seed={seed}", flush=True)
        source_dir = args.source_root / f"results_fives_seed{seed}"
        collision_dir = (
            args.source_root / f"results_fives_g2tm_collision_seed{seed}"
        )
        source_result = json.loads(
            (source_dir / "result.json").read_text(encoding="utf-8")
        )
        selected = source_result["selected_on_dev"]
        threshold = float(selected["mask_threshold"])
        closing = int(selected["closing_iterations"])

        mask_model = CrackUNet().to(device)
        mask_payload = torch.load(
            source_dir / f"mask_predictor_seed{seed}.pt",
            map_location=device,
            weights_only=False,
        )
        mask_model.load_state_dict(mask_payload["model"])
        target_probabilities = predict_masks(
            mask_model,
            target_test["images"],
            args.batch_size,
            device,
        )
        target_mask_f1 = pixel_f1(
            target_probabilities, target_masks, threshold=0.5
        )

        target_artifacts, target_diagnostics = build_artifacts(
            target_test,
            target_probabilities,
            threshold,
            closing,
            args.token_count,
            "mask_topo_recheck",
            20260731,
        )
        target_grid, _ = fives_grid_artifacts(target_test, args.token_count)
        model_scores: dict[str, float] = {}
        model_predictions: dict[str, np.ndarray] = {}
        truth_reference: np.ndarray | None = None
        topo_checkpoints = {
            name: source_dir / f"{name}_seed{seed}.pt"
            for name in TOPO_MODELS
        }
        for name, checkpoint in topo_checkpoints.items():
            model = load_model(checkpoint, name, args, device)
            artifacts = target_grid if name == "grid_external" else target_artifacts
            score, prediction, truth = evaluate_topology_model(
                model, name, target_test, artifacts, args, device
            )
            model_scores[name] = score
            model_predictions[name] = prediction
            if truth_reference is None:
                truth_reference = truth
            elif not np.array_equal(truth_reference, truth):
                raise RuntimeError("Target truth changed between models.")

        reducer_probabilities = target_probabilities.astype(np.float32)
        reducer_checkpoints = {
            name: source_dir / f"{name}_seed{seed}.pt"
            for name in REDUCER_MODELS
        }
        for name, checkpoint in reducer_checkpoints.items():
            model = load_model(checkpoint, name, args, device)
            score, prediction, truth = evaluate_reducer_model(
                model,
                target_test,
                reducer_probabilities,
                args,
                device,
            )
            model_scores[name] = score
            model_predictions[name] = prediction
            if truth_reference is None:
                truth_reference = truth
            elif not np.array_equal(truth_reference, truth):
                raise RuntimeError("Target truth changed between models.")
        for name in COLLISION_MODELS:
            checkpoint = collision_dir / f"{name}_seed{seed}.pt"
            model = load_model(checkpoint, name, args, device)
            score, prediction, truth = evaluate_reducer_model(
                model,
                target_test,
                reducer_probabilities,
                args,
                device,
            )
            model_scores[name] = score
            model_predictions[name] = prediction
            if truth_reference is None:
                truth_reference = truth
            elif not np.array_equal(truth_reference, truth):
                raise RuntimeError("Target truth changed between models.")
        if truth_reference is None:
            raise RuntimeError("No target predictions were generated.")
        np.savez_compressed(
            output / f"heldout_predictions_seed{seed}.npz",
            truth=truth_reference,
            source_id=target_test["source_id"],
            **model_predictions,
        )
        row = {
            "seed": seed,
            "source_mask_threshold": threshold,
            "source_closing_iterations": closing,
            "target_mask_f1_without_target_finetuning": target_mask_f1,
            "target_artifact_diagnostics": target_diagnostics,
            "balanced_accuracy": model_scores,
        }
        per_seed.append(row)
        print(
            f"CROSS_DOMAIN_DONE seed={seed} "
            f"mask_f1={target_mask_f1:.4f} "
            f"masktopo={model_scores['mask_topo_external']:.4f} "
            f"grid={model_scores['grid_external']:.4f}",
            flush=True,
        )

    model_names = list(per_seed[0]["balanced_accuracy"])
    aggregate = {
        name: {
            "scores_by_seed": [
                row["balanced_accuracy"][name] for row in per_seed
            ],
            "mean": float(
                np.mean(
                    [row["balanced_accuracy"][name] for row in per_seed]
                )
            ),
            "sample_std": float(
                np.std(
                    [row["balanced_accuracy"][name] for row in per_seed],
                    ddof=1,
                )
            ),
        }
        for name in model_names
    }
    mask_f1_values = [
        row["target_mask_f1_without_target_finetuning"] for row in per_seed
    ]
    report = {
        "experiment_id": "fives_to_deepcrack_cross_domain_transfer",
        "status": "completed",
        "source_dataset": "FIVES",
        "target_dataset": "DeepCrack",
        "target_fingerprint": deep_fingerprint,
        "target_official_source_counts": {
            split: len(values) for split, values in deep_manifest.items()
        },
        "token_count": args.token_count,
        "no_target_dense_mask_training": True,
        "no_target_classifier_training": True,
        "seeds": list(args.seeds),
        "target_mask_f1_mean": float(np.mean(mask_f1_values)),
        "target_mask_f1_sample_std": float(np.std(mask_f1_values, ddof=1)),
        "aggregates": aggregate,
        "per_seed": per_seed,
        "duration_seconds": time.time() - started,
        "protocol": str(
            (Path(__file__).parent / "FIVES_TO_DEEPCRACK_CROSS_DOMAIN_PROTOCOL.md")
            .resolve()
        ),
    }
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# FIVES → DeepCrack frozen cross-domain transfer",
        "",
        "No DeepCrack dense mask was used for training the mask predictor or "
        "the classifier. Target labels are used only for final evaluation.",
        "",
        f"Frozen target-domain mask F1: {100 * report['target_mask_f1_mean']:.2f} "
        f"± {100 * report['target_mask_f1_sample_std']:.2f}%.",
        "",
        "| Model | Seed scores (%) | Mean ± sample SD |",
        "|---|---:|---:|",
    ]
    for name, values in aggregate.items():
        lines.append(
            f"| {name} | "
            + ", ".join(f"{100 * value:.2f}" for value in values["scores_by_seed"])
            + f" | {100 * values['mean']:.2f} ± {100 * values['sample_std']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a transfer diagnostic, not a claim of "
            "target-domain state of the art. A positive MaskTopo-minus-grid "
            "gap without target dense-mask training supports reduced label "
            "circularity.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crackforest_mask_topo import find_cached_split
from crackforest_real_gate import balanced_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate paired CrackForest MaskTopo and grid runs."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("real_cache"))
    parser.add_argument(
        "--mask-prefix",
        type=str,
        default="results_crackforest_mask_topo_v2_seed",
    )
    parser.add_argument(
        "--grid-dir", type=Path, default=Path("results_crackforest_grid_multiseed")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results_crackforest_real_summary")
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260810, 20260811, 20260812]
    )
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--test-size", type=int, default=320)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--permutation-repetitions", type=int, default=50000)
    parser.add_argument("--statistics-seed", type=int, default=20260730)
    return parser.parse_args()


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(balanced_metrics(truth, prediction)["balanced_accuracy"])


def main() -> None:
    args = parse_args()
    test_arrays = find_cached_split(
        args.cache_dir.resolve(),
        "test",
        args.test_size,
        args.data_seed + 2_000_000,
        args.crop_size,
    )
    source_ids = test_arrays["source_id"].astype(str)
    unique_sources = np.unique(source_ids)
    grid_path = args.grid_dir.resolve() / "heldout_predictions.npz"
    with np.load(grid_path) as archive:
        truth = archive["truth"].astype(np.uint8)
        grid_predictions = {
            seed: archive[f"seed{seed}"].astype(np.uint8)
            for seed in args.seeds
        }
    mask_predictions = {}
    mask_results = {}
    for seed in args.seeds:
        root = Path(f"{args.mask_prefix}{seed}").resolve()
        with np.load(root / "heldout_predictions.npz") as archive:
            if not np.array_equal(truth, archive["truth"]):
                raise RuntimeError(f"Held-out truth mismatch for seed {seed}.")
            mask_predictions[seed] = archive["mask_topocoarsen"].astype(
                np.uint8
            )
        mask_results[seed] = json.loads(
            (root / "result.json").read_text(encoding="utf-8")
        )
    if truth.shape != source_ids.shape:
        raise RuntimeError("Source IDs do not align with held-out predictions.")

    rows = []
    for seed in args.seeds:
        mask_score = balanced_accuracy(truth, mask_predictions[seed])
        grid_score = balanced_accuracy(truth, grid_predictions[seed])
        structural = mask_results[seed]["artifact_diagnostics"]["test"][
            "structural_metrics"
        ]["balanced_accuracy"]
        rows.append(
            {
                "optimization_seed": seed,
                "mask_topo_balanced_accuracy": mask_score,
                "grid_balanced_accuracy": grid_score,
                "paired_gain": mask_score - grid_score,
                "mask_structural_balanced_accuracy": structural,
                "mask_threshold": mask_results[seed]["selected_on_dev"][
                    "mask_threshold"
                ],
                "closing_iterations": mask_results[seed]["selected_on_dev"][
                    "closing_iterations"
                ],
            }
        )

    rng = np.random.default_rng(args.statistics_seed)
    source_indices = {
        source: np.flatnonzero(source_ids == source) for source in unique_sources
    }
    bootstrapped_gains = np.empty(
        args.bootstrap_repetitions, dtype=np.float64
    )
    for repetition in range(args.bootstrap_repetitions):
        sampled_sources = rng.choice(
            unique_sources, size=len(unique_sources), replace=True
        )
        indices = np.concatenate(
            [source_indices[source] for source in sampled_sources]
        )
        bootstrapped_gains[repetition] = np.mean(
            [
                balanced_accuracy(
                    truth[indices], mask_predictions[seed][indices]
                )
                - balanced_accuracy(
                    truth[indices], grid_predictions[seed][indices]
                )
                for seed in args.seeds
            ]
        )

    observed_gain = float(np.mean([row["paired_gain"] for row in rows]))
    null_gains = np.empty(args.permutation_repetitions, dtype=np.float64)
    for repetition in range(args.permutation_repetitions):
        swap_sources = set(
            unique_sources[
                rng.integers(0, 2, size=len(unique_sources), dtype=np.uint8)
                == 1
            ]
        )
        swap_mask = np.array(
            [source in swap_sources for source in source_ids], dtype=bool
        )
        seed_gains = []
        for seed in args.seeds:
            mask_prediction = mask_predictions[seed].copy()
            grid_prediction = grid_predictions[seed].copy()
            temporary = mask_prediction[swap_mask].copy()
            mask_prediction[swap_mask] = grid_prediction[swap_mask]
            grid_prediction[swap_mask] = temporary
            seed_gains.append(
                balanced_accuracy(truth, mask_prediction)
                - balanced_accuracy(truth, grid_prediction)
            )
        null_gains[repetition] = np.mean(seed_gains)
    permutation_p = (
        1 + int(np.sum(null_gains >= observed_gain))
    ) / (args.permutation_repetitions + 1)

    method_scores = np.array(
        [row["mask_topo_balanced_accuracy"] for row in rows]
    )
    grid_scores = np.array([row["grid_balanced_accuracy"] for row in rows])
    gains = np.array([row["paired_gain"] for row in rows])
    structural_scores = np.array(
        [row["mask_structural_balanced_accuracy"] for row in rows]
    )
    report = {
        "experiment_id": "crackforest_real_gate_multiseed_summary",
        "status": "completed",
        "data_seed": args.data_seed,
        "optimization_seeds": args.seeds,
        "test_crop_count": int(truth.size),
        "test_source_image_count": int(unique_sources.size),
        "runs": rows,
        "aggregate": {
            "mask_topo_balanced_accuracy_mean": float(method_scores.mean()),
            "mask_topo_balanced_accuracy_sample_std": float(
                method_scores.std(ddof=1)
            ),
            "grid_balanced_accuracy_mean": float(grid_scores.mean()),
            "grid_balanced_accuracy_sample_std": float(
                grid_scores.std(ddof=1)
            ),
            "paired_gain_mean": observed_gain,
            "paired_gain_sample_std": float(gains.std(ddof=1)),
            "minimum_paired_gain": float(gains.min()),
            "test_structural_balanced_accuracy_mean": float(
                structural_scores.mean()
            ),
            "test_structural_balanced_accuracy_sample_std": float(
                structural_scores.std(ddof=1)
            ),
            "source_cluster_bootstrap_95_ci_for_mean_gain": [
                float(np.quantile(bootstrapped_gains, 0.025)),
                float(np.quantile(bootstrapped_gains, 0.975)),
            ],
            "source_cluster_permutation_one_sided_p": float(permutation_p),
        },
        "precommitted_gates": {
            "all_seed_test_structural_accuracy_at_least_70pct": bool(
                np.all(structural_scores >= 0.70)
            ),
            "all_seed_classifier_gain_at_least_3pp": bool(
                np.all(gains >= 0.03)
            ),
        },
    }
    report["verdict"] = (
        "PROCEED_TO_SECOND_REAL_DATASET"
        if all(report["precommitted_gates"].values())
        else "STOP_CURRENT_GENERAL_CONNECTOR_ROUTE"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

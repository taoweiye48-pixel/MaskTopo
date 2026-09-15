from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crackforest_mask_topo import find_cached_split
from crackforest_real_gate import balanced_metrics


MODES = (
    "grid_recheck",
    "mask_feature_grid",
    "mask_assignment_identity",
    "grid_mask_graph",
    "shuffled_topology",
    "mask_topo_recheck",
)

CONTRASTS = {
    "topology_vs_mask_feature": (
        "mask_topo_recheck",
        "mask_feature_grid",
    ),
    "aligned_vs_shuffled_topology": (
        "mask_topo_recheck",
        "shuffled_topology",
    ),
    "graph_message_contribution": (
        "mask_topo_recheck",
        "mask_assignment_identity",
    ),
    "component_pooling_contribution": (
        "mask_topo_recheck",
        "grid_mask_graph",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the CrackForest mechanism ablation."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("real_cache"))
    parser.add_argument(
        "--input-prefix",
        type=str,
        default="results_crackforest_mechanism_seed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_crackforest_mechanism_summary"),
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


def cluster_statistics(
    truth: np.ndarray,
    sources: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    unique_sources = np.unique(sources)
    source_indices = {
        source: np.flatnonzero(sources == source) for source in unique_sources
    }
    seed_gains = np.array(
        [
            balanced_accuracy(truth, first[index])
            - balanced_accuracy(truth, second[index])
            for index in range(first.shape[0])
        ]
    )
    observed = float(seed_gains.mean())
    bootstrapped = np.empty(bootstrap_repetitions, dtype=np.float64)
    for repetition in range(bootstrap_repetitions):
        sampled_sources = rng.choice(
            unique_sources, size=unique_sources.size, replace=True
        )
        indices = np.concatenate(
            [source_indices[source] for source in sampled_sources]
        )
        bootstrapped[repetition] = np.mean(
            [
                balanced_accuracy(truth[indices], first[seed, indices])
                - balanced_accuracy(truth[indices], second[seed, indices])
                for seed in range(first.shape[0])
            ]
        )
    null_distribution = np.empty(permutation_repetitions, dtype=np.float64)
    for repetition in range(permutation_repetitions):
        swap_sources = set(
            unique_sources[
                rng.integers(
                    0, 2, size=unique_sources.size, dtype=np.uint8
                )
                == 1
            ]
        )
        swap = np.array(
            [source in swap_sources for source in sources], dtype=bool
        )
        gains = []
        for seed in range(first.shape[0]):
            permuted_first = first[seed].copy()
            permuted_second = second[seed].copy()
            temporary = permuted_first[swap].copy()
            permuted_first[swap] = permuted_second[swap]
            permuted_second[swap] = temporary
            gains.append(
                balanced_accuracy(truth, permuted_first)
                - balanced_accuracy(truth, permuted_second)
            )
        null_distribution[repetition] = np.mean(gains)
    p_value = (
        1 + int(np.sum(null_distribution >= observed))
    ) / (permutation_repetitions + 1)
    source_gains = []
    for source in unique_sources:
        indices = source_indices[source]
        source_gains.append(
            np.mean(
                [
                    np.mean(first[seed, indices] == truth[indices])
                    - np.mean(second[seed, indices] == truth[indices])
                    for seed in range(first.shape[0])
                ]
            )
        )
    return {
        "paired_gains_by_seed": seed_gains.tolist(),
        "mean_gain": observed,
        "sample_std_gain": float(seed_gains.std(ddof=1)),
        "source_cluster_bootstrap_95_ci": [
            float(np.quantile(bootstrapped, 0.025)),
            float(np.quantile(bootstrapped, 0.975)),
        ],
        "source_cluster_permutation_one_sided_p_raw": float(p_value),
        "sources_with_positive_mean_accuracy_gain": int(
            np.sum(np.asarray(source_gains) > 0)
        ),
        "sources_with_zero_mean_accuracy_gain": int(
            np.sum(np.asarray(source_gains) == 0)
        ),
        "sources_with_negative_mean_accuracy_gain": int(
            np.sum(np.asarray(source_gains) < 0)
        ),
    }


def holm_adjust(raw_p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p_values, key=raw_p_values.__getitem__)
    adjusted = {}
    running_maximum = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        current = min(1.0, (count - rank) * raw_p_values[name])
        running_maximum = max(running_maximum, current)
        adjusted[name] = running_maximum
    return adjusted


def main() -> None:
    args = parse_args()
    test_arrays = find_cached_split(
        args.cache_dir.resolve(),
        "test",
        args.test_size,
        args.data_seed + 2_000_000,
        args.crop_size,
    )
    sources = test_arrays["source_id"].astype(str)
    predictions = {
        mode: np.empty((len(args.seeds), args.test_size), dtype=np.uint8)
        for mode in MODES
    }
    truth: np.ndarray | None = None
    run_rows = []
    for seed_index, seed in enumerate(args.seeds):
        root = Path(f"{args.input_prefix}{seed}").resolve()
        result = json.loads(
            (root / "result.json").read_text(encoding="utf-8")
        )
        with np.load(root / "heldout_predictions.npz") as archive:
            current_truth = archive["truth"].astype(np.uint8)
            if truth is None:
                truth = current_truth
            elif not np.array_equal(truth, current_truth):
                raise RuntimeError("Held-out labels changed between seeds.")
            for mode in MODES:
                predictions[mode][seed_index] = archive[mode].astype(np.uint8)
                score = balanced_accuracy(
                    current_truth, predictions[mode][seed_index]
                )
                recorded = result["model_results"][mode]["test_metrics"][
                    "balanced_accuracy"
                ]
                if not np.isclose(score, recorded):
                    raise RuntimeError(f"Prediction mismatch: seed={seed}, {mode}")
                run_rows.append(
                    {
                        "optimization_seed": seed,
                        "mode": mode,
                        "balanced_accuracy": score,
                        "parameters": result["model_results"][mode][
                            "parameters"
                        ],
                    }
                )
    if truth is None:
        raise RuntimeError("No runs found.")
    aggregates = {}
    for mode in MODES:
        scores = np.array(
            [
                balanced_accuracy(truth, predictions[mode][seed])
                for seed in range(len(args.seeds))
            ]
        )
        aggregates[mode] = {
            "scores_by_seed": scores.tolist(),
            "mean_balanced_accuracy": float(scores.mean()),
            "sample_std_balanced_accuracy": float(scores.std(ddof=1)),
        }
    rng = np.random.default_rng(args.statistics_seed)
    contrast_results = {}
    for name, (first_name, second_name) in CONTRASTS.items():
        contrast_results[name] = cluster_statistics(
            truth,
            sources,
            predictions[first_name],
            predictions[second_name],
            args.bootstrap_repetitions,
            args.permutation_repetitions,
            rng,
        )
        contrast_results[name]["first"] = first_name
        contrast_results[name]["second"] = second_name
    adjusted = holm_adjust(
        {
            name: float(
                result["source_cluster_permutation_one_sided_p_raw"]
            )
            for name, result in contrast_results.items()
        }
    )
    for name, value in adjusted.items():
        contrast_results[name][
            "source_cluster_permutation_one_sided_p_holm"
        ] = value

    primary_feature = contrast_results["topology_vs_mask_feature"]
    primary_shuffle = contrast_results["aligned_vs_shuffled_topology"]
    graph = contrast_results["graph_message_contribution"]
    pooling = contrast_results["component_pooling_contribution"]
    gates = {
        "topology_beats_mask_feature_by_3pp_mean_and_all_seeds_positive": (
            float(primary_feature["mean_gain"]) >= 0.03
            and all(
                gain > 0
                for gain in primary_feature["paired_gains_by_seed"]
            )
        ),
        "aligned_beats_shuffled_by_3pp_mean_and_all_seeds_positive": (
            float(primary_shuffle["mean_gain"]) >= 0.03
            and all(
                gain > 0
                for gain in primary_shuffle["paired_gains_by_seed"]
            )
        ),
        "graph_message_contribution_at_least_3pp_mean": (
            float(graph["mean_gain"]) >= 0.03
        ),
        "component_pooling_contribution_at_least_3pp_mean": (
            float(pooling["mean_gain"]) >= 0.03
        ),
    }
    report = {
        "experiment_id": "crackforest_masktopo_mechanism_summary",
        "status": "completed",
        "data_seed": args.data_seed,
        "optimization_seeds": args.seeds,
        "test_crop_count": int(truth.size),
        "test_source_image_count": int(np.unique(sources).size),
        "test_was_previously_observed": True,
        "interpretation_scope": "mechanism diagnostic, not confirmatory",
        "runs": run_rows,
        "aggregates": aggregates,
        "contrasts": contrast_results,
        "frozen_gates": gates,
    }
    report["verdict"] = (
        "TOPOLOGY_SPECIFIC_SIGNAL_SUPPORTED_DIAGNOSTIC"
        if gates[
            "topology_beats_mask_feature_by_3pp_mean_and_all_seeds_positive"
        ]
        and gates[
            "aligned_beats_shuffled_by_3pp_mean_and_all_seeds_positive"
        ]
        else "TOPOLOGY_SPECIFIC_SIGNAL_NOT_ISOLATED"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

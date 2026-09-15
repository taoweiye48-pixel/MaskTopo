from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from summarize_crackforest_mechanism_ablation import (
    cluster_statistics,
    holm_adjust,
)


MODES = (
    "grid_external",
    "mask_feature_grid",
    "mask_assignment_identity",
    "grid_mask_graph",
    "shuffled_topology",
    "mask_topo_external",
)

CONTRASTS = {
    "topology_vs_grid": ("mask_topo_external", "grid_external"),
    "topology_vs_mask_feature": (
        "mask_topo_external",
        "mask_feature_grid",
    ),
    "aligned_vs_shuffled_topology": (
        "mask_topo_external",
        "shuffled_topology",
    ),
    "graph_message_contribution": (
        "mask_topo_external",
        "mask_assignment_identity",
    ),
    "component_pooling_contribution": (
        "mask_topo_external",
        "grid_mask_graph",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the one-shot DeepCrack external confirmation."
    )
    parser.add_argument(
        "--input-prefix", type=str, default="results_deepcrack_seed"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_deepcrack_external_summary"),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260810, 20260811, 20260812]
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--permutation-repetitions", type=int, default=50000)
    parser.add_argument("--statistics-seed", type=int, default=20260730)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions: dict[str, list[np.ndarray]] = {mode: [] for mode in MODES}
    truth: np.ndarray | None = None
    sources: np.ndarray | None = None
    structural_scores = []
    run_rows = []
    for seed in args.seeds:
        root = Path(f"{args.input_prefix}{seed}").resolve()
        result = json.loads(
            (root / "result.json").read_text(encoding="utf-8")
        )
        with np.load(root / "heldout_predictions.npz") as archive:
            current_truth = archive["truth"].astype(np.uint8)
            current_sources = archive["source_id"].astype(str)
            if truth is None:
                truth = current_truth
                sources = current_sources
            elif not np.array_equal(truth, current_truth) or not np.array_equal(
                sources, current_sources
            ):
                raise RuntimeError("DeepCrack held-out set changed between seeds.")
            for mode in MODES:
                predictions[mode].append(archive[mode].astype(np.uint8))
                run_rows.append(
                    {
                        "optimization_seed": seed,
                        "mode": mode,
                        "balanced_accuracy": result["model_results"][mode][
                            "test_metrics"
                        ]["balanced_accuracy"],
                    }
                )
        structural_scores.append(
            result["artifact_diagnostics"]["mask_topo_external"]["test"][
                "direct_structural_metrics"
            ]["balanced_accuracy"]
        )
    if truth is None or sources is None:
        raise RuntimeError("No DeepCrack results found.")
    prediction_arrays = {
        mode: np.stack(items) for mode, items in predictions.items()
    }
    aggregates = {}
    for mode, values in prediction_arrays.items():
        scores = np.array(
            [
                np.mean(values[index] == truth)
                for index in range(values.shape[0])
            ],
            dtype=np.float64,
        )
        aggregates[mode] = {
            "scores_by_seed": scores.tolist(),
            "mean_balanced_accuracy": float(scores.mean()),
            "sample_std_balanced_accuracy": float(scores.std(ddof=1)),
        }
    rng = np.random.default_rng(args.statistics_seed)
    contrasts = {}
    for name, (first, second) in CONTRASTS.items():
        contrasts[name] = cluster_statistics(
            truth,
            sources,
            prediction_arrays[first],
            prediction_arrays[second],
            args.bootstrap_repetitions,
            args.permutation_repetitions,
            rng,
        )
        contrasts[name]["first"] = first
        contrasts[name]["second"] = second
    adjusted = holm_adjust(
        {
            name: float(
                result["source_cluster_permutation_one_sided_p_raw"]
            )
            for name, result in contrasts.items()
        }
    )
    for name, value in adjusted.items():
        contrasts[name][
            "source_cluster_permutation_one_sided_p_holm"
        ] = value
    primary_names = (
        "topology_vs_grid",
        "topology_vs_mask_feature",
        "aligned_vs_shuffled_topology",
    )
    gates = {}
    for name in primary_names:
        contrast = contrasts[name]
        gates[f"{name}_mean_3pp_all_seeds_positive"] = (
            float(contrast["mean_gain"]) >= 0.03
            and all(gain > 0 for gain in contrast["paired_gains_by_seed"])
        )
        gates[f"{name}_cluster_ci_lower_above_zero"] = (
            float(contrast["source_cluster_bootstrap_95_ci"][0]) > 0
        )
    gates["all_seed_direct_structural_accuracy_at_least_70pct"] = all(
        score >= 0.70 for score in structural_scores
    )
    report = {
        "experiment_id": "deepcrack_masktopo_external_summary",
        "status": "completed",
        "optimization_seeds": args.seeds,
        "test_crop_count": int(truth.size),
        "test_source_image_count": int(np.unique(sources).size),
        "official_test_used_for_model_selection": False,
        "test_model_performance_previously_observed_before_model_runs": False,
        "test_aggregate_shortcut_audit_performed_before_model_runs": True,
        "runs": run_rows,
        "aggregates": aggregates,
        "direct_structural_accuracy_by_seed": structural_scores,
        "direct_structural_accuracy_mean": float(np.mean(structural_scores)),
        "contrasts": contrasts,
        "frozen_gates": gates,
    }
    report["verdict"] = (
        "EXTERNAL_CONFIRMATION_PASS_STRONG"
        if all(gates.values())
        else "EXTERNAL_CONFIRMATION_INCOMPLETE_OR_FAILED"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

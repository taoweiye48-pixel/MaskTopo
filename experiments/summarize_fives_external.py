from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from fives_external_gate import MECHANISM_MODELS
from strong_reducer_baselines import REDUCERS
from summarize_crackforest_mechanism_ablation import balanced_accuracy
from summarize_strong_reducer_baselines import cluster_bootstrap


SEEDS = (20260810, 20260811, 20260812)
MODELS = (*MECHANISM_MODELS, *REDUCERS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the frozen FIVES third-domain gate."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_fives_external_summary"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--statistics-seed", type=int, default=20260731)
    return parser.parse_args()


def score_summary(values: list[float]) -> dict[str, Any]:
    scores = np.asarray(values, dtype=np.float64)
    return {
        "scores_by_seed": scores.tolist(),
        "mean_balanced_accuracy": float(scores.mean()),
        "sample_std_balanced_accuracy": float(scores.std(ddof=1)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    lines = [
        "# FIVES third-domain external confirmation",
        "",
        "All values are held-out balanced accuracy, mean ± sample SD over",
        "three optimization seeds. Model checkpoints and mask calibration use",
        "development data only.",
        "",
        "| Model | Seed scores (%) | Mean ± sample SD |",
        "|---|---:|---:|",
    ]
    for name, aggregate in dataset["aggregates"].items():
        seeds = ", ".join(
            f"{100 * score:.2f}" for score in aggregate["scores_by_seed"]
        )
        lines.append(
            f"| {name} | {seeds} | "
            f"{100 * aggregate['mean_balanced_accuracy']:.2f} ± "
            f"{100 * aggregate['sample_std_balanced_accuracy']:.2f} |"
        )
    strongest = dataset["strongest_reducer_contrast"]
    shuffled = dataset["shuffled_topology_contrast"]
    identity = dataset["assignment_identity_contrast"]
    lines.extend(
        [
            "",
            f"Strongest reducer by three-seed mean: "
            f"`{dataset['strongest_reducer']}`.",
            "",
            f"MaskTopo gain over strongest reducer: "
            f"{100 * strongest['mean_gain']:+.2f} pp; source-image-cluster "
            f"95% CI [{100 * strongest['source_cluster_bootstrap_95_ci'][0]:+.2f}, "
            f"{100 * strongest['source_cluster_bootstrap_95_ci'][1]:+.2f}] pp.",
            "",
            f"MaskTopo gain over shuffled topology: "
            f"{100 * shuffled['mean_gain']:+.2f} pp; source-image-cluster "
            f"95% CI [{100 * shuffled['source_cluster_bootstrap_95_ci'][0]:+.2f}, "
            f"{100 * shuffled['source_cluster_bootstrap_95_ci'][1]:+.2f}] pp.",
            "",
            f"Exploratory MaskTopo gain over component assignment without graph "
            f"messages: {100 * identity['mean_gain']:+.2f} pp; "
            f"source-image-cluster 95% CI "
            f"[{100 * identity['source_cluster_bootstrap_95_ci'][0]:+.2f}, "
            f"{100 * identity['source_cluster_bootstrap_95_ci'][1]:+.2f}] pp.",
            "",
            f"Direct structural balanced accuracy: "
            f"{100 * dataset['direct_structural_accuracy']['mean']:.2f} ± "
            f"{100 * dataset['direct_structural_accuracy']['sample_std']:.2f}%.",
            "",
            f"Held-out mask F1: "
            f"{100 * dataset['heldout_mask_f1']['mean']:.2f} ± "
            f"{100 * dataset['heldout_mask_f1']['sample_std']:.2f}%.",
            "",
            "## Exploratory disease strata",
            "",
            "| Suffix | Test crops | Source images | MaskTopo | "
            "strongest reducer | gain |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for suffix, values in dataset["disease_stratified_exploratory"].items():
        lines.append(
            f"| {suffix} | {values['crop_count']} | "
            f"{values['source_image_count']} | "
            f"{100 * values['mean_mask_topo_balanced_accuracy']:.2f}% | "
            f"{100 * values['mean_strongest_reducer_balanced_accuracy']:.2f}% | "
            f"{100 * values['mean_gain']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Frozen gate",
            "",
            f"Verdict: **{report['verdict']}**",
            "",
        ]
    )
    for name, value in report["frozen_gates"].items():
        lines.append(f"- {name}: {value}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    truth: np.ndarray | None = None
    sources: np.ndarray | None = None
    prediction_lists: dict[str, list[np.ndarray]] = {
        name: [] for name in MODELS
    }
    scores: dict[str, list[float]] = {name: [] for name in MODELS}
    direct_structural: list[float] = []
    heldout_mask_f1: list[float] = []
    run_metadata: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_root = root / f"results_fives_seed{seed}"
        result = json.loads(
            (run_root / "result.json").read_text(encoding="utf-8")
        )
        if result["status"] != "completed":
            raise RuntimeError(f"FIVES seed {seed} is not completed.")
        if int(result["optimization_seed"]) != seed:
            raise RuntimeError(f"FIVES seed mismatch for {seed}.")
        if int(result["reduced_token_count"]) != 8:
            raise RuntimeError(f"FIVES token-count mismatch for {seed}.")
        with np.load(run_root / "heldout_predictions.npz") as archive:
            current_truth = archive["truth"].astype(np.uint8)
            current_sources = archive["source_id"].astype(str)
            current_predictions = {
                name: archive[name].astype(np.uint8) for name in MODELS
            }
        if truth is None:
            truth = current_truth
            sources = current_sources
        elif not np.array_equal(truth, current_truth) or not np.array_equal(
            sources, current_sources
        ):
            raise RuntimeError("FIVES held-out samples changed across seeds.")
        for name, prediction in current_predictions.items():
            score = balanced_accuracy(current_truth, prediction)
            recorded = result["model_results"][name]["test_metrics"][
                "balanced_accuracy"
            ]
            if not np.isclose(score, recorded):
                raise RuntimeError(
                    f"Prediction/metric mismatch for {seed}, {name}."
                )
            prediction_lists[name].append(prediction)
            scores[name].append(score)
            run_metadata.append(
                {
                    "seed": seed,
                    "model": name,
                    "balanced_accuracy": score,
                    "parameters": result["model_results"][name]["parameters"],
                }
            )
        direct_structural.append(
            result["artifact_diagnostics"]["mask_topo_external"]["test"][
                "direct_structural_metrics"
            ]["balanced_accuracy"]
        )
        heldout_mask_f1.append(result["heldout_mask_metrics"]["f1"])
    if truth is None or sources is None:
        raise RuntimeError("No FIVES runs were found.")
    predictions = {
        name: np.stack(values) for name, values in prediction_lists.items()
    }
    aggregates = {
        name: score_summary(values) for name, values in scores.items()
    }
    strongest = max(
        REDUCERS,
        key=lambda name: aggregates[name]["mean_balanced_accuracy"],
    )
    rng = np.random.default_rng(args.statistics_seed)
    strongest_contrast = cluster_bootstrap(
        truth,
        sources,
        predictions["mask_topo_external"],
        predictions[strongest],
        args.bootstrap_repetitions,
        rng,
    )
    disease_stratified: dict[str, Any] = {}
    for suffix in ("A", "D", "G", "N"):
        indices = np.flatnonzero(
            np.char.endswith(sources.astype(str), f"_{suffix}")
        )
        mask_scores = [
            balanced_accuracy(
                truth[indices],
                predictions["mask_topo_external"][seed_index, indices],
            )
            for seed_index in range(len(args.seeds))
        ]
        baseline_scores = [
            balanced_accuracy(
                truth[indices],
                predictions[strongest][seed_index, indices],
            )
            for seed_index in range(len(args.seeds))
        ]
        disease_stratified[suffix] = {
            "crop_count": int(indices.size),
            "source_image_count": int(np.unique(sources[indices]).size),
            "mask_topo_scores_by_seed": mask_scores,
            "strongest_reducer_scores_by_seed": baseline_scores,
            "mean_mask_topo_balanced_accuracy": float(np.mean(mask_scores)),
            "mean_strongest_reducer_balanced_accuracy": float(
                np.mean(baseline_scores)
            ),
            "mean_gain": float(
                np.mean(np.asarray(mask_scores) - np.asarray(baseline_scores))
            ),
        }
    shuffled_contrast = cluster_bootstrap(
        truth,
        sources,
        predictions["mask_topo_external"],
        predictions["shuffled_topology"],
        args.bootstrap_repetitions,
        rng,
    )
    identity_contrast = cluster_bootstrap(
        truth,
        sources,
        predictions["mask_topo_external"],
        predictions["mask_assignment_identity"],
        args.bootstrap_repetitions,
        rng,
    )
    direct = np.asarray(direct_structural, dtype=np.float64)
    mask_f1 = np.asarray(heldout_mask_f1, dtype=np.float64)
    gates = {
        "mean_gain_over_strongest_reducer_at_least_2pp": (
            strongest_contrast["mean_gain"] >= 0.02
        ),
        "all_seed_gains_over_strongest_reducer_positive": all(
            value > 0
            for value in strongest_contrast["paired_gains_by_seed"]
        ),
        "strongest_reducer_cluster_ci_lower_above_zero": (
            strongest_contrast["source_cluster_bootstrap_95_ci"][0] > 0
        ),
        "mean_direct_structural_accuracy_at_least_70pct": (
            direct.mean() >= 0.70
        ),
        "mean_gain_over_shuffled_topology_at_least_3pp": (
            shuffled_contrast["mean_gain"] >= 0.03
        ),
        "shuffled_topology_cluster_ci_lower_above_zero": (
            shuffled_contrast["source_cluster_bootstrap_95_ci"][0] > 0
        ),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    report = {
        "experiment_id": "fives_masktopo_external_summary",
        "status": "completed",
        "optimization_seeds": args.seeds,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "test_model_performance_previously_observed_before_frozen_run": False,
        "strongest_reducer_definition": (
            "highest three-seed mean held-out balanced accuracy among the four "
            "pre-specified in-framework strong reducers"
        ),
        "dataset": {
            "name": "FIVES",
            "test_crop_count": int(truth.size),
            "test_source_image_count": int(np.unique(sources).size),
            "aggregates": aggregates,
            "strongest_reducer": strongest,
            "strongest_reducer_contrast": strongest_contrast,
            "shuffled_topology_contrast": shuffled_contrast,
            "assignment_identity_contrast": identity_contrast,
            "disease_stratified_exploratory": disease_stratified,
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
            "runs": run_metadata,
        },
        "frozen_gates": gates,
        "verdict": (
            "FIVES_EXTERNAL_CONFIRMATION_PASS"
            if all(gates.values())
            else "FIVES_EXTERNAL_CONFIRMATION_FAIL"
        ),
        "limitations": [
            "FIVES release filenames do not expose subject identifiers, so "
            "patient-level split overlap cannot be independently audited.",
            "Strong reducers are controlled in-framework implementations.",
            "The task is derived connectivity reasoning, not clinical vessel "
            "segmentation or diagnosis.",
        ],
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    with (output / "source_data.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "model",
                "balanced_accuracy",
                "parameters",
                "direct_structural_balanced_accuracy",
                "heldout_mask_f1",
            ),
        )
        writer.writeheader()
        for row in run_metadata:
            seed_index = args.seeds.index(row["seed"])
            writer.writerow(
                {
                    **row,
                    "direct_structural_balanced_accuracy": (
                        direct_structural[seed_index]
                        if row["model"] == "mask_topo_external"
                        else ""
                    ),
                    "heldout_mask_f1": (
                        heldout_mask_f1[seed_index]
                        if row["model"] == "mask_topo_external"
                        else ""
                    ),
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

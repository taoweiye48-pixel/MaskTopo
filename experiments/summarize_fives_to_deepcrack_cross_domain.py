from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from summarize_crackforest_mechanism_ablation import balanced_accuracy
from summarize_strong_reducer_baselines import cluster_bootstrap


SEEDS = (20260810, 20260811, 20260812)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize frozen FIVES-to-DeepCrack transfer."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("results_fives_to_deepcrack_cross_domain")
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--statistics-seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    report = json.loads((root / "result.json").read_text(encoding="utf-8"))
    seed_predictions: dict[int, dict[str, np.ndarray]] = {}
    truth: np.ndarray | None = None
    sources: np.ndarray | None = None
    model_names: list[str] | None = None
    for seed in report["seeds"]:
        with np.load(root / f"heldout_predictions_seed{seed}.npz") as archive:
            current_truth = archive["truth"].astype(np.uint8)
            current_sources = archive["source_id"].astype(str)
            names = [
                name for name in archive.files if name not in {"truth", "source_id"}
            ]
            current = {
                name: archive[name].astype(np.uint8) for name in names
            }
        if truth is None:
            truth = current_truth
            sources = current_sources
            model_names = names
        elif not np.array_equal(truth, current_truth) or not np.array_equal(
            sources, current_sources
        ):
            raise RuntimeError("Target held-out samples changed across seeds.")
        if model_names != names:
            raise RuntimeError("Model set changed across seeds.")
        seed_predictions[int(seed)] = current
    if truth is None or sources is None or model_names is None:
        raise RuntimeError("No prediction files found.")

    stacked = {
        name: np.stack([seed_predictions[seed][name] for seed in report["seeds"]])
        for name in model_names
    }
    aggregates = {
        name: {
            "scores_by_seed": [
                balanced_accuracy(truth, prediction)
                for prediction in stacked[name]
            ],
            "mean": float(
                np.mean(
                    [balanced_accuracy(truth, prediction) for prediction in stacked[name]]
                )
            ),
            "sample_std": float(
                np.std(
                    [balanced_accuracy(truth, prediction) for prediction in stacked[name]],
                    ddof=1,
                )
            ),
        }
        for name in model_names
    }
    rng = np.random.default_rng(args.statistics_seed)
    contrasts: dict[str, Any] = {}
    for name in model_names:
        if name == "mask_topo_external":
            continue
        contrasts[name] = cluster_bootstrap(
            truth,
            sources,
            stacked["mask_topo_external"],
            stacked[name],
            args.bootstrap_repetitions,
            rng,
        )
    source_count = int(np.unique(sources).size)
    output = {
        "experiment_id": "fives_to_deepcrack_cross_domain_transfer_summary",
        "status": "completed",
        "source_dataset": report["source_dataset"],
        "target_dataset": report["target_dataset"],
        "source_image_count": source_count,
        "aggregates": aggregates,
        "masktopo_minus_baseline_source_cluster_bootstrap": contrasts,
        "target_mask_f1_mean": report["target_mask_f1_mean"],
        "target_mask_f1_sample_std": report["target_mask_f1_sample_std"],
        "no_target_dense_mask_training": True,
        "no_target_classifier_training": True,
        "protocol": report["protocol"],
    }
    (root / "statistical_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# FIVES → DeepCrack frozen transfer statistics",
        "",
        "No DeepCrack dense mask or classifier training was used.",
        "",
        "| Model | Seed scores (%) | Mean ± sample SD |",
        "|---|---:|---:|",
    ]
    for name, values in aggregates.items():
        lines.append(
            f"| {name} | "
            + ", ".join(f"{100 * value:.2f}" for value in values["scores_by_seed"])
            + f" | {100 * values['mean']:.2f} ± {100 * values['sample_std']:.2f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Mean gain (pp) | Source-image cluster 95% CI (pp) |",
            "|---|---:|---:|",
        ]
    )
    for name, values in contrasts.items():
        lines.append(
            f"| MaskTopo − {name} | {100 * values['mean_gain']:+.2f} | "
            f"[{100 * values['source_cluster_bootstrap_95_ci'][0]:+.2f}, "
            f"{100 * values['source_cluster_bootstrap_95_ci'][1]:+.2f}] |"
        )
    lines.extend(
        [
            "",
            f"Source-image clusters: {source_count}.",
            f"Frozen target mask F1: {100 * report['target_mask_f1_mean']:.2f} ± "
            f"{100 * report['target_mask_f1_sample_std']:.2f}%.",
            "",
        ]
    )
    (root / "STATISTICAL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

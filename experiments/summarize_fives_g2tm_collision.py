from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from summarize_crackforest_mechanism_ablation import balanced_accuracy
from summarize_strong_reducer_baselines import cluster_bootstrap


SEEDS = (20260810, 20260811, 20260812)
NEW_MODELS = ("g2tm_fixedk_feature", "g2tm_fixedk_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the FIVES G2TM collision gate."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_fives_g2tm_collision_summary"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--statistics-seed", type=int, default=20260731)
    return parser.parse_args()


def aggregate(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "scores_by_seed": array.tolist(),
        "mean_balanced_accuracy": float(array.mean()),
        "sample_std_balanced_accuracy": float(array.std(ddof=1)),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    truth: np.ndarray | None = None
    sources: np.ndarray | None = None
    predictions: dict[str, list[np.ndarray]] = {
        "mask_topo_external": [],
        **{name: [] for name in NEW_MODELS},
    }
    scores: dict[str, list[float]] = {
        name: [] for name in predictions
    }
    latencies: dict[str, list[float]] = {
        name: [] for name in NEW_MODELS
    }
    for seed in args.seeds:
        old_dir = root / f"results_fives_seed{seed}"
        new_dir = root / f"results_fives_g2tm_collision_seed{seed}"
        old_result = json.loads(
            (old_dir / "result.json").read_text(encoding="utf-8")
        )
        new_result = json.loads(
            (new_dir / "result.json").read_text(encoding="utf-8")
        )
        if old_result["status"] != "completed" or new_result["status"] != "completed":
            raise RuntimeError(f"Incomplete run for seed {seed}.")
        with np.load(old_dir / "heldout_predictions.npz") as old_archive:
            old_truth = old_archive["truth"].astype(np.uint8)
            old_sources = old_archive["source_id"].astype(str)
            old_mask_topo = old_archive["mask_topo_external"].astype(np.uint8)
        with np.load(new_dir / "heldout_predictions.npz") as new_archive:
            new_truth = new_archive["truth"].astype(np.uint8)
            new_sources = new_archive["source_id"].astype(str)
            new_predictions = {
                name: new_archive[name].astype(np.uint8)
                for name in NEW_MODELS
            }
        if not np.array_equal(old_truth, new_truth) or not np.array_equal(
            old_sources, new_sources
        ):
            raise RuntimeError(f"Held-out sample order changed for seed {seed}.")
        if truth is None:
            truth = old_truth
            sources = old_sources
        elif not np.array_equal(truth, old_truth) or not np.array_equal(
            sources, old_sources
        ):
            raise RuntimeError("Held-out samples changed across seeds.")

        current_predictions = {
            "mask_topo_external": old_mask_topo,
            **new_predictions,
        }
        for name, prediction in current_predictions.items():
            measured = balanced_accuracy(old_truth, prediction)
            if name == "mask_topo_external":
                recorded = old_result["model_results"][name]["test_metrics"][
                    "balanced_accuracy"
                ]
            else:
                recorded = new_result["model_results"][name]["test_metrics"][
                    "balanced_accuracy"
                ]
                latencies[name].append(
                    float(
                        new_result["model_results"][name]["inference"][
                            "latency_ms_per_sample"
                        ]
                    )
                )
            if not np.isclose(measured, recorded):
                raise RuntimeError(
                    f"Prediction/metric mismatch for seed={seed}, model={name}."
                )
            predictions[name].append(prediction)
            scores[name].append(measured)

    if truth is None or sources is None:
        raise RuntimeError("No completed runs found.")
    stacked = {
        name: np.stack(values) for name, values in predictions.items()
    }
    aggregates = {name: aggregate(values) for name, values in scores.items()}
    rng = np.random.default_rng(args.statistics_seed)
    contrasts: dict[str, Any] = {}
    for name in NEW_MODELS:
        contrasts[name] = cluster_bootstrap(
            truth,
            sources,
            stacked["mask_topo_external"],
            stacked[name],
            args.bootstrap_repetitions,
            rng,
        )

    strongest = max(
        NEW_MODELS,
        key=lambda name: aggregates[name]["mean_balanced_accuracy"],
    )
    gap = (
        aggregates["mask_topo_external"]["mean_balanced_accuracy"]
        - aggregates[strongest]["mean_balanced_accuracy"]
    )
    ci = contrasts[strongest]["source_cluster_bootstrap_95_ci"]
    if gap >= 0.03 and ci[0] > 0:
        verdict = "CURRENT_MASKTOPO_SURVIVES_G2TM_COLLISION"
    elif gap <= 0.02 and ci[0] <= 0:
        verdict = "PIVOT_TO_QUOTIENT_GRAPH_REQUIRED"
    else:
        verdict = "G2TM_COLLISION_INCONCLUSIVE_REQUIRES_STANDARD_VIT"

    report = {
        "experiment_id": "fives_g2tm_collision_gate_summary",
        "status": "completed",
        "dataset": "FIVES",
        "seeds": list(args.seeds),
        "token_count": 8,
        "aggregates": aggregates,
        "contrasts_masktopo_minus_g2tm": contrasts,
        "strongest_g2tm_variant": strongest,
        "strongest_gap": gap,
        "strongest_ci": ci,
        "latency_ms_per_sample": {
            name: aggregate(latencies[name]) for name in NEW_MODELS
        },
        "source_image_count": int(np.unique(sources).size),
        "verdict": verdict,
        "interpretation": (
            "The current MaskTopo mechanism survives this matched collision "
            "test, but the comparison is retrospective and uses a fixed-budget "
            "adaptation rather than the original variable-threshold G2TM "
            "training pipeline."
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# FIVES G2TM collision gate summary",
        "",
        "Retrospective diagnostic: the FIVES held-out results were already "
        "observed before this experiment.",
        "",
        "| Model | Seed scores (%) | Mean ± sample SD |",
        "|---|---:|---:|",
    ]
    for name, values in aggregates.items():
        lines.append(
            f"| {name} | "
            + ", ".join(f"{100 * value:.2f}" for value in values["scores_by_seed"])
            + f" | {100 * values['mean_balanced_accuracy']:.2f} ± "
            + f"{100 * values['sample_std_balanced_accuracy']:.2f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Mean gain (pp) | Source-image cluster 95% CI (pp) |",
            "|---|---:|---:|",
        ]
    )
    for name in NEW_MODELS:
        contrast = contrasts[name]
        lines.append(
            f"| MaskTopo − {name} | {100 * contrast['mean_gain']:+.2f} | "
            f"[{100 * contrast['source_cluster_bootstrap_95_ci'][0]:+.2f}, "
            f"{100 * contrast['source_cluster_bootstrap_95_ci'][1]:+.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Model | Mean latency (ms/sample) | Sample SD |",
            "|---|---:|---:|",
        ]
    )
    for name, values in report["latency_ms_per_sample"].items():
        lines.append(
            f"| {name} | {values['mean_balanced_accuracy']:.3f} | "
            f"{values['sample_std_balanced_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Strongest G2TM variant: `{strongest}`.",
            f"Verdict: **{verdict}**.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

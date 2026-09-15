from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from crackforest_mask_topo import find_cached_split
from strong_reducer_baselines import REDUCERS
from summarize_crackforest_mechanism_ablation import balanced_accuracy


SEEDS = (20260810, 20260811, 20260812)
MASK_TOPO_KEYS = {
    "crackforest": "mask_topo_recheck",
    "deepcrack": "mask_topo_external",
}
OLD_PREFIXES = {
    "crackforest": "results_crackforest_mechanism_seed",
    "deepcrack": "results_deepcrack_seed",
}
NEW_PREFIXES = {
    "crackforest": "results_strong_reducer_crackforest_seed",
    "deepcrack": "results_strong_reducer_deepcrack_seed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize frozen TopoBridge strong-reducer comparisons."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(".")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_strong_reducer_summary"),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS)
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--statistics-seed", type=int, default=20260731)
    parser.add_argument("--crackforest-cache", type=Path, default=Path("real_cache"))
    return parser.parse_args()


def aggregate_scores(
    truth: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    scores = np.asarray(
        [
            balanced_accuracy(truth, predictions[index])
            for index in range(predictions.shape[0])
        ],
        dtype=np.float64,
    )
    return {
        "scores_by_seed": scores.tolist(),
        "mean_balanced_accuracy": float(scores.mean()),
        "sample_std_balanced_accuracy": float(scores.std(ddof=1)),
    }


def cluster_bootstrap(
    truth: np.ndarray,
    sources: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    unique_sources = np.unique(sources)
    source_indices = {
        source: np.flatnonzero(sources == source) for source in unique_sources
    }
    gains_by_seed = np.asarray(
        [
            balanced_accuracy(truth, first[index])
            - balanced_accuracy(truth, second[index])
            for index in range(first.shape[0])
        ],
        dtype=np.float64,
    )
    bootstrap = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(
            unique_sources, size=unique_sources.size, replace=True
        )
        indices = np.concatenate([source_indices[source] for source in sampled])
        bootstrap[repetition] = np.mean(
            [
                balanced_accuracy(truth[indices], first[seed, indices])
                - balanced_accuracy(truth[indices], second[seed, indices])
                for seed in range(first.shape[0])
            ]
        )
    return {
        "paired_gains_by_seed": gains_by_seed.tolist(),
        "mean_gain": float(gains_by_seed.mean()),
        "sample_std_gain": float(gains_by_seed.std(ddof=1)),
        "source_cluster_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "source_image_count": int(unique_sources.size),
    }


def load_dataset(
    root: Path,
    dataset: str,
    seeds: list[int],
    crackforest_cache: Path,
) -> dict[str, Any]:
    predictions: dict[str, list[np.ndarray]] = {
        "mask_topo": [], **{name: [] for name in REDUCERS}
    }
    truth: np.ndarray | None = None
    sources: np.ndarray | None = None
    run_metadata: list[dict[str, Any]] = []
    for seed in seeds:
        old_root = root / f"{OLD_PREFIXES[dataset]}{seed}"
        new_root = root / f"{NEW_PREFIXES[dataset]}{seed}"
        old_result = json.loads(
            (old_root / "result.json").read_text(encoding="utf-8")
        )
        new_result = json.loads(
            (new_root / "result.json").read_text(encoding="utf-8")
        )
        if int(new_result["optimization_seed"]) != seed:
            raise RuntimeError(f"Strong reducer seed mismatch: {dataset}, {seed}")
        with np.load(old_root / "heldout_predictions.npz") as old_archive:
            old_truth = old_archive["truth"].astype(np.uint8)
            mask_topo = old_archive[MASK_TOPO_KEYS[dataset]].astype(np.uint8)
            old_sources = (
                old_archive["source_id"].astype(str)
                if "source_id" in old_archive.files
                else None
            )
        with np.load(new_root / "heldout_predictions.npz") as new_archive:
            new_truth = new_archive["truth"].astype(np.uint8)
            new_sources = (
                new_archive["source_id"].astype(str)
                if "source_id" in new_archive.files
                else None
            )
            current_predictions = {
                name: new_archive[name].astype(np.uint8)
                for name in REDUCERS
            }
        if not np.array_equal(old_truth, new_truth):
            raise RuntimeError(f"Old/new held-out labels differ: {dataset}, {seed}")
        if truth is None:
            truth = old_truth
        elif not np.array_equal(truth, old_truth):
            raise RuntimeError(f"Held-out labels changed across seeds: {dataset}")
        if dataset == "deepcrack":
            if old_sources is None or new_sources is None:
                raise RuntimeError("DeepCrack source IDs are missing.")
            if not np.array_equal(old_sources, new_sources):
                raise RuntimeError("DeepCrack source IDs differ between runs.")
            if sources is None:
                sources = old_sources
            elif not np.array_equal(sources, old_sources):
                raise RuntimeError("DeepCrack source IDs changed across seeds.")
        predictions["mask_topo"].append(mask_topo)
        for name, value in current_predictions.items():
            predictions[name].append(value)
        mask_score = balanced_accuracy(old_truth, mask_topo)
        recorded_mask_score = old_result["model_results"][
            MASK_TOPO_KEYS[dataset]
        ]["test_metrics"]["balanced_accuracy"]
        if not np.isclose(mask_score, recorded_mask_score):
            raise RuntimeError(f"MaskTopo prediction mismatch: {dataset}, {seed}")
        run_metadata.append(
            {
                "seed": seed,
                "model": "mask_topo",
                "balanced_accuracy": mask_score,
                "parameters": old_result["model_results"][
                    MASK_TOPO_KEYS[dataset]
                ]["parameters"],
            }
        )
        for name in REDUCERS:
            score = balanced_accuracy(old_truth, current_predictions[name])
            recorded = new_result["model_results"][name]["test_metrics"][
                "balanced_accuracy"
            ]
            if not np.isclose(score, recorded):
                raise RuntimeError(
                    f"Strong reducer prediction mismatch: {dataset}, {seed}, {name}"
                )
            run_metadata.append(
                {
                    "seed": seed,
                    "model": name,
                    "balanced_accuracy": score,
                    "parameters": new_result["model_results"][name]["parameters"],
                    "latency_ms_per_sample": new_result["model_results"][name][
                        "inference"
                    ]["latency_ms_per_sample"],
                    "peak_cuda_memory_mb": new_result["model_results"][name][
                        "inference"
                    ]["peak_cuda_memory_mb"],
                }
            )
    if truth is None:
        raise RuntimeError(f"No results found for {dataset}.")
    if dataset == "crackforest":
        arrays = find_cached_split(
            crackforest_cache.resolve(), "test", 320, 22260810, 64
        )
        sources = arrays["source_id"].astype(str)
    if sources is None or sources.shape[0] != truth.shape[0]:
        raise RuntimeError(f"Source IDs do not align for {dataset}.")
    stacked = {
        name: np.stack(values) for name, values in predictions.items()
    }
    aggregates = {
        name: aggregate_scores(truth, values)
        for name, values in stacked.items()
    }
    strongest = max(
        REDUCERS,
        key=lambda name: aggregates[name]["mean_balanced_accuracy"],
    )
    return {
        "truth": truth,
        "sources": sources,
        "predictions": stacked,
        "aggregates": aggregates,
        "strongest_baseline": strongest,
        "runs": run_metadata,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# TopoBridge strong reducer benchmark",
        "",
        "This is a frozen retrospective comparison because the held-out labels had",
        "already been observed in earlier mechanism experiments.",
        "",
    ]
    for dataset in ("crackforest", "deepcrack"):
        current = report["datasets"][dataset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Model | Seed scores | Mean ± sample SD |",
                "|---|---:|---:|",
            ]
        )
        for name, values in current["aggregates"].items():
            scores = ", ".join(
                f"{100 * score:.2f}" for score in values["scores_by_seed"]
            )
            lines.append(
                f"| {name} | {scores} | "
                f"{100 * values['mean_balanced_accuracy']:.2f} ± "
                f"{100 * values['sample_std_balanced_accuracy']:.2f} |"
            )
        contrast = current["strongest_baseline_contrast"]
        lines.extend(
            [
                "",
                f"Strongest reducer: `{current['strongest_baseline']}`.",
                "",
                f"MaskTopo mean gain: {100 * contrast['mean_gain']:.2f} pp; "
                f"source-cluster 95% CI "
                f"[{100 * contrast['source_cluster_bootstrap_95_ci'][0]:.2f}, "
                f"{100 * contrast['source_cluster_bootstrap_95_ci'][1]:.2f}] pp.",
                "",
            ]
        )
    lines.extend(
        [
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
    rng = np.random.default_rng(args.statistics_seed)
    datasets = {}
    for dataset in ("crackforest", "deepcrack"):
        loaded = load_dataset(
            root, dataset, args.seeds, args.crackforest_cache
        )
        contrasts = {}
        for name in REDUCERS:
            contrasts[name] = cluster_bootstrap(
                loaded["truth"],
                loaded["sources"],
                loaded["predictions"]["mask_topo"],
                loaded["predictions"][name],
                args.bootstrap_repetitions,
                rng,
            )
        strongest = loaded["strongest_baseline"]
        datasets[dataset] = {
            "test_crop_count": int(loaded["truth"].size),
            "test_source_image_count": int(np.unique(loaded["sources"]).size),
            "aggregates": loaded["aggregates"],
            "strongest_baseline": strongest,
            "contrasts": contrasts,
            "strongest_baseline_contrast": contrasts[strongest],
            "runs": loaded["runs"],
        }
    gates: dict[str, bool] = {}
    for dataset, current in datasets.items():
        contrast = current["strongest_baseline_contrast"]
        gates[f"{dataset}_mean_gain_at_least_2pp"] = (
            contrast["mean_gain"] >= 0.02
        )
        gates[f"{dataset}_all_seed_gains_positive"] = all(
            gain > 0 for gain in contrast["paired_gains_by_seed"]
        )
    gates["deepcrack_source_cluster_ci_lower_above_zero"] = (
        datasets["deepcrack"]["strongest_baseline_contrast"][
            "source_cluster_bootstrap_95_ci"
        ][0]
        > 0
    )
    report = {
        "experiment_id": "topobridge_strong_reducer_summary",
        "status": "completed",
        "optimization_seeds": args.seeds,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "test_was_previously_observed": True,
        "strongest_baseline_definition": (
            "highest three-seed mean held-out balanced accuracy within each dataset"
        ),
        "datasets": datasets,
        "frozen_gates": gates,
        "verdict": (
            "STRONG_REDUCER_GATE_PASS"
            if all(gates.values())
            else "STRONG_REDUCER_GATE_FAIL"
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

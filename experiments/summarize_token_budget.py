from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from crackforest_mask_topo import find_cached_split
from strong_reducer_baselines import REDUCERS
from summarize_crackforest_mechanism_ablation import balanced_accuracy


BUDGETS = (8, 16, 32, 64)
MASK_KEYS = {
    "crackforest": "mask_topo_recheck",
    "deepcrack": "mask_topo_external",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize frozen TopoBridge token-budget experiments."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_token_budget_summary"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260810, 20260811, 20260812],
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--statistics-seed", type=int, default=20260731)
    parser.add_argument("--crackforest-cache", type=Path, default=Path("real_cache"))
    return parser.parse_args()


def aggregate(
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
    indices_by_source = {
        source: np.flatnonzero(sources == source)
        for source in unique_sources
    }
    gains = np.asarray(
        [
            balanced_accuracy(truth, first[index])
            - balanced_accuracy(truth, second[index])
            for index in range(first.shape[0])
        ]
    )
    bootstrap = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(
            unique_sources, size=unique_sources.size, replace=True
        )
        indices = np.concatenate(
            [indices_by_source[source] for source in sampled]
        )
        bootstrap[repetition] = np.mean(
            [
                balanced_accuracy(truth[indices], first[seed, indices])
                - balanced_accuracy(truth[indices], second[seed, indices])
                for seed in range(first.shape[0])
            ]
        )
    return {
        "paired_gains_by_seed": gains.tolist(),
        "mean_gain": float(gains.mean()),
        "sample_std_gain": float(gains.std(ddof=1)),
        "source_cluster_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def old_roots(root: Path, dataset: str, seed: int) -> tuple[Path, Path]:
    mask = (
        root / f"results_crackforest_mechanism_seed{seed}"
        if dataset == "crackforest"
        else root / f"results_deepcrack_seed{seed}"
    )
    reducers = root / f"results_strong_reducer_{dataset}_seed{seed}"
    return reducers, mask


def new_roots(
    root: Path, dataset: str, token_count: int, seed: int
) -> tuple[Path, Path]:
    reducers = (
        root
        / f"results_token_budget_{dataset}_k{token_count}_seed{seed}"
    )
    mask = (
        root
        / f"results_token_budget_masktopo_{dataset}_k{token_count}_seed{seed}"
    )
    return reducers, mask


def load_budget(
    root: Path,
    dataset: str,
    token_count: int,
    seeds: list[int],
) -> dict[str, Any]:
    prediction_lists: dict[str, list[np.ndarray]] = {
        "mask_topo": [],
        **{name: [] for name in REDUCERS},
    }
    truth: np.ndarray | None = None
    sources: np.ndarray | None = None
    rows = []
    preprocessing = []
    for seed in seeds:
        if token_count == 16:
            reducer_root, mask_root = old_roots(root, dataset, seed)
        else:
            reducer_root, mask_root = new_roots(
                root, dataset, token_count, seed
            )
        reducer_result = json.loads(
            (reducer_root / "result.json").read_text(encoding="utf-8")
        )
        mask_result = json.loads(
            (mask_root / "result.json").read_text(encoding="utf-8")
        )
        with np.load(
            reducer_root / "heldout_predictions.npz"
        ) as reducer_archive:
            reducer_truth = reducer_archive["truth"].astype(np.uint8)
            reducer_predictions = {
                name: reducer_archive[name].astype(np.uint8)
                for name in REDUCERS
            }
            reducer_sources = (
                reducer_archive["source_id"].astype(str)
                if "source_id" in reducer_archive.files
                else None
            )
        with np.load(mask_root / "heldout_predictions.npz") as mask_archive:
            mask_truth = mask_archive["truth"].astype(np.uint8)
            mask_prediction = mask_archive[
                MASK_KEYS[dataset] if token_count == 16 else "mask_topo"
            ].astype(np.uint8)
            mask_sources = (
                mask_archive["source_id"].astype(str)
                if "source_id" in mask_archive.files
                else None
            )
        if not np.array_equal(reducer_truth, mask_truth):
            raise RuntimeError(
                f"Reducer/MaskTopo labels differ: {dataset}, K={token_count}, {seed}"
            )
        if truth is None:
            truth = mask_truth
        elif not np.array_equal(truth, mask_truth):
            raise RuntimeError(
                f"Held-out labels changed: {dataset}, K={token_count}"
            )
        if dataset == "deepcrack":
            if reducer_sources is None or mask_sources is None:
                raise RuntimeError("DeepCrack source IDs are missing.")
            if not np.array_equal(reducer_sources, mask_sources):
                raise RuntimeError("DeepCrack source IDs differ.")
            if sources is None:
                sources = mask_sources
            elif not np.array_equal(sources, mask_sources):
                raise RuntimeError("DeepCrack source IDs changed.")
        prediction_lists["mask_topo"].append(mask_prediction)
        for name in REDUCERS:
            prediction_lists[name].append(reducer_predictions[name])
        mask_score = balanced_accuracy(mask_truth, mask_prediction)
        recorded_mask = (
            mask_result["model_results"][MASK_KEYS[dataset]][
                "test_metrics"
            ]["balanced_accuracy"]
            if token_count == 16
            else mask_result["model_result"]["test_metrics"][
                "balanced_accuracy"
            ]
        )
        if not np.isclose(mask_score, recorded_mask):
            raise RuntimeError(
                f"MaskTopo metric mismatch: {dataset}, K={token_count}, {seed}"
            )
        rows.append(
            {
                "dataset": dataset,
                "token_count": token_count,
                "seed": seed,
                "model": "mask_topo",
                "balanced_accuracy": mask_score,
            }
        )
        if token_count != 16:
            preprocessing.append(
                mask_result["topology_preprocessing"]["test"][
                    "milliseconds_per_sample"
                ]
            )
        for name in REDUCERS:
            score = balanced_accuracy(
                reducer_truth, reducer_predictions[name]
            )
            recorded = reducer_result["model_results"][name][
                "test_metrics"
            ]["balanced_accuracy"]
            if not np.isclose(score, recorded):
                raise RuntimeError(
                    f"Reducer metric mismatch: {dataset}, K={token_count}, "
                    f"{seed}, {name}"
                )
            rows.append(
                {
                    "dataset": dataset,
                    "token_count": token_count,
                    "seed": seed,
                    "model": name,
                    "balanced_accuracy": score,
                }
            )
    if truth is None:
        raise RuntimeError("No token-budget results were loaded.")
    stacked = {
        name: np.stack(values)
        for name, values in prediction_lists.items()
    }
    aggregates = {
        name: aggregate(truth, values)
        for name, values in stacked.items()
    }
    return {
        "truth": truth,
        "sources": sources,
        "predictions": stacked,
        "aggregates": aggregates,
        "rows": rows,
        "topology_preprocessing_ms_per_sample_by_seed": preprocessing,
    }


def average_log_budget_auc(values: list[float]) -> float:
    x = np.log2(np.asarray(BUDGETS, dtype=np.float64))
    return float(np.trapezoid(np.asarray(values), x) / (x[-1] - x[0]))


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# TopoBridge token-budget validation",
        "",
        "All values are held-out balanced accuracy, mean ± sample SD over "
        "three optimization seeds.",
        "",
    ]
    for dataset in ("crackforest", "deepcrack"):
        current = report["datasets"][dataset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| K | MaskTopo | strongest reducer | reducer score | gain | "
                "source-cluster 95% CI |",
                "|---:|---:|---|---:|---:|---:|",
            ]
        )
        for token_count in BUDGETS:
            budget = current["budgets"][str(token_count)]
            mask = budget["aggregates"]["mask_topo"]
            strongest = budget["strongest_baseline"]
            base = budget["aggregates"][strongest]
            contrast = budget["strongest_baseline_contrast"]
            ci = contrast["source_cluster_bootstrap_95_ci"]
            lines.append(
                f"| {token_count} | "
                f"{100 * mask['mean_balanced_accuracy']:.2f} ± "
                f"{100 * mask['sample_std_balanced_accuracy']:.2f} | "
                f"{strongest} | "
                f"{100 * base['mean_balanced_accuracy']:.2f} ± "
                f"{100 * base['sample_std_balanced_accuracy']:.2f} | "
                f"{100 * contrast['mean_gain']:+.2f} pp | "
                f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] pp |"
            )
        lines.extend(
            [
                "",
                f"MaskTopo log-budget AUC: "
                f"{100 * current['mask_topo_log_budget_auc']:.2f}%.",
                "",
                f"Strongest-baseline envelope AUC: "
                f"{100 * current['baseline_envelope_log_budget_auc']:.2f}%.",
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
    datasets: dict[str, Any] = {}
    source_rows = []
    for dataset in ("crackforest", "deepcrack"):
        budget_results: dict[str, Any] = {}
        reference_truth: np.ndarray | None = None
        reference_sources: np.ndarray | None = None
        if dataset == "crackforest":
            arrays = find_cached_split(
                args.crackforest_cache.resolve(),
                "test",
                320,
                22260810,
                64,
            )
            reference_sources = arrays["source_id"].astype(str)
        mask_curve = []
        envelope_curve = []
        for token_count in BUDGETS:
            loaded = load_budget(
                root, dataset, token_count, args.seeds
            )
            if reference_truth is None:
                reference_truth = loaded["truth"]
            elif not np.array_equal(reference_truth, loaded["truth"]):
                raise RuntimeError(
                    f"Test labels differ across K for {dataset}."
                )
            if dataset == "deepcrack":
                if reference_sources is None:
                    reference_sources = loaded["sources"]
                elif not np.array_equal(
                    reference_sources, loaded["sources"]
                ):
                    raise RuntimeError(
                        "DeepCrack source IDs differ across K."
                    )
            if reference_sources is None:
                raise RuntimeError(f"Missing source IDs for {dataset}.")
            strongest = max(
                REDUCERS,
                key=lambda name: loaded["aggregates"][name][
                    "mean_balanced_accuracy"
                ],
            )
            contrast = cluster_bootstrap(
                loaded["truth"],
                reference_sources,
                loaded["predictions"]["mask_topo"],
                loaded["predictions"][strongest],
                args.bootstrap_repetitions,
                rng,
            )
            budget_results[str(token_count)] = {
                "aggregates": loaded["aggregates"],
                "strongest_baseline": strongest,
                "strongest_baseline_contrast": contrast,
                "topology_preprocessing_ms_per_sample_by_seed": loaded[
                    "topology_preprocessing_ms_per_sample_by_seed"
                ],
            }
            source_rows.extend(loaded["rows"])
            mask_curve.append(
                loaded["aggregates"]["mask_topo"][
                    "mean_balanced_accuracy"
                ]
            )
            envelope_curve.append(
                loaded["aggregates"][strongest][
                    "mean_balanced_accuracy"
                ]
            )
        mask_auc = average_log_budget_auc(mask_curve)
        envelope_auc = average_log_budget_auc(envelope_curve)
        datasets[dataset] = {
            "budgets": budget_results,
            "mask_topo_log_budget_auc": mask_auc,
            "baseline_envelope_log_budget_auc": envelope_auc,
            "log_budget_auc_gain": mask_auc - envelope_auc,
        }
    gates: dict[str, bool] = {}
    for dataset, current in datasets.items():
        k8 = current["budgets"]["8"]["strongest_baseline_contrast"]
        gates[f"{dataset}_k8_mean_gain_at_least_2pp"] = (
            k8["mean_gain"] >= 0.02
        )
        gates[f"{dataset}_k8_all_seed_gains_positive"] = all(
            gain > 0 for gain in k8["paired_gains_by_seed"]
        )
        gates[f"{dataset}_log_budget_auc_gain_at_least_2pp"] = (
            current["log_budget_auc_gain"] >= 0.02
        )
        k8_mask = current["budgets"]["8"]["aggregates"]["mask_topo"][
            "mean_balanced_accuracy"
        ]
        k16_mask = current["budgets"]["16"]["aggregates"]["mask_topo"][
            "mean_balanced_accuracy"
        ]
        gates[f"{dataset}_k8_no_more_than_5pp_below_k16"] = (
            k8_mask >= k16_mask - 0.05
        )
    gates["deepcrack_k8_cluster_ci_lower_above_zero"] = (
        datasets["deepcrack"]["budgets"]["8"][
            "strongest_baseline_contrast"
        ]["source_cluster_bootstrap_95_ci"][0]
        > 0
    )
    report = {
        "experiment_id": "topobridge_token_budget_summary",
        "status": "completed",
        "optimization_seeds": args.seeds,
        "budgets": list(BUDGETS),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "test_was_previously_observed": True,
        "strongest_baseline_definition": (
            "highest three-seed mean at each dataset and token budget"
        ),
        "datasets": datasets,
        "frozen_gates": gates,
        "verdict": (
            "TOKEN_BUDGET_GATE_PASS"
            if all(gates.values())
            else "TOKEN_BUDGET_GATE_FAIL"
        ),
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
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "token_count",
                "seed",
                "model",
                "balanced_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(source_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


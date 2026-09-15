from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from strong_reducer_baselines import REDUCERS
from summarize_crackforest_mechanism_ablation import balanced_accuracy


DATASETS = ("crackforest", "deepcrack")
BUDGETS = (8, 16, 32, 64)
PH_MODELS = ("ph_only", "ph_guided")
SEEDS = (20260810, 20260811, 20260812)
MASK_KEYS = {
    "crackforest": "mask_topo_recheck",
    "deepcrack": "mask_topo_external",
}
BATCH_SIZES = (1, 8, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize TB-B-260802-023 PH-inclusive Pareto runs."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("results_ph_pareto_summary")
    )
    parser.add_argument("--crackforest-cache", type=Path, default=Path("real_cache"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--statistics-seed", type=int, default=20260802)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prior_roots(
    root: Path, dataset: str, token_count: int, seed: int
) -> tuple[Path, Path]:
    if token_count == 16:
        mask_root = (
            root / f"results_crackforest_mechanism_seed{seed}"
            if dataset == "crackforest"
            else root / f"results_deepcrack_seed{seed}"
        )
        reducer_root = root / f"results_strong_reducer_{dataset}_seed{seed}"
    else:
        mask_root = root / f"results_token_budget_masktopo_{dataset}_k{token_count}_seed{seed}"
        reducer_root = root / f"results_token_budget_{dataset}_k{token_count}_seed{seed}"
    return mask_root, reducer_root


def ph_root(root: Path, dataset: str, token_count: int, seed: int) -> Path:
    retry = "_retry1" if (
        dataset == "crackforest" and token_count == 16 and seed == 20260810
    ) else ""
    return root / f"results_ph_{dataset}_k{token_count}_seed{seed}{retry}"


def crackforest_sources(cache: Path) -> np.ndarray:
    candidates = sorted(cache.glob("crackforest_test_n320_seed22260810_crop64_*.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one CrackForest test cache, found {len(candidates)}."
        )
    with np.load(candidates[0]) as archive:
        return archive["source_id"].astype(str)


def score_summary(scores: list[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        "scores_by_seed": values.tolist(),
        "mean_balanced_accuracy": float(values.mean()),
        "sample_std_balanced_accuracy": float(values.std(ddof=1)),
    }


def scalar_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values_by_seed": array.tolist(),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
    }


def average_log_budget_auc(values: list[float]) -> float:
    x = np.log2(np.asarray(BUDGETS, dtype=np.float64))
    return float(np.trapezoid(np.asarray(values), x) / (x[-1] - x[0]))


def fast_cluster_bootstrap(
    truth: np.ndarray,
    sources: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
    chunk_size: int = 4096,
) -> dict[str, Any]:
    """Source-cluster bootstrap using sufficient confusion-count statistics."""
    unique_sources, inverse = np.unique(sources, return_inverse=True)
    source_count = unique_sources.size
    seed_count = first.shape[0]
    truth_bool = truth.astype(bool)
    positive_count = np.bincount(
        inverse, weights=truth_bool.astype(np.float64), minlength=source_count
    )
    negative_count = np.bincount(
        inverse, weights=(~truth_bool).astype(np.float64), minlength=source_count
    )
    first_positive_correct = np.empty((seed_count, source_count), dtype=np.float64)
    first_negative_correct = np.empty((seed_count, source_count), dtype=np.float64)
    second_positive_correct = np.empty((seed_count, source_count), dtype=np.float64)
    second_negative_correct = np.empty((seed_count, source_count), dtype=np.float64)
    for seed in range(seed_count):
        first_bool = first[seed].astype(bool)
        second_bool = second[seed].astype(bool)
        first_positive_correct[seed] = np.bincount(
            inverse,
            weights=(truth_bool & first_bool).astype(np.float64),
            minlength=source_count,
        )
        first_negative_correct[seed] = np.bincount(
            inverse,
            weights=((~truth_bool) & (~first_bool)).astype(np.float64),
            minlength=source_count,
        )
        second_positive_correct[seed] = np.bincount(
            inverse,
            weights=(truth_bool & second_bool).astype(np.float64),
            minlength=source_count,
        )
        second_negative_correct[seed] = np.bincount(
            inverse,
            weights=((~truth_bool) & (~second_bool)).astype(np.float64),
            minlength=source_count,
        )

    gains_by_seed = np.asarray(
        [
            balanced_accuracy(truth, first[seed])
            - balanced_accuracy(truth, second[seed])
            for seed in range(seed_count)
        ],
        dtype=np.float64,
    )
    bootstrap = np.empty(repetitions, dtype=np.float64)
    offset = 0
    while offset < repetitions:
        current = min(chunk_size, repetitions - offset)
        sampled = rng.choice(
            source_count, size=(current, source_count), replace=True
        )
        multiplicities = np.zeros((current, source_count), dtype=np.float64)
        rows = np.repeat(np.arange(current), source_count)
        np.add.at(multiplicities, (rows, sampled.reshape(-1)), 1.0)
        total_positive = multiplicities @ positive_count
        total_negative = multiplicities @ negative_count
        if np.any(total_positive == 0) or np.any(total_negative == 0):
            raise RuntimeError("A bootstrap resample lacks one outcome class.")
        gains = np.empty((seed_count, current), dtype=np.float64)
        for seed in range(seed_count):
            first_score = 0.5 * (
                (multiplicities @ first_positive_correct[seed]) / total_positive
                + (multiplicities @ first_negative_correct[seed]) / total_negative
            )
            second_score = 0.5 * (
                (multiplicities @ second_positive_correct[seed]) / total_positive
                + (multiplicities @ second_negative_correct[seed]) / total_negative
            )
            gains[seed] = first_score - second_score
        bootstrap[offset : offset + current] = gains.mean(axis=0)
        offset += current
    return {
        "paired_gains_by_seed": gains_by_seed.tolist(),
        "mean_gain": float(gains_by_seed.mean()),
        "sample_std_gain": float(gains_by_seed.std(ddof=1)),
        "source_cluster_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "source_image_count": int(source_count),
    }


def load_budget(
    root: Path,
    dataset: str,
    token_count: int,
    seeds: list[int],
    cache: Path,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    names = ("mask_topo", *REDUCERS, *PH_MODELS)
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in names}
    scores: dict[str, list[float]] = {name: [] for name in names}
    truth_reference: np.ndarray | None = None
    sources_reference: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    diagnostic_values = {
        "selected_h0_mean": [],
        "selected_h1_mean": [],
        "padding_rate": [],
    }
    preprocessing: dict[str, list[float]] = {
        str(batch): [] for batch in BATCH_SIZES
    }
    model_latency: dict[str, dict[str, list[float]]] = {
        name: {str(batch): [] for batch in BATCH_SIZES} for name in PH_MODELS
    }
    end_to_end_latency: dict[str, dict[str, list[float]]] = {
        name: {str(batch): [] for batch in BATCH_SIZES} for name in PH_MODELS
    }
    peak_memory: dict[str, dict[str, list[float]]] = {
        name: {str(batch): [] for batch in BATCH_SIZES} for name in PH_MODELS
    }

    for seed in seeds:
        mask_root, reducer_root = prior_roots(root, dataset, token_count, seed)
        current_ph_root = ph_root(root, dataset, token_count, seed)
        mask_result = json.loads(
            (mask_root / "result.json").read_text(encoding="utf-8")
        )
        reducer_result = json.loads(
            (reducer_root / "result.json").read_text(encoding="utf-8")
        )
        ph_result = json.loads(
            (current_ph_root / "result.json").read_text(encoding="utf-8")
        )
        if ph_result["status"] != "completed":
            raise RuntimeError(
                f"PH run is incomplete: {dataset}, K={token_count}, seed={seed}."
            )
        if int(ph_result["reduced_token_count"]) != token_count:
            raise RuntimeError(
                f"PH token-count mismatch: {dataset}, K={token_count}, seed={seed}."
            )
        if int(ph_result["optimization_seed"]) != seed:
            raise RuntimeError(
                f"PH seed mismatch: {dataset}, K={token_count}, seed={seed}."
            )
        if ph_result.get("massachusetts_test_reopened", True):
            raise RuntimeError("A PH run reports reopening Massachusetts test.")
        if not ph_result.get("test_was_previously_observed", False):
            raise RuntimeError("A PH Pareto run incorrectly claims a fresh test.")
        if ph_result.get("uses_ground_truth_mask_or_topology_at_inference", True):
            raise RuntimeError("A PH run reports ground-truth topology at inference.")

        with np.load(mask_root / "heldout_predictions.npz") as archive:
            truth = archive["truth"].astype(np.uint8)
            mask_key = MASK_KEYS[dataset] if token_count == 16 else "mask_topo"
            mask_prediction = archive[mask_key].astype(np.uint8)
            mask_sources = (
                archive["source_id"].astype(str)
                if "source_id" in archive.files
                else None
            )
        with np.load(reducer_root / "heldout_predictions.npz") as archive:
            reducer_truth = archive["truth"].astype(np.uint8)
            reducer_predictions = {
                name: archive[name].astype(np.uint8) for name in REDUCERS
            }
            reducer_sources = (
                archive["source_id"].astype(str)
                if "source_id" in archive.files
                else None
            )
        with np.load(current_ph_root / "heldout_predictions.npz") as archive:
            ph_truth = archive["truth"].astype(np.uint8)
            ph_predictions = {
                name: archive[name].astype(np.uint8) for name in PH_MODELS
            }
            ph_sources = (
                archive["source_id"].astype(str)
                if "source_id" in archive.files
                else None
            )
        if not np.array_equal(truth, reducer_truth) or not np.array_equal(
            truth, ph_truth
        ):
            raise RuntimeError(
                f"Held-out labels differ: {dataset}, K={token_count}, seed={seed}."
            )
        if dataset == "crackforest":
            sources = crackforest_sources(cache)
        else:
            if mask_sources is None or reducer_sources is None or ph_sources is None:
                raise RuntimeError("DeepCrack source IDs are missing.")
            if not np.array_equal(mask_sources, reducer_sources) or not np.array_equal(
                mask_sources, ph_sources
            ):
                raise RuntimeError("DeepCrack source IDs differ across methods.")
            sources = mask_sources
        if ph_sources is not None and not np.array_equal(sources, ph_sources):
            raise RuntimeError(
                f"PH source IDs differ: {dataset}, K={token_count}, seed={seed}."
            )
        if truth_reference is None:
            truth_reference = truth
            sources_reference = sources
        elif not np.array_equal(truth_reference, truth) or not np.array_equal(
            sources_reference, sources
        ):
            raise RuntimeError(
                f"Held-out sample ordering changed across seeds: {dataset}, K={token_count}."
            )

        if token_count == 16:
            mask_record = mask_result["model_results"][MASK_KEYS[dataset]]
        else:
            mask_record = mask_result["model_result"]
        records = {
            "mask_topo": mask_record,
            **reducer_result["model_results"],
            **ph_result["model_results"],
        }
        current_predictions = {
            "mask_topo": mask_prediction,
            **reducer_predictions,
            **ph_predictions,
        }
        for name, prediction in current_predictions.items():
            score = balanced_accuracy(truth, prediction)
            recorded = records[name]["test_metrics"]["balanced_accuracy"]
            if not np.isclose(score, recorded):
                raise RuntimeError(
                    f"Prediction/metric mismatch: {dataset}, K={token_count}, "
                    f"seed={seed}, model={name}."
                )
            predictions[name].append(prediction)
            scores[name].append(score)
            rows.append(
                {
                    "dataset": dataset,
                    "token_count": token_count,
                    "seed": seed,
                    "model": name,
                    "balanced_accuracy": score,
                    "parameters": records[name]["parameters"],
                }
            )

        diagnostic = ph_result["ph_diagnostics_at_k"]["test"]
        for key in diagnostic_values:
            diagnostic_values[key].append(float(diagnostic[key]))
        for batch in BATCH_SIZES:
            batch_key = str(batch)
            preprocessing[batch_key].append(
                float(
                    ph_result["ph_preprocessing_latency"][batch_key][
                        "latency_ms_per_sample"
                    ]
                )
            )
            for name in PH_MODELS:
                record = ph_result["model_results"][name]
                model_latency[name][batch_key].append(
                    float(record["model_latency"][batch_key]["latency_ms_per_sample"])
                )
                end_to_end_latency[name][batch_key].append(
                    float(
                        record["end_to_end_latency"][batch_key][
                            "conservative_total_ms_per_sample"
                        ]
                    )
                )
                peak_memory[name][batch_key].append(
                    float(record["model_latency"][batch_key]["peak_cuda_memory_mb"])
                )
        artifact = {
            "dataset": dataset,
            "token_count": token_count,
            "seed": seed,
            "ph_root": str(current_ph_root.resolve()),
            "result_sha256": file_sha256(current_ph_root / "result.json"),
            "predictions_sha256": file_sha256(
                current_ph_root / "heldout_predictions.npz"
            ),
        }
        for name in PH_MODELS:
            artifact[f"{name}_checkpoint_sha256"] = file_sha256(
                current_ph_root / f"{name}_seed{seed}.pt"
            )
        artifacts.append(artifact)

    if truth_reference is None or sources_reference is None:
        raise RuntimeError(f"No runs loaded: {dataset}, K={token_count}.")
    stacked = {name: np.stack(values) for name, values in predictions.items()}
    aggregates = {name: score_summary(values) for name, values in scores.items()}
    strongest_reducer = max(
        REDUCERS, key=lambda name: aggregates[name]["mean_balanced_accuracy"]
    )
    best_ph = max(
        PH_MODELS, key=lambda name: aggregates[name]["mean_balanced_accuracy"]
    )
    contrasts = {
        "mask_topo_minus_ph_guided": fast_cluster_bootstrap(
            truth_reference,
            sources_reference,
            stacked["mask_topo"],
            stacked["ph_guided"],
            repetitions,
            rng,
        ),
        "mask_topo_minus_ph_only": fast_cluster_bootstrap(
            truth_reference,
            sources_reference,
            stacked["mask_topo"],
            stacked["ph_only"],
            repetitions,
            rng,
        ),
        "mask_topo_minus_best_ph_envelope": fast_cluster_bootstrap(
            truth_reference,
            sources_reference,
            stacked["mask_topo"],
            stacked[best_ph],
            repetitions,
            rng,
        ),
        "best_ph_minus_strongest_general_reducer": fast_cluster_bootstrap(
            truth_reference,
            sources_reference,
            stacked[best_ph],
            stacked[strongest_reducer],
            repetitions,
            rng,
        ),
    }
    diagnostics = {
        key: scalar_summary(values) for key, values in diagnostic_values.items()
    }
    latency = {
        "ph_preprocessing_ms_per_sample": {
            key: scalar_summary(values) for key, values in preprocessing.items()
        },
        "models": {},
        "flops": {
            "status": "not_measured_in_training_runs",
            "reason": "Requires a same-session frozen-checkpoint profiler pass; not inferred from latency.",
        },
    }
    for name in PH_MODELS:
        latency["models"][name] = {
            "model_ms_per_sample": {
                key: scalar_summary(values)
                for key, values in model_latency[name].items()
            },
            "conservative_end_to_end_ms_per_sample": {
                key: scalar_summary(values)
                for key, values in end_to_end_latency[name].items()
            },
            "peak_cuda_memory_mb": {
                key: scalar_summary(values) for key, values in peak_memory[name].items()
            },
        }
    return {
        "token_count": token_count,
        "test_sample_count": int(truth_reference.size),
        "test_source_count": int(np.unique(sources_reference).size),
        "aggregates": aggregates,
        "strongest_general_reducer": strongest_reducer,
        "best_ph_by_three_seed_test_mean_descriptive": best_ph,
        "contrasts": contrasts,
        "ph_diagnostics": diagnostics,
        "efficiency": latency,
        "runs": rows,
        "artifact_hashes": artifacts,
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TB-B-260802-023 PH-inclusive fixed-K Pareto audit",
        "",
        "All accuracy values are endpoint-connectivity balanced accuracy (mean ± sample SD over three optimization seeds). They are not the separate P1-5 disease-classification result.",
        "All test sets were previously observed, so the comparisons are retrospective. Massachusetts Roads was not reopened.",
        "",
    ]
    for dataset in DATASETS:
        item = report["datasets"][dataset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| K | MaskTopo | PH-only | PH-guided (frozen primary) | Best PH (descriptive) | MaskTopo - best PH, 95% source-cluster CI |",
                "|---:|---:|---:|---:|---|---:|",
            ]
        )
        for token_count in BUDGETS:
            budget = item["budgets"][str(token_count)]
            agg = budget["aggregates"]
            best = budget["best_ph_by_three_seed_test_mean_descriptive"]
            contrast = budget["contrasts"]["mask_topo_minus_best_ph_envelope"]
            ci = contrast["source_cluster_bootstrap_95_ci"]
            lines.append(
                f"| {token_count} | "
                f"{100 * agg['mask_topo']['mean_balanced_accuracy']:.2f} ± {100 * agg['mask_topo']['sample_std_balanced_accuracy']:.2f}% | "
                f"{100 * agg['ph_only']['mean_balanced_accuracy']:.2f} ± {100 * agg['ph_only']['sample_std_balanced_accuracy']:.2f}% | "
                f"{100 * agg['ph_guided']['mean_balanced_accuracy']:.2f} ± {100 * agg['ph_guided']['sample_std_balanced_accuracy']:.2f}% | "
                f"{best} | {100 * contrast['mean_gain']:+.2f} pp "
                f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] |"
            )
        auc = item["log_budget_auc"]
        lines.extend(
            [
                "",
                f"Log-K AUC — MaskTopo: {100 * auc['mask_topo']:.2f}%; PH-guided: {100 * auc['ph_guided']:.2f}%; PH-only: {100 * auc['ph_only']:.2f}%; descriptive best-PH envelope: {100 * auc['best_ph_envelope']:.2f}%.",
                "",
            ]
        )
    lines.extend(
        [
            "## Audit boundaries",
            "",
            "- PH-guided is the frozen primary PH comparator. The per-K best-PH envelope is explicitly descriptive because it is selected using observed test means.",
            "- PH preprocessing is included in the recorded conservative end-to-end PH latency for batch 1/8/32.",
            "- Cross-method FLOPs and same-session batch latency are still pending a frozen-checkpoint profiler pass; no FLOPs are inferred from timing.",
            "- This stage does not use ground-truth masks or topology at inference.",
            "- TopoCL and WACV 2026 remain protocol-context methods; their task scores are not directly compared here.",
            "",
            f"Validation verdict: **{report['verdict']}**",
            "",
        ]
    )
    return "\n".join(lines)


def run_self_test() -> None:
    assert np.isclose(average_log_budget_auc([0.5, 0.5, 0.5, 0.5]), 0.5)
    summary = score_summary([0.5, 0.6, 0.7])
    assert np.isclose(summary["mean_balanced_accuracy"], 0.6)
    assert ph_root(Path("."), "crackforest", 16, 20260810).name.endswith("retry1")
    truth = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.uint8)
    sources = np.asarray(["a", "a", "b", "b", "c", "c"])
    first = np.asarray(
        [[0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, 1]], dtype=np.uint8
    )
    second = np.asarray(
        [[1, 1, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1]], dtype=np.uint8
    )
    result = fast_cluster_bootstrap(
        truth, sources, first, second, 1000, np.random.default_rng(7), chunk_size=1000
    )
    expected = np.asarray(
        [
            balanced_accuracy(truth, first[seed])
            - balanced_accuracy(truth, second[seed])
            for seed in range(2)
        ]
    )
    assert np.allclose(result["paired_gains_by_seed"], expected)
    assert result["source_image_count"] == 3
    source_indices = [np.flatnonzero(sources == source) for source in np.unique(sources)]
    reference_rng = np.random.default_rng(7)
    reference_bootstrap = np.empty(1000, dtype=np.float64)
    for repetition in range(1000):
        sampled = reference_rng.choice(3, size=3, replace=True)
        indices = np.concatenate([source_indices[index] for index in sampled])
        reference_bootstrap[repetition] = np.mean(
            [
                balanced_accuracy(truth[indices], first[seed, indices])
                - balanced_accuracy(truth[indices], second[seed, indices])
                for seed in range(2)
            ]
        )
    reference_ci = np.quantile(reference_bootstrap, [0.025, 0.975])
    assert np.allclose(result["source_cluster_bootstrap_95_ci"], reference_ci)
    print("SUMMARIZE_PH_PARETO_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = args.root.resolve()
    queue_status_path = root / "实验记录" / "TB-B-260802-023_queue_status.json"
    queue_status = json.loads(queue_status_path.read_text(encoding="utf-8"))
    if queue_status["status"] != "completed" or queue_status["jobs_completed"] != 18:
        raise RuntimeError("The fixed 18-job PH Pareto queue is not complete.")
    if queue_status.get("massachusetts_test_reopened", True):
        raise RuntimeError("Queue status reports reopening Massachusetts test.")

    rng = np.random.default_rng(args.statistics_seed)
    datasets: dict[str, Any] = {}
    source_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        budgets: dict[str, Any] = {}
        curves: dict[str, list[float]] = {
            "mask_topo": [],
            "ph_only": [],
            "ph_guided": [],
            "best_ph_envelope": [],
            "general_reducer_envelope": [],
        }
        for token_count in BUDGETS:
            loaded = load_budget(
                root,
                dataset,
                token_count,
                args.seeds,
                args.crackforest_cache.resolve(),
                args.bootstrap_repetitions,
                rng,
            )
            budgets[str(token_count)] = loaded
            source_rows.extend(loaded["runs"])
            agg = loaded["aggregates"]
            curves["mask_topo"].append(agg["mask_topo"]["mean_balanced_accuracy"])
            curves["ph_only"].append(agg["ph_only"]["mean_balanced_accuracy"])
            curves["ph_guided"].append(
                agg["ph_guided"]["mean_balanced_accuracy"]
            )
            curves["best_ph_envelope"].append(
                agg[loaded["best_ph_by_three_seed_test_mean_descriptive"]][
                    "mean_balanced_accuracy"
                ]
            )
            curves["general_reducer_envelope"].append(
                agg[loaded["strongest_general_reducer"]]["mean_balanced_accuracy"]
            )
        datasets[dataset] = {
            "budgets": budgets,
            "log_budget_auc": {
                name: average_log_budget_auc(values)
                for name, values in curves.items()
            },
        }

    report = {
        "experiment_id": "TB-B-260802-023",
        "status": "completed_verified_accuracy_and_ph_efficiency",
        "confirmatory_status": "retrospective_pareto",
        "optimization_seeds": args.seeds,
        "budgets": list(BUDGETS),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "queue_jobs_completed": queue_status["jobs_completed"],
        "datasets": datasets,
        "test_was_previously_observed": True,
        "massachusetts_test_reopened": False,
        "disease_classification_p1_5_used": False,
        "cross_method_same_session_efficiency_status": "pending",
        "protocol": str((root / "PH_PARETO_PROTOCOL.md").resolve()),
        "protocol_sha256": file_sha256(root / "PH_PARETO_PROTOCOL.md"),
        "queue_status_sha256": file_sha256(queue_status_path),
        "code_sha256": file_sha256(Path(__file__)),
        "verdict": "PH_PARETO_ACCURACY_VALIDATED_EFFICIENCY_PENDING",
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(report_markdown(report), encoding="utf-8")
    with (output / "source_data.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "token_count",
                "seed",
                "model",
                "balanced_accuracy",
                "parameters",
            ),
        )
        writer.writeheader()
        writer.writerows(source_rows)
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "status": report["status"],
                "queue_jobs_completed": report["queue_jobs_completed"],
                "verdict": report["verdict"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

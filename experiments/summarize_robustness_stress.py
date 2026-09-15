from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from edge_topocoarsen_connector import balanced_metrics
from robustness_stress import (
    MODELS,
    PH_MODELS,
    SEED_CHOICES,
    SENSITIVITY_CONDITIONS,
    STRESS_CONDITIONS,
)
from summarize_ph_pareto import fast_cluster_bootstrap


DATASETS = ("crackforest", "deepcrack")
FIXED_GENERAL = {
    "crackforest": "perceiver_resampler",
    "deepcrack": "tokenlearner",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TB-B-260802-024.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("results_robustness_summary")
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--statistics-seed", type=int, default=20260824)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_root(root: Path, dataset: str, seed: int) -> Path:
    return root / f"results_robustness_{dataset}_k16_seed{seed}_retry2"


def score_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "scores_by_seed": array.tolist(),
        "mean_balanced_accuracy": float(array.mean()),
        "sample_std_balanced_accuracy": float(array.std(ddof=1)),
    }


def scalar_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values_by_seed": array.tolist(),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
    }


def load_dataset(
    root: Path,
    dataset: str,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    truth_reference: np.ndarray | None = None
    sources_reference: np.ndarray | None = None
    stress_predictions: dict[str, dict[str, list[np.ndarray]]] = {
        condition: {model: [] for model in MODELS}
        for condition in STRESS_CONDITIONS
    }
    sensitivity_predictions: dict[str, list[np.ndarray]] = {
        condition: [] for condition in SENSITIVITY_CONDITIONS
    }
    stress_scores: dict[str, dict[str, list[float]]] = {
        condition: {model: [] for model in MODELS}
        for condition in STRESS_CONDITIONS
    }
    sensitivity_scores: dict[str, list[float]] = {
        condition: [] for condition in SENSITIVITY_CONDITIONS
    }
    mask_f1: dict[str, list[float]] = {
        condition: [] for condition in STRESS_CONDITIONS
    }
    cached_mask_f1: dict[str, list[float]] = {
        condition: [] for condition in STRESS_CONDITIONS
    }
    foreground: dict[str, list[float]] = {
        condition: [] for condition in STRESS_CONDITIONS
    }
    h0: dict[str, list[float]] = {
        condition: [] for condition in STRESS_CONDITIONS
    }
    h1: dict[str, list[float]] = {
        condition: [] for condition in STRESS_CONDITIONS
    }
    rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    code_hashes: set[str] = set()
    protocol_hashes: set[str] = set()
    precision_differences: list[float] = []

    for seed in SEED_CHOICES:
        current_root = run_root(root, dataset, seed)
        result_path = current_root / "result.json"
        prediction_path = current_root / "heldout_predictions.npz"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "completed":
            raise RuntimeError(f"Incomplete robustness result: {dataset}, {seed}.")
        if not result["all_clean_checkpoint_predictions_reproduced"]:
            raise RuntimeError(f"Clean checkpoint gate failed: {dataset}, {seed}.")
        if result.get("massachusetts_test_reopened", True):
            raise RuntimeError("A robustness run reports reopening Massachusetts.")
        if result.get("disease_classification_p1_5_used", True):
            raise RuntimeError("A robustness run reports using P1-5 classification.")
        if not result.get("masktopo_uses_reproduced_float32_probability", False):
            raise RuntimeError("Corrected MaskTopo precision path is absent.")
        if not result.get("reducers_and_ph_clean_use_saved_float16_probability", False):
            raise RuntimeError("Corrected reducer/PH precision path is absent.")
        code_hashes.add(result["code_sha256"])
        protocol_hashes.add(result["protocol_sha256"])
        precision_differences.append(
            float(result["mask_predictor"]["clean_probability_max_abs_difference"])
        )
        stderr_path = (
            root
            / "实验记录"
            / f"TB-B-260802-024_{dataset}_k16_seed{seed}_retry2.stderr.log"
        )
        stderr = stderr_path.read_text(encoding="utf-8")
        if "Traceback" in stderr or "RuntimeError" in stderr:
            raise RuntimeError(f"Unexpected retry2 stderr failure: {stderr_path}")
        with np.load(prediction_path) as archive:
            truth = archive["truth"].astype(np.uint8)
            sources = archive["source_id"].astype(str)
            current_stress = {
                condition: {
                    model: archive[f"stress__{condition}__{model}"].astype(np.uint8)
                    for model in MODELS
                }
                for condition in STRESS_CONDITIONS
            }
            current_sensitivity = {
                condition: archive[
                    f"sensitivity__{condition}__mask_topo"
                ].astype(np.uint8)
                for condition in SENSITIVITY_CONDITIONS
            }
        if truth_reference is None:
            truth_reference = truth
            sources_reference = sources
        elif not np.array_equal(truth_reference, truth) or not np.array_equal(
            sources_reference, sources
        ):
            raise RuntimeError(f"Held-out samples differ across seeds: {dataset}.")
        for condition in STRESS_CONDITIONS:
            condition_result = result["stress_conditions"][condition]
            mask_f1[condition].append(
                float(
                    condition_result["masktopo_native_precision_mask_metrics"][
                        "pixel_f1"
                    ]
                )
            )
            cached_mask_f1[condition].append(
                float(condition_result["mask_metrics"]["pixel_f1"])
            )
            foreground[condition].append(
                float(
                    condition_result["masktopo_native_precision_mask_metrics"][
                        "foreground_fraction"
                    ]
                )
            )
            h0[condition].append(
                float(
                    condition_result["ph_token_diagnostics"]["selected_h0_mean"]
                )
            )
            h1[condition].append(
                float(
                    condition_result["ph_token_diagnostics"]["selected_h1_mean"]
                )
            )
            for model in MODELS:
                prediction = current_stress[condition][model]
                score = float(balanced_metrics(truth, prediction)["balanced_accuracy"])
                recorded = float(
                    condition_result["model_results"][model]["balanced_accuracy"]
                )
                if not np.isclose(score, recorded):
                    raise RuntimeError(
                        f"Prediction/metric mismatch: {dataset}, {seed}, "
                        f"{condition}, {model}."
                    )
                stress_predictions[condition][model].append(prediction)
                stress_scores[condition][model].append(score)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "condition": condition,
                        "model": model,
                        "balanced_accuracy": score,
                        "masktopo_native_pixel_f1": mask_f1[condition][-1],
                        "cached_probability_pixel_f1": cached_mask_f1[condition][-1],
                    }
                )
        for condition in SENSITIVITY_CONDITIONS:
            prediction = current_sensitivity[condition]
            score = float(balanced_metrics(truth, prediction)["balanced_accuracy"])
            recorded = float(
                result["masktopo_sensitivity"][condition]["mask_topo_metrics"][
                    "balanced_accuracy"
                ]
            )
            if not np.isclose(score, recorded):
                raise RuntimeError(
                    f"Sensitivity metric mismatch: {dataset}, {seed}, {condition}."
                )
            sensitivity_predictions[condition].append(prediction)
            sensitivity_scores[condition].append(score)
            sensitivity_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "condition": condition,
                    "mask_threshold": SENSITIVITY_CONDITIONS[condition][0],
                    "closing_iterations": SENSITIVITY_CONDITIONS[condition][1],
                    "balanced_accuracy": score,
                }
            )
        artifacts.append(
            {
                "dataset": dataset,
                "seed": seed,
                "result_path": str(result_path.resolve()),
                "result_sha256": file_sha256(result_path),
                "predictions_path": str(prediction_path.resolve()),
                "predictions_sha256": file_sha256(prediction_path),
                "stderr_path": str(stderr_path.resolve()),
                "stderr_sha256": file_sha256(stderr_path),
            }
        )
    if truth_reference is None or sources_reference is None:
        raise RuntimeError(f"No robustness runs loaded: {dataset}.")
    if len(code_hashes) != 1 or len(protocol_hashes) != 1:
        raise RuntimeError(f"Mixed retry2 code/protocol hashes: {dataset}.")

    stacked = {
        condition: {
            model: np.stack(values)
            for model, values in condition_predictions.items()
        }
        for condition, condition_predictions in stress_predictions.items()
    }
    sensitivity_stacked = {
        condition: np.stack(values)
        for condition, values in sensitivity_predictions.items()
    }
    stress_summary: dict[str, Any] = {}
    fixed_general = FIXED_GENERAL[dataset]
    for condition in STRESS_CONDITIONS:
        aggregates = {
            model: score_summary(stress_scores[condition][model])
            for model in MODELS
        }
        best_ph = max(
            PH_MODELS,
            key=lambda model: aggregates[model]["mean_balanced_accuracy"],
        )
        comparisons = {
            "mask_topo_minus_fixed_general": fast_cluster_bootstrap(
                truth_reference,
                sources_reference,
                stacked[condition]["mask_topo"],
                stacked[condition][fixed_general],
                repetitions,
                rng,
            ),
            "mask_topo_minus_ph_guided": fast_cluster_bootstrap(
                truth_reference,
                sources_reference,
                stacked[condition]["mask_topo"],
                stacked[condition]["ph_guided"],
                repetitions,
                rng,
            ),
            "mask_topo_minus_best_ph_descriptive": fast_cluster_bootstrap(
                truth_reference,
                sources_reference,
                stacked[condition]["mask_topo"],
                stacked[condition][best_ph],
                repetitions,
                rng,
            ),
        }
        drops = {}
        for model in MODELS:
            drops[model] = fast_cluster_bootstrap(
                truth_reference,
                sources_reference,
                stacked["clean"][model],
                stacked[condition][model],
                repetitions,
                rng,
            )
        stress_summary[condition] = {
            "aggregates": aggregates,
            "fixed_general_reducer": fixed_general,
            "best_ph_by_three_seed_condition_mean_descriptive": best_ph,
            "comparisons": comparisons,
            "clean_minus_condition_drops": drops,
            "masktopo_native_pixel_f1": scalar_summary(mask_f1[condition]),
            "cached_probability_pixel_f1": scalar_summary(cached_mask_f1[condition]),
            "masktopo_native_foreground_fraction": scalar_summary(
                foreground[condition]
            ),
            "ph_selected_h0_mean": scalar_summary(h0[condition]),
            "ph_selected_h1_mean": scalar_summary(h1[condition]),
        }
    sensitivity_summary: dict[str, Any] = {}
    reference = sensitivity_stacked["threshold_0.90_c0"]
    for condition in SENSITIVITY_CONDITIONS:
        sensitivity_summary[condition] = {
            "mask_threshold": SENSITIVITY_CONDITIONS[condition][0],
            "closing_iterations": SENSITIVITY_CONDITIONS[condition][1],
            "aggregate": score_summary(sensitivity_scores[condition]),
            "reference_minus_condition": fast_cluster_bootstrap(
                truth_reference,
                sources_reference,
                reference,
                sensitivity_stacked[condition],
                repetitions,
                rng,
            ),
        }
    return {
        "stress": stress_summary,
        "masktopo_sensitivity": sensitivity_summary,
        "run_rows": rows,
        "sensitivity_rows": sensitivity_rows,
        "artifact_hashes": artifacts,
        "retry2_code_sha256": next(iter(code_hashes)),
        "retry2_protocol_sha256": next(iter(protocol_hashes)),
        "float32_vs_float16_probability_max_abs_difference_by_seed": precision_differences,
        "test_sample_count": int(truth_reference.size),
        "test_source_count": int(np.unique(sources_reference).size),
    }


def association_analysis(datasets: dict[str, Any]) -> dict[str, Any]:
    records = []
    for dataset in DATASETS:
        fixed_general = FIXED_GENERAL[dataset]
        rows = datasets[dataset]["run_rows"]
        lookup = {
            (int(row["seed"]), row["condition"], row["model"]): row
            for row in rows
        }
        for seed in SEED_CHOICES:
            for condition in STRESS_CONDITIONS:
                mask_row = lookup[(seed, condition, "mask_topo")]
                general_row = lookup[(seed, condition, fixed_general)]
                records.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "condition": condition,
                        "mask_f1": float(mask_row["masktopo_native_pixel_f1"]),
                        "downstream_gain": float(
                            mask_row["balanced_accuracy"]
                            - general_row["balanced_accuracy"]
                        ),
                    }
                )
    x = np.asarray([row["mask_f1"] for row in records], dtype=np.float64)
    y = np.asarray([row["downstream_gain"] for row in records], dtype=np.float64)
    centered_x = np.empty_like(x)
    centered_y = np.empty_like(y)
    for dataset in DATASETS:
        for seed in SEED_CHOICES:
            indices = np.asarray(
                [
                    index
                    for index, row in enumerate(records)
                    if row["dataset"] == dataset and row["seed"] == seed
                ]
            )
            centered_x[indices] = x[indices] - x[indices].mean()
            centered_y[indices] = y[indices] - y[indices].mean()
    raw_pearson = pearsonr(x, y)
    raw_spearman = spearmanr(x, y)
    centered_pearson = pearsonr(centered_x, centered_y)
    centered_spearman = spearmanr(centered_x, centered_y)
    return {
        "unit": "dataset_seed_condition aggregate; repeated and non-independent",
        "record_count": len(records),
        "fixed_general_reducers": FIXED_GENERAL,
        "raw_pooled": {
            "pearson_r": float(raw_pearson.statistic),
            "pearson_p_nominal": float(raw_pearson.pvalue),
            "spearman_rho": float(raw_spearman.statistic),
            "spearman_p_nominal": float(raw_spearman.pvalue),
        },
        "centered_within_dataset_and_optimization_seed": {
            "pearson_r": float(centered_pearson.statistic),
            "pearson_p_nominal": float(centered_pearson.pvalue),
            "spearman_rho": float(centered_spearman.statistic),
            "spearman_p_nominal": float(centered_spearman.pvalue),
        },
        "interpretation": (
            "Exploratory association only; nominal p-values do not account for "
            "repeated-condition dependence or test reuse."
        ),
        "records": records,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# TB-B-260802-024 K=16 robustness audit",
        "",
        "All values are endpoint-connectivity balanced accuracy, not P1-5 disease classification. All tests were previously observed; Massachusetts Roads was not reopened.",
        "",
    ]
    for dataset in DATASETS:
        item = report["datasets"][dataset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Condition | mask F1 | MaskTopo | PH-only | PH-guided | fixed general reducer | MaskTopo clean drop | MaskTopo − best PH (95% source CI) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition in STRESS_CONDITIONS:
            value = item["stress"][condition]
            agg = value["aggregates"]
            fixed = value["fixed_general_reducer"]
            drop = value["clean_minus_condition_drops"]["mask_topo"]
            contrast = value["comparisons"]["mask_topo_minus_best_ph_descriptive"]
            ci = contrast["source_cluster_bootstrap_95_ci"]
            lines.append(
                f"| {condition} | {value['masktopo_native_pixel_f1']['mean']:.3f} | "
                f"{100 * agg['mask_topo']['mean_balanced_accuracy']:.2f}% | "
                f"{100 * agg['ph_only']['mean_balanced_accuracy']:.2f}% | "
                f"{100 * agg['ph_guided']['mean_balanced_accuracy']:.2f}% | "
                f"{fixed}: {100 * agg[fixed]['mean_balanced_accuracy']:.2f}% | "
                f"{100 * drop['mean_gain']:+.2f} pp | "
                f"{100 * contrast['mean_gain']:+.2f} pp "
                f"[{100 * ci[0]:+.2f},{100 * ci[1]:+.2f}] |"
            )
        lines.extend(
            [
                "",
                "### MaskTopo threshold/morphology sensitivity",
                "",
                "| Condition | MaskTopo | reference − condition (95% source CI) |",
                "|---|---:|---:|",
            ]
        )
        for condition in SENSITIVITY_CONDITIONS:
            value = item["masktopo_sensitivity"][condition]
            contrast = value["reference_minus_condition"]
            ci = contrast["source_cluster_bootstrap_95_ci"]
            lines.append(
                f"| {condition} | "
                f"{100 * value['aggregate']['mean_balanced_accuracy']:.2f}% | "
                f"{100 * contrast['mean_gain']:+.2f} pp "
                f"[{100 * ci[0]:+.2f},{100 * ci[1]:+.2f}] |"
            )
        lines.append("")
    association = report["mask_f1_downstream_gain_association"][
        "centered_within_dataset_and_optimization_seed"
    ]
    lines.extend(
        [
            "## Mask F1 association",
            "",
            f"Within-dataset-and-seed centered Pearson r = {association['pearson_r']:.3f}; Spearman rho = {association['spearman_rho']:.3f}. These are exploratory repeated-measures diagnostics; nominal p-values are not confirmatory.",
            "",
            "## Boundaries",
            "",
            "- Break/false-link are predicted-mask corruptions; low-contrast/noise are image perturbations followed by frozen mask prediction.",
            "- The original historical MaskTopo path used in-memory float32 probabilities, while reducer/PH clean paths consumed a saved float16 cache. retry2 preserves and discloses both native paths.",
            "- The best-PH comparator is descriptive; PH-guided remains the frozen primary PH comparator.",
            "- No result here is an untouched external confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_self_test() -> None:
    truth = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    sources = np.asarray(["a", "a", "b", "b"])
    first = np.asarray([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.uint8)
    second = np.asarray([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.uint8)
    result = fast_cluster_bootstrap(
        truth, sources, first, second, 1000, np.random.default_rng(1)
    )
    assert np.isclose(result["mean_gain"], 0.25)
    print("SUMMARIZE_ROBUSTNESS_STRESS_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = args.root.resolve()
    queue_path = root / "实验记录/TB-B-260802-024_queue_retry2_status.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if queue["status"] != "completed" or queue["jobs_completed"] != 6:
        raise RuntimeError("The consistent retry2 robustness queue is incomplete.")
    if queue.get("massachusetts_test_reopened", True):
        raise RuntimeError("Queue reports reopening Massachusetts.")
    rng = np.random.default_rng(args.statistics_seed)
    datasets = {
        dataset: load_dataset(
            root, dataset, args.bootstrap_repetitions, rng
        )
        for dataset in DATASETS
    }
    association = association_analysis(datasets)
    report = {
        "experiment_id": "TB-B-260802-024",
        "status": "completed_verified_retry2",
        "confirmatory_status": "retrospective_robustness",
        "token_count": 16,
        "optimization_seeds": list(SEED_CHOICES),
        "stress_conditions": list(STRESS_CONDITIONS),
        "sensitivity_conditions": list(SENSITIVITY_CONDITIONS),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "datasets": datasets,
        "mask_f1_downstream_gain_association": association,
        "queue_status": str(queue_path.resolve()),
        "queue_status_sha256": file_sha256(queue_path),
        "protocol": str((root / "ROBUSTNESS_STRESS_PROTOCOL.md").resolve()),
        "protocol_sha256": file_sha256(root / "ROBUSTNESS_STRESS_PROTOCOL.md"),
        "code_sha256": file_sha256(Path(__file__)),
        "test_was_previously_observed": True,
        "massachusetts_test_reopened": False,
        "disease_classification_p1_5_used": False,
        "precision_path_limitation_disclosed": True,
        "verdict": "ROBUSTNESS_AUDIT_COMPLETED_WITH_REPORTED_FAILURES",
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {output}")
    output.mkdir(parents=True)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(markdown_report(report), encoding="utf-8")
    with (output / "source_data.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        rows = [row for dataset in DATASETS for row in datasets[dataset]["run_rows"]]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "sensitivity_data.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        rows = [
            row
            for dataset in DATASETS
            for row in datasets[dataset]["sensitivity_rows"]
        ]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "status": report["status"],
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

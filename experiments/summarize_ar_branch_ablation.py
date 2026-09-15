from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from summarize_crackforest_mechanism_ablation import balanced_accuracy
from summarize_strong_reducer_baselines import cluster_bootstrap


EXPERIMENT_ID = "TB-B-260810-041"
SEEDS = (20260810, 20260811, 20260812)
MODES = (
    "mask_topo_branch_none",
    "mask_topo_a_only",
    "mask_topo_r_only",
    "mask_topo_external",
)
MODE_LABELS = {
    "mask_topo_branch_none": "Assignment; A off; R off",
    "mask_topo_a_only": "Assignment + A only",
    "mask_topo_r_only": "Assignment + R only",
    "mask_topo_external": "Assignment + A + R (MaskTopo)",
}
CONTRASTS = (
    ("a_only_minus_none", "mask_topo_a_only", "mask_topo_branch_none"),
    ("r_only_minus_none", "mask_topo_r_only", "mask_topo_branch_none"),
    ("full_minus_a_only", "mask_topo_external", "mask_topo_a_only"),
    ("full_minus_r_only", "mask_topo_external", "mask_topo_r_only"),
    ("full_minus_none", "mask_topo_external", "mask_topo_branch_none"),
)
RUN_DIRS = {
    "fives": {
        20260810: "fives_seed20260810",
        20260811: "fives_seed20260811_retry1",
        20260812: "fives_seed20260812",
    },
    "deepcrack": {
        20260810: "deepcrack_seed20260810",
        20260811: "deepcrack_seed20260811",
        20260812: "deepcrack_seed20260812",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TB-B-260810-041.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("实验记录/TB-B-260810-041_outputs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("实验记录/TB-B-260810-041_summary"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--statistics-seed", type=int, default=20260810)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_summary(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(
        [balanced_accuracy(truth, row) for row in prediction], dtype=np.float64
    )
    return {
        "scores_by_seed": scores.tolist(),
        "mean_balanced_accuracy": float(scores.mean()),
        "sample_std_balanced_accuracy": float(scores.std(ddof=1)),
    }


def load_dataset(input_root: Path, dataset: str) -> dict[str, Any]:
    truth: np.ndarray | None = None
    sources: np.ndarray | None = None
    predictions: dict[str, list[np.ndarray]] = {mode: [] for mode in MODES}
    run_metadata: list[dict[str, Any]] = []
    parameter_counts: set[int] = set()

    for seed in SEEDS:
        run_dir = input_root / RUN_DIRS[dataset][seed]
        result_path = run_dir / "RESULT.json"
        prediction_path = run_dir / "test_predictions.npz"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            raise RuntimeError(f"Incomplete result: {run_dir}")
        if result.get("dataset") != dataset:
            raise RuntimeError(f"Dataset mismatch: {run_dir}")
        if int(result.get("optimization_seed")) != seed:
            raise RuntimeError(f"Optimization seed mismatch: {run_dir}")
        if not all(result["reproduction_checks"].values()):
            raise RuntimeError(f"Historical reproduction check failed: {run_dir}")
        if not all(result["probability_float16_cache_checks"].values()):
            raise RuntimeError(f"Float16 cache equality failed: {run_dir}")
        current_counts = {int(value) for value in result["parameter_counts"].values()}
        if len(current_counts) != 1:
            raise RuntimeError(f"Within-run parameter mismatch: {run_dir}")
        parameter_counts.update(current_counts)

        with np.load(prediction_path, allow_pickle=False) as archive:
            current_truth = archive["truth"].astype(np.uint8)
            current_sources = archive["source_id"].astype(str)
            current_predictions = {
                mode: archive[mode].astype(np.uint8) for mode in MODES
            }
        if truth is None:
            truth = current_truth
            sources = current_sources
        else:
            if not np.array_equal(truth, current_truth):
                raise RuntimeError(f"Truth drift across seeds: {dataset}")
            if not np.array_equal(sources, current_sources):
                raise RuntimeError(f"Source-ID drift across seeds: {dataset}")

        recorded = {
            "mask_topo_external": result["historical_full_model"]["test_metrics"][
                "balanced_accuracy"
            ],
            **{
                mode: result["new_model_results"][mode]["test_metrics"][
                    "balanced_accuracy"
                ]
                for mode in MODES
                if mode != "mask_topo_external"
            },
        }
        recomputed: dict[str, float] = {}
        for mode in MODES:
            value = balanced_accuracy(current_truth, current_predictions[mode])
            if not np.isclose(value, recorded[mode], rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"Prediction/JSON metric mismatch: {dataset}, {seed}, {mode}"
                )
            predictions[mode].append(current_predictions[mode])
            recomputed[mode] = value

        run_metadata.append(
            {
                "seed": seed,
                "run_directory": str(run_dir.resolve()),
                "result_sha256": sha256(result_path),
                "predictions_sha256": sha256(prediction_path),
                "balanced_accuracy": recomputed,
            }
        )

    assert truth is not None and sources is not None
    if len(parameter_counts) != 1:
        raise RuntimeError(f"Across-run parameter mismatch: {dataset}")
    stacked = {
        mode: np.stack(rows, axis=0) for mode, rows in predictions.items()
    }
    return {
        "truth": truth,
        "sources": sources,
        "predictions": stacked,
        "parameter_count": parameter_counts.pop(),
        "run_metadata": run_metadata,
    }


def summarize_dataset(
    loaded: dict[str, Any], repetitions: int, rng: np.random.Generator
) -> dict[str, Any]:
    truth = loaded["truth"]
    sources = loaded["sources"]
    predictions = loaded["predictions"]
    methods = {
        mode: score_summary(truth, predictions[mode]) for mode in MODES
    }
    contrasts = {
        name: cluster_bootstrap(
            truth,
            sources,
            predictions[first],
            predictions[second],
            repetitions,
            rng,
        )
        for name, first, second in CONTRASTS
    }
    complementarity = (
        contrasts["full_minus_a_only"]["source_cluster_bootstrap_95_ci"][0] > 0
        and contrasts["full_minus_r_only"]["source_cluster_bootstrap_95_ci"][0]
        > 0
    )
    return {
        "sample_count": int(truth.size),
        "source_image_count": int(np.unique(sources).size),
        "positive_count": int(truth.sum()),
        "negative_count": int((truth == 0).sum()),
        "parameter_count_all_conditions": int(loaded["parameter_count"]),
        "methods": methods,
        "contrasts": contrasts,
        "observed_complementarity_gate_pass": bool(complementarity),
        "run_metadata": loaded["run_metadata"],
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def signed_pp(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def write_csvs(output: Path, result: dict[str, Any]) -> None:
    with (output / "seed_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "seed", "method", "balanced_accuracy"])
        for dataset, summary in result["datasets"].items():
            for mode in MODES:
                for seed, score in zip(
                    SEEDS, summary["methods"][mode]["scores_by_seed"], strict=True
                ):
                    writer.writerow([dataset, seed, mode, f"{score:.12f}"])

    with (output / "contrast_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "contrast",
                "mean_gain",
                "sample_std_gain",
                "ci95_low",
                "ci95_high",
                "source_image_count",
            ]
        )
        for dataset, summary in result["datasets"].items():
            for name, contrast in summary["contrasts"].items():
                low, high = contrast["source_cluster_bootstrap_95_ci"]
                writer.writerow(
                    [
                        dataset,
                        name,
                        f"{contrast['mean_gain']:.12f}",
                        f"{contrast['sample_std_gain']:.12f}",
                        f"{low:.12f}",
                        f"{high:.12f}",
                        contrast["source_image_count"],
                    ]
                )


def write_report(output: Path, result: dict[str, Any]) -> None:
    lines = [
        "# TB-B-260810-041 A/R 分支独立消融统计报告",
        "",
        "## 审计结论",
        "",
        "- 六个有效数据集-种子任务全部完成；失败的 FIVES seed 20260811 首次尝试未进入统计，统计使用显式 `retry1` 完整结果。",
        "- 每个有效 run 的历史 A+R 预测、truth、source ID 与记录指标均通过精确复核；原生 float32 概率的 float16 cast 与既有缓存逐元素相等。",
        "- 四条件参数量完全相同。置信区间为 20,000 次配对 source-image cluster bootstrap，不把 crop 或 seed 当独立 source。",
        "- 五个预注册对比报告逐项（pointwise）95% CI，未作多重性校正；这些区间不应解释为同时置信区间。",
        "- 本实验是已观察 test 上的 retrospective mechanism diagnostic，不恢复 untouched-test 资格。",
        "",
        "## 三种子 BA",
        "",
        "| Dataset | Assignment only (A×R×) | A-only | R-only | A+R (MaskTopo) | Params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, summary in result["datasets"].items():
        values = []
        for mode in MODES:
            method = summary["methods"][mode]
            values.append(
                f"{percent(method['mean_balanced_accuracy'])}±"
                f"{percent(method['sample_std_balanced_accuracy'])}"
            )
        lines.append(
            f"| {dataset} | {values[0]} | {values[1]} | {values[2]} | "
            f"{values[3]} | {summary['parameter_count_all_conditions']:,} |"
        )

    lines.extend(
        [
            "",
            "## 配对 source-image 效应",
            "",
            "| Dataset | Contrast | Mean (pp) | 95% CI (pp) |",
            "|---|---|---:|---:|",
        ]
    )
    for dataset, summary in result["datasets"].items():
        for name, _, _ in CONTRASTS:
            contrast = summary["contrasts"][name]
            low, high = contrast["source_cluster_bootstrap_95_ci"]
            lines.append(
                f"| {dataset} | {name} | {signed_pp(contrast['mean_gain'])} | "
                f"[{signed_pp(low)}, {signed_pp(high)}] |"
            )

    lines.extend(["", "## 解释边界", ""])
    for dataset, summary in result["datasets"].items():
        gate = summary["observed_complementarity_gate_pass"]
        if gate:
            wording = "A 与 R 的联合模型相对两种单分支均有正且 CI 不跨零的观察性增益。"
        else:
            wording = (
                "预注册互补性门未通过：不能声称 A 与 R 都是必要的；应表述为单分支已能解释大部分或全部图消息增益。"
            )
        lines.append(f"- **{dataset}**：{wording}")
    lines.extend(
        [
            "- 结果只支持当前 dense-mask-derived endpoint-connectivity、当前轻量架构和预设 K；不外推到自然道路 APLS、高 K 或任意 mask 质量。",
            "- 不得与 FIVES P1–P5 疾病分类的 +1 pp 结果混用。",
            "",
            "## 统计设置",
            "",
            f"- Bootstrap repetitions: {result['statistics']['bootstrap_repetitions']:,}",
            f"- Statistics seed: {result['statistics']['statistics_seed']}",
            "- SD: three optimization seeds 的样本标准差（ddof=1）。",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_root = (root / args.input).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.statistics_seed)
    datasets = {
        dataset: summarize_dataset(
            load_dataset(input_root, dataset), args.bootstrap_repetitions, rng
        )
        for dataset in ("fives", "deepcrack")
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_analyzed_artifact_consistency_verified",
        "analysis_scope": "retrospective mechanism diagnostic",
        "seeds": list(SEEDS),
        "mode_labels": MODE_LABELS,
        "statistics": {
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "statistics_seed": args.statistics_seed,
            "cluster_unit": "source_image",
            "confidence_interval_scope": "pointwise_prespecified_contrasts",
            "multiplicity_adjustment": "none",
            "seed_summary_sd": "sample standard deviation, ddof=1",
        },
        "datasets": datasets,
        "integrity": {
            "valid_dataset_seed_run_count": 6,
            "new_checkpoint_count": 18,
            "failed_partial_run_excluded": "fives_seed20260811",
            "truth_equal_across_seeds": True,
            "source_id_equal_across_seeds": True,
            "historical_full_reproduction_checks_all_passed": True,
            "float16_cache_equality_checks_all_passed": True,
            "prediction_json_metric_checks_all_passed": True,
            "equal_parameter_count_all_conditions": True,
        },
    }
    write_csvs(output, result)
    write_report(output, result)
    (output / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "output": str(output),
        "complementarity": {
            dataset: value["observed_complementarity_gate_pass"]
            for dataset, value in datasets.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

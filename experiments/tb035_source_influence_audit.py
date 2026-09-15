from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


EXPERIMENT_ID = "TB-B-260803-037"
UPSTREAM_EXPERIMENT_ID = "TB-B-260803-035"
SEEDS = (20260810, 20260811, 20260812)
MODELS = (
    "mask_topo_external",
    "mask_conditioned_perceiver_strong",
    "mask_conditioned_slot_param_matched",
    "ph_only",
)
CONTRASTS = {
    "masktopo_minus_mask_perceiver_strong": "mask_conditioned_perceiver_strong",
    "masktopo_minus_mask_slot_param_matched": "mask_conditioned_slot_param_matched",
    "masktopo_minus_best_ph_dev_ph_only": "ph_only",
}
PREDICTIONS_SHA256 = "b82a27c764ed4528929e769b93a806f4cd7cfca0bd27ba242d3d4e22b29595b4"
TB035_RESULT_SHA256 = "ce4c024545f15721116da7e4c43e08f64c04e2cd7d1d206f3dc150b126f6eb24"
TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TB035 deterministic source heterogeneity/influence audit.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("实验记录/TB-B-260803-035_test_once/test_predictions_all_seeds.npz"),
    )
    parser.add_argument(
        "--tb035-result",
        type=Path,
        default=Path("实验记录/TB-B-260803-035_test_once/RESULT.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("实验记录/TB-B-260803-037_source异质性影响审计"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, expected: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Hash mismatch for {path}: {actual} != {expected}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}


def confusion(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.uint8)
    prediction = np.asarray(prediction, dtype=np.uint8)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("Truth/prediction shape mismatch.")
    if not set(np.unique(truth)).issubset({0, 1}) or not set(np.unique(prediction)).issubset({0, 1}):
        raise ValueError("Confusion inputs must be binary.")
    positive = int(np.sum(truth == 1))
    negative = int(np.sum(truth == 0))
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    sensitivity = tp / positive if positive else None
    specificity = tn / negative if negative else None
    balanced_accuracy = (
        0.5 * (sensitivity + specificity)
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "sample_count": int(truth.size),
        "positive_count": positive,
        "negative_count": negative,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": float(np.mean(truth == prediction)),
    }


def require_ba(truth: np.ndarray, prediction: np.ndarray) -> float:
    value = confusion(truth, prediction)["balanced_accuracy"]
    if value is None:
        raise ValueError("Balanced accuracy is undefined because one class is absent.")
    return float(value)


def descriptive(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "positive_count": int(np.sum(array > 0)),
        "zero_count": int(np.sum(np.isclose(array, 0.0, atol=TOLERANCE))),
        "negative_count": int(np.sum(array < 0)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_upstream(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    upstream: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for model in MODELS:
        values = [require_ba(truth, predictions[model][seed_index]) for seed_index in range(3)]
        expected = upstream["aggregates"][model]["balanced_accuracy_by_seed"]
        match = bool(np.allclose(values, expected, atol=TOLERANCE, rtol=0.0))
        checks[f"aggregate_{model}"] = {
            "recomputed": values,
            "upstream": expected,
            "exact_within_1e-12": match,
        }
        if not match:
            raise ValueError(f"Upstream aggregate mismatch for {model}.")
    for contrast, baseline in CONTRASTS.items():
        gains = [
            require_ba(truth, predictions["mask_topo_external"][seed_index])
            - require_ba(truth, predictions[baseline][seed_index])
            for seed_index in range(3)
        ]
        expected = upstream["primary_contrasts"][contrast]["paired_gains_by_seed"]
        match = bool(np.allclose(gains, expected, atol=TOLERANCE, rtol=0.0))
        checks[f"contrast_{contrast}"] = {
            "recomputed": gains,
            "upstream": expected,
            "exact_within_1e-12": match,
        }
        if not match:
            raise ValueError(f"Upstream contrast mismatch for {contrast}.")
    return checks


def fallacy_scan(
    source_count: int,
    source_positive_fraction: list[float],
    source_contrast_means: dict[str, list[float]],
    all_sources_retained: bool,
) -> list[dict[str, str]]:
    negative_counts = {
        contrast: int(np.sum(np.asarray(values) < 0))
        for contrast, values in source_contrast_means.items()
    }
    return [
        {
            "fallacy": "Simpson's paradox",
            "severity": "NOTE",
            "detail": (
                "Aggregate contrasts are positive. Per-source directions are reported rather than hidden; "
                f"negative-source counts are {negative_counts}. No claim of uniform source-level benefit is made."
            ),
        },
        {
            "fallacy": "Ecological fallacy",
            "severity": "NOTE",
            "detail": "Inference remains at source-image level; source summaries are not used to infer biological individual-level effects.",
        },
        {
            "fallacy": "Berkson's paradox",
            "severity": "CAUTION",
            "detail": "The audit concerns a filtered, project-defined Brassica source set and derived balanced endpoint task; population generalization remains limited.",
        },
        {
            "fallacy": "Collider bias",
            "severity": "NOTE",
            "detail": "No covariate adjustment, conditioning model, or post-test source exclusion is used in this audit.",
        },
        {
            "fallacy": "Base-rate neglect",
            "severity": "CAUTION",
            "detail": (
                "The pooled task is 50% positive, while source positive fractions vary from "
                f"{min(source_positive_fraction):.3f} to {max(source_positive_fraction):.3f}; confusion counts, sensitivity, specificity and BA are all reported."
            ),
        },
        {
            "fallacy": "Regression to the mean",
            "severity": "NOTE",
            "detail": "No pre/post comparison or source selection based on extreme model performance is performed.",
        },
        {
            "fallacy": "Survivorship bias",
            "severity": "NOTE" if all_sources_retained else "RED_FLAG",
            "detail": f"All {source_count}/15 frozen test sources are retained in the audit." if all_sources_retained else "A frozen source is missing.",
        },
        {
            "fallacy": "Look-elsewhere effect",
            "severity": "NOTE",
            "detail": "Exactly the three TB035 frozen primary contrasts are audited; no new baseline or outcome is selected from the test results.",
        },
        {
            "fallacy": "Garden of forking paths",
            "severity": "CAUTION",
            "detail": "The exact influence summaries are post-test descriptive analyses. They do not create, replace or relax a confirmatory gate.",
        },
        {
            "fallacy": "Correlation != causation",
            "severity": "CAUTION",
            "detail": "Controlled model comparisons support the frozen task-level representation contrast, not universal causal superiority across domains, K or downstream tasks.",
        },
        {
            "fallacy": "Reverse causality",
            "severity": "NOTE",
            "detail": "No temporal or directional observational exposure-outcome claim is made.",
        },
    ]


def self_test() -> None:
    truth = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    prediction = np.asarray([0, 1, 1, 0], dtype=np.uint8)
    values = confusion(truth, prediction)
    assert values["tp"] == 1 and values["tn"] == 1
    assert values["fp"] == 1 and values["fn"] == 1
    assert values["balanced_accuracy"] == 0.5
    summary = descriptive([-0.1, 0.0, 0.2])
    assert summary["negative_count"] == 1 and summary["positive_count"] == 1
    print(json.dumps({"stage": "self-test", "status": "PASS"}, indent=2))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    predictions_path = args.predictions.resolve()
    result_path = args.tb035_result.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite TB037 output: {output}")
    checked_file(predictions_path, PREDICTIONS_SHA256)
    checked_file(result_path, TB035_RESULT_SHA256)
    upstream = json.loads(result_path.read_text(encoding="utf-8"))
    if upstream["experiment_id"] != UPSTREAM_EXPERIMENT_ID or upstream["status"] != "ONE_SHOT_TEST_COMPLETED":
        raise ValueError("Invalid upstream TB035 result.")

    with np.load(predictions_path, allow_pickle=False) as archive:
        required = {"truth", "source_id", "seeds"}
        required |= {f"prediction_{model}" for model in MODELS}
        if not required.issubset(archive.files):
            raise ValueError(f"Missing prediction keys: {sorted(required - set(archive.files))}")
        truth = archive["truth"].astype(np.uint8)
        sources = archive["source_id"].astype(str)
        seeds = tuple(int(value) for value in archive["seeds"])
        predictions = {
            model: archive[f"prediction_{model}"].astype(np.uint8)
            for model in MODELS
        }
    if truth.shape != (1200,) or sources.shape != (1200,) or seeds != SEEDS:
        raise ValueError("Frozen truth/source/seed shape or identity changed.")
    if any(value.shape != (3, 1200) for value in predictions.values()):
        raise ValueError("Frozen prediction shape changed.")
    unique_sources = np.unique(sources)
    if unique_sources.size != 15:
        raise ValueError(f"Expected 15 sources, got {unique_sources.size}.")

    upstream_reproduction = validate_upstream(truth, predictions, upstream)
    full_gains = {
        contrast: [
            require_ba(truth, predictions["mask_topo_external"][seed_index])
            - require_ba(truth, predictions[baseline][seed_index])
            for seed_index in range(3)
        ]
        for contrast, baseline in CONTRASTS.items()
    }
    full_mean_gain = {contrast: mean(values) for contrast, values in full_gains.items()}

    confusion_rows: list[dict[str, Any]] = []
    source_summary_rows: list[dict[str, Any]] = []
    source_positive_fraction: list[float] = []
    source_contrast_means = {contrast: [] for contrast in CONTRASTS}
    per_source_payload: dict[str, Any] = {}
    for source in unique_sources:
        index = np.flatnonzero(sources == source)
        source_truth = truth[index]
        positive = int(np.sum(source_truth == 1))
        negative = int(np.sum(source_truth == 0))
        if positive == 0 or negative == 0:
            raise ValueError(f"Source {source} lacks one class; per-source BA undefined.")
        source_positive_fraction.append(float(source_truth.mean()))
        model_seed_ba: dict[str, list[float]] = {model: [] for model in MODELS}
        for seed_index, seed in enumerate(SEEDS):
            for model in MODELS:
                values = confusion(source_truth, predictions[model][seed_index, index])
                model_seed_ba[model].append(float(values["balanced_accuracy"]))
                confusion_rows.append(
                    {
                        "source_id": source,
                        "seed": seed,
                        "model": model,
                        **values,
                    }
                )
        contrast_source = {
            contrast: [
                model_seed_ba["mask_topo_external"][seed_index]
                - model_seed_ba[baseline][seed_index]
                for seed_index in range(3)
            ]
            for contrast, baseline in CONTRASTS.items()
        }
        for contrast, values in contrast_source.items():
            source_contrast_means[contrast].append(mean(values))
        joint_adversity = mean(mean(values) for values in contrast_source.values())
        row: dict[str, Any] = {
            "source_id": source,
            "sample_count": int(index.size),
            "positive_count": positive,
            "negative_count": negative,
            "positive_fraction": float(source_truth.mean()),
        }
        for model in MODELS:
            row[f"{model}_ba_mean"] = mean(model_seed_ba[model])
            row[f"{model}_ba_sample_sd"] = stdev(model_seed_ba[model])
        for contrast, values in contrast_source.items():
            row[f"{contrast}_mean_gain"] = mean(values)
            row[f"{contrast}_min_seed_gain"] = min(values)
            row[f"{contrast}_positive_seed_count"] = sum(value > 0 for value in values)
        row["joint_mean_gain_across_three_contrasts"] = joint_adversity
        source_summary_rows.append(row)
        per_source_payload[source] = {
            "sample_count": int(index.size),
            "positive_count": positive,
            "negative_count": negative,
            "model_ba_by_seed": model_seed_ba,
            "contrast_gain_by_seed": contrast_source,
            "joint_mean_gain_across_three_contrasts": joint_adversity,
        }

    leave_one_out_rows: list[dict[str, Any]] = []
    loo_values = {contrast: [] for contrast in CONTRASTS}
    for omitted in unique_sources:
        keep = sources != omitted
        for contrast, baseline in CONTRASTS.items():
            gains = [
                require_ba(truth[keep], predictions["mask_topo_external"][seed_index, keep])
                - require_ba(truth[keep], predictions[baseline][seed_index, keep])
                for seed_index in range(3)
            ]
            value = mean(gains)
            shift = value - full_mean_gain[contrast]
            loo_values[contrast].append(value)
            leave_one_out_rows.append(
                {
                    "omitted_source_id": omitted,
                    "contrast": contrast,
                    "remaining_source_count": 14,
                    "remaining_sample_count": int(np.sum(keep)),
                    "seed20260810_gain": gains[0],
                    "seed20260811_gain": gains[1],
                    "seed20260812_gain": gains[2],
                    "mean_gain": value,
                    "full_15_source_mean_gain": full_mean_gain[contrast],
                    "shift_from_full": shift,
                    "absolute_shift": abs(shift),
                    "relative_absolute_shift": abs(shift) / abs(full_mean_gain[contrast]),
                }
            )

    contrast_summary: dict[str, Any] = {}
    for contrast in CONTRASTS:
        source_values = source_contrast_means[contrast]
        rows = [row for row in leave_one_out_rows if row["contrast"] == contrast]
        worst_source_index = int(np.argmin(source_values))
        max_influence = max(rows, key=lambda row: row["absolute_shift"])
        contrast_summary[contrast] = {
            "full_15_source_mean_gain": full_mean_gain[contrast],
            "per_source_unweighted_gain_distribution": descriptive(source_values),
            "worst_source_by_mean_within_source_gain": {
                "source_id": str(unique_sources[worst_source_index]),
                "mean_gain": source_values[worst_source_index],
            },
            "leave_one_source_out_mean_gain_range": [min(loo_values[contrast]), max(loo_values[contrast])],
            "leave_one_source_out_all_positive": bool(min(loo_values[contrast]) > 0),
            "most_influential_omission_by_absolute_shift": max_influence,
        }

    joint_worst_row = min(
        source_summary_rows,
        key=lambda row: row["joint_mean_gain_across_three_contrasts"],
    )
    all_sources_retained = set(unique_sources) == set(upstream["test_ids"])
    fallacies = fallacy_scan(
        int(unique_sources.size), source_positive_fraction, source_contrast_means, all_sources_retained
    )
    if len(fallacies) != 11:
        raise AssertionError("Fallacy scan is incomplete.")
    any_red_flag = any(item["severity"] == "RED_FLAG" for item in fallacies)
    single_source_dominance = {
        "diagnostic_definition": (
            "No single source is necessary for the positive point-estimate conclusion if every one-source-omitted "
            "mean gain remains above zero. This is descriptive and is not a new confirmatory gate."
        ),
        "all_three_contrasts_retain_positive_gain_under_every_omission": all(
            item["leave_one_source_out_all_positive"] for item in contrast_summary.values()
        ),
        "max_relative_absolute_shift_across_all_omissions": max(
            row["relative_absolute_shift"] for row in leave_one_out_rows
        ),
        "max_absolute_shift_across_all_omissions": max(
            row["absolute_shift"] for row in leave_one_out_rows
        ),
    }

    output.mkdir(parents=True, exist_ok=False)
    confusion_path = output / "source_model_seed_confusion.csv"
    source_summary_path = output / "source_summary.csv"
    loo_path = output / "leave_one_source_out.csv"
    write_csv(confusion_path, confusion_rows)
    write_csv(source_summary_path, source_summary_rows)
    write_csv(loo_path, leave_one_out_rows)

    result = {
        "experiment_id": EXPERIMENT_ID,
        "upstream_experiment_id": UPSTREAM_EXPERIMENT_ID,
        "status": "COMPLETED_DETERMINISTIC_DESCRIPTIVE_AUDIT",
        "completed_at": datetime.now().astimezone().isoformat(),
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run+validate",
            "verification_status": "VERIFIED",
            "version_label": "tb037_v1",
        },
        "scope": {
            "neural_network_loaded": False,
            "checkpoint_loaded": False,
            "model_forward_performed": False,
            "test_predictions_reused_read_only": True,
            "new_confirmatory_gate_created": False,
            "model_or_claim_modified": False,
        },
        "inputs": {
            "predictions": checked_file(predictions_path, PREDICTIONS_SHA256),
            "tb035_result": checked_file(result_path, TB035_RESULT_SHA256),
        },
        "sample_count": int(truth.size),
        "source_count": int(unique_sources.size),
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "contrasts": list(CONTRASTS),
        "upstream_exact_reproduction": upstream_reproduction,
        "per_source": per_source_payload,
        "contrast_summary": contrast_summary,
        "joint_worst_source": joint_worst_row,
        "single_source_dominance": single_source_dominance,
        "fallacy_scan": {
            "coverage": "11/11",
            "overall_confidence": "RED_FLAG" if any_red_flag else "CAUTION",
            "items": fallacies,
        },
        "interpretation_boundary": (
            "Post-test descriptive influence audit only. It does not change TB035 model selection, thresholds, "
            "confidence intervals, claim gates or the one-shot test result."
        ),
        "endpoint_disease_firewall": (
            "All effects here are Brassica endpoint-connectivity effects and are separate from P1-P5 disease classification +1.00 pp."
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    result_path_out = output / "RESULT.json"
    result_path_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run + validate",
        f"- Origin Date: {datetime.now().astimezone().date().isoformat()}",
        "- Verification Status: VERIFIED",
        "- Version Label: tb037_v1",
        "",
        "# TB-B-260803-037：TB035 source 异质性与影响审计",
        "",
        "本审计只读取冻结的 TB035 test predictions；没有加载 checkpoint、神经网络或执行前向，也没有修改模型、阈值、CI 或主张门。",
        "",
        "## Source 构成",
        "",
        "| source | n | positive | negative | positive fraction | MaskTopo BA | Perceiver BA | Slot BA | PH-only BA | 三对比联合均值差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_summary_rows:
        report_lines.append(
            f"| {row['source_id']} | {row['sample_count']} | {row['positive_count']} | {row['negative_count']} | "
            f"{row['positive_fraction']:.3f} | {100*row['mask_topo_external_ba_mean']:.2f}% | "
            f"{100*row['mask_conditioned_perceiver_strong_ba_mean']:.2f}% | "
            f"{100*row['mask_conditioned_slot_param_matched_ba_mean']:.2f}% | "
            f"{100*row['ph_only_ba_mean']:.2f}% | "
            f"{100*row['joint_mean_gain_across_three_contrasts']:+.2f} pp |"
        )
    report_lines.extend(["", "## Leave-one-source-out 影响", ""])
    for contrast, item in contrast_summary.items():
        lower, upper = item["leave_one_source_out_mean_gain_range"]
        worst = item["worst_source_by_mean_within_source_gain"]
        influence = item["most_influential_omission_by_absolute_shift"]
        report_lines.append(
            f"- `{contrast}`：完整15-source差值 {100*item['full_15_source_mean_gain']:+.2f} pp；"
            f"LOO范围 [{100*lower:+.2f},{100*upper:+.2f}] pp；所有遗漏后仍为正："
            f"`{str(item['leave_one_source_out_all_positive']).lower()}`。"
        )
        report_lines.append(
            f"  - 最不利 source（按source内三种子均值差）：`{worst['source_id']}`，{100*worst['mean_gain']:+.2f} pp。"
        )
        report_lines.append(
            f"  - 最大绝对影响来自遗漏 `{influence['omitted_source_id']}`："
            f"相对完整效应偏移 {100*influence['shift_from_full']:+.2f} pp。"
        )
    report_lines.extend(
        [
            "",
            "## 单一 source 主导诊断",
            "",
            f"- 每一个 source 分别剔除后，三项平均差值均保持为正：`{str(single_source_dominance['all_three_contrasts_retain_positive_gain_under_every_omission']).lower()}`。",
            f"- 全部45个遗漏×对比组合中的最大绝对偏移：{100*single_source_dominance['max_absolute_shift_across_all_omissions']:.2f} pp。",
            f"- 最大相对偏移占对应完整差值的 {100*single_source_dominance['max_relative_absolute_shift_across_all_omissions']:.2f}%。",
            f"- 联合最不利 source：`{joint_worst_row['source_id']}`，三项source内平均差值的联合均值为 {100*joint_worst_row['joint_mean_gain_across_three_contrasts']:+.2f} pp。",
            "- 该诊断只说明没有任何单个 source 是保持正点估计所必需的；它不证明15个 source 足以代表更广泛总体。",
            "",
            "## Source 间异质性",
            "",
        ]
    )
    for contrast, item in contrast_summary.items():
        values = item["per_source_unweighted_gain_distribution"]
        report_lines.append(
            f"- `{contrast}`：source内均值差范围 [{100*values['min']:+.2f},{100*values['max']:+.2f}] pp，"
            f"中位数 {100*values['median']:+.2f} pp，IQR [{100*values['q25']:+.2f},{100*values['q75']:+.2f}] pp，"
            f"正/零/负 source 数={values['positive_count']}/{values['zero_count']}/{values['negative_count']}。"
        )
    report_lines.extend(
        [
            "",
            "## 统计谬误扫描",
            "",
            "- 覆盖：11/11。",
            f"- 总体置信：`{'RED_FLAG' if any_red_flag else 'CAUTION'}`；CAUTION 来自项目定义过滤样本、source base-rate 不均衡和审计为 post-test 描述性分析，并非计算错误。",
            "- 未创建新的显著性门、多重比较筛选或因果外推。完整11项见 `RESULT.json`。",
            "",
            "## 结论边界",
            "",
            "本审计不改变 TB035 已冻结的 source-cluster bootstrap CI 或两个主张门。结果只能补充 source 异质性与单源影响描述；不得改写成所有 source、所有域或所有 K 均获益。",
            "",
            "以上均为 Brassica endpoint-connectivity 效应，严禁与 P1–P5 疾病分类 `+1.00 pp` 混合。",
            "",
        ]
    )
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": result["status"],
        "outputs": {
            "result": checked_file(result_path_out),
            "report": checked_file(report_path),
            "source_model_seed_confusion": checked_file(confusion_path),
            "source_summary": checked_file(source_summary_path),
            "leave_one_source_out": checked_file(loo_path),
        },
        "neural_network_loaded": False,
        "model_forward_performed": False,
        "upstream_exact_reproduction_pass": all(
            item["exact_within_1e-12"] for item in upstream_reproduction.values()
        ),
        "all_loo_contrasts_positive": single_source_dominance[
            "all_three_contrasts_retain_positive_gain_under_every_omission"
        ],
        "joint_worst_source": joint_worst_row["source_id"],
    }
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "manifest": checked_file(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

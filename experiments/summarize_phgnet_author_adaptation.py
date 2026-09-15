from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from summarize_crackforest_mechanism_ablation import balanced_accuracy
from summarize_strong_reducer_baselines import cluster_bootstrap


EXPERIMENT_ID = "TB-B-260803-031"
SEEDS = (20260810, 20260811, 20260812)
MODELS = (
    "aligned_masktopo",
    "ph_only",
    "ph_guided",
    "phgnet_author_adaptation",
)
MODEL_LABEL = "PHG-Net author-code-based adaptation"
EXPECTED_AUTHOR_COMMIT = "6daa5f7dba556e9882611eb4e2e1c89a67f0d2c5"
EXPECTED_AUTHOR_MODULE_SHA256 = (
    "88535E17B99B6DE7BAD8CA4202698CBDBA1DD056CA1F0ADB67BD005729312B20"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize DeepCrack K16 PHG-Net author-code adaptation."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_phgnet_author_deepcrack_k16_summary"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--statistics-seed", type=int, default=20260803)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_summary(values: list[float]) -> dict[str, Any]:
    scores = np.asarray(values, dtype=np.float64)
    return {
        "balanced_accuracy_by_seed": scores.tolist(),
        "mean_balanced_accuracy": float(scores.mean()),
        "sample_std_balanced_accuracy": float(scores.std(ddof=1)),
    }


def load_and_validate(
    root: Path,
    seeds: list[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    truth_reference: np.ndarray | None = None
    source_reference: np.ndarray | None = None
    prediction_lists: dict[str, list[np.ndarray]] = {name: [] for name in MODELS}
    score_lists: dict[str, list[float]] = {name: [] for name in MODELS}
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []

    for seed in seeds:
        adaptation_root = root / f"results_phgnet_author_deepcrack_k16_seed{seed}"
        adaptation_result_path = adaptation_root / "result.json"
        adaptation_prediction_path = adaptation_root / "heldout_predictions.npz"
        adaptation_result = json.loads(
            adaptation_result_path.read_text(encoding="utf-8")
        )
        if adaptation_result["experiment_id"] != EXPERIMENT_ID:
            raise RuntimeError(f"Experiment ID mismatch at seed {seed}.")
        if adaptation_result["status"] != "completed":
            raise RuntimeError(f"Adaptation run not complete at seed {seed}.")
        if adaptation_result["method_label"] != MODEL_LABEL:
            raise RuntimeError(f"Method label changed at seed {seed}.")
        if adaptation_result["original_protocol_reproduction"]:
            raise RuntimeError("Adaptation incorrectly claims original-protocol reproduction.")
        if adaptation_result["confirmatory_status"] != "retrospective_collision":
            raise RuntimeError(f"Confirmatory status changed at seed {seed}.")
        if int(adaptation_result["reduced_token_count"]) != 16:
            raise RuntimeError(f"K changed at seed {seed}.")
        if not adaptation_result["exact_k_interface_verified"]:
            raise RuntimeError(f"Exact-K assertion missing at seed {seed}.")
        if adaptation_result["p1_to_p5_result_used"]:
            raise RuntimeError("P1--P5 evidence leaked into the endpoint experiment.")
        if adaptation_result["massachusetts_test_reopened"]:
            raise RuntimeError("Massachusetts test was unexpectedly reopened.")
        author = adaptation_result["author_source"]
        if author["commit"] != EXPECTED_AUTHOR_COMMIT:
            raise RuntimeError(f"Author commit mismatch at seed {seed}.")
        if author["module_sha256"] != EXPECTED_AUTHOR_MODULE_SHA256:
            raise RuntimeError(f"Author module hash mismatch at seed {seed}.")
        if any(
            dtype != "float16"
            for dtype in adaptation_result["mask_probabilities"][
                "storage_dtype_by_split"
            ].values()
        ):
            raise RuntimeError(f"Non-float16 mask storage at seed {seed}.")

        alignment_path = (
            root
            / "results_deepcrack_float16_alignment"
            / f"predictions_seed{seed}.npz"
        )
        with np.load(alignment_path, allow_pickle=False) as archive:
            truth = archive["truth"].astype(np.uint8)
            sources = archive["source_id"].astype(str)
            aligned = {
                "aligned_masktopo": archive["aligned_masktopo"].astype(np.uint8),
                "ph_only": archive["ph_only"].astype(np.uint8),
                "ph_guided": archive["ph_guided"].astype(np.uint8),
            }
        with np.load(adaptation_prediction_path, allow_pickle=False) as archive:
            if not np.array_equal(truth, archive["truth"].astype(np.uint8)):
                raise RuntimeError(f"Adaptation truth mismatch at seed {seed}.")
            if not np.array_equal(sources, archive["source_id"].astype(str)):
                raise RuntimeError(f"Adaptation source IDs mismatch at seed {seed}.")
            adaptation_prediction = archive[
                "phgnet_author_adaptation"
            ].astype(np.uint8)
        ph_path = root / f"results_ph_deepcrack_k16_seed{seed}/heldout_predictions.npz"
        with np.load(ph_path, allow_pickle=False) as archive:
            if not np.array_equal(truth, archive["truth"].astype(np.uint8)):
                raise RuntimeError(f"Frozen PH truth mismatch at seed {seed}.")
            if "source_id" in archive.files and not np.array_equal(
                sources, archive["source_id"].astype(str)
            ):
                raise RuntimeError(f"Frozen PH source IDs mismatch at seed {seed}.")
            for name in ("ph_only", "ph_guided"):
                if not np.array_equal(aligned[name], archive[name].astype(np.uint8)):
                    raise RuntimeError(
                        f"Float16 alignment and frozen PH predictions differ: {seed}, {name}."
                    )

        if truth_reference is None:
            truth_reference = truth
            source_reference = sources
        elif not np.array_equal(truth_reference, truth) or not np.array_equal(
            source_reference, sources
        ):
            raise RuntimeError("DeepCrack held-out samples changed across seeds.")

        current = {**aligned, "phgnet_author_adaptation": adaptation_prediction}
        recorded_adaptation = adaptation_result["model_results"][
            "phgnet_author_adaptation"
        ]["test_metrics"]["balanced_accuracy"]
        for name, prediction in current.items():
            score = balanced_accuracy(truth, prediction)
            if name == "phgnet_author_adaptation" and not np.isclose(
                score, recorded_adaptation
            ):
                raise RuntimeError(f"Stored adaptation metric mismatch at seed {seed}.")
            prediction_lists[name].append(prediction)
            score_lists[name].append(score)
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "balanced_accuracy": score,
                    "source_image_count": int(np.unique(sources).size),
                }
            )

        checkpoint_path = (
            adaptation_root / f"phgnet_author_adaptation_seed{seed}.pt"
        )
        artifacts.append(
            {
                "seed": seed,
                "result_path": str(adaptation_result_path.resolve()),
                "result_sha256": file_sha256(adaptation_result_path),
                "prediction_path": str(adaptation_prediction_path.resolve()),
                "prediction_sha256": file_sha256(adaptation_prediction_path),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "alignment_prediction_path": str(alignment_path.resolve()),
                "alignment_prediction_sha256": file_sha256(alignment_path),
                "ph_prediction_path": str(ph_path.resolve()),
                "ph_prediction_sha256": file_sha256(ph_path),
            }
        )

    if truth_reference is None or source_reference is None:
        raise RuntimeError("No runs were loaded.")
    predictions = {
        name: np.stack(values) for name, values in prediction_lists.items()
    }
    aggregates = {
        name: score_summary(values) for name, values in score_lists.items()
    }
    return truth_reference, source_reference, predictions, aggregates, rows, artifacts


def make_report(result: dict[str, Any]) -> str:
    aggregates = result["aggregates"]
    contrasts = result["contrasts"]
    primary = contrasts["aligned_masktopo_minus_phgnet_author_adaptation"]
    primary_ci = primary["source_cluster_bootstrap_95_ci"]
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run + statistical aggregation",
        "- Origin Date: 2026-08-03",
        "- Verification Status: ANALYZED",
        "- Version Label: TB-B-260803-031_v1",
        "",
        "# TB-B-260803-031: DeepCrack K16 PHG-Net 作者代码适配",
        "",
        "> 方法名称固定为 **PHG-Net author-code-based adaptation（基于作者代码的适配）**。这不是 PHG-Net 原协议复现，也不与原论文准确率直接比较。",
        "",
        "## 执行口径",
        "",
        "- DeepCrack 端点连通任务，K=16，训练/dev/test 为 2,400/600/1,200 crops；统计单位为 222 个 source images。",
        "- 三个优化种子：20260810、20260811、20260812；22 epochs、batch 32、AdamW、相同 dev 选模和统一分类头。",
        "- 三个 mask archive 的原始存储 dtype 均为 float16；复用相同 H0/H1 持久图缓存。",
        "- 作者仓库 commit `6daa5f7dba556e9882611eb4e2e1c89a67f0d2c5`；复用未修改的 `PointNetEncoder`，通过适配层精确产生 16 个 token。",
        "- DeepCrack test 已在历史实验中观察，本实验属于 retrospective collision，不是 untouched confirmation。",
        "- P1--P5 疾病分类结果未进入本实验或统计，不能与端点连通增益混合。",
        "",
        "## 三种子结果",
        "",
        "| Seed | float16-aligned MaskTopo | PH-only | PH-guided | PHG-Net author-code-based adaptation |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, seed in enumerate(result["seeds"]):
        lines.append(
            f"| {seed} | "
            f"{100 * aggregates['aligned_masktopo']['balanced_accuracy_by_seed'][index]:.2f}% | "
            f"{100 * aggregates['ph_only']['balanced_accuracy_by_seed'][index]:.2f}% | "
            f"{100 * aggregates['ph_guided']['balanced_accuracy_by_seed'][index]:.2f}% | "
            f"{100 * aggregates['phgnet_author_adaptation']['balanced_accuracy_by_seed'][index]:.2f}% |"
        )
    lines.extend(
        [
            "",
            "三种子均值 ± 样本标准差：",
            "",
            f"- float16-aligned MaskTopo: {100 * aggregates['aligned_masktopo']['mean_balanced_accuracy']:.2f}±{100 * aggregates['aligned_masktopo']['sample_std_balanced_accuracy']:.2f}%.",
            f"- PH-only: {100 * aggregates['ph_only']['mean_balanced_accuracy']:.2f}±{100 * aggregates['ph_only']['sample_std_balanced_accuracy']:.2f}%.",
            f"- PH-guided: {100 * aggregates['ph_guided']['mean_balanced_accuracy']:.2f}±{100 * aggregates['ph_guided']['sample_std_balanced_accuracy']:.2f}%.",
            f"- PHG-Net author-code-based adaptation: {100 * aggregates['phgnet_author_adaptation']['mean_balanced_accuracy']:.2f}±{100 * aggregates['phgnet_author_adaptation']['sample_std_balanced_accuracy']:.2f}%.",
            "",
            "## Source-image cluster-bootstrap 对比",
            "",
            "| 对比 | 均值差 | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    labels = {
        "aligned_masktopo_minus_phgnet_author_adaptation": "aligned MaskTopo − PHG-Net adaptation",
        "phgnet_author_adaptation_minus_ph_only": "PHG-Net adaptation − PH-only",
        "phgnet_author_adaptation_minus_ph_guided": "PHG-Net adaptation − PH-guided",
        "aligned_masktopo_minus_ph_only_sanity": "aligned MaskTopo − PH-only（sanity）",
    }
    for key, label in labels.items():
        contrast = contrasts[key]
        ci = contrast["source_cluster_bootstrap_95_ci"]
        lines.append(
            f"| {label} | {100 * contrast['mean_gain']:+.2f} pp | "
            f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] pp |"
        )
    lines.extend(
        [
            "",
            "## 裁决",
            "",
            f"- 冻结主对比：aligned MaskTopo − PHG-Net adaptation = {100 * primary['mean_gain']:+.2f} pp，95% CI [{100 * primary_ci[0]:+.2f}, {100 * primary_ci[1]:+.2f}] pp。",
            f"- 预注册判定：`{result['verdict']}`。",
            f"- 主张影响：{result['claim_impact']} ",
            "- 该结果只支撑 DeepCrack K=16、当前端点连通协议下的比较；不能外推为普遍优于 PHG-Net、医学分类或所有 K。",
            "",
            "## 完整性检查",
            "",
            "- 3/3 运行完成；truth、source IDs、PH-only/PH-guided 冻结预测逐元素一致。",
            "- 每个适配运行记录 exact-K=True、float16 train/dev/test 输入以及 P1--P5 未使用。",
            "- 20,000 次配对 source-image cluster bootstrap；统计种子 20260803。",
            "- 作者仓库未提供显式 LICENSE 文件；仅记录本地研究适配，不推断再分发许可。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_self_test() -> None:
    truth = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    sources = np.asarray(["a", "a", "b", "b"])
    first = np.asarray([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    second = np.asarray([[0, 1, 0, 1], [1, 1, 0, 1]], dtype=np.uint8)
    result = cluster_bootstrap(
        truth, sources, first, second, 100, np.random.default_rng(1)
    )
    if not np.isfinite(result["mean_gain"]):
        raise AssertionError("Bootstrap self-test failed.")
    print("PHGNET_AUTHOR_SUMMARY_SELF_TEST_PASS", flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    if seeds != list(SEEDS):
        raise ValueError(f"Frozen seeds changed: {seeds}")
    if args.bootstrap_repetitions != 20000 or args.statistics_seed != 20260803:
        raise ValueError("Frozen bootstrap settings changed.")
    truth, sources, predictions, aggregates, rows, artifacts = load_and_validate(
        root, seeds
    )
    rng = np.random.default_rng(args.statistics_seed)
    contrast_pairs = {
        "aligned_masktopo_minus_phgnet_author_adaptation": (
            "aligned_masktopo",
            "phgnet_author_adaptation",
        ),
        "phgnet_author_adaptation_minus_ph_only": (
            "phgnet_author_adaptation",
            "ph_only",
        ),
        "phgnet_author_adaptation_minus_ph_guided": (
            "phgnet_author_adaptation",
            "ph_guided",
        ),
        "aligned_masktopo_minus_ph_only_sanity": (
            "aligned_masktopo",
            "ph_only",
        ),
    }
    contrasts = {
        name: cluster_bootstrap(
            truth,
            sources,
            predictions[first],
            predictions[second],
            args.bootstrap_repetitions,
            rng,
        )
        for name, (first, second) in contrast_pairs.items()
    }
    primary = contrasts["aligned_masktopo_minus_phgnet_author_adaptation"]
    lower, upper = primary["source_cluster_bootstrap_95_ci"]
    if lower > 0:
        verdict = "MASKTOPO_OUTPERFORMS_PHGNET_AUTHOR_CODE_BASED_ADAPTATION"
        claim_impact = (
            "DeepCrack K16 上对更强作者 PD 编码器适配的优势仍成立；主张范围保持为当前协议，不能升级为普遍优于 PHG-Net。"
        )
    elif upper < 0:
        verdict = "PHGNET_AUTHOR_CODE_BASED_ADAPTATION_OUTPERFORMS_MASKTOPO"
        claim_impact = (
            "撤回 DeepCrack 上优于强 PH 编码器的主张，并将贡献改写为与任务/效率相关的有限结果。"
        )
    else:
        verdict = "MASKTOPO_VS_PHGNET_AUTHOR_CODE_BASED_ADAPTATION_INCONCLUSIVE"
        claim_impact = (
            "DeepCrack 上相对强 PH 编码器的优势不确定；不得声称击败 PHG-Net 类方法。"
        )

    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_analyzed",
        "method_label": MODEL_LABEL,
        "original_protocol_reproduction": False,
        "dataset": "deepcrack",
        "task": "endpoint_connectivity",
        "token_count": 16,
        "seeds": seeds,
        "test_crop_count": int(truth.size),
        "test_source_image_count": int(np.unique(sources).size),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "statistics_seed": args.statistics_seed,
        "aggregates": aggregates,
        "contrasts": contrasts,
        "verdict": verdict,
        "claim_impact": claim_impact,
        "artifact_consistency": {
            "truth_equal_across_all_runs": True,
            "source_ids_equal_across_all_runs": True,
            "frozen_ph_predictions_exactly_reproduced": True,
            "adaptation_prediction_metrics_reproduced": True,
            "float16_storage_verified": True,
            "exact_k_verified": True,
            "p1_to_p5_evidence_used": False,
        },
        "artifacts": artifacts,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "source_data.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "model", "balanced_accuracy", "source_image_count"),
        )
        writer.writeheader()
        writer.writerows(rows)
    report = make_report(result)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "artifact_hashes.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report, flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

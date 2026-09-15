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


EXPERIMENT_ID = "TB-B-260803-032"
SEEDS = (20260810, 20260811, 20260812)
FROZEN_DATA_SEED = 20260810
FROZEN_SPLIT_SEED = 20260730
STATISTICS_SEED = 20260803
BOOTSTRAP_REPETITIONS = 20_000
MASKTOPO_PARAMETER_ANCHOR = 149_123
MODELS = (
    "aligned_masktopo",
    "mask_conditioned_perceiver_strong",
    "mask_conditioned_slot_param_matched",
    "existing_mask_guided_queries",
)
NEW_MODELS = (
    "mask_conditioned_perceiver_strong",
    "mask_conditioned_slot_param_matched",
)
EXPECTED_PARAMETERS = {
    "mask_conditioned_perceiver_strong": 259_043,
    "mask_conditioned_slot_param_matched": 150_163,
    "existing_mask_guided_queries": 259_011,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize TB032 DeepCrack K16 mask-aware fairness baselines."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_mask_aware_equal_supervision_deepcrack_k16_summary"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS
    )
    parser.add_argument("--statistics-seed", type=int, default=STATISTICS_SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_summary(scores: list[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        "balanced_accuracy_by_seed": values.tolist(),
        "mean_balanced_accuracy": float(values.mean()),
        "sample_std_balanced_accuracy": float(values.std(ddof=1)),
    }


def validate_new_result(result: dict[str, Any], seed: int) -> None:
    if result["experiment_id"] != EXPERIMENT_ID or result["status"] != "completed":
        raise RuntimeError(f"Incomplete or foreign TB032 result at seed {seed}.")
    expected_scalars = {
        "dataset": "deepcrack",
        "task": "endpoint_connectivity",
        "optimization_seed": seed,
        "data_seed": FROZEN_DATA_SEED,
        "split_seed": FROZEN_SPLIT_SEED,
        "reduced_token_count": 16,
        "common_token_dimension": 64,
        "confirmatory_status": "retrospective_collision",
    }
    for key, expected in expected_scalars.items():
        if result[key] != expected:
            raise RuntimeError(
                f"Frozen field changed at seed {seed}: {key}={result[key]!r}."
            )
    if result["sample_counts"] != {"train": 2400, "dev": 600, "test": 1200}:
        raise RuntimeError(f"Sample counts changed at seed {seed}.")
    if result["models"] != list(NEW_MODELS):
        raise RuntimeError(f"Frozen model list changed at seed {seed}.")
    if not result["selection_uses_development_only"]:
        raise RuntimeError(f"Development-only selection missing at seed {seed}.")
    if not result["test_was_previously_observed"]:
        raise RuntimeError(f"Retrospective status missing at seed {seed}.")
    if result["massachusetts_test_reopened"] or result["p1_to_p5_result_used"]:
        raise RuntimeError(f"Out-of-scope evidence leaked at seed {seed}.")
    if any(
        dtype != "float16"
        for dtype in result["mask_probabilities"]["storage_dtype_by_split"].values()
    ):
        raise RuntimeError(f"Mask archive was not stored float16 at seed {seed}.")
    if result["mask_probabilities"]["loaded_compute_dtype"] != "float32":
        raise RuntimeError(f"Unexpected compute dtype at seed {seed}.")
    equal = result["equal_supervision"]
    if not equal["same_saved_predicted_mask_as_masktopo"]:
        raise RuntimeError(f"Equal mask supervision not asserted at seed {seed}.")
    if equal["ground_truth_mask_used_at_inference"]:
        raise RuntimeError(f"Ground-truth mask used at seed {seed}.")
    firewall = result["non_topological_firewall"]
    prohibited_flags = (
        "connected_components",
        "component_assignment",
        "adjacency_graph_construction",
        "reachability_graph_construction",
        "persistent_homology",
        "skeletonization",
    )
    if any(firewall[key] for key in prohibited_flags):
        raise RuntimeError(f"Topology firewall failed at seed {seed}.")
    if firewall["classifier_graph_inputs"] != "identity_only_common_head":
        raise RuntimeError(f"Classifier graph inputs changed at seed {seed}.")
    for name in NEW_MODELS:
        model = result["model_results"][name]
        if model["parameters"] != EXPECTED_PARAMETERS[name]:
            raise RuntimeError(f"Parameter count changed: {seed}, {name}.")
        if model["exact_output_tokens"] != 16:
            raise RuntimeError(f"Exact-K failed: {seed}, {name}.")
        if any(
            model[key]
            for key in (
                "uses_ground_truth_mask_or_topology",
                "uses_connected_components",
                "uses_graph_construction",
                "uses_persistent_homology",
            )
        ):
            raise RuntimeError(f"Model topology flag failed: {seed}, {name}.")


def load_and_validate(
    root: Path, seeds: list[int]
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    truth_reference: np.ndarray | None = None
    source_reference: np.ndarray | None = None
    prediction_lists: dict[str, list[np.ndarray]] = {name: [] for name in MODELS}
    score_lists: dict[str, list[float]] = {name: [] for name in MODELS}
    seed_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    code_hashes: set[str] = set()
    protocol_hashes: set[str] = set()

    for seed in seeds:
        run_root = root / f"results_mask_aware_equal_supervision_deepcrack_k16_seed{seed}"
        result_path = run_root / "result.json"
        prediction_path = run_root / "heldout_predictions.npz"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_new_result(result, seed)
        code_hashes.add(result["code_sha256"])
        protocol_hashes.add(result["protocol_sha256"])

        alignment_path = (
            root / "results_deepcrack_float16_alignment" / f"predictions_seed{seed}.npz"
        )
        existing_root = root / f"results_strong_reducer_deepcrack_seed{seed}"
        existing_result_path = existing_root / "result.json"
        existing_prediction_path = existing_root / "heldout_predictions.npz"
        existing_result = json.loads(existing_result_path.read_text(encoding="utf-8"))
        if existing_result["optimization_seed"] != seed:
            raise RuntimeError(f"Existing baseline optimization seed mismatch: {seed}.")
        if existing_result["data_seed"] != FROZEN_DATA_SEED:
            raise RuntimeError(f"Existing baseline data seed mismatch: {seed}.")
        if existing_result["mask_probabilities"]["sha256"] != result[
            "mask_probabilities"
        ]["sha256"]:
            raise RuntimeError(f"Mask archive hash mismatch: {seed}.")

        with np.load(alignment_path, allow_pickle=False) as aligned_archive:
            truth = aligned_archive["truth"].astype(np.uint8)
            sources = aligned_archive["source_id"].astype(str)
            aligned_masktopo = aligned_archive["aligned_masktopo"].astype(np.uint8)
        with np.load(prediction_path, allow_pickle=False) as new_archive:
            if not np.array_equal(truth, new_archive["truth"].astype(np.uint8)):
                raise RuntimeError(f"New-run truth mismatch: {seed}.")
            if not np.array_equal(sources, new_archive["source_id"].astype(str)):
                raise RuntimeError(f"New-run source mismatch: {seed}.")
            new_predictions = {
                name: new_archive[name].astype(np.uint8) for name in NEW_MODELS
            }
        with np.load(existing_prediction_path, allow_pickle=False) as old_archive:
            if not np.array_equal(truth, old_archive["truth"].astype(np.uint8)):
                raise RuntimeError(f"Existing-run truth mismatch: {seed}.")
            if not np.array_equal(sources, old_archive["source_id"].astype(str)):
                raise RuntimeError(f"Existing-run source mismatch: {seed}.")
            existing_prediction = old_archive["mask_guided_queries"].astype(np.uint8)

        if truth_reference is None:
            truth_reference = truth
            source_reference = sources
        elif not np.array_equal(truth_reference, truth) or not np.array_equal(
            source_reference, sources
        ):
            raise RuntimeError("Frozen DeepCrack held-out samples changed across seeds.")

        current = {
            "aligned_masktopo": aligned_masktopo,
            **new_predictions,
            "existing_mask_guided_queries": existing_prediction,
        }
        for name, prediction in current.items():
            score = balanced_accuracy(truth, prediction)
            if name in NEW_MODELS:
                recorded = result["model_results"][name]["test_metrics"][
                    "balanced_accuracy"
                ]
            elif name == "existing_mask_guided_queries":
                recorded = existing_result["model_results"]["mask_guided_queries"][
                    "test_metrics"
                ]["balanced_accuracy"]
                if existing_result["model_results"]["mask_guided_queries"][
                    "parameters"
                ] != EXPECTED_PARAMETERS[name]:
                    raise RuntimeError(f"Existing parameter count changed: {seed}.")
            else:
                recorded = score
            if not np.isclose(score, recorded):
                raise RuntimeError(f"Stored metric mismatch: {seed}, {name}.")
            prediction_lists[name].append(prediction)
            score_lists[name].append(score)
            seed_rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "balanced_accuracy": score,
                    "source_image_count": int(np.unique(sources).size),
                }
            )

        for source in np.unique(sources):
            indices = np.flatnonzero(sources == source)
            row: dict[str, Any] = {
                "seed": seed,
                "source_id": source,
                "crop_count": int(indices.size),
                "positive_count": int(truth[indices].sum()),
                "negative_count": int(indices.size - truth[indices].sum()),
            }
            for name, prediction in current.items():
                row[f"{name}_correct_count"] = int(
                    np.count_nonzero(prediction[indices] == truth[indices])
                )
            source_rows.append(row)

        artifacts.append(
            {
                "seed": seed,
                "result_path": str(result_path.resolve()),
                "result_sha256": file_sha256(result_path),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": file_sha256(prediction_path),
                "strong_checkpoint_sha256": file_sha256(
                    run_root / f"mask_conditioned_perceiver_strong_seed{seed}.pt"
                ),
                "matched_checkpoint_sha256": file_sha256(
                    run_root / f"mask_conditioned_slot_param_matched_seed{seed}.pt"
                ),
                "alignment_prediction_sha256": file_sha256(alignment_path),
                "existing_prediction_sha256": file_sha256(existing_prediction_path),
                "mask_archive_sha256": result["mask_probabilities"]["sha256"],
            }
        )

    if truth_reference is None or source_reference is None:
        raise RuntimeError("No TB032 runs were loaded.")
    if len(code_hashes) != 1 or len(protocol_hashes) != 1:
        raise RuntimeError("Final TB032 runs do not share one code/protocol hash.")
    predictions = {
        name: np.stack(values) for name, values in prediction_lists.items()
    }
    aggregates = {
        name: score_summary(values) for name, values in score_lists.items()
    }

    repeat_root = (
        root
        / "results_mask_aware_equal_supervision_deepcrack_k16_seed20260810_VALID_pre_data_seed_hardgate"
    )
    repeat_checks: dict[str, bool] = {}
    with np.load(
        repeat_root / "heldout_predictions.npz", allow_pickle=False
    ) as original, np.load(
        root
        / "results_mask_aware_equal_supervision_deepcrack_k16_seed20260810"
        / "heldout_predictions.npz",
        allow_pickle=False,
    ) as final:
        for key in final.files:
            repeat_checks[key] = bool(np.array_equal(final[key], original[key]))
    if not all(repeat_checks.values()):
        raise RuntimeError("Seed 20260810 reproducibility repeat changed predictions.")

    shared_hashes = {
        "code_sha256": next(iter(code_hashes)),
        "protocol_sha256": next(iter(protocol_hashes)),
        "seed20260810_pre_hardgate_repeat_equal": repeat_checks,
    }
    return (
        truth_reference,
        source_reference,
        predictions,
        aggregates,
        seed_rows,
        source_rows,
        artifacts,
        shared_hashes,
    )


def make_report(result: dict[str, Any]) -> str:
    aggregates = result["aggregates"]
    contrasts = result["contrasts"]
    labels = {
        "aligned_masktopo": "float16-aligned MaskTopo",
        "mask_conditioned_perceiver_strong": "Mask-conditioned Perceiver (strong)",
        "mask_conditioned_slot_param_matched": "Mask-conditioned Slot (parameter-matched)",
        "existing_mask_guided_queries": "Existing mask-guided queries (frozen sanity)",
    }
    lines = [
        "---",
        f"experiment_id: {EXPERIMENT_ID}",
        "date: 2026-08-03",
        "status: completed_analyzed",
        "dataset: DeepCrack",
        "task: endpoint_connectivity",
        "token_count: 16",
        "primary_statistical_unit: source_image",
        "confirmatory_status: retrospective_collision",
        f"verdict: {result['verdict']}",
        "---",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run + statistical aggregation",
        "- Origin Date: 2026-08-03",
        "- Verification Status: ANALYZED",
        "- Version Label: TB-B-260803-032_v1",
        "",
        "# TB-B-260803-032：DeepCrack K=16 等监督 mask-aware 非拓扑强基线",
        "",
        "## 结论",
        "",
        f"冻结裁决：`{result['verdict']}`。{result['claim_impact']}",
        "",
        "这是已经观察过 test 的 DeepCrack retrospective collision，不是 untouched 外部确认。结果只涉及端点连通任务；P1–P5 疾病分类的 `+1.00 pp` 未进入输入、训练、统计或主张，严禁与这里的增益混合。",
        "",
        "## 公平性与实现",
        "",
        "- 三种子均固定 DeepCrack train/dev/test 为 2,400/600/1,200 crops、222 source images，`data_seed=20260810`、`split_seed=20260730`。",
        "- 两个新基线均输入相同 PatchStem 图像特征、固定坐标和逐种子相同 SHA256 的 float16 保存 mask 概率，精确输出 K=16。",
        "- 强基线使用两层 mask-biased Perceiver cross-attention；参数匹配版使用两轮 mask-biased slot pooling。",
        "- 两者均禁止连通分量、component assignment、邻接/可达图构造、PH、骨架化和拓扑消息传递；统一分类头只接收 identity adjacency/reachability。",
        "- 22 epochs、batch 32、AdamW、lr 1e-3、weight decay 1e-4、相同增强与 dev-only checkpoint selection。",
        "",
        "## 容量核对",
        "",
        "| 模型 | 参数量 | 相对 MaskTopo |",
        "|---|---:|---:|",
        f"| MaskTopo anchor | {MASKTOPO_PARAMETER_ANCHOR:,} | 1.000× |",
        f"| Mask-conditioned Perceiver (strong) | {EXPECTED_PARAMETERS['mask_conditioned_perceiver_strong']:,} | {EXPECTED_PARAMETERS['mask_conditioned_perceiver_strong']/MASKTOPO_PARAMETER_ANCHOR:.3f}× |",
        f"| Mask-conditioned Slot (parameter-matched) | {EXPECTED_PARAMETERS['mask_conditioned_slot_param_matched']:,} | {EXPECTED_PARAMETERS['mask_conditioned_slot_param_matched']/MASKTOPO_PARAMETER_ANCHOR:.3f}× |",
        f"| Existing mask-guided queries | {EXPECTED_PARAMETERS['existing_mask_guided_queries']:,} | {EXPECTED_PARAMETERS['existing_mask_guided_queries']/MASKTOPO_PARAMETER_ANCHOR:.3f}× |",
        "",
        "参数匹配版相对 MaskTopo 的误差为 +0.697%，通过预设 ±1% 门；主强基线和既有 mask-guided 对照都比 MaskTopo 参数更多，因此结果不能解释为 MaskTopo 仅靠更大容量获益。",
        "",
        "## 三种子 balanced accuracy",
        "",
        "| Seed | aligned MaskTopo | Mask-Perceiver strong | Mask-Slot matched | Existing mask-guided |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, seed in enumerate(result["seeds"]):
        lines.append(
            f"| {seed} | "
            f"{100*aggregates['aligned_masktopo']['balanced_accuracy_by_seed'][index]:.2f}% | "
            f"{100*aggregates['mask_conditioned_perceiver_strong']['balanced_accuracy_by_seed'][index]:.2f}% | "
            f"{100*aggregates['mask_conditioned_slot_param_matched']['balanced_accuracy_by_seed'][index]:.2f}% | "
            f"{100*aggregates['existing_mask_guided_queries']['balanced_accuracy_by_seed'][index]:.2f}% |"
        )
    lines.extend(["", "三种子均值 ± 样本标准差：", ""])
    for name in MODELS:
        values = aggregates[name]
        lines.append(
            f"- {labels[name]}：{100*values['mean_balanced_accuracy']:.2f}±"
            f"{100*values['sample_std_balanced_accuracy']:.2f}%。"
        )
    lines.extend(
        [
            "",
            "## Source-image cluster-bootstrap",
            "",
            "20,000 次配对 source-image cluster bootstrap，统计种子 20260803；差值方向均为 MaskTopo 减基线。",
            "",
            "| 冻结对比 | 均值差 | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    contrast_labels = {
        "aligned_masktopo_minus_mask_conditioned_perceiver_strong": "MaskTopo − Mask-Perceiver strong（主对比）",
        "aligned_masktopo_minus_mask_conditioned_slot_param_matched": "MaskTopo − Mask-Slot matched（容量隔离）",
        "aligned_masktopo_minus_existing_mask_guided_queries_sanity": "MaskTopo − existing mask-guided（冻结 sanity）",
    }
    for key, label in contrast_labels.items():
        contrast = contrasts[key]
        lower, upper = contrast["source_cluster_bootstrap_95_ci"]
        lines.append(
            f"| {label} | {100*contrast['mean_gain']:+.2f} pp | "
            f"[{100*lower:+.2f}, {100*upper:+.2f}] pp |"
        )
    lines.extend(
        [
            "",
            "## 主张边界",
            "",
            result["claim_boundary"],
            "",
            "不得据此写成：在所有数据集、所有 K 或 untouched 外部域上普遍优于 mask-aware 非拓扑方法；也不得把端点连通增益与 P1–P5 疾病分类 `+1.00 pp` 合并。",
            "",
            "## 完整性与异常闭环",
            "",
            "- 3/3 最终运行共享完全相同的 code SHA、protocol SHA、固定 data seed；float16 存储、exact-K、参数量和非拓扑 firewall 均通过。",
            "- 每个新运行的 truth/source_id 与对应 aligned MaskTopo 逐项一致，保存预测重新计算的 balanced accuracy 与 result.json 一致。",
            "- 首次 20260811/12 因错误 data seed 产生的结果已标记 `INVALID_data_seed` 并完全排除；异常在统计前发现并记录。",
            "- 为统一代码哈希而重跑的 20260810，与原有效运行的 truth、source_id、二值预测和概率全部逐项相同。",
            "- Massachusetts 自然道路 test 未重新打开；本实验不改变该失败扩展的结论。",
            "",
            "## 产物",
            "",
            "- 协议：`MASK_AWARE_EQUAL_SUPERVISION_PROTOCOL.md`",
            "- 训练代码：`mask_aware_equal_supervision.py`",
            "- 汇总代码：`summarize_mask_aware_equal_supervision.py`",
            "- 三种子目录：`results_mask_aware_equal_supervision_deepcrack_k16_seed20260810/11/12`",
            "- 汇总目录：`results_mask_aware_equal_supervision_deepcrack_k16_summary/`",
            "- 异常记录：`实验记录/异常记录_TB-B-260803-032_data_seed偏差.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_self_test() -> None:
    truth = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    sources = np.asarray(["a", "a", "b", "b"])
    first = np.asarray([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.uint8)
    second = np.asarray([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)
    result = cluster_bootstrap(
        truth, sources, first, second, 100, np.random.default_rng(1)
    )
    if not np.isfinite(result["mean_gain"]):
        raise AssertionError("Cluster bootstrap self-test failed.")
    print("MASK_AWARE_EQUAL_SUPERVISION_SUMMARY_SELF_TEST_PASS", flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if list(args.seeds) != list(SEEDS):
        raise ValueError("Frozen optimization seeds changed.")
    if (
        args.bootstrap_repetitions != BOOTSTRAP_REPETITIONS
        or args.statistics_seed != STATISTICS_SEED
    ):
        raise ValueError("Frozen bootstrap protocol changed.")

    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (
        truth,
        sources,
        predictions,
        aggregates,
        seed_rows,
        source_rows,
        artifacts,
        shared_hashes,
    ) = load_and_validate(root, list(args.seeds))

    rng = np.random.default_rng(args.statistics_seed)
    pairs = {
        "aligned_masktopo_minus_mask_conditioned_perceiver_strong": (
            "aligned_masktopo",
            "mask_conditioned_perceiver_strong",
        ),
        "aligned_masktopo_minus_mask_conditioned_slot_param_matched": (
            "aligned_masktopo",
            "mask_conditioned_slot_param_matched",
        ),
        "aligned_masktopo_minus_existing_mask_guided_queries_sanity": (
            "aligned_masktopo",
            "existing_mask_guided_queries",
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
        for name, (first, second) in pairs.items()
    }
    primary = contrasts[
        "aligned_masktopo_minus_mask_conditioned_perceiver_strong"
    ]
    matched = contrasts[
        "aligned_masktopo_minus_mask_conditioned_slot_param_matched"
    ]
    primary_lower, primary_upper = primary["source_cluster_bootstrap_95_ci"]
    matched_lower, matched_upper = matched["source_cluster_bootstrap_95_ci"]
    if primary_lower > 0 and matched_lower > 0:
        verdict = "TOPOLOGY_SPECIFIC_DEEPCRACK_FAIRNESS_GATE_PASS"
        claim_impact = (
            "在当前 retrospective DeepCrack K=16 端点连通协议内，MaskTopo 对容量更大的强 mask-aware 非拓扑基线和参数匹配版均保持正且 CI 不跨零的优势。"
        )
        claim_boundary = (
            "允许将结果写成：在当前 DeepCrack K=16 retrospective endpoint-connectivity fairness test 中，MaskTopo 的优势不能由相同 mask 监督、显式 mask attention bias 或参数量差异解释。这里的“topology-specific”仅指该冻结对照排除了这些替代解释，不等于 untouched 外部泛化已确认。"
        )
    elif primary_upper < 0 or matched_upper < 0:
        verdict = "TOPOLOGY_SPECIFIC_DEEPCRACK_CLAIM_WITHDRAW"
        claim_impact = "至少一个冻结非拓扑基线的 CI 整体高于 MaskTopo，撤回 DeepCrack topology-specific superiority 主张。"
        claim_boundary = "DeepCrack 只能报告描述性结果，不得声称 topology-specific superiority。"
    else:
        verdict = "TOPOLOGY_SPECIFIC_DEEPCRACK_CLAIM_NARROW_TO_MASK_CONDITIONED_REDUCTION"
        claim_impact = "至少一个冻结公平对比跨零，不能把 DeepCrack 增益明确归因于拓扑组织。"
        claim_boundary = "主张必须收窄为有效的 mask-conditioned token reduction，不得声称 topology-specific superiority。"

    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_analyzed",
        "dataset": "deepcrack",
        "task": "endpoint_connectivity",
        "token_count": 16,
        "seeds": list(args.seeds),
        "data_seed": FROZEN_DATA_SEED,
        "split_seed": FROZEN_SPLIT_SEED,
        "test_crop_count": int(truth.size),
        "test_source_image_count": int(np.unique(sources).size),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "statistics_seed": args.statistics_seed,
        "aggregates": aggregates,
        "contrasts": contrasts,
        "verdict": verdict,
        "claim_impact": claim_impact,
        "claim_boundary": claim_boundary,
        "artifact_consistency": {
            "truth_equal_to_alignment_for_all_runs": True,
            "source_ids_equal_to_alignment_for_all_runs": True,
            "truth_and_sources_equal_across_seeds": True,
            "prediction_metrics_reproduced": True,
            "same_mask_archive_hash_as_existing_baseline": True,
            "float16_storage_verified": True,
            "exact_k_verified": True,
            "non_topological_firewall_verified": True,
            "parameter_match_within_one_percent": True,
            "shared_code_and_protocol_hashes": True,
            "invalid_data_seed_runs_excluded": True,
            "p1_to_p5_evidence_used": False,
        },
        "shared_hashes": shared_hashes,
        "artifacts": artifacts,
        "summary_code_sha256": file_sha256(Path(__file__)),
    }
    report = make_report(result)
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "artifact_hashes.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "seed_scores.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    with (output / "source_level_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    print(report, flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEED_RUNS = {
    20260820: Path("results_massroads_seed20260820_train_dev_retry1"),
    20260821: Path("results_massroads_seed20260821_train_dev"),
    20260822: Path("results_massroads_seed20260822_train_dev"),
}
MODELS = (
    "grid_external",
    "mask_assignment_identity",
    "shuffled_topology",
    "mask_topo_external",
    "tokenlearner",
    "perceiver_resampler",
    "tome_style",
    "g2tm_fixedk_mask",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze and verify Massachusetts Roads train/dev checkpoints."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("实验记录/TB-B-260802-021_train_dev_freeze_manifest.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("实验记录/TB-B-260802-021_train_dev_freeze_manifest.md"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> None:
    assert len(SEED_RUNS) == 3
    assert len(MODELS) == 8
    assert len(set(MODELS)) == len(MODELS)
    print("MASSROADS_FREEZE_MANIFEST_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = args.root.resolve()
    output = (root / args.output).resolve()
    report_path = (root / args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("Freeze manifest output already exists; refusing overwrite.")
    if (root / "real_data/MassachusettsRoads/test").exists():
        raise RuntimeError("Test directory exists before checkpoint freeze manifest.")

    audit_path = root / "results_massroads_data_audit/data_gate.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["verdict"] != "TRAIN_DEV_DATA_GATE_PASS" or audit["test_accessed"]:
        raise RuntimeError("Train/dev data gate is not frozen and clean.")
    fingerprint = str(audit["dataset_fingerprint"])
    runs: list[dict[str, Any]] = []
    all_hashes: set[str] = set()
    for seed, relative in SEED_RUNS.items():
        run_root = (root / relative).resolve()
        result_path = run_root / "train_dev_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "completed_train_dev_frozen":
            raise RuntimeError(f"Seed {seed} is not complete.")
        if result["test_accessed"] or int(result["optimization_seed"]) != seed:
            raise RuntimeError(f"Seed/test-access mismatch for {seed}.")
        if str(result["dataset_fingerprint"]) != fingerprint:
            raise RuntimeError(f"Dataset fingerprint mismatch for {seed}.")
        if int(result["token_count"]) != 8:
            raise RuntimeError(f"Token budget mismatch for {seed}.")
        if set(result["model_results"]) != set(MODELS):
            raise RuntimeError(f"Model set mismatch for {seed}.")
        checkpoints: dict[str, Any] = {}
        for model in MODELS:
            entry = result["model_results"][model]
            path = Path(entry["checkpoint"]).resolve()
            observed = sha256_file(path)
            if observed != entry["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint hash mismatch: {seed}, {model}.")
            if observed in all_hashes:
                raise RuntimeError(f"Duplicate checkpoint hash: {seed}, {model}.")
            all_hashes.add(observed)
            checkpoints[model] = {
                "path": str(path),
                "sha256": observed,
                "parameters": int(entry["parameters"]),
                "best_dev_balanced_accuracy": float(
                    entry["best_dev_balanced_accuracy"]
                ),
            }
        mask_path = Path(result["mask_checkpoint"]["path"]).resolve()
        mask_hash = sha256_file(mask_path)
        if mask_hash != result["mask_checkpoint"]["sha256"]:
            raise RuntimeError(f"Mask checkpoint hash mismatch for {seed}.")
        expected_pt_count = len(MODELS) + 1
        if len(list(run_root.glob("*.pt"))) != expected_pt_count:
            raise RuntimeError(f"Unexpected checkpoint count for {seed}.")
        runs.append(
            {
                "seed": seed,
                "run_root": str(run_root),
                "result_json": {
                    "path": str(result_path),
                    "sha256": sha256_file(result_path),
                },
                "selected_on_validation": result["selected_on_validation"],
                "mask_checkpoint": {"path": str(mask_path), "sha256": mask_hash},
                "checkpoints": checkpoints,
            }
        )

    protocol_path = root / "MASSACHUSETTS_ROADS_EXTERNAL_CONFIRMATION_PROTOCOL.md"
    training_script = root / "massroads_train_dev.py"
    manifest = {
        "experiment_id": "TB-B-260802-021_massroads_train_dev_freeze_manifest",
        "status": "TRAIN_DEV_CHECKPOINTS_FROZEN_BEFORE_TEST_ACCESS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "dataset_fingerprint": fingerprint,
        "token_count": 8,
        "optimization_seeds": list(SEED_RUNS),
        "models": list(MODELS),
        "data_gate": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "version": "massroads_protocol_v1.2",
        },
        "training_script": {
            "path": str(training_script),
            "sha256": sha256_file(training_script),
        },
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# TB-B-260802-021 train/dev 冻结清单",
        "",
        "- 状态：`TRAIN_DEV_CHECKPOINTS_FROZEN_BEFORE_TEST_ACCESS`",
        f"- 数据指纹：`{fingerprint}`",
        "- test 已访问：否",
        "- K：8",
        "",
        "| Seed | Mask threshold / closing | 端点模型 | checkpoint 数 |",
        "|---:|---:|---:|---:|",
    ]
    for run in runs:
        selected = run["selected_on_validation"]
        lines.append(
            f"| {run['seed']} | {selected['mask_threshold']} / "
            f"{selected['closing_iterations']} | {len(run['checkpoints'])} | 9 |"
        )
    lines.extend(
        [
            "",
            "每个 checkpoint 均已按训练结果 JSON 重新计算并核对 SHA-256；失败的首次 seed 20260820 目录不在本冻结清单中。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

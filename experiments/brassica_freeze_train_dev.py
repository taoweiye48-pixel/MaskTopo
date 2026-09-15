from __future__ import annotations

import argparse
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
import torch


EXPERIMENT_ID = "TB-B-260803-035"
SEEDS = (20260810, 20260811, 20260812)
ALL_MODELS = (
    "mask_topo_external",
    "mask_conditioned_perceiver_strong",
    "mask_conditioned_slot_param_matched",
    "ph_only",
    "ph_guided",
)
PH_MODELS = ("ph_only", "ph_guided")
FROZEN_PROTOCOL_HASHES = {
    "ROOTNAV_BRASSICA_UNTOUCHED_PROTOCOL.md": "d9725bff3f3865dd7e1f3f6c5bcb4aba6e10eafaccdfe53d58eeb0a1936f979f",
    "ROOTNAV_BRASSICA_UNTOUCHED_PROTOCOL_V2.md": "db40091a6d78d1ef980a3d93ca01f5467fb7d45ed7410b97ac233f0c030af1fb",
}
FROZEN_INPUT_HASHES = {
    "实验记录/TB-B-260803-034_source_manifest.json": "102ad36ce884045141cf5b71919667fa81a09e2d56829ebbc60022be0ae84de0",
    "实验记录/TB-B-260803-034_train_dev_extraction_manifest.json": "9735c85ae3186839b6a56b9091478a654ce9a82af377cf654e401ee756dfb457",
    "实验记录/TB-B-260803-035_data_gate/data_gate.json": "aefa91fb24a9ae596de55ca1d143d6006914e1e24b9bd89fe9eefbe8137a6e39",
}
CODE_FILES = (
    "brassica_train_dev.py",
    "crackforest_mask_topo.py",
    "crackforest_mechanism_ablation.py",
    "crackforest_real_gate.py",
    "mask_aware_equal_supervision.py",
    "ph_token_baselines.py",
    "strong_reducer_baselines.py",
    "topobridge_mvp.py",
    "topocoarsen_oracle.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze audited TB035 train/dev artifacts before test.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("实验记录/TB-B-260803-035_train_dev_freeze_manifest.json"),
    )
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


def verify_result(seed: int, result_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["experiment_id"] != EXPERIMENT_ID or result["status"] != "completed_train_dev_seed_frozen":
        raise ValueError(f"Seed {seed} result status invalid.")
    if result["optimization_seed"] != seed or result["data_seed"] != 20260810:
        raise ValueError(f"Seed metadata invalid for {seed}.")
    if result["sample_counts"] != {"train": 2400, "dev": 600} or result["token_count"] != 16:
        raise ValueError(f"Sample or K mismatch for {seed}.")
    if (
        result["test_accessed"]
        or result["test_image_member_request_count"] != 0
        or result["test_rsml_member_request_count"] != 0
    ):
        raise ValueError(f"Seed {seed} reports test access.")
    if result["mask_probability_storage_dtypes"] != {"train": "float16", "dev": "float16"}:
        raise ValueError(f"Seed {seed} probability storage is not float16.")
    if result["mask_probability_loaded_compute_dtype"] != "float32":
        raise ValueError(f"Seed {seed} compute reload dtype changed.")
    if result["dev_mask_metrics"]["f1"] < 0.30:
        raise ValueError(f"Seed {seed} mask F1 gate failed.")
    if not result["parameter_gate"]["pass"] or result["parameter_gate"]["relative_error"] > 0.01:
        raise ValueError(f"Seed {seed} parameter gate failed.")
    if not result["self_test_checks"] or not all(result["self_test_checks"].values()):
        raise ValueError(f"Seed {seed} self-test gate failed.")
    if tuple(result["model_results"].keys()) != ALL_MODELS:
        raise ValueError(f"Seed {seed} model set or ordering changed.")

    referenced = {
        "result": checked_file(result_path),
        "mask_checkpoint": checked_file(
            Path(result["mask_checkpoint"]["path"]), result["mask_checkpoint"]["sha256"]
        ),
        "mask_probabilities": checked_file(
            Path(result["mask_probabilities"]["path"]), result["mask_probabilities"]["sha256"]
        ),
        "ph_cache": checked_file(Path(result["ph_cache"]["path"]), result["ph_cache"]["sha256"]),
        "dev_predictions": checked_file(
            Path(result["dev_predictions"]["path"]), result["dev_predictions"]["sha256"]
        ),
        "model_checkpoints": {},
    }
    for model_name in ALL_MODELS:
        model = result["model_results"][model_name]
        if model["seed"] != seed or not np.isfinite(model["best_dev_balanced_accuracy"]):
            raise ValueError(f"Seed/model metadata invalid: {seed}/{model_name}")
        referenced["model_checkpoints"][model_name] = checked_file(
            Path(model["checkpoint"]), model["checkpoint_sha256"]
        )

    with np.load(result["mask_probabilities"]["path"], allow_pickle=False) as archive:
        if archive["train"].shape != (2400, 64, 64) or archive["dev"].shape != (600, 64, 64):
            raise ValueError(f"Seed {seed} probability shape changed.")
        if archive["train"].dtype != np.float16 or archive["dev"].dtype != np.float16:
            raise ValueError(f"Seed {seed} probability dtype changed.")

    return result, referenced


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite freeze manifest: {output}")
    marker = root / "实验记录" / f"{EXPERIMENT_ID}_TEST_INFERENCE_STARTED.json"
    if marker.exists():
        raise ValueError("Test marker exists before train/dev freeze.")

    frozen_inputs: dict[str, Any] = {}
    for relative, expected in {**FROZEN_PROTOCOL_HASHES, **FROZEN_INPUT_HASHES}.items():
        frozen_inputs[relative] = checked_file(root / relative, expected)

    source_manifest = json.loads(
        (root / "实验记录/TB-B-260803-034_source_manifest.json").read_text(encoding="utf-8")
    )
    if (
        source_manifest["test_image_member_request_count"] != 0
        or source_manifest["test_rsml_member_request_count"] != 0
    ):
        raise ValueError("Source manifest reports pre-freeze test access.")
    test_ids = [item["source_id"] for item in source_manifest["sources"]["test"]]
    if len(test_ids) != 15 or len(set(test_ids)) != 15:
        raise ValueError("Frozen test ID set invalid.")

    results: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for seed in SEEDS:
        result_path = root / "实验记录" / f"{EXPERIMENT_ID}_seed{seed}" / "train_dev_result.json"
        result, referenced = verify_result(seed, result_path)
        results[str(seed)] = result
        artifacts[str(seed)] = referenced

    dev_values = {
        model: [results[str(seed)]["model_results"][model]["best_dev_balanced_accuracy"] for seed in SEEDS]
        for model in ALL_MODELS
    }
    dev_summary = {
        model: {
            "values": values,
            "mean": mean(values),
            "sample_sd": stdev(values),
        }
        for model, values in dev_values.items()
    }
    ph_diff = dev_summary["ph_only"]["mean"] - dev_summary["ph_guided"]["mean"]
    best_ph = "ph_guided" if abs(ph_diff) <= 1e-12 else max(PH_MODELS, key=lambda x: dev_summary[x]["mean"])
    selected_models = (
        "mask_topo_external",
        "mask_conditioned_perceiver_strong",
        "mask_conditioned_slot_param_matched",
        best_ph,
    )

    code = {relative: checked_file(root / relative) for relative in CODE_FILES}
    logs = {}
    for seed in SEEDS:
        logs[str(seed)] = {
            stream: checked_file(root / "实验记录" / f"{EXPERIMENT_ID}_seed{seed}.{stream}.log")
            for stream in ("stdout", "stderr")
        }

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "TRAIN_DEV_FROZEN_TEST_STILL_SEALED",
        "created_at": datetime.now().astimezone().isoformat(),
        "claim_boundary": "untouched-test external validation after disclosed train/dev task-shortcut repair",
        "endpoint_disease_firewall": "Endpoint-connectivity effects are separate from P1-P5 disease classification +1.00 pp.",
        "test_accessed": False,
        "test_image_member_request_count": 0,
        "test_rsml_member_request_count": 0,
        "test_marker_exists": False,
        "test_ids": test_ids,
        "test_source_count": len(test_ids),
        "test_sample_count": 1200,
        "optimization_seeds": list(SEEDS),
        "data_seed": 20260810,
        "bootstrap_seed": 20260803,
        "bootstrap_replicates": 20000,
        "token_count": 16,
        "dev_summary": dev_summary,
        "ph_selection_rule": "three-seed mean dev balanced accuracy; ties within 1e-12 select ph_guided",
        "selected_best_ph": best_ph,
        "selected_test_models": list(selected_models),
        "frozen_inputs": frozen_inputs,
        "training_code": code,
        "seed_artifacts": artifacts,
        "logs": logs,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_best_ph": best_ph,
                "selected_test_models": list(selected_models),
                "output": str(output),
                "sha256": sha256_file(output),
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

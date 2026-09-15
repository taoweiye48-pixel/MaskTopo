from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from robustness_stress import MODELS, mask_predictor_path, model_spec, prior_root


JOBS = tuple(
    (dataset, seed)
    for dataset in ("crackforest", "deepcrack")
    for seed in (20260810, 20260811, 20260812)
)


def write_status(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(".").resolve()
    log_root = root / "实验记录"
    planned = []
    for dataset, seed in JOBS:
        output = root / f"results_robustness_{dataset}_k16_seed{seed}_retry2"
        stem = f"TB-B-260802-024_{dataset}_k16_seed{seed}_retry2"
        stdout = log_root / f"{stem}.stdout.log"
        stderr = log_root / f"{stem}.stderr.log"
        required = [
            prior_root(root, dataset, seed) / "mask_probabilities.npz",
            mask_predictor_path(root, dataset, seed),
            root / f"ph_cache/{dataset}_seed{seed}_k64.npz",
            *[model_spec(root, dataset, seed, model)["checkpoint"] for model in MODELS],
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing retry2 input: {missing}")
        if output.exists() or stdout.exists() or stderr.exists():
            raise FileExistsError(
                f"Refusing to overwrite retry2 target: {output} / {stdout} / {stderr}"
            )
        planned.append(
            {
                "dataset": dataset,
                "seed": seed,
                "output": output,
                "stdout": stdout,
                "stderr": stderr,
                "command": [
                    sys.executable,
                    "robustness_stress.py",
                    "--dataset",
                    dataset,
                    "--seed",
                    str(seed),
                    "--output",
                    output.name,
                ],
            }
        )
    status_path = log_root / "TB-B-260802-024_queue_retry2_status.json"
    if status_path.exists():
        raise FileExistsError(f"Refusing to overwrite retry2 status: {status_path}")
    status: dict[str, Any] = {
        "experiment_id": "TB-B-260802-024",
        "queue": "retry2_all_six_consistent_precision_protocol",
        "status": "running",
        "started_unix": time.time(),
        "jobs_total": len(planned),
        "jobs_completed": 0,
        "current_job": None,
        "completed": [],
        "supersedes_outputs": [
            "results_robustness_crackforest_k16_seed20260810",
            "results_robustness_crackforest_k16_seed20260811_retry1",
        ],
        "massachusetts_test_reopened": False,
    }
    write_status(status_path, status)
    for index, job in enumerate(planned, start=1):
        status["current_job"] = {
            "index": index,
            "dataset": job["dataset"],
            "seed": job["seed"],
            "started_unix": time.time(),
        }
        write_status(status_path, status)
        print(
            f"ROBUSTNESS_RETRY2_START {index}/{len(planned)} "
            f"dataset={job['dataset']} seed={job['seed']}",
            flush=True,
        )
        with job["stdout"].open("w", encoding="utf-8") as out_handle, job[
            "stderr"
        ].open("w", encoding="utf-8") as err_handle:
            completed = subprocess.run(
                job["command"], cwd=root, stdout=out_handle, stderr=err_handle, check=False
            )
        record = {
            "index": index,
            "dataset": job["dataset"],
            "seed": job["seed"],
            "returncode": completed.returncode,
            "output": str(job["output"]),
            "stdout": str(job["stdout"]),
            "stderr": str(job["stderr"]),
            "finished_unix": time.time(),
        }
        status["completed"].append(record)
        if completed.returncode != 0:
            status["status"] = "failed_stop_no_further_retry"
            status["current_job"] = record
            write_status(status_path, status)
            print(
                f"ROBUSTNESS_RETRY2_FAIL index={index} returncode={completed.returncode}",
                flush=True,
            )
            raise SystemExit(completed.returncode)
        if not (job["output"] / "result.json").exists():
            status["status"] = "failed_stop_missing_result"
            status["current_job"] = record
            write_status(status_path, status)
            raise RuntimeError(f"Missing retry2 result: {job['output']}")
        status["jobs_completed"] = index
        status["current_job"] = None
        write_status(status_path, status)
        print(
            f"ROBUSTNESS_RETRY2_DONE {index}/{len(planned)} "
            f"dataset={job['dataset']} seed={job['seed']}",
            flush=True,
        )
    status["status"] = "completed"
    status["finished_unix"] = time.time()
    write_status(status_path, status)
    print("ROBUSTNESS_RETRY2_COMPLETED", flush=True)


if __name__ == "__main__":
    main()

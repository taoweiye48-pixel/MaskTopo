from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from robustness_stress import MODELS, mask_predictor_path, model_spec, ph_run_root, prior_root


DATASETS = ("crackforest", "deepcrack")
SEEDS = (20260810, 20260811, 20260812)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen TB-B-260802-024 queue.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_status(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    log_root = root / "实验记录"
    jobs = []
    for dataset in DATASETS:
        for seed in SEEDS:
            output = root / f"results_robustness_{dataset}_k16_seed{seed}"
            stem = f"TB-B-260802-024_{dataset}_k16_seed{seed}"
            stdout = log_root / f"{stem}.stdout.log"
            stderr = log_root / f"{stem}.stderr.log"
            command = [
                sys.executable,
                "robustness_stress.py",
                "--dataset",
                dataset,
                "--seed",
                str(seed),
                "--output",
                output.name,
            ]
            required = [
                prior_root(root, dataset, seed) / "mask_probabilities.npz",
                mask_predictor_path(root, dataset, seed),
                root / f"ph_cache/{dataset}_seed{seed}_k64.npz",
                *[
                    model_spec(root, dataset, seed, model)["checkpoint"]
                    for model in MODELS
                ],
            ]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Frozen robustness inputs missing for {dataset}/{seed}: {missing}"
                )
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite output: {output}")
            if stdout.exists() or stderr.exists():
                raise FileExistsError(f"Refusing to overwrite logs: {stdout} / {stderr}")
            jobs.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "command": command,
                    "output": str(output),
                    "stdout": str(stdout),
                    "stderr": str(stderr),
                }
            )
    if args.dry_run:
        print(json.dumps(jobs, ensure_ascii=False, indent=2))
        return
    status_path = log_root / "TB-B-260802-024_queue_status.json"
    if status_path.exists():
        raise FileExistsError(f"Refusing to overwrite queue status: {status_path}")
    status: dict[str, Any] = {
        "experiment_id": "TB-B-260802-024",
        "status": "running",
        "started_unix": time.time(),
        "jobs_total": len(jobs),
        "jobs_completed": 0,
        "current_job": None,
        "completed": [],
        "massachusetts_test_reopened": False,
    }
    write_status(status_path, status)
    for index, job in enumerate(jobs, start=1):
        status["current_job"] = {
            "index": index,
            "dataset": job["dataset"],
            "seed": job["seed"],
            "started_unix": time.time(),
        }
        write_status(status_path, status)
        print(
            f"ROBUSTNESS_QUEUE_START {index}/{len(jobs)} "
            f"dataset={job['dataset']} seed={job['seed']}",
            flush=True,
        )
        with Path(job["stdout"]).open("w", encoding="utf-8") as out_handle, Path(
            job["stderr"]
        ).open("w", encoding="utf-8") as err_handle:
            completed = subprocess.run(
                job["command"],
                cwd=root,
                stdout=out_handle,
                stderr=err_handle,
                check=False,
            )
        record = {
            "index": index,
            "dataset": job["dataset"],
            "seed": job["seed"],
            "returncode": completed.returncode,
            "output": job["output"],
            "stdout": job["stdout"],
            "stderr": job["stderr"],
            "finished_unix": time.time(),
        }
        status["completed"].append(record)
        if completed.returncode != 0:
            status["status"] = "failed_stop_no_retry"
            status["current_job"] = record
            write_status(status_path, status)
            print(
                f"ROBUSTNESS_QUEUE_FAIL index={index} returncode={completed.returncode}",
                flush=True,
            )
            raise SystemExit(completed.returncode)
        result_path = Path(job["output"]) / "result.json"
        if not result_path.exists():
            status["status"] = "failed_stop_missing_result"
            status["current_job"] = record
            write_status(status_path, status)
            raise RuntimeError(f"Completed job produced no result: {result_path}")
        status["jobs_completed"] = index
        status["current_job"] = None
        write_status(status_path, status)
        print(
            f"ROBUSTNESS_QUEUE_DONE {index}/{len(jobs)} "
            f"dataset={job['dataset']} seed={job['seed']}",
            flush=True,
        )
    status["status"] = "completed"
    status["finished_unix"] = time.time()
    write_status(status_path, status)
    print("ROBUSTNESS_QUEUE_COMPLETED", flush=True)


if __name__ == "__main__":
    main()

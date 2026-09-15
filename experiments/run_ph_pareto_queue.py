from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DATASETS = ("deepcrack", "crackforest")
TOKEN_COUNTS = (8, 32, 64)
SEEDS = (20260810, 20260811, 20260812)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen TB-B-260802-023 PH Pareto queue."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_for(
    root: Path, dataset: str, token_count: int, seed: int
) -> tuple[list[str], Path, Path, Path]:
    if dataset == "deepcrack":
        cache_dir = "deepcrack_cache"
        mask = f"results_deepcrack_seed{seed}\\mask_probabilities.npz"
    else:
        cache_dir = "real_cache"
        mask = f"results_crackforest_mechanism_seed{seed}\\mask_probabilities.npz"
    ph_cache = f"ph_cache\\{dataset}_seed{seed}_k64.npz"
    output = root / f"results_ph_{dataset}_k{token_count}_seed{seed}"
    log_stem = f"TB-B-260802-023_{dataset}_k{token_count}_seed{seed}"
    stdout = root / "实验记录" / f"{log_stem}.stdout.log"
    stderr = root / "实验记录" / f"{log_stem}.stderr.log"
    command = [
        sys.executable,
        "ph_token_baselines.py",
        "--dataset",
        dataset,
        "--cache-dir",
        cache_dir,
        "--mask-probabilities",
        mask,
        "--ph-cache",
        ph_cache,
        "--output",
        output.name,
        "--token-count",
        str(token_count),
        "--max-ph-tokens",
        "64",
        "--seed",
        str(seed),
        "--split-seed",
        "20260730",
        "--models",
        "ph_only",
        "ph_guided",
    ]
    return command, output, stdout, stderr


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    jobs = [
        (dataset, token_count, seed)
        for dataset in DATASETS
        for token_count in TOKEN_COUNTS
        for seed in SEEDS
    ]
    planned = []
    for dataset, token_count, seed in jobs:
        command, output, stdout, stderr = command_for(
            root, dataset, token_count, seed
        )
        mask_index = command.index("--mask-probabilities") + 1
        ph_index = command.index("--ph-cache") + 1
        required = [root / command[mask_index], root / command[ph_index]]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Frozen queue input missing: {path}")
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output directory: {output}"
            )
        if stdout.exists() or stderr.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing log: {stdout} / {stderr}"
            )
        planned.append(
            {
                "dataset": dataset,
                "token_count": token_count,
                "seed": seed,
                "command": command,
                "output": str(output),
                "stdout": str(stdout),
                "stderr": str(stderr),
            }
        )
    if args.dry_run:
        print(json.dumps(planned, ensure_ascii=False, indent=2))
        return
    status_path = root / "实验记录" / "TB-B-260802-023_queue_status.json"
    status: dict[str, Any] = {
        "experiment_id": "TB-B-260802-023",
        "status": "running",
        "started_unix": time.time(),
        "jobs_total": len(planned),
        "jobs_completed": 0,
        "current_job": None,
        "completed": [],
        "massachusetts_test_reopened": False,
    }
    write_status(status_path, status)
    for index, job in enumerate(planned, start=1):
        status["current_job"] = {
            "index": index,
            "dataset": job["dataset"],
            "token_count": job["token_count"],
            "seed": job["seed"],
            "started_unix": time.time(),
        }
        write_status(status_path, status)
        print(
            f"PH_PARETO_QUEUE_START {index}/{len(planned)} "
            f"dataset={job['dataset']} K={job['token_count']} seed={job['seed']}",
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
            "token_count": job["token_count"],
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
                f"PH_PARETO_QUEUE_FAIL index={index} "
                f"returncode={completed.returncode}",
                flush=True,
            )
            raise SystemExit(completed.returncode)
        result_path = Path(job["output"]) / "result.json"
        if not result_path.exists():
            status["status"] = "failed_stop_missing_result"
            status["current_job"] = record
            write_status(status_path, status)
            raise RuntimeError(f"Completed command produced no result.json: {result_path}")
        status["jobs_completed"] = index
        status["current_job"] = None
        write_status(status_path, status)
        print(
            f"PH_PARETO_QUEUE_DONE {index}/{len(planned)} "
            f"dataset={job['dataset']} K={job['token_count']} seed={job['seed']}",
            flush=True,
        )
    status["status"] = "completed"
    status["finished_unix"] = time.time()
    write_status(status_path, status)
    print("PH_PARETO_QUEUE_COMPLETED", flush=True)


if __name__ == "__main__":
    main()

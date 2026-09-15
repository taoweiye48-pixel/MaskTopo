from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from edge_topocoarsen_connector import (
    EdgeCoarsenDataset,
    train_connector,
)
from topobridge_mvp import load_or_generate
from topocoarsen_learned import slice_arrays
from topocoarsen_oracle import coarse_graph, grid_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired fixed-split multi-seed grid control."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260810, 20260811, 20260812, 20260813, 20260814],
    )
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--dev-size", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument(
        "--output", type=Path, default=Path("results_grid_multiseed")
    )
    return parser.parse_args()


def grid_artifacts(
    sample_count: int, token_count: int
) -> dict[str, np.ndarray]:
    assignment = grid_assignment(token_count)
    adjacency, reachability = coarse_graph(
        assignment,
        np.zeros((16, 16), dtype=np.int16),
        token_count,
        topology_aware=False,
    )
    return {
        "assignment": np.repeat(
            assignment[None], sample_count, axis=0
        ),
        "adjacency": np.repeat(
            adjacency[None], sample_count, axis=0
        ),
        "reachability": np.repeat(
            reachability[None], sample_count, axis=0
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_arrays = load_or_generate(
        args.cache_dir.resolve(),
        "train",
        args.train_size,
        args.data_seed,
    )
    validation_arrays = load_or_generate(
        args.cache_dir.resolve(),
        "val",
        args.val_size,
        args.data_seed + 1_000_000,
    )
    dev_arrays = slice_arrays(validation_arrays, 0, args.dev_size)
    test_arrays = slice_arrays(
        validation_arrays, args.dev_size, args.val_size
    )
    train_dataset = EdgeCoarsenDataset(
        train_arrays, grid_artifacts(args.train_size, args.token_count)
    )
    dev_dataset = EdgeCoarsenDataset(
        dev_arrays, grid_artifacts(args.dev_size, args.token_count)
    )
    test_dataset = EdgeCoarsenDataset(
        test_arrays,
        grid_artifacts(args.val_size - args.dev_size, args.token_count),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = []
    histories = []
    for seed in args.seeds:
        args.seed = seed
        summary, history, _, _ = train_connector(
            args,
            train_dataset,
            dev_dataset,
            test_dataset,
            output,
            device,
            model_name="grid",
        )
        summaries.append(summary)
        histories.extend(history)
    values = np.asarray(
        [
            row["test_metrics"]["balanced_accuracy"]
            for row in summaries
        ]
    )
    result = {
        "experiment_id": "grid_fixed_split_multiseed_control",
        "data_seed": args.data_seed,
        "optimization_seeds": args.seeds,
        "per_seed": summaries,
        "mean_test_balanced_accuracy": float(values.mean()),
        "sample_standard_deviation": float(values.std(ddof=1)),
        "minimum_test_balanced_accuracy": float(values.min()),
        "maximum_test_balanced_accuracy": float(values.max()),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output / "history.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

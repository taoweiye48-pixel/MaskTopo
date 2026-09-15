from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from crackforest_mask_topo import find_cached_split
from crackforest_real_gate import RealDataset, fixed_artifacts, train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matched fixed-grid controls for the CrackForest MaskTopo runs."
        )
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("real_cache"))
    parser.add_argument(
        "--output", type=Path, default=Path("results_crackforest_grid_multiseed")
    )
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--dev-size", type=int, default=320)
    parser.add_argument("--test-size", type=int, default=320)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260810, 20260811, 20260812]
    )
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.crop_size != 64 or args.token_count != 16:
        raise ValueError("This control requires crop-size=64 and 16 tokens.")
    arrays = {
        "train": find_cached_split(
            args.cache_dir.resolve(),
            "train",
            args.train_size,
            args.data_seed,
            args.crop_size,
        ),
        "dev": find_cached_split(
            args.cache_dir.resolve(),
            "dev",
            args.dev_size,
            args.data_seed + 1_000_000,
            args.crop_size,
        ),
        "test": find_cached_split(
            args.cache_dir.resolve(),
            "test",
            args.test_size,
            args.data_seed + 2_000_000,
            args.crop_size,
        ),
    }
    artifacts = {
        split: fixed_artifacts(current, args.token_count, "grid")[0]
        for split, current in arrays.items()
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    predictions = {}
    truth_reference: np.ndarray | None = None
    for seed in args.seeds:
        current_args = argparse.Namespace(**vars(args))
        current_args.seed = seed
        seed_output = output / f"seed{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        summary, _, prediction, truth = train_classifier(
            "grid",
            current_args,
            RealDataset(arrays["train"], artifacts["train"], augment=True),
            RealDataset(arrays["dev"], artifacts["dev"]),
            RealDataset(arrays["test"], artifacts["test"]),
            seed_output,
            device,
        )
        if truth_reference is None:
            truth_reference = truth
        elif not np.array_equal(truth_reference, truth):
            raise RuntimeError("Held-out labels changed between matched runs.")
        results.append(
            {
                "optimization_seed": seed,
                "test_metrics": summary["test_metrics"],
                "best_dev_balanced_accuracy": summary[
                    "best_dev_balanced_accuracy"
                ],
            }
        )
        predictions[f"seed{seed}"] = prediction
    scores = np.array(
        [item["test_metrics"]["balanced_accuracy"] for item in results],
        dtype=np.float64,
    )
    report = {
        "experiment_id": "crackforest_fixed_grid_multiseed_control",
        "data_seed": args.data_seed,
        "optimization_seeds": args.seeds,
        "runs": results,
        "mean_test_balanced_accuracy": float(scores.mean()),
        "sample_std_test_balanced_accuracy": float(scores.std(ddof=1)),
    }
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if truth_reference is None:
        raise RuntimeError("No runs were requested.")
    np.savez_compressed(
        output / "heldout_predictions.npz",
        truth=truth_reference,
        **predictions,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

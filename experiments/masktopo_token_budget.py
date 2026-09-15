from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from crackforest_mechanism_ablation import build_artifacts
from crackforest_real_gate import RealDataset, train_classifier, write_csv
from strong_reducer_baselines import load_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Variable-token MaskTopo accuracy experiment."
    )
    parser.add_argument(
        "--dataset", choices=("crackforest", "deepcrack"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-run-dir", type=Path, required=True)
    parser.add_argument("--mask-probabilities", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("real_data/DeepCrack/dataset/extracted"),
    )
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--dev-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--shuffle-seed", type=int, default=20260731)
    return parser.parse_args()


def selected_mask_rule(
    prior: dict[str, Any], dataset: str
) -> tuple[float, int]:
    key = (
        "selected_on_prior_dev"
        if dataset == "crackforest"
        else "selected_on_dev"
    )
    selected = prior[key]
    return (
        float(selected["mask_threshold"]),
        int(selected["closing_iterations"]),
    )


def main() -> None:
    args = parse_args()
    if args.token_count not in {8, 16, 32, 64}:
        raise ValueError("token-count must be one of 8, 16, 32, or 64.")
    if args.crop_size != 64 or args.output_size != 64:
        raise ValueError("The frozen budget protocol requires 64x64 inputs.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior_root = args.prior_run_dir.resolve()
    prior = json.loads(
        (prior_root / "result.json").read_text(encoding="utf-8")
    )
    if int(prior["optimization_seed"]) != args.seed:
        raise ValueError("Prior mask run and optimization seed do not match.")
    threshold, closing = selected_mask_rule(prior, args.dataset)
    arrays = load_arrays(args)
    mask_path = args.mask_probabilities.resolve()
    with np.load(mask_path) as archive:
        probabilities = {
            split: archive[split][
                : arrays[split]["images"].shape[0]
            ].astype(np.float32)
            for split in ("train", "dev", "test")
        }
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    preprocessing: dict[str, Any] = {}
    for split_index, split in enumerate(("train", "dev", "test")):
        started = time.perf_counter()
        artifacts[split], diagnostics[split] = build_artifacts(
            arrays[split],
            probabilities[split],
            threshold,
            closing,
            args.token_count,
            "mask_topo_recheck",
            args.shuffle_seed + 100_000 * split_index,
        )
        elapsed = time.perf_counter() - started
        preprocessing[split] = {
            "seconds": elapsed,
            "milliseconds_per_sample": (
                1000.0 * elapsed / arrays[split]["images"].shape[0]
            ),
        }
    model_name = f"mask_topo_budget_k{args.token_count}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary, history, prediction, truth = train_classifier(
        model_name,
        args,
        RealDataset(arrays["train"], artifacts["train"], augment=True),
        RealDataset(arrays["dev"], artifacts["dev"]),
        RealDataset(arrays["test"], artifacts["test"]),
        output,
        device,
    )
    write_csv(output / "history.csv", history)
    prediction_payload: dict[str, np.ndarray] = {
        "truth": truth,
        "mask_topo": prediction,
    }
    if args.dataset == "deepcrack":
        prediction_payload["source_id"] = arrays["test"]["source_id"]
    np.savez_compressed(
        output / "heldout_predictions.npz", **prediction_payload
    )
    result = {
        "experiment_id": "masktopo_token_budget",
        "status": "completed",
        "dataset": args.dataset,
        "optimization_seed": args.seed,
        "data_seed": args.data_seed,
        "fine_token_count": 256,
        "reduced_token_count": args.token_count,
        "selected_on_prior_dev": {
            "mask_threshold": threshold,
            "closing_iterations": closing,
        },
        "prior_run_dir": str(prior_root),
        "mask_probabilities": str(mask_path),
        "model_result": summary,
        "artifact_diagnostics": diagnostics,
        "topology_preprocessing": preprocessing,
        "test_was_previously_observed": True,
        "selection_uses_development_only": True,
        "protocol": str(
            (Path(__file__).parent / "TOKEN_BUDGET_PROTOCOL.md").resolve()
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


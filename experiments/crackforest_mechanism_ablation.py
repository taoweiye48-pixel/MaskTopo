from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crackforest_mask_topo import (
    CrackUNet,
    find_cached_split,
    mask_groups,
    predict_masks,
)
from crackforest_real_gate import (
    RealDataset,
    balanced_metrics,
    coarse_graph8,
    evaluate_classifier,
    fixed_artifacts,
    group_structural_prediction,
    model_forward,
    train_classifier,
    write_csv,
)
from topobridge_mvp import set_seed
from topocoarsen_oracle import TopoCoarsenModel, oracle_assignment


MODES = (
    "grid_recheck",
    "mask_feature_grid",
    "mask_assignment_identity",
    "grid_mask_graph",
    "shuffled_topology",
    "mask_topo_recheck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mechanism ablations for the CrackForest MaskTopo gain."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("real_cache"))
    parser.add_argument("--mask-run-dir", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--dev-size", type=int, default=320)
    parser.add_argument("--test-size", type=int, default=320)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--shuffle-seed", type=int, default=20260731)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def limit_groups(groups: np.ndarray, maximum_foreground: int = 15) -> np.ndarray:
    groups = groups.astype(np.int16, copy=True)
    components = [int(item) for item in np.unique(groups) if int(item) > 0]
    if len(components) > maximum_foreground:
        sizes = {
            component: int(np.sum(groups == component))
            for component in components
        }
        keep = set(
            sorted(components, key=sizes.__getitem__, reverse=True)[
                :maximum_foreground
            ]
        )
        for component in components:
            if component not in keep:
                groups[groups == component] = 0
    ordered = [
        0,
        *sorted(int(item) for item in np.unique(groups) if int(item) > 0),
    ]
    remap = {old: new for new, old in enumerate(ordered)}
    return np.vectorize(remap.__getitem__, otypes=[np.int16])(groups)


def predicted_groups(
    probability: np.ndarray,
    threshold: float,
    closing_iterations: int,
    maximum_foreground: int = 15,
) -> np.ndarray:
    _, groups = mask_groups(probability, threshold, closing_iterations)
    return limit_groups(groups, maximum_foreground)


def build_artifacts(
    arrays: dict[str, np.ndarray],
    probabilities: np.ndarray,
    threshold: float,
    closing_iterations: int,
    token_count: int,
    mode: str,
    shuffle_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample_count = arrays["images"].shape[0]
    assignments = np.empty((sample_count, 16, 16), dtype=np.int16)
    adjacency = np.empty(
        (sample_count, token_count, token_count), dtype=np.uint8
    )
    reachability = np.empty_like(adjacency)
    structural = np.empty(sample_count, dtype=np.uint8)
    graph_density = np.empty(sample_count, dtype=np.float64)
    reachability_density = np.empty(sample_count, dtype=np.float64)
    base = None
    if mode == "grid_mask_graph":
        side = int(round(np.sqrt(token_count)))
        if side * side != token_count or 16 % side:
            raise ValueError(
                "grid_mask_graph requires a square token count whose side "
                "divides 16."
            )
        base = np.arange(token_count, dtype=np.int16).reshape(side, side)
        base = base.repeat(16 // side, axis=0).repeat(
            16 // side, axis=1
        )
    identity = np.eye(token_count, dtype=np.uint8)
    rng = np.random.default_rng(shuffle_seed)
    for index in range(sample_count):
        groups = predicted_groups(
            probabilities[index],
            threshold,
            closing_iterations,
            maximum_foreground=token_count - 1,
        )
        artifact_groups = groups
        if mode == "shuffled_topology":
            artifact_groups = groups.reshape(-1)[
                rng.permutation(groups.size)
            ].reshape(groups.shape)
        if mode == "grid_mask_graph":
            if base is None:
                raise AssertionError("Missing grid assignment.")
            assignment = base
            graph, closure = coarse_graph8(
                assignment, artifact_groups, token_count, topology_aware=True
            )
        elif mode == "mask_assignment_identity":
            assignment = oracle_assignment(
                artifact_groups, token_count, bridge_aware=False
            )
            graph = identity
            closure = identity
        elif mode in {"shuffled_topology", "mask_topo_recheck"}:
            assignment = oracle_assignment(
                artifact_groups, token_count, bridge_aware=False
            )
            graph, closure = coarse_graph8(
                assignment, artifact_groups, token_count, topology_aware=True
            )
        else:
            raise ValueError(mode)
        if np.unique(assignment).size != token_count:
            raise RuntimeError(f"{mode} produced empty tokens at sample {index}.")
        assignments[index] = assignment
        adjacency[index] = graph
        reachability[index] = closure
        structural[index] = group_structural_prediction(
            artifact_groups, arrays["endpoint_patches"][index]
        )
        graph_density[index] = float(graph.mean())
        reachability_density[index] = float(closure.mean())
    return {
        "assignment": assignments,
        "adjacency": adjacency,
        "reachability": reachability,
    }, {
        "direct_structural_metrics": balanced_metrics(
            arrays["connected"], structural
        ),
        "mean_adjacency_density": float(graph_density.mean()),
        "mean_reachability_density": float(reachability_density.mean()),
        "all_tokens_nonempty": True,
    }


class MaskFeatureDataset(RealDataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        artifacts: dict[str, np.ndarray],
        probabilities: np.ndarray,
        augment: bool,
    ) -> None:
        super().__init__(arrays, artifacts, augment)
        if probabilities.shape != (
            arrays["images"].shape[0],
            64,
            64,
        ):
            raise ValueError("Mask probabilities do not align with images.")
        self.probabilities = probabilities.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        probability = torch.from_numpy(self.probabilities[index]).unsqueeze(0)
        item["image"] = torch.cat((item["image"], probability), dim=0)
        return item


class FourChannelTopoCoarsen(nn.Module):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__()
        self.model = TopoCoarsenModel(
            "mask_feature_grid", dim, token_count
        )
        self.model.stem.net[0] = nn.Conv2d(
            4, 32, kernel_size=3, stride=2, padding=1
        )

    def forward(
        self,
        image: torch.Tensor,
        assignment: torch.Tensor,
        adjacency: torch.Tensor,
        reachability: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(image, assignment, adjacency, reachability)


def train_mask_feature_grid(
    args: argparse.Namespace,
    train_dataset: MaskFeatureDataset,
    dev_dataset: MaskFeatureDataset,
    test_dataset: MaskFeatureDataset,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    model_name = "mask_feature_grid"
    set_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = FourChannelTopoCoarsen(args.dim, args.token_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = output_dir / f"{model_name}_seed{args.seed}.pt"
    best = -1.0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = batch["connected"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model_forward(model, model_name, batch, device)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            batches += 1
        dev_metrics, _, _ = evaluate_classifier(
            model, model_name, dev_loader, device
        )
        record = {
            "stage": "classifier",
            "seed": args.seed,
            "model": model_name,
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"REAL_CLASSIFIER model={model_name} "
            f"epoch={epoch}/{args.epochs} "
            f"loss={record['train_loss']:.4f} "
            f"dev_bal={dev_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        if dev_metrics["balanced_accuracy"] > best:
            best = dev_metrics["balanced_accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "model_name": model_name,
                    "dev_metrics": dev_metrics,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    test_metrics, prediction, truth = evaluate_classifier(
        model, model_name, test_loader, device, return_predictions=True
    )
    if prediction is None or truth is None:
        raise AssertionError("Missing test predictions.")
    return {
        "model": model_name,
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_dev_balanced_accuracy": best,
        "test_metrics": test_metrics,
        "training_seconds": time.time() - started,
        "test_inference_uses_ground_truth_topology": False,
    }, history, prediction, truth


def run_self_test() -> None:
    arrays = {
        "images": np.zeros((4, 3, 64, 64), dtype=np.uint8),
        "connected": np.array([0, 1, 0, 1], dtype=np.uint8),
        "patch_ids": np.zeros((4, 16, 16), dtype=np.int16),
        "endpoint_patches": np.array(
            [[0, 255], [1, 2], [17, 34], [33, 34]], dtype=np.int16
        ),
    }
    probabilities = np.zeros((4, 64, 64), dtype=np.float32)
    probabilities[:, 4:60, 30:34] = 0.99
    for mode in (
        "mask_assignment_identity",
        "grid_mask_graph",
        "shuffled_topology",
        "mask_topo_recheck",
    ):
        artifacts, diagnostics = build_artifacts(
            arrays, probabilities, 0.9, 0, 16, mode, 123
        )
        assert artifacts["assignment"].shape == (4, 16, 16)
        assert artifacts["adjacency"].shape == (4, 16, 16)
        assert diagnostics["all_tokens_nonempty"]
    for token_count in (8, 32, 64):
        artifacts, diagnostics = build_artifacts(
            arrays,
            probabilities,
            0.9,
            0,
            token_count,
            "mask_topo_recheck",
            123,
        )
        assert artifacts["assignment"].shape == (4, 16, 16)
        assert artifacts["adjacency"].shape == (
            4,
            token_count,
            token_count,
        )
        assert diagnostics["all_tokens_nonempty"]
    grid = fixed_artifacts(arrays, 16, "grid")[0]
    dataset = MaskFeatureDataset(arrays, grid, probabilities, False)
    batch = dataset[0]
    model = FourChannelTopoCoarsen(16, 16)
    logits = model(
        batch["image"].unsqueeze(0),
        batch["assignment"].unsqueeze(0),
        batch["adjacency"].unsqueeze(0),
        batch["reachability"].unsqueeze(0),
    )
    assert logits.shape == (1,)
    print("CRACKFOREST_MECHANISM_ABLATION_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.mask_run_dir is None or args.output is None:
        raise ValueError("--mask-run-dir and --output are required.")
    if args.crop_size != 64 or args.token_count != 16:
        raise ValueError("This ablation requires crop-size=64 and 16 tokens.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
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
    mask_run_dir = args.mask_run_dir.resolve()
    prior = json.loads(
        (mask_run_dir / "result.json").read_text(encoding="utf-8")
    )
    if int(prior["optimization_seed"]) != args.seed:
        raise ValueError("Mask checkpoint seed does not match optimization seed.")
    threshold = float(prior["selected_on_dev"]["mask_threshold"])
    closing_iterations = int(
        prior["selected_on_dev"]["closing_iterations"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_model = CrackUNet().to(device)
    checkpoint = torch.load(
        mask_run_dir / f"mask_predictor_seed{args.seed}.pt",
        map_location=device,
        weights_only=False,
    )
    mask_model.load_state_dict(checkpoint["model"])
    probabilities = {
        split: predict_masks(
            mask_model, current["images"], args.batch_size, device
        )
        for split, current in arrays.items()
    }
    np.savez_compressed(
        output / "mask_probabilities.npz",
        **{
            split: value.astype(np.float16)
            for split, value in probabilities.items()
        },
    )

    grid_artifacts = {
        split: fixed_artifacts(current, args.token_count, "grid")[0]
        for split, current in arrays.items()
    }
    artifact_bank: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    diagnostics: dict[str, Any] = {}
    for mode in (
        "mask_assignment_identity",
        "grid_mask_graph",
        "shuffled_topology",
        "mask_topo_recheck",
    ):
        artifact_bank[mode] = {}
        diagnostics[mode] = {}
        for split_index, split in enumerate(("train", "dev", "test")):
            artifact_bank[mode][split], diagnostics[mode][split] = (
                build_artifacts(
                    arrays[split],
                    probabilities[split],
                    threshold,
                    closing_iterations,
                    args.token_count,
                    mode,
                    args.shuffle_seed
                    + 100_000 * split_index,
                )
            )
    summaries = {}
    histories = []
    predictions = {}
    truth_reference: np.ndarray | None = None
    for mode in args.modes:
        print(f"ABLATION_START mode={mode}", flush=True)
        if mode in {"grid_recheck", "mask_feature_grid"}:
            current_artifacts = grid_artifacts
        else:
            current_artifacts = artifact_bank[mode]
        if mode == "mask_feature_grid":
            summary, history, prediction, truth = train_mask_feature_grid(
                args,
                MaskFeatureDataset(
                    arrays["train"],
                    current_artifacts["train"],
                    probabilities["train"],
                    True,
                ),
                MaskFeatureDataset(
                    arrays["dev"],
                    current_artifacts["dev"],
                    probabilities["dev"],
                    False,
                ),
                MaskFeatureDataset(
                    arrays["test"],
                    current_artifacts["test"],
                    probabilities["test"],
                    False,
                ),
                output,
                device,
            )
        else:
            summary, history, prediction, truth = train_classifier(
                mode,
                args,
                RealDataset(
                    arrays["train"], current_artifacts["train"], augment=True
                ),
                RealDataset(arrays["dev"], current_artifacts["dev"]),
                RealDataset(arrays["test"], current_artifacts["test"]),
                output,
                device,
            )
        if truth_reference is None:
            truth_reference = truth
        elif not np.array_equal(truth_reference, truth):
            raise RuntimeError("Held-out truth changed between ablations.")
        summaries[mode] = summary
        histories.extend(history)
        predictions[mode] = prediction
        print(
            f"ABLATION_DONE mode={mode} "
            f"test_bal={summary['test_metrics']['balanced_accuracy']:.4f}",
            flush=True,
        )
    if truth_reference is None:
        raise RuntimeError("No ablation modes were run.")
    write_csv(output / "history.csv", histories)
    np.savez_compressed(
        output / "heldout_predictions.npz",
        truth=truth_reference,
        **predictions,
    )
    result = {
        "experiment_id": "crackforest_masktopo_mechanism_ablation",
        "status": "completed",
        "optimization_seed": args.seed,
        "data_seed": args.data_seed,
        "mask_checkpoint": str(mask_run_dir),
        "selected_on_prior_dev": {
            "mask_threshold": threshold,
            "closing_iterations": closing_iterations,
        },
        "modes": list(args.modes),
        "model_results": summaries,
        "artifact_diagnostics": diagnostics,
        "test_was_previously_observed": True,
        "interpretation_scope": "mechanism diagnostic, not confirmatory",
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

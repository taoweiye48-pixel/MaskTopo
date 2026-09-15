from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import deepcrack_external_gate
import fives_external_gate
from crackforest_mask_topo import CrackUNet, predict_masks
from crackforest_mechanism_ablation import build_artifacts
from crackforest_real_gate import (
    RealDataset,
    evaluate_classifier,
    train_classifier,
    write_csv,
)
from topocoarsen_oracle import TopoCoarsenModel


MODES = (
    "mask_topo_branch_none",
    "mask_topo_a_only",
    "mask_topo_r_only",
)
BRANCH_CONTROLS = {
    "mask_topo_branch_none": (False, False),
    "mask_topo_a_only": (True, False),
    "mask_topo_r_only": (False, True),
    "mask_topo_external": (True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parameter-matched A/R branch ablation for MaskTopo."
    )
    parser.add_argument("--dataset", choices=("fives", "deepcrack"))
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--prior-run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def configure_graph_branches(
    model: TopoCoarsenModel, use_local: bool, use_reachability: bool
) -> None:
    for graph_layer in model.head.graph_layers:
        graph_layer.use_local = use_local
        graph_layer.use_reachability = use_reachability


def frozen_namespace(
    dataset: str,
    seed: int,
    output: Path,
) -> argparse.Namespace:
    common = {
        "output": output,
        "train_size": 2400,
        "dev_size": 600,
        "test_size": 1200,
        "candidate_multiplier": 3,
        "crop_size": 64,
        "output_size": 64,
        "min_component_pixels": 12,
        "epochs": 22,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dim": 64,
        "num_workers": 0,
        "seed": seed,
        "data_seed": 20260810,
        "shuffle_seed": 20260731,
    }
    if dataset == "fives":
        common.update(
            {
                "dataset": "fives",
                "dataset_root": Path(
                    "real_data/FIVES/dataset/preprocessed512_v2"
                ),
                "cache_dir": Path("fives_cache"),
                "token_count": 8,
                "split_seed": 20260731,
            }
        )
    else:
        common.update(
            {
                "dataset": "deepcrack",
                "dataset_root": Path(
                    "real_data/DeepCrack/dataset/extracted"
                ),
                "cache_dir": Path("deepcrack_cache"),
                "token_count": 16,
                "split_seed": 20260730,
            }
        )
    return argparse.Namespace(**common)


def load_dataset(
    args: argparse.Namespace,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, list[str]],
    dict[str, tuple[np.ndarray, np.ndarray]],
    str,
]:
    if args.dataset == "fives":
        return fives_external_gate.load_all_arrays(args)
    return deepcrack_external_gate.load_all_arrays(args)


def run_self_test() -> None:
    torch.manual_seed(1234)
    reference = TopoCoarsenModel("mask_topo_external", 16, 4)
    state = reference.state_dict()
    parameter_count = sum(parameter.numel() for parameter in reference.parameters())
    models: dict[str, TopoCoarsenModel] = {}
    for name, controls in BRANCH_CONTROLS.items():
        model = TopoCoarsenModel(name, 16, 4)
        model.load_state_dict(state)
        configure_graph_branches(model, *controls)
        for graph_layer in model.head.graph_layers:
            graph_layer.local_gate.data.fill_(0.3)
            graph_layer.component_gate.data.fill_(-0.4)
        if sum(parameter.numel() for parameter in model.parameters()) != parameter_count:
            raise AssertionError("Branch control changed parameter count.")
        models[name] = model.eval()
    tokens = torch.randn(2, 4, 16)
    adjacency_a = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    adjacency_b = torch.ones_like(adjacency_a)
    reachability_a = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    reachability_b = torch.ones_like(reachability_a)
    layer_none = models["mask_topo_branch_none"].head.graph_layers[0]
    if not torch.equal(layer_none(tokens, adjacency_a, reachability_a), tokens):
        raise AssertionError("branch-none must be an exact identity forward.")
    layer_a = models["mask_topo_a_only"].head.graph_layers[0]
    if not torch.equal(
        layer_a(tokens, adjacency_a, reachability_a),
        layer_a(tokens, adjacency_a, reachability_b),
    ):
        raise AssertionError("A-only unexpectedly depends on R.")
    layer_r = models["mask_topo_r_only"].head.graph_layers[0]
    if not torch.equal(
        layer_r(tokens, adjacency_a, reachability_a),
        layer_r(tokens, adjacency_b, reachability_a),
    ):
        raise AssertionError("R-only unexpectedly depends on A.")
    print("AR_BRANCH_ABLATION_SELF_TEST_PASS", flush=True)


def main() -> None:
    cli = parse_args()
    if cli.self_test:
        run_self_test()
        return
    if cli.dataset is None or cli.prior_run_dir is None or cli.output is None:
        raise ValueError("--dataset, --prior-run-dir and --output are required.")
    if cli.seed not in {20260810, 20260811, 20260812}:
        raise ValueError("Seed is outside the frozen TB041 set.")

    root = Path(__file__).resolve().parent
    prior_run = cli.prior_run_dir.resolve()
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args = frozen_namespace(cli.dataset, cli.seed, output)
    args.dataset_root = (root / args.dataset_root).resolve()
    args.cache_dir = (root / args.cache_dir).resolve()

    prior_result_path = prior_run / "result.json"
    prior_prediction_path = prior_run / "heldout_predictions.npz"
    mask_checkpoint_path = prior_run / f"mask_predictor_seed{cli.seed}.pt"
    full_checkpoint_path = prior_run / f"mask_topo_external_seed{cli.seed}.pt"
    prior_result = json.loads(prior_result_path.read_text(encoding="utf-8"))
    if prior_result.get("status") != "completed":
        raise RuntimeError("Prior run is not completed.")
    if int(prior_result["optimization_seed"]) != cli.seed:
        raise RuntimeError("Prior optimization seed mismatch.")
    if int(prior_result["data_seed"]) != args.data_seed:
        raise RuntimeError("Prior data seed mismatch.")
    if int(prior_result["split_seed"]) != args.split_seed:
        raise RuntimeError("Prior split seed mismatch.")
    selected = prior_result["selected_on_dev"]
    threshold = float(selected["mask_threshold"])
    closing = int(selected["closing_iterations"])

    print(
        f"TB041_DATA_LOAD dataset={cli.dataset} seed={cli.seed}", flush=True
    )
    arrays, manifest, _, fingerprint = load_dataset(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TB041 frozen execution requires CUDA.")

    mask_model = CrackUNet().to(device)
    mask_checkpoint = torch.load(
        mask_checkpoint_path, map_location=device, weights_only=False
    )
    mask_model.load_state_dict(mask_checkpoint["model"])
    probabilities = {
        split: predict_masks(
            mask_model, arrays[split]["images"], args.batch_size, device
        )
        for split in ("train", "dev", "test")
    }
    probability_cache_checks: dict[str, bool] = {}
    probability_hashes: dict[str, dict[str, str]] = {}
    with np.load(prior_run / "mask_probabilities.npz") as cached:
        for split, probability in probabilities.items():
            cast = probability.astype(np.float16)
            probability_cache_checks[split] = bool(
                np.array_equal(cast, cached[split])
            )
            probability_hashes[split] = {
                "regenerated_float32_sha256": sha256_array(probability),
                "regenerated_float16_sha256": sha256_array(cast),
                "prior_cached_float16_sha256": sha256_array(cached[split]),
            }
    if not all(probability_cache_checks.values()):
        raise RuntimeError("Regenerated mask probabilities do not match prior cache.")

    artifacts: dict[str, dict[str, np.ndarray]] = {}
    artifact_diagnostics: dict[str, Any] = {}
    for split_index, split in enumerate(("train", "dev", "test")):
        artifacts[split], artifact_diagnostics[split] = build_artifacts(
            arrays[split],
            probabilities[split],
            threshold,
            closing,
            args.token_count,
            "mask_topo_recheck",
            args.shuffle_seed + 100_000 * split_index,
        )

    test_dataset = RealDataset(arrays["test"], artifacts["test"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    full_model = TopoCoarsenModel(
        "mask_topo_external", args.dim, args.token_count
    ).to(device)
    full_checkpoint = torch.load(
        full_checkpoint_path, map_location=device, weights_only=False
    )
    full_model.load_state_dict(full_checkpoint["model"])
    full_metrics, full_prediction, full_truth = evaluate_classifier(
        full_model,
        "mask_topo_external",
        test_loader,
        device,
        return_predictions=True,
    )
    if full_prediction is None or full_truth is None:
        raise AssertionError("Missing reproduced full-model predictions.")
    with np.load(prior_prediction_path) as prior_archive:
        prior_truth = prior_archive["truth"].astype(np.uint8)
        prior_sources = prior_archive["source_id"].astype(str)
        prior_full_prediction = prior_archive["mask_topo_external"].astype(
            np.uint8
        )
    source_ids = arrays["test"]["source_id"].astype(str)
    reproduction_checks = {
        "truth_equal_prior": bool(np.array_equal(full_truth, prior_truth)),
        "source_id_equal_prior": bool(
            np.array_equal(source_ids, prior_sources)
        ),
        "full_prediction_equal_prior": bool(
            np.array_equal(full_prediction, prior_full_prediction)
        ),
        "full_metric_equal_prior": bool(
            np.isclose(
                full_metrics["balanced_accuracy"],
                prior_result["model_results"]["mask_topo_external"][
                    "test_metrics"
                ]["balanced_accuracy"],
            )
        ),
    }
    if not all(reproduction_checks.values()):
        raise RuntimeError(
            f"Historical full-model reproduction failed: {reproduction_checks}"
        )
    print(
        f"TB041_REPRO_PASS dataset={cli.dataset} seed={cli.seed} "
        f"test_bal={full_metrics['balanced_accuracy']:.4f}",
        flush=True,
    )

    if cli.preflight:
        preflight = {
            "experiment_id": "TB-B-260810-041",
            "status": "preflight_pass",
            "dataset": cli.dataset,
            "optimization_seed": cli.seed,
            "parameter_count_historical_full": sum(
                parameter.numel() for parameter in full_model.parameters()
            ),
            "probability_float16_cache_checks": probability_cache_checks,
            "reproduction_checks": reproduction_checks,
            "historical_full_test_metrics": full_metrics,
            "artifact_diagnostics": artifact_diagnostics,
            "no_new_model_training_performed": True,
        }
        (output / "PREFLIGHT.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
        return

    datasets = {
        split: RealDataset(
            arrays[split], artifacts[split], augment=split == "train"
        )
        for split in ("train", "dev", "test")
    }
    histories: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for mode in MODES:
        print(
            f"TB041_TRAIN_START dataset={cli.dataset} seed={cli.seed} "
            f"mode={mode}",
            flush=True,
        )
        summary, history, prediction, truth = train_classifier(
            mode,
            args,
            datasets["train"],
            datasets["dev"],
            datasets["test"],
            output,
            device,
        )
        if not np.array_equal(truth, prior_truth):
            raise RuntimeError(f"Truth changed while training {mode}.")
        summaries[mode] = summary
        histories.extend(history)
        predictions[mode] = prediction
        print(
            f"TB041_TRAIN_DONE dataset={cli.dataset} seed={cli.seed} "
            f"mode={mode} test_bal="
            f"{summary['test_metrics']['balanced_accuracy']:.4f}",
            flush=True,
        )

    parameter_counts = {
        "mask_topo_external": int(
            prior_result["model_results"]["mask_topo_external"]["parameters"]
        ),
        **{
            mode: int(summary["parameters"])
            for mode, summary in summaries.items()
        },
    }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"Parameter counts are not matched: {parameter_counts}")

    write_csv(output / "history.csv", histories)
    np.savez_compressed(
        output / "test_predictions.npz",
        truth=prior_truth,
        source_id=prior_sources,
        mask_topo_external=prior_full_prediction,
        **predictions,
    )
    result = {
        "experiment_id": "TB-B-260810-041",
        "status": "completed",
        "dataset": cli.dataset,
        "optimization_seed": cli.seed,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "token_count": args.token_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model_dim": args.dim,
        "dataset_fingerprint": fingerprint,
        "source_counts": {
            split: len(values) for split, values in manifest.items()
        },
        "selected_on_prior_dev": {
            "mask_threshold": threshold,
            "closing_iterations": closing,
        },
        "branch_controls": {
            name: {
                "use_local_adjacency_A": controls[0],
                "use_component_reachability_R": controls[1],
            }
            for name, controls in BRANCH_CONTROLS.items()
        },
        "parameter_counts": parameter_counts,
        "historical_full_model": {
            "test_metrics": full_metrics,
            "prediction_sha256": sha256_array(prior_full_prediction),
        },
        "new_model_results": summaries,
        "artifact_diagnostics": artifact_diagnostics,
        "probability_float16_cache_checks": probability_cache_checks,
        "probability_hashes": probability_hashes,
        "reproduction_checks": reproduction_checks,
        "prior_artifacts": {
            "run_dir": str(prior_run),
            "result_json_sha256": sha256_file(prior_result_path),
            "predictions_sha256": sha256_file(prior_prediction_path),
            "mask_checkpoint_sha256": sha256_file(mask_checkpoint_path),
            "full_checkpoint_sha256": sha256_file(full_checkpoint_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "test_previously_observed": True,
        "interpretation_scope": "retrospective mechanism diagnostic",
        "protocol": str(
            (
                root
                / "实验记录"
                / "TB-B-260810-041_A_R分支独立消融协议.md"
            ).resolve()
        ),
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "TB041_RUN_COMPLETE",
                "dataset": cli.dataset,
                "seed": cli.seed,
                "test_balanced_accuracy": {
                    "mask_topo_external": full_metrics["balanced_accuracy"],
                    **{
                        mode: summaries[mode]["test_metrics"][
                            "balanced_accuracy"
                        ]
                        for mode in MODES
                    },
                },
                "parameter_counts": parameter_counts,
                "reproduction_checks": reproduction_checks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

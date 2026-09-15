from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from crackforest_mask_topo import CrackUNet
from crackforest_mechanism_ablation import build_artifacts
from robustness_stress import (
    array_sha256,
    file_sha256,
    load_model,
    load_test_and_masks,
    mask_predictor_path,
    ph_run_root,
    predict_classifier,
    predict_masks,
    prior_root,
)
from summarize_crackforest_mechanism_ablation import balanced_accuracy
from summarize_strong_reducer_baselines import cluster_bootstrap


EXPERIMENT_ID = "TB-B-260802-026"
SEEDS = (20260810, 20260811, 20260812)
TOKEN_COUNT = 16
BOOTSTRAP_REPETITIONS = 20_000
STATISTICS_SEED = 20260826
PH_MODELS = ("ph_only", "ph_guided")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepCrack K16 float16-aligned MaskTopo counterfactual."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("results_deepcrack_float16_alignment")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def score_summary(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    values = np.asarray(
        [balanced_accuracy(truth, row) for row in predictions], dtype=np.float64
    )
    return {
        "balanced_accuracy_by_seed": values.tolist(),
        "mean_balanced_accuracy": float(values.mean()),
        "sample_std_balanced_accuracy": float(values.std(ddof=1)),
    }


def artifact_change_summary(
    native: dict[str, np.ndarray], aligned: dict[str, np.ndarray]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in ("assignment", "adjacency", "reachability"):
        first = native[name]
        second = aligned[name]
        if first.shape != second.shape:
            raise RuntimeError(f"Artifact shape changed for {name}: {first.shape} != {second.shape}")
        changed = first != second
        sample_axes = tuple(range(1, changed.ndim))
        changed_samples = np.any(changed, axis=sample_axes)
        output[name] = {
            "element_change_count": int(np.count_nonzero(changed)),
            "sample_change_count": int(np.count_nonzero(changed_samples)),
            "sample_change_fraction": float(changed_samples.mean()),
            "native_sha256": array_sha256(first),
            "aligned_sha256": array_sha256(second),
        }
    return output


def run_self_test() -> None:
    native = {
        "assignment": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        "adjacency": np.zeros((2, 2, 2), dtype=np.uint8),
        "reachability": np.zeros((2, 2, 2), dtype=np.uint8),
    }
    aligned = {name: value.copy() for name, value in native.items()}
    aligned["assignment"][1, 0] = 0
    summary = artifact_change_summary(native, aligned)
    assert summary["assignment"]["sample_change_count"] == 1
    assert summary["adjacency"]["sample_change_count"] == 0
    truth = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    predictions = np.asarray([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    scores = score_summary(truth, predictions)
    assert np.isclose(scores["mean_balanced_accuracy"], 0.875)
    print("DEEPCRACK_FLOAT16_ALIGNMENT_SELF_TEST_PASS", flush=True)


def report_markdown(result: dict[str, Any]) -> str:
    aggregates = result["aggregates"]
    contrast = result["contrasts"]["aligned_masktopo_minus_ph_only"]
    ci = contrast["source_cluster_bootstrap_95_ci"]
    precision = result["contrasts"]["aligned_masktopo_minus_native_masktopo"]
    precision_ci = precision["source_cluster_bootstrap_95_ci"]
    lines = [
        "# DeepCrack K=16 float16 probability-path alignment",
        "",
        "This is a no-training retrospective endpoint-connectivity counterfactual. It does not use the P1-5 disease-classification result and does not reopen Massachusetts Roads test.",
        "",
        "| Seed | Native float32 MaskTopo | Aligned float16 MaskTopo | PH-only | PH-guided | MaskTopo prediction flips |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["runs"]:
        lines.append(
            f"| {row['seed']} | {100 * row['native_masktopo_balanced_accuracy']:.2f}% | "
            f"{100 * row['aligned_masktopo_balanced_accuracy']:.2f}% | "
            f"{100 * row['ph_only_balanced_accuracy']:.2f}% | "
            f"{100 * row['ph_guided_balanced_accuracy']:.2f}% | "
            f"{row['aligned_vs_native_prediction_flip_count']} |"
        )
    lines.extend(
        [
            "",
            "## Three-seed result",
            "",
            f"- Native float32 MaskTopo: {100 * aggregates['native_masktopo']['mean_balanced_accuracy']:.2f}±{100 * aggregates['native_masktopo']['sample_std_balanced_accuracy']:.2f}%.",
            f"- Aligned float16 MaskTopo: {100 * aggregates['aligned_masktopo']['mean_balanced_accuracy']:.2f}±{100 * aggregates['aligned_masktopo']['sample_std_balanced_accuracy']:.2f}%.",
            f"- PH-only: {100 * aggregates['ph_only']['mean_balanced_accuracy']:.2f}±{100 * aggregates['ph_only']['sample_std_balanced_accuracy']:.2f}%.",
            f"- PH-guided: {100 * aggregates['ph_guided']['mean_balanced_accuracy']:.2f}±{100 * aggregates['ph_guided']['sample_std_balanced_accuracy']:.2f}%.",
            f"- Aligned MaskTopo − PH-only: {100 * contrast['mean_gain']:+.2f} pp, source-image cluster bootstrap 95% CI [{100 * ci[0]:+.2f},{100 * ci[1]:+.2f}].",
            f"- Aligned MaskTopo − native MaskTopo: {100 * precision['mean_gain']:+.2f} pp, 95% CI [{100 * precision_ci[0]:+.2f},{100 * precision_ci[1]:+.2f}].",
            "",
            "## Verdict",
            "",
            f"`{result['verdict']}`",
            "",
            result["interpretation"],
            "",
            "The historical float32/float16 provenance difference remains disclosed. This counterfactual does not overwrite the native-path result; it tests whether precision explains the MaskTopo-versus-PH contrast.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    protocol = root / "DEEPCRACK_FLOAT16_ALIGNMENT_PROTOCOL.md"
    if not protocol.exists():
        raise FileNotFoundError(protocol)
    output.mkdir(parents=True)
    shutil.copyfile(protocol, output / "PROTOCOL.md")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen protocol requires CUDA to reproduce the original path.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    prediction_lists: dict[str, list[np.ndarray]] = {
        "native_masktopo": [],
        "aligned_masktopo": [],
        "ph_only": [],
        "ph_guided": [],
    }
    truth_reference: np.ndarray | None = None
    source_reference: np.ndarray | None = None
    runs: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        print(f"FLOAT16_ALIGNMENT_LOAD seed={seed}", flush=True)
        arrays, _ = load_test_and_masks(root, "deepcrack", seed)
        old_root = prior_root(root, "deepcrack", seed)
        cache_path = old_root / "mask_probabilities.npz"
        with np.load(cache_path) as archive:
            cached_raw = archive["test"]
            cached_dtype = str(cached_raw.dtype)
            cached_probability = cached_raw.astype(np.float32)
        if cached_dtype != "float16":
            raise RuntimeError(f"Expected float16 cache for seed {seed}, got {cached_dtype}.")

        predictor_path = mask_predictor_path(root, "deepcrack", seed)
        predictor_checkpoint = torch.load(
            predictor_path, map_location="cpu", weights_only=False
        )
        predictor = CrackUNet()
        predictor.load_state_dict(predictor_checkpoint["model"])
        predictor = predictor.to(device).eval()
        reproduced_probability = predict_masks(
            predictor, arrays["images"], args.batch_size, device
        )
        del predictor
        torch.cuda.empty_cache()
        probability_delta = np.abs(reproduced_probability - cached_probability)
        if float(probability_delta.max()) > 1e-3:
            raise RuntimeError(
                f"Seed {seed} predictor/cache reproduction exceeds 1e-3: "
                f"{float(probability_delta.max())}."
            )

        print(f"FLOAT16_ALIGNMENT_ARTIFACTS seed={seed}", flush=True)
        native_artifacts, native_diagnostics = build_artifacts(
            arrays,
            reproduced_probability,
            0.90,
            0,
            TOKEN_COUNT,
            "mask_topo_recheck",
            20260731,
        )
        aligned_artifacts, aligned_diagnostics = build_artifacts(
            arrays,
            cached_probability,
            0.90,
            0,
            TOKEN_COUNT,
            "mask_topo_recheck",
            20260731,
        )
        artifact_changes = artifact_change_summary(native_artifacts, aligned_artifacts)

        model, spec = load_model(root, "deepcrack", seed, "mask_topo")
        model = model.to(device).eval()
        with np.load(old_root / "heldout_predictions.npz") as archive:
            truth = archive["truth"].astype(np.uint8)
            sources = archive["source_id"].astype(str)
            saved_native_prediction = archive["mask_topo_external"].astype(np.uint8)
        print(f"FLOAT16_ALIGNMENT_FORWARD seed={seed}", flush=True)
        reproduced_native_prediction = predict_classifier(
            model,
            "mask_topo",
            arrays["images"],
            reproduced_probability,
            native_artifacts,
            None,
            args.batch_size,
            device,
        )
        native_mismatch = int(
            np.count_nonzero(reproduced_native_prediction != saved_native_prediction)
        )
        if native_mismatch:
            raise RuntimeError(
                f"Seed {seed} native MaskTopo prediction mismatch: {native_mismatch}."
            )
        aligned_prediction = predict_classifier(
            model,
            "mask_topo",
            arrays["images"],
            cached_probability,
            aligned_artifacts,
            None,
            args.batch_size,
            device,
        )
        del model
        torch.cuda.empty_cache()

        ph_root = ph_run_root(root, "deepcrack", seed)
        with np.load(ph_root / "heldout_predictions.npz") as archive:
            if not np.array_equal(truth, archive["truth"].astype(np.uint8)):
                raise RuntimeError(f"Seed {seed} PH truth mismatch.")
            if "source_id" in archive.files and not np.array_equal(
                sources, archive["source_id"].astype(str)
            ):
                raise RuntimeError(f"Seed {seed} PH source mismatch.")
            ph_predictions = {
                name: archive[name].astype(np.uint8) for name in PH_MODELS
            }
        old_result = json.loads((old_root / "result.json").read_text(encoding="utf-8"))
        ph_result = json.loads((ph_root / "result.json").read_text(encoding="utf-8"))
        recorded_native = old_result["model_results"]["mask_topo_external"][
            "test_metrics"
        ]["balanced_accuracy"]
        if not np.isclose(
            balanced_accuracy(truth, saved_native_prediction), recorded_native
        ):
            raise RuntimeError(f"Seed {seed} native recorded metric mismatch.")
        for name in PH_MODELS:
            recorded_ph = ph_result["model_results"][name]["test_metrics"][
                "balanced_accuracy"
            ]
            if not np.isclose(
                balanced_accuracy(truth, ph_predictions[name]), recorded_ph
            ):
                raise RuntimeError(f"Seed {seed} {name} recorded metric mismatch.")

        if truth_reference is None:
            truth_reference = truth
            source_reference = sources
        elif not np.array_equal(truth_reference, truth) or not np.array_equal(
            source_reference, sources
        ):
            raise RuntimeError("Frozen DeepCrack samples changed across seeds.")

        current_predictions = {
            "native_masktopo": saved_native_prediction,
            "aligned_masktopo": aligned_prediction,
            **ph_predictions,
        }
        for name, prediction in current_predictions.items():
            prediction_lists[name].append(prediction)
            source_rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "balanced_accuracy": balanced_accuracy(truth, prediction),
                }
            )
        flips = aligned_prediction != saved_native_prediction
        probability_threshold_flips = (
            (reproduced_probability >= 0.90) != (cached_probability >= 0.90)
        )
        run = {
            "seed": seed,
            "native_masktopo_balanced_accuracy": balanced_accuracy(
                truth, saved_native_prediction
            ),
            "aligned_masktopo_balanced_accuracy": balanced_accuracy(
                truth, aligned_prediction
            ),
            "ph_only_balanced_accuracy": balanced_accuracy(
                truth, ph_predictions["ph_only"]
            ),
            "ph_guided_balanced_accuracy": balanced_accuracy(
                truth, ph_predictions["ph_guided"]
            ),
            "aligned_vs_native_prediction_flip_count": int(np.count_nonzero(flips)),
            "aligned_vs_native_prediction_flip_source_count": int(
                np.unique(sources[flips]).size
            ),
            "probability_max_abs_difference": float(probability_delta.max()),
            "probability_mean_abs_difference": float(probability_delta.mean()),
            "threshold_pixel_flip_count": int(
                np.count_nonzero(probability_threshold_flips)
            ),
            "threshold_sample_change_count": int(
                np.count_nonzero(np.any(probability_threshold_flips, axis=(1, 2)))
            ),
            "artifact_changes": artifact_changes,
            "native_artifact_diagnostics": native_diagnostics,
            "aligned_artifact_diagnostics": aligned_diagnostics,
            "checkpoint_path": str(spec["checkpoint"].resolve()),
            "checkpoint_sha256": file_sha256(spec["checkpoint"]),
            "mask_predictor_path": str(predictor_path.resolve()),
            "mask_predictor_sha256": file_sha256(predictor_path),
            "float16_cache_path": str(cache_path.resolve()),
            "float16_cache_file_sha256": file_sha256(cache_path),
            "float16_cache_raw_array_sha256": array_sha256(cached_raw),
            "float16_quantized_values_as_float32_sha256": array_sha256(
                cached_probability
            ),
            "native_reproduced_probability_sha256": array_sha256(
                reproduced_probability
            ),
            "native_saved_prediction_mismatch_count": native_mismatch,
        }
        runs.append(run)
        np.savez_compressed(
            output / f"predictions_seed{seed}.npz",
            truth=truth,
            source_id=sources,
            native_masktopo=saved_native_prediction,
            aligned_masktopo=aligned_prediction,
            ph_only=ph_predictions["ph_only"],
            ph_guided=ph_predictions["ph_guided"],
        )

    if truth_reference is None or source_reference is None:
        raise RuntimeError("No DeepCrack runs loaded.")
    predictions = {
        name: np.stack(values) for name, values in prediction_lists.items()
    }
    aggregates = {
        name: score_summary(truth_reference, value)
        for name, value in predictions.items()
    }
    if aggregates["ph_only"]["mean_balanced_accuracy"] < aggregates["ph_guided"][
        "mean_balanced_accuracy"
    ]:
        raise RuntimeError("Frozen DeepCrack best-PH identity changed from PH-only.")

    contrast_specs = {
        "aligned_masktopo_minus_ph_only": ("aligned_masktopo", "ph_only"),
        "aligned_masktopo_minus_ph_guided": ("aligned_masktopo", "ph_guided"),
        "native_masktopo_minus_ph_only": ("native_masktopo", "ph_only"),
        "aligned_masktopo_minus_native_masktopo": (
            "aligned_masktopo",
            "native_masktopo",
        ),
    }
    contrasts = {
        name: cluster_bootstrap(
            truth_reference,
            source_reference,
            predictions[first],
            predictions[second],
            BOOTSTRAP_REPETITIONS,
            np.random.default_rng(STATISTICS_SEED + index),
        )
        for index, (name, (first, second)) in enumerate(contrast_specs.items())
    }
    primary = contrasts["aligned_masktopo_minus_ph_only"]
    lower = primary["source_cluster_bootstrap_95_ci"][0]
    if primary["mean_gain"] <= 0:
        verdict = "WITHDRAW_DEEPCRACK_PH_SUPERIORITY"
        interpretation = (
            "After float16 alignment, MaskTopo no longer exceeds PH-only. The original "
            "DeepCrack PH-superiority claim is withdrawn."
        )
    elif lower <= 0:
        verdict = "DOWNGRADE_DEEPCRACK_PH_CONTRAST_TO_INCONCLUSIVE"
        interpretation = (
            "After float16 alignment, the point estimate remains positive but the source-level "
            "interval crosses zero. DeepCrack cannot support a positive PH contrast."
        )
    else:
        verdict = "FLOAT16_ALIGNMENT_PASSES_PH_CONTRAST_REMAINS_POSITIVE"
        interpretation = (
            "After MaskTopo consumes the same float16 probability cache as PH, the source-level "
            "confidence interval remains above zero. Probability precision does not explain away "
            "the DeepCrack MaskTopo-versus-PH result."
        )

    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_verified",
        "confirmatory_status": "retrospective_precision_alignment_counterfactual",
        "dataset": "deepcrack",
        "task": "endpoint_connectivity",
        "token_count": TOKEN_COUNT,
        "optimization_seeds": list(SEEDS),
        "test_sample_count": int(truth_reference.size),
        "test_source_count": int(np.unique(source_reference).size),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "statistics_seed": STATISTICS_SEED,
        "aggregates": aggregates,
        "contrasts": contrasts,
        "runs": runs,
        "verdict": verdict,
        "interpretation": interpretation,
        "native_path_preserved_not_overwritten": True,
        "aligned_masktopo_consumes_same_float16_cache_values_as_ph": True,
        "training_performed": False,
        "test_was_previously_observed": True,
        "massachusetts_test_reopened": False,
        "disease_classification_p1_5_used": False,
        "protocol": str(protocol.resolve()),
        "protocol_sha256": file_sha256(protocol),
        "code": str(Path(__file__).resolve()),
        "code_sha256": file_sha256(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(report_markdown(result), encoding="utf-8")
    with (output / "source_data.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("seed", "model", "balanced_accuracy")
        )
        writer.writeheader()
        writer.writerows(source_rows)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": result["status"],
                "verdict": verdict,
                "aligned_masktopo_mean": aggregates["aligned_masktopo"][
                    "mean_balanced_accuracy"
                ],
                "ph_only_mean": aggregates["ph_only"]["mean_balanced_accuracy"],
                "aligned_minus_ph_only": primary,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

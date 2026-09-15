from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy
import torch
import torch.nn as nn
from scipy.ndimage import binary_closing, binary_erosion

from crackforest_mask_topo import CrackUNet, build_pixel_masks
from crackforest_mechanism_ablation import build_artifacts
from crackforest_real_gate import load_sources as load_crackforest_sources
from crackforest_real_gate import source_split as crackforest_source_split
from deepcrack_external_gate import load_all_arrays
from edge_topocoarsen_connector import balanced_metrics
from ph_token_baselines import make_model as make_ph_model
from ph_token_baselines import ph_descriptors
from strong_reducer_baselines import REDUCERS, load_arrays, make_model as make_reducer
from topocoarsen_oracle import TopoCoarsenModel


SEED_CHOICES = (20260810, 20260811, 20260812)
TOKEN_COUNT = 16
PH_MODELS = ("ph_only", "ph_guided")
MODELS = ("mask_topo", *REDUCERS, *PH_MODELS)
MASK_SENSITIVE = {"mask_topo", "mask_guided_queries", *PH_MODELS}
STRESS_CONDITIONS = (
    "clean",
    "mask_break_erosion1",
    "mask_false_link_closing2",
    "low_contrast_0.35",
    "cross_domain_noise",
)
SENSITIVITY_CONDITIONS = {
    "threshold_0.80_c0": (0.80, 0),
    "threshold_0.85_c0": (0.85, 0),
    "threshold_0.90_c0": (0.90, 0),
    "threshold_0.95_c0": (0.95, 0),
    "threshold_0.90_c1": (0.90, 1),
    "threshold_0.90_c2": (0.90, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TB-B-260802-024 robustness job.")
    parser.add_argument("--dataset", choices=("crackforest", "deepcrack"), required=True)
    parser.add_argument("--seed", type=int, choices=SEED_CHOICES, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def loader_args(root: Path, dataset: str, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset,
        cache_dir=root / ("real_cache" if dataset == "crackforest" else "deepcrack_cache"),
        dataset_root=root / "real_data/DeepCrack/dataset/extracted",
        train_size=None,
        dev_size=None,
        test_size=None,
        candidate_multiplier=3,
        crop_size=64,
        output_size=64,
        min_component_pixels=12,
        # Dataset crops are frozen once with data seed 20260810 and reused by
        # all three optimization seeds.
        data_seed=20260810,
        split_seed=20260730,
    )


def load_test_and_masks(
    root: Path, dataset: str, seed: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if dataset == "deepcrack":
        args = loader_args(root, dataset, seed)
        args.train_size, args.dev_size, args.test_size = 2400, 600, 1200
        arrays, _, sources, _ = load_all_arrays(args)
        test = arrays["test"]
        source_masks = {key: value[1] for key, value in sources.items()}
    else:
        test = load_arrays(loader_args(root, dataset, seed))["test"]
        dataset_root = root / "real_data/CrackForest-dataset"
        test_sources = crackforest_source_split(dataset_root, 20260730)["test"]
        sources = load_crackforest_sources(dataset_root, test_sources)
        source_masks = {key: value[1] for key, value in sources.items()}
    masks = build_pixel_masks(test, source_masks)
    if masks.shape != (test["images"].shape[0], 64, 64):
        raise RuntimeError("Ground-truth pixel masks do not align with test inputs.")
    return test, masks


def prior_root(root: Path, dataset: str, seed: int) -> Path:
    return (
        root / f"results_crackforest_mechanism_seed{seed}"
        if dataset == "crackforest"
        else root / f"results_deepcrack_seed{seed}"
    )


def mask_predictor_path(root: Path, dataset: str, seed: int) -> Path:
    return (
        root / f"results_crackforest_mask_topo_v2_seed{seed}/mask_predictor_seed{seed}.pt"
        if dataset == "crackforest"
        else root / f"results_deepcrack_seed{seed}/mask_predictor_seed{seed}.pt"
    )


def ph_run_root(root: Path, dataset: str, seed: int) -> Path:
    retry = "_retry1" if dataset == "crackforest" and seed == 20260810 else ""
    return root / f"results_ph_{dataset}_k16_seed{seed}{retry}"


def model_spec(root: Path, dataset: str, seed: int, model_name: str) -> dict[str, Any]:
    if model_name == "mask_topo":
        run_root = prior_root(root, dataset, seed)
        key = "mask_topo_recheck" if dataset == "crackforest" else "mask_topo_external"
        checkpoint = run_root / f"{key}_seed{seed}.pt"
    elif model_name in REDUCERS:
        run_root = root / f"results_strong_reducer_{dataset}_seed{seed}"
        key = model_name
        checkpoint = run_root / f"{model_name}_seed{seed}.pt"
    elif model_name in PH_MODELS:
        run_root = ph_run_root(root, dataset, seed)
        key = model_name
        checkpoint = run_root / f"{model_name}_seed{seed}.pt"
    else:
        raise ValueError(model_name)
    return {
        "run_root": run_root,
        "prediction_key": key,
        "checkpoint": checkpoint,
    }


def load_model(
    root: Path, dataset: str, seed: int, model_name: str
) -> tuple[nn.Module, dict[str, Any]]:
    spec = model_spec(root, dataset, seed, model_name)
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    if model_name == "mask_topo":
        model = TopoCoarsenModel(str(checkpoint["model_name"]), 64, TOKEN_COUNT)
    elif model_name in REDUCERS:
        model = make_reducer(model_name, 64, TOKEN_COUNT)
    else:
        model = make_ph_model(model_name, 64, TOKEN_COUNT)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, spec


def photometric_images(
    images: np.ndarray, dataset: str
) -> dict[str, np.ndarray]:
    clean = images.copy()
    value = clean[:, 0].astype(np.float32) / 255.0
    low = clean.copy()
    low[:, 0] = np.rint(
        255.0 * np.clip(0.5 + 0.35 * (value - 0.5), 0.0, 1.0)
    ).astype(np.uint8)
    rng = np.random.default_rng(20268024 + (0 if dataset == "crackforest" else 1000))
    noise_value = np.clip(
        0.85 * value + 0.075 + rng.normal(0.0, 0.08, size=value.shape),
        0.0,
        1.0,
    )
    noisy = clean.copy()
    noisy[:, 0] = np.rint(255.0 * noise_value).astype(np.uint8)
    if not np.array_equal(low[:, 1:], clean[:, 1:]) or not np.array_equal(
        noisy[:, 1:], clean[:, 1:]
    ):
        raise AssertionError("Endpoint marker channels changed under photometric stress.")
    return {
        "clean": clean,
        "low_contrast_0.35": low,
        "cross_domain_noise": noisy,
    }


@torch.inference_mode()
def predict_masks(
    model: CrackUNet,
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, images.shape[0], batch_size):
        image = torch.from_numpy(
            images[start : start + batch_size, 0].astype(np.float32) / 255.0
        ).unsqueeze(1).to(device)
        parts.append(torch.sigmoid(model(image)).cpu().numpy().astype(np.float32))
    return np.concatenate(parts)


def mask_corruptions(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    binary = probabilities >= 0.90
    structure = np.ones((3, 3), dtype=bool)
    broken = probabilities.copy()
    false_link = probabilities.copy()
    for index in range(probabilities.shape[0]):
        eroded = binary_erosion(binary[index], structure=structure, iterations=1)
        removed = binary[index] & ~eroded
        broken[index, removed] = 0.0
        closed = binary_closing(binary[index], structure=structure, iterations=2)
        added = closed & ~binary[index]
        false_link[index, added] = 1.0
    return {
        "mask_break_erosion1": broken,
        "mask_false_link_closing2": false_link,
    }


def pixel_mask_metrics(
    truth: np.ndarray, probability: np.ndarray, clean: np.ndarray
) -> dict[str, float]:
    target = truth.astype(bool)
    prediction = probability >= 0.90
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "pixel_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "pixel_precision": precision,
        "pixel_recall": recall,
        "foreground_fraction": float(prediction.mean()),
        "binary_changed_fraction_vs_clean": float(
            np.mean(prediction != (clean >= 0.90))
        ),
        "probability_mean_absolute_change_vs_clean": float(
            np.mean(np.abs(probability - clean))
        ),
    }


def ph_token_diagnostics(descriptors: np.ndarray) -> dict[str, float]:
    valid = descriptors[..., 6] > 0.5
    h0 = descriptors[..., 3] > 0.5
    h1 = descriptors[..., 4] > 0.5
    return {
        "valid_pair_mean": float(valid.sum(axis=1).mean()),
        "padding_rate": float(1.0 - valid.mean()),
        "selected_h0_mean": float((valid & h0).sum(axis=1).mean()),
        "selected_h1_mean": float((valid & h1).sum(axis=1).mean()),
    }


def compute_descriptors(probabilities: np.ndarray) -> np.ndarray:
    output = np.empty((probabilities.shape[0], 64, 11), dtype=np.float32)
    for index in range(probabilities.shape[0]):
        output[index] = ph_descriptors(probabilities[index], 64)[0]
    return output


@torch.inference_mode()
def predict_classifier(
    model: nn.Module,
    model_name: str,
    images: np.ndarray,
    probabilities: np.ndarray,
    artifacts: dict[str, np.ndarray] | None,
    descriptors: np.ndarray | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, images.shape[0], batch_size):
        stop = min(start + batch_size, images.shape[0])
        image = torch.from_numpy(
            images[start:stop].astype(np.float32) / 255.0
        ).to(device)
        if model_name == "mask_topo":
            if artifacts is None:
                raise ValueError("MaskTopo artifacts are missing.")
            inputs = (
                image,
                torch.from_numpy(artifacts["assignment"][start:stop].astype(np.int64)).to(device),
                torch.from_numpy(artifacts["adjacency"][start:stop].astype(np.int64)).to(device),
                torch.from_numpy(artifacts["reachability"][start:stop].astype(np.int64)).to(device),
            )
        else:
            mask = torch.from_numpy(probabilities[start:stop]).unsqueeze(1).to(device)
            if model_name in PH_MODELS:
                if descriptors is None:
                    raise ValueError("PH descriptors are missing.")
                descriptor = torch.from_numpy(
                    descriptors[start:stop, :TOKEN_COUNT]
                ).to(device)
                inputs = (image, mask, descriptor)
            else:
                inputs = (image, mask)
        logits = model(*inputs)
        parts.append((torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.uint8))
    return np.concatenate(parts)


def load_expected_prediction(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(spec["run_root"] / "heldout_predictions.npz") as archive:
        return (
            archive["truth"].astype(np.uint8),
            archive[spec["prediction_key"]].astype(np.uint8),
        )


def run_self_test() -> None:
    images = np.zeros((2, 3, 64, 64), dtype=np.uint8)
    images[:, 1, 2, 3] = 255
    images[:, 2, 5, 7] = 255
    perturbed = photometric_images(images, "crackforest")
    assert np.array_equal(perturbed["low_contrast_0.35"][:, 1:], images[:, 1:])
    probability = np.zeros((2, 64, 64), dtype=np.float32)
    probability[:, 20:44, 31:33] = 0.95
    corrupt = mask_corruptions(probability)
    assert np.count_nonzero(corrupt["mask_break_erosion1"] >= 0.90) < np.count_nonzero(
        probability >= 0.90
    )
    metrics = pixel_mask_metrics(probability >= 0.90, probability, probability)
    assert np.isclose(metrics["pixel_f1"], 1.0)
    assert metrics["binary_changed_fraction_vs_clean"] == 0.0
    print("ROBUSTNESS_STRESS_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen robustness protocol requires CUDA.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    print(f"ROBUSTNESS_LOAD dataset={args.dataset} seed={args.seed}", flush=True)
    arrays, ground_truth_masks = load_test_and_masks(root, args.dataset, args.seed)
    current_prior = prior_root(root, args.dataset, args.seed)
    with np.load(current_prior / "mask_probabilities.npz") as archive:
        cached_clean_probability = archive["test"].astype(np.float32)
    image_conditions = photometric_images(arrays["images"], args.dataset)

    predictor_checkpoint_path = mask_predictor_path(root, args.dataset, args.seed)
    predictor_checkpoint = torch.load(
        predictor_checkpoint_path, map_location="cpu", weights_only=False
    )
    predictor = CrackUNet()
    predictor.load_state_dict(predictor_checkpoint["model"])
    predictor = predictor.to(device).eval()
    reproduced_probability = predict_masks(
        predictor, image_conditions["clean"], args.batch_size, device
    )
    probability_difference = np.abs(
        reproduced_probability - cached_clean_probability
    )
    if float(probability_difference.max()) > 1e-3:
        raise RuntimeError(
            "Mask predictor failed frozen clean-probability reproduction: "
            f"{float(probability_difference.max())}."
        )
    # MaskTopo was trained/evaluated with the in-memory float32 predictor output
    # before the probability cache was quantized to float16. Reducer/PH runs
    # consumed the saved float16 cache. Preserve both native frozen paths.
    stress_probabilities = {
        "clean": cached_clean_probability,
        **mask_corruptions(cached_clean_probability),
        "low_contrast_0.35": predict_masks(
            predictor, image_conditions["low_contrast_0.35"], args.batch_size, device
        ),
        "cross_domain_noise": predict_masks(
            predictor, image_conditions["cross_domain_noise"], args.batch_size, device
        ),
    }
    masktopo_probabilities = {
        "clean": reproduced_probability,
        **mask_corruptions(reproduced_probability),
        "low_contrast_0.35": stress_probabilities["low_contrast_0.35"],
        "cross_domain_noise": stress_probabilities["cross_domain_noise"],
    }
    del predictor
    torch.cuda.empty_cache()

    ph_cache_path = root / f"ph_cache/{args.dataset}_seed{args.seed}_k64.npz"
    with np.load(ph_cache_path) as archive:
        clean_descriptors = archive["test"].astype(np.float32)
    descriptors = {"clean": clean_descriptors}
    for condition in STRESS_CONDITIONS[1:]:
        print(
            f"ROBUSTNESS_PH dataset={args.dataset} seed={args.seed} condition={condition}",
            flush=True,
        )
        descriptors[condition] = compute_descriptors(stress_probabilities[condition])

    stress_artifacts: dict[str, dict[str, np.ndarray]] = {}
    stress_artifact_diagnostics: dict[str, Any] = {}
    for condition in STRESS_CONDITIONS:
        print(
            f"ROBUSTNESS_ARTIFACT dataset={args.dataset} seed={args.seed} condition={condition}",
            flush=True,
        )
        stress_artifacts[condition], stress_artifact_diagnostics[condition] = build_artifacts(
            arrays,
            masktopo_probabilities[condition],
            0.90,
            0,
            TOKEN_COUNT,
            "mask_topo_recheck",
            20260731,
        )

    sensitivity_artifacts: dict[str, dict[str, np.ndarray]] = {}
    sensitivity_diagnostics: dict[str, Any] = {}
    for condition, (threshold, closing) in SENSITIVITY_CONDITIONS.items():
        if condition == "threshold_0.90_c0":
            sensitivity_artifacts[condition] = stress_artifacts["clean"]
            sensitivity_diagnostics[condition] = stress_artifact_diagnostics["clean"]
            continue
        print(
            f"ROBUSTNESS_SENSITIVITY dataset={args.dataset} seed={args.seed} condition={condition}",
            flush=True,
        )
        sensitivity_artifacts[condition], sensitivity_diagnostics[condition] = build_artifacts(
            arrays,
            reproduced_probability,
            threshold,
            closing,
            TOKEN_COUNT,
            "mask_topo_recheck",
            20260731,
        )

    predictions: dict[str, dict[str, np.ndarray]] = {
        condition: {} for condition in STRESS_CONDITIONS
    }
    sensitivity_predictions: dict[str, np.ndarray] = {}
    checkpoint_audit: dict[str, Any] = {}
    truth_reference: np.ndarray | None = None
    for model_name in MODELS:
        print(
            f"ROBUSTNESS_MODEL dataset={args.dataset} seed={args.seed} model={model_name}",
            flush=True,
        )
        model, spec = load_model(root, args.dataset, args.seed, model_name)
        expected_truth, expected_clean = load_expected_prediction(spec)
        if truth_reference is None:
            truth_reference = expected_truth
        elif not np.array_equal(truth_reference, expected_truth):
            raise RuntimeError("Clean truth differs across model archives.")
        model = model.to(device).eval()
        clean_prediction = predict_classifier(
            model,
            model_name,
            image_conditions["clean"],
            (
                reproduced_probability
                if model_name == "mask_topo"
                else cached_clean_probability
            ),
            stress_artifacts["clean"] if model_name == "mask_topo" else None,
            descriptors["clean"] if model_name in PH_MODELS else None,
            args.batch_size,
            device,
        )
        mismatch = int(np.count_nonzero(clean_prediction != expected_clean))
        if mismatch:
            raise RuntimeError(
                f"Clean checkpoint prediction mismatch: {model_name}, {mismatch}."
            )
        predictions["clean"][model_name] = clean_prediction
        for condition in STRESS_CONDITIONS[1:]:
            if condition in {"mask_break_erosion1", "mask_false_link_closing2"} and (
                model_name not in MASK_SENSITIVE
            ):
                predictions[condition][model_name] = clean_prediction.copy()
                continue
            current_images = image_conditions.get(condition, image_conditions["clean"])
            predictions[condition][model_name] = predict_classifier(
                model,
                model_name,
                current_images,
                stress_probabilities[condition],
                stress_artifacts[condition] if model_name == "mask_topo" else None,
                descriptors[condition] if model_name in PH_MODELS else None,
                args.batch_size,
                device,
            )
        if model_name == "mask_topo":
            for condition in SENSITIVITY_CONDITIONS:
                sensitivity_predictions[condition] = predict_classifier(
                    model,
                    model_name,
                    image_conditions["clean"],
                    cached_clean_probability,
                    sensitivity_artifacts[condition],
                    None,
                    args.batch_size,
                    device,
                )
        checkpoint_audit[model_name] = {
            "path": str(spec["checkpoint"].resolve()),
            "sha256": file_sha256(spec["checkpoint"]),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "clean_saved_prediction_mismatch_count": mismatch,
        }
        del model
        torch.cuda.empty_cache()
    if truth_reference is None or not np.array_equal(truth_reference, arrays["connected"]):
        raise RuntimeError("Truth does not align with frozen dataset arrays.")

    stress_results: dict[str, Any] = {}
    for condition in STRESS_CONDITIONS:
        stress_results[condition] = {
            "mask_metrics": pixel_mask_metrics(
                    ground_truth_masks,
                    stress_probabilities[condition],
                    cached_clean_probability,
            ),
            "masktopo_native_precision_mask_metrics": pixel_mask_metrics(
                ground_truth_masks,
                masktopo_probabilities[condition],
                reproduced_probability,
            ),
            "mask_probability_sha256": array_sha256(stress_probabilities[condition]),
            "image_sha256": array_sha256(
                image_conditions.get(condition, image_conditions["clean"])
            ),
            "ph_token_diagnostics": ph_token_diagnostics(descriptors[condition]),
            "mask_topo_artifact_diagnostics": stress_artifact_diagnostics[condition],
            "model_results": {
                model_name: balanced_metrics(
                    truth_reference, predictions[condition][model_name]
                )
                for model_name in MODELS
            },
        }
    sensitivity_results = {
        condition: {
            "mask_threshold": threshold,
            "closing_iterations": closing,
            "mask_topo_metrics": balanced_metrics(
                truth_reference, sensitivity_predictions[condition]
            ),
            "artifact_diagnostics": sensitivity_diagnostics[condition],
        }
        for condition, (threshold, closing) in SENSITIVITY_CONDITIONS.items()
    }

    prediction_payload: dict[str, np.ndarray] = {
        "truth": truth_reference,
        "source_id": arrays["source_id"].astype(str),
    }
    for condition in STRESS_CONDITIONS:
        for model_name in MODELS:
            prediction_payload[f"stress__{condition}__{model_name}"] = predictions[
                condition
            ][model_name]
    for condition, prediction in sensitivity_predictions.items():
        prediction_payload[f"sensitivity__{condition}__mask_topo"] = prediction
    np.savez_compressed(output / "heldout_predictions.npz", **prediction_payload)
    result = {
        "experiment_id": "TB-B-260802-024",
        "status": "completed",
        "confirmatory_status": "retrospective_robustness",
        "dataset": args.dataset,
        "optimization_seed": args.seed,
        "input_perturbation_seed": 20268024
        + (0 if args.dataset == "crackforest" else 1000),
        "token_count": TOKEN_COUNT,
        "test_sample_count": int(truth_reference.size),
        "test_source_count": int(np.unique(arrays["source_id"].astype(str)).size),
        "stress_conditions": stress_results,
        "masktopo_sensitivity": sensitivity_results,
        "checkpoint_audit": checkpoint_audit,
        "mask_predictor": {
            "path": str(predictor_checkpoint_path.resolve()),
            "sha256": file_sha256(predictor_checkpoint_path),
            "clean_probability_max_abs_difference": float(probability_difference.max()),
            "clean_probability_mean_abs_difference": float(probability_difference.mean()),
        },
        "clean_probability_path": str(
            (current_prior / "mask_probabilities.npz").resolve()
        ),
        "clean_probability_sha256": file_sha256(
            current_prior / "mask_probabilities.npz"
        ),
        "masktopo_uses_reproduced_float32_probability": True,
        "reducers_and_ph_clean_use_saved_float16_probability": True,
        "ph_cache_path": str(ph_cache_path.resolve()),
        "ph_cache_sha256": file_sha256(ph_cache_path),
        "protocol": str((root / "ROBUSTNESS_STRESS_PROTOCOL.md").resolve()),
        "protocol_sha256": file_sha256(root / "ROBUSTNESS_STRESS_PROTOCOL.md"),
        "code_sha256": file_sha256(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "all_clean_checkpoint_predictions_reproduced": all(
            item["clean_saved_prediction_mismatch_count"] == 0
            for item in checkpoint_audit.values()
        ),
        "uses_ground_truth_mask_or_topology_at_inference": False,
        "test_was_previously_observed": True,
        "massachusetts_test_reopened": False,
        "disease_classification_p1_5_used": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "dataset": args.dataset,
                "seed": args.seed,
                "clean_checkpoints_reproduced": result[
                    "all_clean_checkpoint_predictions_reproduced"
                ],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

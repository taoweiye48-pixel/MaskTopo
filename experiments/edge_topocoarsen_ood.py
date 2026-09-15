from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from edge_graph_diagnostic import predict_edges
from edge_topocoarsen_connector import (
    EdgeCoarsenDataset,
    balanced_metrics,
    bootstrap_balanced_accuracy,
    canonicalize_images,
    evaluate,
    make_artifacts,
)
from topobridge_mvp import (
    choose_endpoint_pair,
    connected_components,
    load_or_generate,
    remove_small_components,
    render_sample,
    set_seed,
    smooth_field,
)
from topocoarsen_learned import LearnableTopoCoarsen, slice_arrays
from topocoarsen_oracle import TopoCoarsenModel, coarse_graph, grid_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EdgeTopoCoarsen image and topology OOD validation."
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--dev-size", type=int, default=250)
    parser.add_argument("--topology-ood-size", type=int, default=500)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--canonicalize", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument(
        "--edge-predictor-checkpoint",
        type=Path,
        default=Path(
            "results_learned_topocoarsen/"
            "learned_topocoarsen_seed20260810.pt"
        ),
    )
    parser.add_argument(
        "--connector-checkpoint",
        type=Path,
        default=Path(
            "results_edge_topocoarsen/"
            "edge_topocoarsen_seed20260810.pt"
        ),
    )
    parser.add_argument(
        "--grid-checkpoint",
        type=Path,
        default=Path(
            "results_learned_topocoarsen_v2/grid_seed20260810.pt"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results_edge_topocoarsen_ood")
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def transformed_variants(
    arrays: dict[str, np.ndarray], seed: int
) -> dict[str, dict[str, np.ndarray]]:
    images = arrays["images"]
    labels = arrays["connected"].copy()
    normalized = images.astype(np.float32) / 255.0
    rng = np.random.default_rng(seed + 41)

    photometric = 0.12 + 0.76 * normalized
    photometric += rng.normal(
        0.0, 0.035, size=photometric.shape
    ).astype(np.float32)
    photometric = np.clip(photometric, 0.0, 1.0)

    tensor = torch.from_numpy(normalized)
    downsampled = F.interpolate(
        tensor, size=(48, 48), mode="bilinear", align_corners=False
    )
    resampled = F.interpolate(
        downsampled, size=(64, 64), mode="bilinear", align_corners=False
    ).numpy()

    variants = {
        "id_heldout": images.copy(),
        "rotate_90": np.rot90(images, k=1, axes=(-2, -1)).copy(),
        "horizontal_flip": np.flip(images, axis=-1).copy(),
        "low_contrast_noise": np.round(photometric * 255.0).astype(np.uint8),
        "resolution_48_to_64": np.round(
            np.clip(resampled, 0.0, 1.0) * 255.0
        ).astype(np.uint8),
    }
    return {
        name: {"images": value, "connected": labels.copy()}
        for name, value in variants.items()
    }


def generate_topology_ood(
    count: int, seed: int, cache_path: Path
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        with np.load(cache_path) as archive:
            return {key: archive[key] for key in archive.files}
    images = np.empty((count, 3, 64, 64), dtype=np.uint8)
    labels_out = np.empty(count, dtype=np.int64)
    component_counts = np.empty(count, dtype=np.int64)
    for index in range(count):
        rng = np.random.default_rng(seed + index * 104729)
        desired_connected = index % 2
        target_distance = [0.18, 0.30, 0.42][(index // 2) % 3]
        accepted = False
        for _ in range(1200):
            field = smooth_field(
                rng.normal(size=(32, 32)), int(rng.integers(1, 4))
            )
            quantile = rng.uniform(0.60, 0.74)
            mask = field > np.quantile(field, quantile)
            mask = remove_small_components(mask, minimum_size=4)
            component_map, sizes = connected_components(mask)
            component_count = len(sizes)
            foreground_fraction = float(np.mean(mask))
            if not (
                5 <= component_count <= 7
                and 0.12 <= foreground_fraction <= 0.43
            ):
                continue
            pair = choose_endpoint_pair(
                component_map, desired_connected, target_distance, rng
            )
            if pair is None:
                continue
            first, second = pair
            rendered = render_sample(mask, first, second, rng)
            images[index] = np.round(rendered * 255.0).astype(np.uint8)
            labels_out[index] = desired_connected
            component_counts[index] = component_count
            accepted = True
            break
        if not accepted:
            raise RuntimeError(
                f"Could not generate topology-OOD sample {index}."
            )
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(
                f"TOPOLOGY_OOD_DATA {index + 1}/{count}", flush=True
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        images=images,
        connected=labels_out,
        component_count=component_counts,
    )
    return {
        "images": images,
        "connected": labels_out,
        "component_count": component_counts,
    }


def fixed_grid_artifacts(
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


def load_model(
    name: str,
    checkpoint_path: Path,
    dim: int,
    token_count: int,
    device: torch.device,
) -> TopoCoarsenModel:
    model = TopoCoarsenModel(name, dim, token_count).to(device)
    checkpoint = torch.load(
        checkpoint_path.resolve(), map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate_variant(
    name: str,
    arrays: dict[str, np.ndarray],
    edge_predictor: LearnableTopoCoarsen,
    connector: TopoCoarsenModel,
    grid: TopoCoarsenModel,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if args.canonicalize:
        arrays = {
            **arrays,
            "images": canonicalize_images(arrays["images"]),
        }
    normalized = arrays["images"].astype(np.float32) / 255.0
    horizontal, vertical = predict_edges(
        edge_predictor, normalized, device, args.batch_size
    )
    artifacts, artifact_diagnostics = make_artifacts(
        normalized,
        horizontal,
        vertical,
        args.threshold,
        args.token_count,
    )
    structural_prediction = artifact_diagnostics.pop(
        "structural_predictions"
    )
    structural = balanced_metrics(
        arrays["connected"], structural_prediction
    )
    edge_dataset = EdgeCoarsenDataset(arrays, artifacts)
    edge_loader = torch.utils.data.DataLoader(
        edge_dataset, batch_size=args.batch_size, shuffle=False
    )
    edge_metrics, edge_prediction, truth = evaluate(
        connector, edge_loader, device, return_predictions=True
    )
    if edge_prediction is None or truth is None:
        raise AssertionError("Missing OOD predictions.")

    grid_dataset = EdgeCoarsenDataset(
        arrays,
        fixed_grid_artifacts(len(edge_dataset), args.token_count),
    )
    grid_loader = torch.utils.data.DataLoader(
        grid_dataset, batch_size=args.batch_size, shuffle=False
    )
    grid_metrics, _, _ = evaluate(grid, grid_loader, device)
    return {
        "variant": name,
        "sample_count": int(arrays["images"].shape[0]),
        "structural_balanced_accuracy": structural["balanced_accuracy"],
        "edge_topocoarsen_balanced_accuracy": edge_metrics[
            "balanced_accuracy"
        ],
        "edge_topocoarsen_positive_accuracy": edge_metrics[
            "positive_accuracy"
        ],
        "edge_topocoarsen_negative_accuracy": edge_metrics[
            "negative_accuracy"
        ],
        "edge_topocoarsen_bootstrap_95_ci": bootstrap_balanced_accuracy(
            truth, edge_prediction, args.seed + sum(map(ord, name))
        ),
        "grid_balanced_accuracy": grid_metrics["balanced_accuracy"],
        "edge_gain_over_grid": (
            edge_metrics["balanced_accuracy"]
            - grid_metrics["balanced_accuracy"]
        ),
        "maximum_predicted_foreground_components": (
            artifact_diagnostics["foreground_component_count"]["maximum"]
        ),
        "all_16_tokens_nonempty": artifact_diagnostics[
            "all_16_tokens_nonempty"
        ],
    }


def run_self_test() -> None:
    arrays = {
        "images": np.zeros((2, 3, 64, 64), dtype=np.uint8),
        "connected": np.array([0, 1], dtype=np.int64),
    }
    variants = transformed_variants(arrays, 123)
    assert set(variants) == {
        "id_heldout",
        "rotate_90",
        "horizontal_flip",
        "low_contrast_noise",
        "resolution_48_to_64",
    }
    assert all(value["images"].shape == (2, 3, 64, 64) for value in variants.values())
    artifacts = fixed_grid_artifacts(2, 16)
    assert artifacts["assignment"].shape == (2, 16, 16)
    print("EDGE_TOPOCOARSEN_OOD_SELF_TEST_PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    set_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validation = load_or_generate(
        args.cache_dir.resolve(),
        "val",
        args.val_size,
        args.seed + 1_000_000,
    )
    heldout = slice_arrays(
        validation, args.dev_size, args.val_size
    )
    variants = transformed_variants(heldout, args.seed)
    topology_cache = (
        args.cache_dir.resolve()
        / f"topology_ood_n{args.topology_ood_size}_seed{args.seed + 2_000_000}.npz"
    )
    topology_ood = generate_topology_ood(
        args.topology_ood_size,
        args.seed + 2_000_000,
        topology_cache,
    )
    variants["topology_5_to_7_components"] = {
        "images": topology_ood["images"],
        "connected": topology_ood["connected"],
    }

    edge_predictor = LearnableTopoCoarsen(
        args.dim, args.token_count, args.temperature
    ).to(device)
    edge_checkpoint = torch.load(
        args.edge_predictor_checkpoint.resolve(),
        map_location=device,
        weights_only=False,
    )
    edge_predictor.load_state_dict(edge_checkpoint["model"])
    edge_predictor.eval()
    connector = load_model(
        "edge_topocoarsen",
        args.connector_checkpoint,
        args.dim,
        args.token_count,
        device,
    )
    grid = load_model(
        "grid",
        args.grid_checkpoint,
        args.dim,
        args.token_count,
        device,
    )

    rows = []
    for name, arrays in variants.items():
        row = evaluate_variant(
            name,
            arrays,
            edge_predictor,
            connector,
            grid,
            args,
            device,
        )
        rows.append(row)
        print(
            f"OOD variant={name} "
            f"structural={row['structural_balanced_accuracy']:.4f} "
            f"edge={row['edge_topocoarsen_balanced_accuracy']:.4f} "
            f"grid={row['grid_balanced_accuracy']:.4f}",
            flush=True,
        )

    id_score = next(
        row["edge_topocoarsen_balanced_accuracy"]
        for row in rows
        if row["variant"] == "id_heldout"
    )
    ood_rows = [row for row in rows if row["variant"] != "id_heldout"]
    worst_ood = min(
        row["edge_topocoarsen_balanced_accuracy"] for row in ood_rows
    )
    minimum_gain = min(row["edge_gain_over_grid"] for row in ood_rows)
    result = {
        "experiment_id": "edge_topocoarsen_ood_gate",
        "status": "completed",
        "fixed_threshold_selected_before_ood": args.threshold,
        "per_image_channel_contrast_canonicalization": args.canonicalize,
        "variants": rows,
        "topology_ood_definition": {
            "training_component_range": [2, 4],
            "ood_component_range": [5, 7],
            "ood_sample_count": args.topology_ood_size,
            "balanced_labels": True,
        },
        "inference_uses_images_only": True,
        "summary": {
            "id_balanced_accuracy": id_score,
            "worst_ood_balanced_accuracy": worst_ood,
            "worst_ood_drop": id_score - worst_ood,
            "minimum_ood_gain_over_grid": minimum_gain,
        },
        "gates": {
            "all_ood_structural_scores_at_least_80pct": all(
                row["structural_balanced_accuracy"] >= 0.80
                for row in ood_rows
            ),
            "all_ood_connector_scores_at_least_75pct": all(
                row["edge_topocoarsen_balanced_accuracy"] >= 0.75
                for row in ood_rows
            ),
            "all_ood_gains_over_grid_at_least_10pp": all(
                row["edge_gain_over_grid"] >= 0.10 for row in ood_rows
            ),
        },
    }
    result["verdict"] = (
        "OOD_GATE_PASS"
        if all(result["gates"].values())
        else "OOD_WEAKNESS_FOUND"
    )
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    flat_rows = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if key != "edge_topocoarsen_bootstrap_95_ci"
        }
        flat["ci_low"], flat["ci_high"] = row[
            "edge_topocoarsen_bootstrap_95_ci"
        ]
        flat_rows.append(flat)
    with (output / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    names = [row["variant"] for row in rows]
    edge_values = [
        row["edge_topocoarsen_balanced_accuracy"] for row in rows
    ]
    grid_values = [row["grid_balanced_accuracy"] for row in rows]
    positions = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(
        positions - width / 2,
        grid_values,
        width,
        label="grid",
        color="#4C78A8",
    )
    ax.bar(
        positions + width / 2,
        edge_values,
        width,
        label="EdgeTopoCoarsen",
        color="#E45756",
    )
    ax.set_xticks(positions, names, rotation=20, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("EdgeTopoCoarsen distribution-shift validation")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.savefig(output / "ood_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

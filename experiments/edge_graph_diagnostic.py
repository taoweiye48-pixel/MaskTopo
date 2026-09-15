from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from topobridge_mvp import load_or_generate, set_seed
from topocoarsen_learned import LearnableTopoCoarsen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict image-only edge graph reachability diagnostic."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results_learned_topocoarsen/"
            "learned_topocoarsen_seed20260810.pt"
        ),
    )
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--dev-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999],
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument(
        "--output", type=Path, default=Path("results_edge_graph")
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def image_foreground(image: np.ndarray) -> np.ndarray:
    channel = image[0]
    return channel.reshape(16, 4, 16, 4).max(axis=(1, 3)) > 0.5


def image_endpoints(image: np.ndarray) -> tuple[int, int]:
    endpoints = []
    for channel in (1, 2):
        coordinates = np.argwhere(image[channel] > 0.5)
        if coordinates.size == 0:
            raise ValueError("Endpoint marker was not found in the image.")
        center = np.mean(coordinates, axis=0)
        row = int(np.clip(np.floor(center[0] / 4), 0, 15))
        col = int(np.clip(np.floor(center[1] / 4), 0, 15))
        endpoints.append(row * 16 + col)
    return int(endpoints[0]), int(endpoints[1])


@torch.no_grad()
def predict_edges(
    model: LearnableTopoCoarsen,
    images: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    horizontal = []
    vertical = []
    model.eval()
    for start in range(0, images.shape[0], batch_size):
        image = torch.from_numpy(images[start : start + batch_size]).to(device)
        features = model.stem(image)
        horizontal.append(
            torch.sigmoid(
                model.edge_logits(
                    features[:, :, :, :-1], features[:, :, :, 1:]
                )
            )
            .cpu()
            .numpy()
        )
        vertical.append(
            torch.sigmoid(
                model.edge_logits(
                    features[:, :, :-1, :], features[:, :, 1:, :]
                )
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(horizontal), np.concatenate(vertical)


def graph_reachable(
    foreground: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    endpoints: tuple[int, int],
    threshold: float,
) -> int:
    first, second = endpoints
    if first == second:
        return 1
    start = divmod(first, 16)
    target = divmod(second, 16)
    if not foreground[start] or not foreground[target]:
        return 0
    queue: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while queue:
        row, col = queue.popleft()
        neighbors = (
            (
                row,
                col - 1,
                horizontal[row, col - 1] if col > 0 else -1.0,
            ),
            (
                row,
                col + 1,
                horizontal[row, col] if col < 15 else -1.0,
            ),
            (
                row - 1,
                col,
                vertical[row - 1, col] if row > 0 else -1.0,
            ),
            (
                row + 1,
                col,
                vertical[row, col] if row < 15 else -1.0,
            ),
        )
        for nr, nc, probability in neighbors:
            point = (nr, nc)
            if (
                0 <= nr < 16
                and 0 <= nc < 16
                and foreground[point]
                and probability >= threshold
                and point not in seen
            ):
                if point == target:
                    return 1
                seen.add(point)
                queue.append(point)
    return 0


def raw_mask_reachable(
    foreground: np.ndarray, endpoints: tuple[int, int]
) -> int:
    horizontal = np.ones((16, 15), dtype=np.float32)
    vertical = np.ones((15, 16), dtype=np.float32)
    return graph_reachable(
        foreground, horizontal, vertical, endpoints, threshold=0.5
    )


def metrics(
    indices: np.ndarray,
    labels: np.ndarray,
    foreground: np.ndarray,
    endpoints: list[tuple[int, int]],
    horizontal: np.ndarray,
    vertical: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = np.array(
        [
            graph_reachable(
                foreground[index],
                horizontal[index],
                vertical[index],
                endpoints[index],
                threshold,
            )
            for index in indices
        ]
    )
    truth = labels[indices]
    positive = float(np.mean(prediction[truth == 1] == 1))
    negative = float(np.mean(prediction[truth == 0] == 0))
    return {
        "balanced_accuracy": 0.5 * (positive + negative),
        "positive_accuracy": positive,
        "negative_accuracy": negative,
    }


def run_self_test() -> None:
    image = np.zeros((3, 64, 64), dtype=np.float32)
    image[0, 8:24, 8:24] = 1.0
    image[1, 9:12, 9:12] = 1.0
    image[2, 20:23, 20:23] = 1.0
    foreground = image_foreground(image)
    endpoints = image_endpoints(image)
    assert foreground.shape == (16, 16)
    assert endpoints == (2 * 16 + 2, 5 * 16 + 5)
    assert raw_mask_reachable(foreground, endpoints) == 1
    print("EDGE_GRAPH_SELF_TEST_PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if not (0 < args.dev_size < args.val_size):
        raise ValueError("dev-size must be between 0 and val-size.")
    set_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays = load_or_generate(
        args.cache_dir.resolve(),
        "val",
        args.val_size,
        args.seed + 1_000_000,
    )
    images = arrays["images"].astype(np.float32) / 255.0
    labels = arrays["connected"]
    foreground = np.stack([image_foreground(image) for image in images])
    endpoints = [image_endpoints(image) for image in images]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnableTopoCoarsen(
        args.dim, args.token_count, args.temperature
    ).to(device)
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    horizontal, vertical = predict_edges(model, images, device)

    dev_indices = np.arange(0, args.dev_size)
    test_indices = np.arange(args.dev_size, args.val_size)
    rows = []
    for threshold in args.thresholds:
        dev_metrics = metrics(
            dev_indices,
            labels,
            foreground,
            endpoints,
            horizontal,
            vertical,
            threshold,
        )
        rows.append({"threshold": threshold, **dev_metrics})
    selected = max(
        rows,
        key=lambda row: (row["balanced_accuracy"], row["threshold"]),
    )
    test_metrics = metrics(
        test_indices,
        labels,
        foreground,
        endpoints,
        horizontal,
        vertical,
        float(selected["threshold"]),
    )
    raw_prediction = np.array(
        [
            raw_mask_reachable(foreground[index], endpoints[index])
            for index in test_indices
        ]
    )
    raw_truth = labels[test_indices]
    raw_positive = float(np.mean(raw_prediction[raw_truth == 1] == 1))
    raw_negative = float(np.mean(raw_prediction[raw_truth == 0] == 0))
    raw_balanced = 0.5 * (raw_positive + raw_negative)

    result = {
        "experiment_id": "image_only_edge_graph_diagnostic",
        "status": "completed",
        "checkpoint": str(args.checkpoint.resolve()),
        "dev_count": int(dev_indices.size),
        "test_count": int(test_indices.size),
        "selected_threshold": float(selected["threshold"]),
        "selected_dev_metrics": {
            key: value for key, value in selected.items() if key != "threshold"
        },
        "held_out_test_metrics": test_metrics,
        "raw_mask_test_balanced_accuracy": raw_balanced,
        "inference_inputs": [
            "image channel 0",
            "image channel 1",
            "image channel 2",
            "learned edge probabilities from image",
        ],
        "uses_patch_ids_at_dev_or_test": False,
        "uses_endpoint_metadata_at_dev_or_test": False,
        "is_connector_evaluation": False,
        "verdict": (
            "EDGE_GRAPH_LEARNABILITY_SUPPORTED"
            if test_metrics["balanced_accuracy"] >= 0.70
            else "EDGE_GRAPH_LEARNABILITY_NOT_SUPPORTED"
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "threshold_curve.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(
        [row["threshold"] for row in rows],
        [row["balanced_accuracy"] for row in rows],
        marker="o",
    )
    ax.axvline(
        selected["threshold"], color="#E45756", linestyle="--", linewidth=1
    )
    ax.set_ylim(0.45, 1.01)
    ax.set_xlabel("edge probability threshold (selected on dev)")
    ax.set_ylabel("dev balanced accuracy")
    ax.set_title("Image-only learned edge graph calibration")
    ax.grid(alpha=0.25)
    fig.savefig(output / "threshold_curve.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

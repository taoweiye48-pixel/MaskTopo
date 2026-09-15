from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from topobridge_mvp import (
    ConvConnector,
    PatchStem,
    coordinate_shortcut_accuracy,
    load_or_generate,
    set_seed,
)


MODEL_NAMES = [
    "random",
    "uniform",
    "foreground",
    "conv",
    "topology_oracle",
    "topology_coords_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle upper-bound test for topology-critical token retention."
    )
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--output", type=Path, default=Path("results_oracle"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def endpoint_patch_indices(endpoint_features: np.ndarray) -> list[int]:
    indices: list[int] = []
    for offset in (0, 2):
        row32 = int(np.clip(np.rint(endpoint_features[offset] * 31.0), 0, 31))
        col32 = int(np.clip(np.rint(endpoint_features[offset + 1] * 31.0), 0, 31))
        row16 = row32 // 2
        col16 = col32 // 2
        index = row16 * 16 + col16
        if index not in indices:
            indices.append(index)
    return indices


def topology_critical_scores(patch_ids: np.ndarray) -> np.ndarray:
    """Exact graph-theoretic damage score, independent of endpoint labels."""
    if patch_ids.shape != (16, 16):
        raise ValueError("Expected 16x16 patch component IDs.")
    scores = np.zeros(256, dtype=np.float32)
    for component_id in np.unique(patch_ids):
        if component_id <= 0:
            continue
        coordinates = [tuple(x) for x in np.argwhere(patch_ids == component_id)]
        node_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
        count = len(coordinates)
        if count <= 2:
            continue
        adjacency: list[list[int]] = [[] for _ in range(count)]
        for index, (row, col) in enumerate(coordinates):
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                other = node_index.get(neighbor)
                if other is not None:
                    adjacency[index].append(other)

        discovery = [-1] * count
        low = [0] * count
        parent = [-1] * count
        subtree = [1] * count
        children: list[list[int]] = [[] for _ in range(count)]
        tick = 0

        def dfs(node: int) -> None:
            nonlocal tick
            discovery[node] = low[node] = tick
            tick += 1
            for neighbor in adjacency[node]:
                if discovery[neighbor] < 0:
                    parent[neighbor] = node
                    children[node].append(neighbor)
                    dfs(neighbor)
                    subtree[node] += subtree[neighbor]
                    low[node] = min(low[node], low[neighbor])
                elif neighbor != parent[node]:
                    low[node] = min(low[node], discovery[neighbor])

        dfs(0)
        denominator = max((count - 1) * (count - 2) / 2.0, 1.0)
        for node in range(count):
            separated: list[int] = []
            for child in children[node]:
                if parent[node] < 0 or low[child] >= discovery[node]:
                    separated.append(subtree[child])
                if low[child] > discovery[node]:
                    bridge_weight = min(subtree[child], count - subtree[child]) / count
                    first_flat = coordinates[node][0] * 16 + coordinates[node][1]
                    child_flat = coordinates[child][0] * 16 + coordinates[child][1]
                    scores[first_flat] += float(bridge_weight)
                    scores[child_flat] += float(bridge_weight)
            remainder = count - 1 - sum(separated)
            if remainder > 0:
                separated.append(remainder)
            if len(separated) >= 2:
                disconnected_pairs = 0.0
                running = 0
                for part in separated:
                    disconnected_pairs += running * part
                    running += part
                flat = coordinates[node][0] * 16 + coordinates[node][1]
                scores[flat] += float(disconnected_pairs / denominator)
    return scores


def farthest_fill(
    selected: list[int], candidates: list[int], token_count: int
) -> list[int]:
    result = list(dict.fromkeys(selected))
    remaining = np.array(
        [
            candidate
            for candidate in dict.fromkeys(candidates)
            if candidate not in result
        ],
        dtype=np.int64,
    )
    if remaining.size == 0:
        return result
    remaining_rc = np.stack((remaining // 16, remaining % 16), axis=1)
    minimum_distance = np.full(remaining.shape[0], np.inf, dtype=np.float64)
    for item in result:
        point = np.array([item // 16, item % 16])
        minimum_distance = np.minimum(
            minimum_distance, np.sum((remaining_rc - point) ** 2, axis=1)
        )
    available = np.ones(remaining.shape[0], dtype=bool)
    while len(result) < token_count and np.any(available):
        if not result:
            chosen_position = int(np.flatnonzero(available)[np.sum(available) // 2])
        else:
            masked_distance = np.where(available, minimum_distance, -1.0)
            chosen_position = int(np.argmax(masked_distance))
        chosen = int(remaining[chosen_position])
        result.append(chosen)
        available[chosen_position] = False
        chosen_point = remaining_rc[chosen_position]
        minimum_distance = np.minimum(
            minimum_distance,
            np.sum((remaining_rc - chosen_point) ** 2, axis=1),
        )
    return result


def build_selections(
    arrays: dict[str, np.ndarray], token_count: int, policy_seed: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    count = arrays["images"].shape[0]
    selections = {
        name: np.empty((count, token_count), dtype=np.int64)
        for name in ("random", "uniform", "foreground", "topology")
    }
    topology_recalls = []
    foreground_recalls = []
    topology_critical_counts = []
    overlaps = []
    endpoint_failures = 0
    uniform_candidates = [
        row * 16 + col for row in (1, 5, 9, 13) for col in (1, 5, 9, 13)
    ]

    for sample_index in range(count):
        patch_ids = arrays["patch_ids"][sample_index]
        endpoints = endpoint_patch_indices(arrays["endpoint_features"][sample_index])
        foreground = np.flatnonzero(patch_ids.reshape(-1) > 0).tolist()
        all_nodes = list(range(256))
        rng = np.random.default_rng(policy_seed + sample_index * 104729)

        random_candidates = np.array(all_nodes, dtype=np.int64)
        rng.shuffle(random_candidates)
        random_selected = endpoints + [
            int(item) for item in random_candidates if int(item) not in endpoints
        ]
        random_selected = random_selected[:token_count]

        uniform_selected = farthest_fill(
            endpoints, uniform_candidates + all_nodes, token_count
        )
        foreground_selected = farthest_fill(
            endpoints, foreground + all_nodes, token_count
        )

        scores = topology_critical_scores(patch_ids)
        critical = np.flatnonzero(scores > 0)
        ranked_critical = sorted(
            (int(item) for item in critical),
            key=lambda item: (-float(scores[item]), item),
        )
        topology_selected = list(dict.fromkeys(endpoints + ranked_critical))
        if len(topology_selected) < token_count:
            topology_selected = farthest_fill(
                topology_selected, foreground + all_nodes, token_count
            )
        topology_selected = topology_selected[:token_count]

        selections["random"][sample_index] = random_selected
        selections["uniform"][sample_index] = uniform_selected
        selections["foreground"][sample_index] = foreground_selected
        selections["topology"][sample_index] = topology_selected

        if not set(endpoints).issubset(topology_selected):
            endpoint_failures += 1
        critical_set = set(int(item) for item in critical)
        recall_denominator = min(token_count, len(critical_set))
        if recall_denominator > 0:
            topology_recalls.append(
                len(critical_set.intersection(topology_selected)) / recall_denominator
            )
            foreground_recalls.append(
                len(critical_set.intersection(foreground_selected)) / recall_denominator
            )
        topology_critical_counts.append(len(critical_set))
        first = set(topology_selected)
        second = set(foreground_selected)
        overlaps.append(len(first & second) / len(first | second))

    for name, selected in selections.items():
        if selected.shape != (count, token_count):
            raise AssertionError(f"Bad selection shape for {name}.")
        if np.any(np.apply_along_axis(lambda row: len(set(row)) != token_count, 1, selected)):
            raise AssertionError(f"Duplicate tokens in {name}.")

    diagnostics = {
        "samples": int(count),
        "token_count": int(token_count),
        "mean_critical_nodes": float(np.mean(topology_critical_counts)),
        "median_critical_nodes": float(np.median(topology_critical_counts)),
        "topology_normalized_critical_recall": float(np.mean(topology_recalls)),
        "foreground_normalized_critical_recall": float(np.mean(foreground_recalls)),
        "topology_foreground_mean_jaccard": float(np.mean(overlaps)),
        "topology_endpoint_failures": int(endpoint_failures),
    }
    return selections, diagnostics


class OracleDataset(Dataset):
    def __init__(
        self, arrays: dict[str, np.ndarray], selections: dict[str, np.ndarray]
    ) -> None:
        self.arrays = arrays
        self.selections = selections

    def __len__(self) -> int:
        return int(self.arrays["images"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "image": torch.from_numpy(
                self.arrays["images"][index].astype(np.float32) / 255.0
            ),
            "connected": torch.tensor(
                self.arrays["connected"][index], dtype=torch.float32
            ),
        }
        for name, selected in self.selections.items():
            item[f"select_{name}"] = torch.from_numpy(selected[index])
        return item


def grid_coordinates(side: int) -> torch.Tensor:
    axis = (torch.arange(side, dtype=torch.float32) + 0.5) / side
    rows, cols = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((2.0 * rows - 1.0, 2.0 * cols - 1.0), dim=-1).reshape(-1, 2)


class TokenHead(nn.Module):
    def __init__(self, dim: int, token_count: int) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_position = nn.Parameter(torch.zeros(1, 1, dim))
        self.coordinate = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.cls_position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=4 * dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(dim)
        self.connected = nn.Linear(dim, 1)
        self.token_count = token_count

    def forward(self, tokens: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != self.token_count:
            raise ValueError("Unexpected token count.")
        encoded_tokens = tokens + self.coordinate(coordinates)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1) + self.cls_position
        sequence = torch.cat((cls, encoded_tokens), dim=1)
        encoded = self.encoder(sequence)
        return self.connected(self.norm(encoded[:, 0])).squeeze(-1)


class OracleModel(nn.Module):
    def __init__(self, name: str, dim: int, token_count: int) -> None:
        super().__init__()
        self.name = name
        self.token_count = token_count
        self.stem = PatchStem(dim)
        self.conv = ConvConnector(dim, int(math.isqrt(token_count))) if name == "conv" else None
        self.project = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.head = TokenHead(dim, token_count)
        self.register_buffer("fine_coordinates", grid_coordinates(16), persistent=False)
        conv_side = int(math.isqrt(token_count))
        self.register_buffer(
            "coarse_coordinates", grid_coordinates(conv_side), persistent=False
        )

    def forward(
        self, image: torch.Tensor, selection: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self.stem(image)
        if self.name == "conv":
            if self.conv is None:
                raise AssertionError("Missing convolutional connector.")
            compressed, _ = self.conv(features)
            tokens = compressed.flatten(2).transpose(1, 2)
            coordinates = self.coarse_coordinates.unsqueeze(0).expand(
                image.shape[0], -1, -1
            )
        else:
            if selection is None:
                raise ValueError("A token selection is required.")
            flat = features.flatten(2).transpose(1, 2)
            gather_index = selection.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
            tokens = torch.gather(flat, 1, gather_index)
            coordinate_bank = self.fine_coordinates.unsqueeze(0).expand(
                image.shape[0], -1, -1
            )
            coordinates = torch.gather(
                coordinate_bank, 1, selection.unsqueeze(-1).expand(-1, -1, 2)
            )
            if self.name == "topology_coords_only":
                tokens = torch.zeros_like(tokens)
        return self.head(self.project(tokens), coordinates)


def selection_key(model_name: str) -> str | None:
    if model_name == "conv":
        return None
    if model_name in {"topology_oracle", "topology_coords_only"}:
        return "select_topology"
    return f"select_{model_name}"


@torch.no_grad()
def evaluate(
    model: OracleModel,
    loader: DataLoader,
    device: torch.device,
    override_selection: str | None = None,
) -> dict[str, float]:
    model.eval()
    truths = []
    predictions = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        key = override_selection or selection_key(model.name)
        selected = (
            batch[key].to(device, non_blocking=True) if key is not None else None
        )
        logits = model(image, selected)
        truths.append(batch["connected"].bool())
        predictions.append((torch.sigmoid(logits) >= 0.5).cpu())
    truth = torch.cat(truths)
    prediction = torch.cat(predictions)
    positive = float((prediction[truth] == truth[truth]).float().mean())
    negative = float((prediction[~truth] == truth[~truth]).float().mean())
    return {
        "connected_accuracy": float((prediction == truth).float().mean()),
        "connected_balanced_accuracy": 0.5 * (positive + negative),
        "positive_accuracy": positive,
        "negative_accuracy": negative,
    }


def train_one(
    model_name: str,
    args: argparse.Namespace,
    train_dataset: OracleDataset,
    val_dataset: OracleDataset,
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = OracleModel(model_name, args.dim, args.token_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_metric = -1.0
    best_path = output / f"{model_name}_seed{args.seed}.pt"
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["connected"].to(device, non_blocking=True)
            key = selection_key(model_name)
            selected = (
                batch[key].to(device, non_blocking=True) if key is not None else None
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(image, selected)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batch_count += 1
        metrics = evaluate(model, val_loader, device)
        record = {
            "seed": args.seed,
            "model": model_name,
            "epoch": epoch,
            "train_loss": loss_sum / max(batch_count, 1),
            **metrics,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"ORACLE_TRAIN seed={args.seed} model={model_name} "
            f"epoch={epoch}/{args.epochs} loss={record['train_loss']:.4f} "
            f"conn_bal={metrics['connected_balanced_accuracy']:.4f}",
            flush=True,
        )
        if metrics["connected_balanced_accuracy"] > best_metric:
            best_metric = metrics["connected_balanced_accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "model_name": model_name,
                    "metrics": metrics,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, val_loader, device)
    summary: dict[str, Any] = {
        "seed": args.seed,
        "model": model_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch_connected_balanced_accuracy": best_metric,
        **final_metrics,
        "training_seconds": time.time() - started,
    }
    if model_name == "topology_oracle":
        for override in ("foreground", "uniform", "random"):
            metrics = evaluate(
                model, val_loader, device, override_selection=f"select_{override}"
            )
            summary[f"counterfactual_{override}_balanced_accuracy"] = metrics[
                "connected_balanced_accuracy"
            ]
    return summary, history


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_result(
    output: Path,
    summaries: list[dict[str, Any]],
    shortcut_accuracy: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    by_model = {row["model"]: row for row in summaries}
    baseline_names = [
        name for name in ("random", "uniform", "foreground", "conv") if name in by_model
    ]
    strongest_baseline = max(
        baseline_names,
        key=lambda name: by_model[name]["connected_balanced_accuracy"],
    )
    oracle = by_model["topology_oracle"]["connected_balanced_accuracy"]
    baseline = by_model[strongest_baseline]["connected_balanced_accuracy"]
    gain = oracle - baseline
    coordinate_only = by_model["topology_coords_only"][
        "connected_balanced_accuracy"
    ]
    counterfactual_foreground = by_model["topology_oracle"].get(
        "counterfactual_foreground_balanced_accuracy", float("nan")
    )
    counterfactual_drop = oracle - counterfactual_foreground
    leakage_pass = coordinate_only <= 0.60
    oracle_gain_pass = gain >= 0.03
    topology_dependence_pass = counterfactual_drop >= 0.02

    if not leakage_pass:
        verdict = "INVALID_ORACLE_SELECTION_LEAKAGE"
    elif oracle_gain_pass and topology_dependence_pass:
        verdict = "TOPOLOGY_DIRECTION_SUPPORTED"
    elif gain <= 0.0:
        verdict = "NO_GO_FOR_CRITICAL_TOKEN_SELECTION"
    else:
        verdict = "INCONCLUSIVE_REDESIGN_REQUIRED"

    result = {
        "experiment_id": "topobridge_topology_oracle_upper_bound",
        "status": "completed",
        "coordinate_shortcut_accuracy": shortcut_accuracy,
        "selection_diagnostics": diagnostics,
        "scores": {
            name: row["connected_balanced_accuracy"] for name, row in by_model.items()
        },
        "strongest_non_topology_baseline": strongest_baseline,
        "topology_oracle_gain": gain,
        "topology_coordinate_only_accuracy": coordinate_only,
        "topology_to_foreground_counterfactual_drop": counterfactual_drop,
        "gates": {
            "dataset_shortcut_pass": shortcut_accuracy <= 0.60,
            "coordinate_selection_leakage_pass": leakage_pass,
            "oracle_gain_at_least_3pp": oracle_gain_pass,
            "counterfactual_drop_at_least_2pp": topology_dependence_pass,
        },
        "verdict": verdict,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    names = [row["model"] for row in summaries]
    scores = [row["connected_balanced_accuracy"] for row in summaries]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = ["#4C78A8" if "topology" not in name else "#E45756" for name in names]
    ax.bar(names, scores, color=colors)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("validation connected balanced accuracy")
    ax.set_title("Topology-critical token Oracle upper-bound test")
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(output / "oracle_comparison.png", dpi=180)
    plt.close(fig)
    return result


def run_self_test() -> None:
    path = np.zeros((16, 16), dtype=np.int16)
    path[8, 2:14] = 1
    scores = topology_critical_scores(path)
    assert np.count_nonzero(scores) >= 10
    assert scores[8 * 16 + 8] > 0
    endpoint_features = np.array(
        [8 / 31, 4 / 31, 24 / 31, 26 / 31, 0, 0], dtype=np.float32
    )
    endpoints = endpoint_patch_indices(endpoint_features)
    assert endpoints == [4 * 16 + 2, 12 * 16 + 13]
    dummy = {
        "images": np.zeros((2, 3, 64, 64), dtype=np.uint8),
        "connected": np.array([0, 1]),
        "patch_ids": np.stack((path, path)),
        "endpoint_features": np.stack((endpoint_features, endpoint_features)),
    }
    selections, diagnostics = build_selections(dummy, 16, 123)
    for selected in selections.values():
        assert selected.shape == (2, 16)
        assert all(len(set(row)) == 16 for row in selected)
        assert all(set(endpoints).issubset(row) for row in selected)
    model = OracleModel("topology_oracle", 32, 16)
    logits = model(
        torch.zeros(2, 3, 64, 64), torch.from_numpy(selections["topology"])
    )
    assert logits.shape == (2,)
    assert diagnostics["topology_endpoint_failures"] == 0
    print("ORACLE_SELF_TEST_PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    unknown = sorted(set(args.models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    side = int(math.isqrt(args.token_count))
    if side * side != args.token_count:
        raise ValueError("token-count must be a perfect square for the conv baseline.")
    required = {"topology_oracle", "topology_coords_only"}
    if not required.issubset(args.models):
        raise ValueError(f"Models must include {sorted(required)} for gate evaluation.")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "args": vars(args),
            },
            default=str,
        ),
        flush=True,
    )

    train_arrays = load_or_generate(
        cache_dir, "train", args.train_size, args.seed
    )
    val_arrays = load_or_generate(
        cache_dir, "val", args.val_size, args.seed + 1_000_000
    )
    shortcut = coordinate_shortcut_accuracy(train_arrays, val_arrays)
    print(f"ORACLE_SHORTCUT distance_accuracy={shortcut:.4f}", flush=True)

    train_selections, train_diagnostics = build_selections(
        train_arrays, args.token_count, args.seed + 10_000_000
    )
    val_selections, val_diagnostics = build_selections(
        val_arrays, args.token_count, args.seed + 20_000_000
    )
    diagnostics = {
        "train": train_diagnostics,
        "validation": val_diagnostics,
    }
    (output / "selection_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False), flush=True)

    train_dataset = OracleDataset(train_arrays, train_selections)
    val_dataset = OracleDataset(val_arrays, val_selections)
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for model_name in args.models:
        summary, records = train_one(
            model_name, args, train_dataset, val_dataset, output, device
        )
        summaries.append(summary)
        history.extend(records)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "history.csv", history)
    result = render_result(output, summaries, shortcut, diagnostics)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import strong_reducer_baselines as reducer_baselines
from crackforest_real_gate import write_csv
from fives_external_gate import load_all_arrays
from strong_reducer_baselines import CommonTokenClassifier, ReducerDataset
from topobridge_mvp import PatchStem
from topocoarsen_oracle import fine_coordinates


COLLISION_MODELS = (
    "g2tm_fixedk_feature",
    "g2tm_fixedk_mask",
)
OFFICIAL_G2TM_REVISION = "f17d1c8374a6f09365d856ff40d5aaf6d0bcf5d4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matched fixed-budget G2TM collision baselines for the frozen "
            "FIVES TopoBridge experiment."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("real_data/FIVES/dataset/preprocessed512_v2"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("fives_cache_v2"))
    parser.add_argument("--mask-run-dir", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--dev-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=1200)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--data-seed", type=int, default=20260810)
    parser.add_argument("--split-seed", type=int, default=20260731)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=COLLISION_MODELS,
        default=list(COLLISION_MODELS),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grid_edges(side: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    source: list[int] = []
    target: list[int] = []
    for row in range(side):
        for col in range(side):
            node = row * side + col
            if col + 1 < side:
                source.append(node)
                target.append(node + 1)
            if row + 1 < side:
                source.append(node)
                target.append(node + side)
    return (
        torch.tensor(source, dtype=torch.long),
        torch.tensor(target, dtype=torch.long),
    )


def maximum_spanning_forest_assignments(
    features: torch.Tensor,
    token_count: int,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
) -> torch.Tensor:
    """Return exactly K connected feature-similarity groups.

    G2TM thresholds cosine similarities on the four-neighbour patch graph and
    merges each resulting connected component. For a matched fixed-token
    comparison, this function processes the same graph edges in descending
    cosine-similarity order and stops Kruskal union at exactly K components.
    The discrete grouping follows the official G2TM no-gradient convention.
    """
    if features.ndim != 3:
        raise ValueError("Expected [batch, token, dim] features.")
    batch_size, fine_count, _ = features.shape
    if not 0 < token_count <= fine_count:
        raise ValueError("token_count must be between 1 and the fine-token count.")
    if fine_count != 256:
        raise ValueError("The frozen FIVES protocol requires a 16x16 token grid.")

    with torch.no_grad():
        normalized = F.normalize(features.float(), dim=-1)
        source = edge_source.to(features.device)
        target = edge_target.to(features.device)
        similarities = (
            normalized[:, source] * normalized[:, target]
        ).sum(dim=-1)
        similarity_array = similarities.detach().cpu().numpy()
        source_array = edge_source.cpu().numpy()
        target_array = edge_target.cpu().numpy()

        assignments = np.empty((batch_size, fine_count), dtype=np.int64)
        for batch_index in range(batch_size):
            parent = np.arange(fine_count, dtype=np.int64)
            rank = np.zeros(fine_count, dtype=np.int8)

            def find(node: int) -> int:
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = int(parent[node])
                return node

            components = fine_count
            order = np.argsort(
                -similarity_array[batch_index],
                kind="stable",
            )
            for edge_index in order:
                left = find(int(source_array[edge_index]))
                right = find(int(target_array[edge_index]))
                if left == right:
                    continue
                if rank[left] < rank[right]:
                    left, right = right, left
                parent[right] = left
                if rank[left] == rank[right]:
                    rank[left] += 1
                components -= 1
                if components == token_count:
                    break
            if components != token_count:
                raise AssertionError(
                    f"Expected {token_count} components, found {components}."
                )

            roots = np.asarray([find(node) for node in range(fine_count)])
            unique_roots = np.unique(roots)
            ordered_roots = sorted(
                unique_roots.tolist(),
                key=lambda root: float(np.flatnonzero(roots == root).mean()),
            )
            remap = {root: index for index, root in enumerate(ordered_roots)}
            assignments[batch_index] = np.asarray(
                [remap[int(root)] for root in roots],
                dtype=np.int64,
            )

    return torch.from_numpy(assignments).to(features.device)


def assignments_are_connected(
    assignment: np.ndarray,
    token_count: int,
    side: int = 16,
) -> bool:
    for group in range(token_count):
        nodes = set(np.flatnonzero(assignment == group).tolist())
        if not nodes:
            return False
        pending = [next(iter(nodes))]
        visited: set[int] = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            row, col = divmod(node, side)
            for next_row, next_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                neighbor = next_row * side + next_col
                if (
                    0 <= next_row < side
                    and 0 <= next_col < side
                    and neighbor in nodes
                    and neighbor not in visited
                ):
                    pending.append(neighbor)
        if visited != nodes:
            return False
    return True


class G2TMFixedKModel(CommonTokenClassifier):
    def __init__(self, dim: int, token_count: int, mask_guided: bool) -> None:
        super().__init__(dim, token_count)
        self.stem = PatchStem(dim)
        self.mask_guided = mask_guided
        self.mask_embedding = (
            nn.Sequential(
                nn.Linear(1, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            if mask_guided
            else None
        )
        source, target = grid_edges()
        self.register_buffer("edge_source", source, persistent=False)
        self.register_buffer("edge_target", target, persistent=False)
        self.register_buffer("coordinates", fine_coordinates(), persistent=False)

    def forward(
        self,
        image: torch.Tensor,
        mask_probability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.stem(image).flatten(2).transpose(1, 2)
        grouping_features = values
        if self.mask_guided:
            if mask_probability is None or self.mask_embedding is None:
                raise ValueError(
                    "g2tm_fixedk_mask requires a predicted mask probability."
                )
            mask_tokens = F.adaptive_avg_pool2d(
                mask_probability, (16, 16)
            ).flatten(2).transpose(1, 2)
            grouping_features = values + self.mask_embedding(mask_tokens)
            values = grouping_features

        assignment = maximum_spanning_forest_assignments(
            grouping_features,
            self.token_count,
            self.edge_source,
            self.edge_target,
        )
        one_hot = F.one_hot(
            assignment,
            num_classes=self.token_count,
        ).to(values.dtype)
        counts = one_hot.sum(dim=1).clamp_min(1.0)
        pooled = torch.bmm(one_hot.transpose(1, 2), values) / counts.unsqueeze(-1)
        coordinate_bank = self.coordinates.to(values.dtype).unsqueeze(0).expand(
            values.shape[0], -1, -1
        )
        centroids = (
            torch.bmm(one_hot.transpose(1, 2), coordinate_bank)
            / counts.unsqueeze(-1)
        )
        mass = (counts / float(values.shape[1])).unsqueeze(-1)
        metadata = torch.cat((centroids, mass), dim=-1)
        return self.classify(pooled, metadata)


def make_collision_model(name: str, dim: int, token_count: int) -> nn.Module:
    if name == "g2tm_fixedk_feature":
        return G2TMFixedKModel(dim, token_count, mask_guided=False)
    if name == "g2tm_fixedk_mask":
        return G2TMFixedKModel(dim, token_count, mask_guided=True)
    raise ValueError(name)


def run_self_test() -> None:
    torch.manual_seed(123)
    source, target = grid_edges()
    features = torch.randn(3, 256, 16)
    first = maximum_spanning_forest_assignments(
        features, 8, source, target
    )
    second = maximum_spanning_forest_assignments(
        features, 8, source, target
    )
    if not torch.equal(first, second):
        raise AssertionError("Fixed-K assignment is not deterministic.")
    for assignment in first.numpy():
        if np.unique(assignment).size != 8:
            raise AssertionError("Assignment does not contain exactly K groups.")
        if not assignments_are_connected(assignment, 8):
            raise AssertionError("A fixed-K group is disconnected.")

    image = torch.rand(2, 3, 64, 64)
    mask = torch.rand(2, 1, 64, 64)
    for name in COLLISION_MODELS:
        model = make_collision_model(name, 16, 8)
        logits = model(image, mask)
        if logits.shape != (2,):
            raise AssertionError((name, logits.shape))
        logits.square().mean().backward()
        if not all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        ):
            raise AssertionError(f"Missing gradient in {name}.")
    print("FIVES_G2TM_COLLISION_SELF_TEST_PASS", flush=True)


def main() -> None:
    experiment_started = time.time()
    args = parse_args()
    args.dataset = "fives"
    if args.self_test:
        run_self_test()
        return
    if args.mask_run_dir is None or args.output is None:
        raise ValueError("--mask-run-dir and --output are required.")
    if (
        args.crop_size != 64
        or args.output_size != 64
        or args.token_count != 8
    ):
        raise ValueError("The collision gate requires 64x64 crops and K=8.")

    mask_run_dir = args.mask_run_dir.resolve()
    prior = json.loads(
        (mask_run_dir / "result.json").read_text(encoding="utf-8")
    )
    if prior["status"] != "completed":
        raise RuntimeError("The referenced FIVES run is not complete.")
    if int(prior["optimization_seed"]) != args.seed:
        raise ValueError("Mask run and collision run optimization seeds differ.")
    if int(prior["reduced_token_count"]) != args.token_count:
        raise ValueError("Mask run and collision run token budgets differ.")

    arrays, manifest, _, fingerprint = load_all_arrays(args)
    if fingerprint != prior["dataset"]["fingerprint"]:
        raise RuntimeError("FIVES dataset fingerprint changed.")
    mask_path = mask_run_dir / "mask_probabilities.npz"
    with np.load(mask_path) as archive:
        probabilities = {
            split: archive[split][
                : arrays[split]["images"].shape[0]
            ].astype(np.float32)
            for split in ("train", "dev", "test")
        }
    datasets = {
        split: ReducerDataset(
            arrays[split],
            probabilities[split],
            augment=split == "train",
        )
        for split in ("train", "dev", "test")
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    truth_reference: np.ndarray | None = None

    original_factory = reducer_baselines.make_model
    reducer_baselines.make_model = make_collision_model
    try:
        for name in args.models:
            print(
                f"FIVES_COLLISION_START model={name} seed={args.seed}",
                flush=True,
            )
            summary, history, prediction, truth = reducer_baselines.train_one(
                name,
                args,
                datasets,
                output,
                device,
            )
            summary["uses_predicted_mask"] = name == "g2tm_fixedk_mask"
            summary["method_status"] = (
                "matched_fixed_budget_adaptation_of_official_g2tm"
            )
            if truth_reference is None:
                truth_reference = truth
            elif not np.array_equal(truth_reference, truth):
                raise RuntimeError("Held-out truth changed between models.")
            summaries[name] = summary
            histories.extend(history)
            predictions[name] = prediction
            print(
                f"FIVES_COLLISION_DONE model={name} seed={args.seed} "
                f"test_bal={summary['test_metrics']['balanced_accuracy']:.4f}",
                flush=True,
            )
    finally:
        reducer_baselines.make_model = original_factory

    if truth_reference is None:
        raise RuntimeError("No collision model was run.")
    write_csv(output / "history.csv", histories)
    np.savez_compressed(
        output / "heldout_predictions.npz",
        truth=truth_reference,
        source_id=arrays["test"]["source_id"],
        **predictions,
    )
    result = {
        "experiment_id": "fives_g2tm_collision_gate",
        "status": "completed",
        "dataset": {
            "name": "FIVES",
            "fingerprint": fingerprint,
            "official_source_counts": {
                split: len(values) for split, values in manifest.items()
            },
        },
        "optimization_seed": args.seed,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "fine_token_count": 256,
        "reduced_token_count": args.token_count,
        "models": list(args.models),
        "model_results": summaries,
        "mask_probabilities": {
            "path": str(mask_path),
            "sha256": file_sha256(mask_path),
        },
        "official_g2tm": {
            "repository": "https://github.com/vbercy/g2tm-segmenter",
            "branch": "torch2",
            "revision": OFFICIAL_G2TM_REVISION,
            "core_preserved": (
                "four-neighbour cosine-similarity graph and connected "
                "feature grouping with detached assignments"
            ),
            "adaptation": (
                "Kruskal maximum-spanning forest stops at exactly K connected "
                "components instead of selecting a global similarity threshold"
            ),
        },
        "test_was_previously_observed": True,
        "selection_uses_development_only": True,
        "inference_uses_ground_truth_mask": False,
        "protocol": str(
            (
                Path(__file__).parent
                / "FIVES_G2TM_COLLISION_PROTOCOL.md"
            ).resolve()
        ),
        "duration_seconds": time.time() - experiment_started,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

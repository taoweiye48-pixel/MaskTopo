"""Deterministic component-aligned coarsening (CAC) and coarse graphs.

Assignments are hard and non-differentiable. Mask probabilities, not annotation
masks, are the structural input at inference. Float16/float32 input values are
preserved because threshold rounding can change the graph.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
from scipy.ndimage import binary_closing, label


def patch_component_ids(component_map: np.ndarray) -> np.ndarray:
    if (
        component_map.ndim != 2
        or component_map.shape[0] != component_map.shape[1]
        or component_map.shape[0] % 16 != 0
    ):
        raise ValueError("Component crop must be square and divisible by 16.")
    block_size = component_map.shape[0] // 16
    patches = np.zeros((16, 16), dtype=np.int16)
    for row in range(16):
        for col in range(16):
            block = component_map[
                block_size * row : block_size * (row + 1),
                block_size * col : block_size * (col + 1),
            ]
            positive = block[block > 0]
            if positive.size:
                values, counts = np.unique(positive, return_counts=True)
                patches[row, col] = int(values[np.argmax(counts)])
    return patches


def coarse_graph8(
    assignment: np.ndarray,
    patch_ids: np.ndarray,
    token_count: int,
    topology_aware: bool,
) -> tuple[np.ndarray, np.ndarray]:
    adjacency = np.eye(token_count, dtype=np.uint8)
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(16):
        for col in range(16):
            for delta_row, delta_col in directions:
                new_row = row + delta_row
                new_col = col + delta_col
                if not (0 <= new_row < 16 and 0 <= new_col < 16):
                    continue
                if topology_aware:
                    first_id = int(patch_ids[row, col])
                    second_id = int(patch_ids[new_row, new_col])
                    valid = (first_id == 0 and second_id == 0) or (
                        first_id > 0 and first_id == second_id
                    )
                    if not valid:
                        continue
                first = int(assignment[row, col])
                second = int(assignment[new_row, new_col])
                adjacency[first, second] = 1
                adjacency[second, first] = 1
    reachability = adjacency.astype(bool)
    for pivot in range(token_count):
        reachability |= (
            reachability[:, pivot, None] & reachability[pivot, None, :]
        )
    return adjacency, reachability.astype(np.uint8)

def mask_groups(
    probability: np.ndarray,
    threshold: float,
    closing_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    binary = probability >= threshold
    if closing_iterations:
        binary = binary_closing(
            binary,
            structure=np.ones((3, 3), dtype=bool),
            iterations=closing_iterations,
        )
    component_map, _ = label(
        binary, structure=np.ones((3, 3), dtype=np.uint8)
    )
    patches = patch_component_ids(component_map)
    return component_map, patches

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

def allocate_cluster_counts(
    patch_ids: np.ndarray,
    token_count: int,
    critical_scores: np.ndarray | None,
) -> dict[int, int]:
    groups = [int(item) for item in np.unique(patch_ids)]
    if token_count < len(groups):
        raise ValueError("Token budget is smaller than the number of semantic groups.")
    sizes = {group: int(np.sum(patch_ids == group)) for group in groups}
    counts = {group: 1 for group in groups}
    weights = {}
    for group in groups:
        weight = float(sizes[group])
        if critical_scores is not None and group > 0:
            mask = patch_ids.reshape(-1) == group
            weight += 2.0 * float(np.count_nonzero(critical_scores[mask] > 0))
        weights[group] = weight
    while sum(counts.values()) < token_count:
        candidates = [
            group for group in groups if counts[group] < sizes[group]
        ]
        if not candidates:
            raise RuntimeError("Could not allocate all coarse tokens.")
        chosen = max(
            candidates,
            key=lambda group: (
                weights[group] / counts[group],
                sizes[group],
                -group,
            ),
        )
        counts[chosen] += 1
    return counts


def farthest_seeds(
    nodes: np.ndarray,
    count: int,
    critical_scores: np.ndarray | None,
) -> list[int]:
    flat_nodes = nodes[:, 0] * 16 + nodes[:, 1]
    selected: list[int] = []
    if critical_scores is not None:
        ranked = sorted(
            (int(item) for item in flat_nodes if critical_scores[int(item)] > 0),
            key=lambda item: (-float(critical_scores[item]), item),
        )
        critical_quota = min(len(ranked), max(1, count // 2))
        selected.extend(ranked[:critical_quota])
    if not selected:
        centroid = np.mean(nodes, axis=0)
        distances = np.sum((nodes - centroid) ** 2, axis=1)
        selected.append(int(flat_nodes[int(np.argmin(distances))]))
    candidate_rc = nodes.astype(np.float64)
    while len(selected) < count:
        selected_rc = np.array(
            [(item // 16, item % 16) for item in selected], dtype=np.float64
        )
        distances = np.min(
            np.sum(
                (candidate_rc[:, None, :] - selected_rc[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        for item in selected:
            position = np.flatnonzero(flat_nodes == item)
            if position.size:
                distances[position[0]] = -1.0
        selected.append(int(flat_nodes[int(np.argmax(distances))]))
    return selected


def assign_group(nodes: np.ndarray, seeds: list[int]) -> dict[int, int]:
    node_set = {int(row) * 16 + int(col) for row, col in nodes}
    labels = {node: -1 for node in node_set}
    distances = {node: 10**9 for node in node_set}
    queue: deque[int] = deque()
    for label, seed in enumerate(seeds):
        labels[seed] = label
        distances[seed] = 0
        queue.append(seed)
    while queue:
        node = queue.popleft()
        row, col = divmod(node, 16)
        for nr, nc in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            neighbor = nr * 16 + nc
            if 0 <= nr < 16 and 0 <= nc < 16 and neighbor in node_set:
                proposed = distances[node] + 1
                if proposed < distances[neighbor]:
                    distances[neighbor] = proposed
                    labels[neighbor] = labels[node]
                    queue.append(neighbor)
    seed_rc = np.array([(item // 16, item % 16) for item in seeds])
    for node, label in list(labels.items()):
        if label >= 0:
            continue
        point = np.array([node // 16, node % 16])
        labels[node] = int(np.argmin(np.sum((seed_rc - point) ** 2, axis=1)))
    return labels


def canonicalize_assignment(assignment: np.ndarray, token_count: int) -> np.ndarray:
    centroids = []
    for cluster in range(token_count):
        coordinates = np.argwhere(assignment == cluster)
        if coordinates.size == 0:
            raise AssertionError("Empty coarse cluster.")
        centroid = np.mean(coordinates, axis=0)
        centroids.append((float(centroid[0]), float(centroid[1]), cluster))
    order = [item[2] for item in sorted(centroids)]
    remap = {old: new for new, old in enumerate(order)}
    return np.vectorize(remap.__getitem__, otypes=[np.int16])(assignment)

def component_assignment(
    patch_ids: np.ndarray, token_count: int
) -> np.ndarray:
    scores = None
    counts = allocate_cluster_counts(patch_ids, token_count, scores)
    assignment = np.full((16, 16), -1, dtype=np.int16)
    next_cluster = 0
    for group in sorted(counts):
        nodes = np.argwhere(patch_ids == group)
        group_scores = None
        seeds = farthest_seeds(nodes, counts[group], group_scores)
        local = assign_group(nodes, seeds)
        for flat, local_label in local.items():
            assignment[flat // 16, flat % 16] = next_cluster + local_label
        next_cluster += counts[group]
    if np.any(assignment < 0) or next_cluster != token_count:
        raise AssertionError("Oracle coarsening did not cover the full grid.")
    return canonicalize_assignment(assignment, token_count)

@dataclass
class Topology:
    """Batched groups/assignments [B,16,16] and binary A/R [B,K,K]."""
    groups: np.ndarray
    assignment: np.ndarray
    adjacency: np.ndarray
    reachability: np.ndarray

    def tensors(self, device="cpu"):
        """Return assignment, adjacency, reachability for MaskTopo.forward."""
        return (
            torch.as_tensor(self.assignment, dtype=torch.long, device=device),
            torch.as_tensor(self.adjacency, dtype=torch.float32, device=device),
            torch.as_tensor(self.reachability, dtype=torch.float32, device=device),
        )


def build_topology(probabilities, token_count=8, threshold=0.5, closing_iterations=0):
    """Construct exactly K nonempty tokens per predicted 64x64 mask.

    probabilities is [64,64] or [B,64,64], with values in [0,1].
    Threshold/closing settings must be chosen using development data.
    Graph self-loops and spatial edges between background tokens are retained.
    """
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 2:
        probabilities = probabilities[None]
    if probabilities.ndim != 3 or probabilities.shape[1:] != (64, 64) or len(probabilities) == 0:
        raise ValueError("Expected a nonempty [B,64,64] batch or a [64,64] mask")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and within [0,1]")
    if isinstance(token_count, bool) or int(token_count) != token_count or not 1 <= token_count <= 256:
        raise ValueError("token_count must be an integer in [1,256]")
    if closing_iterations not in (0, 1, 2) or not 0 <= threshold <= 1:
        raise ValueError("Use a threshold in [0,1] and closing_iterations in {0,1,2}")
    token_count = int(token_count)
    groups, assignments, adjacency, reachability = [], [], [], []
    for probability in probabilities:
        _, group = mask_groups(probability, threshold, closing_iterations)
        group = limit_groups(group, token_count - 1)
        assigned = component_assignment(group, token_count)
        graph, closure = coarse_graph8(assigned, group, token_count, topology_aware=True)
        groups.append(group)
        assignments.append(assigned)
        adjacency.append(graph)
        reachability.append(closure)
    return Topology(*(np.stack(x) for x in (groups, assignments, adjacency, reachability)))

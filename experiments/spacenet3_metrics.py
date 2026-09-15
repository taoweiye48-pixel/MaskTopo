from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass
class SkeletonGraph:
    coordinates: np.ndarray
    matrix: csr_matrix
    degree: np.ndarray


def skeleton_graph(mask: np.ndarray) -> SkeletonGraph:
    skeleton = skeletonize(mask.astype(bool))
    coordinates = np.argwhere(skeleton).astype(np.float64)
    if coordinates.size == 0:
        return SkeletonGraph(coordinates.reshape(0, 2), csr_matrix((0, 0), dtype=np.float64), np.empty(0, dtype=np.int16))
    index = {tuple(int(value) for value in coordinate): position for position, coordinate in enumerate(coordinates)}
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for position, (row, column) in enumerate(coordinates.astype(np.int16)):
        for delta_row, delta_column in NEIGHBORS:
            other = index.get((int(row + delta_row), int(column + delta_column)))
            if other is None or other <= position:
                continue
            weight = math.sqrt(2.0) if delta_row and delta_column else 1.0
            rows.extend((position, other))
            columns.extend((other, position))
            weights.extend((weight, weight))
    matrix = csr_matrix((weights, (rows, columns)), shape=(coordinates.shape[0], coordinates.shape[0]))
    degree = np.diff(matrix.indptr).astype(np.int16)
    return SkeletonGraph(coordinates, matrix, degree)


def control_indices(graph: SkeletonGraph, maximum: int = 16) -> np.ndarray:
    count = graph.coordinates.shape[0]
    if count == 0:
        return np.empty(0, dtype=np.int64)
    critical = np.flatnonzero(graph.degree != 2).tolist()
    stride_candidates = np.linspace(0, count - 1, min(count, maximum), dtype=np.int64).tolist()
    candidates = list(dict.fromkeys(critical + stride_candidates))
    if len(candidates) <= maximum:
        return np.asarray(candidates, dtype=np.int64)
    selected = [candidates[0]]
    remaining = candidates[1:]
    while remaining and len(selected) < maximum:
        selected_coordinates = graph.coordinates[np.asarray(selected)]
        distances = [float(np.min(np.linalg.norm(selected_coordinates - graph.coordinates[item], axis=1))) for item in remaining]
        chosen_position = int(np.argmax(distances))
        selected.append(remaining.pop(chosen_position))
    return np.asarray(selected, dtype=np.int64)


def directed_path_score(reference: SkeletonGraph, proposal: SkeletonGraph, snap_tolerance: float = 3.0, minimum_path: float = 4.0) -> tuple[float, float, int] | None:
    controls = control_indices(reference)
    if controls.size < 2:
        return None
    reference_distances = dijkstra(reference.matrix, directed=False, indices=controls)
    if proposal.coordinates.shape[0] == 0:
        valid_pairs = sum(bool(np.isfinite(reference_distances[i, controls[j]]) and reference_distances[i, controls[j]] >= minimum_path) for i in range(controls.size) for j in range(i + 1, controls.size))
        return (0.0, 0.0, valid_pairs) if valid_pairs else None
    tree = cKDTree(proposal.coordinates)
    snap_distances, snapped = tree.query(reference.coordinates[controls], k=1)
    snapped = snapped.astype(np.int64)
    unique_snapped, inverse = np.unique(snapped, return_inverse=True)
    proposal_distances = dijkstra(proposal.matrix, directed=False, indices=unique_snapped)
    scores: list[float] = []
    connected: list[float] = []
    for first in range(controls.size):
        for second in range(first + 1, controls.size):
            reference_length = float(reference_distances[first, controls[second]])
            if not np.isfinite(reference_length) or reference_length < minimum_path:
                continue
            if snap_distances[first] > snap_tolerance or snap_distances[second] > snap_tolerance:
                scores.append(0.0)
                connected.append(0.0)
                continue
            proposal_length = float(proposal_distances[inverse[first], snapped[second]])
            if not np.isfinite(proposal_length):
                scores.append(0.0)
                connected.append(0.0)
                continue
            scores.append(1.0 - min(1.0, abs(proposal_length - reference_length) / max(reference_length, 1e-6)))
            connected.append(1.0)
    if not scores:
        return None
    return float(np.mean(scores)), float(np.mean(connected)), len(scores)


def raster_apls_proxy(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    target_graph = skeleton_graph(target)
    prediction_graph = skeleton_graph(prediction)
    forward = directed_path_score(target_graph, prediction_graph)
    reverse = directed_path_score(prediction_graph, target_graph)
    directional = [item[0] for item in (forward, reverse) if item is not None]
    symmetric = float(np.mean(directional)) if directional else float("nan")
    return {
        "raster_apls_proxy": symmetric,
        "gt_to_pred_path_score": float(forward[0]) if forward is not None else float("nan"),
        "pred_to_gt_path_score": float(reverse[0]) if reverse is not None else float("nan"),
        "path_recall": float(forward[1]) if forward is not None else float("nan"),
        "disconnection_rate": float(1.0 - forward[1]) if forward is not None else float("nan"),
        "gt_path_pair_count": int(forward[2]) if forward is not None else 0,
        "pred_path_pair_count": int(reverse[2]) if reverse is not None else 0,
    }


def cldice(target: np.ndarray, prediction: np.ndarray) -> float:
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    target_skeleton = skeletonize(target)
    prediction_skeleton = skeletonize(prediction)
    if not target_skeleton.any() and not prediction_skeleton.any():
        return float("nan")
    topology_precision = float((prediction_skeleton & target).sum()) / max(int(prediction_skeleton.sum()), 1)
    topology_sensitivity = float((target_skeleton & prediction).sum()) / max(int(target_skeleton.sum()), 1)
    return 2.0 * topology_precision * topology_sensitivity / max(topology_precision + topology_sensitivity, 1e-12)


def binary_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    true_positive = int((target & prediction).sum())
    false_positive = int((~target & prediction).sum())
    false_negative = int((target & ~prediction).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else float("nan")


def soft_dice(target: np.ndarray, probability: np.ndarray) -> float:
    target = target.astype(np.float64)
    probability = probability.astype(np.float64)
    return float((2.0 * np.sum(target * probability) + 1.0) / (np.sum(target) + np.sum(probability) + 1.0))


def sample_metrics(target: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    prediction = probability >= threshold
    return {"road_f1": binary_f1(target, prediction), "soft_dice": soft_dice(target, probability), "cldice": cldice(target, prediction), **raster_apls_proxy(target, prediction)}


def run_self_test() -> None:
    line = np.zeros((64, 64), dtype=np.uint8)
    line[32, 8:56] = 1
    same = raster_apls_proxy(line, line)
    assert abs(float(same["raster_apls_proxy"]) - 1.0) < 1e-9
    shifted = np.zeros_like(line)
    shifted[34, 8:56] = 1
    assert float(raster_apls_proxy(line, shifted)["raster_apls_proxy"]) > 0.95
    broken = line.copy()
    broken[32, 30:35] = 0
    assert float(raster_apls_proxy(line, broken)["raster_apls_proxy"]) < 0.9
    empty = np.zeros_like(line)
    assert np.isnan(float(raster_apls_proxy(empty, empty)["raster_apls_proxy"]))
    assert cldice(line, line) == 1.0 and binary_f1(line, line) == 1.0
    print("SPACENET3_METRICS_SELF_TEST_PASS")


if __name__ == "__main__":
    run_self_test()

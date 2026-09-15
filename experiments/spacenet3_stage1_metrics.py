from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from crackforest_real_gate import patch_component_ids
from spacenet3_metrics import SkeletonGraph, binary_f1, cldice, raster_apls_proxy, skeleton_graph


THRESHOLDS = tuple(round(index * 0.05, 2) for index in range(1, 20))
BOOTSTRAP_SEED = 20260851
BOOTSTRAP_REPLICATES = 20_000
STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


def patch_ids(mask: np.ndarray) -> np.ndarray:
    component_map = label(mask.astype(bool), structure=STRUCTURE_8)[0]
    return patch_component_ids(component_map)


def kappa_from_occupancy(pred_ids_list: list[np.ndarray], gt_ids_list: list[np.ndarray]) -> float:
    if not pred_ids_list or len(pred_ids_list) != len(gt_ids_list):
        return float("nan")
    prediction = np.concatenate([(item > 0).ravel() for item in pred_ids_list])
    target = np.concatenate([(item > 0).ravel() for item in gt_ids_list])
    agreement = float(np.mean(prediction == target))
    prediction_rate = float(np.mean(prediction))
    target_rate = float(np.mean(target))
    chance = prediction_rate * target_rate + (1 - prediction_rate) * (1 - target_rate)
    return (agreement - chance) / (1 - chance) if chance < 1 else float("nan")


def fixed_spatial_edges(ids: np.ndarray) -> np.ndarray:
    if ids.shape != (16, 16):
        raise ValueError(f"Expected 16x16 patch ids, got {ids.shape}")
    edges: list[bool] = []
    for row in range(16):
        for column in range(16):
            for delta_row, delta_column in ((0, 1), (1, 0), (1, 1), (1, -1)):
                next_row, next_column = row + delta_row, column + delta_column
                if 0 <= next_row < 16 and 0 <= next_column < 16:
                    first, second = int(ids[row, column]), int(ids[next_row, next_column])
                    edges.append(first > 0 and first == second)
    return np.asarray(edges, dtype=bool)


def fixed_edge_iou(pred_ids: np.ndarray, gt_ids: np.ndarray) -> float:
    prediction = fixed_spatial_edges(pred_ids)
    target = fixed_spatial_edges(gt_ids)
    if not target.any():
        return float("nan")
    union = np.logical_or(prediction, target)
    return float(np.logical_and(prediction, target).sum() / union.sum())


def patch_foreground_fraction(mask: np.ndarray) -> float:
    return float(mask.reshape(16, 16, 16, 16).max(axis=(1, 3)).mean())


def limit_groups_diagnostics(mask: np.ndarray, maximum_foreground: int = 15) -> dict[str, float | int]:
    component_map, component_count = label(mask.astype(bool), structure=STRUCTURE_8)
    ids = patch_component_ids(component_map)
    represented = [int(item) for item in np.unique(ids) if int(item) > 0]
    sizes = {item: int(np.sum(ids == item)) for item in represented}
    keep = set(sorted(represented, key=lambda item: (sizes[item], -item), reverse=True)[:maximum_foreground])
    dropped = set(range(1, int(component_count) + 1)) - keep
    foreground_pixels = int((component_map > 0).sum())
    dropped_pixels = int(np.isin(component_map, list(dropped)).sum()) if dropped else 0
    return {
        "component_count": int(component_count),
        "dropped_component_count": len(dropped),
        "dropped_pixel_fraction": dropped_pixels / foreground_pixels if foreground_pixels else 0.0,
    }


def nanmean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(finite.mean()) if finite.size else float("nan")


def bootstrap_summary(values: list[float], seed: int = BOOTSTRAP_SEED, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, float | int]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "bootstrap_sd": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "valid_sources": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        stop = min(replicates, start + chunk)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "bootstrap_sd": float(means.std(ddof=1)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "valid_sources": int(array.size),
    }


def select_threshold(targets: np.ndarray, probabilities: np.ndarray, source_ids: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    unique_sources = np.unique(source_ids)
    for threshold in THRESHOLDS:
        source_values: list[float] = []
        valid_crops = 0
        for source in unique_sources:
            indices = np.flatnonzero(source_ids == source)
            crop_values: list[float] = []
            for index in indices:
                if not targets[index].any():
                    continue
                crop_values.append(binary_f1(targets[index], probabilities[index] >= threshold))
            finite = [item for item in crop_values if np.isfinite(item)]
            if finite:
                source_values.append(float(np.mean(finite)))
                valid_crops += len(finite)
        rows.append(
            {
                "threshold": threshold,
                "source_mean_road_f1": float(np.mean(source_values)) if source_values else float("nan"),
                "valid_sources": len(source_values),
                "valid_crops": valid_crops,
            }
        )
    best = max(rows, key=lambda row: (float(row["source_mean_road_f1"]), float(row["threshold"])))
    return float(best["threshold"]), rows


def gate_a_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    source_ids: np.ndarray,
    crop_rows: np.ndarray,
    crop_columns: np.ndarray,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    crop_output: list[dict[str, Any]] = []
    source_output: list[dict[str, Any]] = []
    for source in np.unique(source_ids):
        indices = np.flatnonzero(source_ids == source)
        source_f1: list[float] = []
        source_cldice: list[float] = []
        source_edges: list[float] = []
        pred_ids_nonempty: list[np.ndarray] = []
        gt_ids_nonempty: list[np.ndarray] = []
        empty_pixel_fp: list[float] = []
        empty_patch_fp: list[float] = []
        dropped_components: list[float] = []
        dropped_pixels: list[float] = []
        for index in indices:
            target = targets[index].astype(np.uint8)
            prediction = (probabilities[index] >= threshold).astype(np.uint8)
            gt_nonempty = bool(target.any())
            prediction_ids = patch_ids(prediction)
            target_ids = patch_ids(target)
            road_f1 = binary_f1(target, prediction) if gt_nonempty else float("nan")
            crop_cldice = cldice(target, prediction) if gt_nonempty else float("nan")
            edge_iou = fixed_edge_iou(prediction_ids, target_ids)
            drop = limit_groups_diagnostics(prediction)
            if np.isfinite(road_f1):
                source_f1.append(float(road_f1))
            if np.isfinite(crop_cldice):
                source_cldice.append(float(crop_cldice))
            if np.isfinite(edge_iou):
                source_edges.append(float(edge_iou))
            if gt_nonempty:
                pred_ids_nonempty.append(prediction_ids)
                gt_ids_nonempty.append(target_ids)
            else:
                empty_pixel_fp.append(float(prediction.mean()))
                empty_patch_fp.append(patch_foreground_fraction(prediction))
            dropped_components.append(float(drop["dropped_component_count"]))
            dropped_pixels.append(float(drop["dropped_pixel_fraction"]))
            crop_output.append(
                {
                    "source_id": str(source),
                    "crop_row": int(crop_rows[index]),
                    "crop_column": int(crop_columns[index]),
                    "gt_nonempty": gt_nonempty,
                    "road_f1": road_f1,
                    "cldice": crop_cldice,
                    "fixed_spatial_edge_iou": edge_iou,
                    "predicted_foreground_fraction": float(prediction.mean()),
                    "target_foreground_fraction": float(target.mean()),
                    "predicted_patch_foreground_fraction": patch_foreground_fraction(prediction),
                    **drop,
                }
            )
        predictions = (probabilities[indices] >= threshold).astype(np.uint8)
        current_targets = targets[indices].astype(np.uint8)
        source_output.append(
            {
                "source_id": str(source),
                "road_f1": nanmean(source_f1),
                "cldice": nanmean(source_cldice),
                "fixed_spatial_edge_iou": nanmean(source_edges),
                "patch_kappa": kappa_from_occupancy(pred_ids_nonempty, gt_ids_nonempty),
                "predicted_foreground_fraction": float(predictions.mean()),
                "target_foreground_fraction": float(current_targets.mean()),
                "empty_crop_predicted_pixel_fraction": nanmean(empty_pixel_fp),
                "empty_crop_predicted_patch_fraction": nanmean(empty_patch_fp),
                "dropped_component_count_mean": nanmean(dropped_components),
                "dropped_pixel_fraction_mean": nanmean(dropped_pixels),
                "valid_nonempty_crops": len(source_f1),
                "valid_edge_crops": len(source_edges),
                "empty_crops": len(empty_pixel_fp),
            }
        )
    fields = {
        "A1_road_f1": "road_f1",
        "A2_fixed_spatial_edge_iou": "fixed_spatial_edge_iou",
        "A3_patch_kappa": "patch_kappa",
        "A4_empty_crop_predicted_pixel_fraction": "empty_crop_predicted_pixel_fraction",
    }
    summaries = {
        gate: bootstrap_summary([float(row[field]) for row in source_output], seed=BOOTSTRAP_SEED + offset)
        for offset, (gate, field) in enumerate(fields.items())
    }
    point = {gate: float(summary["mean"]) for gate, summary in summaries.items()}
    requirements = {
        "A1_road_f1_at_least_0_45": bool(point["A1_road_f1"] >= 0.45),
        "A2_fixed_spatial_edge_iou_at_least_0_60": bool(point["A2_fixed_spatial_edge_iou"] >= 0.60),
        "A3_patch_kappa_at_least_0_50": bool(point["A3_patch_kappa"] >= 0.50),
        "A4_empty_crop_predicted_pixel_fraction_at_most_0_01": bool(point["A4_empty_crop_predicted_pixel_fraction"] <= 0.01),
        "A2_effective_sources_at_least_24": bool(int(summaries["A2_fixed_spatial_edge_iou"]["valid_sources"]) >= 24),
        "A3_effective_sources_at_least_24": bool(int(summaries["A3_patch_kappa"]["valid_sources"]) >= 24),
    }
    summary = {
        "threshold": threshold,
        "source_count": len(source_output),
        "crop_count": len(crop_output),
        "metrics": summaries,
        "requirements": requirements,
        "verdict": "PASS" if all(requirements.values()) else "FAIL",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    return crop_output, source_output, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stitch_mosaic(values: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    if values.shape[0] != 25 or values.shape[1:] != (256, 256):
        raise ValueError(f"Expected 25x256x256 values, got {values.shape}")
    mosaic = np.zeros((1280, 1280), dtype=values.dtype)
    seen: set[tuple[int, int]] = set()
    for value, row, column in zip(values, rows, columns, strict=True):
        location = (int(row), int(column))
        if location in seen or location[0] not in (0, 256, 512, 768, 1024) or location[1] not in (0, 256, 512, 768, 1024):
            raise RuntimeError(f"Invalid or duplicate mosaic window: {location}")
        seen.add(location)
        mosaic[location[0] : location[0] + 256, location[1] : location[1] + 256] = value
    if len(seen) != 25:
        raise RuntimeError(f"Expected 25 mosaic windows, found {len(seen)}")
    return mosaic


def directed_path_score(
    reference: SkeletonGraph,
    proposal: SkeletonGraph,
    maximum_controls: int = 64,
    snap_tolerance: float = 3.0,
    minimum_path: float = 16.0,
) -> tuple[float, float, int] | None:
    from spacenet3_metrics import control_indices

    controls = control_indices(reference, maximum=maximum_controls)
    if controls.size < 2:
        return None
    reference_distances = dijkstra(reference.matrix, directed=False, indices=controls)
    if proposal.coordinates.shape[0] == 0:
        valid_pairs = sum(
            bool(np.isfinite(reference_distances[first, controls[second]]) and reference_distances[first, controls[second]] >= minimum_path)
            for first in range(controls.size)
            for second in range(first + 1, controls.size)
        )
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


def source_mosaic_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    if target.shape != (1280, 1280) or prediction.shape != target.shape:
        raise ValueError(f"Expected aligned 1280x1280 mosaics, got {target.shape}, {prediction.shape}")
    target_graph = skeleton_graph(target)
    prediction_graph = skeleton_graph(prediction)
    if target_graph.coordinates.shape[0] == 0 and prediction_graph.coordinates.shape[0] == 0:
        return {
            "bidirectional_raster_path_quality": float("nan"),
            "gt_to_pred_path_score": float("nan"),
            "pred_to_gt_path_score": float("nan"),
            "path_recall": float("nan"),
            "disconnection_rate": float("nan"),
            "gt_path_pair_count": 0,
            "pred_path_pair_count": 0,
        }
    forward = directed_path_score(target_graph, prediction_graph)
    reverse = directed_path_score(prediction_graph, target_graph)
    forward_score = float(forward[0]) if forward is not None and np.isfinite(forward[0]) else 0.0
    reverse_score = float(reverse[0]) if reverse is not None and np.isfinite(reverse[0]) else 0.0
    harmonic = 2 * forward_score * reverse_score / (forward_score + reverse_score) if forward_score + reverse_score else 0.0
    recall = float(forward[1]) if forward is not None and np.isfinite(forward[1]) else 0.0
    return {
        "bidirectional_raster_path_quality": harmonic,
        "gt_to_pred_path_score": forward_score,
        "pred_to_gt_path_score": reverse_score,
        "path_recall": recall,
        "disconnection_rate": 1.0 - recall,
        "gt_path_pair_count": int(forward[2]) if forward is not None else 0,
        "pred_path_pair_count": int(reverse[2]) if reverse is not None else 0,
    }


def run_self_test() -> None:
    perfect = np.zeros((256, 256), dtype=np.uint8)
    perfect[120:136, 20:236] = 1
    perfect_ids = patch_ids(perfect)
    assert abs(fixed_edge_iou(perfect_ids, perfect_ids) - 1.0) < 1e-12
    assert abs(kappa_from_occupancy([perfect_ids], [perfect_ids]) - 1.0) < 1e-12
    permuted = perfect_ids.copy()
    permuted[permuted > 0] += 7
    assert abs(fixed_edge_iou(permuted, perfect_ids) - 1.0) < 1e-12
    broken = perfect.copy()
    broken[:, 124:132] = 0
    assert fixed_edge_iou(patch_ids(broken), perfect_ids) < 1.0
    empty_ids = patch_ids(np.zeros_like(perfect))
    assert np.isnan(fixed_edge_iou(empty_ids, empty_ids))

    values = np.stack([np.full((256, 256), index, dtype=np.int16) for index in range(25)])
    rows = np.asarray([row for row in (0, 256, 512, 768, 1024) for _ in range(5)])
    columns = np.asarray([column for _ in range(5) for column in (0, 256, 512, 768, 1024)])
    mosaic = stitch_mosaic(values, rows, columns)
    assert mosaic.shape == (1280, 1280) and mosaic[0, 0] == 0 and mosaic[-1, -1] == 24

    line = np.zeros((1280, 1280), dtype=np.uint8)
    line[640, 100:1180] = 1
    same = source_mosaic_metrics(line, line)
    assert abs(float(same["bidirectional_raster_path_quality"]) - 1.0) < 1e-12
    line_broken = line.copy()
    line_broken[640, 620:660] = 0
    assert float(source_mosaic_metrics(line, line_broken)["bidirectional_raster_path_quality"]) < 1.0
    assert float(source_mosaic_metrics(line, np.zeros_like(line))["bidirectional_raster_path_quality"]) == 0.0
    double_empty = source_mosaic_metrics(np.zeros_like(line), np.zeros_like(line))
    assert np.isnan(float(double_empty["bidirectional_raster_path_quality"]))

    counterexample = np.zeros((64, 64), np.uint8)
    counterexample[16, 5:30] = 1
    counterexample[48, 34:59] = 1
    false_link = counterexample.copy()
    false_link[16:49, 29:35] = 1
    old = raster_apls_proxy(counterexample, false_link)
    forward = float(old["gt_to_pred_path_score"])
    reverse = float(old["pred_to_gt_path_score"])
    harmonic = 2 * forward * reverse / (forward + reverse)
    assert old["path_recall"] == 1.0 and harmonic < 1.0
    print(
        {
            "status": "SPACENET3_STAGE1_METRICS_SELF_TEST_PASS",
            "perfect": same,
            "counterexample_path_recall": old["path_recall"],
            "counterexample_bidirectional_raster_path_quality": harmonic,
            "mosaic_shape": mosaic.shape,
            "unused_source_border": 20,
        },
        flush=True,
    )


if __name__ == "__main__":
    run_self_test()

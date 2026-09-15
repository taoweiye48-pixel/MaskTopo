from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from spacenet3_metrics import binary_f1, sample_metrics


THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
MODELS = ("mask_only", "grid", "tokenlearner", "perceiver_resampler", "tome_style", "mask_guided_queries", "ph_only", "ph_guided", "mask_assignment_identity", "shuffled_topology", "mask_topo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose dev threshold recoverability after TB027.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_mean(values: np.ndarray, source_ids: np.ndarray) -> float:
    means = []
    for source in sorted(set(str(item) for item in source_ids)):
        selected = values[source_ids.astype(str) == source]
        finite = selected[np.isfinite(selected)]
        means.append(float(np.mean(finite)) if finite.size else float("nan"))
    return float(np.nanmean(means))


def select_threshold(target: np.ndarray, probability: np.ndarray, source_ids: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for threshold in THRESHOLDS:
        values = np.asarray([binary_f1(target[index], probability[index] >= threshold) for index in range(target.shape[0])], dtype=np.float64)
        rows.append({"threshold": threshold, "source_mean_road_f1": source_mean(values, source_ids), "foreground_fraction": float((probability >= threshold).mean()), "valid_f1_crop_count": float(np.isfinite(values).sum())})
    selected = max(rows, key=lambda row: (row["source_mean_road_f1"], row["threshold"]))
    return float(selected["threshold"]), rows


def graph_metrics(target: np.ndarray, probability: np.ndarray, source_ids: np.ndarray, threshold: float, model: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for index in range(target.shape[0]):
        rows.append({"model": model, "index": index, "source_id": str(source_ids[index]), "threshold": threshold, **sample_metrics(target[index], probability[index], threshold)})
        if (index + 1) % 200 == 0 or index + 1 == target.shape[0]:
            print(f"DIAGNOSTIC_METRICS model={model} progress={index + 1}/{target.shape[0]}", flush=True)
    names = ("road_f1", "soft_dice", "cldice", "raster_apls_proxy", "path_recall", "disconnection_rate")
    summary = {}
    for name in names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        summary[name] = source_mean(values, source_ids)
    summary["threshold"] = threshold
    summary["foreground_fraction"] = float((probability >= threshold).mean())
    summary["valid_apls_crop_count"] = float(sum(np.isfinite(float(row["raster_apls_proxy"])) for row in rows))
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.self_test:
        target = np.zeros((2, 8, 8), dtype=np.uint8)
        target[:, 3, 1:7] = 1
        probability = np.full((2, 8, 8), 0.55, dtype=np.float32)
        probability[target > 0] = 0.9
        source_ids = np.asarray(["a", "b"])
        selected, _ = select_threshold(target, probability, source_ids)
        assert selected > 0.55
        print("SPACENET3_THRESHOLD_DIAGNOSTIC_SELF_TEST_PASS")
        return
    run_dir = args.run_dir.resolve()
    dev_cache = args.dev_cache.resolve()
    protocol = args.protocol.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    parent = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if parent["gate"]["verdict"] != "FAIL" or parent["gate"]["heldout_content_access_authorized"]:
        raise RuntimeError("Expected failed TB027 gate with heldout sealed")
    with np.load(dev_cache, allow_pickle=False) as archive:
        target = archive["masks"].astype(np.uint8)
        source_ids = archive["source_ids"].astype(str)
    freeze = {"experiment_id": "TB-B-260803-028", "status": "frozen_before_execution", "parent_result_sha256": sha256_file(run_dir / "result.json"), "dev_cache_sha256": sha256_file(dev_cache), "protocol_sha256": sha256_file(protocol), "code_sha256": sha256_file(Path(__file__).resolve()), "thresholds": THRESHOLDS, "heldout_content_accessed": False}
    with (output / "run_freeze.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(freeze, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    sweep_rows = []
    selected_results = {}
    all_metric_rows = []
    for model in MODELS:
        prediction_path = run_dir / f"dev_predictions_{model}_seed20260830.npz"
        with np.load(prediction_path, allow_pickle=False) as archive:
            probability = archive["probability"].astype(np.float32)
        selected, rows = select_threshold(target, probability, source_ids)
        for row in rows:
            sweep_rows.append({"model": model, **row})
        summary, metric_rows = graph_metrics(target, probability, source_ids, selected, model)
        summary["prediction_path"] = str(prediction_path)
        summary["prediction_sha256"] = sha256_file(prediction_path)
        summary["fixed_0_5_road_f1"] = parent["mask_only_dev_metrics"]["road_f1"] if model == "mask_only" else parent["models"][model]["dev_metrics"]["road_f1"]
        summary["fixed_0_5_apls"] = parent["mask_only_dev_metrics"]["raster_apls_proxy"] if model == "mask_only" else parent["models"][model]["dev_metrics"]["raster_apls_proxy"]
        selected_results[model] = summary
        all_metric_rows.extend(metric_rows)
        print(f"THRESHOLD_SELECTED model={model} threshold={selected} road_f1={summary['road_f1']:.4f} apls={summary['raster_apls_proxy']:.4f}", flush=True)
    write_csv(output / "threshold_sweep.csv", sweep_rows)
    write_csv(output / "selected_crop_metrics.csv", all_metric_rows)
    mask = selected_results["mask_only"]
    requirements = {"road_f1_gain_at_least_0_05": mask["road_f1"] - mask["fixed_0_5_road_f1"] >= 0.05, "apls_gain_at_least_0_03": mask["raster_apls_proxy"] - mask["fixed_0_5_apls"] >= 0.03, "foreground_fraction_at_most_0_10": mask["foreground_fraction"] <= 0.10}
    verdict = "CALIBRATION_MATERIAL" if all(requirements.values()) else "RETRAIN_MASK_REQUIRED"
    result = {"experiment_id": "TB-B-260803-028", "status": "completed_diagnostic", "verdict": verdict, "requirements": requirements, "parent_gate_remains": "FAIL", "heldout_content_accessed": False, "selected": selected_results, "run_freeze_sha256": sha256_file(output / "run_freeze.json")}
    result_path = output / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "verdict": verdict, "requirements": requirements, "mask_only": mask, "result": str(result_path), "sha256": sha256_file(result_path)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from strong_reducer_baselines import REDUCERS
from summarize_crackforest_mechanism_ablation import balanced_accuracy
from summarize_strong_reducer_baselines import cluster_bootstrap


SEEDS = (20260810, 20260811, 20260812)
DATASETS = ("fives", "deepcrack", "crackforest")
PH_MODELS = ("ph_only", "ph_guided")
TOKEN_COUNTS = {"fives": 8, "deepcrack": 16, "crackforest": 16}
MASK_KEYS = {
    "fives": "mask_topo_external",
    "deepcrack": "mask_topo_external",
    "crackforest": "mask_topo_recheck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize TB-B-260802-022 PH direct topology baselines."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("results_ph_primary_summary")
    )
    parser.add_argument("--crackforest-cache", type=Path, default=Path("real_cache"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--statistics-seed", type=int, default=20260802)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_paths(root: Path, dataset: str, seed: int) -> tuple[Path, Path, Path]:
    if dataset == "fives":
        old = root / f"results_fives_seed{seed}"
        strong = old
        ph = root / f"results_ph_fives_k8_seed{seed}"
    elif dataset == "deepcrack":
        old = root / f"results_deepcrack_seed{seed}"
        strong = root / f"results_strong_reducer_deepcrack_seed{seed}"
        ph = root / f"results_ph_deepcrack_k16_seed{seed}"
    else:
        old = root / f"results_crackforest_mechanism_seed{seed}"
        strong = root / f"results_strong_reducer_crackforest_seed{seed}"
        suffix = "_retry1" if seed == 20260810 else ""
        ph = root / f"results_ph_crackforest_k16_seed{seed}{suffix}"
    return old, strong, ph


def crackforest_sources(cache: Path) -> np.ndarray:
    candidates = sorted(
        cache.glob("crackforest_test_n320_seed22260810_crop64_*.npz")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one CrackForest test cache, found {len(candidates)}."
        )
    with np.load(candidates[0]) as archive:
        return archive["source_id"].astype(str)


def score_summary(values: list[float]) -> dict[str, Any]:
    scores = np.asarray(values, dtype=np.float64)
    return {
        "scores_by_seed": scores.tolist(),
        "mean_balanced_accuracy": float(scores.mean()),
        "sample_std_balanced_accuracy": float(scores.std(ddof=1)),
    }


def load_dataset(
    root: Path,
    dataset: str,
    seeds: list[int],
    crackforest_cache: Path,
    bootstrap_repetitions: int,
    statistics_seed: int,
) -> dict[str, Any]:
    model_names = ("mask_topo", *PH_MODELS, *REDUCERS)
    prediction_lists: dict[str, list[np.ndarray]] = {
        name: [] for name in model_names
    }
    scores: dict[str, list[float]] = {name: [] for name in model_names}
    truth_reference: np.ndarray | None = None
    source_reference: np.ndarray | None = None
    run_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    ph_latency: list[float] = []
    ph_h0: list[float] = []
    ph_h1: list[float] = []
    ph_padding: list[float] = []
    for seed in seeds:
        old_root, strong_root, ph_root = run_paths(root, dataset, seed)
        old_result = json.loads(
            (old_root / "result.json").read_text(encoding="utf-8")
        )
        strong_result = (
            old_result
            if strong_root == old_root
            else json.loads((strong_root / "result.json").read_text(encoding="utf-8"))
        )
        ph_result = json.loads((ph_root / "result.json").read_text(encoding="utf-8"))
        if ph_result["status"] != "completed":
            raise RuntimeError(f"PH run not completed: {dataset}, {seed}")
        if ph_result["confirmatory_status"] != "retrospective_collision":
            raise RuntimeError(f"PH confirmatory status changed: {dataset}, {seed}")
        if ph_result["massachusetts_test_reopened"]:
            raise RuntimeError("A PH run reports reopening Massachusetts test.")
        if int(ph_result["reduced_token_count"]) != TOKEN_COUNTS[dataset]:
            raise RuntimeError(f"PH token count changed: {dataset}, {seed}")
        with np.load(old_root / "heldout_predictions.npz") as archive:
            truth = archive["truth"].astype(np.uint8)
            mask_prediction = archive[MASK_KEYS[dataset]].astype(np.uint8)
            old_source = (
                archive["source_id"].astype(str)
                if "source_id" in archive.files
                else None
            )
            fives_reducers = (
                {name: archive[name].astype(np.uint8) for name in REDUCERS}
                if dataset == "fives"
                else None
            )
        with np.load(ph_root / "heldout_predictions.npz") as archive:
            if not np.array_equal(truth, archive["truth"].astype(np.uint8)):
                raise RuntimeError(f"PH truth mismatch: {dataset}, {seed}")
            ph_predictions = {
                name: archive[name].astype(np.uint8) for name in PH_MODELS
            }
            ph_source = (
                archive["source_id"].astype(str)
                if "source_id" in archive.files
                else None
            )
        if dataset == "fives":
            if fives_reducers is None:
                raise AssertionError("FIVES reducers were not loaded.")
            reducer_predictions = fives_reducers
            strong_prediction_path = old_root / "heldout_predictions.npz"
        else:
            with np.load(strong_root / "heldout_predictions.npz") as archive:
                if not np.array_equal(truth, archive["truth"].astype(np.uint8)):
                    raise RuntimeError(f"Reducer truth mismatch: {dataset}, {seed}")
                reducer_predictions = {
                    name: archive[name].astype(np.uint8) for name in REDUCERS
                }
                strong_source = (
                    archive["source_id"].astype(str)
                    if "source_id" in archive.files
                    else None
                )
            strong_prediction_path = strong_root / "heldout_predictions.npz"
            if dataset == "deepcrack" and (
                old_source is None
                or strong_source is None
                or not np.array_equal(old_source, strong_source)
            ):
                raise RuntimeError("DeepCrack old/strong source IDs differ.")
        if dataset == "crackforest":
            sources = crackforest_sources(crackforest_cache)
        else:
            if old_source is None:
                raise RuntimeError(f"Source IDs missing for {dataset}.")
            sources = old_source
        if ph_source is not None and not np.array_equal(sources, ph_source):
            raise RuntimeError(f"PH source IDs differ: {dataset}, {seed}")
        if truth_reference is None:
            truth_reference = truth
            source_reference = sources
        elif not np.array_equal(truth_reference, truth) or not np.array_equal(
            source_reference, sources
        ):
            raise RuntimeError(f"Held-out samples changed across seeds: {dataset}")
        current = {
            "mask_topo": mask_prediction,
            **ph_predictions,
            **reducer_predictions,
        }
        result_sources = {
            "mask_topo": old_result["model_results"][MASK_KEYS[dataset]],
            **ph_result["model_results"],
            **strong_result["model_results"],
        }
        for name, prediction in current.items():
            score = balanced_accuracy(truth, prediction)
            recorded = result_sources[name]["test_metrics"]["balanced_accuracy"]
            if not np.isclose(score, recorded):
                raise RuntimeError(
                    f"Prediction/metric mismatch: {dataset}, {seed}, {name}"
                )
            prediction_lists[name].append(prediction)
            scores[name].append(score)
            run_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model": name,
                    "balanced_accuracy": score,
                    "parameters": result_sources[name]["parameters"],
                }
            )
        diagnostic = ph_result["ph_diagnostics_at_k"]["test"]
        ph_h0.append(float(diagnostic["selected_h0_mean"]))
        ph_h1.append(float(diagnostic["selected_h1_mean"]))
        ph_padding.append(float(diagnostic["padding_rate"]))
        ph_latency.append(
            float(ph_result["ph_preprocessing_latency"]["1"]["latency_ms_per_sample"])
        )
        artifact_entry = {
            "dataset": dataset,
            "seed": seed,
            "ph_result_path": str(ph_root / "result.json"),
            "ph_result_sha256": file_sha256(ph_root / "result.json"),
            "ph_prediction_path": str(ph_root / "heldout_predictions.npz"),
            "ph_prediction_sha256": file_sha256(
                ph_root / "heldout_predictions.npz"
            ),
            "prior_prediction_path": str(old_root / "heldout_predictions.npz"),
            "prior_prediction_sha256": file_sha256(
                old_root / "heldout_predictions.npz"
            ),
            "strong_prediction_path": str(strong_prediction_path),
            "strong_prediction_sha256": file_sha256(strong_prediction_path),
        }
        for name in PH_MODELS:
            checkpoint = ph_root / f"{name}_seed{seed}.pt"
            artifact_entry[f"{name}_checkpoint_sha256"] = file_sha256(checkpoint)
        artifacts.append(artifact_entry)
    if truth_reference is None or source_reference is None:
        raise RuntimeError(f"No runs loaded for {dataset}.")
    predictions = {
        name: np.stack(values) for name, values in prediction_lists.items()
    }
    aggregates = {
        name: score_summary(values) for name, values in scores.items()
    }
    strongest_reducer = max(
        REDUCERS, key=lambda name: aggregates[name]["mean_balanced_accuracy"]
    )
    best_ph = max(
        PH_MODELS, key=lambda name: aggregates[name]["mean_balanced_accuracy"]
    )
    rng = np.random.default_rng(1000 * DATASETS.index(dataset) + statistics_seed)
    contrasts = {
        "mask_topo_minus_ph_guided": cluster_bootstrap(
            truth_reference,
            source_reference,
            predictions["mask_topo"],
            predictions["ph_guided"],
            bootstrap_repetitions,
            rng,
        ),
        "mask_topo_minus_ph_only": cluster_bootstrap(
            truth_reference,
            source_reference,
            predictions["mask_topo"],
            predictions["ph_only"],
            bootstrap_repetitions,
            rng,
        ),
        "mask_topo_minus_best_ph_envelope": cluster_bootstrap(
            truth_reference,
            source_reference,
            predictions["mask_topo"],
            predictions[best_ph],
            bootstrap_repetitions,
            rng,
        ),
        "best_ph_minus_strongest_reducer": cluster_bootstrap(
            truth_reference,
            source_reference,
            predictions[best_ph],
            predictions[strongest_reducer],
            bootstrap_repetitions,
            rng,
        ),
    }
    return {
        "dataset": dataset,
        "token_count": TOKEN_COUNTS[dataset],
        "test_sample_count": int(truth_reference.size),
        "test_source_count": int(np.unique(source_reference).size),
        "aggregates": aggregates,
        "strongest_reducer": strongest_reducer,
        "best_ph_by_three_seed_mean": best_ph,
        "contrasts": contrasts,
        "ph_diagnostics": {
            "test_selected_h0_mean_by_seed": ph_h0,
            "test_selected_h1_mean_by_seed": ph_h1,
            "test_padding_rate_by_seed": ph_padding,
            "ph_preprocessing_ms_per_sample_batch1_by_seed": ph_latency,
            "ph_preprocessing_ms_per_sample_batch1_mean": float(np.mean(ph_latency)),
        },
        "runs": run_rows,
        "artifact_hashes": artifacts,
        "test_was_previously_observed": True,
        "interpretation_scope": (
            "retrospective_mechanism_only"
            if dataset == "crackforest"
            else "retrospective_collision"
        ),
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TB-B-260802-022 PH direct topology baseline",
        "",
        "All results below are endpoint-connectivity balanced accuracy, not FIVES disease classification.",
        "All three dataset tests were previously observed; this is retrospective collision/mechanism evidence.",
        "Massachusetts Roads one-shot test was not reopened.",
        "",
        "| Dataset | K | MaskTopo | PH-only | PH-guided | Best PH | MaskTopo - best PH (95% source-cluster CI) |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for dataset in DATASETS:
        item = report["datasets"][dataset]
        aggregates = item["aggregates"]
        best_ph = item["best_ph_by_three_seed_mean"]
        contrast = item["contrasts"]["mask_topo_minus_best_ph_envelope"]
        ci = contrast["source_cluster_bootstrap_95_ci"]
        lines.append(
            f"| {dataset} | {item['token_count']} | "
            f"{100 * aggregates['mask_topo']['mean_balanced_accuracy']:.2f}±"
            f"{100 * aggregates['mask_topo']['sample_std_balanced_accuracy']:.2f}% | "
            f"{100 * aggregates['ph_only']['mean_balanced_accuracy']:.2f}±"
            f"{100 * aggregates['ph_only']['sample_std_balanced_accuracy']:.2f}% | "
            f"{100 * aggregates['ph_guided']['mean_balanced_accuracy']:.2f}±"
            f"{100 * aggregates['ph_guided']['sample_std_balanced_accuracy']:.2f}% | "
            f"{best_ph} | {100 * contrast['mean_gain']:+.2f} pp "
            f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- The frozen primary PH comparator is PH-guided; the best-PH envelope is also reported conservatively.",
            "- PH-only failure alone is not evidence against PH; conclusions use PH-guided and the best-PH envelope.",
            "- TopoCL and the WACV 2026 multi-filtration classifier remain protocol-related work, not directly comparable task scores.",
            "- The PH CPU preprocessing time is included in the later Pareto analysis.",
            "- No claim from this report uses the separate P1-5 disease-classification +1 pp result.",
            "",
        ]
    )
    return "\n".join(lines)


def run_self_test() -> None:
    truth = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    sources = np.asarray(["a", "a", "b", "b"])
    first = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8)
    second = np.asarray([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=np.uint8)
    result = cluster_bootstrap(
        truth, sources, first, second, 100, np.random.default_rng(1)
    )
    assert result["mean_gain"] == 0.5
    print("SUMMARIZE_PH_TOKEN_BASELINES_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = args.root.resolve()
    datasets = {
        dataset: load_dataset(
            root,
            dataset,
            args.seeds,
            args.crackforest_cache.resolve(),
            args.bootstrap_repetitions,
            args.statistics_seed,
        )
        for dataset in DATASETS
    }
    report = {
        "experiment_id": "TB-B-260802-022",
        "status": "completed_verified",
        "confirmatory_status": "retrospective_collision_and_mechanism",
        "optimization_seeds": args.seeds,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "datasets": datasets,
        "massachusetts_test_reopened": False,
        "disease_classification_p1_5_used": False,
        "protocol": str((root / "PH_TOKEN_BASELINE_PROTOCOL.md").resolve()),
        "protocol_sha256": file_sha256(root / "PH_TOKEN_BASELINE_PROTOCOL.md"),
        "code_sha256": file_sha256(Path(__file__)),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(report_markdown(report), encoding="utf-8")
    with (output / "source_data.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = ("dataset", "seed", "model", "balanced_accuracy", "parameters")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in DATASETS:
            writer.writerows(report["datasets"][dataset]["runs"])
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

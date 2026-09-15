from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from summarize_crackforest_mechanism_ablation import balanced_accuracy


SEEDS = (20260820, 20260821, 20260822)
MODELS = (
    "grid_external",
    "mask_assignment_identity",
    "shuffled_topology",
    "mask_topo_external",
    "tokenlearner",
    "perceiver_resampler",
    "tome_style",
    "g2tm_fixedk_mask",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "results_massroads_external_summary"
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["status"] != "completed_one_shot_test":
        raise RuntimeError("Result status is not complete.")
    if result["verdict"] != "MASSROADS_EXTERNAL_CONFIRMATION_PASS":
        raise RuntimeError("Frozen verdict is not PASS.")
    truth_reference = None
    source_reference = None
    recomputed = {name: [] for name in MODELS}
    for seed in SEEDS:
        with np.load(output / f"seed{seed}_predictions.npz") as archive:
            truth = archive["truth"].astype(np.uint8)
            sources = archive["source_id"].astype(str)
            if set(archive.files) != {"truth", "source_id", *MODELS}:
                raise RuntimeError(f"Prediction key mismatch for seed {seed}.")
            if truth_reference is None:
                truth_reference = truth
                source_reference = sources
            elif not np.array_equal(truth_reference, truth) or not np.array_equal(
                source_reference, sources
            ):
                raise RuntimeError(f"Truth/source mismatch for seed {seed}.")
            for name in MODELS:
                score = balanced_accuracy(truth, archive[name].astype(np.uint8))
                recomputed[name].append(score)
    for name in MODELS:
        recorded = result["aggregates"][name]
        if not np.allclose(recomputed[name], recorded["scores_by_seed"]):
            raise RuntimeError(f"Saved BA mismatch for {name}.")
        if not np.isclose(np.mean(recomputed[name]), recorded["mean_balanced_accuracy"]):
            raise RuntimeError(f"Saved mean BA mismatch for {name}.")
    if truth_reference is None or source_reference is None:
        raise RuntimeError("No predictions found.")
    if truth_reference.size != 1200 or not np.isclose(truth_reference.mean(), 0.5):
        raise RuntimeError("Frozen test truth count/balance changed.")
    if np.unique(source_reference).size != 49:
        raise RuntimeError("Frozen test source count changed.")
    if not all(bool(value) for value in result["frozen_gates"].values()):
        raise RuntimeError("At least one frozen gate is false.")

    marker_path = root / "实验记录/TB-B-260802-021_TEST_INFERENCE_STARTED.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker["status"] != "TEST_INFERENCE_COMPLETED_DO_NOT_RERUN":
        raise RuntimeError("One-shot marker is not complete.")
    completed = marker["completed_checkpoints"]
    if len(completed) != 27 or len(
        {(int(item["seed"]), item["model"]) for item in completed}
    ) != 27:
        raise RuntimeError("One-shot checkpoint receipt is incomplete or duplicated.")
    if marker["result"]["sha256"] != sha256_file(result_path):
        raise RuntimeError("One-shot result hash mismatch.")

    with (output / "source_data.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(SEEDS) * len(MODELS):
        raise RuntimeError("Source-data row count mismatch.")
    gate = json.loads((output / "test_data_gate.json").read_text(encoding="utf-8"))
    if gate["verdict"] != "TEST_DATA_INTEGRITY_PASS" or not all(
        bool(value) for value in gate["gates"].values()
    ):
        raise RuntimeError("Test data gate is not fully passing.")
    print(
        json.dumps(
            {
                "status": "MASSROADS_EXTERNAL_OFFLINE_VERIFICATION_PASS",
                "recomputed_model_seed_scores": 24,
                "truth_count": int(truth_reference.size),
                "positive_fraction": float(truth_reference.mean()),
                "source_tiles": int(np.unique(source_reference).size),
                "one_shot_checkpoint_receipts": len(completed),
                "frozen_gates_passed": sum(
                    bool(value) for value in result["frozen_gates"].values()
                ),
                "result_sha256": sha256_file(result_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

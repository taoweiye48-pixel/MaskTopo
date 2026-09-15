from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
EDGE_RUNS = [
    ROOT / "results_edge_topocoarsen" / "result.json",
    *[
        ROOT / f"results_edge_topocoarsen_seed{seed}" / "result.json"
        for seed in (20260811, 20260812, 20260813, 20260814)
    ],
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    output = ROOT / "results_edge_topocoarsen_stage2"
    output.mkdir(parents=True, exist_ok=True)
    initial = load(ROOT / "results_edge_topocoarsen" / "result.json")
    grid_multiseed = load(
        ROOT / "results_grid_multiseed" / "result.json"
    )
    diagnostic_ood = load(
        ROOT / "results_edge_topocoarsen_ood" / "result.json"
    )
    robust_connector = load(
        ROOT / "results_edge_topocoarsen_robust" / "result.json"
    )
    robust_ood = load(
        ROOT / "results_edge_topocoarsen_ood_robust" / "result.json"
    )
    confirmatory = load(
        ROOT
        / "results_edge_topocoarsen_ood_robust_confirmatory"
        / "result.json"
    )

    edge_values = np.asarray(
        [
            load(path)["connector"]["test_metrics"]["balanced_accuracy"]
            for path in EDGE_RUNS
        ]
    )
    grid_values = np.asarray(
        [
            row["test_metrics"]["balanced_accuracy"]
            for row in grid_multiseed["per_seed"]
        ]
    )
    seeds = [20260810, 20260811, 20260812, 20260813, 20260814]
    gains = edge_values - grid_values
    rows = [
        {
            "optimization_seed": seed,
            "edge_topocoarsen": edge,
            "grid": grid,
            "paired_gain": gain,
        }
        for seed, edge, grid, gain in zip(
            seeds, edge_values, grid_values, gains
        )
    ]
    with (output / "paired_multiseed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    initial_photo = next(
        row
        for row in diagnostic_ood["variants"]
        if row["variant"] == "low_contrast_noise"
    )
    robust_photo = next(
        row
        for row in robust_ood["variants"]
        if row["variant"] == "low_contrast_noise"
    )
    summary = {
        "experiment_id": "edge_topocoarsen_stage2_summary",
        "status": "completed",
        "initial_connector_gate": {
            "edge_topocoarsen": initial["connector"]["test_metrics"][
                "balanced_accuracy"
            ],
            "grid": initial["reference_scores_same_split_seed_training_protocol"][
                "grid"
            ],
            "topology_oracle": initial[
                "reference_scores_same_split_seed_training_protocol"
            ]["topology_oracle"],
            "edge_structural_accuracy": initial["artifact_diagnostics"][
                "test"
            ]["structural_endpoint_metrics"]["balanced_accuracy"],
        },
        "paired_fixed_split_five_seed": {
            "seeds": seeds,
            "edge_scores": edge_values.tolist(),
            "grid_scores": grid_values.tolist(),
            "paired_gains": gains.tolist(),
            "edge_mean": float(edge_values.mean()),
            "edge_sample_sd": float(edge_values.std(ddof=1)),
            "grid_mean": float(grid_values.mean()),
            "grid_sample_sd": float(grid_values.std(ddof=1)),
            "mean_paired_gain": float(gains.mean()),
            "paired_gain_sample_sd": float(gains.std(ddof=1)),
            "minimum_paired_gain": float(gains.min()),
        },
        "diagnostic_stress_failure": {
            "variant": "low_contrast_noise",
            "before_robustification": initial_photo[
                "edge_topocoarsen_balanced_accuracy"
            ],
            "structural_accuracy_before_robustification": initial_photo[
                "structural_balanced_accuracy"
            ],
            "after_canonicalization_and_augmentation": robust_photo[
                "edge_topocoarsen_balanced_accuracy"
            ],
            "robust_grid_control": robust_photo["grid_balanced_accuracy"],
        },
        "robust_connector_id_gate": {
            "edge_topocoarsen": robust_connector["connector"][
                "test_metrics"
            ]["balanced_accuracy"],
            "retrained_grid": robust_connector["retrained_grid_control"][
                "test_metrics"
            ]["balanced_accuracy"],
        },
        "fresh_confirmatory_ood": {
            "sample_seed": 20260999,
            "summary": confirmatory["summary"],
            "variants": confirmatory["variants"],
            "verdict": confirmatory["verdict"],
        },
        "overall_verdict": (
            "CONTINUE_TO_REAL_DATA"
            if confirmatory["verdict"] == "OOD_GATE_PASS"
            and float(gains.min()) >= 0.10
            else "REDESIGN_OR_STOP"
        ),
    }
    (output / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(
        ["grid", "EdgeTopo", "oracle"],
        [
            summary["initial_connector_gate"]["grid"],
            summary["initial_connector_gate"]["edge_topocoarsen"],
            summary["initial_connector_gate"]["topology_oracle"],
        ],
        color=["#4C78A8", "#E45756", "#72B7B2"],
    )
    axes[0].set_title("Initial held-out connector gate")
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_ylabel("balanced accuracy")

    x = np.arange(len(seeds))
    axes[1].plot(x, edge_values, marker="o", label="EdgeTopoCoarsen")
    axes[1].plot(x, grid_values, marker="o", label="grid")
    axes[1].set_xticks(x, [str(seed)[-2:] for seed in seeds])
    axes[1].set_title("Fixed-split optimization seeds")
    axes[1].set_xlabel("seed suffix")
    axes[1].set_ylim(0.45, 1.0)
    axes[1].legend()

    confirm_rows = confirmatory["variants"]
    labels = [row["variant"] for row in confirm_rows]
    positions = np.arange(len(labels))
    width = 0.38
    axes[2].bar(
        positions - width / 2,
        [row["grid_balanced_accuracy"] for row in confirm_rows],
        width,
        label="grid",
        color="#4C78A8",
    )
    axes[2].bar(
        positions + width / 2,
        [
            row["edge_topocoarsen_balanced_accuracy"]
            for row in confirm_rows
        ],
        width,
        label="robust EdgeTopo",
        color="#E45756",
    )
    axes[2].set_xticks(
        positions,
        ["ID", "rot90", "flip", "photo", "resize", "topology"],
        rotation=20,
    )
    axes[2].set_title("Fresh confirmatory shifts")
    axes[2].set_ylim(0.45, 1.0)
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25, axis="y")
    fig.savefig(output / "stage2_overview.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

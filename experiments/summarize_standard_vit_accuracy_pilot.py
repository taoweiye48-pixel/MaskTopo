from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


MODES = ("full", "grid", "g2tm_fixedk", "hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired ViT accuracy pilot runs.")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--efficiency-result",
        type=Path,
        default=Path("results_standard_vit_efficiency/result.json"),
    )
    parser.add_argument(
        "--e2e-result",
        type=Path,
        default=Path("results_topology_postprocess_e2e_v2/result.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results_standard_vit_accuracy_summary")
    )
    return parser.parse_args()


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(value) for value in array],
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for directory in args.runs:
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "completed":
            raise ValueError(f"Incomplete run: {result_path}")
        if result["experiment_class"] != "formal_partial_finetuning_pilot":
            raise ValueError(f"Non-formal run cannot enter summary: {result_path}")
        runs.append(result)
    seeds = [int(run["configuration"]["seed"]) for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds supplied.")
    common_modes = set(MODES)
    for run in runs:
        common_modes &= set(run["models"])
    if common_modes != set(MODES):
        raise ValueError(f"All four modes are required; found {sorted(common_modes)}")

    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        mode_summary: dict[str, Any] = {}
        for metric in (
            "balanced_accuracy",
            "positive_accuracy",
            "negative_accuracy",
            "accuracy",
        ):
            values = [float(run["models"][mode]["test"][metric]) for run in runs]
            mode_summary[metric] = mean_sd(values)
        mode_summary["best_dev_balanced_accuracy"] = mean_sd(
            [float(run["models"][mode]["best_dev_balanced_accuracy"]) for run in runs]
        )
        mode_summary["training_seconds"] = mean_sd(
            [float(run["models"][mode]["training_seconds"]) for run in runs]
        )
        mode_summary["trainable_parameters"] = int(
            runs[0]["models"][mode]["trainable_parameters"]
        )
        summary[mode] = mode_summary
        rows.append(
            {
                "model": mode,
                "test_balanced_accuracy_mean": mode_summary["balanced_accuracy"]["mean"],
                "test_balanced_accuracy_sd": mode_summary["balanced_accuracy"]["sd"],
                "positive_accuracy_mean": mode_summary["positive_accuracy"]["mean"],
                "negative_accuracy_mean": mode_summary["negative_accuracy"]["mean"],
                "accuracy_mean": mode_summary["accuracy"]["mean"],
                "trainable_parameters": mode_summary["trainable_parameters"],
            }
        )

    paired = {}
    hybrid_values = np.asarray(summary["hybrid"]["balanced_accuracy"]["values"])
    for baseline in ("full", "grid", "g2tm_fixedk"):
        baseline_values = np.asarray(summary[baseline]["balanced_accuracy"]["values"])
        paired[baseline] = mean_sd((hybrid_values - baseline_values).tolist())

    efficiency = json.loads(args.efficiency_result.read_text(encoding="utf-8"))
    e2e = json.loads(args.e2e_result.read_text(encoding="utf-8"))
    full_e2e = float(e2e["end_to_end"]["8"]["full_vit"]["mean_ms_per_sample"])
    hybrid_e2e = float(e2e["end_to_end"]["8"]["masktopo_hybrid"]["mean_ms_per_sample"])
    latency_reduction = 100.0 * (full_e2e - hybrid_e2e) / full_e2e
    latency = {
        "batch_size": 8,
        "full_e2e_ms_per_sample": full_e2e,
        "hybrid_e2e_ms_per_sample": hybrid_e2e,
        "hybrid_reduction_percent": latency_reduction,
        "grid_cached_connector_ms_per_sample": float(
            efficiency["modes"]["grid"]["latency_ms_per_sample"]
        ),
        "g2tm_cached_connector_ms_per_sample": float(
            efficiency["modes"]["g2tm_fixedk"]["latency_ms_per_sample"]
        ),
        "warning": "Grid/G2TM values exclude dynamic topology prediction/build and are not directly comparable end-to-end values.",
    }

    non_hybrid_best = max(
        summary[mode]["balanced_accuracy"]["mean"]
        for mode in ("full", "grid", "g2tm_fixedk")
    )
    hybrid_mean = summary["hybrid"]["balanced_accuracy"]["mean"]
    gate_one = hybrid_mean >= non_hybrid_best + 0.03
    gate_two = (
        hybrid_mean >= summary["full"]["balanced_accuracy"]["mean"] - 0.02
        and hybrid_mean >= summary["grid"]["balanced_accuracy"]["mean"] + 0.03
        and hybrid_mean >= summary["g2tm_fixedk"]["balanced_accuracy"]["mean"] + 0.03
        and latency_reduction >= 20.0
    )
    decision = {
        "gate_one_accuracy_lead_3pp": bool(gate_one),
        "gate_two_tradeoff": bool(gate_two),
        "pilot_pass": bool(gate_one or gate_two),
        "recommendation": (
            "PROCEED_TO_FULL_FINETUNING_AND_EXTERNAL_VALIDATION"
            if gate_one or gate_two
            else "REVISE_CONNECTOR_BEFORE_EXPENSIVE_SCALING"
        ),
    }
    output = {
        "experiment_id": "TB-B-260801-007-summary",
        "status": "completed",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "models": summary,
        "paired_hybrid_minus_baseline": paired,
        "efficiency": latency,
        "decision_gate": decision,
        "limitations": [
            "Partial fine-tuning pilot; not a final full-fine-tuning result.",
            "n=3 seeds; paired differences are descriptive and no definitive p value is claimed.",
            "Accuracy and latency were measured in separate controlled runs on the same GPU class.",
        ],
        "run_paths": [str(path.resolve()) for path in args.runs],
    }
    (args.output / "result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# Standard ViT accuracy--efficiency pilot",
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        "",
        "| Model | Test balanced accuracy | Positive accuracy | Negative accuracy |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        item = summary[mode]
        report.append(
            f"| {mode} | {100*item['balanced_accuracy']['mean']:.2f} +/- "
            f"{100*item['balanced_accuracy']['sd']:.2f}% | "
            f"{100*item['positive_accuracy']['mean']:.2f}% | "
            f"{100*item['negative_accuracy']['mean']:.2f}% |"
        )
    report.extend(
        [
            "",
            "## Paired Hybrid differences",
            "",
            *[
                f"- Hybrid - {baseline}: {100*value['mean']:+.2f} +/- {100*value['sd']:.2f} pp"
                for baseline, value in paired.items()
            ],
            "",
            "## Efficiency",
            "",
            f"At batch 8, measured end-to-end latency was {full_e2e:.3f} ms/sample "
            f"for Full and {hybrid_e2e:.3f} ms/sample for Hybrid "
            f"({latency_reduction:.2f}% reduction).",
            "",
            "## Pilot decision",
            "",
            f"- Gate 1: {gate_one}",
            f"- Gate 2: {gate_two}",
            f"- Decision: {decision['recommendation']}",
            "",
            "This is a partial fine-tuning pilot and must not be described as the final model result.",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


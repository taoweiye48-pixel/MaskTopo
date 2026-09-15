from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from p0_full_finetune import MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize formal TopoBridge P0 runs.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results_p0_full_finetune_summary")
    )
    return parser.parse_args()


def mean_sd(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": array.tolist(),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [
        json.loads((path / "result.json").read_text(encoding="utf-8"))
        for path in args.runs
    ]
    for run in runs:
        if run["status"] != "completed" or run["experiment_class"] != "formal_p0_full_finetuning":
            raise ValueError("Only completed formal P0 runs may enter the summary.")
        if set(run["models"]) != set(MODES):
            raise ValueError("A formal P0 run is missing one or more conditions.")
    seeds = [int(run["configuration"]["seed"]) for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds supplied.")
    model_summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        item: dict[str, Any] = {}
        for metric in (
            "balanced_accuracy",
            "positive_accuracy",
            "negative_accuracy",
            "accuracy",
        ):
            item[metric] = mean_sd(
                [float(run["models"][mode]["test"][metric]) for run in runs]
            )
        item["best_dev_balanced_accuracy"] = mean_sd(
            [float(run["models"][mode]["best_dev_balanced_accuracy"]) for run in runs]
        )
        item["training_seconds"] = mean_sd(
            [float(run["models"][mode]["training_seconds"]) for run in runs]
        )
        item["peak_cuda_memory_mb"] = mean_sd(
            [float(run["models"][mode]["peak_cuda_memory_mb"]) for run in runs]
        )
        item["trainable_parameters"] = int(
            runs[0]["models"][mode]["trainable_parameters"]
        )
        model_summary[mode] = item
        rows.append(
            {
                "model": mode,
                "balanced_accuracy_mean": item["balanced_accuracy"]["mean"],
                "balanced_accuracy_sd": item["balanced_accuracy"]["sd"],
                "positive_accuracy_mean": item["positive_accuracy"]["mean"],
                "negative_accuracy_mean": item["negative_accuracy"]["mean"],
                "peak_cuda_memory_mb_mean": item["peak_cuda_memory_mb"]["mean"],
                "training_seconds_mean": item["training_seconds"]["mean"],
                "trainable_parameters": item["trainable_parameters"],
            }
        )
    hybrid = np.asarray(model_summary["hybrid"]["balanced_accuracy"]["values"])
    paired: dict[str, Any] = {}
    for mode in MODES:
        if mode == "hybrid":
            continue
        values = np.asarray(model_summary[mode]["balanced_accuracy"]["values"])
        paired[mode] = mean_sd((hybrid - values).tolist())
    full_gate = (
        model_summary["hybrid"]["balanced_accuracy"]["mean"]
        >= model_summary["full"]["balanced_accuracy"]["mean"] - 0.02
        and model_summary["hybrid"]["balanced_accuracy"]["mean"]
        >= model_summary["g2tm_fixedk"]["balanced_accuracy"]["mean"] + 0.03
    )
    counterfactual_gate = (
        paired["shuffled_global"]["mean"] >= 0.05
        and paired["random_connected"]["mean"] >= 0.05
    )
    mechanism = {
        mode: {
            **paired[mode],
            "independent_component_supported_at_1pp": paired[mode]["mean"] >= 0.01,
        }
        for mode in ("assignment_only", "local_only", "reachability_only")
    }
    within_class_drop = paired["shuffled_within_class"]
    decision = {
        "full_finetuning_gate": bool(full_gate),
        "counterfactual_gate": bool(counterfactual_gate),
        "p0_pass": bool(full_gate and counterfactual_gate),
        "within_class_alignment_supported_at_3pp": within_class_drop["mean"] >= 0.03,
        "recommendation": (
            "P0_COMPLETE_PROCEED_TO_EXTERNAL_VALIDATION"
            if full_gate and counterfactual_gate
            else "P0_FAILED_REVISE_METHOD_OR_CLAIMS"
        ),
    }
    output = {
        "experiment_id": "TB-B-260801-P0-summary",
        "status": "completed",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "models": model_summary,
        "paired_hybrid_minus_condition": paired,
        "mechanism_assessment": mechanism,
        "decision": decision,
        "limitations": [
            "Three seeds support stability estimates but not definitive p-values.",
            "Within-class shuffling diagnoses label-level topology information; it is not a causal intervention on mask supervision.",
            "External-domain validation remains a P1 experiment.",
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
        "# TopoBridge P0 full-fine-tuning summary",
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        "",
        "| Condition | Test balanced accuracy | Hybrid minus condition |",
        "|---|---:|---:|",
    ]
    for mode in MODES:
        metric = model_summary[mode]["balanced_accuracy"]
        difference = "--" if mode == "hybrid" else (
            f"{100*paired[mode]['mean']:+.2f} +/- {100*paired[mode]['sd']:.2f} pp"
        )
        report.append(
            f"| {mode} | {100*metric['mean']:.2f} +/- {100*metric['sd']:.2f}% | {difference} |"
        )
    report.extend(
        [
            "",
            "## P0 gates",
            "",
            f"- Full-fine-tuning fairness gate: {decision['full_finetuning_gate']}",
            f"- Counterfactual topology gate: {decision['counterfactual_gate']}",
            f"- Within-class alignment diagnostic >=3 pp: {decision['within_class_alignment_supported_at_3pp']}",
            f"- Overall: {decision['recommendation']}",
            "",
            "## Mechanism assessment",
            "",
            *[
                f"- Hybrid - {mode}: {100*value['mean']:+.2f} +/- {100*value['sd']:.2f} pp; independent >=1 pp: {value['independent_component_supported_at_1pp']}"
                for mode, value in mechanism.items()
            ],
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


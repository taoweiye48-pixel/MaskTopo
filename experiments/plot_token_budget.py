from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BUDGETS = (8, 16, 32, 64)
MODELS = (
    "mask_topo",
    "tokenlearner",
    "perceiver_resampler",
    "tome_style",
    "mask_guided_queries",
)
LABELS = {
    "mask_topo": "MaskTopo",
    "tokenlearner": "TokenLearner",
    "perceiver_resampler": "Perceiver",
    "tome_style": "ToMe-style",
    "mask_guided_queries": "Mask-guided queries",
}
COLORS = {
    "mask_topo": "#2864A5",
    "tokenlearner": "#D9822B",
    "perceiver_resampler": "#4B8B69",
    "tome_style": "#7A7F87",
    "mask_guided_queries": "#8A63A8",
}
MARKERS = {
    "mask_topo": "o",
    "tokenlearner": "s",
    "perceiver_resampler": "^",
    "tome_style": "D",
    "mask_guided_queries": "P",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication figure for TopoBridge token-budget results."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results_token_budget_summary/result.json"),
    )
    parser.add_argument(
        "--efficiency",
        type=Path,
        default=Path("results_token_budget_efficiency/result.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_token_budget_summary"),
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "legend.frameon": False,
        }
    )


def efficiency_index(result: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (row["model"], int(row["token_count"])): row
        for row in result["rows"]
    }


def preprocessing_index(
    result: dict[str, Any], dataset: str
) -> dict[int, float]:
    return {
        int(row["token_count"]): float(row["milliseconds_per_sample"])
        for row in result["topology_preprocessing"]
        if row["dataset"] == dataset
    }


def accuracy_panel(
    axis: plt.Axes,
    summary: dict[str, Any],
    dataset: str,
    title: str,
) -> None:
    x = np.log2(np.asarray(BUDGETS))
    for model in MODELS:
        means = np.asarray(
            [
                summary["datasets"][dataset]["budgets"][str(token_count)][
                    "aggregates"
                ][model]["mean_balanced_accuracy"]
                for token_count in BUDGETS
            ]
        )
        errors = np.asarray(
            [
                summary["datasets"][dataset]["budgets"][str(token_count)][
                    "aggregates"
                ][model]["sample_std_balanced_accuracy"]
                for token_count in BUDGETS
            ]
        )
        axis.errorbar(
            x,
            100 * means,
            yerr=100 * errors,
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=4.2 if model == "mask_topo" else 3.6,
            linewidth=1.8 if model == "mask_topo" else 1.1,
            capsize=2,
            capthick=0.7,
            alpha=1.0 if model == "mask_topo" else 0.88,
            label=LABELS[model],
            zorder=5 if model == "mask_topo" else 3,
        )
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xticks(x, [str(value) for value in BUDGETS])
    axis.set_xlabel("Reduced token count, K")
    axis.set_ylabel("Balanced accuracy (%)")
    axis.set_ylim(43, 84)
    axis.grid(axis="y", color="#D9DDE2", linewidth=0.55, alpha=0.75)


def connector_latency(
    index: dict[tuple[str, int], dict[str, Any]],
    model: str,
    token_count: int,
) -> float:
    return float(
        index[(model, token_count)]["connector_runtime"][
            "latency_ms_per_sample"
        ]
    )


def end_to_end_latency(
    efficiency: dict[str, Any],
    index: dict[tuple[str, int], dict[str, Any]],
    preprocessing: dict[int, float],
    model: str,
    token_count: int,
) -> float:
    value = connector_latency(index, model, token_count)
    mask_latency = float(
        efficiency["mask_predictor"]["runtime"]["latency_ms_per_sample"]
    )
    if model == "mask_topo":
        value += mask_latency + preprocessing[token_count]
    elif model == "mask_guided_queries":
        value += mask_latency
    return value


def latency_panel(
    axis: plt.Axes,
    efficiency: dict[str, Any],
    index: dict[tuple[str, int], dict[str, Any]],
    preprocessing: dict[int, float],
) -> None:
    x = np.log2(np.asarray(BUDGETS))
    for model in MODELS:
        connector = [
            connector_latency(index, model, token_count)
            for token_count in BUDGETS
        ]
        axis.plot(
            x,
            connector,
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=3.4,
            linewidth=1.0,
            alpha=0.78,
        )
    mask_e2e = [
        end_to_end_latency(
            efficiency,
            index,
            preprocessing,
            "mask_topo",
            token_count,
        )
        for token_count in BUDGETS
    ]
    query_e2e = [
        end_to_end_latency(
            efficiency,
            index,
            preprocessing,
            "mask_guided_queries",
            token_count,
        )
        for token_count in BUDGETS
    ]
    axis.plot(
        x,
        mask_e2e,
        color=COLORS["mask_topo"],
        linewidth=2.0,
        linestyle="--",
        label="MaskTopo end-to-end",
    )
    axis.plot(
        x,
        query_e2e,
        color=COLORS["mask_guided_queries"],
        linewidth=1.4,
        linestyle="--",
        label="Mask-query end-to-end",
    )
    axis.set_yscale("log")
    axis.set_xticks(x, [str(value) for value in BUDGETS])
    axis.set_xlabel("Reduced token count, K")
    axis.set_ylabel("Latency per sample (ms, log scale)")
    axis.set_title(
        "c  Connector and estimated end-to-end latency",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="y", which="both", color="#D9DDE2", linewidth=0.55)
    axis.legend(loc="upper left")


def pareto_panel(
    axis: plt.Axes,
    summary: dict[str, Any],
    efficiency: dict[str, Any],
    index: dict[tuple[str, int], dict[str, Any]],
    preprocessing: dict[int, float],
) -> None:
    for model in MODELS:
        latency = np.asarray(
            [
                end_to_end_latency(
                    efficiency,
                    index,
                    preprocessing,
                    model,
                    token_count,
                )
                for token_count in BUDGETS
            ]
        )
        accuracy = np.asarray(
            [
                summary["datasets"]["deepcrack"]["budgets"][
                    str(token_count)
                ]["aggregates"][model]["mean_balanced_accuracy"]
                for token_count in BUDGETS
            ]
        )
        axis.plot(
            latency,
            100 * accuracy,
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=3.8,
            linewidth=1.1 if model != "mask_topo" else 1.8,
            alpha=0.9,
        )
        if model == "mask_topo":
            for x_value, y_value, token_count in zip(
                latency, 100 * accuracy, BUDGETS
            ):
                axis.annotate(
                    f"K={token_count}",
                    (x_value, y_value),
                    xytext=(3, 2),
                    textcoords="offset points",
                    fontsize=5.7,
                    color=COLORS[model],
                )
    axis.set_xscale("log")
    axis.set_xlabel("Estimated end-to-end latency (ms, log scale)")
    axis.set_ylabel("DeepCrack balanced accuracy (%)")
    axis.set_title(
        "d  Accuracy–latency trade-off",
        loc="left",
        fontweight="bold",
    )
    axis.grid(color="#D9DDE2", linewidth=0.55, alpha=0.75)


def main() -> None:
    args = parse_args()
    if np.any(np.asarray(BUDGETS, dtype=float) <= 0):
        raise ValueError("Token budgets must be strictly positive for log axes.")
    summary = json.loads(
        args.summary.read_text(encoding="utf-8")
    )
    efficiency = json.loads(
        args.efficiency.read_text(encoding="utf-8")
    )
    configure_style()
    index = efficiency_index(efficiency)
    preprocessing = preprocessing_index(efficiency, "deepcrack")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.1, 5.2),
        constrained_layout=True,
    )
    accuracy_panel(
        axes[0, 0],
        summary,
        "crackforest",
        "a  CrackForest: accuracy versus token budget",
    )
    accuracy_panel(
        axes[0, 1],
        summary,
        "deepcrack",
        "b  DeepCrack: accuracy versus token budget",
    )
    latency_panel(axes[1, 0], efficiency, index, preprocessing)
    pareto_panel(
        axes[1, 1],
        summary,
        efficiency,
        index,
        preprocessing,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.035),
    )
    figure.text(
        0.5,
        -0.015,
        "Accuracy: mean ± sample SD over 3 optimization seeds. "
        "End-to-end latency is a serial estimate including mask prediction "
        "and CPU topology construction where required.",
        ha="center",
        va="top",
        fontsize=6,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = output / "FIG_TOKEN_BUDGET"
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"TOKEN_BUDGET_FIGURE_WRITTEN {base}", flush=True)


if __name__ == "__main__":
    main()

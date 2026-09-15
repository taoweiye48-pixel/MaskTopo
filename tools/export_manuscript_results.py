"""Export the paper's aggregate tables without reconstructing seed observations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reported_results"
LABELS = ("main", "mech", "effects", "ar")
DOMAINS = ("FIVES", "DeepCrack", "Massachusetts Roads", "Brassica")


def table_rows(table: str) -> list[list[str]]:
    body = table.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    rows = []
    for line in body.strip().splitlines():
        line = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", line)
        rows.append([cell.strip().rstrip("\\").strip() for cell in line.split("&")])
    return rows


def mean_sd(cell: str) -> tuple[str, str]:
    values = re.findall(r"\d+\.\d+", cell)
    if len(values) != 2:
        raise ValueError(f"Expected a mean and SD: {cell}")
    return values[0], values[1]


def write_csv(name: str, fields: list[str], rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, help="Refresh excerpts from main.tex.")
    parser.add_argument("--pdf", type=Path, help="Record the matching paper PDF checksum.")
    args = parser.parse_args()
    snapshot = OUT / "manuscript_tables.tex"
    source = args.manuscript or snapshot
    text = source.read_text(encoding="utf-8")
    matches = re.findall(r"\\begin\{table\}\[t\].*?\\end\{table\}", text, re.S)
    tables = {}
    for block in matches:
        label = re.search(r"\\label\{tab:([^}]+)\}", block)
        if label and label[1] in LABELS:
            tables[label[1]] = block
    if set(tables) != set(LABELS):
        raise ValueError("Expected manuscript tables main, mech, effects, and ar.")
    if args.manuscript:
        snapshot.write_text(
            "% Exact aggregate table excerpts from the current manuscript.\n\n"
            + "\n\n".join(tables[label] for label in LABELS) + "\n",
            encoding="utf-8", newline="\n",
        )
        metadata = {
            "title": "MaskTopo: Component-Aligned Token Coarsening for Thin-Structure Connectivity",
            "authority": "current manuscript",
            "source_file": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "data_level": "reported aggregates; no individual seed observations",
        }
        if args.pdf:
            metadata["pdf_sha256"] = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
        (OUT / "manuscript_source.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    rows = table_rows(tables["main"])
    assert len(rows) == 4 and all(len(row) == 5 for row in rows)
    ci_text = tables["main"].split("Source-cluster", 1)[1].split("Best-PH", 1)[0]
    intervals = re.findall(r"\[([\d.]+),([\d.]+)\]", ci_text)
    assert len(intervals) == 4
    comparators = ("TokenLearner", "Mask-Perceiver", "Matched G2TM", "Mask-Slot")
    primary = []
    for domain, comparator, row, interval in zip(DOMAINS, comparators, rows, intervals):
        mean, sd = mean_sd(row[2])
        primary.append(dict(
            dataset=domain, split="test", token_count=row[1],
            masktopo_ba_mean_pct=mean, masktopo_ba_sd_pct=sd,
            primary_comparator=comparator,
            comparator_ba_mean_pct=re.findall(r"\d+\.\d+", row[3])[0],
            gain_pp=row[4].lstrip("+"), ci95_low_pp=interval[0], ci95_high_pp=interval[1],
        ))
    write_csv("table1_primary_results.csv", list(primary[0]), primary)

    splits = ("test", "test", "post_hoc_development", "post_hoc_development")
    protocols = ("FIVES", "historical_float32", "development_ablation", "development_ablation")
    rows = table_rows(tables["mech"])
    assert len(rows) == 5 and all(len(row) == 5 for row in rows)
    conditions = {"Grid": "Grid", "Grid + graph": "Grid + graph",
                  "Assign. + id.": "Assignment+identity", "MaskTopo": "MaskTopo", "Shuffled": "Shuffled"}
    mechanism = []
    for row in rows:
        for domain, split, protocol, value in zip(DOMAINS, splits, protocols, row[1:]):
            mechanism.append(dict(dataset=domain, split=split, protocol=protocol,
                                  condition=conditions[row[0]], ba_mean_pct=value))
    write_csv("table2_component_alignment.csv", list(mechanism[0]), mechanism)

    rows = table_rows(tables["effects"])
    assert len(rows) == 2 and all(len(row) == 5 for row in rows)
    effects = []
    for row, effect, comparator in zip(rows, ("graph", "correspondence"), ("Assignment+identity", "Shuffled")):
        for domain, split, protocol, value in zip(DOMAINS, splits, protocols, row[1:]):
            effects.append(dict(dataset=domain, split=split, protocol=protocol,
                                effect=effect, comparison=f"MaskTopo minus {comparator}", gain_pp=value.lstrip("+")))
    write_csv("table3_graph_correspondence_gains.csv", list(effects[0]), effects)

    rows = table_rows(tables["ar"])
    assert len(rows) == 4 and all(len(row) == 5 for row in rows)
    branches = []
    for row in rows:
        for domain, protocol, cell in zip(DOMAINS[:2], protocols[:2], row[3:]):
            mean, sd = mean_sd(cell)
            branches.append(dict(dataset=domain, split="test", protocol=protocol,
                                 condition=row[0], adjacency=row[1], reachability=row[2],
                                 ba_mean_pct=mean, ba_sd_pct=sd, classifier_parameters="149123"))
    write_csv("table4_adjacency_reachability.csv", list(branches[0]), branches)
    print("Exported manuscript Tables 1-4: 4, 20, 8, and 8 aggregate rows.")


if __name__ == "__main__":
    main()

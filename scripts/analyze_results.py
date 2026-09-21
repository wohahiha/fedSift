"""Reproduce paired OOF loss intervals and the Figure 9 vector graphic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from fedsift.experiment_io import results_root
from fedsift.group_manifest import build_group_manifest, load_arff_rows, load_csv_rows
from fedsift.paired_bootstrap import paired_group_intervals
from fedsift.probability_contract import PROBABILITY_CLIP_EPSILON

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "fedavg_nonprivate", "dp_fedavg", "dp_fedadam", "fedsift",
    "fedsift_public_argmin_rule", "fedsift_uniform_schedule", "fedsift_without_sift",
)
SEEDS = {"pima": 20260907, "retinopathy": 20260908}


def analyze(predictions_path: Path):
    with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    output, evidence = [], {}
    for dataset, seed in SEEDS.items():
        if dataset == "pima":
            source = load_csv_rows(ROOT / "data/diabetes.csv", dataset=dataset, target="Outcome")
        else:
            source = load_arff_rows(
                ROOT / "data/raw/diabetic_retinopathy_debrecen/messidor_features.arff",
                dataset=dataset, target="Class",
            )
        manifest = build_group_manifest(source)
        group_ids = [None] * len(source.row_ids)
        for group in manifest["groups"]:
            for row_id in group["row_ids"]:
                group_ids[row_id] = group["group_id"]
        losses, folds = {}, None
        labels = np.asarray(source.labels)
        for method in METHODS:
            rows = sorted(
                (r for r in predictions if r["dataset"] == dataset and r["method"] == method),
                key=lambda r: int(r["row_id"]),
            )
            if [int(r["row_id"]) for r in rows] != list(source.row_ids):
                raise ValueError(f"{dataset}/{method}: OOF rows are incomplete or duplicated")
            if [int(r["label"]) for r in rows] != list(source.labels):
                raise ValueError(f"{dataset}/{method}: labels differ from the raw source")
            assignments = [int(r["outer_fold"]) for r in rows]
            if folds is not None and folds != assignments:
                raise ValueError("methods have different outer-fold assignments")
            folds = assignments
            p = np.asarray([float(r["probability"]) for r in rows])
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError("OOF probabilities must be finite and in [0, 1]")
            clipped = np.clip(p, PROBABILITY_CLIP_EPSILON, 1 - PROBABILITY_CLIP_EPSILON)
            losses[method] = {
                "log_loss": -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)),
                "brier_score": (p - labels) ** 2,
            }
        for group in manifest["groups"]:
            if len({folds[i] for i in group["row_ids"]}) != 1:
                raise ValueError("an exact feature group crosses outer folds")
        comparisons = [(metric, method) for metric in ("log_loss", "brier_score")
                       for method in METHODS if method != "fedsift"]
        differences = np.column_stack([
            losses["fedsift"][metric] - losses[method][metric] for metric, method in comparisons
        ])
        intervals = paired_group_intervals(differences, group_ids, seed=seed)
        for index, (metric, comparator) in enumerate(comparisons):
            output.append({
                "dataset": dataset, "metric": metric, "comparator": comparator,
                "effect": intervals.effect[index], "ci95_low": intervals.lower[index],
                "ci95_high": intervals.upper[index], "replicates": intervals.replicates,
                "unit": "exact_feature_group",
                "estimand": "pooled_out_of_fold_FedSift_minus_comparator_conditional_on_frozen_predictions",
            })
        evidence[dataset] = {
            "records": len(labels), "groups": intervals.group_count,
            "bootstrap_replicates": intervals.replicates, "seed": seed,
            "raw_data_sha256": source.source_sha256,
            "group_assignment_sha256": manifest["row_to_group_sha256"],
        }
    evidence["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    return output, evidence


def write_figure(rows, destination: Path):
    """Write a dependency-free, editable vector plot with four aligned panels."""
    comparators = (
        ("dp_fedavg", "DP-FedAvg"),
        ("fedsift_public_argmin_rule", "Public argmin"),
        ("fedsift_uniform_schedule", "Uniform noise"),
        ("fedsift_without_sift", "Without sift"),
    )
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="640" viewBox="0 0 1000 640">',
           '<rect width="1000" height="640" fill="white"/>',
           '<g font-family="Arial, sans-serif" font-size="15" fill="#202830">']
    for row_index, dataset in enumerate(SEEDS):
        for column, metric in enumerate(("log_loss", "brier_score")):
            selected = [next(r for r in rows if r["dataset"] == dataset and r["metric"] == metric
                             and r["comparator"] == method) for method, _ in comparators]
            low = min(0, *(r["ci95_low"] for r in selected))
            high = max(0, *(r["ci95_high"] for r in selected))
            pad = max((high - low) * 0.12, 1e-5)
            low, high = low - pad, high + pad
            x0, y0 = column * 500 + 150, row_index * 295 + 55
            position = lambda value: x0 + (value - low) / (high - low) * 320
            name = "Pima" if dataset == "pima" else "Debrecen"
            svg.append(f'<text x="{x0}" y="{y0 - 25}" font-weight="bold">{name}</text>')
            svg.append(f'<path d="M {position(0)} {y0 - 8} v 192" stroke="#888" stroke-dasharray="5 4"/>')
            for index, ((_, label), value) in enumerate(zip(comparators, selected)):
                y = y0 + index * 48
                left, right, point = [position(value[k]) for k in ("ci95_low", "ci95_high", "effect")]
                svg.append(f'<text x="{x0 - 12}" y="{y + 5}" text-anchor="end">{label}</text>')
                svg.append(f'<path d="M {left} {y} H {right} M {left} {y-5} v 10 M {right} {y-5} v 10" stroke="#245c7a" stroke-width="2" fill="none"/>')
                svg.append(f'<circle cx="{point}" cy="{y}" r="4" fill="#245c7a"/>')
            for tick in np.linspace(low, high, 5):
                svg.append(f'<text x="{position(tick)}" y="{y0 + 202}" text-anchor="middle" font-size="12">{tick:.3f}</text>')
            xlabel = "Log loss" if metric == "log_loss" else "Brier score"
            svg.append(f'<text x="{x0 + 160}" y="{y0 + 234}" text-anchor="middle">{xlabel}: FedSift minus comparator</text>')
    svg.append('</g></svg>')
    destination.write_text("\n".join(svg), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    arguments = parser.parse_args()
    destination = results_root() / "main_summary"
    source = arguments.predictions or destination / "oof_predictions.csv"
    rows, evidence = analyze(source)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "paired_group_bootstrap.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "paired_group_bootstrap.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    write_figure(rows, destination / "paired_effects.svg")
    print(json.dumps({"comparisons": len(rows), "output": str(destination), "datasets": list(SEEDS)}))


if __name__ == "__main__":
    main()

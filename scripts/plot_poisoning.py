"""Render the twelve poisoning results as an editable two-panel SVG."""
import argparse
import csv
import math
from pathlib import Path

from fedsift.experiment_io import results_root

METHODS = ("fedavg_nonprivate", "dp_fedavg", "fedsift")
ATTACKS = ("malicious_client_label_flip", "malicious_client_sign_scale")
DATASETS = ("pima", "retinopathy")
COLORS = ("#666666", "#c87924", "#28668a")


def write_figure(source: Path, destination: Path, language: str = "en"):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = {}
    for row in rows:
        key = (row["dataset_id"], row["method_id"], row["attack"])
        value = float(row["delta_log_loss"])
        if key in values or not math.isfinite(value):
            raise ValueError("Poisoning results contain duplicate or nonfinite values")
        values[key] = value
    expected = {(d, m, a) for d in DATASETS for m in METHODS for a in ATTACKS}
    if values.keys() != expected:
        raise ValueError("The figure requires all twelve registered poisoning results")
    chinese = language == "zh"
    labels = ("非私有 FedAvg", "DP-FedAvg", "FedSift") if chinese else ("Non-private FedAvg", "DP-FedAvg", "FedSift")
    attacks = ("标签翻转", "符号翻转并放大") if chinese else ("Label flipping", "Sign flip and scaling")
    ylabel = "Δ log loss（攻击后 − 干净训练）" if chinese else "Δ log loss (attacked − clean)"
    lower = min(-0.08, min(values.values()) - 0.05)
    upper = max(1.62, max(values.values()) + 0.08)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="604" viewBox="0 0 1200 604">',
           '<rect width="1200" height="604" fill="white"/>',
           '<g font-family="Times New Roman, SimSun, serif" font-size="24" fill="#222">']

    def marker(x, y, index):
        color = COLORS[index]
        if index == 0:
            return f'<rect x="{x-6}" y="{y-6}" width="12" height="12" fill="{color}"/>'
        if index == 1:
            return f'<path d="M {x} {y-8} L {x-8} {y+6} L {x+8} {y+6} Z" fill="{color}"/>'
        return f'<circle cx="{x}" cy="{y}" r="7" fill="{color}"/>'

    for index, label in enumerate(labels):
        x = 310 + index * 255
        svg.extend((marker(x, 32, index), f'<text x="{x+18}" y="40">{label}</text>'))
    for panel, dataset in enumerate(DATASETS):
        left, top, width, height = 105 + panel * 560, 80, 495, 430
        y = lambda value: top + height * (upper - value) / (upper - lower)
        title = "(a) Pima" if panel == 0 else "(b) Debrecen"
        svg.append(f'<text x="{left+width/2}" y="70" text-anchor="middle">{title}</text>')
        tick = math.ceil(lower / 0.25) * 0.25
        while tick <= upper:
            position = y(tick)
            svg.append(f'<path d="M {left} {position} h {width}" stroke="#e3e3e3"/>')
            svg.append(f'<text x="{left-12}" y="{position+8}" text-anchor="end">{tick:.2f}</text>')
            tick += 0.25
        svg.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#555"/>')
        svg.append(f'<path d="M {left} {y(0)} h {width}" stroke="#777" stroke-dasharray="7 6"/>')
        for attack_index, attack in enumerate(ATTACKS):
            center = left + width * (0.27 + attack_index * 0.46)
            svg.append(f'<text x="{center}" y="550" text-anchor="middle">{attacks[attack_index]}</text>')
            for method_index, method in enumerate(METHODS):
                svg.append(marker(center + (method_index - 1) * 19, y(values[(dataset, method, attack)]), method_index))
    svg.append(f'<text transform="translate(30 295) rotate(-90)" text-anchor="middle">{ylabel}</text>')
    svg.append('</g></svg>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    args = parser.parse_args()
    directory = results_root() / "followup/poisoning"
    write_figure(args.summary or directory / "poisoning_summary.csv",
                 args.output or directory / "poisoning_effects.svg", args.language)


if __name__ == "__main__":
    main()

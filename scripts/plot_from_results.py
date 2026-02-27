#!/usr/bin/env python3
"""
Generate presentation plots from an existing results.json.
No model, no GPU, no project imports needed — just numpy + matplotlib.

Produces:
  - confusion_self.png   (Reverse Classification: Generated Embeddings)
  - opposite_pairs.png   (Opposite Attribute Pair JSD Distances)

Usage:
    python scripts/plot_from_results.py
    python scripts/plot_from_results.py --results_json outputs/results.json --output_dir outputs/plots_local
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _abbrev(name: str) -> str:
    """'male, high-pitched, fast speed' -> 'M-Hi-Fast'"""
    lut = {
        "male": "M", "female": "F",
        "high-pitched": "Hi", "medium-pitched": "Med", "low-pitched": "Lo",
        "fast speed": "Fast", "measured speed": "Meas", "slow speed": "Slow",
    }
    parts = [p.strip() for p in name.split(",")]
    return "-".join(lut.get(p, p[:4]) for p in parts)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — Reverse Classification confusion matrix
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_data, group_names, title, save_path):
    cm = np.array(cm_data, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm / row_sums

    short = [_abbrev(g) for g in group_names]
    n = len(group_names)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)

    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5, color="white" if v > 0.5 else "black")

    ax.set_xlabel("Predicted Group", fontsize=11)
    ax.set_ylabel("True Group", fontsize=11)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — JSD Opposite Attribute Pair Distances
# ──────────────────────────────────────────────────────────────────────────────

def plot_opposite_pairs(opposite_analysis, save_path):
    flip_specs = [
        ("gender_flip", "Gender Flip (M<->F)"),
        ("pitch_flip",  "Pitch Flip (High<->Low)"),
        ("rate_flip",   "Rate Flip (Fast<->Slow)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (flip_key, label) in zip(axes, flip_specs):
        data   = opposite_analysis[flip_key]
        pairs  = data["pairs"]

        if not pairs:
            ax.set_title(label)
            ax.text(0.5, 0.5, "No pairs found",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        pair_labels, distances = [], []
        for p in pairs:
            a_parts = [x.strip() for x in p["group_a"].split(",")]
            b_parts = [x.strip() for x in p["group_b"].split(",")]
            shared  = [a for a, b in zip(a_parts, b_parts) if a == b]
            pair_labels.append(", ".join(shared) if shared else p["group_a"][:18])
            distances.append(p["distance"])

        y_pos = range(len(pairs))
        ax.barh(y_pos, distances, color="#5C6BC0", edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pair_labels, fontsize=7)
        ax.set_xlabel("JSD Distance")
        ax.set_title(label, fontsize=11)

        mean_val = data["stats"]["mean"]
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_val:.4f}")
        ax.legend(fontsize=8)

    fig.suptitle("Opposite Attribute Pair Distances", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Re-generate presentation plots from results.json"
    )
    p.add_argument(
        "--results_json",
        default="outputs/results.json",
        help="Path to results.json (default: outputs/results.json)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Where to save PNGs (default: <results_json parent>/plots_local)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    results_path = Path(args.results_json)
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found: {results_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else results_path.parent / "plots_local"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {results_path}")
    with open(results_path) as f:
        results = json.load(f)

    print(f"Output:  {output_dir.resolve()}\n")

    # ── Plot 1: Reverse Classification (self-consistency matrix) ─────────────
    if "reverse_classification" in results:
        data = results["reverse_classification"]["self"]
        plot_confusion_matrix(
            data["confusion_matrix"],
            data["group_names"],
            "Reverse Classification: Generated Embeddings",
            str(output_dir / "confusion_self.png"),
        )
    else:
        print("  [skip] reverse_classification not in results.json")

    # ── Plot 2: JSD Opposite Attribute Pairs ──────────────────────────────────
    gc = results.get("gmm_comparison", {})
    if "opposite_analysis" in gc:
        plot_opposite_pairs(
            gc["opposite_analysis"],
            str(output_dir / "opposite_pairs.png"),
        )
    else:
        print("  [skip] gmm_comparison.opposite_analysis not in results.json")

    print(f"\nDone. {len(list(output_dir.glob('*.png')))} PNG(s) in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

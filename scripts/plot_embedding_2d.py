#!/usr/bin/env python3
"""
Generate 2D LDA + seaborn KDE contour plots locally from sampled_embeddings.npz.

No model, no GPU, no project imports needed.
Requires: numpy, matplotlib, seaborn, scikit-learn

Usage:
    python scripts/plot_embedding_2d.py
    python scripts/plot_embedding_2d.py \
        --npz outputs/analysis_<timestamp>/sampled_embeddings.npz \
        --output_dir outputs/plots_local
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Attribute helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_attrs(group: str):
    parts = [p.strip() for p in group.split(",")]
    return (
        parts[0] if len(parts) > 0 else "?",
        parts[1] if len(parts) > 1 else "?",
        parts[2] if len(parts) > 2 else "?",
    )


PANELS = [
    (
        0,  # attr index
        {"male": "#4472C4", "female": "#ED7D31"},
        "Gender",
    ),
    (
        1,
        {"high-pitched": "#C00000", "medium-pitched": "#7030A0", "low-pitched": "#00B0F0"},
        "Pitch",
    ),
    (
        2,
        {"fast speed": "#E84040", "measured speed": "#F0A800", "slow speed": "#3BAA55"},
        "Speaking Rate",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Main plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_2d_lda(sampled_embeddings: dict, save_path: str) -> None:
    import seaborn as sns
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA

    # Stack all embeddings
    all_embs, all_groups = [], []
    for group, embs in sampled_embeddings.items():
        all_embs.append(embs)
        all_groups.extend([group] * len(embs))
    all_embs   = np.vstack(all_embs)
    all_groups = np.array(all_groups)

    attr_labels = [
        np.array([_get_attrs(g)[idx] for g in all_groups])
        for idx in range(3)
    ]

    # PCA-2 pre-computed for gender panel (LDA → 1 component only)
    pca2 = PCA(n_components=2, random_state=42).fit_transform(all_embs)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (attr_idx, color_map, attr_label) in zip(axes, PANELS):
        labels     = attr_labels[attr_idx]
        n_classes  = len(np.unique(labels))
        n_comp     = min(n_classes - 1, 2)

        lda      = LDA(n_components=n_comp)
        lda_proj = lda.fit_transform(all_embs, labels)

        if n_comp == 1:
            coords = np.column_stack([lda_proj.ravel(), pca2[:, 1]])
            xlabel, ylabel = "LDA 1 (gender axis)", "PCA 2"
        else:
            coords = lda_proj
            xlabel, ylabel = "LDA 1", "LDA 2"

        for attr_val in sorted(np.unique(labels)):
            mask  = labels == attr_val
            color = color_map.get(attr_val, "#888888")
            try:
                sns.kdeplot(
                    x=coords[mask, 0], y=coords[mask, 1],
                    fill=True, alpha=0.20, color=color, levels=5, ax=ax,
                )
                sns.kdeplot(
                    x=coords[mask, 0], y=coords[mask, 1],
                    fill=False, alpha=0.85, color=color, levels=5, ax=ax,
                    label=attr_val,
                )
            except Exception as e:
                print(f"  [warn] KDE failed for {attr_val}: {e}")

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"By {attr_label}", fontsize=12)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Embedding Space: LDA Projection + 2D KDE Contours (GMM-MDN samples)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Local 2D LDA + KDE contour plots from sampled_embeddings.npz"
    )
    p.add_argument(
        "--npz",
        default=None,
        help="Path to sampled_embeddings.npz "
             "(default: newest analysis_* directory under outputs/)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Where to save PNG (default: same directory as the .npz file)",
    )
    return p.parse_args()


def _find_latest_npz() -> Path:
    """Auto-discover the most recent sampled_embeddings.npz under outputs/."""
    candidates = sorted(
        Path("outputs").glob("analysis_*/sampled_embeddings.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # Fallback: flat outputs/
    flat = Path("outputs/sampled_embeddings.npz")
    if flat.exists():
        return flat
    raise FileNotFoundError(
        "No sampled_embeddings.npz found. "
        "Pass --npz explicitly or run analysis 2.8 first."
    )


def main():
    args  = parse_args()

    npz_path = Path(args.npz) if args.npz else _find_latest_npz()
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")

    output_dir = Path(args.output_dir) if args.output_dir else npz_path.parent / "plots_local"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {npz_path}")
    data   = np.load(str(npz_path), allow_pickle=True)
    embs   = data["embeddings"]   # [G, N, D]
    groups = data["groups"]       # [G] str array

    sampled_embeddings = {
        str(groups[i]): embs[i] for i in range(len(groups))
    }
    print(f"  {len(groups)} groups, {embs.shape[1]} samples each, dim={embs.shape[2]}")
    print(f"Output: {output_dir.resolve()}\n")

    save_path = str(output_dir / "embedding_lda_kde.png")
    plot_2d_lda(sampled_embeddings, save_path)

    print(f"\nDone. Plot saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

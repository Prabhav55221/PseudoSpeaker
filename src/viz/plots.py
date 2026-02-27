"""
Plotting functions for post-training GMM analysis.

All matplotlib rendering, headless-safe via Agg backend.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Dimensionality reduction
# =============================================================================

def run_dimensionality_reduction(
    embeddings: np.ndarray,
    method: str = "tsne",
    perplexity: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """
    Reduce high-dimensional embeddings to 2D for visualization.

    Falls back from UMAP to t-SNE if umap is not installed.

    Args:
        embeddings: [N, D] input embeddings
        method: "tsne" or "umap"
        perplexity: t-SNE perplexity parameter
        seed: random seed

    Returns:
        [N, 2] reduced coordinates
    """
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=seed)
            coords = reducer.fit_transform(embeddings)
            logger.info("Dimensionality reduction via UMAP complete")
            return coords
        except ImportError:
            logger.warning("umap-learn not installed, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(embeddings)
        logger.info("Dimensionality reduction via t-SNE complete")
        return coords

    raise ValueError(f"Unknown method: {method}")


# =============================================================================
# 2.5 — t-SNE / UMAP plots
# =============================================================================

def plot_tsne_by_source(
    coords: np.ndarray,
    labels_source: List[str],
    save_path: str,
) -> None:
    """
    Scatter plot: real (blue) vs generated (red).

    Args:
        coords: [N, 2] reduced coordinates
        labels_source: "real" or "generated" per point
        save_path: output image path
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    labels_arr = np.array(labels_source)
    for source, color, marker, alpha in [
        ("real", "#2196F3", "o", 0.4),
        ("generated", "#F44336", "x", 0.4),
    ]:
        mask = labels_arr == source
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, marker=marker, alpha=alpha, s=8, label=source,
        )

    ax.legend(fontsize=12)
    ax.set_title("t-SNE: Real vs Generated Embeddings", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_tsne_by_group(
    coords: np.ndarray,
    labels_group: List[str],
    save_path: str,
) -> None:
    """
    Scatter plot colored by attribute group (18 colors).

    Args:
        coords: [N, 2] reduced coordinates
        labels_group: group name per point
        save_path: output image path
    """
    unique_groups = sorted(set(labels_group))
    cmap = cm.get_cmap("tab20", len(unique_groups))

    fig, ax = plt.subplots(figsize=(14, 10))

    for i, group in enumerate(unique_groups):
        mask = np.array(labels_group) == group
        # Abbreviate label for legend
        short = _abbreviate_group(group)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)], s=8, alpha=0.4, label=short,
        )

    ax.legend(
        fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5),
        ncol=1, markerscale=3,
    )
    ax.set_title("t-SNE: Embeddings by Attribute Group", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_tsne_by_attribute(
    coords: np.ndarray,
    labels: List[str],
    attr_name: str,
    labels_source: List[str],
    save_path: str,
) -> None:
    """
    Scatter plot: color = attribute value, marker = source (o vs x).

    Args:
        coords: [N, 2] reduced coordinates
        labels: attribute value per point (e.g., gender)
        attr_name: attribute name for title ("gender", "pitch", "rate")
        labels_source: "real" or "generated" per point
        save_path: output image path
    """
    unique_vals = sorted(set(labels))
    colors = cm.get_cmap("Set1", len(unique_vals))
    color_map = {val: colors(i) for i, val in enumerate(unique_vals)}

    fig, ax = plt.subplots(figsize=(10, 8))

    labels_arr = np.array(labels)
    source_arr = np.array(labels_source)

    for val in unique_vals:
        for source, marker in [("real", "o"), ("generated", "x")]:
            mask = (labels_arr == val) & (source_arr == source)
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[color_map[val]], marker=marker, s=8, alpha=0.4,
                label=f"{val} ({source})",
            )

    ax.legend(fontsize=9, markerscale=3)
    ax.set_title(f"t-SNE by {attr_name.title()}", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# =============================================================================
# 2.7 — Confusion matrices
# =============================================================================

def plot_confusion_matrix(
    cm_data: np.ndarray,
    group_names: List[str],
    title: str,
    save_path: str,
    normalize: bool = True,
) -> None:
    """
    Plot confusion matrix as heatmap.

    Args:
        cm_data: [G, G] confusion matrix
        group_names: axis labels
        title: plot title
        save_path: output image path
        normalize: if True, normalize rows to sum to 1
    """
    if normalize:
        row_sums = cm_data.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_norm = cm_data.astype(float) / row_sums
    else:
        cm_norm = cm_data.astype(float)

    short_names = [_abbreviate_group(g) for g in group_names]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax)

    n = len(group_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color=text_color)

    ax.set_xlabel("Predicted Group", fontsize=11)
    ax.set_ylabel("True Group", fontsize=11)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# =============================================================================
# 2.6 — GMM distance plots
# =============================================================================

def plot_gmm_distance_heatmap(
    matrix: np.ndarray,
    group_names: List[str],
    metric_name: str,
    save_path: str,
) -> None:
    """
    Plot pairwise GMM distance matrix as heatmap with annotations.

    Args:
        matrix: [G, G] distance matrix
        group_names: axis labels
        metric_name: e.g., "JSD" or "KL"
        save_path: output image path
    """
    short_names = [_abbreviate_group(g) for g in group_names]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, interpolation="nearest", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label=metric_name)

    n = len(group_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)

    # Annotate cells
    vmax = matrix.max()
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text_color = "white" if val > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=4.5, color=text_color)

    ax.set_title(f"Pairwise {metric_name} Between Group GMMs", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_opposite_pair_bars(
    opposite_analysis: dict,
    save_path: str,
) -> None:
    """
    Grouped bar chart showing distances for gender/pitch/rate flips.

    Args:
        opposite_analysis: output of analyze_opposite_pairs()
        save_path: output image path
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (flip_type, label) in zip(axes, [
        ("gender_flip", "Gender Flip (M<->F)"),
        ("pitch_flip", "Pitch Flip (High<->Low)"),
        ("rate_flip", "Rate Flip (Fast<->Slow)"),
    ]):
        data = opposite_analysis[flip_type]
        pairs = data["pairs"]
        if not pairs:
            ax.set_title(label)
            ax.text(0.5, 0.5, "No pairs found", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        pair_labels = []
        distances = []
        for p in pairs:
            # Extract the differing attributes for label
            a_parts = [x.strip() for x in p["group_a"].split(",")]
            b_parts = [x.strip() for x in p["group_b"].split(",")]
            # Find the shared attributes
            shared = [a for a, b in zip(a_parts, b_parts) if a == b]
            short_label = ", ".join(shared) if shared else f"{p['group_a'][:15]}"
            pair_labels.append(short_label)
            distances.append(p["distance"])

        y_pos = range(len(pairs))
        ax.barh(y_pos, distances, color="#5C6BC0", edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pair_labels, fontsize=7)
        ax.set_xlabel("JSD Distance")
        ax.set_title(label, fontsize=11)

        # Add mean line
        mean_val = data["stats"]["mean"]
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean: {mean_val:.4f}")
        ax.legend(fontsize=8)

    fig.suptitle("Opposite Attribute Pair Distances", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# =============================================================================
# 2.1 — KL divergence bars
# =============================================================================

def plot_kl_divergence_bars(
    kl_values: Dict[str, float],
    save_path: str,
) -> None:
    """
    Horizontal bar chart of KL(predicted || reference) per group.

    Sorted by value, colored by gender, dashed mean line.

    Args:
        kl_values: group -> KL value
        save_path: output image path
    """
    # Sort by KL value
    sorted_groups = sorted(kl_values.keys(), key=lambda g: kl_values[g])
    values = [kl_values[g] for g in sorted_groups]
    short_names = [_abbreviate_group(g) for g in sorted_groups]

    # Color by gender
    colors = []
    for g in sorted_groups:
        if g.startswith("male"):
            colors.append("#42A5F5")
        elif g.startswith("female"):
            colors.append("#EF5350")
        else:
            colors.append("#78909C")

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_groups) * 0.4)))
    y_pos = range(len(sorted_groups))
    ax.barh(y_pos, values, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("KL Divergence (predicted || reference)", fontsize=11)
    ax.set_title("KL Divergence: Predicted vs Reference GMMs", fontsize=13)

    # Mean line
    mean_kl = np.mean(values)
    ax.axvline(mean_kl, color="black", linestyle="--", linewidth=1.5,
                label=f"Mean: {mean_kl:.4f}")
    ax.legend(fontsize=10)

    # Legend for gender colors
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#42A5F5", label="Male"),
        Patch(facecolor="#EF5350", label="Female"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="black", linestyle="--", label=f"Mean: {mean_kl:.4f}")
    ], fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# =============================================================================
# Report generation
# =============================================================================

def generate_report(
    output_dir: str,
    results: dict,
) -> None:
    """
    Write a markdown report summarizing all analysis results.

    Args:
        output_dir: directory containing plots/ subdirectory
        results: dict with keys for each analysis that was run
    """
    lines = [
        "# Post-Training Analysis Report",
        "",
        f"Generated from model checkpoint analysis.",
        "",
    ]

    # 2.5 — Visualization
    if "viz" in results:
        lines.extend([
            "## 2.5 Embedding Visualization",
            "",
            "### Real vs Generated",
            "![Real vs Generated](plots/tsne_real_vs_gen.png)",
            "",
            "### By Attribute Group",
            "![By Group](plots/tsne_by_group.png)",
            "",
            "### By Gender",
            "![By Gender](plots/tsne_by_gender.png)",
            "",
            "### By Pitch",
            "![By Pitch](plots/tsne_by_pitch.png)",
            "",
            "### By Rate",
            "![By Rate](plots/tsne_by_rate.png)",
            "",
        ])

    # 2.7 — Reverse Classification
    if "reverse_classification" in results:
        rc = results["reverse_classification"]
        lines.extend([
            "## 2.7 Reverse Classification",
            "",
            f"**Overall accuracy (real embeddings):** {rc['real']['overall_accuracy']:.4f}",
            "",
        ])
        # Per-group table
        lines.append("| Group | Accuracy |")
        lines.append("|-------|----------|")
        for group in rc["real"]["group_names"]:
            acc = rc["real"]["per_group_accuracy"].get(group, float("nan"))
            lines.append(f"| {group} | {acc:.4f} |")
        lines.append("")

        if "self" in rc:
            lines.extend([
                f"**Self-consistency accuracy (generated):** {rc['self']['overall_accuracy']:.4f}",
                "",
            ])

        lines.extend([
            "### Confusion Matrix (Real Embeddings)",
            "![Confusion Real](plots/confusion_real.png)",
            "",
        ])
        if "self" in rc:
            lines.extend([
                "### Confusion Matrix (Self-Consistency)",
                "![Confusion Self](plots/confusion_self.png)",
                "",
            ])

    # 2.6 — GMM Comparison
    if "gmm_comparison" in results:
        gc = results["gmm_comparison"]
        lines.extend([
            "## 2.6 GMM Comparison",
            "",
            "### Pairwise JSD Heatmap",
            "![JSD Heatmap](plots/gmm_distance_heatmap.png)",
            "",
        ])

        if "opposite_analysis" in gc:
            oa = gc["opposite_analysis"]
            lines.extend([
                "### Opposite Attribute Pair Distances",
                "![Opposite Pairs](plots/opposite_pairs.png)",
                "",
                "| Flip Type | Mean JSD | Std | Min | Max |",
                "|-----------|----------|-----|-----|-----|",
            ])
            for flip_type in ["gender_flip", "pitch_flip", "rate_flip"]:
                s = oa[flip_type]["stats"]
                lines.append(
                    f"| {flip_type} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |"
                )
            lines.append("")

    # 2.1 — KL Divergence
    if "kl_divergence" in results:
        kl = results["kl_divergence"]
        mean_kl = np.mean(list(kl.values()))
        lines.extend([
            "## 2.1 KL Divergence (Predicted vs Reference)",
            "",
            f"**Mean KL divergence:** {mean_kl:.4f}",
            "",
            "![KL Divergence](plots/kl_divergence.png)",
            "",
            "| Group | KL(pred \\|\\| ref) |",
            "|-------|-------------------|",
        ])
        for group in sorted(kl.keys(), key=lambda g: kl[g]):
            lines.append(f"| {group} | {kl[group]:.4f} |")
        lines.append("")

    report_path = Path(output_dir) / "report.md"
    report_path.write_text("\n".join(lines))
    logger.info(f"Report saved: {report_path}")


# =============================================================================
# 2.8 — Embedding distribution (KDE)
# =============================================================================

def plot_embedding_kde(
    sampled_embeddings: Dict[str, np.ndarray],
    save_path: str,
) -> None:
    """
    Smoothed KDE of GMM-MDN sampled embeddings, faceted by attribute.

    Projects all embeddings to the first PCA component, then aggregates
    across groups sharing the same attribute value (gender / pitch / rate)
    so each panel shows 2-3 clean curves rather than 18 noisy ones.

    Args:
        sampled_embeddings: group_name -> [N, D] array of sampled embeddings
        save_path: output image path
    """
    from scipy.stats import gaussian_kde
    from sklearn.decomposition import PCA

    # ── PCA projection ────────────────────────────────────────────────────────
    all_embs = np.vstack(list(sampled_embeddings.values()))
    pca = PCA(n_components=1, random_state=42)
    all_proj = pca.fit_transform(all_embs).ravel()

    idx = 0
    group_proj: Dict[str, np.ndarray] = {}
    for group, embs in sampled_embeddings.items():
        n = len(embs)
        group_proj[group] = all_proj[idx : idx + n]
        idx += n

    # ── Aggregate by attribute value ─────────────────────────────────────────
    def _get_attrs(group: str):
        parts = [p.strip() for p in group.split(",")]
        return (
            parts[0] if len(parts) > 0 else "?",
            parts[1] if len(parts) > 1 else "?",
            parts[2] if len(parts) > 2 else "?",
        )

    def _aggregate(attr_idx: int) -> Dict[str, np.ndarray]:
        buckets: Dict[str, list] = {}
        for group, proj in group_proj.items():
            key = _get_attrs(group)[attr_idx]
            buckets.setdefault(key, []).extend(proj.tolist())
        return {k: np.array(v) for k, v in buckets.items()}

    panels = [
        (
            _aggregate(0),
            {"male": "#4472C4", "female": "#ED7D31"},
            "Gender",
        ),
        (
            _aggregate(1),
            {
                "high-pitched": "#C00000",
                "medium-pitched": "#7030A0",
                "low-pitched": "#00B0F0",
            },
            "Pitch",
        ),
        (
            _aggregate(2),
            {
                "fast speed": "#E84040",
                "measured speed": "#F0A800",
                "slow speed": "#3BAA55",
            },
            "Speaking Rate",
        ),
    ]

    # ── KDE grid ─────────────────────────────────────────────────────────────
    x_pad = (all_proj.max() - all_proj.min()) * 0.15
    x_grid = np.linspace(all_proj.min() - x_pad, all_proj.max() + x_pad, 400)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (attr_data, color_map, attr_label) in zip(axes, panels):
        for attr_val, proj_vals in sorted(attr_data.items()):
            if len(proj_vals) < 2:
                continue
            color = color_map.get(attr_val, "#888888")
            try:
                kde = gaussian_kde(proj_vals, bw_method="silverman")
                density = kde(x_grid)
                density = density / density.max()  # normalise to [0, 1]
                ax.plot(
                    x_grid, density,
                    color=color, linewidth=2.2, label=attr_val,
                )
                ax.fill_between(x_grid, density, alpha=0.12, color=color)
            except Exception:
                pass

        ax.set_xlabel("PCA Component 1", fontsize=10)
        ax.set_ylabel("Density (normalised)", fontsize=10)
        ax.set_title(f"By {attr_label}", fontsize=12)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Sampled Embedding Distributions (GMM-MDN, smoothed KDE)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# =============================================================================
# Helpers
# =============================================================================

def _abbreviate_group(name: str) -> str:
    """Abbreviate group name for axis labels.

    e.g. "male, high-pitched, fast speed" -> "M-Hi-Fast"
    """
    parts = [p.strip() for p in name.split(",")]
    abbrevs = []
    for p in parts:
        p_lower = p.lower()
        if p_lower == "male":
            abbrevs.append("M")
        elif p_lower == "female":
            abbrevs.append("F")
        elif "high" in p_lower:
            abbrevs.append("Hi")
        elif "medium" in p_lower:
            abbrevs.append("Med")
        elif "low" in p_lower:
            abbrevs.append("Lo")
        elif "fast" in p_lower:
            abbrevs.append("Fast")
        elif "measured" in p_lower:
            abbrevs.append("Meas")
        elif "slow" in p_lower:
            abbrevs.append("Slow")
        else:
            abbrevs.append(p[:4])
    return "-".join(abbrevs)

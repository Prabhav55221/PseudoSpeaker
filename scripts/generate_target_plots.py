#!/usr/bin/env python3
"""
Generate synthetic "target" plots showing what good analysis results
should look like after training with contrastive loss.

Uses the same plotting functions as the real analysis pipeline.

Usage:
    python scripts/generate_target_plots.py --output_dir outputs/target_plots
    python scripts/generate_target_plots.py  # defaults to outputs/target_plots
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.viz.plots import (
    plot_tsne_by_source,
    plot_tsne_by_group,
    plot_tsne_by_attribute,
    plot_confusion_matrix,
    plot_gmm_distance_heatmap,
    plot_opposite_pair_bars,
    plot_kl_divergence_bars,
    generate_report,
)

# The 18 attribute groups (sorted)
GROUP_NAMES = [
    "female, high-pitched, fast speed",
    "female, high-pitched, measured speed",
    "female, high-pitched, slow speed",
    "female, low-pitched, fast speed",
    "female, low-pitched, measured speed",
    "female, low-pitched, slow speed",
    "female, medium-pitched, fast speed",
    "female, medium-pitched, measured speed",
    "female, medium-pitched, slow speed",
    "male, high-pitched, fast speed",
    "male, high-pitched, measured speed",
    "male, high-pitched, slow speed",
    "male, low-pitched, fast speed",
    "male, low-pitched, measured speed",
    "male, low-pitched, slow speed",
    "male, medium-pitched, fast speed",
    "male, medium-pitched, measured speed",
    "male, medium-pitched, slow speed",
]

# Per-group "difficulty" — groups with fewer real samples are harder.
# female-low is rarest in CapSpeech, male-medium is most common.
GROUP_DIFFICULTY = {
    "female, high-pitched, fast speed": 0.52,
    "female, high-pitched, measured speed": 0.58,
    "female, high-pitched, slow speed": 0.46,
    "female, low-pitched, fast speed": 0.32,    # rare group
    "female, low-pitched, measured speed": 0.38,  # rare
    "female, low-pitched, slow speed": 0.34,      # rare
    "female, medium-pitched, fast speed": 0.50,
    "female, medium-pitched, measured speed": 0.61,
    "female, medium-pitched, slow speed": 0.48,
    "male, high-pitched, fast speed": 0.56,
    "male, high-pitched, measured speed": 0.64,
    "male, high-pitched, slow speed": 0.50,
    "male, low-pitched, fast speed": 0.54,
    "male, low-pitched, measured speed": 0.60,
    "male, low-pitched, slow speed": 0.52,
    "male, medium-pitched, fast speed": 0.58,
    "male, medium-pitched, measured speed": 0.67,
    "male, medium-pitched, slow speed": 0.55,
}


def parse_attributes(name):
    parts = [p.strip() for p in name.split(",")]
    return parts[0], parts[1], parts[2]


def gender_idx(name):
    return 0 if name.startswith("female") else 1


def pitch_idx(name):
    if "high" in name:
        return 0
    elif "medium" in name:
        return 1
    return 2


def rate_idx(name):
    if "fast" in name:
        return 0
    elif "measured" in name:
        return 1
    return 2


# =========================================================================
# Synthetic t-SNE coordinates — mimics real x-vector t-SNE structure
# =========================================================================

def _speaker_island(rng, cx, cy, n_pts, tightness=0.8):
    """
    Generate a very tight speaker-level cluster (hotspot) at (cx, cy).
    Real t-SNE of x-vectors shows dense little clumps ~1-3 units across
    in a 120-unit space.
    """
    angle = rng.uniform(0, np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Tight elongated ellipse
    sx = tightness * rng.uniform(0.5, 1.3)
    sy = tightness * rng.uniform(0.3, 0.8)

    pts = rng.normal(0, 1, size=(n_pts, 2))
    pts[:, 0] *= sx
    pts[:, 1] *= sy

    rotated = np.column_stack([
        pts[:, 0] * cos_a - pts[:, 1] * sin_a,
        pts[:, 0] * sin_a + pts[:, 1] * cos_a,
    ])
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated


def generate_tsne_data(rng, n_real=200, n_gen=200):
    """
    Generate 2D t-SNE-like coordinates that mimic real x-vector t-SNE.

    Key visual features to match from real model output:
    - Dense little hotspot clumps (speaker islands) ~1-3 units across
    - Clear gaps between hotspots
    - Hotspots scattered across full -60 to +60 range
    - Some isolated peripheral hotspots
    - For target (improved): generated points also form hotspots that
      overlap with real, slightly more diffuse

    Structure:
    - Gender: broad spatial separation (female left/top, male right/bottom)
    - Pitch: secondary separation within gender region
    - Rate: subtle, mostly overlapping
    """
    all_coords = []
    labels_source = []
    labels_group = []
    labels_gender = []
    labels_pitch = []
    labels_rate = []

    # Broad spatial regions (range: -60 to +60)
    gender_base = {"female": (-15.0, 10.0), "male": (12.0, -8.0)}
    gender_scatter = {"female": 28.0, "male": 32.0}

    pitch_shift = {
        "high-pitched": (5.0, 12.0),
        "medium-pitched": (-2.0, 0.0),
        "low-pitched": (-4.0, -10.0),
    }

    rate_shift = {
        "fast speed": (-4.0, 2.0),
        "measured speed": (0.0, -1.0),
        "slow speed": (3.5, -1.5),
    }

    # Fewer speakers per group → more points per speaker → denser hotspots
    speakers_per_group = {
        "female, low-pitched, fast speed": 6,
        "female, low-pitched, measured speed": 7,
        "female, low-pitched, slow speed": 6,
        "female, medium-pitched, fast speed": 8,
        "female, medium-pitched, measured speed": 9,
        "female, medium-pitched, slow speed": 7,
        "female, high-pitched, fast speed": 8,
        "female, high-pitched, measured speed": 9,
        "female, high-pitched, slow speed": 7,
        "male, low-pitched, fast speed": 9,
        "male, low-pitched, measured speed": 10,
        "male, low-pitched, slow speed": 8,
        "male, medium-pitched, fast speed": 9,
        "male, medium-pitched, measured speed": 11,
        "male, medium-pitched, slow speed": 9,
        "male, high-pitched, fast speed": 8,
        "male, high-pitched, measured speed": 10,
        "male, high-pitched, slow speed": 8,
    }

    for group in GROUP_NAMES:
        gender, pitch, rate = parse_attributes(group)
        bx, by = gender_base[gender]
        scatter = gender_scatter[gender]
        px, py = pitch_shift[pitch]
        rx, ry = rate_shift[rate]

        region_cx = bx + px + rx
        region_cy = by + py + ry

        n_speakers = speakers_per_group[group]
        pts_per_speaker = n_real // n_speakers
        remainder = n_real - pts_per_speaker * n_speakers

        # Place speaker centers — scattered with some peripheral outliers
        speaker_centers = []
        for s in range(n_speakers):
            sx = region_cx + rng.normal(0, scatter * 0.38)
            sy = region_cy + rng.normal(0, scatter * 0.35)
            # ~20% chance of a peripheral outlier island (far from center)
            if rng.random() < 0.20:
                sx += rng.choice([-1, 1]) * rng.uniform(12, 30)
                sy += rng.choice([-1, 1]) * rng.uniform(8, 20)
            speaker_centers.append((sx, sy))

        # Real points: very tight clusters (hotspots)
        real_pts = []
        for s, (sx, sy) in enumerate(speaker_centers):
            n_pts = pts_per_speaker + (1 if s < remainder else 0)
            island = _speaker_island(rng, sx, sy, n_pts, tightness=0.8)
            real_pts.append(island)

        real_all = np.concatenate(real_pts, axis=0)
        all_coords.append(real_all)
        labels_source.extend(["real"] * len(real_all))
        labels_group.extend([group] * len(real_all))
        labels_gender.extend([gender] * len(real_all))
        labels_pitch.extend([pitch] * len(real_all))
        labels_rate.extend([rate] * len(real_all))

        # Generated points: form hotspots near same speaker centers
        # but slightly less tight and with small spatial offset.
        gen_pts_per_speaker = n_gen // n_speakers
        gen_remainder = n_gen - gen_pts_per_speaker * n_speakers

        gen_pts = []
        for s, (sx, sy) in enumerate(speaker_centers):
            n_pts = gen_pts_per_speaker + (1 if s < gen_remainder else 0)
            # Small offset + slightly wider clusters
            gx = sx + rng.normal(0, 1.5)
            gy = sy + rng.normal(0, 1.5)
            island = _speaker_island(rng, gx, gy, n_pts, tightness=1.6)
            gen_pts.append(island)

        # ~8% bridge/stray points scattered in the region (model imperfections)
        n_bridge = int(n_gen * 0.08)
        bridge_x = region_cx + rng.normal(0, scatter * 0.25, size=n_bridge)
        bridge_y = region_cy + rng.normal(0, scatter * 0.25, size=n_bridge)
        bridge_pts = np.column_stack([bridge_x, bridge_y])
        gen_pts.append(bridge_pts)

        gen_all = np.concatenate(gen_pts, axis=0)
        n_gen_actual = len(gen_all)
        all_coords.append(gen_all)
        labels_source.extend(["generated"] * n_gen_actual)
        labels_group.extend([group] * n_gen_actual)
        labels_gender.extend([gender] * n_gen_actual)
        labels_pitch.extend([pitch] * n_gen_actual)
        labels_rate.extend([rate] * n_gen_actual)

    coords = np.concatenate(all_coords, axis=0)
    return coords, labels_source, labels_group, labels_gender, labels_pitch, labels_rate


# =========================================================================
# Synthetic confusion matrix — realistic per-group variance
# =========================================================================

def generate_confusion_matrix(rng, accuracy_map, self_mode=False):
    """
    Generate confusion matrix with per-group accuracy from accuracy_map,
    and realistic off-diagonal confusion patterns.
    """
    n = len(GROUP_NAMES)
    n_samples = 200  # per group
    cm = np.zeros((n, n), dtype=int)

    for i in range(n):
        gi = gender_idx(GROUP_NAMES[i])
        pi = pitch_idx(GROUP_NAMES[i])
        ri = rate_idx(GROUP_NAMES[i])

        target_acc = accuracy_map[GROUP_NAMES[i]]
        if self_mode:
            target_acc = min(target_acc + rng.uniform(0.08, 0.15), 0.72)

        n_correct = int(round(target_acc * n_samples))
        n_wrong = n_samples - n_correct
        cm[i, i] = n_correct

        # Distribute errors with realistic confusion hierarchy
        confusion_weights = np.zeros(n)
        for j in range(n):
            if i == j:
                continue
            gj = gender_idx(GROUP_NAMES[j])
            pj = pitch_idx(GROUP_NAMES[j])
            rj = rate_idx(GROUP_NAMES[j])

            w = 0.0
            if gi == gj:
                # Same gender: much more likely confusion
                if pi == pj:
                    w = 3.0  # same gender+pitch
                elif ri == rj:
                    w = 2.5  # same gender+rate
                else:
                    w = 1.5  # same gender only
            else:
                # Cross-gender: rare confusion
                if pi == pj and ri == rj:
                    w = 0.4  # same pitch+rate, different gender
                elif pi == pj or ri == rj:
                    w = 0.15
                else:
                    w = 0.05

            # Add per-pair noise
            w *= rng.uniform(0.5, 1.8)
            confusion_weights[j] = w

        # Normalize and allocate errors
        if confusion_weights.sum() > 0:
            confusion_probs = confusion_weights / confusion_weights.sum()
            errors = rng.multinomial(n_wrong, confusion_probs)
            cm[i] += errors

    return cm


def generate_self_confusion(rng, accuracy_map):
    return generate_confusion_matrix(rng, accuracy_map, self_mode=True)


# =========================================================================
# Synthetic JSD distance matrix
# =========================================================================

def generate_jsd_matrix(rng):
    """
    Generate pairwise JSD matrix with realistic 3-tier structure.

    Higher JSD values than before, with more per-pair variance.
    Gender flip: ~0.04-0.07
    Pitch flip (high<->low): ~0.015-0.035
    Rate flip (fast<->slow): ~0.008-0.020
    """
    n = len(GROUP_NAMES)
    matrix = np.zeros((n, n))

    for i in range(n):
        gi = gender_idx(GROUP_NAMES[i])
        pi = pitch_idx(GROUP_NAMES[i])
        ri = rate_idx(GROUP_NAMES[i])
        for j in range(i + 1, n):
            gj = gender_idx(GROUP_NAMES[j])
            pj = pitch_idx(GROUP_NAMES[j])
            rj = rate_idx(GROUP_NAMES[j])

            val = 0.0

            # Gender effect: dominant
            if gi != gj:
                val += rng.uniform(0.048, 0.078)

            # Pitch effect
            pitch_dist = abs(pi - pj)
            if pitch_dist == 2:
                val += rng.uniform(0.022, 0.042)
            elif pitch_dist == 1:
                val += rng.uniform(0.010, 0.024)

            # Rate effect
            rate_dist = abs(ri - rj)
            if rate_dist == 2:
                val += rng.uniform(0.012, 0.028)
            elif rate_dist == 1:
                val += rng.uniform(0.005, 0.015)

            val = max(val, 0.0005)
            matrix[i, j] = val
            matrix[j, i] = val

    return matrix


def analyze_opposite_pairs_synthetic(matrix):
    from src.viz.gmm_analysis import analyze_opposite_pairs
    return analyze_opposite_pairs(matrix, GROUP_NAMES)


# =========================================================================
# Synthetic KL divergence — realistic per-group variance
# =========================================================================

def generate_kl_values(rng):
    """
    Generate per-group KL values with realistic variance.

    Groups with fewer samples (female-low) have higher KL.
    Some groups are clearly harder than others.
    """
    kl = {}
    for group in GROUP_NAMES:
        gender, pitch, rate = parse_attributes(group)

        # Base: male groups have more data → lower KL
        base = 14.0 if gender == "male" else 17.0

        # Low-pitched female is rare → higher KL (but not catastrophic)
        if gender == "female" and "low" in pitch:
            base += rng.uniform(6, 14)
        elif gender == "female" and "high" in pitch:
            base += rng.uniform(1, 5)
        elif gender == "male" and "high" in pitch:
            base += rng.uniform(0, 4)

        # Rate adds some noise
        if "fast" in rate:
            base += rng.uniform(-1, 3)
        elif "slow" in rate:
            base += rng.uniform(-1, 3)

        # Per-group noise
        base += rng.normal(0, 2)
        kl[group] = max(base, 6.0)

    return kl


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate target analysis plots")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/target_plots",
        help="Output directory (default: outputs/target_plots)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    print("Generating target plots...")
    print(f"Output: {output_dir}")

    # --- t-SNE plots ---
    print("  Generating t-SNE data...")
    coords, src, grp, gnd, pit, rat = generate_tsne_data(rng)

    plot_tsne_by_source(coords, src, str(plots_dir / "tsne_real_vs_gen.png"))
    plot_tsne_by_group(coords, grp, str(plots_dir / "tsne_by_group.png"))
    plot_tsne_by_attribute(coords, gnd, "gender", src, str(plots_dir / "tsne_by_gender.png"))
    plot_tsne_by_attribute(coords, pit, "pitch", src, str(plots_dir / "tsne_by_pitch.png"))
    plot_tsne_by_attribute(coords, rat, "rate", src, str(plots_dir / "tsne_by_rate.png"))

    # --- Confusion matrices ---
    print("  Generating confusion matrices...")
    cm_real = generate_confusion_matrix(rng, GROUP_DIFFICULTY)
    cm_self = generate_self_confusion(rng, GROUP_DIFFICULTY)

    plot_confusion_matrix(
        cm_real, GROUP_NAMES, "Reverse Classification: Real Embeddings",
        str(plots_dir / "confusion_real.png"), normalize=True,
    )
    plot_confusion_matrix(
        cm_self, GROUP_NAMES, "Self-Consistency: Generated Embeddings",
        str(plots_dir / "confusion_self.png"), normalize=True,
    )

    # --- JSD heatmap + opposite pairs ---
    print("  Generating JSD distance matrix...")
    jsd_matrix = generate_jsd_matrix(rng)
    plot_gmm_distance_heatmap(
        jsd_matrix, GROUP_NAMES, "JSD",
        str(plots_dir / "gmm_distance_heatmap.png"),
    )

    opposite_analysis = analyze_opposite_pairs_synthetic(jsd_matrix)
    plot_opposite_pair_bars(opposite_analysis, str(plots_dir / "opposite_pairs.png"))

    # --- KL divergence ---
    print("  Generating KL divergence values...")
    kl_values = generate_kl_values(rng)
    plot_kl_divergence_bars(kl_values, str(plots_dir / "kl_divergence.png"))

    # --- Collect results for report ---
    cm_real_norm = cm_real.astype(float) / cm_real.sum(axis=1, keepdims=True)
    cm_self_norm = cm_self.astype(float) / cm_self.sum(axis=1, keepdims=True)

    real_per_group = {}
    self_per_group = {}
    for i, g in enumerate(GROUP_NAMES):
        real_per_group[g] = float(cm_real_norm[i, i])
        self_per_group[g] = float(cm_self_norm[i, i])

    results = {
        "viz": {"num_points": len(coords)},
        "reverse_classification": {
            "real": {
                "overall_accuracy": float(np.diag(cm_real).sum() / cm_real.sum()),
                "per_group_accuracy": real_per_group,
                "confusion_matrix": cm_real,
                "group_names": GROUP_NAMES,
            },
            "self": {
                "overall_accuracy": float(np.diag(cm_self).sum() / cm_self.sum()),
                "per_group_accuracy": self_per_group,
                "confusion_matrix": cm_self,
                "group_names": GROUP_NAMES,
            },
        },
        "gmm_comparison": {
            "distance_matrix": jsd_matrix.tolist(),
            "opposite_analysis": opposite_analysis,
        },
        "kl_divergence": kl_values,
    }

    generate_report(str(output_dir), results)

    # Print summary
    real_acc = float(np.diag(cm_real).sum() / cm_real.sum())
    self_acc = float(np.diag(cm_self).sum() / cm_self.sum())
    mean_kl = float(np.mean(list(kl_values.values())))

    print()
    print("=" * 50)
    print("TARGET RESULTS SUMMARY")
    print("=" * 50)
    print(f"  Reverse classification (real):  {real_acc:.1%}")
    print(f"  Self-consistency (generated):   {self_acc:.1%}")
    print(f"  Gender flip mean JSD:           {opposite_analysis['gender_flip']['stats']['mean']:.4f}")
    print(f"  Pitch flip mean JSD:            {opposite_analysis['pitch_flip']['stats']['mean']:.4f}")
    print(f"  Rate flip mean JSD:             {opposite_analysis['rate_flip']['stats']['mean']:.4f}")
    print(f"  Mean KL divergence:             {mean_kl:.1f}")
    print()
    print("  Per-group classification accuracy:")
    for g in GROUP_NAMES:
        print(f"    {g:45s}  {real_per_group[g]:.1%}")
    print("=" * 50)
    print(f"Plots saved to: {plots_dir}")
    print(f"Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Post-training analysis CLI for GMM-MDN models.

Runs analyses 2.1, 2.5, 2.6, 2.7 and generates visualizations + report.

Usage:
    python scripts/analyze_model.py \
        --checkpoint outputs/best_model.pth \
        --embedding_dir /path/to/xvectors \
        --mapping_dir data/mappings \
        --augmented_texts data_augment/augmented_texts.json \
        --output_dir outputs/analysis
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch


def setup_logging(verbose: int) -> logging.Logger:
    """Configure logging based on verbosity level."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-training GMM analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required paths
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to best_model.pth checkpoint",
    )
    parser.add_argument(
        "--embedding_dir", type=str, required=True,
        help="Path to Kaldi ARK xvector directory",
    )
    parser.add_argument(
        "--mapping_dir", type=str, required=True,
        help="Path to directory with test.json",
    )
    parser.add_argument(
        "--augmented_texts", type=str, required=True,
        help="Path to augmented_texts.json",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for plots/ and report.md",
    )

    # Analysis selection
    parser.add_argument(
        "--analyses", type=str, nargs="+",
        default=["2.5", "2.6", "2.7", "2.1"],
        help="Which analyses to run",
    )

    # Parameters
    parser.add_argument(
        "--max_per_group", type=int, default=200,
        help="Max real embeddings per group",
    )
    parser.add_argument(
        "--num_generated", type=int, default=200,
        help="Generated embeddings per group for visualization",
    )
    parser.add_argument(
        "--num_mc_samples", type=int, default=10000,
        help="MC samples for KL/JSD estimation",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Computation device (cuda or cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )

    # Logging
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.verbose)

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    logger.info(f"Using device: {args.device}")

    # Create output directories
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # =========================================================================
    # Phase 1: Load model and data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Phase 1: Loading model and data")
    logger.info("=" * 60)

    # Load model
    from src.models.gmm_mdn import GMMMDN

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint = GMMMDN.load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    logger.info(f"Model loaded: {model.num_components} components, {model.embedding_dim}D embeddings")

    # Load augmented texts to get group names
    with open(args.augmented_texts, "r") as f:
        augmented_texts = json.load(f)
    group_names = sorted(augmented_texts.keys())
    logger.info(f"Found {len(group_names)} attribute groups")

    # Get GMM params for all groups
    from src.viz.gmm_analysis import (
        get_all_group_gmm_params,
        load_reference_embeddings_by_group,
    )

    logger.info("Getting GMM parameters for all groups...")
    group_gmm_params = get_all_group_gmm_params(
        model, args.augmented_texts, text_index=0
    )

    # Load reference embeddings (do this early due to HyperionEmbeddingLoader's os.chdir)
    test_json_path = Path(args.mapping_dir) / "test.json"
    logger.info(f"Loading reference embeddings from: {test_json_path}")
    real_embeddings_by_group = load_reference_embeddings_by_group(
        str(test_json_path),
        args.embedding_dir,
        max_per_group=args.max_per_group,
        seed=args.seed,
    )
    total_real = sum(len(e) for e in real_embeddings_by_group.values())
    logger.info(f"Loaded {total_real} real embeddings across {len(real_embeddings_by_group)} groups")

    # Results dict to collect all analysis outputs
    results = {}

    # =========================================================================
    # Phase 2: Run analyses
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Phase 2: Running analyses")
    logger.info("=" * 60)

    # Import analysis functions
    from src.viz.gmm_analysis import (
        prepare_visualization_data,
        compute_reverse_classification_metrics,
        compute_self_consistency,
        compute_pairwise_gmm_distances,
        analyze_opposite_pairs,
        compute_kl_real_vs_predicted,
    )
    from src.viz.plots import (
        run_dimensionality_reduction,
        plot_tsne_by_source,
        plot_tsne_by_group,
        plot_tsne_by_attribute,
        plot_confusion_matrix,
        plot_gmm_distance_heatmap,
        plot_opposite_pair_bars,
        plot_kl_divergence_bars,
        generate_report,
    )

    # --- 2.5: Visualization ---
    if "2.5" in args.analyses:
        logger.info("-" * 40)
        logger.info("Running 2.5: Embedding Visualization")
        logger.info("-" * 40)

        viz_data = prepare_visualization_data(
            model,
            real_embeddings_by_group,
            args.augmented_texts,
            num_generated_per_group=args.num_generated,
            temperature=args.temperature,
            device=args.device,
        )
        logger.info(f"Prepared {len(viz_data['embeddings'])} embeddings for visualization")

        logger.info("Running t-SNE dimensionality reduction...")
        coords = run_dimensionality_reduction(
            viz_data["embeddings"],
            method="tsne",
            perplexity=30,
            seed=args.seed,
        )

        # Generate plots
        plot_tsne_by_source(
            coords, viz_data["labels_source"],
            str(plots_dir / "tsne_real_vs_gen.png"),
        )
        plot_tsne_by_group(
            coords, viz_data["labels_group"],
            str(plots_dir / "tsne_by_group.png"),
        )
        plot_tsne_by_attribute(
            coords, viz_data["labels_gender"], "gender",
            viz_data["labels_source"],
            str(plots_dir / "tsne_by_gender.png"),
        )
        plot_tsne_by_attribute(
            coords, viz_data["labels_pitch"], "pitch",
            viz_data["labels_source"],
            str(plots_dir / "tsne_by_pitch.png"),
        )
        plot_tsne_by_attribute(
            coords, viz_data["labels_rate"], "rate",
            viz_data["labels_source"],
            str(plots_dir / "tsne_by_rate.png"),
        )

        results["viz"] = {"num_points": len(viz_data["embeddings"])}
        logger.info("2.5 complete")

    # --- 2.7: Reverse Classification ---
    if "2.7" in args.analyses:
        logger.info("-" * 40)
        logger.info("Running 2.7: Reverse Classification")
        logger.info("-" * 40)

        # Real embeddings classification
        rc_real = compute_reverse_classification_metrics(
            real_embeddings_by_group,
            group_gmm_params,
            group_names,
            device=args.device,
        )
        logger.info(f"Real embeddings overall accuracy: {rc_real['overall_accuracy']:.4f}")

        plot_confusion_matrix(
            rc_real["confusion_matrix"],
            group_names,
            "Reverse Classification: Real Embeddings",
            str(plots_dir / "confusion_real.png"),
            normalize=True,
        )

        # Self-consistency
        rc_self = compute_self_consistency(
            model,
            group_gmm_params,
            group_names,
            num_samples_per_group=args.num_generated,
            temperature=args.temperature,
            device=args.device,
        )
        logger.info(f"Self-consistency overall accuracy: {rc_self['overall_accuracy']:.4f}")

        plot_confusion_matrix(
            rc_self["confusion_matrix"],
            group_names,
            "Self-Consistency: Generated Embeddings",
            str(plots_dir / "confusion_self.png"),
            normalize=True,
        )

        results["reverse_classification"] = {
            "real": rc_real,
            "self": rc_self,
        }
        logger.info("2.7 complete")

    # --- 2.6: GMM Comparison ---
    if "2.6" in args.analyses:
        logger.info("-" * 40)
        logger.info("Running 2.6: GMM Comparison")
        logger.info("-" * 40)

        logger.info("Computing pairwise JSD matrix...")
        distance_matrix = compute_pairwise_gmm_distances(
            group_gmm_params,
            group_names,
            metric="jsd",
            num_samples=args.num_mc_samples,
            device=args.device,
        )

        plot_gmm_distance_heatmap(
            distance_matrix,
            group_names,
            "JSD",
            str(plots_dir / "gmm_distance_heatmap.png"),
        )

        opposite_analysis = analyze_opposite_pairs(distance_matrix, group_names)
        plot_opposite_pair_bars(
            opposite_analysis,
            str(plots_dir / "opposite_pairs.png"),
        )

        # Log summary stats
        for flip_type in ["gender_flip", "pitch_flip", "rate_flip"]:
            stats = opposite_analysis[flip_type]["stats"]
            logger.info(f"{flip_type}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

        results["gmm_comparison"] = {
            "distance_matrix": distance_matrix.tolist(),
            "opposite_analysis": opposite_analysis,
        }
        logger.info("2.6 complete")

    # --- 2.1: KL Divergence ---
    if "2.1" in args.analyses:
        logger.info("-" * 40)
        logger.info("Running 2.1: KL Divergence (Real vs Predicted)")
        logger.info("-" * 40)

        kl_values = compute_kl_real_vs_predicted(
            real_embeddings_by_group,
            group_gmm_params,
            n_components=15,
            num_mc_samples=args.num_mc_samples,
            seed=args.seed,
            device=args.device,
        )

        plot_kl_divergence_bars(
            kl_values,
            str(plots_dir / "kl_divergence.png"),
        )

        mean_kl = np.mean(list(kl_values.values()))
        logger.info(f"Mean KL divergence: {mean_kl:.4f}")

        results["kl_divergence"] = kl_values
        logger.info("2.1 complete")

    # =========================================================================
    # Phase 3: Generate report
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Phase 3: Generating report")
    logger.info("=" * 60)

    generate_report(str(output_dir), results)

    # Save raw results as JSON
    results_json_path = output_dir / "results.json"

    def _convert_for_json(obj):
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert_for_json(v) for v in obj]
        return obj

    with open(results_json_path, "w") as f:
        json.dump(_convert_for_json(results), f, indent=2)
    logger.info(f"Results saved: {results_json_path}")

    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Report: {output_dir / 'report.md'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

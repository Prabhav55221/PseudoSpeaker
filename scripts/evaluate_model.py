#!/usr/bin/env python3
"""
Comprehensive evaluation script for trained GMM-MDN model.

Generates embeddings for multiple text descriptions and computes metrics
to evaluate the quality of the generated speaker embeddings.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial.distance import mahalanobis, cdist
from scipy.stats import wasserstein_distance

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gmm_mdn import GMMMDN
from src.utils.embedding_loader import HyperionEmbeddingLoader


def setup_logging(verbosity: int) -> logging.Logger:
    """Setup logging with specified verbosity."""
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if verbosity >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbosity >= 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    return logger


def load_model(checkpoint_path: Path, device: str, logger: logging.Logger) -> GMMMDN:
    """Load trained GMM-MDN model from checkpoint."""
    logger.info(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint
    config = checkpoint.get('config', {})

    # Create model
    model = GMMMDN(
        num_components=config.get('num_gmm_components', 15),
        embedding_dim=config.get('embedding_dim', 192),
        hidden_dim=config.get('hidden_dim', 512),
        text_encoder_name=config.get('text_encoder_name', 'all-MiniLM-L6-v2'),
        freeze_encoder=config.get('freeze_encoder', True),
        device=device
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded successfully!")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"  Validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    logger.info(f"\n{model}")

    return model


def generate_samples(
    model: GMMMDN,
    text: str,
    num_samples: int,
    temperature: float,
    device: str,
    logger: logging.Logger
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate speaker embeddings from text description.

    Returns:
        embeddings: [num_samples, embedding_dim]
        gmm_params: Dict with 'means', 'stds', 'weights'
    """
    logger.info(f"Generating {num_samples} samples for: '{text}'")

    with torch.no_grad():
        # Sample from model
        samples = model.sample(
            text=text,
            num_samples=num_samples,
            temperature=temperature
        )

        embeddings = samples.cpu().numpy()  # [num_samples, D]

        # Get GMM parameters for analysis
        weights, means, log_vars = model.get_gmm_params(text)

        # Convert to numpy
        weights_np = weights.cpu().numpy()  # [K]
        means_np = means.cpu().numpy()  # [K, D]
        stds_np = torch.exp(0.5 * log_vars).cpu().numpy()  # [K, D]

    gmm_params = {
        'means': means_np,
        'stds': stds_np,
        'weights': weights_np
    }

    logger.info(f"Generated embeddings shape: {embeddings.shape}")

    return embeddings, gmm_params


def compute_diversity_score(embeddings: np.ndarray) -> float:
    """
    Compute diversity score: average pairwise distance.

    Higher is better (more diverse samples).
    """
    if len(embeddings) < 2:
        return 0.0

    # Compute pairwise Euclidean distances
    distances = cdist(embeddings, embeddings, metric='euclidean')

    # Get upper triangular (excluding diagonal)
    mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
    pairwise_dists = distances[mask]

    # Average distance
    diversity = float(np.mean(pairwise_dists))

    return diversity


def compute_coverage_score(
    embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    percentile: float = 95
) -> float:
    """
    Compute coverage score: fraction of reference data within threshold.

    For each generated sample, find nearest reference embedding.
    Coverage = fraction of reference embeddings that are "close" to generated samples.

    Higher is better (better coverage of real distribution).
    """
    if len(reference_embeddings) == 0:
        return 0.0

    # Compute distances from each reference to nearest generated sample
    distances = cdist(reference_embeddings, embeddings, metric='euclidean')
    min_distances = np.min(distances, axis=1)

    # Threshold: 95th percentile of distances
    threshold = np.percentile(min_distances, percentile)

    # Coverage: fraction within threshold
    coverage = float(np.mean(min_distances <= threshold))

    return coverage


def compute_mahalanobis_distance(
    predicted_gmm: Dict[str, np.ndarray],
    reference_embeddings: np.ndarray
) -> float:
    """
    Compute Mahalanobis distance between predicted GMM and reference distribution.

    Uses the covariance of reference data to measure distance from predicted mean.
    """
    # Compute reference statistics
    ref_mean = np.mean(reference_embeddings, axis=0)
    ref_cov = np.cov(reference_embeddings, rowvar=False)

    # Regularize covariance to ensure invertibility
    reg = 1e-6
    ref_cov_reg = ref_cov + reg * np.eye(ref_cov.shape[0])

    try:
        ref_cov_inv = np.linalg.inv(ref_cov_reg)
    except np.linalg.LinAlgError:
        # If still singular, use pseudo-inverse
        ref_cov_inv = np.linalg.pinv(ref_cov_reg)

    # Compute weighted mean of predicted GMM
    pred_mean = np.average(predicted_gmm['means'], weights=predicted_gmm['weights'], axis=0)

    # Mahalanobis distance
    diff = pred_mean - ref_mean
    maha_dist = float(np.sqrt(diff @ ref_cov_inv @ diff))

    return maha_dist


def compute_kl_divergence(
    predicted_gmm: Dict[str, np.ndarray],
    reference_embeddings: np.ndarray,
    num_samples: int = 10000
) -> float:
    """
    Estimate KL divergence between predicted GMM and reference distribution.

    Uses Monte Carlo sampling to approximate KL(predicted || reference).
    """
    # Sample from predicted GMM
    K, D = predicted_gmm['means'].shape
    weights = predicted_gmm['weights']

    # Sample component indices
    component_indices = np.random.choice(K, size=num_samples, p=weights)

    # Sample from each component
    samples = []
    for k in range(K):
        num_k = np.sum(component_indices == k)
        if num_k > 0:
            mean_k = predicted_gmm['means'][k]
            std_k = predicted_gmm['stds'][k]
            samples_k = np.random.normal(mean_k, std_k, size=(num_k, D))
            samples.append(samples_k)

    pred_samples = np.vstack(samples)

    # Estimate densities (simplified: use Gaussian KDE or histogram)
    # For high-dimensional data, we'll use average log-likelihood as proxy

    # Compute log p(x) under predicted GMM for pred_samples
    log_prob_pred = 0.0
    for i in range(num_samples):
        x = pred_samples[i]
        # Log probability under GMM
        log_probs = []
        for k in range(K):
            mean_k = predicted_gmm['means'][k]
            std_k = predicted_gmm['stds'][k]
            # Log probability under Gaussian
            log_p = -0.5 * np.sum(((x - mean_k) / std_k) ** 2)
            log_p -= 0.5 * D * np.log(2 * np.pi)
            log_p -= np.sum(np.log(std_k))
            log_p += np.log(weights[k])
            log_probs.append(log_p)
        # Log-sum-exp
        max_log_p = np.max(log_probs)
        log_prob_pred += max_log_p + np.log(np.sum(np.exp(log_probs - max_log_p)))

    log_prob_pred /= num_samples

    # Estimate log q(x) under reference distribution (use Gaussian approximation)
    ref_mean = np.mean(reference_embeddings, axis=0)
    ref_cov = np.cov(reference_embeddings, rowvar=False)
    ref_cov_reg = ref_cov + 1e-6 * np.eye(ref_cov.shape[0])

    log_prob_ref = 0.0
    for i in range(num_samples):
        x = pred_samples[i]
        diff = x - ref_mean
        try:
            log_p = -0.5 * diff @ np.linalg.inv(ref_cov_reg) @ diff
            log_p -= 0.5 * D * np.log(2 * np.pi)
            log_p -= 0.5 * np.linalg.slogdet(ref_cov_reg)[1]
            log_prob_ref += log_p
        except:
            log_prob_ref += -1e10  # Large penalty if computation fails

    log_prob_ref /= num_samples

    # KL divergence ≈ E[log p(x)] - E[log q(x)]
    kl_div = float(log_prob_pred - log_prob_ref)

    return kl_div


def load_reference_embeddings(
    embedding_dir: Path,
    mapping_dir: Path,
    split: str,
    max_samples: int,
    logger: logging.Logger
) -> np.ndarray:
    """Load reference embeddings from dataset."""
    logger.info(f"Loading reference embeddings from {split} split...")

    # Load mapping
    mapping_path = mapping_dir / f"{split}.json"
    with open(mapping_path, 'r') as f:
        data = json.load(f)

    logger.info(f"Found {len(data)} samples in {split} split")

    # Subsample if too many
    if len(data) > max_samples:
        logger.info(f"Subsampling to {max_samples} samples")
        indices = np.random.choice(len(data), size=max_samples, replace=False)
        data = [data[i] for i in indices]

    # Load embeddings
    loader = HyperionEmbeddingLoader(embedding_dir, logger=logger)

    embeddings = []
    for i, item in enumerate(data):
        if i % 100 == 0:
            logger.debug(f"Loading embedding {i+1}/{len(data)}")

        try:
            emb = loader.load_embedding(item['audio_id'])
            embeddings.append(emb)
        except Exception as e:
            logger.warning(f"Failed to load {item['audio_id']}: {e}")

    loader.close()

    embeddings = np.array(embeddings)
    logger.info(f"Loaded {len(embeddings)} reference embeddings, shape: {embeddings.shape}")

    return embeddings


def save_results(
    output_dir: Path,
    text_descriptions: List[str],
    all_embeddings: Dict[str, np.ndarray],
    all_gmm_params: Dict[str, Dict[str, np.ndarray]],
    all_metrics: Dict[str, Dict[str, float]],
    logger: logging.Logger
):
    """Save evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    for i, text in enumerate(text_descriptions):
        emb_path = output_dir / f"embeddings_{i+1}.npy"
        np.save(emb_path, all_embeddings[text])
        logger.info(f"Saved embeddings to: {emb_path}")

    # Save GMM parameters
    gmm_path = output_dir / "gmm_params.json"
    gmm_serializable = {}
    for i, text in enumerate(text_descriptions):
        gmm_serializable[f"description_{i+1}"] = {
            'text': text,
            'means': all_gmm_params[text]['means'].tolist(),
            'stds': all_gmm_params[text]['stds'].tolist(),
            'weights': all_gmm_params[text]['weights'].tolist()
        }

    with open(gmm_path, 'w') as f:
        json.dump(gmm_serializable, f, indent=2)
    logger.info(f"Saved GMM parameters to: {gmm_path}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    metrics_serializable = {}
    for i, text in enumerate(text_descriptions):
        metrics_serializable[f"description_{i+1}"] = {
            'text': text,
            **all_metrics[text]
        }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_path}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)

    for i, text in enumerate(text_descriptions):
        metrics = all_metrics[text]
        logger.info(f"\nDescription {i+1}: '{text}'")
        logger.info(f"  Diversity Score:     {metrics['diversity_score']:.4f}")
        logger.info(f"  Coverage Score:      {metrics['coverage_score']:.4f}")
        logger.info(f"  Mahalanobis Dist:    {metrics['mahalanobis_distance']:.4f}")
        logger.info(f"  KL Divergence:       {metrics['kl_divergence']:.4f}")
        logger.info(f"  Avg Pairwise Dist:   {metrics['avg_pairwise_distance']:.4f}")

    # Overall statistics
    logger.info("\n" + "="*80)
    logger.info("OVERALL STATISTICS")
    logger.info("="*80)

    all_diversity = [m['diversity_score'] for m in all_metrics.values()]
    all_coverage = [m['coverage_score'] for m in all_metrics.values()]
    all_mahalanobis = [m['mahalanobis_distance'] for m in all_metrics.values()]

    logger.info(f"Average Diversity Score:     {np.mean(all_diversity):.4f} ± {np.std(all_diversity):.4f}")
    logger.info(f"Average Coverage Score:      {np.mean(all_coverage):.4f} ± {np.std(all_coverage):.4f}")
    logger.info(f"Average Mahalanobis Dist:    {np.mean(all_mahalanobis):.4f} ± {np.std(all_mahalanobis):.4f}")

    logger.info("\nGuidelines:")
    logger.info("  Diversity Score:  0.7-0.95 is good, >0.85 is excellent")
    logger.info("  Coverage Score:   0.8-0.98 is good, >0.90 is excellent")
    logger.info("  Mahalanobis Dist: Lower is better (closer to real distribution)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GMM-MDN model")

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    # Data
    parser.add_argument(
        "--embedding_dir",
        type=str,
        help="Path to x-vector embeddings directory (for reference)"
    )

    parser.add_argument(
        "--mapping_dir",
        type=str,
        help="Path to train/dev/test mapping files (for reference)"
    )

    parser.add_argument(
        "--reference_split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Which split to use for reference embeddings"
    )

    parser.add_argument(
        "--max_reference_samples",
        type=int,
        default=1000,
        help="Maximum number of reference samples to load"
    )

    # Sampling
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate per description"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more diverse)"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Verbosity level: (default INFO), -v for INFO, -vv for DEBUG"
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.verbose)

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Paths
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    # Define test descriptions (diverse set covering different attributes)
    text_descriptions = [
        "A male speaker with deep voice and slow speaking rate",
        "A female speaker with high-pitched voice and fast speaking rate",
        "A male speaker with medium pitch and normal speaking rate",
        "A female speaker with low voice and measured pace",
        "A speaker with neutral tone and moderate tempo"
    ]

    logger.info("="*80)
    logger.info("GMM-MDN Model Evaluation")
    logger.info("="*80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Num samples per description: {args.num_samples}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info("="*80)

    # Load model
    model = load_model(checkpoint_path, args.device, logger)

    # Load reference embeddings if provided
    reference_embeddings = None
    if args.embedding_dir and args.mapping_dir:
        reference_embeddings = load_reference_embeddings(
            Path(args.embedding_dir),
            Path(args.mapping_dir),
            args.reference_split,
            args.max_reference_samples,
            logger
        )
    else:
        logger.warning("No reference embeddings provided, skipping coverage/Mahalanobis metrics")

    # Generate samples and compute metrics
    all_embeddings = {}
    all_gmm_params = {}
    all_metrics = {}

    for i, text in enumerate(text_descriptions):
        logger.info("\n" + "="*80)
        logger.info(f"Description {i+1}/{len(text_descriptions)}")
        logger.info("="*80)

        # Generate samples
        embeddings, gmm_params = generate_samples(
            model=model,
            text=text,
            num_samples=args.num_samples,
            temperature=args.temperature,
            device=args.device,
            logger=logger
        )

        all_embeddings[text] = embeddings
        all_gmm_params[text] = gmm_params

        # Compute metrics
        metrics = {}

        # Diversity score (always computable)
        diversity = compute_diversity_score(embeddings)
        metrics['diversity_score'] = diversity
        logger.info(f"Diversity Score: {diversity:.4f}")

        # Average pairwise distance
        if len(embeddings) > 1:
            distances = cdist(embeddings, embeddings, metric='euclidean')
            mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
            avg_dist = float(np.mean(distances[mask]))
            metrics['avg_pairwise_distance'] = avg_dist
            logger.info(f"Average Pairwise Distance: {avg_dist:.4f}")
        else:
            metrics['avg_pairwise_distance'] = 0.0

        # Reference-based metrics (if available)
        if reference_embeddings is not None:
            # Coverage score
            coverage = compute_coverage_score(embeddings, reference_embeddings)
            metrics['coverage_score'] = coverage
            logger.info(f"Coverage Score: {coverage:.4f}")

            # Mahalanobis distance
            maha_dist = compute_mahalanobis_distance(gmm_params, reference_embeddings)
            metrics['mahalanobis_distance'] = maha_dist
            logger.info(f"Mahalanobis Distance: {maha_dist:.4f}")

            # KL divergence (expensive, optional)
            kl_div = compute_kl_divergence(gmm_params, reference_embeddings, num_samples=1000)
            metrics['kl_divergence'] = kl_div
            logger.info(f"KL Divergence: {kl_div:.4f}")
        else:
            metrics['coverage_score'] = None
            metrics['mahalanobis_distance'] = None
            metrics['kl_divergence'] = None

        all_metrics[text] = metrics

    # Save results
    save_results(
        output_dir=output_dir,
        text_descriptions=text_descriptions,
        all_embeddings=all_embeddings,
        all_gmm_params=all_gmm_params,
        all_metrics=all_metrics,
        logger=logger
    )

    logger.info("\n" + "="*80)
    logger.info("Evaluation complete!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\nTo convert embeddings to audio, see TTS_CONVERSION_GUIDE.md")


if __name__ == "__main__":
    main()

"""
GMM (Gaussian Mixture Model) utilities for training and inference.

Implements:
- NLL (Negative Log-Likelihood) loss computation
- Sampling from GMM distributions
- Numerically stable log-space operations
"""

import torch
import torch.nn.functional as F
from typing import Tuple

# Numerical stability constants
EPSILON = 1e-6
LOG_2PI = 1.8378770664093453  # log(2 * pi)


def compute_gmm_nll(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    log_vars: torch.Tensor
) -> torch.Tensor:
    """
    Compute negative log-likelihood of embeddings under GMM.

    Uses diagonal covariance matrices for efficiency.
    All computations in log-space for numerical stability.

    Args:
        embeddings: Target embeddings [batch_size, embedding_dim]
        weights: GMM mixing coefficients [batch_size, num_components]
                 (unnormalized, will be softmaxed)
        means: GMM component means [batch_size, num_components, embedding_dim]
        log_vars: GMM component log-variances [batch_size, num_components, embedding_dim]
                  (diagonal covariance, log-space for stability)

    Returns:
        NLL loss scalar (mean over batch)
    """
    batch_size, num_components, embedding_dim = means.shape

    # Normalize weights to mixing coefficients (softmax ensures sum to 1)
    log_weights = F.log_softmax(weights, dim=1)  # [batch_size, num_components]

    # Expand embeddings for broadcasting: [batch_size, 1, embedding_dim]
    embeddings = embeddings.unsqueeze(1)

    # Compute squared Mahalanobis distance for each component
    # (x - μ)^T Σ^{-1} (x - μ) where Σ is diagonal
    diff = embeddings - means  # [batch_size, num_components, embedding_dim]
    inv_vars = torch.exp(-log_vars)  # [batch_size, num_components, embedding_dim]
    mahal_dist = torch.sum(diff ** 2 * inv_vars, dim=2)  # [batch_size, num_components]

    # Compute log probability for each component (multivariate Gaussian)
    # log p(x|k) = -0.5 * (D*log(2π) + sum(log_vars) + mahal_dist)
    log_det = torch.sum(log_vars, dim=2)  # [batch_size, num_components]
    log_probs = -0.5 * (
        embedding_dim * LOG_2PI + log_det + mahal_dist
    )  # [batch_size, num_components]

    # Combine with mixing coefficients
    # log p(x) = log(sum_k(π_k * p(x|k)))
    log_weighted_probs = log_weights + log_probs  # [batch_size, num_components]

    # Log-sum-exp for numerical stability
    log_likelihood = torch.logsumexp(log_weighted_probs, dim=1)  # [batch_size]

    # Return negative log-likelihood (mean over batch)
    nll = -torch.mean(log_likelihood)

    return nll


def compute_contrastive_nll(
    embeddings: torch.Tensor,
    all_weights: torch.Tensor,
    all_means: torch.Tensor,
    all_log_vars: torch.Tensor,
    target_group_idx: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Cross-group contrastive loss via GMM log-likelihoods.

    Computes log-likelihood of batch embeddings under each group's GMM,
    then applies cross-entropy so the correct group has highest likelihood.

    Args:
        embeddings: [B, D] batch embeddings from one group
        all_weights: [G, K] mixing coefficients (unnormalized, from forward())
        all_means: [G, K, D] component means
        all_log_vars: [G, K, D] component log-variances
        target_group_idx: correct group index for this batch
        temperature: cross-entropy temperature (lower = sharper)

    Returns:
        Cross-entropy loss (scalar)
    """
    G, K, D = all_means.shape
    B = embeddings.shape[0]

    # Normalize weights per group
    log_weights = F.log_softmax(all_weights, dim=1)  # [G, K]

    # Broadcast: embeddings [B,1,1,D] vs means [1,G,K,D]
    x = embeddings.unsqueeze(1).unsqueeze(2)          # [B, 1, 1, D]
    means = all_means.unsqueeze(0)                     # [1, G, K, D]
    log_vars = all_log_vars.unsqueeze(0)               # [1, G, K, D]

    # Mahalanobis distance
    diff = x - means                                   # [B, G, K, D]
    inv_vars = torch.exp(-log_vars)                    # [1, G, K, D]
    mahal = torch.sum(diff ** 2 * inv_vars, dim=3)     # [B, G, K]

    # Log determinant per component per group
    log_det = torch.sum(all_log_vars, dim=2)           # [G, K]

    # Log probability per component
    log_probs = -0.5 * (
        D * LOG_2PI + log_det.unsqueeze(0) + mahal
    )  # [B, G, K]

    # Combine with mixing coefficients
    log_weighted = log_weights.unsqueeze(0) + log_probs  # [B, G, K]

    # Log-sum-exp over components → per-group log-likelihood
    log_likelihood = torch.logsumexp(log_weighted, dim=2)  # [B, G]

    # Cross-entropy: correct group should score highest
    targets = torch.full(
        (B,), target_group_idx, dtype=torch.long, device=embeddings.device
    )
    loss = F.cross_entropy(log_likelihood / temperature, targets)

    return loss


def sample_from_gmm(
    weights: torch.Tensor,
    means: torch.Tensor,
    log_vars: torch.Tensor,
    num_samples: int = 1,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample embeddings from GMM distribution.

    Two-step sampling:
    1. Sample component indices from categorical distribution (weights)
    2. Sample from selected Gaussian components

    Args:
        weights: GMM mixing coefficients [num_components]
                 (unnormalized, will be softmaxed)
        means: GMM component means [num_components, embedding_dim]
        log_vars: GMM component log-variances [num_components, embedding_dim]
        num_samples: Number of samples to generate
        temperature: Sampling temperature (higher = more diversity)
                     Applied to both component selection and Gaussian sampling

    Returns:
        Sampled embeddings [num_samples, embedding_dim]
    """
    num_components, embedding_dim = means.shape

    # Normalize weights to probabilities
    probs = F.softmax(weights / temperature, dim=0)  # [num_components]

    # Sample component indices
    component_indices = torch.multinomial(
        probs, num_samples, replacement=True
    )  # [num_samples]

    # Gather selected means and variances
    selected_means = means[component_indices]  # [num_samples, embedding_dim]
    selected_log_vars = log_vars[component_indices]  # [num_samples, embedding_dim]
    selected_vars = torch.exp(selected_log_vars)  # [num_samples, embedding_dim]

    # Sample from Gaussians (reparameterization trick)
    # x = μ + σ * ε, where ε ~ N(0, I)
    epsilon = torch.randn_like(selected_means)
    samples = selected_means + torch.sqrt(selected_vars * temperature) * epsilon

    return samples


def diversity_score(samples: torch.Tensor) -> float:
    """
    Compute diversity score for sampled embeddings.

    Measures average pairwise distance (higher = more diverse).

    Args:
        samples: Sampled embeddings [num_samples, embedding_dim]

    Returns:
        Average pairwise cosine distance
    """
    # Normalize to unit vectors
    samples_norm = F.normalize(samples, p=2, dim=1)  # [num_samples, embedding_dim]

    # Compute pairwise cosine similarities
    similarities = torch.mm(samples_norm, samples_norm.t())  # [num_samples, num_samples]

    # Compute cosine distances (1 - similarity)
    distances = 1.0 - similarities

    # Average over off-diagonal elements (exclude self-similarity)
    num_samples = samples.shape[0]
    mask = ~torch.eye(num_samples, dtype=torch.bool, device=samples.device)
    avg_distance = distances[mask].mean().item()

    return avg_distance


def coverage_score(
    samples: torch.Tensor,
    reference_embeddings: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute coverage score for sampled embeddings.

    Measures how well samples cover the reference embedding space.
    Uses k-nearest neighbor distances.

    Args:
        samples: Sampled embeddings [num_samples, embedding_dim]
        reference_embeddings: Ground-truth embeddings [num_references, embedding_dim]
        k: Number of nearest neighbors to consider

    Returns:
        Average k-NN distance (lower = better coverage)
    """
    # Normalize to unit vectors
    samples_norm = F.normalize(samples, p=2, dim=1)
    refs_norm = F.normalize(reference_embeddings, p=2, dim=1)

    # Compute cosine distances from each reference to all samples
    similarities = torch.mm(refs_norm, samples_norm.t())  # [num_refs, num_samples]
    distances = 1.0 - similarities

    # Find k-nearest samples for each reference
    k = min(k, samples.shape[0])
    knn_distances, _ = torch.topk(distances, k, dim=1, largest=False)

    # Average k-NN distance
    avg_knn_distance = knn_distances.mean().item()

    return avg_knn_distance


def validate_gmm_params(
    weights: torch.Tensor,
    means: torch.Tensor,
    log_vars: torch.Tensor
) -> Tuple[bool, str]:
    """
    Validate GMM parameters for numerical issues.

    Args:
        weights: GMM mixing coefficients [batch_size, num_components]
        means: GMM component means [batch_size, num_components, embedding_dim]
        log_vars: GMM component log-variances [batch_size, num_components, embedding_dim]

    Returns:
        (is_valid, error_message) tuple
    """
    # Check for NaN or Inf
    if torch.isnan(weights).any() or torch.isinf(weights).any():
        return False, "Invalid values in weights (NaN or Inf)"

    if torch.isnan(means).any() or torch.isinf(means).any():
        return False, "Invalid values in means (NaN or Inf)"

    if torch.isnan(log_vars).any() or torch.isinf(log_vars).any():
        return False, "Invalid values in log_vars (NaN or Inf)"

    # Check variance bounds (prevent collapse)
    vars = torch.exp(log_vars)
    if (vars < EPSILON).any():
        return False, f"Variances too small (< {EPSILON})"

    if (vars > 1e6).any():
        return False, "Variances too large (> 1e6)"

    # Check weight distribution (prevent mode collapse)
    probs = F.softmax(weights, dim=1)
    max_prob = probs.max(dim=1).values
    if (max_prob > 0.99).any():
        return False, "Mode collapse detected (single component dominates)"

    return True, ""

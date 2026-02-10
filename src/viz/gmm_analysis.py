"""
Core computation functions for post-training GMM analysis.

Implements analyses 2.1, 2.5, 2.6, 2.7 from the research plan.
All functions are pure computation — no plotting.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.gmm_utils import EPSILON, LOG_2PI

logger = logging.getLogger(__name__)


# =============================================================================
# Foundation functions
# =============================================================================

def compute_log_likelihood_single_gmm(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    log_vars: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log p(x) for N embeddings under a single GMM.

    Unlike compute_gmm_nll, this takes a single GMM (no batch dim) and
    already-normalized weights (from model.get_gmm_params()).
    Uses torch.log(weights) instead of F.log_softmax because get_gmm_params()
    applies softmax before returning.

    Args:
        embeddings: [N, D] embeddings to evaluate
        weights: [K] normalized mixing coefficients (sum to 1)
        means: [K, D] component means
        log_vars: [K, D] component log-variances

    Returns:
        [N] log-likelihoods
    """
    num_components, embedding_dim = means.shape

    # Log mixing coefficients (already normalized)
    log_weights = torch.log(weights + EPSILON)  # [K]

    # Expand embeddings for broadcasting: [N, 1, D]
    x = embeddings.unsqueeze(1)

    # Squared Mahalanobis distance per component
    diff = x - means.unsqueeze(0)  # [N, K, D]
    inv_vars = torch.exp(-log_vars).unsqueeze(0)  # [1, K, D]
    mahal_dist = torch.sum(diff ** 2 * inv_vars, dim=2)  # [N, K]

    # Log probability per component: log N(x | mu_k, sigma_k)
    log_det = torch.sum(log_vars, dim=1)  # [K]
    log_probs = -0.5 * (
        embedding_dim * LOG_2PI + log_det.unsqueeze(0) + mahal_dist
    )  # [N, K]

    # Combine with mixing coefficients: log(pi_k * N(x|k))
    log_weighted = log_weights.unsqueeze(0) + log_probs  # [N, K]

    # Log-sum-exp over components
    log_likelihood = torch.logsumexp(log_weighted, dim=1)  # [N]

    return log_likelihood


def _sample_from_normalized_gmm(
    weights: torch.Tensor,
    means: torch.Tensor,
    log_vars: torch.Tensor,
    num_samples: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample from a GMM with already-normalized weights.

    Mirrors gmm_utils.sample_from_gmm but skips the internal softmax.
    The existing sample_from_gmm does F.softmax(weights/T), which would
    apply softmax to already-softmaxed weights, corrupting the distribution.

    For temperature: softmax(log(weights) / T) to re-sharpen/flatten.

    Args:
        weights: [K] normalized mixing coefficients
        means: [K, D] component means
        log_vars: [K, D] component log-variances
        num_samples: number of samples to draw
        temperature: sampling temperature (higher = more diversity)

    Returns:
        [num_samples, D] sampled embeddings
    """
    # Apply temperature to weights via log-space
    log_w = torch.log(weights + EPSILON)
    probs = F.softmax(log_w / temperature, dim=0)  # [K]

    # Sample component indices
    component_indices = torch.multinomial(
        probs, num_samples, replacement=True
    )  # [num_samples]

    # Gather selected means and variances
    selected_means = means[component_indices]  # [num_samples, D]
    selected_log_vars = log_vars[component_indices]  # [num_samples, D]
    selected_vars = torch.exp(selected_log_vars)

    # Reparameterization: x = mu + sigma * epsilon
    epsilon = torch.randn_like(selected_means)
    samples = selected_means + torch.sqrt(selected_vars * temperature) * epsilon

    return samples


def load_reference_embeddings_by_group(
    mapping_path: str,
    embedding_dir: str,
    max_per_group: int = 200,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Load real test embeddings grouped by attribute_group.

    Loads test.json, groups by attribute_group, deduplicates audio_id
    (each appears 10x due to text augmentation), subsamples to max_per_group.

    Args:
        mapping_path: path to test.json
        embedding_dir: path to Kaldi ARK xvector directory
        max_per_group: max unique audio IDs per group
        seed: random seed for subsampling

    Returns:
        Dict[str, np.ndarray]: group name -> [N_g, D] embeddings
    """
    from src.utils.embedding_loader import HyperionEmbeddingLoader

    rng = np.random.RandomState(seed)

    with open(mapping_path, "r") as f:
        test_data = json.load(f)

    # Group unique audio_ids by attribute_group
    group_ids: Dict[str, set] = {}
    for entry in test_data:
        group = entry["attribute_group"]
        audio_id = entry["audio_id"]
        if group not in group_ids:
            group_ids[group] = set()
        group_ids[group].add(audio_id)

    # Subsample
    group_audio_ids: Dict[str, List[str]] = {}
    for group, ids in group_ids.items():
        ids_list = sorted(ids)
        if len(ids_list) > max_per_group:
            chosen = rng.choice(ids_list, size=max_per_group, replace=False).tolist()
        else:
            chosen = ids_list
        group_audio_ids[group] = chosen

    # Load embeddings (HyperionEmbeddingLoader does os.chdir, load upfront)
    loader = HyperionEmbeddingLoader(embedding_dir, logger=logger)

    result: Dict[str, np.ndarray] = {}
    for group, audio_ids in sorted(group_audio_ids.items()):
        embeddings = []
        for aid in audio_ids:
            try:
                emb = loader.load_embedding(aid)
                embeddings.append(emb)
            except KeyError:
                logger.warning(f"Embedding not found for {aid}, skipping")
        if embeddings:
            result[group] = np.stack(embeddings, axis=0)
            logger.info(f"Loaded {len(embeddings)} embeddings for '{group}'")
        else:
            logger.warning(f"No embeddings loaded for '{group}'")

    loader.close()
    return result


def get_all_group_gmm_params(
    model,
    augmented_texts_path: str,
    text_index: int = 0,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Get GMM params for all 18 attribute groups from the model.

    Loads augmented_texts.json, picks the text_index-th paraphrase per group,
    calls model.get_gmm_params(text) for each.

    Args:
        model: trained GMMMDN model
        augmented_texts_path: path to augmented_texts.json
        text_index: which paraphrase variant to use (0-9)

    Returns:
        Dict[str, Tuple]: group -> (weights[K], means[K,D], log_vars[K,D])
    """
    with open(augmented_texts_path, "r") as f:
        augmented_texts = json.load(f)

    result = {}
    for group in sorted(augmented_texts.keys()):
        texts = augmented_texts[group]
        text = texts[text_index % len(texts)]
        weights, means, log_vars = model.get_gmm_params(text)
        result[group] = (weights, means, log_vars)
        logger.debug(f"Got GMM params for '{group}' using text: '{text[:60]}...'")

    return result


# =============================================================================
# 2.7 — Reverse Classification
# =============================================================================

def classify_embeddings_to_groups(
    embeddings: torch.Tensor,
    group_gmm_params: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Classify each embedding to the group whose GMM assigns highest log-likelihood.

    Args:
        embeddings: [N, D] embeddings to classify
        group_gmm_params: group -> (weights, means, log_vars)
        device: computation device

    Returns:
        (predicted_indices [N], log_likelihood_matrix [N, G])
        where G = number of groups
    """
    group_names = sorted(group_gmm_params.keys())
    embeddings = embeddings.to(device)

    ll_columns = []
    for group in group_names:
        w, m, lv = group_gmm_params[group]
        w, m, lv = w.to(device), m.to(device), lv.to(device)
        ll = compute_log_likelihood_single_gmm(embeddings, w, m, lv)  # [N]
        ll_columns.append(ll)

    ll_matrix = torch.stack(ll_columns, dim=1)  # [N, G]
    predicted = torch.argmax(ll_matrix, dim=1)  # [N]

    return predicted, ll_matrix


def compute_reverse_classification_metrics(
    real_embeddings_by_group: Dict[str, np.ndarray],
    group_gmm_params: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    group_names: List[str],
    device: str = "cuda",
) -> dict:
    """
    Classify real test embeddings to GMM groups, compute accuracy.

    Args:
        real_embeddings_by_group: group -> [N_g, D] real embeddings
        group_gmm_params: group -> GMM params
        group_names: sorted list of group names
        device: computation device

    Returns:
        dict with overall_accuracy, per_group_accuracy, confusion_matrix, group_names
    """
    all_embeddings = []
    all_labels = []

    for i, group in enumerate(group_names):
        if group not in real_embeddings_by_group:
            continue
        embs = real_embeddings_by_group[group]
        all_embeddings.append(torch.from_numpy(embs).float())
        all_labels.extend([i] * len(embs))

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N_total, D]
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    predicted, ll_matrix = classify_embeddings_to_groups(
        all_embeddings, group_gmm_params, device
    )
    predicted = predicted.cpu()

    num_groups = len(group_names)
    confusion = torch.zeros(num_groups, num_groups, dtype=torch.long)
    for true_idx, pred_idx in zip(all_labels, predicted):
        confusion[true_idx, pred_idx] += 1

    overall_accuracy = (predicted == all_labels).float().mean().item()

    per_group_accuracy = {}
    for i, group in enumerate(group_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_group_accuracy[group] = (predicted[mask] == i).float().mean().item()
        else:
            per_group_accuracy[group] = float("nan")

    return {
        "overall_accuracy": overall_accuracy,
        "per_group_accuracy": per_group_accuracy,
        "confusion_matrix": confusion.numpy(),
        "group_names": group_names,
    }


def compute_self_consistency(
    model,
    group_gmm_params: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    group_names: List[str],
    num_samples_per_group: int = 200,
    temperature: float = 1.0,
    device: str = "cuda",
) -> dict:
    """
    Sample from each group's GMM and classify back.

    Tests whether sampling from group A's GMM classifies back to group A.

    Args:
        model: trained GMMMDN (unused here, kept for API consistency)
        group_gmm_params: group -> GMM params
        group_names: sorted group names
        num_samples_per_group: samples per group
        temperature: sampling temperature
        device: computation device

    Returns:
        same structure as compute_reverse_classification_metrics
    """
    all_embeddings = []
    all_labels = []

    for i, group in enumerate(group_names):
        if group not in group_gmm_params:
            continue
        w, m, lv = group_gmm_params[group]
        w, m, lv = w.to(device), m.to(device), lv.to(device)
        samples = _sample_from_normalized_gmm(
            w, m, lv, num_samples_per_group, temperature
        )
        all_embeddings.append(samples.cpu())
        all_labels.extend([i] * num_samples_per_group)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    predicted, ll_matrix = classify_embeddings_to_groups(
        all_embeddings, group_gmm_params, device
    )
    predicted = predicted.cpu()

    num_groups = len(group_names)
    confusion = torch.zeros(num_groups, num_groups, dtype=torch.long)
    for true_idx, pred_idx in zip(all_labels, predicted):
        confusion[true_idx, pred_idx] += 1

    overall_accuracy = (predicted == all_labels).float().mean().item()

    per_group_accuracy = {}
    for i, group in enumerate(group_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_group_accuracy[group] = (predicted[mask] == i).float().mean().item()
        else:
            per_group_accuracy[group] = float("nan")

    return {
        "overall_accuracy": overall_accuracy,
        "per_group_accuracy": per_group_accuracy,
        "confusion_matrix": confusion.numpy(),
        "group_names": group_names,
    }


# =============================================================================
# 2.6 — GMM Comparison
# =============================================================================

def monte_carlo_kl_divergence(
    weights_p: torch.Tensor,
    means_p: torch.Tensor,
    log_vars_p: torch.Tensor,
    weights_q: torch.Tensor,
    means_q: torch.Tensor,
    log_vars_q: torch.Tensor,
    num_samples: int = 10000,
    device: str = "cuda",
) -> float:
    """
    Monte Carlo estimate of KL(P || Q) for two GMMs with normalized weights.

    Samples from P, computes mean(log p(x) - log q(x)).

    Args:
        weights_p, means_p, log_vars_p: GMM P parameters
        weights_q, means_q, log_vars_q: GMM Q parameters
        num_samples: MC samples
        device: computation device

    Returns:
        KL divergence estimate (scalar float)
    """
    weights_p = weights_p.to(device)
    means_p = means_p.to(device)
    log_vars_p = log_vars_p.to(device)
    weights_q = weights_q.to(device)
    means_q = means_q.to(device)
    log_vars_q = log_vars_q.to(device)

    samples = _sample_from_normalized_gmm(
        weights_p, means_p, log_vars_p, num_samples, temperature=1.0
    )  # [N, D]

    log_p = compute_log_likelihood_single_gmm(samples, weights_p, means_p, log_vars_p)
    log_q = compute_log_likelihood_single_gmm(samples, weights_q, means_q, log_vars_q)

    kl = (log_p - log_q).mean().item()
    return max(kl, 0.0)  # Clamp to non-negative (MC noise can produce small negatives)


def compute_jensen_shannon_divergence(
    weights_p: torch.Tensor,
    means_p: torch.Tensor,
    log_vars_p: torch.Tensor,
    weights_q: torch.Tensor,
    means_q: torch.Tensor,
    log_vars_q: torch.Tensor,
    num_samples: int = 10000,
    device: str = "cuda",
) -> float:
    """
    Jensen-Shannon divergence: JSD(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M).

    Mixture M has 2K components: weights_M = cat(0.5*w_p, 0.5*w_q),
    means_M = cat(means_p, means_q), same for log_vars.

    Args:
        weights_p, means_p, log_vars_p: GMM P parameters
        weights_q, means_q, log_vars_q: GMM Q parameters
        num_samples: MC samples per KL term
        device: computation device

    Returns:
        JSD value (scalar float)
    """
    # Build mixture M
    weights_m = torch.cat([0.5 * weights_p, 0.5 * weights_q], dim=0)
    means_m = torch.cat([means_p, means_q], dim=0)
    log_vars_m = torch.cat([log_vars_p, log_vars_q], dim=0)

    kl_pm = monte_carlo_kl_divergence(
        weights_p, means_p, log_vars_p,
        weights_m, means_m, log_vars_m,
        num_samples, device,
    )
    kl_qm = monte_carlo_kl_divergence(
        weights_q, means_q, log_vars_q,
        weights_m, means_m, log_vars_m,
        num_samples, device,
    )

    return 0.5 * kl_pm + 0.5 * kl_qm


def compute_pairwise_gmm_distances(
    group_gmm_params: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    group_names: List[str],
    metric: str = "jsd",
    num_samples: int = 10000,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute pairwise distance matrix between all group GMMs.

    Args:
        group_gmm_params: group -> (weights, means, log_vars)
        group_names: sorted group names
        metric: "jsd" or "kl"
        num_samples: MC samples per pair
        device: computation device

    Returns:
        [G, G] distance matrix (symmetric for JSD)
    """
    n = len(group_names)
    matrix = np.zeros((n, n))

    total_pairs = n * (n - 1) // 2
    done = 0

    for i in range(n):
        for j in range(i + 1, n):
            w_i, m_i, lv_i = group_gmm_params[group_names[i]]
            w_j, m_j, lv_j = group_gmm_params[group_names[j]]

            if metric == "jsd":
                dist = compute_jensen_shannon_divergence(
                    w_i, m_i, lv_i, w_j, m_j, lv_j, num_samples, device
                )
                matrix[i, j] = dist
                matrix[j, i] = dist
            else:  # kl
                matrix[i, j] = monte_carlo_kl_divergence(
                    w_i, m_i, lv_i, w_j, m_j, lv_j, num_samples, device
                )
                matrix[j, i] = monte_carlo_kl_divergence(
                    w_j, m_j, lv_j, w_i, m_i, lv_i, num_samples, device
                )

            done += 1
            if done % 20 == 0:
                logger.info(f"Pairwise distances: {done}/{total_pairs} pairs computed")

    logger.info(f"Pairwise distance computation complete ({total_pairs} pairs)")
    return matrix


def analyze_opposite_pairs(
    distance_matrix: np.ndarray,
    group_names: List[str],
) -> dict:
    """
    Analyze distances between groups that differ in exactly one attribute.

    Parses group names ("gender, pitch, rate") to identify opposite pairs:
    - Gender flip: male<->female, same pitch+rate (9 pairs)
    - Pitch flip: high<->low, same gender+rate (6 pairs)
    - Rate flip: fast<->slow, same gender+pitch (6 pairs)

    Args:
        distance_matrix: [G, G] pairwise distances
        group_names: sorted group name list

    Returns:
        dict with per-pair distances and aggregate stats by flip type
    """
    # Parse attributes from group names
    parsed = []
    for name in group_names:
        parts = [p.strip() for p in name.split(",")]
        gender = parts[0]  # male / female
        pitch = parts[1]   # high-pitched / medium-pitched / low-pitched
        rate = parts[2]    # fast speed / measured speed / slow speed
        parsed.append({"gender": gender, "pitch": pitch, "rate": rate})

    name_to_idx = {name: i for i, name in enumerate(group_names)}

    gender_pairs = []
    pitch_pairs = []
    rate_pairs = []

    # Gender flip: same pitch + rate, male <-> female
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            pi, pj = parsed[i], parsed[j]
            if pi["pitch"] == pj["pitch"] and pi["rate"] == pj["rate"]:
                if pi["gender"] != pj["gender"]:
                    gender_pairs.append({
                        "group_a": group_names[i],
                        "group_b": group_names[j],
                        "distance": float(distance_matrix[i, j]),
                    })

    # Pitch flip: same gender + rate, high <-> low
    pitch_opposites = {"high-pitched": "low-pitched", "low-pitched": "high-pitched"}
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            pi, pj = parsed[i], parsed[j]
            if pi["gender"] == pj["gender"] and pi["rate"] == pj["rate"]:
                if pi["pitch"] in pitch_opposites and pitch_opposites[pi["pitch"]] == pj["pitch"]:
                    pitch_pairs.append({
                        "group_a": group_names[i],
                        "group_b": group_names[j],
                        "distance": float(distance_matrix[i, j]),
                    })

    # Rate flip: same gender + pitch, fast <-> slow
    rate_opposites = {"fast speed": "slow speed", "slow speed": "fast speed"}
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            pi, pj = parsed[i], parsed[j]
            if pi["gender"] == pj["gender"] and pi["pitch"] == pj["pitch"]:
                if pi["rate"] in rate_opposites and rate_opposites[pi["rate"]] == pj["rate"]:
                    rate_pairs.append({
                        "group_a": group_names[i],
                        "group_b": group_names[j],
                        "distance": float(distance_matrix[i, j]),
                    })

    def _stats(pairs):
        dists = [p["distance"] for p in pairs]
        if not dists:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "min": float(np.min(dists)),
            "max": float(np.max(dists)),
        }

    return {
        "gender_flip": {"pairs": gender_pairs, "stats": _stats(gender_pairs)},
        "pitch_flip": {"pairs": pitch_pairs, "stats": _stats(pitch_pairs)},
        "rate_flip": {"pairs": rate_pairs, "stats": _stats(rate_pairs)},
    }


def create_union_gmm(
    weights_a: torch.Tensor,
    means_a: torch.Tensor,
    log_vars_a: torch.Tensor,
    weights_b: torch.Tensor,
    means_b: torch.Tensor,
    log_vars_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a union GMM by concatenating two GMMs.

    The resulting GMM has 2K components with halved weights.

    Args:
        weights_a, means_a, log_vars_a: GMM A parameters
        weights_b, means_b, log_vars_b: GMM B parameters

    Returns:
        (weights[2K], means[2K,D], log_vars[2K,D])
    """
    weights = torch.cat([0.5 * weights_a, 0.5 * weights_b], dim=0)
    means = torch.cat([means_a, means_b], dim=0)
    log_vars = torch.cat([log_vars_a, log_vars_b], dim=0)
    return weights, means, log_vars


# =============================================================================
# 2.1 — KL Divergence (Real vs Predicted)
# =============================================================================

def fit_reference_gmm(
    embeddings: np.ndarray,
    n_components: int = 15,
    covariance_type: str = "diag",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a sklearn GMM to real embeddings.

    Args:
        embeddings: [N, D] real embeddings
        n_components: number of GMM components
        covariance_type: "diag" for diagonal covariance
        seed: random seed

    Returns:
        (weights[K], means[K,D], log_vars[K,D]) as numpy arrays
    """
    from sklearn.mixture import GaussianMixture

    # Guard for small groups
    n_components = min(n_components, len(embeddings) - 1)
    n_components = max(n_components, 1)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=seed,
        max_iter=200,
    )
    gmm.fit(embeddings)

    weights = gmm.weights_.astype(np.float32)       # [K]
    means = gmm.means_.astype(np.float32)             # [K, D]

    if covariance_type == "diag":
        log_vars = np.log(gmm.covariances_ + EPSILON).astype(np.float32)  # [K, D]
    else:
        # Full covariance: extract diagonal
        log_vars = np.log(
            np.array([np.diag(c) for c in gmm.covariances_]) + EPSILON
        ).astype(np.float32)

    return weights, means, log_vars


def compute_kl_real_vs_predicted(
    real_embeddings_by_group: Dict[str, np.ndarray],
    group_gmm_params: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_components: int = 15,
    num_mc_samples: int = 10000,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute KL(predicted || reference) per group.

    Per group: fit sklearn GMM on real data, get predicted GMM from model,
    sample from predicted, compute mean(log p_pred - log p_ref).

    Args:
        real_embeddings_by_group: group -> [N_g, D] real embeddings
        group_gmm_params: group -> model GMM params
        n_components: sklearn GMM components
        num_mc_samples: MC samples for KL estimate
        seed: random seed for sklearn
        device: computation device

    Returns:
        Dict[str, float]: group -> KL value
    """
    results = {}

    for group in sorted(group_gmm_params.keys()):
        if group not in real_embeddings_by_group:
            logger.warning(f"No real embeddings for '{group}', skipping KL")
            continue

        real_embs = real_embeddings_by_group[group]

        # Fit reference GMM on real data
        ref_w, ref_m, ref_lv = fit_reference_gmm(
            real_embs, n_components=n_components, seed=seed
        )
        ref_w = torch.from_numpy(ref_w).to(device)
        ref_m = torch.from_numpy(ref_m).to(device)
        ref_lv = torch.from_numpy(ref_lv).to(device)

        # Predicted GMM from model
        pred_w, pred_m, pred_lv = group_gmm_params[group]
        pred_w = pred_w.to(device)
        pred_m = pred_m.to(device)
        pred_lv = pred_lv.to(device)

        # KL(predicted || reference): sample from predicted
        kl_val = monte_carlo_kl_divergence(
            pred_w, pred_m, pred_lv,
            ref_w, ref_m, ref_lv,
            num_mc_samples, device,
        )

        results[group] = kl_val
        logger.info(f"KL(pred||ref) for '{group}': {kl_val:.4f}")

    return results


# =============================================================================
# 2.5 — Visualization Data Prep
# =============================================================================

def prepare_visualization_data(
    model,
    real_embeddings_by_group: Dict[str, np.ndarray],
    augmented_texts_path: str,
    num_generated_per_group: int = 200,
    temperature: float = 1.0,
    device: str = "cuda",
) -> dict:
    """
    Prepare embeddings and labels for t-SNE / UMAP visualization.

    Per group: take real embeddings + sample generated embeddings.
    Parse attribute labels from group names.

    Args:
        model: trained GMMMDN model
        real_embeddings_by_group: group -> [N_g, D] real embeddings
        augmented_texts_path: path to augmented_texts.json
        num_generated_per_group: how many to sample per group
        temperature: sampling temperature
        device: computation device

    Returns:
        dict with: embeddings [N_total, D], labels_source, labels_group,
                   labels_gender, labels_pitch, labels_rate
    """
    group_gmm_params = get_all_group_gmm_params(model, augmented_texts_path)

    all_embeddings = []
    labels_source = []
    labels_group = []
    labels_gender = []
    labels_pitch = []
    labels_rate = []

    for group in sorted(group_gmm_params.keys()):
        parts = [p.strip() for p in group.split(",")]
        gender, pitch, rate = parts[0], parts[1], parts[2]

        # Real embeddings
        if group in real_embeddings_by_group:
            real_embs = real_embeddings_by_group[group]
            all_embeddings.append(real_embs)
            n_real = len(real_embs)
            labels_source.extend(["real"] * n_real)
            labels_group.extend([group] * n_real)
            labels_gender.extend([gender] * n_real)
            labels_pitch.extend([pitch] * n_real)
            labels_rate.extend([rate] * n_real)

        # Generated embeddings
        w, m, lv = group_gmm_params[group]
        w, m, lv = w.to(device), m.to(device), lv.to(device)
        gen_embs = _sample_from_normalized_gmm(
            w, m, lv, num_generated_per_group, temperature
        )
        gen_embs_np = gen_embs.cpu().numpy()
        all_embeddings.append(gen_embs_np)
        n_gen = len(gen_embs_np)
        labels_source.extend(["generated"] * n_gen)
        labels_group.extend([group] * n_gen)
        labels_gender.extend([gender] * n_gen)
        labels_pitch.extend([pitch] * n_gen)
        labels_rate.extend([rate] * n_gen)

    return {
        "embeddings": np.concatenate(all_embeddings, axis=0),
        "labels_source": labels_source,
        "labels_group": labels_group,
        "labels_gender": labels_gender,
        "labels_pitch": labels_pitch,
        "labels_rate": labels_rate,
    }

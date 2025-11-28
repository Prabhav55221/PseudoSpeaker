"""
Evaluator for GMM-MDN validation.

Computes metrics:
- NLL loss (primary metric)
- Diversity (average pairwise distance)
- Coverage (k-NN distance to ground truth)
"""

import logging
from typing import Dict, Optional
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.gmm_mdn import GMMMDN
from ..models.gmm_utils import diversity_score, coverage_score


class Evaluator:
    """
    Evaluator for GMM-MDN validation.

    Computes NLL loss, diversity, and coverage metrics.
    """

    def __init__(
        self,
        model: GMMMDN,
        dataloader: DataLoader,
        device: str = "cuda",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize evaluator.

        Args:
            model: GMM-MDN model
            dataloader: Validation DataLoader
            device: Device to run evaluation on
            logger: Logger instance
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

    @torch.no_grad()
    def evaluate(
        self,
        num_samples: int = 10,
        temperature: float = 1.0,
        compute_diversity: bool = True,
        compute_coverage: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            num_samples: Number of samples to generate per text (for diversity/coverage)
            temperature: Sampling temperature
            compute_diversity: Whether to compute diversity score
            compute_coverage: Whether to compute coverage score

        Returns:
            Dict of metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        diversity_scores = []
        coverage_scores = []

        self.logger.info("Running evaluation...")

        for texts, embeddings, audio_ids, attribute_group in tqdm(
            self.dataloader,
            desc="Evaluating",
            disable=not self.logger.isEnabledFor(logging.INFO)
        ):
            embeddings = embeddings.to(self.device)

            # Compute NLL loss
            loss = self.model.compute_loss(texts, embeddings)
            total_loss += loss.item()
            num_batches += 1

            # Compute diversity and coverage (sample once per unique text)
            if compute_diversity or compute_coverage:
                # Use first text in batch (all texts in group are similar)
                text = texts[0]

                # Sample embeddings
                samples = self.model.sample(
                    text=text,
                    num_samples=num_samples,
                    temperature=temperature
                )

                # Diversity: average pairwise distance among samples
                if compute_diversity:
                    div_score = diversity_score(samples)
                    diversity_scores.append(div_score)

                # Coverage: k-NN distance to ground truth
                if compute_coverage:
                    cov_score = coverage_score(
                        samples=samples,
                        reference_embeddings=embeddings,
                        k=min(5, embeddings.shape[0])
                    )
                    coverage_scores.append(cov_score)

        # Aggregate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        metrics = {
            "val_loss": avg_loss,
        }

        if compute_diversity and diversity_scores:
            metrics["diversity"] = sum(diversity_scores) / len(diversity_scores)

        if compute_coverage and coverage_scores:
            metrics["coverage"] = sum(coverage_scores) / len(coverage_scores)

        # Log metrics
        self.logger.info(
            f"Validation metrics: "
            f"loss={metrics['val_loss']:.4f}"
            + (f", diversity={metrics.get('diversity', 0):.4f}" if compute_diversity else "")
            + (f", coverage={metrics.get('coverage', 0):.4f}" if compute_coverage else "")
        )

        return metrics

    @torch.no_grad()
    def evaluate_per_group(
        self,
        num_samples: int = 10,
        temperature: float = 1.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model per attribute group.

        Args:
            num_samples: Number of samples to generate per text
            temperature: Sampling temperature

        Returns:
            Dict mapping attribute_group -> metrics dict
        """
        self.model.eval()

        group_losses = defaultdict(list)
        group_diversities = defaultdict(list)
        group_coverages = defaultdict(list)

        self.logger.info("Running per-group evaluation...")

        for texts, embeddings, audio_ids, attribute_group in tqdm(
            self.dataloader,
            desc="Evaluating per group",
            disable=not self.logger.isEnabledFor(logging.INFO)
        ):
            embeddings = embeddings.to(self.device)

            # Compute NLL loss
            loss = self.model.compute_loss(texts, embeddings)
            group_losses[attribute_group].append(loss.item())

            # Sample and compute diversity/coverage
            text = texts[0]
            samples = self.model.sample(
                text=text,
                num_samples=num_samples,
                temperature=temperature
            )

            div_score = diversity_score(samples)
            group_diversities[attribute_group].append(div_score)

            cov_score = coverage_score(
                samples=samples,
                reference_embeddings=embeddings,
                k=min(5, embeddings.shape[0])
            )
            group_coverages[attribute_group].append(cov_score)

        # Aggregate per-group metrics
        group_metrics = {}

        for group in group_losses.keys():
            group_metrics[group] = {
                "loss": sum(group_losses[group]) / len(group_losses[group]),
                "diversity": sum(group_diversities[group]) / len(group_diversities[group]),
                "coverage": sum(group_coverages[group]) / len(group_coverages[group]),
                "num_batches": len(group_losses[group])
            }

        # Log per-group statistics
        self.logger.info("\nPer-group metrics:")
        for group in sorted(group_metrics.keys()):
            metrics = group_metrics[group]
            self.logger.info(
                f"  {group:45s}: "
                f"loss={metrics['loss']:.4f}, "
                f"diversity={metrics['diversity']:.4f}, "
                f"coverage={metrics['coverage']:.4f}"
            )

        return group_metrics

    @torch.no_grad()
    def sample_for_visualization(
        self,
        texts: list[str],
        num_samples: int = 100,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Sample embeddings for visualization.

        Args:
            texts: List of text descriptions to sample from
            num_samples: Number of samples per text
            temperature: Sampling temperature

        Returns:
            Dict mapping text -> sampled embeddings [num_samples, 512]
        """
        self.model.eval()

        samples_dict = {}

        for text in texts:
            samples = self.model.sample(
                text=text,
                num_samples=num_samples,
                temperature=temperature
            )
            samples_dict[text] = samples.cpu()

        return samples_dict

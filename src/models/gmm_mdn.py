"""
GMM-MDN: Gaussian Mixture Model - Mixture Density Network.

Main model for pseudo-speaker generation from text descriptions.
Predicts GMM parameters conditioned on text input.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Union

from .text_encoder import TextEncoder
from .gmm_utils import compute_gmm_nll, sample_from_gmm


class GMMMDN(nn.Module):
    """
    GMM-MDN model for pseudo-speaker generation.

    Architecture:
        Text → SentenceBERT (384-dim) → Dense layers → GMM parameters

    GMM parameters:
        - weights: [batch_size, K] mixing coefficients
        - means: [batch_size, K, 512] component means
        - log_vars: [batch_size, K, 512] component log-variances (diagonal)
    """

    def __init__(
        self,
        num_components: int = 15,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        text_encoder_name: str = "all-MiniLM-L6-v2",
        freeze_encoder: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize GMM-MDN model.

        Args:
            num_components: Number of GMM components (K)
            embedding_dim: X-vector embedding dimension (512)
            hidden_dim: Hidden layer dimension
            text_encoder_name: SentenceBERT model name
            freeze_encoder: If True, freeze text encoder weights
            device: Device to load model on
        """
        super().__init__()

        self.num_components = num_components
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Text encoder (SentenceBERT)
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
            freeze=freeze_encoder,
            device=device
        )

        text_dim = self.text_encoder.get_output_dim()

        # MDN head: 3-layer dense network
        self.mdn_head = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output layers for GMM parameters
        # Total output size: K + K*D + K*D = K * (1 + 2*D)
        self.weight_layer = nn.Linear(hidden_dim, num_components)
        self.mean_layer = nn.Linear(hidden_dim, num_components * embedding_dim)
        self.logvar_layer = nn.Linear(hidden_dim, num_components * embedding_dim)

        # Initialize weights
        self._init_weights()

        # Move all layers to device
        self.to(device)

    def _init_weights(self):
        """Initialize MDN head weights."""
        for module in self.mdn_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize output layers
        nn.init.xavier_uniform_(self.weight_layer.weight)
        nn.init.zeros_(self.weight_layer.bias)

        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.zeros_(self.mean_layer.bias)

        nn.init.xavier_uniform_(self.logvar_layer.weight)
        nn.init.constant_(self.logvar_layer.bias, -1.0)  # Initialize to small variance

    def forward(
        self,
        texts: Union[str, List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: text → GMM parameters.

        Args:
            texts: Single text or list of text descriptions

        Returns:
            Tuple of (weights, means, log_vars):
                - weights: [batch_size, num_components]
                - means: [batch_size, num_components, embedding_dim]
                - log_vars: [batch_size, num_components, embedding_dim]
        """
        # Encode text
        text_embeddings = self.text_encoder(texts)  # [batch_size, text_dim]

        # Clone to convert inference tensors to regular tensors for autograd
        # (sentence-transformers uses inference_mode internally)
        # Also move to model's device
        text_embeddings = text_embeddings.clone().to(self.device)

        batch_size = text_embeddings.shape[0]

        # MDN head
        hidden = self.mdn_head(text_embeddings)  # [batch_size, hidden_dim]

        # Predict GMM parameters
        weights = self.weight_layer(hidden)  # [batch_size, K]

        means = self.mean_layer(hidden)  # [batch_size, K*D]
        means = means.view(batch_size, self.num_components, self.embedding_dim)

        log_vars = self.logvar_layer(hidden)  # [batch_size, K*D]
        log_vars = log_vars.view(batch_size, self.num_components, self.embedding_dim)

        return weights, means, log_vars

    def compute_loss(
        self,
        texts: Union[str, List[str]],
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NLL loss for text-embedding pairs.

        Args:
            texts: Text descriptions
            embeddings: Target x-vector embeddings [batch_size, embedding_dim]

        Returns:
            NLL loss (scalar)
        """
        # Predict GMM parameters
        weights, means, log_vars = self.forward(texts)

        # Compute NLL loss
        loss = compute_gmm_nll(embeddings, weights, means, log_vars)

        return loss

    def compute_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        group_texts: List[str],
        target_group_idx: int,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Compute contrastive loss: embeddings should be most likely under
        the correct group's GMM vs all other groups.

        Forwards all group representative texts through the model in one pass,
        then uses cross-entropy over per-group log-likelihoods.

        Args:
            embeddings: [B, D] batch embeddings (already on device)
            group_texts: list of G representative texts (one per group, sorted)
            target_group_idx: index of the correct group in group_texts
            temperature: cross-entropy temperature

        Returns:
            Contrastive loss (scalar)
        """
        from .gmm_utils import compute_contrastive_nll

        # Forward all G group texts → [G, K], [G, K, D], [G, K, D]
        all_weights, all_means, all_log_vars = self.forward(group_texts)

        return compute_contrastive_nll(
            embeddings, all_weights, all_means, all_log_vars,
            target_group_idx, temperature,
        )

    def sample(
        self,
        text: str,
        num_samples: int = 1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Sample pseudo-speaker embeddings from text description.

        Args:
            text: Text description (single string)
            num_samples: Number of embeddings to sample
            temperature: Sampling temperature (higher = more diversity)

        Returns:
            Sampled embeddings [num_samples, embedding_dim]
        """
        self.eval()

        with torch.no_grad():
            # Predict GMM parameters
            weights, means, log_vars = self.forward(text)

            # Sample from GMM (squeeze batch dimension)
            samples = sample_from_gmm(
                weights=weights.squeeze(0),
                means=means.squeeze(0),
                log_vars=log_vars.squeeze(0),
                num_samples=num_samples,
                temperature=temperature
            )

        return samples

    def get_gmm_params(
        self,
        text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get GMM parameters for a text description.

        Useful for analysis and visualization.

        Args:
            text: Text description (single string)

        Returns:
            Tuple of (weights, means, log_vars):
                - weights: [num_components] (normalized probabilities)
                - means: [num_components, embedding_dim]
                - log_vars: [num_components, embedding_dim]
        """
        self.eval()

        with torch.no_grad():
            weights, means, log_vars = self.forward(text)

            # Squeeze batch dimension and normalize weights
            weights = torch.softmax(weights.squeeze(0), dim=0)
            means = means.squeeze(0)
            log_vars = log_vars.squeeze(0)

        return weights, means, log_vars

    def save_checkpoint(self, path: str, epoch: int, optimizer_state=None, **kwargs):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict (optional)
            **kwargs: Additional metadata to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': {
                'num_components': self.num_components,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'text_encoder_name': self.text_encoder.model_name,
                'freeze_encoder': self.text_encoder.freeze,
            },
            **kwargs
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cuda"):
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model on

        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=device)

        # Create model from saved config
        config = checkpoint['model_config']
        model = cls(
            num_components=config['num_components'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            text_encoder_name=config['text_encoder_name'],
            freeze_encoder=config['freeze_encoder'],
            device=device
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model, checkpoint

    def __repr__(self) -> str:
        """String representation."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"GMMMDN(\n"
            f"  num_components={self.num_components},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  text_encoder={self.text_encoder.model_name},\n"
            f"  freeze_encoder={self.text_encoder.freeze},\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,}\n"
            f")"
        )

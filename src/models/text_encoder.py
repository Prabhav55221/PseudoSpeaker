"""
Text encoder using SentenceBERT for speaker description embeddings.

Wraps sentence-transformers for easy integration and optional freezing.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List, Union


class TextEncoder(nn.Module):
    """
    SentenceBERT text encoder wrapper.

    Encodes natural language speaker descriptions to dense vectors.
    Default model: all-MiniLM-L6-v2 (384-dim output, 22M params).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        freeze: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize text encoder.

        Args:
            model_name: SentenceTransformer model name
            freeze: If True, freeze encoder weights (no gradient updates)
            device: Device to load model on
        """
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.device = device

        # Load pre-trained SentenceBERT model
        self.encoder = SentenceTransformer(model_name, device=device)

        # Freeze weights if requested
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        # Get output dimension
        self.output_dim = self.encoder.get_sentence_embedding_dimension()

    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text descriptions to embeddings.

        Args:
            texts: Single text string or list of texts

        Returns:
            Text embeddings [batch_size, output_dim]
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]

        # Encode texts
        if self.freeze:
            with torch.no_grad():
                embeddings = self.encoder.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device
                )
        else:
            embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device
            )

        return embeddings

    def get_output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_dim

    def unfreeze(self):
        """Unfreeze encoder weights for fine-tuning."""
        self.freeze = False
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()

    def freeze_encoder(self):
        """Freeze encoder weights (disable gradient updates)."""
        self.freeze = True
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        """
        Set training mode.

        If encoder is frozen, keeps it in eval mode.

        Args:
            mode: If True, set to training mode; else eval mode
        """
        if self.freeze:
            # Keep encoder in eval mode even during training
            self.encoder.eval()
        else:
            super().train(mode)
            self.encoder.train(mode)

        return self

    def eval(self):
        """Set evaluation mode."""
        super().eval()
        self.encoder.eval()
        return self

    def to(self, device):
        """Move model to device."""
        self.device = str(device)
        self.encoder = self.encoder.to(device)
        return super().to(device)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TextEncoder(\n"
            f"  model={self.model_name},\n"
            f"  output_dim={self.output_dim},\n"
            f"  freeze={self.freeze},\n"
            f"  device={self.device}\n"
            f")"
        )

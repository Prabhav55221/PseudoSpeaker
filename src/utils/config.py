"""
Configuration dataclass for GMM-MDN training.

Defines all hyperparameters and paths needed for training and inference.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GMMMDNConfig:
    """Configuration for GMM-MDN training and inference.

    Attributes:
        Data paths:
            data_dir: Path to CapSpeech dataset
            embedding_dir: Path to x-vector ARK files
            augmented_texts_path: Path to augmented_texts.json
            mapping_dir: Path to train/dev/test mapping files

        Model architecture:
            text_encoder_name: SentenceBERT model name
            freeze_encoder: Whether to freeze text encoder weights
            num_gmm_components: Number of GMM components (K)
            embedding_dim: X-vector dimension (fixed at 512)
            hidden_dim: Hidden layer dimension

        Training:
            batch_size: Embeddings per group per batch
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization
            grad_clip: Gradient clipping max norm

        System:
            device: cuda or cpu
            num_workers: DataLoader workers
            seed: Random seed for reproducibility

        Logging:
            output_dir: Directory for checkpoints and logs
            log_interval: Log every N batches
            save_interval: Save checkpoint every N epochs
    """

    # Data paths
    data_dir: str
    embedding_dir: str
    augmented_texts_path: str
    mapping_dir: Optional[str] = None

    # Model architecture
    text_encoder_name: str = "all-MiniLM-L6-v2"
    freeze_encoder: bool = True
    num_gmm_components: int = 15
    embedding_dim: int = 512  # X-vector dimension (confirmed from logs)
    hidden_dim: int = 512

    # Training hyperparameters
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # System
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42

    # Logging
    output_dir: str = "./outputs"
    log_interval: int = 100
    save_interval: int = 5

    # Additional training settings
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 10

    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Convert string paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.embedding_dir = Path(self.embedding_dir)
        self.augmented_texts_path = Path(self.augmented_texts_path)
        self.output_dir = Path(self.output_dir)

        if self.mapping_dir:
            self.mapping_dir = Path(self.mapping_dir)

        # Validate paths exist (for data loading)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

        if not self.embedding_dir.exists():
            raise ValueError(f"Embedding directory not found: {self.embedding_dir}")

        if not self.augmented_texts_path.exists():
            raise ValueError(f"Augmented texts file not found: {self.augmented_texts_path}")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate hyperparameters
        if self.num_gmm_components < 1:
            raise ValueError(f"num_gmm_components must be >= 1, got {self.num_gmm_components}")

        if self.embedding_dim != 512:
            raise ValueError(f"embedding_dim must be 512 (x-vector dimension), got {self.embedding_dim}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")

    def to_dict(self) -> dict:
        """Convert config to dictionary (for saving)."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GMMMDNConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def save(self, path: Path):
        """Save config to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "GMMMDNConfig":
        """Load config from JSON file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

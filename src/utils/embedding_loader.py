"""
Embedding loader for Kaldi ARK format x-vectors using Hyperion.

Loads 512-dimensional x-vector embeddings from CapSpeech dataset.
Uses Hyperion's RandomAccessDataReaderFactory for proper ARK file handling.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np

from hyperion.io import RandomAccessDataReaderFactory as DRF


class HyperionEmbeddingLoader:
    """
    Loader for x-vector embeddings stored in Kaldi ARK format.

    Uses Hyperion's RandomAccessDataReaderFactory for proper ARK handling.
    Loads from multiple ARK files in the embedding directory.
    """

    def __init__(self, embedding_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize embedding loader.

        Args:
            embedding_dir: Path to directory containing ARK files
            logger: Logger instance (creates default if None)
        """
        self.embedding_dir = Path(embedding_dir)
        self.logger = logger or logging.getLogger(__name__)

        # Create reader using main xvector.csv
        self.reader = None
        self._load_ark_files()

    def _load_ark_files(self):
        """
        Load embeddings using Hyperion's RandomAccessDataReaderFactory.

        Uses the main xvector.csv file which indexes all ARK files.
        """
        # Use the main xvector.csv file (not the individual xvector.1.csv, etc.)
        main_csv = self.embedding_dir / "xvector.csv"

        if not main_csv.exists():
            raise ValueError(f"Main xvector.csv not found in {self.embedding_dir}")

        self.logger.info(f"Loading embeddings from: {main_csv}")

        try:
            self.reader = DRF.create(str(main_csv))
            self.logger.info(f"Successfully loaded embedding reader")
        except Exception as e:
            raise ValueError(f"Failed to load xvector.csv: {e}")

    def load_embedding(self, audio_id: str) -> np.ndarray:
        """
        Load x-vector embedding for given audio_id.

        Args:
            audio_id: Audio identifier (e.g., "id10230-WxmrMgdkqOw-00004.wav")

        Returns:
            X-vector embedding as numpy array (shape: [512])

        Raises:
            KeyError: If audio_id not found
        """
        # Keys in xvector.csv DO include .wav extension
        # Try with .wav first (most common case)
        if audio_id.endswith('.wav'):
            key = audio_id
            key_alt = audio_id[:-4]  # without .wav
        else:
            key = f"{audio_id}.wav"  # add .wav
            key_alt = audio_id  # keep as-is

        try:
            embedding = self.reader.read([key], squeeze=True)
        except:
            # Try alternate format if first attempt failed
            try:
                embedding = self.reader.read([key_alt], squeeze=True)
            except:
                raise KeyError(f"Audio ID not found: {audio_id} (tried: {key}, {key_alt})")

        if embedding is None or embedding.size == 0:
            raise KeyError(f"Audio ID not found: {audio_id}")

        # Validate embedding shape
        if embedding.shape[0] != 512:
            raise RuntimeError(
                f"Expected 512-dim embedding for {audio_id}, "
                f"got {embedding.shape[0]}"
            )

        return embedding.astype(np.float32)

    def load_batch(self, audio_ids: list[str]) -> np.ndarray:
        """
        Load multiple embeddings.

        Args:
            audio_ids: List of audio identifiers

        Returns:
            Stacked embeddings as numpy array (shape: [N, 512])
        """
        embeddings = []
        for audio_id in audio_ids:
            emb = self.load_embedding(audio_id)
            embeddings.append(emb)

        return np.stack(embeddings, axis=0).astype(np.float32)

    def has_embedding(self, audio_id: str) -> bool:
        """
        Check if embedding exists for audio_id.

        Args:
            audio_id: Audio identifier

        Returns:
            True if embedding available, False otherwise
        """
        try:
            self.load_embedding(audio_id)
            return True
        except KeyError:
            return False

    def get_available_ids(self) -> list[str]:
        """
        Get list of all audio_ids with available embeddings.

        Note: This is expensive as it requires iterating through all ARK files.
        Not recommended for large datasets.

        Returns:
            List of audio identifiers
        """
        self.logger.warning("get_available_ids() is expensive and not recommended")
        # This would require reading all keys from all ARK files
        # Not implemented - use dataset filtering instead
        raise NotImplementedError("Use dataset-level filtering instead")

    def __len__(self) -> int:
        """Return number of available embeddings (not implemented)."""
        raise NotImplementedError("Use dataset-level counting instead")

    def close(self):
        """Close the embedding reader."""
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all readers."""
        self.close()

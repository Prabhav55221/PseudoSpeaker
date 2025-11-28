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

        # Create readers for each ARK file using Hyperion factory
        self.readers = {}
        self._load_ark_files()

    def _load_ark_files(self):
        """
        Load ARK files using Hyperion's RandomAccessDataReaderFactory.

        Finds all .ark files in embedding_dir and creates readers for them.
        """
        ark_files = sorted(self.embedding_dir.glob("xvector.*.ark"))

        if not ark_files:
            raise ValueError(f"No xvector ARK files found in {self.embedding_dir}")

        self.logger.info(f"Loading {len(ark_files)} ARK files...")

        for ark_file in ark_files:
            try:
                reader = DRF.create(str(ark_file))
                self.readers[ark_file] = reader
                self.logger.info(f"  Loaded: {ark_file.name}")
            except Exception as e:
                self.logger.warning(f"  Failed to load {ark_file.name}: {e}")

        if not self.readers:
            raise ValueError("Failed to load any ARK files!")

        self.logger.info(f"Successfully loaded {len(self.readers)} ARK files")

    def load_embedding(self, audio_id: str) -> np.ndarray:
        """
        Load x-vector embedding for given audio_id.

        Searches across all ARK files for the embedding.

        Args:
            audio_id: Audio identifier (e.g., "id10230-WxmrMgdkqOw-00004.wav")

        Returns:
            X-vector embedding as numpy array (shape: [512])

        Raises:
            KeyError: If audio_id not found in any ARK file
        """
        # Try each reader until we find the embedding
        for ark_file, reader in self.readers.items():
            try:
                # Try with and without .wav extension
                key = audio_id.replace('.wav', '') if audio_id.endswith('.wav') else audio_id
                embedding = reader.read([key], squeeze=True)

                if embedding is not None and embedding.size > 0:
                    # Validate embedding shape
                    if embedding.shape[0] != 512:
                        raise RuntimeError(
                            f"Expected 512-dim embedding for {audio_id}, "
                            f"got {embedding.shape[0]}"
                        )
                    return embedding.astype(np.float32)

                # Try with .wav extension
                key_alt = f"{key}.wav" if not audio_id.endswith('.wav') else key[:-4]
                embedding = reader.read([key_alt], squeeze=True)

                if embedding is not None and embedding.size > 0:
                    if embedding.shape[0] != 512:
                        raise RuntimeError(
                            f"Expected 512-dim embedding for {audio_id}, "
                            f"got {embedding.shape[0]}"
                        )
                    return embedding.astype(np.float32)

            except Exception:
                continue

        raise KeyError(f"Audio ID not found in any ARK file: {audio_id}")

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
        """Close all ARK file readers."""
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all readers."""
        self.close()

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
        self.recipe_root = None  # Will be set in _load_ark_files
        self._load_ark_files()

    def _load_ark_files(self):
        """
        Load embeddings using Hyperion's RandomAccessDataReaderFactory.

        Uses the main xvector.csv file which indexes all ARK files.
        The CSV contains relative paths to ARK files that are relative to the recipe root.
        We need to change directory to the recipe root before creating the reader.
        """
        import os

        # Use the main xvector.csv file (not the individual xvector.1.csv, etc.)
        main_csv = self.embedding_dir / "xvector.csv"

        if not main_csv.exists():
            raise ValueError(f"Main xvector.csv not found in {self.embedding_dir}")

        self.logger.info(f"Loading embeddings from: {main_csv}")

        # The CSV paths are relative to the recipe root directory
        # embedding_dir format: .../recipes/voxceleb_eval/v3.6.xs/exp/xvectors/.../CapSpeech-real
        # recipe_root is: .../recipes/voxceleb_eval/v3.6.xs/
        # We need to find the recipe root by going up to the 'exp' directory

        embedding_dir_str = str(self.embedding_dir)
        if '/exp/' in embedding_dir_str:
            self.recipe_root = embedding_dir_str.split('/exp/')[0]
        else:
            # Fallback: assume embedding_dir itself contains the ARK files
            self.recipe_root = str(self.embedding_dir)

        self.logger.info(f"Recipe root directory: {self.recipe_root}")

        try:
            # Save current directory
            original_dir = os.getcwd()

            # Change to recipe root so relative paths in CSV resolve correctly
            os.chdir(self.recipe_root)
            self.logger.info(f"Changed working directory to: {self.recipe_root}")

            try:
                # Pass path directly - Hyperion auto-detects the CSV format
                # Matches the pattern from old code: reader = DRF.create(feats_file)
                self.reader = DRF.create(str(main_csv))
                self.logger.info(f"Successfully loaded embedding reader")
            finally:
                # Always restore original directory
                os.chdir(original_dir)
                self.logger.info(f"Restored working directory to: {original_dir}")

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

        import os

        # Need to be in recipe root directory for ARK file paths to resolve
        original_dir = os.getcwd()

        try:
            os.chdir(self.recipe_root)

            # Try primary key
            try:
                # Read returns a list when passed a list of keys
                embedding = self.reader.read([key])

                # Convert to numpy array if it's a list
                if isinstance(embedding, list):
                    embedding = np.array(embedding)

                # Get first element if batch dimension exists
                if len(embedding.shape) > 1:
                    embedding = embedding[0]

                # Success with primary key
                return embedding.astype(np.float32)
            except Exception as e:
                self.logger.debug(f"Failed to load with key '{key}': {type(e).__name__}: {e}")

            # Try alternate key
            try:
                # Read returns a list when passed a list of keys
                embedding = self.reader.read([key_alt])

                # Convert to numpy array if it's a list
                if isinstance(embedding, list):
                    embedding = np.array(embedding)

                # Get first element if batch dimension exists
                if len(embedding.shape) > 1:
                    embedding = embedding[0]

                # Success with alternate key
                return embedding.astype(np.float32)
            except Exception as e:
                self.logger.debug(f"Failed to load with alternate key '{key_alt}': {type(e).__name__}: {e}")

            # Both attempts failed
            raise KeyError(f"Audio ID not found: {audio_id} (tried: {key}, {key_alt})")

        finally:
            os.chdir(original_dir)

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

"""
Embedding loader for Kaldi ARK format x-vectors using Hyperion.

Loads 512-dimensional x-vector embeddings from CapSpeech dataset.
Filters to sources with available embeddings (voxceleb, ears, expresso).
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from hyperion.io import RandomAccessDataReader


# Sources with available embeddings (LibriTTS-R excluded)
VALID_SOURCES = {"voxceleb", "ears", "expresso"}


class HyperionEmbeddingLoader:
    """
    Loader for x-vector embeddings stored in Kaldi ARK format.

    Uses Hyperion's RandomAccessDataReader for efficient loading.
    Maintains mapping from audio_id to (ark_file, byte_offset).
    """

    def __init__(self, embedding_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize embedding loader.

        Args:
            embedding_dir: Path to directory containing ARK files and CSVs
            logger: Logger instance (creates default if None)
        """
        self.embedding_dir = Path(embedding_dir)
        self.logger = logger or logging.getLogger(__name__)

        # Mapping: audio_id -> (ark_file_path, byte_offset)
        self.metadata: Dict[str, Tuple[Path, int]] = {}

        # Cache of RandomAccessDataReader instances per ARK file
        self.readers: Dict[Path, RandomAccessDataReader] = {}

        # Load metadata from CSV files
        self._load_metadata()

    def _load_metadata(self):
        """
        Load metadata from CSV files in embedding directory.

        CSV format: audio_id, ark_file, byte_offset
        Filters to only include valid sources (voxceleb, ears, expresso).
        """
        csv_files = sorted(self.embedding_dir.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {self.embedding_dir}")

        self.logger.info(f"Loading metadata from {len(csv_files)} CSV files...")

        total_loaded = 0
        filtered_count = 0

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            # Validate CSV format
            required_cols = {"audio_id", "ark_file", "byte_offset"}
            if not required_cols.issubset(df.columns):
                self.logger.warning(
                    f"Skipping {csv_file.name}: missing required columns. "
                    f"Expected {required_cols}, got {set(df.columns)}"
                )
                continue

            # Filter to valid sources
            for _, row in df.iterrows():
                audio_id = row["audio_id"]

                # Extract source from audio_id (format: source_*)
                source = audio_id.split("_")[0]

                if source not in VALID_SOURCES:
                    filtered_count += 1
                    continue

                ark_file = self.embedding_dir / row["ark_file"]
                byte_offset = int(row["byte_offset"])

                # Validate ARK file exists
                if not ark_file.exists():
                    self.logger.warning(
                        f"ARK file not found for {audio_id}: {ark_file}"
                    )
                    continue

                self.metadata[audio_id] = (ark_file, byte_offset)
                total_loaded += 1

        self.logger.info(
            f"Loaded metadata for {total_loaded:,} embeddings "
            f"({filtered_count:,} filtered from invalid sources)"
        )

        if total_loaded == 0:
            raise ValueError("No valid embeddings found!")

    def _get_reader(self, ark_file: Path) -> RandomAccessDataReader:
        """
        Get or create RandomAccessDataReader for ARK file.

        Args:
            ark_file: Path to ARK file

        Returns:
            RandomAccessDataReader instance (cached)
        """
        if ark_file not in self.readers:
            self.readers[ark_file] = RandomAccessDataReader(str(ark_file))
        return self.readers[ark_file]

    def load_embedding(self, audio_id: str) -> np.ndarray:
        """
        Load x-vector embedding for given audio_id.

        Args:
            audio_id: Audio identifier (e.g., "voxceleb_id00001_00001")

        Returns:
            X-vector embedding as numpy array (shape: [512])

        Raises:
            KeyError: If audio_id not found in metadata
            RuntimeError: If embedding loading fails
        """
        if audio_id not in self.metadata:
            raise KeyError(f"Audio ID not found: {audio_id}")

        ark_file, byte_offset = self.metadata[audio_id]

        reader = self._get_reader(ark_file)
        embedding = reader.read([audio_id])[0]

        # Validate embedding shape
        if embedding.shape[0] != 512:
            raise RuntimeError(
                f"Expected 512-dim embedding for {audio_id}, "
                f"got {embedding.shape[0]}"
            )

        return embedding.astype(np.float32)

    def load_batch(self, audio_ids: list[str]) -> np.ndarray:
        """
        Load multiple embeddings efficiently.

        Groups by ARK file to minimize file operations.

        Args:
            audio_ids: List of audio identifiers

        Returns:
            Stacked embeddings as numpy array (shape: [N, 512])
        """
        # Group audio_ids by ARK file
        ark_groups: Dict[Path, list[str]] = {}

        for audio_id in audio_ids:
            if audio_id not in self.metadata:
                raise KeyError(f"Audio ID not found: {audio_id}")

            ark_file, _ = self.metadata[audio_id]

            if ark_file not in ark_groups:
                ark_groups[ark_file] = []
            ark_groups[ark_file].append(audio_id)

        # Load embeddings grouped by ARK file
        embeddings = []

        for ark_file, ids in ark_groups.items():
            reader = self._get_reader(ark_file)
            batch_embeddings = reader.read(ids)
            embeddings.extend(batch_embeddings)

        # Stack and validate
        embeddings = np.stack(embeddings, axis=0).astype(np.float32)

        if embeddings.shape != (len(audio_ids), 512):
            raise RuntimeError(
                f"Expected shape ({len(audio_ids)}, 512), "
                f"got {embeddings.shape}"
            )

        return embeddings

    def has_embedding(self, audio_id: str) -> bool:
        """
        Check if embedding exists for audio_id.

        Args:
            audio_id: Audio identifier

        Returns:
            True if embedding available, False otherwise
        """
        return audio_id in self.metadata

    def get_available_ids(self) -> list[str]:
        """
        Get list of all audio_ids with available embeddings.

        Returns:
            Sorted list of audio identifiers
        """
        return sorted(self.metadata.keys())

    def __len__(self) -> int:
        """Return number of available embeddings."""
        return len(self.metadata)

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

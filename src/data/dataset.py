"""
PyTorch Dataset for GMM-MDN training.

Loads text-embedding pairs with group-based organization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import numpy as np

from ..utils.embedding_loader import HyperionEmbeddingLoader


class GMMMDNDataset(Dataset):
    """
    Dataset for GMM-MDN training.

    Loads text descriptions and corresponding x-vector embeddings.
    Organizes samples by attribute groups for group-based batching.

    Each sample: (text, embedding, audio_id, attribute_group)
    """

    def __init__(
        self,
        json_path: Path,
        embedding_dir: Path,
        logger: logging.Logger = None
    ):
        """
        Initialize dataset.

        Args:
            json_path: Path to JSON file (train.json, dev.json, or test.json)
            embedding_dir: Path to directory with ARK files and CSVs
            logger: Logger instance
        """
        self.json_path = Path(json_path)
        self.embedding_dir = Path(embedding_dir)
        self.logger = logger or logging.getLogger(__name__)

        # Load samples from JSON
        self.samples = self._load_samples()

        # Initialize embedding loader
        self.embedding_loader = HyperionEmbeddingLoader(
            embedding_dir=self.embedding_dir,
            logger=self.logger
        )

        # Filter samples to only those with available embeddings
        self._filter_valid_samples()

        # Create group-based organization
        self.groups_to_indices = self._create_group_indices()

        self.logger.info(
            f"Loaded dataset: {len(self.samples):,} samples, "
            f"{len(self.groups_to_indices)} groups"
        )

    def _load_samples(self) -> List[Dict]:
        """
        Load samples from JSON file.

        Returns:
            List of sample dicts with keys: audio_id, text, attribute_group
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, 'r') as f:
            samples = json.load(f)

        self.logger.info(f"Loaded {len(samples):,} samples from {self.json_path.name}")
        return samples

    def _filter_valid_samples(self):
        """
        Filter samples to only those with available embeddings.

        Updates self.samples in-place.
        """
        valid_samples = []
        missing_count = 0

        for sample in self.samples:
            audio_id = sample["audio_id"]
            if self.embedding_loader.has_embedding(audio_id):
                valid_samples.append(sample)
            else:
                missing_count += 1

        self.samples = valid_samples

        if missing_count > 0:
            self.logger.warning(
                f"Filtered out {missing_count:,} samples with missing embeddings"
            )

    def _create_group_indices(self) -> Dict[str, List[int]]:
        """
        Create mapping from attribute_group to sample indices.

        Returns:
            Dict mapping attribute_group -> list of indices
        """
        groups_to_indices = defaultdict(list)

        for idx, sample in enumerate(self.samples):
            attr_group = sample["attribute_group"]
            groups_to_indices[attr_group].append(idx)

        return dict(groups_to_indices)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, str, str]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (text, embedding, audio_id, attribute_group)
        """
        sample = self.samples[idx]

        text = sample["text"]
        audio_id = sample["audio_id"]
        attribute_group = sample["attribute_group"]

        # Load embedding
        embedding = self.embedding_loader.load_embedding(audio_id)
        embedding = torch.from_numpy(embedding).float()

        return text, embedding, audio_id, attribute_group

    def get_group_samples(self, attribute_group: str) -> List[int]:
        """
        Get all sample indices for a given attribute group.

        Args:
            attribute_group: Attribute group name

        Returns:
            List of sample indices
        """
        return self.groups_to_indices.get(attribute_group, [])

    def get_all_groups(self) -> List[str]:
        """
        Get list of all attribute groups in dataset.

        Returns:
            List of attribute group names
        """
        return list(self.groups_to_indices.keys())

    def get_group_stats(self) -> Dict[str, int]:
        """
        Get statistics about samples per group.

        Returns:
            Dict mapping attribute_group -> sample count
        """
        return {
            group: len(indices)
            for group, indices in self.groups_to_indices.items()
        }

    def close(self):
        """Close embedding loader."""
        self.embedding_loader.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class GroupBatchSampler:
    """
    Batch sampler that samples from a single group per batch.

    Ensures each batch contains embeddings from the same attribute group.
    """

    def __init__(
        self,
        dataset: GMMMDNDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42
    ):
        """
        Initialize group batch sampler.

        Args:
            dataset: GMMMDNDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle groups and samples
            drop_last: Whether to drop last incomplete batch
            seed: Random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self.groups = dataset.get_all_groups()
        self.groups_to_indices = dataset.groups_to_indices

        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        """Generate batches."""
        # Shuffle groups if requested
        if self.shuffle:
            groups = self.rng.permutation(self.groups).tolist()
        else:
            groups = self.groups.copy()

        # Generate batches for each group
        for group in groups:
            indices = self.groups_to_indices[group].copy()

            # Shuffle indices within group
            if self.shuffle:
                self.rng.shuffle(indices)

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]

                # Skip incomplete batches if requested
                if self.drop_last and len(batch_indices) < self.batch_size:
                    continue

                yield batch_indices

    def __len__(self) -> int:
        """Return number of batches."""
        total_batches = 0

        for group in self.groups:
            num_samples = len(self.groups_to_indices[group])

            if self.drop_last:
                total_batches += num_samples // self.batch_size
            else:
                total_batches += (num_samples + self.batch_size - 1) // self.batch_size

        return total_batches

    def set_epoch(self, epoch: int):
        """Set epoch for reproducibility with shuffling."""
        self.rng = np.random.RandomState(self.seed + epoch)

"""
DataLoader utilities for GMM-MDN training.

Custom collation and group-based batching.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import GMMMDNDataset, GroupBatchSampler


def collate_fn(batch: List[Tuple]) -> Tuple[List[str], torch.Tensor, List[str], str]:
    """
    Custom collate function for GMM-MDN batches.

    Args:
        batch: List of (text, embedding, audio_id, attribute_group) tuples

    Returns:
        Tuple of:
            - texts: List of text strings [batch_size]
            - embeddings: Stacked embeddings [batch_size, 512]
            - audio_ids: List of audio IDs [batch_size]
            - attribute_group: Shared attribute group (all samples in batch have same group)
    """
    texts = []
    embeddings = []
    audio_ids = []
    attribute_groups = set()

    for text, embedding, audio_id, attribute_group in batch:
        texts.append(text)
        embeddings.append(embedding)
        audio_ids.append(audio_id)
        attribute_groups.add(attribute_group)

    # Stack embeddings
    embeddings = torch.stack(embeddings, dim=0)

    # Verify all samples are from same group (sanity check)
    if len(attribute_groups) > 1:
        raise ValueError(
            f"Batch contains samples from multiple groups: {attribute_groups}. "
            "This should not happen with GroupBatchSampler."
        )

    attribute_group = attribute_groups.pop()

    return texts, embeddings, audio_ids, attribute_group


def create_dataloader(
    json_path: Path,
    embedding_dir: Path,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: int = 42,
    logger: logging.Logger = None
) -> DataLoader:
    """
    Create DataLoader with group-based batching.

    Args:
        json_path: Path to JSON file (train.json, dev.json, or test.json)
        embedding_dir: Path to directory with ARK files
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle groups and samples
        num_workers: Number of DataLoader workers
        drop_last: Whether to drop last incomplete batch
        seed: Random seed for reproducibility
        logger: Logger instance

    Returns:
        DataLoader instance
    """
    logger = logger or logging.getLogger(__name__)

    # Create dataset
    dataset = GMMMDNDataset(
        json_path=json_path,
        embedding_dir=embedding_dir,
        logger=logger
    )

    # Log dataset statistics
    group_stats = dataset.get_group_stats()
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total samples: {len(dataset):,}")
    logger.info(f"  Number of groups: {len(group_stats)}")
    logger.info(f"  Samples per group: {min(group_stats.values()):,} - {max(group_stats.values()):,}")

    # Create group batch sampler
    sampler = GroupBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed
    )

    logger.info(f"  Total batches: {len(sampler):,}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader


def create_train_val_dataloaders(
    train_json: Path,
    dev_json: Path,
    embedding_dir: Path,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    logger: logging.Logger = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        train_json: Path to train.json
        dev_json: Path to dev.json
        embedding_dir: Path to directory with ARK files
        batch_size: Number of samples per batch
        num_workers: Number of DataLoader workers
        seed: Random seed
        logger: Logger instance

    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger = logger or logging.getLogger(__name__)

    logger.info("Creating training DataLoader...")
    train_loader = create_dataloader(
        json_path=train_json,
        embedding_dir=embedding_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        seed=seed,
        logger=logger
    )

    logger.info("Creating validation DataLoader...")
    val_loader = create_dataloader(
        json_path=dev_json,
        embedding_dir=embedding_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        seed=seed,
        logger=logger
    )

    return train_loader, val_loader

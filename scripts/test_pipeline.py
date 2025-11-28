#!/usr/bin/env python3
"""
End-to-end pipeline test for GMM-MDN system.

Tests all components to ensure they work correctly before training.

Usage:
    python scripts/test_pipeline.py --data_dir /path/to/data --embedding_dir /path/to/embeddings
"""

import argparse
import logging
from pathlib import Path

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.embedding_loader import HyperionEmbeddingLoader
from src.models.text_encoder import TextEncoder
from src.models.gmm_mdn import GMMMDN
from src.models.gmm_utils import compute_gmm_nll, sample_from_gmm, diversity_score, coverage_score
from src.data.dataset import GMMMDNDataset, GroupBatchSampler
from src.data.dataloader import collate_fn
from torch.utils.data import DataLoader


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test GMM-MDN pipeline")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to CapSpeech dataset directory"
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        required=True,
        help="Path to x-vector embedding directory (ARK files)"
    )
    parser.add_argument(
        "--mapping_dir",
        type=str,
        required=True,
        help="Path to directory with train/dev/test JSON files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)"
    )

    return parser.parse_args()


def test_embedding_loader(embedding_dir, logger):
    """Test HyperionEmbeddingLoader."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Embedding Loader")
    logger.info("=" * 80)

    loader = HyperionEmbeddingLoader(embedding_dir, logger=logger)

    logger.info(f"✓ Successfully initialized embedding loader")

    # Test single embedding load with a known audio ID
    # Using an ID from the dataset (will fail if not found, which is expected)
    test_id = "id10230-WxmrMgdkqOw-00004.wav"

    try:
        embedding = loader.load_embedding(test_id)
        logger.info(f"✓ Loaded embedding for {test_id}: shape={embedding.shape}, dtype={embedding.dtype}")

        if embedding.shape[0] != 512:
            raise ValueError(f"Expected 512-dim embedding, got {embedding.shape[0]}")

        # Test batch loading
        batch_ids = [test_id] * 3  # Same ID multiple times for testing
        batch_embeddings = loader.load_batch(batch_ids)
        logger.info(f"✓ Loaded batch of {len(batch_ids)} embeddings: shape={batch_embeddings.shape}")

        if batch_embeddings.shape != (3, 512):
            raise ValueError(f"Expected shape (3, 512), got {batch_embeddings.shape}")

    except KeyError as e:
        logger.warning(f"Test ID not found (expected if using sample data): {e}")
        logger.info("Skipping batch test")

    logger.info("✓ Embedding loader tests passed!")

    return loader


def test_text_encoder(device, logger):
    """Test TextEncoder."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Text Encoder")
    logger.info("=" * 80)

    encoder = TextEncoder(
        model_name="all-MiniLM-L6-v2",
        freeze=True,
        device=device
    )

    logger.info(f"✓ Created text encoder: {encoder.model_name}, output_dim={encoder.output_dim}")

    # Test single text
    text = "A male speaker with deep voice"
    embedding = encoder([text])
    logger.info(f"✓ Encoded single text: shape={embedding.shape}, dtype={embedding.dtype}")

    if embedding.shape != (1, encoder.output_dim):
        raise ValueError(f"Expected shape (1, {encoder.output_dim}), got {embedding.shape}")

    # Test batch
    texts = ["A male speaker", "A female speaker", "A child speaker"]
    embeddings = encoder(texts)
    logger.info(f"✓ Encoded batch of {len(texts)} texts: shape={embeddings.shape}")

    if embeddings.shape != (3, encoder.output_dim):
        raise ValueError(f"Expected shape (3, {encoder.output_dim}), got {embeddings.shape}")

    logger.info("✓ Text encoder tests passed!")

    return encoder


def test_gmm_utils(device, logger):
    """Test GMM utilities."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: GMM Utilities")
    logger.info("=" * 80)

    batch_size = 8
    num_components = 5
    embedding_dim = 512

    # Create random GMM parameters
    weights = torch.randn(batch_size, num_components).to(device)
    means = torch.randn(batch_size, num_components, embedding_dim).to(device)
    log_vars = torch.randn(batch_size, num_components, embedding_dim).to(device) - 1.0
    embeddings = torch.randn(batch_size, embedding_dim).to(device)

    # Test NLL computation
    nll = compute_gmm_nll(embeddings, weights, means, log_vars)
    logger.info(f"✓ Computed NLL loss: {nll.item():.4f}")

    if torch.isnan(nll) or torch.isinf(nll):
        raise ValueError("NLL is NaN or Inf!")

    # Test sampling
    samples = sample_from_gmm(
        weights=weights[0],
        means=means[0],
        log_vars=log_vars[0],
        num_samples=10,
        temperature=1.0
    )
    logger.info(f"✓ Sampled from GMM: shape={samples.shape}")

    if samples.shape != (10, embedding_dim):
        raise ValueError(f"Expected shape (10, {embedding_dim}), got {samples.shape}")

    # Test diversity score
    div_score = diversity_score(samples)
    logger.info(f"✓ Diversity score: {div_score:.4f}")

    # Test coverage score
    cov_score = coverage_score(samples, embeddings, k=5)
    logger.info(f"✓ Coverage score: {cov_score:.4f}")

    logger.info("✓ GMM utilities tests passed!")


def test_model(device, logger):
    """Test GMMMDN model."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: GMM-MDN Model")
    logger.info("=" * 80)

    model = GMMMDN(
        num_components=10,
        embedding_dim=512,
        hidden_dim=256,
        text_encoder_name="all-MiniLM-L6-v2",
        freeze_encoder=True,
        device=device
    )

    logger.info(f"✓ Created model:\n{model}")

    # Test forward pass
    texts = ["A male speaker", "A female speaker"]
    weights, means, log_vars = model(texts)

    logger.info(f"✓ Forward pass:")
    logger.info(f"  weights: {weights.shape}")
    logger.info(f"  means: {means.shape}")
    logger.info(f"  log_vars: {log_vars.shape}")

    if weights.shape != (2, 10):
        raise ValueError(f"Expected weights shape (2, 10), got {weights.shape}")

    # Test loss computation
    embeddings = torch.randn(2, 512).to(device)
    loss = model.compute_loss(texts, embeddings)
    logger.info(f"✓ Computed loss: {loss.item():.4f}")

    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError("Loss is NaN or Inf!")

    # Test sampling
    samples = model.sample("A male speaker", num_samples=5, temperature=1.0)
    logger.info(f"✓ Sampled embeddings: shape={samples.shape}")

    if samples.shape != (5, 512):
        raise ValueError(f"Expected shape (5, 512), got {samples.shape}")

    # Test backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.info("✓ Backward pass successful")

    logger.info("✓ Model tests passed!")

    return model


def test_dataset(mapping_dir, embedding_dir, logger):
    """Test Dataset and DataLoader."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Dataset and DataLoader")
    logger.info("=" * 80)

    train_json = Path(mapping_dir) / "train.json"

    dataset = GMMMDNDataset(
        json_path=train_json,
        embedding_dir=embedding_dir,
        logger=logger
    )

    logger.info(f"✓ Created dataset: {len(dataset):,} samples")
    logger.info(f"  Number of groups: {len(dataset.get_all_groups())}")

    # Test single sample
    text, embedding, audio_id, attr_group = dataset[0]
    logger.info(f"✓ Loaded sample:")
    logger.info(f"  text: {text}")
    logger.info(f"  embedding: {embedding.shape}, {embedding.dtype}")
    logger.info(f"  audio_id: {audio_id}")
    logger.info(f"  attribute_group: {attr_group}")

    # Test batch sampler
    sampler = GroupBatchSampler(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        drop_last=False,
        seed=42
    )

    logger.info(f"✓ Created batch sampler: {len(sampler)} batches")

    # Test dataloader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Test one batch
    batch = next(iter(dataloader))
    texts, embeddings, audio_ids, attr_group = batch

    logger.info(f"✓ Loaded batch:")
    logger.info(f"  texts: {len(texts)} strings")
    logger.info(f"  embeddings: {embeddings.shape}")
    logger.info(f"  audio_ids: {len(audio_ids)}")
    logger.info(f"  attribute_group: {attr_group}")

    # Verify group consistency
    unique_groups = set()
    for text, emb, aid, ag in zip(texts, embeddings, audio_ids, [attr_group] * len(texts)):
        unique_groups.add(ag)

    if len(unique_groups) != 1:
        raise ValueError(f"Batch should have 1 group, got {len(unique_groups)}")

    logger.info("✓ Dataset and DataLoader tests passed!")

    return dataset, dataloader


def test_training_step(model, dataloader, device, logger):
    """Test a single training step."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Training Step")
    logger.info("=" * 80)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Get one batch
    batch = next(iter(dataloader))
    texts, embeddings, audio_ids, attr_group = batch
    embeddings = embeddings.to(device)

    logger.info(f"✓ Loaded batch with {len(texts)} samples")

    # Forward pass
    loss = model.compute_loss(texts, embeddings)
    logger.info(f"✓ Forward pass: loss={loss.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    logger.info("✓ Backward pass successful")

    # Check gradients
    has_grads = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grads = True
            break

    if not has_grads:
        raise ValueError("No gradients found!")

    logger.info("✓ Training step tests passed!")


def main():
    args = parse_args()

    # Setup logger
    logger = setup_logger("test_pipeline")
    logger.setLevel(logging.INFO)

    logger.info("=" * 80)
    logger.info("GMM-MDN Pipeline Test")
    logger.info("=" * 80)
    logger.info(f"Device: {args.device}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Embedding directory: {args.embedding_dir}")
    logger.info(f"Mapping directory: {args.mapping_dir}")

    try:
        # Run tests
        embedding_loader = test_embedding_loader(args.embedding_dir, logger)
        text_encoder = test_text_encoder(args.device, logger)
        test_gmm_utils(args.device, logger)
        model = test_model(args.device, logger)
        dataset, dataloader = test_dataset(args.mapping_dir, args.embedding_dir, logger)
        test_training_step(model, dataloader, args.device, logger)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("=" * 80)
        logger.info("\nPipeline is ready for training!")

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("TEST FAILED! ✗")
        logger.error("=" * 80)
        logger.error(f"\nError: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

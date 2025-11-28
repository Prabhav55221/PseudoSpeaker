#!/usr/bin/env python3
"""
Main training script for GMM-MDN pseudo-speaker generation.

Usage:
    python scripts/train.py --data_dir /path/to/data --embedding_dir /path/to/embeddings
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import GMMMDNConfig
from src.utils.logger import setup_logger, log_config
from src.models.gmm_mdn import GMMMDN
from src.data.dataloader import create_train_val_dataloaders
from src.training.trainer import Trainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GMM-MDN for pseudo-speaker generation"
    )

    # Required paths
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
        "--augmented_texts_path",
        type=str,
        required=True,
        help="Path to augmented_texts.json"
    )

    # Data preparation
    parser.add_argument(
        "--mapping_dir",
        type=str,
        default=None,
        help="Path to directory with train/dev/test JSON files (if not provided, will be created)"
    )

    # Model architecture
    parser.add_argument(
        "--num_gmm_components",
        type=int,
        default=15,
        help="Number of GMM components (default: 15)"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden layer dimension (default: 512)"
    )
    parser.add_argument(
        "--text_encoder_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceBERT model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=True,
        help="Freeze text encoder weights (default: True)"
    )
    parser.add_argument(
        "--finetune_encoder",
        action="store_true",
        help="Fine-tune text encoder (overrides --freeze_encoder)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (embeddings per group) (default: 64)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="L2 regularization weight (default: 1e-5)"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (default: 1.0)"
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu) (default: cuda if available)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs (default: ./outputs)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Log every N batches (default: 100)"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)"
    )

    # Scheduler and early stopping
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=5,
        help="LR scheduler patience (default: 5)"
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.5,
        help="LR scheduler reduction factor (default: 0.5)"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = output_dir / "train.log"
    logger = setup_logger("gmm_mdn", log_file=log_file)

    logger.info("=" * 80)
    logger.info("GMM-MDN Pseudo-Speaker Generation - Training")
    logger.info("=" * 80)

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Handle encoder freezing
    freeze_encoder = args.freeze_encoder and not args.finetune_encoder
    if args.finetune_encoder:
        logger.info("Fine-tuning text encoder (overriding freeze_encoder)")

    # Create configuration
    config = GMMMDNConfig(
        data_dir=args.data_dir,
        embedding_dir=args.embedding_dir,
        augmented_texts_path=args.augmented_texts_path,
        mapping_dir=args.mapping_dir,
        text_encoder_name=args.text_encoder_name,
        freeze_encoder=freeze_encoder,
        num_gmm_components=args.num_gmm_components,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        early_stopping_patience=args.early_stopping_patience
    )

    # Log configuration
    log_config(logger, config)

    # Save configuration
    config.save(output_dir / "config.json")
    logger.info(f"Saved config to {output_dir / 'config.json'}")

    # Check if mapping files exist
    if config.mapping_dir is None:
        logger.error("Mapping directory not provided!")
        logger.error("Please run src/data/prepare_dataset.py first to create train/dev/test splits")
        return

    train_json = config.mapping_dir / "train.json"
    dev_json = config.mapping_dir / "dev.json"

    if not train_json.exists() or not dev_json.exists():
        logger.error(f"Mapping files not found in {config.mapping_dir}")
        logger.error("Please run src/data/prepare_dataset.py first to create train/dev/test splits")
        return

    # Create dataloaders
    logger.info("\nCreating DataLoaders...")
    train_loader, val_loader = create_train_val_dataloaders(
        train_json=train_json,
        dev_json=dev_json,
        embedding_dir=config.embedding_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        logger=logger
    )

    # Create model
    logger.info("\nCreating model...")
    model = GMMMDN(
        num_components=config.num_gmm_components,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        text_encoder_name=config.text_encoder_name,
        freeze_encoder=config.freeze_encoder,
        device=config.device
    )

    model.to(config.device)
    logger.info(f"\n{model}")

    # Create trainer
    logger.info("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger
    )

    # Resume from checkpoint if provided
    if args.resume:
        trainer.resume_from_checkpoint(Path(args.resume))

    # Train
    trainer.train()

    logger.info("\nTraining finished!")
    logger.info(f"Best model saved to: {trainer.checkpoint_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()

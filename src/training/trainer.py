"""
Trainer for GMM-MDN model.

Handles training loop, validation, checkpointing, and early stopping.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.gmm_mdn import GMMMDN
from ..models.gmm_utils import validate_gmm_params
from ..utils.logger import MetricsLogger
from ..utils.config import GMMMDNConfig
from .evaluator import Evaluator


class Trainer:
    """
    Trainer for GMM-MDN model.

    Handles training loop with:
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    """

    def __init__(
        self,
        model: GMMMDN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: GMMMDNConfig,
        logger: logging.Logger
    ):
        """
        Initialize trainer.

        Args:
            model: GMM-MDN model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training configuration
            logger: Logger instance
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler (ReduceLROnPlateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True
        )

        # Metrics logger
        self.metrics_logger = MetricsLogger(
            logger=logger,
            log_interval=config.log_interval
        )

        # Evaluator
        self.evaluator = Evaluator(
            model=model,
            dataloader=val_loader,
            device=config.device,
            logger=logger
        )

        # Checkpointing
        self.checkpoint_dir = config.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = config.early_stopping_patience

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        self.metrics_logger.reset()

        epoch_loss = 0.0
        num_batches = 0

        # Set epoch for reproducible shuffling
        if hasattr(self.train_loader.batch_sampler, 'set_epoch'):
            self.train_loader.batch_sampler.set_epoch(epoch)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.epochs}",
            disable=not self.logger.isEnabledFor(logging.INFO)
        )

        for texts, embeddings, audio_ids, attribute_group in pbar:
            embeddings = embeddings.to(self.config.device)

            # Forward pass
            loss = self.model.compute_loss(texts, embeddings)

            # Check for NaN loss
            if torch.isnan(loss):
                self.logger.error(f"NaN loss detected at step {self.global_step}")

                # Validate GMM parameters
                weights, means, log_vars = self.model.forward(texts)
                is_valid, error_msg = validate_gmm_params(weights, means, log_vars)
                if not is_valid:
                    self.logger.error(f"Invalid GMM params: {error_msg}")

                raise ValueError("NaN loss encountered!")

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

            # Optimizer step
            self.optimizer.step()

            # Update metrics
            self.metrics_logger.update(loss=loss.item())
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log metrics
            self.metrics_logger.log(prefix="Train")

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        # Average epoch loss
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        return avg_loss

    def validate(self) -> dict:
        """
        Run validation.

        Returns:
            Dict of validation metrics
        """
        metrics = self.evaluator.evaluate(
            num_samples=10,
            temperature=1.0,
            compute_diversity=True,
            compute_coverage=True
        )

        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

        self.model.save_checkpoint(
            path=str(checkpoint_path),
            epoch=epoch,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            metrics=metrics,
            global_step=self.global_step
        )

        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            self.model.save_checkpoint(
                path=str(best_path),
                epoch=epoch,
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict(),
                metrics=metrics,
                global_step=self.global_step
            )
            self.logger.info(f"Saved best model: {best_path}")

    def train(self):
        """
        Full training loop.

        Trains for config.epochs with validation and checkpointing.
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting training")
        self.logger.info("=" * 80)
        self.logger.info(f"Model: {self.model}")
        self.logger.info(f"Train batches: {len(self.train_loader):,}")
        self.logger.info(f"Val batches: {len(self.val_loader):,}")
        self.logger.info("=" * 80)

        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss = self.train_epoch(epoch)

            self.logger.info(
                f"Epoch {epoch}/{self.config.epochs} - "
                f"Train loss: {train_loss:.4f}"
            )

            # Validate
            val_metrics = self.validate()

            # Update learning rate scheduler
            self.scheduler.step(val_metrics['val_loss'])

            # Check for best model
            val_loss = val_metrics['val_loss']
            is_best = val_loss < self.best_val_loss

            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.logger.info(f"New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"No improvement for {self.patience_counter} epochs "
                    f"(patience: {self.early_stopping_patience})"
                )

            # Save checkpoint
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {self.early_stopping_patience} epochs)"
                )
                break

        self.logger.info("=" * 80)
        self.logger.info("Training complete!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info("=" * 80)

    def resume_from_checkpoint(self, checkpoint_path: Path):
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)

        # Load best validation loss
        if 'metrics' in checkpoint:
            self.best_val_loss = checkpoint['metrics'].get('val_loss', float('inf'))

        self.logger.info(
            f"Resumed from epoch {self.current_epoch}, "
            f"step {self.global_step}, "
            f"best val loss {self.best_val_loss:.4f}"
        )

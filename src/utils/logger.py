"""
Logging utilities for GMM-MDN training.

Simple logger that writes to both console and file.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Path to log file (if None, only console logging)
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file provided)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


class MetricsLogger:
    """
    Simple metrics logger for tracking training stats.

    Maintains running averages and logs periodically.
    """

    def __init__(self, logger: logging.Logger, log_interval: int = 100):
        """
        Initialize metrics logger.

        Args:
            logger: Logger instance to use
            log_interval: Log every N updates
        """
        self.logger = logger
        self.log_interval = log_interval
        self.metrics = {}
        self.counts = {}
        self.step = 0

    def update(self, **kwargs):
        """
        Update metrics with new values.

        Args:
            **kwargs: Metric name and value pairs
        """
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value
            self.counts[key] += 1

        self.step += 1

    def log(self, prefix: str = "Train", force: bool = False):
        """
        Log accumulated metrics if interval reached.

        Args:
            prefix: Prefix for log message
            force: Force logging regardless of interval
        """
        if not force and self.step % self.log_interval != 0:
            return

        if not self.metrics:
            return

        # Compute averages
        avg_metrics = {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
        }

        # Format message
        metrics_str = " | ".join([
            f"{key}: {value:.4f}"
            for key, value in avg_metrics.items()
        ])

        self.logger.info(f"{prefix} | Step {self.step} | {metrics_str}")

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
        self.step = 0

    def get_average(self, key: str) -> float:
        """
        Get average value for a specific metric.

        Args:
            key: Metric name

        Returns:
            Average value (0.0 if metric doesn't exist)
        """
        if key not in self.metrics or self.counts[key] == 0:
            return 0.0
        return self.metrics[key] / self.counts[key]


def log_config(logger: logging.Logger, config):
    """
    Log configuration settings.

    Args:
        logger: Logger instance
        config: Configuration object (GMMMDNConfig)
    """
    logger.info("="*80)
    logger.info("Configuration")
    logger.info("="*80)

    config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)

    for key, value in sorted(config_dict.items()):
        logger.info(f"  {key:30s}: {value}")

    logger.info("="*80)

"""
Logging utilities for experiment tracking and monitoring
Supports TensorBoard, structured logging, and metric tracking
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


def setup_logger(name='pxgan', log_dir=None, level=logging.INFO):
    """
    Setup structured logger with file and console handlers

    Args:
        name: Logger name
        log_dir: Directory for log files (None = console only)
        level: Logging level

    Returns:
        logging.Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")

    return logger


class MetricLogger:
    """
    Tracks and aggregates metrics during training/evaluation
    """

    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = {}

    def update(self, **kwargs):
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)

    def get_average(self, key):
        """Get average of a metric"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return sum(self.metrics[key]) / len(self.metrics[key])

    def get_last(self, key):
        """Get last value of a metric"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return self.metrics[key][-1]

    def get_all_averages(self):
        """Get averages of all metrics"""
        return {key: self.get_average(key) for key in self.metrics.keys()}

    def reset(self):
        """Reset all metrics"""
        self.metrics = defaultdict(list)

    def save_epoch(self, epoch):
        """Save current averages for an epoch"""
        self.epoch_metrics[epoch] = self.get_all_averages()

    def get_summary(self):
        """Get summary of all epoch metrics"""
        return self.epoch_metrics

    def save_to_file(self, path):
        """Save metrics to JSON file"""
        with open(path, 'w') as f:
            json.dump({
                'current': dict(self.metrics),
                'epochs': self.epoch_metrics
            }, f, indent=2)


class ExperimentTracker:
    """
    Comprehensive experiment tracking with TensorBoard support
    """

    def __init__(self, exp_dir, config=None, enabled=True, backend='tensorboard'):
        """
        Args:
            exp_dir: Experiment directory
            config: Configuration dict to save
            enabled: Whether tracking is enabled
            backend: 'tensorboard', 'wandb', or 'none'
        """
        self.exp_dir = Path(exp_dir)
        self.enabled = enabled
        self.backend = backend
        self.step = 0

        if not self.enabled:
            return

        # Create directories
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.exp_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)

        # Save config
        if config is not None:
            config_path = self.exp_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

        # Setup backend
        if self.backend == 'tensorboard':
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None

        # Metric logger
        self.metric_logger = MetricLogger()

        print(f"Experiment tracking initialized at {self.exp_dir}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix=''):
        """
        Log metrics to tracker

        Args:
            metrics: Dictionary of metric names to values
            step: Step number (uses internal counter if None)
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        if not self.enabled:
            return

        if step is None:
            step = self.step
            self.step += 1

        # Update metric logger
        self.metric_logger.update(**metrics)

        # Log to backend
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f'{prefix}{name}', value, step)

    def log_images(self, images: torch.Tensor, step: Optional[int] = None, tag='images'):
        """
        Log images to TensorBoard

        Args:
            images: Tensor of shape (N, C, H, W) or (C, H, W)
            step: Step number
            tag: Image tag/name
        """
        if not self.enabled or self.writer is None:
            return

        if step is None:
            step = self.step

        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Normalize to [0, 1] if needed
        if images.min() < 0:
            images = (images - images.min()) / (images.max() - images.min())

        self.writer.add_images(tag, images, step)

    def log_histogram(self, values: torch.Tensor, step: Optional[int] = None, tag='histogram'):
        """Log histogram of values"""
        if not self.enabled or self.writer is None:
            return

        if step is None:
            step = self.step

        self.writer.add_histogram(tag, values, step)

    def log_text(self, text: str, step: Optional[int] = None, tag='text'):
        """Log text"""
        if not self.enabled or self.writer is None:
            return

        if step is None:
            step = self.step

        self.writer.add_text(tag, text, step)

    def log_model_graph(self, model, input_tensor):
        """Log model computational graph"""
        if not self.enabled or self.writer is None:
            return

        try:
            self.writer.add_graph(model, input_tensor)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")

    def save_checkpoint(self, checkpoint_dict: Dict[str, Any], filename: str, is_best=False):
        """
        Save checkpoint

        Args:
            checkpoint_dict: Dictionary containing model states, optimizer, etc.
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        if not self.enabled:
            return

        checkpoint_path = self.exp_dir / filename
        torch.save(checkpoint_dict, checkpoint_path)

        if is_best:
            best_path = self.exp_dir / 'best_model.pt'
            torch.save(checkpoint_dict, best_path)

    def close(self):
        """Close tracker and flush logs"""
        if not self.enabled:
            return

        # Save metric summary
        metrics_path = self.exp_dir / 'metrics_summary.json'
        self.metric_logger.save_to_file(metrics_path)

        # Close writer
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        print(f"Experiment tracking closed. Logs saved to {self.exp_dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

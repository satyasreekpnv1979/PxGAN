"""
Checkpointing utilities for saving and loading model states
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Any
import json


class CheckpointManager:
    """
    Manages model checkpoints during training

    Handles saving, loading, and tracking best models
    """

    def __init__(self,
                 checkpoint_dir: str,
                 max_to_keep: int = 5,
                 save_best_only: bool = False):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep (0 = keep all)
            save_best_only: Only save best model based on metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_to_keep = max_to_keep
        self.save_best_only = save_best_only

        self.best_metric = None
        self.best_epoch = None
        self.checkpoints = []  # List of saved checkpoint paths

    def save(self,
             epoch: int,
             models: Dict[str, torch.nn.Module],
             optimizers: Dict[str, torch.optim.Optimizer],
             schedulers: Optional[Dict] = None,
             metric: Optional[float] = None,
             metadata: Optional[Dict[str, Any]] = None):
        """
        Save checkpoint

        Args:
            epoch: Current epoch
            models: Dictionary of models to save
            optimizers: Dictionary of optimizers
            schedulers: Optional dictionary of schedulers
            metric: Validation metric (for best model tracking)
            metadata: Additional metadata to save
        """
        # Check if this is the best model
        is_best = False
        if metric is not None:
            if self.best_metric is None or metric > self.best_metric:
                self.best_metric = metric
                self.best_epoch = epoch
                is_best = True

        # Don't save if save_best_only and not best
        if self.save_best_only and not is_best:
            return

        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'models': {name: model.state_dict() for name, model in models.items()},
            'optimizers': {name: opt.state_dict() for name, opt in optimizers.items()},
            'metric': metric,
            'is_best': is_best,
        }

        if schedulers is not None:
            checkpoint['schedulers'] = {name: sched.state_dict() for name, sched in schedulers.items()}

        if metadata is not None:
            checkpoint['metadata'] = metadata

        # Save checkpoint
        checkpoint_path =self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, checkpoint_path)

        self.checkpoints.append(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  → New best model (metric={metric:.4f})")

        # Clean up old checkpoints if needed
        if self.max_to_keep > 0 and len(self.checkpoints) > self.max_to_keep:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to stay within max_to_keep"""
        while len(self.checkpoints) > self.max_to_keep:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                print(f"Removed old checkpoint: {old_checkpoint}")

    def load_latest(self) -> Optional[Dict]:
        """Load the latest checkpoint"""
        if not self.checkpoints:
            # Search for checkpoints in directory
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if checkpoints:
                return load_checkpoint(str(checkpoints[-1]))
        else:
            if self.checkpoints:
                return load_checkpoint(str(self.checkpoints[-1]))

        return None

    def load_best(self) -> Optional[Dict]:
        """Load the best checkpoint"""
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            return load_checkpoint(str(best_path))
        return None


def save_checkpoint(path: str,
                   epoch: int,
                   models: Dict[str, torch.nn.Module],
                   optimizers: Dict[str, torch.optim.Optimizer],
                   schedulers: Optional[Dict] = None,
                   metadata: Optional[Dict] = None):
    """
    Save a checkpoint to a specific path

    Args:
        path: Path to save checkpoint
        epoch: Current epoch
        models: Dictionary of models
        optimizers: Dictionary of optimizers
        schedulers: Optional dictionary of schedulers
        metadata: Optional metadata
    """
    checkpoint = {
        'epoch': epoch,
        'models': {name: model.state_dict() for name, model in models.items()},
        'optimizers': {name: opt.state_dict() for name, opt in optimizers.items()},
    }

    if schedulers is not None:
        checkpoint['schedulers'] = {name: sched.state_dict() for name, sched in schedulers.items()}

    if metadata is not None:
        checkpoint['metadata'] = metadata

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str,
                   models: Optional[Dict[str, torch.nn.Module]] = None,
                   optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
                   schedulers: Optional[Dict] = None,
                   device: Optional[torch.device] = None) -> Dict:
    """
    Load a checkpoint

    Args:
        path: Path to checkpoint file
        models: Optional dictionary of models to load state into
        optimizers: Optional dictionary of optimizers to load state into
        schedulers: Optional dictionary of schedulers to load state into
        device: Device to load checkpoint on

    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(path, map_location=device)

    # Load model states
    if models is not None:
        for name, model in models.items():
            if name in checkpoint['models']:
                model.load_state_dict(checkpoint['models'][name])
                print(f"Loaded {name} from checkpoint")

    # Load optimizer states
    if optimizers is not None:
        for name, opt in optimizers.items():
            if name in checkpoint['optimizers']:
                opt.load_state_dict(checkpoint['optimizers'][name])
                print(f"Loaded {name} optimizer from checkpoint")

    # Load scheduler states
    if schedulers is not None and 'schedulers' in checkpoint:
        for name, sched in schedulers.items():
            if name in checkpoint['schedulers']:
                sched.load_state_dict(checkpoint['schedulers'][name])
                print(f"Loaded {name} scheduler from checkpoint")

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    return checkpoint

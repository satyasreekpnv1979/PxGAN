"""
Visualization utilities for pixel grids, loss curves, and metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch


def plot_pixel_grid(image_tensor,
                   channel_names=None,
                   save_path=None,
                   title='Pixel Grid Visualization',
                   cmap='viridis'):
    """
    Visualize multi-channel pixel grid

    Args:
        image_tensor: Tensor of shape (C, H, W) or (H, W)
        channel_names: List of channel names (length C)
        save_path: Path to save figure (None = display only)
        title: Figure title
        cmap: Colormap name
    """
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.detach().cpu().numpy()

    # Handle single channel
    if image_tensor.ndim == 2:
        image_tensor = image_tensor[np.newaxis, :, :]

    C, H, W = image_tensor.shape

    # Create subplot grid
    ncols = min(4, C)
    nrows = (C + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i in range(C):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        # Plot channel
        im = ax.imshow(image_tensor[i], cmap=cmap, aspect='auto')

        # Channel title
        if channel_names and i < len(channel_names):
            ax.set_title(f'Ch {i}: {channel_names[i]}', fontsize=10)
        else:
            ax.set_title(f'Channel {i}', fontsize=10)

        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xlabel('Width')
        ax.set_ylabel('Height')

    # Hide empty subplots
    for i in range(C, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved pixel grid to {save_path}")

    plt.show()


def plot_loss_curves(loss_history: Dict[str, List[float]],
                     save_path=None,
                     title='Training Loss Curves',
                     figsize=(12, 6)):
    """
    Plot training loss curves

    Args:
        loss_history: Dict mapping loss names to lists of values
        save_path: Path to save figure
        title: Figure title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Separate discriminator and generator losses
    d_losses = {k: v for k, v in loss_history.items() if 'd_' in k.lower() or 'disc' in k.lower()}
    g_losses = {k: v for k, v in loss_history.items() if 'g_' in k.lower() or 'gen' in k.lower()}

    # Plot discriminator losses
    ax = axes[0]
    for name, values in d_losses.items():
        ax.plot(values, label=name, linewidth=2, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Discriminator Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot generator losses
    ax = axes[1]
    for name, values in g_losses.items():
        ax.plot(values, label=name, linewidth=2, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Generator Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")

    plt.show()


def plot_metrics(metrics: Dict[str, float],
                save_path=None,
                title='Evaluation Metrics',
                figsize=(10, 6)):
    """
    Plot evaluation metrics as bar chart

    Args:
        metrics: Dictionary of metric names to values
        save_path: Path to save figure
        title: Figure title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = list(metrics.keys())
    values = list(metrics.values())

    # Create bar chart
    bars = ax.bar(range(len(names)), values, color='steelblue', alpha=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics plot to {save_path}")

    plt.show()


def plot_roc_pr_curves(labels: np.ndarray,
                       scores: np.ndarray,
                       save_path=None,
                       title='ROC and PR Curves',
                       figsize=(12, 5)):
    """
    Plot ROC and Precision-Recall curves

    Args:
        labels: Binary labels (0/1)
        scores: Anomaly scores
        save_path: Path to save figure
        title: Figure title
        figsize: Figure size
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    # Compute curves
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ROC curve
    ax = axes[0]
    ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PR curve
    ax = axes[1]
    ax.plot(recall, precision, linewidth=2, label=f'AUC = {pr_auc:.4f}')
    baseline = labels.sum() / len(labels)
    ax.axhline(baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline = {baseline:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC/PR curves to {save_path}")

    plt.show()


def plot_confusion_matrix(cm: np.ndarray,
                         class_names=['Normal', 'Anomaly'],
                         save_path=None,
                         title='Confusion Matrix',
                         figsize=(8, 6)):
    """
    Plot confusion matrix heatmap

    Args:
        cm: Confusion matrix (2x2)
        class_names: Class names
        save_path: Path to save figure
        title: Figure title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Percentage'})

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


def plot_anomaly_score_distribution(scores: np.ndarray,
                                    labels: np.ndarray,
                                    threshold: Optional[float] = None,
                                    save_path=None,
                                    title='Anomaly Score Distribution',
                                    figsize=(10, 6)):
    """
    Plot distribution of anomaly scores for normal vs anomalous samples

    Args:
        scores: Anomaly scores
        labels: Binary labels (0=normal, 1=anomaly)
        threshold: Decision threshold (optional)
        save_path: Path to save figure
        title: Figure title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Separate normal and anomaly scores
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    # Plot histograms
    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)

    # Plot threshold if provided
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')

    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved score distribution to {save_path}")

    plt.show()


def plot_comparison_grid(real_images: torch.Tensor,
                        fake_images: torch.Tensor,
                        num_samples=4,
                        channel_idx=0,
                        save_path=None,
                        title='Real vs Generated Comparison'):
    """
    Plot comparison grid of real vs generated images

    Args:
        real_images: Real images (B, C, H, W)
        fake_images: Generated images (B, C, H, W)
        num_samples: Number of samples to show
        channel_idx: Which channel to visualize
        save_path: Path to save figure
        title: Figure title
    """
    if isinstance(real_images, torch.Tensor):
        real_images = real_images.detach().cpu().numpy()
    if isinstance(fake_images, torch.Tensor):
        fake_images = fake_images.detach().cpu().numpy()

    num_samples = min(num_samples, real_images.shape[0], fake_images.shape[0])

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

    for i in range(num_samples):
        # Real
        axes[0, i].imshow(real_images[i, channel_idx], cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Real {i+1}')
        axes[0, i].axis('off')

        # Fake
        axes[1, i].imshow(fake_images[i, channel_idx], cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'Generated {i+1}')
        axes[1, i].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison grid to {save_path}")

    plt.show()

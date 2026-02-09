"""
PyTorch Dataset and DataLoader for PxGAN
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Callable, Tuple
import pandas as pd

from .pixel_mapper import PixelMapper


class PxGANDataset(Dataset):
    """
    PyTorch Dataset for pixelated telemetry data

    Converts time-windowed DataFrames to pixel grid images using PixelMapper
    """

    def __init__(self,
                 windows: List[pd.DataFrame],
                 mapper: PixelMapper,
                 cond_extractor: Optional[Callable] = None,
                 transform=None):
        """
        Args:
            windows: List of time-windowed DataFrames
            mapper: Fitted PixelMapper instance
            cond_extractor: Optional function to extract conditional features
                           from window DataFrame (returns numpy array)
            transform: Optional transforms to apply to images
        """
        self.windows = windows
        self.mapper = mapper
        self.cond_extractor = cond_extractor or self._default_cond_extractor
        self.transform = transform

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample

        Returns:
            dict with keys:
                - 'image': Tensor of shape (C, H, W)
                - 'cond': Conditional vector tensor
                - 'label': Anomaly label (0=normal, 1=anomaly)
                - 'window_idx': Index of the window
        """
        window_df = self.windows[idx]

        # Convert to pixel image
        image = self.mapper.transform(window_df)  # (C, H, W)

        # Extract conditional features
        cond = self.cond_extractor(window_df)

        # Get label (from window attrs)
        label = window_df.attrs.get('label', 0)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        cond_tensor = torch.from_numpy(cond).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Apply transforms if any
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return {
            'image': image_tensor,
            'cond': cond_tensor,
            'label': label_tensor,
            'window_idx': idx
        }

    def _default_cond_extractor(self, window_df: pd.DataFrame) -> np.ndarray:
        """
        Default conditional feature extractor

        Extracts simple metadata from window

        Returns:
            1D numpy array of conditional features
        """
        features = []

        # Time-based features
        if 'timestamp' in window_df.columns and len(window_df) > 0:
            if pd.api.types.is_datetime64_any_dtype(window_df['timestamp']):
                # Hour of day (normalized to [0, 1])
                hour = window_df['timestamp'].iloc[0].hour / 24.0
                features.append(hour)

                # Day of week (normalized to [0, 1])
                day = window_df['timestamp'].iloc[0].dayofweek / 7.0
                features.append(day)
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])

        # Host role (if available) - placeholder encoding
        # In practice, you'd encode this from a known set of roles
        if 'host_role' in window_df.columns and len(window_df) > 0:
            role_map = {'server': 0.0, 'client': 0.33, 'gateway': 0.66, 'other': 1.0}
            role = window_df['host_role'].mode()[0] if len(window_df) > 0 else 'other'
            features.append(role_map.get(role, 1.0))
        else:
            features.append(0.5)  # Default

        # Network segment (placeholder)
        features.append(0.0)

        # Pad to specified cond_dim if needed (will be handled in model)
        return np.array(features, dtype=np.float32)


class SequenceDataset(Dataset):
    """
    Dataset for sequence of windows (for Sequence Critic)

    Returns sequences of consecutive windows
    """

    def __init__(self,
                 windows: List[pd.DataFrame],
                 mapper: PixelMapper,
                 seq_len: int = 5,
                 cond_extractor: Optional[Callable] = None):
        """
        Args:
            windows: List of time windows
            mapper: Fitted PixelMapper
            seq_len: Number of consecutive windows in a sequence
            cond_extractor: Optional conditional feature extractor
        """
        self.windows = windows
        self.mapper = mapper
        self.seq_len = seq_len
        self.cond_extractor = cond_extractor or PxGANDataset([], mapper)._default_cond_extractor

        # Compute valid start indices (where seq_len consecutive windows exist)
        self.valid_indices = list(range(len(windows) - seq_len + 1))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sequence of windows

        Returns:
            dict with keys:
                - 'sequence': Tensor of shape (seq_len, C, H, W)
                - 'cond': Conditional features for the sequence
                - 'label': Sequence label (1 if any window is anomaly)
        """
        start_idx = self.valid_indices[idx]

        # Get sequence of windows
        seq_windows = self.windows[start_idx:start_idx + self.seq_len]

        # Convert each to pixel image
        seq_images = [self.mapper.transform(w) for w in seq_windows]
        seq_tensor = torch.from_numpy(np.stack(seq_images, axis=0)).float()  # (seq_len, C, H, W)

        # Conditional features from first window in sequence
        cond = self.cond_extractor(seq_windows[0])
        cond_tensor = torch.from_numpy(cond).float()

        # Label: 1 if any window in sequence is anomaly
        labels = [w.attrs.get('label', 0) for w in seq_windows]
        label = 1 if any(labels) else 0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'sequence': seq_tensor,
            'cond': cond_tensor,
            'label': label_tensor,
            'start_idx': start_idx
        }


def create_dataloaders(train_windows: List[pd.DataFrame],
                      val_windows: List[pd.DataFrame],
                      test_windows: List[pd.DataFrame],
                      mapper: PixelMapper,
                      batch_size: int = 64,
                      num_workers: int = 4,
                      cond_extractor: Optional[Callable] = None,
                      create_sequence_loaders: bool = False,
                      seq_len: int = 5) -> dict:
    """
    Create train/val/test DataLoaders

    Args:
        train_windows: Training windows
        val_windows: Validation windows
        test_windows: Test windows
        mapper: Fitted PixelMapper
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        cond_extractor: Optional conditional feature extractor
        create_sequence_loaders: Whether to create sequence DataLoaders
        seq_len: Sequence length for sequence loaders

    Returns:
        Dictionary with DataLoader objects
    """
    print("Creating DataLoaders...")

    # Create datasets
    train_dataset = PxGANDataset(train_windows, mapper, cond_extractor)
    val_dataset = PxGANDataset(val_windows, mapper, cond_extractor)
    test_dataset = PxGANDataset(test_windows, mapper, cond_extractor)

    # Create main dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Set True if using GPU
        drop_last=True  # For stable GAN training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Create sequence loaders if requested
    if create_sequence_loaders:
        print(f"Creating sequence DataLoaders (seq_len={seq_len})...")

        train_seq_dataset = SequenceDataset(train_windows, mapper, seq_len, cond_extractor)
        val_seq_dataset = SequenceDataset(val_windows, mapper, seq_len, cond_extractor)

        train_seq_loader = DataLoader(
            train_seq_dataset,
            batch_size=batch_size // 2,  # Smaller batch for sequences
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True
        )

        val_seq_loader = DataLoader(
            val_seq_dataset,
            batch_size=batch_size // 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )

        loaders['train_seq'] = train_seq_loader
        loaders['val_seq'] = val_seq_loader

        print(f"  Train seq batches: {len(train_seq_loader)}")
        print(f"  Val seq batches: {len(val_seq_loader)}")

    return loaders


def collate_fn_with_padding(batch: List[dict]) -> dict:
    """
    Custom collate function to handle variable-size conditional features

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary
    """
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    window_indices = [item['window_idx'] for item in batch]

    # Pad conditional features to max length in batch
    conds = [item['cond'] for item in batch]
    max_cond_len = max(len(c) for c in conds)

    padded_conds = []
    for c in conds:
        if len(c) < max_cond_len:
            padded = torch.cat([c, torch.zeros(max_cond_len - len(c))])
        else:
            padded = c
        padded_conds.append(padded)

    conds_tensor = torch.stack(padded_conds)

    return {
        'image': images,
        'cond': conds_tensor,
        'label': labels,
        'window_idx': window_indices
    }

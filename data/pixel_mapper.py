"""
Pixel Mapper: Deterministic telemetry → 2D image conversion
Core component for pixelation-guided GAN
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml


class PixelMapper:
    """
    Maps time-windowed telemetry data to fixed-size 2D image tensors

    Features are grouped into channels, aggregated with statistics,
    and tiled into H×W grids. Normalization is applied per channel.
    """

    def __init__(self,
                 feature_groups_config: str,
                 image_shape: Tuple[int, int] = (32, 32),
                 scaler_path: Optional[str] = None):
        """
        Args:
            feature_groups_config: Path to feature groups YAML config
            image_shape: (H, W) output image dimensions
            scaler_path: Path to pre-fitted scaler (None = fit new)
        """
        self.image_shape = image_shape
        self.H, self.W = image_shape

        # Load feature groups configuration
        with open(feature_groups_config, 'r') as f:
            config = yaml.safe_load(f)

        self.feature_groups_cfg = config['feature_groups']
        self.aggregation_functions = config['aggregation']['functions']
        self.normalization_cfg = config['normalization']
        self.fill_strategy = config['mapping']['fill_strategy']

        # Extract feature group names and features
        self.channel_names = list(self.feature_groups_cfg.keys())
        self.num_channels = len(self.channel_names)

        # Build feature list per channel
        self.features_per_channel = {}
        for ch_name in self.channel_names:
            self.features_per_channel[ch_name] = (
                self.feature_groups_cfg[ch_name]['features']
            )

        # Normalization statistics (fitted on training data)
        self.normalization_stats = None
        self.is_fitted = False

        # Load scaler if provided
        if scaler_path is not None:
            self.load_scaler(scaler_path)

        print(f"PixelMapper initialized:")
        print(f"  Image shape: {image_shape}")
        print(f"  Channels: {self.num_channels}")
        print(f"  Aggregations: {self.aggregation_functions}")

    def fit(self, train_windows: List[pd.DataFrame]):
        """
        Fit normalization statistics on training data

        Args:
            train_windows: List of time-windowed DataFrames
        """
        print("Fitting PixelMapper normalization statistics...")

        # Collect all channel values for statistics
        all_channel_values = {ch: [] for ch in self.channel_names}

        for window_df in train_windows:
            for ch_name in self.channel_names:
                channel_vec = self._aggregate_channel(window_df, ch_name)
                all_channel_values[ch_name].append(channel_vec)

        # Compute normalization stats per channel
        self.normalization_stats = {}

        for ch_name in self.channel_names:
            # Stack all vectors for this channel
            values = np.vstack(all_channel_values[ch_name])

            method = self.normalization_cfg['method']

            if method == 'zscore':
                mean = np.mean(values, axis=0)
                std = np.std(values, axis=0) + 1e-8  # Avoid division by zero
                self.normalization_stats[ch_name] = {'mean': mean, 'std': std, 'method': 'zscore'}

            elif method == 'minmax':
                min_val = np.min(values, axis=0)
                max_val = np.max(values, axis=0)
                range_val = max_val - min_val + 1e-8
                self.normalization_stats[ch_name] = {'min': min_val, 'range': range_val, 'method': 'minmax'}

            elif method == 'robust':
                # Robust scaling using median and IQR
                median = np.median(values, axis=0)
                q25 = np.percentile(values, 25, axis=0)
                q75 = np.percentile(values, 75, axis=0)
                iqr = q75 - q25 + 1e-8
                self.normalization_stats[ch_name] = {'median': median, 'iqr': iqr, 'method': 'robust'}

        self.is_fitted = True
        print(f"PixelMapper fitted on {len(train_windows)} windows")

    def transform(self, window_df: pd.DataFrame) -> np.ndarray:
        """
        Transform a time window to pixel image

        Args:
            window_df: DataFrame with telemetry events for one time window

        Returns:
            numpy array of shape (C, H, W)
        """
        if not self.is_fitted:
            raise ValueError("PixelMapper must be fitted before transform. Call .fit() first.")

        # Initialize output image
        image = np.zeros((self.num_channels, self.H, self.W), dtype=np.float32)

        # Process each channel
        for ch_idx, ch_name in enumerate(self.channel_names):
            # Aggregate features for this channel
            channel_vec = self._aggregate_channel(window_df, ch_name)

            # Normalize
            channel_vec = self._normalize_channel(channel_vec, ch_name)

            # Fill into H×W grid
            channel_grid = self._fill_grid(channel_vec)

            image[ch_idx] = channel_grid

        return image

    def fit_transform(self, train_windows: List[pd.DataFrame]) -> List[np.ndarray]:
        """
        Fit on training data and transform

        Args:
            train_windows: List of training time windows

        Returns:
            List of transformed images
        """
        self.fit(train_windows)
        return [self.transform(w) for w in train_windows]

    def _aggregate_channel(self, window_df: pd.DataFrame, channel_name: str) -> np.ndarray:
        """
        Aggregate features for a single channel using configured aggregation functions

        Args:
            window_df: Time window DataFrame
            channel_name: Name of the channel

        Returns:
            1D numpy array of aggregated values
        """
        features = self.features_per_channel[channel_name]
        agg_values = []

        for feat in features:
            # Get feature column (may not exist in all windows)
            if feat in window_df.columns:
                col = window_df[feat]

                # Apply each aggregation function
                for agg_func in self.aggregation_functions:
                    if agg_func == 'mean':
                        val = col.mean() if len(col) > 0 else 0.0
                    elif agg_func == 'std':
                        val = col.std() if len(col) > 0 else 0.0
                    elif agg_func == 'min':
                        val = col.min() if len(col) > 0 else 0.0
                    elif agg_func == 'max':
                        val = col.max() if len(col) > 0 else 0.0
                    elif agg_func == 'percentile_25':
                        val = col.quantile(0.25) if len(col) > 0 else 0.0
                    elif agg_func == 'percentile_75':
                        val = col.quantile(0.75) if len(col) > 0 else 0.0
                    else:
                        val = 0.0

                    # Handle NaN/inf
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0

                    agg_values.append(val)
            else:
                # Feature not present: fill with zeros for all aggregations
                agg_values.extend([0.0] * len(self.aggregation_functions))

        return np.array(agg_values, dtype=np.float32)

    def _normalize_channel(self, channel_vec: np.ndarray, channel_name: str) -> np.ndarray:
        """
        Normalize channel vector using fitted statistics

        Args:
            channel_vec: Raw aggregated channel vector
            channel_name: Channel name

        Returns:
            Normalized vector
        """
        stats = self.normalization_stats[channel_name]
        method = stats['method']

        if method == 'zscore':
            normalized = (channel_vec - stats['mean']) / stats['std']

            # Clip outliers if configured
            if self.normalization_cfg.get('clip_outliers', True):
                clip_thresh = self.normalization_cfg.get('clip_std_threshold', 3.0)
                normalized = np.clip(normalized, -clip_thresh, clip_thresh)

        elif method == 'minmax':
            normalized = (channel_vec - stats['min']) / stats['range']

        elif method == 'robust':
            normalized = (channel_vec - stats['median']) / stats['iqr']

            if self.normalization_cfg.get('clip_outliers', True):
                clip_thresh = self.normalization_cfg.get('clip_std_threshold', 3.0)
                normalized = np.clip(normalized, -clip_thresh, clip_thresh)

        else:
            normalized = channel_vec

        return normalized.astype(np.float32)

    def _fill_grid(self, vector: np.ndarray) -> np.ndarray:
        """
        Fill 1D vector into H×W grid using configured strategy

        Args:
            vector: 1D aggregated and normalized vector

        Returns:
            2D grid (H, W)
        """
        target_size = self.H * self.W

        if self.fill_strategy == 'tile':
            # Repeat/tile vector to fill grid
            if len(vector) >= target_size:
                # Truncate
                filled = vector[:target_size]
            else:
                # Tile (repeat)
                repeats = target_size // len(vector) + 1
                tiled = np.tile(vector, repeats)
                filled = tiled[:target_size]

        elif self.fill_strategy == 'pad':
            # Zero-pad to target size
            if len(vector) >= target_size:
                filled = vector[:target_size]
            else:
                filled = np.pad(vector, (0, target_size - len(vector)), 'constant')

        elif self.fill_strategy == 'resize':
            # Use interpolation to resize (requires scipy or image library)
            # For simplicity, using tile here as well
            if len(vector) >= target_size:
                filled = vector[:target_size]
            else:
                repeats = target_size // len(vector) + 1
                tiled = np.tile(vector, repeats)
                filled = tiled[:target_size]

        else:
            # Default to tile
            if len(vector) >= target_size:
                filled = vector[:target_size]
            else:
                repeats = target_size // len(vector) + 1
                tiled = np.tile(vector, repeats)
                filled = tiled[:target_size]

        # Reshape to grid
        grid = filled.reshape(self.H, self.W)
        return grid.astype(np.float32)

    def save_scaler(self, path: str):
        """
        Save fitted normalization statistics

        Args:
            path: File path to save scaler
        """
        if not self.is_fitted:
            raise ValueError("PixelMapper not fitted. Cannot save scaler.")

        scaler_data = {
            'normalization_stats': self.normalization_stats,
            'image_shape': self.image_shape,
            'channel_names': self.channel_names,
            'num_channels': self.num_channels,
            'features_per_channel': self.features_per_channel,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(scaler_data, f)

        print(f"PixelMapper scaler saved to {path}")

    def load_scaler(self, path: str):
        """
        Load pre-fitted normalization statistics

        Args:
            path: File path to load scaler from
        """
        with open(path, 'rb') as f:
            scaler_data = pickle.load(f)

        self.normalization_stats = scaler_data['normalization_stats']
        self.image_shape = scaler_data['image_shape']
        self.H, self.W = self.image_shape
        self.channel_names = scaler_data['channel_names']
        self.num_channels = scaler_data['num_channels']
        self.features_per_channel = scaler_data['features_per_channel']
        self.is_fitted = True

        print(f"PixelMapper scaler loaded from {path}")

    def get_channel_names(self) -> List[str]:
        """Get list of channel names"""
        return self.channel_names

    def get_num_channels(self) -> int:
        """Get number of channels"""
        return self.num_channels

    def get_image_shape(self) -> Tuple[int, int]:
        """Get output image shape"""
        return self.image_shape

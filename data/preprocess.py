"""
Data preprocessing pipeline
Handles feature engineering, train/val/test splits, and pixel mapper fitting
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split

from .pixel_mapper import PixelMapper
from .loaders import create_time_windows


class DataPreprocessor:
    """
    Preprocessing pipeline for PxGAN telemetry data

    Handles:
    1. Feature engineering (derived features, encodings)
    2. Temporal train/val/test splitting
    3. Pixel mapper fitting on training data only
    4. Saving processed windows and scaler
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']

        # Initialize pixel mapper
        self.pixel_mapper = PixelMapper(
            feature_groups_config=self.data_config['feature_groups_config'],
            image_shape=tuple(self.data_config['image_shape'])
        )

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived/engineered features to telemetry data

        Args:
            df: Raw telemetry DataFrame

        Returns:
            DataFrame with additional features
        """
        df = df.copy()

        # Bytes per second (if flow data)
        if 'bytes_sent' in df.columns and 'flow_duration' in df.columns:
            df['bytes_per_second'] = df['bytes_sent'] / (df['flow_duration'] + 1e-6)

        if 'bytes_received' in df.columns and 'flow_duration' in df.columns:
            df['bytes_received_per_second'] = df['bytes_received'] / (df['flow_duration'] + 1e-6)

        # Packets per second
        if 'packets_sent' in df.columns and 'flow_duration' in df.columns:
            df['packets_per_second'] = df['packets_sent'] / (df['flow_duration'] + 1e-6)

        # Flow symmetry (bidirectional traffic ratio)
        if 'bytes_sent' in df.columns and 'bytes_received' in df.columns:
            total = df['bytes_sent'] + df['bytes_received'] + 1e-6
            df['flow_symmetry'] = np.minimum(df['bytes_sent'], df['bytes_received']) / total

        # Port encoding (categorize common port ranges)
        if 'dst_port' in df.columns:
            df['dst_port_encoded'] = df['dst_port'].apply(self._encode_port)

        if 'src_port' in df.columns:
            df['src_port_encoded'] = df['src_port'].apply(self._encode_port)

        # Protocol encoding
        if 'protocol' in df.columns:
            protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2, 'OTHER': 3}
            df['protocol_type'] = df['protocol'].map(protocol_map).fillna(3)

        # Time-based features (if timestamp present)
        if 'timestamp' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['hour_of_day'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        return df

    def _encode_port(self, port: int) -> int:
        """Encode port into categorical ranges"""
        if port < 0:
            return 0
        elif port < 1024:
            return 1  # Well-known ports
        elif port < 49152:
            return 2  # Registered ports
        else:
            return 3  # Dynamic/private ports

    def temporal_split(self,
                      windows: List[pd.DataFrame],
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> Tuple[List, List, List]:
        """
        Split windows temporally (not randomly) to preserve time ordering

        Args:
            windows: List of time windows
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing

        Returns:
            (train_windows, val_windows, test_windows)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        n = len(windows)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_windows = windows[:train_end]
        val_windows = windows[train_end:val_end]
        test_windows = windows[val_end:]

        print(f"Temporal split:")
        print(f"  Train: {len(train_windows)} windows")
        print(f"  Val: {len(val_windows)} windows")
        print(f"  Test: {len(test_windows)} windows")

        return train_windows, val_windows, test_windows

    def process_and_split(self,
                         raw_df: pd.DataFrame,
                         time_col: str = 'timestamp',
                         label_col: Optional[str] = None) -> Tuple[List, List, List]:
        """
        Complete preprocessing pipeline

        Args:
            raw_df: Raw telemetry DataFrame
            time_col: Timestamp column name
            label_col: Optional label column for anomalies

        Returns:
            (train_windows, val_windows, test_windows)
        """
        print("Starting preprocessing pipeline...")

        # 1. Feature engineering
        print("Step 1: Feature engineering...")
        df = self.add_derived_features(raw_df)

        # 2. Create time windows
        print("Step 2: Creating time windows...")
        windows = create_time_windows(
            df,
            time_col=time_col,
            window_size=self.data_config['window_size'],
            stride=self.data_config['stride'],
            label_col=label_col
        )

        # 3. Temporal split
        print("Step 3: Temporal splitting...")
        train_windows, val_windows, test_windows = self.temporal_split(
            windows,
            train_ratio=self.data_config.get('train_ratio', 0.7),
            val_ratio=self.data_config.get('val_ratio', 0.15),
            test_ratio=self.data_config.get('test_ratio', 0.15)
        )

        # 4. Fit pixel mapper on training data only
        print("Step 4: Fitting pixel mapper...")
        self.pixel_mapper.fit(train_windows)

        print("Preprocessing complete!")

        return train_windows, val_windows, test_windows

    def save_processed_data(self,
                           output_dir: str,
                           train_windows: List,
                           val_windows: List,
                           test_windows: List):
        """
        Save processed windows and fitted pixel mapper

        Args:
            output_dir: Output directory
            train_windows: Training windows
            val_windows: Validation windows
            test_windows: Test windows
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving processed data to {output_dir}...")

        # Save windows
        with open(output_path / 'train_windows.pkl', 'wb') as f:
            pickle.dump(train_windows, f)

        with open(output_path / 'val_windows.pkl', 'wb') as f:
            pickle.dump(val_windows, f)

        with open(output_path / 'test_windows.pkl', 'wb') as f:
            pickle.dump(test_windows, f)

        # Save pixel mapper scaler
        self.pixel_mapper.save_scaler(str(output_path / 'scaler.pkl'))

        # Save feature stats for analysis
        stats = {
            'num_train_windows': len(train_windows),
            'num_val_windows': len(val_windows),
            'num_test_windows': len(test_windows),
            'window_size': self.data_config['window_size'],
            'stride': self.data_config['stride'],
            'image_shape': self.data_config['image_shape'],
            'num_channels': self.pixel_mapper.get_num_channels(),
            'channel_names': self.pixel_mapper.get_channel_names(),
        }

        import json
        with open(output_path / 'feature_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Saved processed data:")
        print(f"  {output_path / 'train_windows.pkl'}")
        print(f"  {output_path / 'val_windows.pkl'}")
        print(f"  {output_path / 'test_windows.pkl'}")
        print(f"  {output_path / 'scaler.pkl'}")
        print(f"  {output_path / 'feature_stats.json'}")

    def load_processed_data(self, data_dir: str) -> Tuple[List, List, List, PixelMapper]:
        """
        Load preprocessed windows and pixel mapper

        Args:
            data_dir: Directory containing processed data

        Returns:
            (train_windows, val_windows, test_windows, pixel_mapper)
        """
        data_path = Path(data_dir)

        print(f"Loading processed data from {data_dir}...")

        with open(data_path / 'train_windows.pkl', 'rb') as f:
            train_windows = pickle.load(f)

        with open(data_path / 'val_windows.pkl', 'rb') as f:
            val_windows = pickle.load(f)

        with open(data_path / 'test_windows.pkl', 'rb') as f:
            test_windows = pickle.load(f)

        # Load pixel mapper
        self.pixel_mapper.load_scaler(str(data_path / 'scaler.pkl'))

        print(f"Loaded {len(train_windows)} train, {len(val_windows)} val, {len(test_windows)} test windows")

        return train_windows, val_windows, test_windows, self.pixel_mapper

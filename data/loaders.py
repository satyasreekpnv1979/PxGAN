"""
Data loaders for multi-format telemetry data
Supports CSV flows, JSON logs, and Parquet metrics
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime, timedelta


def load_csv_flows(path: Union[str, Path],
                   time_col: str = 'timestamp',
                   sort: bool = True,
                   parse_dates: bool = True) -> pd.DataFrame:
    """
    Load network flow data from CSV

    Args:
        path: Path to CSV file
        time_col: Name of timestamp column
        sort: Sort by timestamp
        parse_dates: Parse timestamp column as datetime

    Returns:
        DataFrame with flow data
    """
    print(f"Loading CSV flows from {path}...")

    # Try different encodings if needed
    try:
        df = pd.read_csv(path, parse_dates=[time_col] if parse_dates else None)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1', parse_dates=[time_col] if parse_dates else None)

    # Sort by time if requested
    if sort and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)

    print(f"  Loaded {len(df)} flow records")
    print(f"  Columns: {list(df.columns)}")

    return df


def load_json_logs(path: Union[str, Path],
                   flatten: bool = True,
                   time_col: str = 'timestamp',
                   sort: bool = True) -> pd.DataFrame:
    """
    Load semi-structured log data from JSON/JSONL

    Args:
        path: Path to JSON or JSONL file
        flatten: Flatten nested JSON structures
        time_col: Name of timestamp field
        sort: Sort by timestamp

    Returns:
        DataFrame with log data
    """
    print(f"Loading JSON logs from {path}...")

    path = Path(path)

    # Detect JSONL vs JSON
    with open(path, 'r') as f:
        first_line = f.readline()
        f.seek(0)

        try:
            json.loads(first_line)
            is_jsonl = True
        except:
            is_jsonl = False

    if is_jsonl:
        # Line-delimited JSON
        records = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        df = pd.DataFrame(records)
    else:
        # Single JSON object or array
        with open(path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try to find the main data array
            if 'logs' in data:
                df = pd.DataFrame(data['logs'])
            elif 'events' in data:
                df = pd.DataFrame(data['events'])
            elif 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                # Assume dict is a single record
                df = pd.DataFrame([data])

    # Flatten nested structures if requested
    if flatten and len(df) > 0:
        df = pd.json_normalize(df.to_dict('records'))

    # Parse timestamp
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        # Sort by time if requested
        if sort:
            df = df.sort_values(time_col).reset_index(drop=True)

    print(f"  Loaded {len(df)} log records")
    print(f"  Columns: {list(df.columns)}")

    return df


def load_parquet_metrics(path: Union[str, Path],
                        time_col: str = 'timestamp',
                        sort: bool = True) -> pd.DataFrame:
    """
    Load telemetry metrics from Parquet format

    Args:
        path: Path to Parquet file
        time_col: Name of timestamp column
        sort: Sort by timestamp

    Returns:
        DataFrame with metrics data
    """
    print(f"Loading Parquet metrics from {path}...")

    df = pd.read_parquet(path)

    # Parse timestamp if present
    if time_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        # Sort by time if requested
        if sort:
            df = df.sort_values(time_col).reset_index(drop=True)

    print(f"  Loaded {len(df)} metric records")
    print(f"  Columns: {list(df.columns)}")

    return df


def create_time_windows(df: pd.DataFrame,
                       time_col: str = 'timestamp',
                       window_size: int = 30,
                       stride: int = 10,
                       label_col: Optional[str] = None) -> List[pd.DataFrame]:
    """
    Create sliding time windows from telemetry DataFrame

    Args:
        df: Input DataFrame with timestamp column
        time_col: Name of timestamp column
        window_size: Window duration in seconds
        stride: Stride/step size in seconds
        label_col: Optional label column (for anomalies)

    Returns:
        List of DataFrames, one per window
    """
    print(f"Creating time windows (size={window_size}s, stride={stride}s)...")

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # Drop rows with invalid timestamps
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    if len(df) == 0:
        print("  Warning: No valid timestamps found")
        return []

    # Get time range
    start_time = df[time_col].min()
    end_time = df[time_col].max()

    window_duration = timedelta(seconds=window_size)
    stride_duration = timedelta(seconds=stride)

    windows = []
    current_start = start_time

    while current_start + window_duration <= end_time:
        current_end = current_start + window_duration

        # Extract window
        window_df = df[
            (df[time_col] >= current_start) &
            (df[time_col] < current_end)
        ].copy()

        # Add metadata as attrs
        window_df.attrs['window_start'] = current_start
        window_df.attrs['window_end'] = current_end
        window_df.attrs['window_size'] = window_size

        # Determine label (majority vote if label_col provided)
        if label_col is not None and label_col in df.columns:
            if len(window_df) > 0 and label_col in window_df.columns:
                # Majority vote: 1 if >50% of events are anomalies
                label = 1 if window_df[label_col].mean() > 0.5 else 0
            else:
                label = 0
            window_df.attrs['label'] = label
        else:
            window_df.attrs['label'] = 0  # Default: normal

        windows.append(window_df)

        # Move to next window
        current_start += stride_duration

    print(f"  Created {len(windows)} windows")

    # Print summary
    if label_col is not None:
        num_anomalies = sum(1 for w in windows if w.attrs.get('label', 0) == 1)
        num_normal = len(windows) - num_anomalies
        print(f"  Normal windows: {num_normal}, Anomaly windows: {num_anomalies}")

    return windows


def merge_multi_source_data(flow_df: Optional[pd.DataFrame] = None,
                            log_df: Optional[pd.DataFrame] = None,
                            metrics_df: Optional[pd.DataFrame] = None,
                            time_col: str = 'timestamp',
                            merge_tolerance: int = 1) -> pd.DataFrame:
    """
    Merge data from multiple sources (flows, logs, metrics) based on timestamp

    Args:
        flow_df: Flow data DataFrame
        log_df: Log data DataFrame
        metrics_df: Metrics data DataFrame
        time_col: Timestamp column name
        merge_tolerance: Time tolerance for merging (seconds)

    Returns:
        Merged DataFrame
    """
    print("Merging multi-source data...")

    dfs_to_merge = []

    if flow_df is not None:
        flow_df = flow_df.copy()
        flow_df.columns = ['flow_' + col if col != time_col else col for col in flow_df.columns]
        dfs_to_merge.append(flow_df)

    if log_df is not None:
        log_df = log_df.copy()
        log_df.columns = ['log_' + col if col != time_col else col for col in log_df.columns]
        dfs_to_merge.append(log_df)

    if metrics_df is not None:
        metrics_df = metrics_df.copy()
        metrics_df.columns = ['metric_' + col if col != time_col else col for col in metrics_df.columns]
        dfs_to_merge.append(metrics_df)

    if len(dfs_to_merge) == 0:
        raise ValueError("No data sources provided")

    if len(dfs_to_merge) == 1:
        return dfs_to_merge[0]

    # Merge using timestamp with tolerance
    merged = dfs_to_merge[0]

    for df in dfs_to_merge[1:]:
        # Use merge_asof for time-based merging with tolerance
        merged = pd.merge_asof(
            merged.sort_values(time_col),
            df.sort_values(time_col),
            on=time_col,
            direction='nearest',
            tolerance=pd.Timedelta(seconds=merge_tolerance)
        )

    print(f"  Merged data: {len(merged)} records, {len(merged.columns)} columns")

    return merged

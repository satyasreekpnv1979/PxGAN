"""
Data preprocessing script for PxGAN

Usage:
    python scripts/preprocess_data.py \
        --flow_csv ./raw_data/flows.csv \
        --config config/default.yaml \
        --output_dir ./processed_data
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from data.loaders import load_csv_flows, load_json_logs, load_parquet_metrics, merge_multi_source_data
from data.preprocess import DataPreprocessor
from utils import setup_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess telemetry data for PxGAN')

    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--flow_csv', type=str, default=None,
                       help='Path to flow CSV file')
    parser.add_argument('--log_json', type=str, default=None,
                       help='Path to log JSON/JSONL file')
    parser.add_argument('--metrics_parquet', type=str, default=None,
                       help='Path to metrics Parquet file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--time_col', type=str, default='timestamp',
                       help='Timestamp column name')
    parser.add_argument('--label_col', type=str, default=None,
                       help='Label column name (for supervised learning)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger('preprocess', log_dir=str(output_dir))
    logger.info("Starting data preprocessing")

    # Set seed for reproducibility
    set_seed(config['reproducibility'].get('seed', 42))

    # Load data from multiple sources
    logger.info("Loading data...")

    flow_df = None
    log_df = None
    metrics_df = None

    if args.flow_csv:
        logger.info(f"Loading flow data from {args.flow_csv}")
        flow_df = load_csv_flows(args.flow_csv, time_col=args.time_col)

    if args.log_json:
        logger.info(f"Loading log data from {args.log_json}")
        log_df = load_json_logs(args.log_json, time_col=args.time_col)

    if args.metrics_parquet:
        logger.info(f"Loading metrics data from {args.metrics_parquet}")
        metrics_df = load_parquet_metrics(args.metrics_parquet, time_col=args.time_col)

    # Merge data sources if multiple provided
    if sum([flow_df is not None, log_df is not None, metrics_df is not None]) > 1:
        logger.info("Merging multiple data sources...")
        combined_df = merge_multi_source_data(
            flow_df=flow_df,
            log_df=log_df,
            metrics_df=metrics_df,
            time_col=args.time_col,
            merge_tolerance=config['data'].get('merge_tolerance', 1)
        )
    else:
        # Use single datasource
        combined_df = flow_df or log_df or metrics_df

    if combined_df is None:
        logger.error("No data sources provided! Specify at least one of: --flow_csv, --log_json, --metrics_parquet")
        sys.exit(1)

    logger.info(f"Combined data: {len(combined_df)} records, {len(combined_df.columns)} columns")

    # Initialize preprocessor
    logger.info("Initializing preprocessor...")
    preprocessor = DataPreprocessor(config)

    # Process and split data
    logger.info("Processing and splitting data...")
    train_windows, val_windows, test_windows = preprocessor.process_and_split(
        raw_df=combined_df,
        time_col=args.time_col,
        label_col=args.label_col
    )

    # Save processed data
    logger.info(f"Saving processed data to {output_dir}...")
    preprocessor.save_processed_data(
        output_dir=str(output_dir),
        train_windows=train_windows,
        val_windows=val_windows,
        test_windows=test_windows
    )

    logger.info("Preprocessing complete!")
    logger.info(f"Processed data saved to {output_dir}")
    logger.info("You can now train with:")
    logger.info(f"  python scripts/train.py --config {args.config} --data_dir {output_dir}")


if __name__ == '__main__':
    main()

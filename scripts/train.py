"""
Main training script for PxGAN

Usage:
    python scripts/train.py --config config/default.yaml --data_dir ./processed_data
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data import PxGANDataset, create_dataloaders
from data.preprocess import DataPreprocessor
from models import ConditionalGenerator, PatchDiscriminator, SequenceCritic
from training import PxGANTrainer
from utils import (
    set_seed,
    set_deterministic,
    get_device,
    setup_logger,
    ExperimentTracker
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train PxGAN')

    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with processed data (windows, scaler)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for experiments (default: from config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cpu, cuda, or auto (default: auto)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command-line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir

    # Setup reproducibility
    seed = config['reproducibility'].get('seed', 42)
    deterministic = config['reproducibility'].get('deterministic', True)
    num_threads = config['reproducibility'].get('num_threads', None)

    set_seed(seed)
    set_deterministic(enabled=deterministic, num_threads=num_threads)

    # Get device
    device_str = args.device or config.get('device', 'auto')
    device = get_device(device_str)

    # Setup logging
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger('pxgan', log_dir=str(output_dir / 'logs'))
    logger.info(f"Starting PxGAN training")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {output_dir}")

    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    preprocessor = DataPreprocessor(config)
    train_windows, val_windows, test_windows, pixel_mapper = preprocessor.load_processed_data(args.data_dir)

    logger.info(f"Loaded {len(train_windows)} train, {len(val_windows)} val, {len(test_windows)} test windows")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    loaders = create_dataloaders(
        train_windows=train_windows,
        val_windows=val_windows,
        test_windows=test_windows,
        mapper=pixel_mapper,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        create_sequence_loaders=config['model'].get('discriminator_seq', {}) is not None
    )

    # Initialize models
    logger.info("Initializing models...")

    model_config = config['model']

    generator = ConditionalGenerator(
        z_dim=model_config['generator']['z_dim'],
        cond_dim=model_config['generator']['cond_dim'],
        out_channels=model_config['generator']['out_channels'],
        img_size=model_config['generator']['img_size'],
        use_spectral_norm=model_config['generator'].get('use_spectral_norm', False)
    )

    discriminator_patch = PatchDiscriminator(
        in_channels=model_config['discriminator_patch']['in_channels'],
        ndf=model_config['discriminator_patch']['ndf'],
        n_layers=model_config['discriminator_patch'].get('n_layers', 3),
        use_spectral_norm=model_config['discriminator_patch'].get('use_spectral_norm', True)
    )

    # Optional sequence critic
    discriminator_seq = None
    if 'discriminator_seq' in model_config and model_config['discriminator_seq']:
        discriminator_seq = SequenceCritic(
            in_channels=model_config['discriminator_seq']['in_channels'],
            seq_len=model_config['discriminator_seq']['seq_len'],
            cnn_embed_dim=model_config['discriminator_seq']['cnn_embed_dim'],
            lstm_hidden=model_config['discriminator_seq']['lstm_hidden'],
            lstm_layers=model_config['discriminator_seq']['lstm_layers'],
            dropout=model_config['discriminator_seq'].get('dropout', 0.3)
        )

    # Experiment tracker
    tracker_config = config.get('tracking', {})
    if tracker_config.get('enabled', True):
        experiment_tracker = ExperimentTracker(
            exp_dir=str(output_dir),
            config=config,
            enabled=True,
            backend=tracker_config.get('backend', 'tensorboard')
        )
    else:
        experiment_tracker = None

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = PxGANTrainer(
        config=config,
        generator=generator,
        discriminator_patch=discriminator_patch,
        discriminator_seq=discriminator_seq,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        device=device,
        experiment_tracker=experiment_tracker
    )

    # Train
    num_epochs = config['training']['epochs']
    logger.info(f"Starting training for {num_epochs} epochs...")

    try:
        trainer.train(num_epochs=num_epochs, resume_from=args.resume)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Save final model
        final_model_path = output_dir / 'final_model.pt'
        trainer.save_final_model(str(final_model_path))
        logger.info(f"Saved final model to {final_model_path}")

        # Close tracker
        if experiment_tracker:
            experiment_tracker.close()

    logger.info("Training complete!")


if __name__ == '__main__':
    main()

# PxGAN: Pixelation-Guided GAN for Telemetry Anomaly Detection

Complete implementation of Pixelation-Guided GAN for multivariate telemetry anomaly detection with adversarial hardening.

## Features

- **Multi-format data loading**: CSV flows, JSON logs, Parquet metrics
- **Flexible pixel mapping**: Deterministic telemetry → 2D image conversion with configurable feature groups
- **Dual discriminator architecture**:
  - PatchGAN for spatial quality
  - Sequence Critic for temporal coherence
- **Full training pipeline**: Multi-discriminator GAN training with feature matching and gradient penalty
- **Anomaly scoring**: Combined reconstruction + discriminator + statistical scoring
- **CPU-optimized**: Efficient multi-core training (no GPU required)
- **Reproducible**: Deterministic operations, checkpointing, experiment tracking

## Architecture Overview

```
Telemetry Data → Pixel Mapper → (C, H, W) Images
                     ↓
            Generator (G) ← noise (z) + conditions
                     ↓
              Generated Images
                     ↓
         ┌───────────┴──────────┐
          ↓                     ↓
  PatchGAN (spatial)    SeqCritic (temporal)
  ```

## Installation

```bash
# Clone/navigate to project
cd PxGAN

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, pandas, yaml; print('✓ Dependencies installed')"
```

## Quick Start

### 1. Preprocess Data

Prepare your telemetry data (CSV, JSON, or Parquet format):

```bash
python scripts/preprocess_data.py \
    --flow_csv ./raw_data/flows.csv \
    --config config/default.yaml \
    --output_dir ./processed_data \
    --time_col timestamp \
    --label_col label  # optional, for labeled anomalies
```

**Multi-source data** (combine flows, logs, metrics):
```bash
python scripts/preprocess_data.py \
    --flow_csv ./raw_data/flows.csv \
    --log_json ./raw_data/logs.jsonl \
    --metrics_parquet ./raw_data/metrics.parquet \
    --config config/default.yaml \
    --output_dir ./processed_data
```

### 2. Train PxGAN

```bash
python scripts/train.py \
    --config config/default.yaml \
    --data_dir ./processed_data \
    --output_dir ./experiments/run_001 \
    --epochs 100 \
    --batch_size 64
```

**Resume training**:
```bash
python scripts/train.py \
    --config config/default.yaml \
    --data_dir ./processed_data \
    --output_dir ./experiments/run_001 \
    --resume ./experiments/run_001/checkpoints/best_model.pt
```

### 3. Monitor Training

TensorBoard logging is enabled by default:

```bash
tensorboard --logdir ./experiments/run_001/logs
# Open http://localhost:6006
```

## Configuration

All hyperparameters are in `config/default.yaml`:

### Key Settings

**Data**:
- `window_size`: Time window duration (seconds)
- `stride`: Sliding window stride (seconds)
- `image_shape`: [H, W] pixel grid dimensions

**Models**:
- `generator.z_dim`: Latent noise dimension
- `discriminator_patch.ndf`: Base discriminator filters
- `discriminator_seq.seq_len`: Sequence length for temporal critic

**Training**:
- `epochs`: Number of training epochs
- `lr`: Learning rate (default: 0.0002)
- `loss_weights`: Weights for different loss components

**Feature Groups** (`config/feature_groups.yaml`):
Define how telemetry features map to image channels. Customize for your data:

```yaml
feature_groups:
  flow_stats:
    features: [bytes_sent, bytes_received, packets_sent, ...]
  timing:
    features: [iat_mean, iat_std, burst_count, ...]
  # Add your custom feature groups
```

## Project Structure

```
pxgan/
├── config/                   # Configuration files
│   ├── default.yaml         # Main config
│   └── feature_groups.yaml  # Feature→channel mapping
├── data/                    # Data pipeline
│   ├── pixel_mapper.py     # Telemetry → image conversion
│   ├── loaders.py          # Multi-format data loading
│   ├── preprocess.py       # Preprocessing pipeline
│   └── dataset.py          # PyTorch datasets
├── models/                  # Model architectures
│   ├── generator.py
│   ├── discriminator_patch.py
│   ├── discriminator_seq.py
│   └── losses.py
├── training/                # Training system
│   ├── trainer.py
│   ├── checkpointing.py
│   └── adversarial.py
├── evaluation/              # Evaluation & metrics
├── utils/                   # Utilities
│   ├── reproducibility.py
│   ├── logging.py
│   └── visualization.py
└── scripts/                 # Execution scripts
    ├── preprocess_data.py
    ├── train.py
    └── eval.py
```

## Data Format

Your telemetry data should include at minimum:
- **Timestamp column**: For temporal windowing
- **Feature columns**: Numeric telemetry features (flow stats, timing, protocol info, etc.)
- **Label column** (optional): Binary labels (0 = normal, 1 = anomaly)

Example CSV format:
```csv
timestamp,bytes_sent,bytes_received,packets_sent,duration,label
2024-01-01 00:00:00,1024,2048,10,5.2,0
2024-01-01 00:00:05,2048,4096,20,3.1,0
...
```

## CPU Optimization

For your 32-core Xeon server, the code is pre-configured for CPU training:

- `num_threads: 32` in `config/default.yaml`
- `num_workers: 8` for DataLoader parallelism
- Batch processing optimized for CPU

**Tip**: Monitor CPU usage with `htop` during training.

## Advanced Usage

### Custom Feature Engineering

Edit `data/preprocess.py` → `add_derived_features()` to add domain-specific features:

```python
def add_derived_features(self, df):
    df = df.copy()

    # Add your custom features
    df['my_custom_ratio'] = df['feature_a'] / (df['feature_b'] + 1e-6)

    return df
```

### Adversarial Hardening

Enable adversarial training in config:

```yaml
training:
  adversarial_training:
    enabled: true
    start_epoch: 10
    frequency: 5
    epsilon: 0.01
```

### Visualization

Visualize pixel grids:

```python
from utils.visualization import plot_pixel_grid
from data import PixelMapper

mapper = PixelMapper('config/feature_groups.yaml')
# ... fit mapper ...

image = mapper.transform(window_df)
plot_pixel_grid(image, channel_names=mapper.get_channel_names(),
                save_path='pixel_grid.png')
```

## Troubleshooting

**Out of Memory**:
- Reduce `batch_size` in config
- Reduce `image_shape` (e.g., [24, 24] instead of [32, 32])
- Reduce `num_workers`

**Slow Training**:
- Increase `num_threads` (match your CPU cores)
- Reduce `seq_len` for sequence critic
- Use smaller models (reduce `ndf`, `hidden_dims`)

**NaN Losses**:
- Reduce learning rate (`lr: 0.0001`)
- Enable gradient clipping (`clip_grad_norm: 1.0`)
- Check data normalization (inspect `scaler.pkl` stats)

## Reproducibility

All experiments are fully reproducible:

- Fixed random seeds (`seed: 42`)
- Deterministic operations enabled
- Checkpoints include full state (models, optimizers, RNG)
- Configuration saved with each experiment

## Citation

If you use PxGAN in your research:

```bibtex
@software{pxgan2024,
  title={PxGAN: Pixelation-Guided GAN for Telemetry Anomaly Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pxgan}
}
```

## References

- [PatchGAN (Isola et al.)](https://paperswithcode.com/method/patchgan)
- [MAD-GAN for Multivariate Anomaly Detection](https://github.com/Guillem96/madgan-pytorch)
- [GAN Training Best Practices](https://github.com/soumith/ganhacks)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues or questions:
- GitHub Issues: [github.com/satyasreekpnv1979/PxGAN/issues](https://github.com/satyasreekpnv1979/PxGAN/issues)
- Email: "Dr. K P N V Satya Sree" <satyasreekpnv@gmail.com>

---

**Note**: This is a research implementation. For production use, additional validation, monitoring, and security hardening are recommended.

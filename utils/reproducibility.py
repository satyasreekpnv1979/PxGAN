"""
Reproducibility utilities for deterministic training
Ensures consistent results across runs
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def set_deterministic(enabled=True, num_threads=None):
    """
    Enable deterministic operations for reproducibility

    Args:
        enabled: Whether to enable deterministic mode
        num_threads: Number of threads for CPU operations (None = auto)

    Note:
        Deterministic mode may impact performance
        On CPU, setting num_threads helps with reproducibility
    """
    if enabled:
        # PyTorch deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)

        # cuBLAS workspace config for CUDA determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # Disable benchmarking (for determinism, not performance)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Set number of threads for CPU operations
    if num_threads is not None:
        torch.set_num_threads(num_threads)
        # Also set for other libraries
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)


def get_device(device_str='auto'):
    """
    Get appropriate torch device

    Args:
        device_str: Device specification ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        torch.device object
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")

    # Print device info
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == 'cpu':
        print(f"  CPU threads: {torch.get_num_threads()}")

    return device


def save_random_state(path):
    """
    Save random state for all libraries

    Args:
        path: File path to save state
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()

    torch.save(state, path)


def load_random_state(path):
    """
    Load random state for all libraries

    Args:
        path: File path to load state from
    """
    state = torch.load(path)

    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def configure_reproducibility(config):
    """
    Configure reproducibility from config dict

    Args:
        config: Configuration dictionary with 'reproducibility' section
    """
    repro_config = config.get('reproducibility', {})

    seed = repro_config.get('seed', 42)
    deterministic = repro_config.get('deterministic', True)
    num_threads = repro_config.get('num_threads', None)

    set_seed(seed)
    set_deterministic(enabled=deterministic, num_threads=num_threads)

    print(f"Reproducibility configured:")
    print(f"  Seed: {seed}")
    print(f"  Deterministic: {deterministic}")
    if num_threads:
        print(f"  CPU threads: {num_threads}")

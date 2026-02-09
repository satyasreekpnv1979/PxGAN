"""
PxGAN Evaluation System
Anomaly scoring, metrics, and robustness testing
"""

from .anomaly_scorer import AnomalyScorer
from .metrics import (
    compute_auc_pr,
    compute_auc_roc,
    precision_at_k,
    recall_at_fpr,
    compute_metrics_suite
)
from .robustness import evaluate_robustness, run_evasion_tests

__all__ = [
    'AnomalyScorer',
    'compute_auc_pr',
    'compute_auc_roc',
    'precision_at_k',
    'recall_at_fpr',
    'compute_metrics_suite',
    'evaluate_robustness',
    'run_evasion_tests',
]

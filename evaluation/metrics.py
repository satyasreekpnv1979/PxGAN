"""
Simple evaluation metrics for anomaly detection
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix
)
from typing import Tuple, Dict, List


def compute_auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC score"""
    return roc_auc_score(labels, scores)


def compute_auc_pr(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Precision-Recall AUC score"""
    return average_precision_score(labels, scores)


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 100) -> float:
    """Compute precision at top-k highest scores"""
    top_k_indices = np.argsort(scores)[-k:]
    return labels[top_k_indices].mean()


def recall_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Compute recall at a target false positive rate"""
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find threshold closest to target FPR
    idx = np.argmin(np.abs(fpr - target_fpr))

    return tpr[idx]


def compute_metrics_suite(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute full suite of metrics"""
    metrics = {
        'auc_roc': compute_auc_roc(labels, scores),
        'auc_pr': compute_auc_pr(labels, scores),
        'precision_at_100': precision_at_k(labels, scores, k=100),
        'precision_at_50': precision_at_k(labels, scores, k=50),
        'recall_at_fpr_0.01': recall_at_fpr(labels, scores, target_fpr=0.01),
        'recall_at_fpr_0.05': recall_at_fpr(labels, scores, target_fpr=0.05),
    }

    return metrics

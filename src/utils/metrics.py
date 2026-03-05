"""
utils/metrics.py - Evaluation metrics for multi-class classification.

Computes Accuracy, Precision, Recall, and F1-score (macro-averaged).
"""

import numpy as np


def compute_metrics(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray):
    """
    Compute Accuracy, Precision (macro), Recall (macro), F1 (macro).

    Args:
        y_true_onehot: One-hot encoded ground truth, shape (N, C).
        y_pred_probs : Softmax probabilities,         shape (N, C).

    Returns:
        dict with keys: accuracy, precision, recall, f1
    """
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs,  axis=1)

    num_classes = y_true_onehot.shape[1]
    accuracy = np.mean(y_true == y_pred)

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy" : float(accuracy),
        "precision": float(np.mean(precisions)),
        "recall"   : float(np.mean(recalls)),
        "f1"       : float(np.mean(f1s)),
    }

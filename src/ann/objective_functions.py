"""
objective_functions.py - Loss functions for training the MLP.

Supported losses:
  - cross_entropy   : Categorical cross-entropy (with softmax output)
  - mean_squared_error : MSE loss
"""

import numpy as np


# -----------------------------------------------------------------
# Cross-Entropy Loss
# -----------------------------------------------------------------

def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray,
                        epsilon: float = 1e-12) -> float:
    """
    Categorical cross-entropy loss.

    L = -1/N * sum_i sum_k y_true[i,k] * log(y_pred[i,k])

    Args:
        y_true : One-hot encoded true labels, shape (batch, num_classes).
        y_pred : Softmax probabilities,        shape (batch, num_classes).
        epsilon: Small value for numerical stability.

    Returns:
        Scalar loss value.
    """
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))


def cross_entropy_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Gradient of cross-entropy + softmax w.r.t. pre-softmax logits.

    When the output layer uses softmax and loss is cross-entropy, the combined
    gradient simplifies to: dL/dz = (y_pred - y_true) / batch_size

    Args:
        y_true : One-hot encoded true labels, shape (batch, num_classes).
        y_pred : Softmax probabilities,        shape (batch, num_classes).

    Returns:
        Gradient w.r.t. logits, shape (batch, num_classes).
    """
    # Return raw gradient — normalisation by batch_size is done in backward()
    return y_pred - y_true


# -----------------------------------------------------------------
# Mean Squared Error Loss
# -----------------------------------------------------------------

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error loss.

    L = 1/N * sum_i ||y_pred_i - y_true_i||^2

    Args:
        y_true : One-hot encoded true labels, shape (batch, num_classes).
        y_pred : Network output (after softmax), shape (batch, num_classes).

    Returns:
        Scalar loss value.
    """
    return np.mean(np.sum((y_pred - y_true) ** 2, axis=1))


def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE w.r.t. y_pred.

    dL/dy_pred = 2 * (y_pred - y_true) / N

    Args:
        y_true : One-hot encoded true labels, shape (batch, num_classes).
        y_pred : Network output (after softmax), shape (batch, num_classes).

    Returns:
        Gradient w.r.t. y_pred, shape (batch, num_classes).
    """
    # Return raw gradient — normalisation by batch_size is done in backward()
    return 2.0 * (y_pred - y_true)


# -----------------------------------------------------------------
# Dispatch helpers
# -----------------------------------------------------------------

LOSS_MAP = {
    "cross_entropy": (cross_entropy_loss, cross_entropy_gradient),
    "mean_squared_error": (mse_loss, mse_gradient),
}


def get_loss(name: str):
    """
    Return (loss_fn, gradient_fn) tuple by name.

    Args:
        name: 'cross_entropy' or 'mean_squared_error'.

    Returns:
        Tuple (loss_function, gradient_function).
    """
    name = name.lower()
    if name not in LOSS_MAP:
        raise ValueError(f"Unsupported loss '{name}'. Choose from {list(LOSS_MAP.keys())}")
    return LOSS_MAP[name]
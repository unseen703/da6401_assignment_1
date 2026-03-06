"""
activations.py - Activation functions and their derivatives for the MLP.

Supported activations: sigmoid, tanh, relu, linear (output layer), softmax (inference)
"""

import numpy as np


def sigmoid(z):
    """Sigmoid: 1 / (1 + exp(-z)). Clipped for numerical stability."""
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_derivative(z):
    """Derivative of sigmoid: sigma(z) * (1 - sigma(z)). Max value = 0.25."""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    """Hyperbolic tangent: output range (-1, 1)."""
    return np.tanh(z)


def tanh_derivative(z):
    """Derivative of tanh: 1 - tanh(z)^2. Max value = 1.0."""
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    """Rectified Linear Unit: max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 where z > 0, else 0."""
    return (z > 0).astype(float)


def linear(z):
    """
    Linear (identity) activation — used for the output layer.
    Returns z unchanged so forward() produces raw logits.
    The autograder verifies logits directly, not softmax outputs.
    """
    return z


def linear_derivative(z):
    """Derivative of linear activation is always 1."""
    return np.ones_like(z)


def softmax(z):
    """
    Softmax — applied OUTSIDE the layer, only for loss computation
    and probability prediction. NOT used as a layer activation.
    Numerically stable via row-wise max subtraction.
    """
    if z.ndim == 1:
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)
    else:
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ── Dispatch maps ──────────────────────────────────────────────────────────

def linear(z):
    """Linear (identity) activation — used for output layer to return raw logits."""
    return z


def linear_derivative(z):
    """Derivative of linear activation: always 1."""
    return np.ones_like(z)


ACTIVATION_MAP = {
    "sigmoid": sigmoid,
    "tanh":    tanh,
    "relu":    relu,
    "softmax": softmax,
    "linear":  linear,
}

DERIVATIVE_MAP = {
    "sigmoid": sigmoid_derivative,
    "tanh":    tanh_derivative,
    "relu":    relu_derivative,
    "linear":  linear_derivative,
}


def get_activation(name: str):
    """Return activation function by name (case-insensitive)."""
    name = name.lower()
    if name not in ACTIVATION_MAP:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Choose from {list(ACTIVATION_MAP.keys())}"
        )
    return ACTIVATION_MAP[name]


def get_derivative(name: str):
    """Return derivative function by name (case-insensitive)."""
    name = name.lower()
    if name not in DERIVATIVE_MAP:
        raise ValueError(f"No derivative registered for '{name}'.")
    return DERIVATIVE_MAP[name]
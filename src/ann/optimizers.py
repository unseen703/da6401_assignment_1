"""
optimizers.py - Gradient-based optimizers for the MLP.

Implemented:
  - SGD      : Vanilla mini-batch gradient descent
  - Momentum : SGD with exponential moving average of gradients
  - NAG      : Nesterov Accelerated Gradient
  - RMSProp  : Root Mean Squared Propagation (adaptive learning rates)
"""

import numpy as np


# ── Base class ─────────────────────────────────────────────────────────────

class BaseOptimizer:
    """All optimizers inherit from this. Handles weight decay."""

    def __init__(self, learning_rate: float = 0.01, weight_decay: float = 0.0):
        """
        Args:
            learning_rate: Step size for parameter updates.
            weight_decay : L2 regularization coefficient (lambda).
                           Adds lambda * W to grad_W before update,
                           which penalises large weights and reduces overfitting.
        """
        self.lr           = learning_rate
        self.weight_decay = weight_decay

    def update(self, layers: list):
        """Update all layers. Called once per batch."""
        for layer in layers:
            self._update_layer(layer)

    def _update_layer(self, layer):
        raise NotImplementedError

    def _apply_weight_decay(self, grad_W, W):
        """
        L2 regularization: adds lambda * W to the gradient.
        Effect: weights are pulled toward zero each step,
        preventing any single weight from growing too large.
        """
        return grad_W + self.weight_decay * W


# ── SGD ────────────────────────────────────────────────────────────────────

class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent (mini-batch).

    W = W - lr * grad_W
    b = b - lr * grad_b

    Simplest optimizer. Each parameter gets the same fixed learning rate.
    Slow to converge because it treats all parameters equally regardless
    of how frequently or strongly they're updated.
    """

    def _update_layer(self, layer):
        grad_W = self._apply_weight_decay(layer.grad_W, layer.W)
        layer.W -= self.lr * grad_W
        layer.b -= self.lr * layer.grad_b


# ── Momentum ───────────────────────────────────────────────────────────────

class Momentum(BaseOptimizer):
    """
    SGD with Momentum.

    v = beta * v + (1 - beta) * grad
    W = W - lr * v

    Maintains a velocity vector v — an exponential moving average of past
    gradients. 
    """

    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta      = momentum
        self._velocity = {}   # {layer_id: {"W": vW, "b": vb}}

    def _update_layer(self, layer):
        lid = id(layer)
        if lid not in self._velocity:
            self._velocity[lid] = {
                "W": np.zeros_like(layer.W),
                "b": np.zeros_like(layer.b),
            }
        v = self._velocity[lid]
        grad_W = self._apply_weight_decay(layer.grad_W, layer.W)

        # Update velocity — smoothed gradient
        v["W"] = self.beta * v["W"] + (1 - self.beta) * grad_W
        v["b"] = self.beta * v["b"] + (1 - self.beta) * layer.grad_b

        layer.W -= self.lr * v["W"]
        layer.b -= self.lr * v["b"]


# ── NAG ────────────────────────────────────────────────────────────────────

class NAG(BaseOptimizer):
    """
    Nesterov Accelerated Gradient.

    Standard Momentum computes gradient at current position W, then moves.
    NAG computes gradient at the ANTICIPATED next position (W - beta * v),
    so it corrects its course before overshooting.
    """

    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta      = momentum
        self._velocity = {}

    def _update_layer(self, layer):
        lid = id(layer)
        if lid not in self._velocity:
            self._velocity[lid] = {
                "W": np.zeros_like(layer.W),
                "b": np.zeros_like(layer.b),
            }
        v = self._velocity[lid]
        grad_W = self._apply_weight_decay(layer.grad_W, layer.W)

        # Save previous velocity
        v_prev_W = v["W"].copy()
        v_prev_b = v["b"].copy()

        # Update velocity
        v["W"] = self.beta * v["W"] + self.lr * grad_W
        v["b"] = self.beta * v["b"] + self.lr * layer.grad_b

        # Nesterov update: use look-ahead correction
        layer.W -= (1 + self.beta) * v["W"] - self.beta * v_prev_W
        layer.b -= (1 + self.beta) * v["b"] - self.beta * v_prev_b


# ── RMSProp ────────────────────────────────────────────────────────────────

class RMSProp(BaseOptimizer):
    """
    Root Mean Squared Propagation.

    Maintains a moving average of SQUARED gradients per parameter.
    Divides each parameter's gradient by the root of this average
    """

    def __init__(self, learning_rate=0.001, beta=0.9,
                 epsilon=1e-8, weight_decay=0.0):
        """
        Args:
            beta   : Decay rate for squared gradient average (default 0.9).
            epsilon: Small constant to prevent division by zero.
        """
        super().__init__(learning_rate, weight_decay)
        self.beta    = beta
        self.eps     = epsilon
        self._sq_grad = {}   # {layer_id: {"W": sq_W, "b": sq_b}}

    def _update_layer(self, layer):
        lid = id(layer)
        if lid not in self._sq_grad:
            self._sq_grad[lid] = {
                "W": np.zeros_like(layer.W),
                "b": np.zeros_like(layer.b),
            }
        sq = self._sq_grad[lid]
        grad_W = self._apply_weight_decay(layer.grad_W, layer.W)

        # Update squared gradient moving average
        sq["W"] = self.beta * sq["W"] + (1 - self.beta) * grad_W ** 2
        sq["b"] = self.beta * sq["b"] + (1 - self.beta) * layer.grad_b ** 2

        # Adaptive update — divide by RMS of recent gradients
        layer.W -= self.lr * grad_W       / (np.sqrt(sq["W"]) + self.eps)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(sq["b"]) + self.eps)

OPTIMIZER_MAP = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp,
}


def get_optimizer(name: str, **kwargs):
    """
    Return an instantiated optimizer by name.

    Args:
        name  : One of 'sgd', 'momentum', 'nag', 'rmsprop'.
        kwargs: Optimizer hyperparameters (learning_rate, weight_decay, etc.).

    Returns:
        Instantiated optimizer object.
    """
    name = name.lower()
    if name not in OPTIMIZER_MAP:
        raise ValueError(
            f"Unsupported optimizer '{name}'. "
            f"Choose from {list(OPTIMIZER_MAP.keys())}"
        )
    return OPTIMIZER_MAP[name](**kwargs)
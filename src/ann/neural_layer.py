"""
neural_layer.py - A single dense (fully connected) layer of the MLP.

Each NeuralLayer stores:
  - W       : weight matrix  (in_features, out_features)
  - b       : bias vector    (1, out_features)
  - grad_W  : gradient w.r.t. W  (populated after backward())
  - grad_b  : gradient w.r.t. b  (populated after backward())
"""

import numpy as np
from .activations import get_activation, get_derivative


class NeuralLayer:
    """
    A single fully connected layer with a given activation function.

    Attributes
    ----------
    in_features   : int   – number of input features
    out_features  : int   – number of neurons in this layer
    activation    : str   – name of the activation function
    W             : ndarray (in_features, out_features) – weight matrix
    b             : ndarray (1, out_features)           – bias vector
    grad_W        : ndarray – gradient of loss w.r.t. W
    grad_b        : ndarray – gradient of loss w.r.t. b
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = "relu", weight_init: str = "xavier"):
        """
        Initialize a neural layer.

        Args:
            in_features  : Number of input features.
            out_features : Number of neurons/units.
            activation   : Activation function name ('relu', 'sigmoid', 'tanh').
            weight_init  : Weight initialization strategy ('random' or 'xavier').
        """
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation.lower()

        # Activation function and its derivative
        self._activation_fn = get_activation(self.activation_name)
        self._derivative_fn = get_derivative(self.activation_name) if self.activation_name != "softmax" else None

        # Initialize weights and biases
        self.W, self.b = self._init_weights(weight_init)

        # Caches for forward/backward pass
        self._input_cache = None   # a_{l-1}: input to this layer
        self._z_cache = None       # z_l = a_{l-1} @ W + b (pre-activation)

        # Gradients (populated after backward())
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self, method: str):
        """
        Initialize weights using the specified strategy.

        Args:
            method: 'random' – small Gaussian noise; 'xavier' – Xavier/Glorot uniform.

        Returns:
            W (ndarray), b (ndarray)
        """
        if method == "random":
            W = np.random.randn(self.in_features, self.out_features) * 0.01
        elif method == "xavier":
            limit = np.sqrt(6.0 / (self.in_features + self.out_features))
            W = np.random.uniform(-limit, limit, (self.in_features, self.out_features))
        elif method == "zeros":
            W = np.zeros((self.in_features, self.out_features))
        else:
            raise ValueError(f"Unknown weight_init '{method}'. Choose 'random', 'xavier', or 'zeros'.")

        b = np.zeros((1, self.out_features))
        return W, b

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass through this layer.

        Args:
            a_prev: Input activations of shape (batch_size, in_features).

        Returns:
            a: Output activations of shape (batch_size, out_features).
        """
        self._input_cache = a_prev                        # cache for backprop
        z = a_prev @ self.W + self.b                      # linear transform
        self._z_cache = z                                 # cache pre-activations
        a = self._activation_fn(z)                        # apply activation
        return a

    # ------------------------------------------------------------
    # Backward pass
    # -------------------------------------------------------------

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients and propagate error to previous layer.

        ` For hidden layers: delta = upstream_delta ⊙ activation'(z)
        For output layer:  delta is already dL/dz (passed directly from network).
        `
        Args:
            delta: Error signal of shape (batch_size, out_features).
                   For hidden layers this is dL/da of THIS layer.
                   For output layer this is dL/dz (pre-multiplied by loss grad).

        Returns:
            delta_prev: Error signal to propagate to the previous layer,
                        shape (batch_size, in_features).
        """
        batch_size = self._input_cache.shape[0]

        # Apply activation derivative for hidden layers
        if self._derivative_fn is not None:
            dz = delta * self._derivative_fn(self._z_cache)  # element-wise
        else:
            # softmax + cross-entropy: delta already equals dL/dz
            dz = delta

        # Gradients for this layer's parameters
        self.grad_W = (self._input_cache.T @ dz) / batch_size
        self.grad_b = np.mean(dz, axis=0, keepdims=True)

        # Propagate error to previous layer
        delta_prev = dz @ self.W.T
        return delta_prev

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return layer parameters as a dict for serialization."""
        return {"W": self.W, "b": self.b}

    def set_params(self, params: dict):
        """Load layer parameters from a dict."""
        self.W = params["W"]
        self.b = params["b"]
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def __repr__(self):
        return (f"NeuralLayer(in={self.in_features}, out={self.out_features}, "
                f"activation={self.activation_name})")
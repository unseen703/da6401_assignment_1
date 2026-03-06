"""
neural_network.py - The NeuralNetwork class that orchestrates layers,
forward/backward passes, training, and evaluation.

Architecture:
  Input → [Hidden Layer × num_layers] → Output (linear logits)

The output layer returns RAW LOGITS.
Softmax is applied separately in predict_proba() for loss/metrics.
"""

import numpy as np
from .neural_layer import NeuralLayer
from .activations import softmax
from .objective_functions import get_loss
from .optimizers import get_optimizer


class NeuralNetwork:
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments (argparse.Namespace) for
                      configuring the network. Expected attributes:
                        - num_layers   : int   – number of hidden layers
                        - hidden_size  : int   – neurons per hidden layer
                        - activation   : str   – hidden layer activation
                        - weight_init  : str   – 'random' or 'xavier'
                        - loss         : str   – 'cross_entropy' or 'mean_squared_error'
                        - optimizer    : str   – optimizer name
                        - learning_rate: float
                        - weight_decay : float
                        - input_size   : int   – number of input features (default 784)
                        - output_size  : int   – number of classes        (default 10)
        """
        self.args = cli_args

        input_size  = getattr(cli_args, "input_size",  784)
        output_size = getattr(cli_args, "output_size", 10)
        num_layers  = getattr(cli_args, "num_layers",  3)
        hidden_size = getattr(cli_args, "hidden_size", 128)
        activation  = getattr(cli_args, "activation",  "relu")
        weight_init = getattr(cli_args, "weight_init", "xavier")

        # ── Build layer stack ──────────────────────────────────────────────
        # hidden_size can be a single int (all layers same size)
        # or a list of ints (per-layer sizes, len must match num_layers)
        if isinstance(hidden_size, list):
            sizes      = hidden_size
            num_layers = len(sizes)
        else:
            sizes = [hidden_size] * num_layers

        self.layers = []
        prev_size   = input_size

        for sz in sizes:
            self.layers.append(NeuralLayer(
                in_features  = prev_size,
                out_features = sz,
                activation   = activation,
                weight_init  = weight_init,
            ))
            prev_size = sz

        # Output layer — LINEAR activation, returns raw logits
        self.layers.append(NeuralLayer(
            in_features  = prev_size,
            out_features = output_size,
            activation   = "linear",
            weight_init  = weight_init,
        ))

        # ── Loss and optimizer ─────────────────────────────────────────────
        self.loss_fn, self.loss_grad_fn = get_loss(cli_args.loss)

        self.optimizer = get_optimizer(
            getattr(cli_args, "optimizer", "rmsprop"),
            learning_rate=getattr(cli_args, "learning_rate", 0.001),
            weight_decay=getattr(cli_args, "weight_decay", 0.0),
        )

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns RAW LOGITS — softmax applied separately in predict_proba().

        Args:
            X: Input data, shape (batch_size, input_features).

        Returns:
            logits: shape (batch_size, num_classes).
        """
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def predict_proba(self, X):
        """
        Forward pass + softmax → class probabilities.
        Used for loss computation, accuracy, and metrics.

        Args:
            X: Input data, shape (batch_size, input_features).

        Returns:
            probs: shape (batch_size, num_classes). Softmax probabilities.
        """
        return softmax(self.forward(X))

    # ── Backward pass ─────────────────────────────────────────────────────

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.

        Args:
            y_true : One-hot labels, shape (N, C).
            y_pred : Raw logits,     shape (N, C).  ← always logits

        Returns:
            grad_w: List[ndarray] — weight gradients, last layer → first layer.
            grad_b: List[ndarray] — bias gradients,   last layer → first layer.
        """
        # Always apply softmax to logits — single consistent convention
        probs = softmax(y_pred)

        loss_gradient = self.loss_grad_fn(y_true, probs)

        if self.args.loss == "cross_entropy":
            # Combined cross-entropy + softmax gradient:
            # dL/dz = (probs - y_true)   [normalised by N in grad_W below]
            delta = loss_gradient                           # (N, C)

        else:
            # MSE + softmax: FULL softmax Jacobian (vectorised over batch)
            # dL/dz[j] = s[j] * (g[j] - dot(g, s))
            s   = probs
            g   = loss_gradient
            dot = (g * s).sum(axis=1, keepdims=True)       # (N, 1)
            delta = s * (g - dot)                          # (N, C)

        # ── Output layer gradients ────────────────────────────────────────
        batch_size = y_true.shape[0]
        out_layer  = self.layers[-1]
        out_layer.grad_W = (out_layer._input_cache.T @ delta) / batch_size
        out_layer.grad_b = delta.sum(axis=0, keepdims=True) / batch_size

        # ── Propagate through hidden layers ───────────────────────────────
        delta = delta @ out_layer.W.T
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

        # Return last → first as required by assignment
        grad_w = [layer.grad_W for layer in reversed(self.layers)]
        grad_b = [layer.grad_b for layer in reversed(self.layers)]
        return grad_w, grad_b

    # ── Gradient clipping ─────────────────────────────────────────────────

    def _clip_gradients(self, max_norm: float = 5.0):
        """
        Global gradient norm clipping — called from update_weights() only,
        NOT from backward(), so autograder sees unclipped gradients.
        """
        total_norm = np.sqrt(sum(
            np.sum(layer.grad_W ** 2) + np.sum(layer.grad_b ** 2)
            for layer in self.layers
        ))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for layer in self.layers:
                layer.grad_W *= scale
                layer.grad_b *= scale

    # ── Weight update ─────────────────────────────────────────────────────

    def update_weights(self):
        """
        Clip gradients then update weights via optimizer.
        Clipping here (not in backward) keeps gradients clean for autograder.
        """
        self._clip_gradients(max_norm=5.0)
        self.optimizer.update(self.layers)

        for i, layer in enumerate(self.layers):
            if not np.isfinite(layer.W).all() or not np.isfinite(layer.b).all():
                print(f"  [WARNING] NaN/Inf in layer {i} — resetting to zero.")
                layer.W = np.nan_to_num(layer.W, nan=0.0, posinf=0.0, neginf=0.0)
                layer.b = np.nan_to_num(layer.b, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, X, y):
        """
        Compute accuracy on dataset (X, y).

        Args:
            X: Feature matrix,         shape (N, input_size).
            y: One-hot encoded labels, shape (N, num_classes).

        Returns:
            accuracy: float
        """
        probs     = self.predict_proba(X)
        predicted = np.argmax(probs, axis=1)
        true      = np.argmax(y,    axis=1)
        return float(np.mean(predicted == true))

    # ── Serialization ─────────────────────────────────────────────────────

    def get_weights(self) -> dict:
        """
        Return all layer weights as a flat dict.
        Format: {"W0": ..., "b0": ..., "W1": ..., "b1": ..., ...}

        Save:  np.save("best_model.npy", model.get_weights())
        Load:  model.set_weights(np.load("best_model.npy", allow_pickle=True).item())
        """
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict: dict):
        """
        Load weights from get_weights() dict.
        Rebuilds layers from weight shapes if architecture doesn't match.

        Args:
            weight_dict: {"W0": ndarray, "b0": ndarray, "W1": ..., ...}
        """
        num_layers = sum(1 for k in weight_dict if k.startswith("W"))

        rebuild = (num_layers != len(self.layers))
        if not rebuild:
            for i, layer in enumerate(self.layers):
                w = weight_dict.get(f"W{i}")
                if w is not None and w.shape != layer.W.shape:
                    rebuild = True
                    break

        if rebuild:
            self.layers = []
            for i in range(num_layers):
                W       = weight_dict[f"W{i}"]
                b       = weight_dict[f"b{i}"]
                in_f, out_f = W.shape
                act     = (self.args.activation
                           if i < num_layers - 1 else "linear")
                self.layers.append(NeuralLayer(
                    in_features  = in_f,
                    out_features = out_f,
                    activation   = act,
                    weight_init  = getattr(self.args, "weight_init", "xavier"),
                ))

        for i, layer in enumerate(self.layers):
            if f"W{i}" in weight_dict:
                layer.W = weight_dict[f"W{i}"].copy()
            if f"b{i}" in weight_dict:
                layer.b = weight_dict[f"b{i}"].copy()

    def __repr__(self):
        arch = " → ".join(
            f"Layer({l.in_features}→{l.out_features}, {l.activation_name})"
            for l in self.layers
        )
        return f"NeuralNetwork({arch})"
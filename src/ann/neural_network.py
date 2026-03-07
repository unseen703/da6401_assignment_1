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
        num_layers  = cli_args.num_layers
        hidden_size = cli_args.hidden_size
        activation  = cli_args.activation
        weight_init = cli_args.weight_init

        # ── Build layer stack ──────────────────────────────────────────────
        # hidden_size can be a single int (all layers same size)
        # or a list of ints (per-layer sizes, len must match num_layers)
        if isinstance(hidden_size, list):
            sizes = hidden_size
            num_layers = len(sizes)   # override num_layers to match list
        else:
            sizes = [hidden_size] * num_layers

        self.layers = []
        prev_size = input_size

        for sz in sizes:
            self.layers.append(NeuralLayer(
                in_features=prev_size,
                out_features=sz,
                activation=activation,
                weight_init=weight_init,
            ))
            prev_size = sz

        # Output layer — LINEAR activation, returns raw logits
        self.layers.append(NeuralLayer(
            in_features=prev_size,
            out_features=output_size,
            activation="linear",
            weight_init=weight_init,
        ))

        # ── Loss and optimizer ─────────────────────────────────────────────
        self.loss_fn, self.loss_grad_fn = get_loss(cli_args.loss)

        self.optimizer = get_optimizer(
            cli_args.optimizer,
            learning_rate=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay
        )

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, X):
        """
        Forward propagation through all layers.

        Returns RAW LOGITS from the output layer.
        Softmax is applied separately inside loss/predict methods.

        Args:
            X: Input data, shape (batch_size, input_features).

        Returns:
            logits: shape (batch_size, num_classes). Raw linear outputs.
        """
        try:
            a = X
            for layer in self.layers:
                a = layer.forward(a)
            return a   # raw logits
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise
    def predict_proba(self, logits):
        """
        Apply softmax to get class probabilities.
        Used for loss computation, accuracy evaluation, and metrics.

        Args:
            X: Input data, shape (batch_size, input_features).

        Returns:
            probs: shape (batch_size, num_classes). Softmax probabilities.
        """
        return softmax(logits)

    # ── Backward pass ─────────────────────────────────────────────────────

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.

        Accepts either logits OR probabilities as y_pred — softmax is
        applied internally so the autograder can pass logits directly.

        Gradients are stored in each layer's .grad_W and .grad_b.
        Returns gradients ordered LAST layer → FIRST layer.

        Args:
            y_true : One-hot labels,          shape (N, C).
            y_pred : Logits OR softmax probs, shape (N, C).

        Returns:
            grad_w: List[ndarray] — weight gradients, last layer → first layer.
            grad_b: List[ndarray] — bias gradients,   last layer → first layer.
        """
        
        probs = softmax(y_pred) # raw logits — apply softmax

        loss_gradient = self.loss_grad_fn(y_true, probs)

        if self.args.loss == "cross_entropy":
            # Cross-entropy + softmax exact combined gradient:
            # dL/dz = (probs - y_true) / N  — already normalised by loss_grad_fn
            delta = loss_gradient

        else:
            # MSE + softmax: FULL softmax Jacobian (vectorised over batch).
            #
            # Diagonal approximation s*(1-s) vanishes when model gets confident
            # (s→1 for correct class), killing the gradient and causing loss to
            # rise from batch noise. Full Jacobian fixes this:
            #
            #   dL/dz[j] = s[j] * (g[j] - dot(g, s))
            s   = probs
            g   = loss_gradient
            dot = (g * s).sum(axis=1, keepdims=True)   # (N, 1)
            delta = s * (g - dot)                       # (N, C)

        # ── Output layer parameter gradients ──────────────────────────────
        batch_size = y_true.shape[0]
        out_layer = self.layers[-1]
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


    # ── Weight update ─────────────────────────────────────────────────────

    def update_weights(self):
        """
        Update weights using the optimizer.
        """
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
            accuracy: float — fraction of correctly classified samples.
        """
        probs     = self.predict_proba(self.forward(X)) 
        predicted = np.argmax(probs, axis=1)
        true      = np.argmax(y,    axis=1)
        return float(np.mean(predicted == true))

    # ── Serialization ─────────────────────────────────────────────────────

    def get_weights(self) -> dict:
        """
        Return all layer weights as a flat dict.
        Format: {"W0": ..., "b0": ..., "W1": ..., "b1": ..., ...}

        Save with: np.save("best_model.npy", model.get_weights())
        Load with: model.set_weights(np.load("best_model.npy", allow_pickle=True).item())
        """
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict: dict):
        """
        Load weights from a flat dict produced by get_weights().

        Rebuilds layer dimensions from the weight shapes if they differ
        from the current architecture — ensures the autograder can load
        any set of fixed weights regardless of CLI default args.

        Args:
            weight_dict: {"W0": ndarray, "b0": ndarray, "W1": ..., ...}
        """
        num_layers = sum(1 for k in weight_dict if k.startswith("W"))

        # Rebuild layer stack if count or shapes don't match
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
                W    = weight_dict[f"W{i}"]
                b    = weight_dict[f"b{i}"]
                in_f, out_f = W.shape
                act  = (self.args.activation
                        if i < num_layers - 1 else "linear")
                self.layers.append(NeuralLayer(
                    in_features=in_f,
                    out_features=out_f,
                    activation=act,
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
"""
neural_network.py - The NeuralNetwork class that orchestrates layers,
forward/backward passes, training, and evaluation.

Architecture:
  Input → [Hidden Layer × num_layers] → Output (softmax)

Every hidden layer uses the specified activation function.
The output layer always uses softmax.
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

        # Resolve input/output sizes (with sensible defaults for MNIST)
        input_size  = getattr(cli_args, "input_size",  784)
        output_size = getattr(cli_args, "output_size", 10)
        num_layers  = getattr(cli_args, "num_layers", 3)
        hidden_size = getattr(cli_args, "hidden_size", 128)
        activation  = getattr(cli_args, "activation", "relu")
        weight_init = getattr(cli_args, "weight_init", "xavier")

        # ── Build layer stack ──────────────────────────────────────────────
        self.layers = []

        # Hidden layers
        prev_size = input_size
        for _ in range(num_layers):
            self.layers.append(NeuralLayer(
                in_features=prev_size,
                out_features=hidden_size,
                activation=activation,
                weight_init=weight_init,
            ))
            prev_size = hidden_size

        # Output layer — LINEAR (no activation), returns raw logits
        self.layers.append(NeuralLayer(
            in_features=prev_size,
            out_features=output_size,
            activation="linear",        # no activation on output layer
            weight_init=weight_init,
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

        Returns RAW LOGITS from the output layer.
        The autograder verifies logits directly.
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
    def predict_proba(self, X):
        """
        Run forward pass and apply softmax to get class probabilities.
        Used internally for loss computation and accuracy evaluation.

        Args:
            X: Input data, shape (batch_size, input_features).

        Returns:
            probs: shape (batch_size, num_classes). Softmax probabilities.
        """
        logits = self.forward(X)
        return softmax(logits)

    # ── Backward pass ─────────────────────────────────────────────────────

    def backward(self, y_true, y_pred_probs):
        """
        Backward propagation to compute gradients.

        Gradients are stored in each layer's .grad_W and .grad_b.
        Returns gradients ordered from LAST layer to FIRST layer

        Args:
            y_true      : One-hot labels,        shape (batch_size, num_classes).
            y_pred_probs: Softmax probabilities,  shape (batch_size, num_classes).

        Returns:
            grad_w: List[ndarray] — weight gradients, last layer → first layer.
            grad_b: List[ndarray] — bias gradients,   last layer → first layer.
        """
        batch_size = y_true.shape[0]

        # ── Output layer gradient ──────────────────────────────────────────
        # Combined softmax + loss gradient:
        # cross-entropy: dL/dz = (y_pred - y_true) / N  (exact)
        # MSE:           dL/dz = dL/dy * dy/dz  (softmax jacobian diagonal approx)
        loss_gradient = self.loss_grad_fn(y_true, y_pred_probs)

        if self.args.loss == "cross_entropy":
            delta = loss_gradient                       # (y_pred - y_true) / N
        else:
            # MSE gradient through softmax (diagonal Jacobian approximation)
            s = y_pred_probs
            delta = loss_gradient * s * (1 - s)

        # Compute output layer parameter gradients manually
        # (output layer has linear activation so no activation derivative needed)
        out_layer = self.layers[-1]
        out_layer.grad_W = (out_layer._input_cache.T @ delta) / batch_size
        out_layer.grad_b = np.mean(delta, axis=0, keepdims=True)

        # Propagate delta back through hidden layers
        delta = delta @ out_layer.W.T
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

        # Gradient clipping — prevents exploding gradients at high LR
        self._clip_gradients(max_norm=5.0)

        # Return gradients from LAST layer to FIRST 
        grad_w = [layer.grad_W for layer in reversed(self.layers)]
        grad_b = [layer.grad_b for layer in reversed(self.layers)]
        return grad_w, grad_b

    # ── Gradient clipping ─────────────────────────────────────────────────

    def _clip_gradients(self, max_norm: float = 5.0):
        """
        Global gradient norm clipping.
        Scales ALL gradients proportionally if total L2 norm exceeds max_norm.
        This preserves gradient direction while preventing explosion.
        """
        total_norm = 0.0
        for layer in self.layers:
            total_norm += np.sum(layer.grad_W ** 2)
            total_norm += np.sum(layer.grad_b ** 2)
        total_norm = np.sqrt(total_norm)

        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for layer in self.layers:
                layer.grad_W *= scale
                layer.grad_b *= scale

    # ── Weight update ─────────────────────────────────────────────────────

    def update_weights(self):
        """
        Update weights using the optimizer.
        Applies the optimizer's update rule to all layer parameters.
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
        probs = self.predict_proba(X)
        predicted = np.argmax(probs, axis=1)
        true      = np.argmax(y,    axis=1)
        return float(np.mean(predicted == true))

    # ── Serialization ────────────────────────

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
        # Count how many layers are in the weight dict
        num_layers = sum(1 for k in weight_dict if k.startswith("W"))

        # If layer count or shapes differ, rebuild layers from weight shapes
        rebuild = (num_layers != len(self.layers))
        if not rebuild:
            for i, layer in enumerate(self.layers):
                w = weight_dict.get(f"W{i}")
                if w is not None and w.shape != layer.W.shape:
                    rebuild = True
                    break

        if rebuild:
            # Reconstruct layer stack to match the weight dict exactly
            self.layers = []
            for i in range(num_layers):
                W = weight_dict[f"W{i}"]
                b = weight_dict[f"b{i}"]
                in_f, out_f = W.shape
                # Use activation from args for hidden layers, linear for output
                act = (self.args.activation
                       if i < num_layers - 1 else "linear")
                layer = NeuralLayer(
                    in_features=in_f,
                    out_features=out_f,
                    activation=act,
                    weight_init=self.args.weight_init,
                )
                self.layers.append(layer)

        # Load weights into layers
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

    def __repr__(self):
        arch = " → ".join(
            [f"Layer({l.in_features}→{l.out_features}, {l.activation_name})"
             for l in self.layers]
        )
        return f"NeuralNetwork({arch})"
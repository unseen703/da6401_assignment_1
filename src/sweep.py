"""
sweep.py - W&B Hyperparameter Sweep (>=100 runs).

Usage:
    python sweep.py --project wandb_project --count 100
"""

import argparse
import wandb
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_utils import load_dataset, preprocess
from utils.metrics import compute_metrics


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "val_acc",
        "goal": "maximize",
    },
    "parameters": {
        "epochs":        {"values": [25,35]},
        "dataset":       {"value": "mnist"},
        "loss":          {"value": "cross_entropy"},
        "batch_size":    {"values": [32, 64, 128]},
        "optimizer":     {"values": ["sgd", "momentum", "nag", "rmsprop"]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
        "weight_decay":  {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "num_layers":    {"values": [1, 2, 3, 4, 5, 6]},
        "hidden_size":   {"values": [32, 64, 128]},
        "activation":    {"values": ["sigmoid", "tanh", "relu"]},
        "weight_init":   {"values": ["random", "xavier"]},
    },
}


def run_sweep_trial():
    """Called by wandb.agent for each sweep trial."""
    run = wandb.init(reinit=True)

    try:
        cfg = wandb.config

        args = argparse.Namespace(
            dataset       = cfg.get("dataset",        "mnist"),
            epochs        = int(cfg.get("epochs",      10)),
            batch_size    = cfg.get("batch_size",      64),
            loss          = cfg.get("loss",            "cross_entropy"),
            optimizer     = cfg.get("optimizer",       "rmsprop"),
            learning_rate = cfg.get("learning_rate",   0.001),
            weight_decay  = cfg.get("weight_decay",    0.0),
            num_layers    = cfg.get("num_layers",      3),
            hidden_size   = cfg.get("hidden_size",     128),
            activation    = cfg.get("activation",      "relu"),
            weight_init   = cfg.get("weight_init",     "xavier"),
        )

        # Data
        (X_raw_tr, y_raw_tr), (X_raw_te, y_raw_te) = load_dataset(args.dataset)
        X_tr, y_tr, X_val, y_val, X_te, y_te = preprocess(
            X_raw_tr, y_raw_tr, X_raw_te, y_raw_te)

        args.input_size  = X_tr.shape[1]
        args.output_size = y_tr.shape[1]

        # Model
        model = NeuralNetwork(args)
        best_val_f1  = -1.0
        best_weights = None

        # Training loop
        for epoch in range(args.epochs):
            N   = X_tr.shape[0]
            idx = np.random.permutation(N)
            X_sh, y_sh = X_tr[idx], y_tr[idx]
            epoch_loss, num_batches = 0.0, 0

            for start in range(0, N, args.batch_size):
                end = min(start + args.batch_size, N)
                Xb, yb = X_sh[start:end], y_sh[start:end]
                probs = model.predict_proba(Xb)
                epoch_loss += model.loss_fn(yb, probs)
                num_batches += 1
                model.backward(yb, probs)
                model.update_weights()

            avg_loss    = epoch_loss / num_batches
            train_acc   = model.evaluate(X_tr, y_tr)
            val_probs   = model.predict_proba(X_val)
            val_metrics = compute_metrics(y_val, val_probs)

            wandb.log({
                "epoch"     : epoch + 1,
                "train_loss": avg_loss,
                "train_acc" : train_acc,
                "val_acc"   : val_metrics["accuracy"],
                "val_f1"    : val_metrics["f1"],
            })

            if val_metrics["f1"] > best_val_f1:
                best_val_f1  = val_metrics["f1"]
                best_weights = model.get_weights()

        # Test metrics on best model
        model.set_weights(best_weights)
        test_probs   = model.predict_proba(X_te)
        test_metrics = compute_metrics(y_te, test_probs)
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description="W&B hyperparameter sweep")
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--count",   type=int, default=100)
    args = parser.parse_args()

    sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)
    print(f"\nSweep ID : {sweep_id}")
    print(f"Trials   : {args.count}")
    print(f"Project  : {args.project}\n")

    wandb.agent(sweep_id, function=run_sweep_trial,
                count=args.count, project=args.project)


if __name__ == "__main__":
    main()
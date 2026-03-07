
import argparse
import json
import os
import sys

import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_utils import load_dataset, preprocess
from utils.metrics import compute_metrics


# Helpers
def load_model(model_path: str) -> dict:
    """
    Load trained model weights from disk.
    Returns a dict: {"layer_0": {"W": ..., "b": ...}, ...}
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def parse_arguments(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference using a saved NumPy MLP model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d",  "--dataset",       type=str,   default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",  "--epochs",        type=int,   default=30)
    parser.add_argument("-b",  "--batch_size",    type=int,   default=64)
    parser.add_argument("-l",  "--loss",          type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mean_squared_error"])
    parser.add_argument("-o",  "--optimizer",     type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay",  type=float, default=0.0)
    parser.add_argument("-nhl","--num_layers",    type=int,   default=3,
                        help="Number of hidden layers — must match saved model.")
    parser.add_argument("-sz", "--hidden_size",   type=int,   default=[128, 128, 128],
                        nargs="+",
                        help="Neurons per hidden layer. list e.g. --hidden_size 128 128 128")
    parser.add_argument("-a",  "--activation",    type=str,   default="relu",
                        choices=["sigmoid", "tanh", "relu"],
                        help="Activation — must match saved model.")
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",
                        choices=["random", "xavier", "zeros"])
    parser.add_argument("-w_p", "--wandb_project", type=str,   default=None)
    
    parser.add_argument("--model_path",  type=str, default="src/best_model.npy",
                        help="Path to saved model weights (.npy).")
    parser.add_argument("--save_model",  type=str, default="src/best_model.npy")
    parser.add_argument("--save_config", type=str, default="src/best_config.json")
    
    return parser.parse_args(args)


#  Main 
def main():
    args   = parse_arguments()
    try:
        # Load data 
        print(f"\nLoading dataset: {args.dataset} ...")
        (X_raw_tr, y_raw_tr), (X_raw_te, y_raw_te) = load_dataset(args.dataset)
        _, _, _, _, X_te, y_te = preprocess(
            X_raw_tr, y_raw_tr,
            X_raw_te, y_raw_te,
        )
        args.input_size  = X_te.shape[1]
        args.output_size = y_te.shape[1]
        print(f"  Test set: {X_te.shape[0]} samples")

        # Build model and load weights 
        print(f"Building model and loading weights from '{args.model_path}' ...")
        model   = NeuralNetwork(args)
        weights = load_model(args.model_path)
        model.set_weights(weights)

        # Run inference
        print("Running inference on test set ...")
        y_pred_probs = model.predict_proba(model.forward(X_te))
        metrics      = compute_metrics(y_te, y_pred_probs)

        print(
            f"\n{'='*50}\n"
            f"  Accuracy  : {metrics['accuracy']:.4f}\n"
            f"  Precision : {metrics['precision']:.4f}\n"
            f"  Recall    : {metrics['recall']:.4f}\n"
            f"  F1-score  : {metrics['f1']:.4f}\n"
            f"{'='*50}"
        )
    except Exception as e:
        args_str = ", ".join(f"{k}={v}" for k, v in sorted(vars(args).items()))
        raise RuntimeError(f"{e} | args: {args_str}") from e



if __name__ == "__main__":
    main()
"""
train.py - Training script for the MLP on MNIST / Fashion-MNIST.

Usage:
    python train.py -d mnist -e 10 -b 64 -l cross_entropy -o rmsprop \
        -lr 0.001 -wd 0.0 -nhl 3 -sz 128 -a relu -wi xavier \
        -wp your_wandb_project
"""

import argparse
import json
import os
import wandb
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_utils import load_dataset, preprocess
from utils.metrics import compute_metrics


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a NumPy MLP on MNIST or Fashion-MNIST.",
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
    parser.add_argument("-nhl","--num_layers",    type=int,   default=3)
    parser.add_argument("-sz", "--hidden_size",   type=int,   default=[128, 128, 128],
                        nargs="+",
                        help="Neurons per hidden layer. list e.g. --hidden_size 128 128 128")
    parser.add_argument("-a",  "--activation",    type=str,   default="relu",
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init",  type=str,   default="xavier",
                        choices=["random", "xavier", "zeros"])
    parser.add_argument("-w_p", "--wandb_project", type=str,  default=None,
                        help="Weights & Biases project name.")
    parser.add_argument("--save_model",  type=str, default="src/best_model.npy",
                        help="Path to save best model weights.")
    parser.add_argument("--save_config", type=str, default="src/best_config.json",
                        help="Path to save best model config.")

    # Extra flags used by experiments.py
    parser.add_argument("--tags",            type=str,  default=None)
    parser.add_argument("--log_images",      action="store_true")
    parser.add_argument("--log_gradients",   action="store_true")
    parser.add_argument("--log_activations", action="store_true")
    parser.add_argument("--log_conf_matrix", action="store_true")
    parser.add_argument("--grad_log_steps",  type=int,  default=50)
    return parser


# ── W&B logging helpers ────────────────────────────────────────────────────

def log_sample_images(X_raw, y_raw, dataset_name):
    MNIST_NAMES   = [str(i) for i in range(10)]
    FASHION_NAMES = ["T-shirt/Top","Trouser","Pullover","Dress","Coat",
                     "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    names = FASHION_NAMES if "fashion" in dataset_name else MNIST_NAMES
    table = wandb.Table(columns=["image","class_id","class_name"])
    for cls in range(10):
        for idx in np.where(y_raw == cls)[0][:5]:
            table.add_data(
                wandb.Image(X_raw[idx], caption=f"{cls}: {names[cls]}"),
                cls, names[cls])
    wandb.log({"sample_images_per_class": table})
    print("  [W&B] Logged sample image table.")


def log_gradient_norms(model, step):
    """Layer gradient norms + per-neuron norms for layer 0."""
    log_dict = {}
    for i, layer in enumerate(model.layers):
        log_dict[f"layer_{i}_W - grad_norm"] = float(np.linalg.norm(layer.grad_W))
        log_dict[f"layer_{i}_b - grad_norm"] = float(np.linalg.norm(layer.grad_b))
    gW0 = model.layers[0].grad_W
    for n in range(min(5, gW0.shape[1])):
        log_dict[f"neuron_grad/layer0_neuron{n}"] = float(np.linalg.norm(gW0[:, n]))
    wandb.log(log_dict, step=step)


def log_activation_stats(model, X_sample, step):
    """Dead-neuron fraction + mean/std per hidden layer."""
    a = X_sample
    log_dict = {}
    for i, layer in enumerate(model.layers[:-1]):
        a = layer.forward(a)
        log_dict[f"layer_{i}_dead_frac activations"] = float(np.mean(a == 0))
        log_dict[f"layer_{i}_mean activations"]       = float(np.mean(a))
        log_dict[f"layer_{i}_std activations"]        = float(np.std(a))
    wandb.log(log_dict, step=step)


def log_confusion_matrix(y_true_oh, y_pred_probs, dataset_name):
    from sklearn.metrics import confusion_matrix

    FASHION_NAMES = ["T-shirt","Trouser","Pullover","Dress","Coat",
                     "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    MNIST_NAMES   = [str(i) for i in range(10)]
    names  = FASHION_NAMES if "fashion" in dataset_name else MNIST_NAMES
    y_true = np.argmax(y_true_oh,    axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Native W&B confusion matrix — renders as interactive heatmap, no slider
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=names,
        )
    })

    # Native W&B bar chart — renders as clean bar widget, no slider
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    bar_data = [[name, float(acc)] for name, acc in zip(names, per_class_acc)]
    wandb.log({
        "per_class_accuracy": wandb.plot.bar(
            wandb.Table(data=bar_data, columns=["class", "accuracy"]),
            "class",
            "accuracy",
            title="Per-class Accuracy",
        )
    })

    print("  [W&B] Logged confusion matrix & per-class accuracy.")


# ── Helpers for safe save ──────────────────────────────────────────────────

def _existing_best_f1(config_path: str) -> float:
    """Read best_val_f1 from an existing config file. Returns -1 if not found."""
    if not os.path.exists(config_path):
        return -1.0
    try:
        with open(config_path, "r") as f:
            return float(json.load(f).get("best_val_f1", -1.0))
    except Exception:
        return -1.0


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = parse_arguments()
    args   = parser.parse_args()

    # hidden_size comes in as a list due to nargs="+"

    try:
        # ── W&B init ──────────────────────────────────────────────────────────
        wandb_run = None
        if args.wandb_project:
            run_name = (f"{args.optimizer}-{args.activation}-"
                        f"{args.num_layers}L-{args.hidden_size}n-"
                        f"lr{args.learning_rate}")
            tags = args.tags.split(",") if args.tags else []
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=run_name,
                tags=tags,
                reinit=True,
            )
            # Allow sweep to override args
            for k, v in dict(wandb.config).items():
                if hasattr(args, k):
                    setattr(args, k, v)

        # ── Data ──────────────────────────────────────────────────────────────
        print(f"\nLoading dataset: {args.dataset} ...")
        (X_raw_tr, y_raw_tr), (X_raw_te, y_raw_te) = load_dataset(args.dataset)

        if wandb_run and args.log_images:
            log_sample_images(X_raw_tr, y_raw_tr, args.dataset)

        X_tr, y_tr, X_val, y_val, X_te, y_te = preprocess(
            X_raw_tr, y_raw_tr, X_raw_te, y_raw_te)

        args.input_size  = X_tr.shape[1]
        args.output_size = y_tr.shape[1]
        act_sample = X_tr[:256]

        print(f"  Train:{X_tr.shape[0]}  Val:{X_val.shape[0]}  Test:{X_te.shape[0]}")

        # ── Model ─────────────────────────────────────────────────────────────
        model = NeuralNetwork(args)
        print(f"  {model}")

        # ── Training loop ─────────────────────────────────────────────────────
        best_val_f1  = -1.0
        best_weights = None
        global_step  = 0

        for epoch in range(args.epochs):
            N   = X_tr.shape[0]
            idx = np.random.permutation(N)
            X_sh, y_sh = X_tr[idx], y_tr[idx]
            epoch_loss, num_batches = 0.0, 0

            for start in range(0, N, args.batch_size):
                end  = min(start + args.batch_size, N)
                Xb, yb = X_sh[start:end], y_sh[start:end]

                # forward() returns logits — backward() always expects logits
                logits = model.forward(Xb)
                probs  = model.predict_proba(logits)
                epoch_loss  += model.loss_fn(yb, probs)
                num_batches += 1

                model.backward(yb, logits)

                if wandb_run and args.log_gradients:
                    if global_step % args.grad_log_steps == 0:
                        log_gradient_norms(model, step=global_step)

                model.update_weights()
                global_step += 1

            if wandb_run and args.log_activations and epoch % 2 == 0:
                log_activation_stats(model, act_sample, step=global_step)

            # ── Epoch metrics ─────────────────────────────────────────────────
            avg_loss    = epoch_loss / num_batches
            train_acc   = model.evaluate(X_tr, y_tr)
            val_probs   = model.predict_proba(model.forward(X_val))
            val_metrics = compute_metrics(y_val, val_probs)

            print(
                f"Epoch [{epoch+1:>{len(str(args.epochs))}}/{args.epochs}] "
                f"Loss:{avg_loss:.4f}  TrainAcc:{train_acc:.4f}  "
                f"ValAcc:{val_metrics['accuracy']:.4f}  ValF1:{val_metrics['f1']:.4f}"
            )

            if wandb_run:
                wandb.log({
                    "epoch"     : epoch + 1,
                    "train_loss": avg_loss,
                    "train_acc" : train_acc,
                    "val_acc"   : val_metrics["accuracy"],
                    "val_f1"    : val_metrics["f1"],
                    "val_prec"  : val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                }, step=global_step)

            # Keep best weights in memory
            if val_metrics["f1"] > best_val_f1:
                best_val_f1  = val_metrics["f1"]
                best_weights = model.get_weights()

        # ── Final test evaluation ─────────────────────────────────────────────
        print("\nRestoring best weights for test evaluation...")
        model.set_weights(best_weights)

        test_probs   = model.predict_proba(model.forward(X_te))
        test_metrics = compute_metrics(y_te, test_probs)

        print(
            f"\n{'='*55}\n"
            f"  Test Accuracy  : {test_metrics['accuracy']:.4f}\n"
            f"  Test Precision : {test_metrics['precision']:.4f}\n"
            f"  Test Recall    : {test_metrics['recall']:.4f}\n"
            f"  Test F1-score  : {test_metrics['f1']:.4f}\n"
            f"{'='*55}"
        )

        if wandb_run:
            wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
            if args.log_conf_matrix:
                log_confusion_matrix(y_te, test_probs, args.dataset)
            wandb_run.finish()

        # ── Save best model and config — only if this run beat existing ────────
        existing_f1 = _existing_best_f1(args.save_config)

        if best_val_f1 > existing_f1:
            # Save weights
            os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
            np.save(args.save_model, best_weights)
            print(f"\nNew best model saved → '{args.save_model}' "
                f"(F1: {existing_f1:.4f} → {best_val_f1:.4f})")

            # Save config
            config = {
                **vars(args),
                "best_val_f1": best_val_f1,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            os.makedirs(os.path.dirname(args.save_config) or ".", exist_ok=True)
            with open(args.save_config, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Config saved → '{args.save_config}'")
        else:
            print(f"\nSkipping save — existing model is better "
                f"(saved F1={existing_f1:.4f} vs this run F1={best_val_f1:.4f})")
    except Exception as e:
        
        args_str = ", ".join(f"{k}={v}" for k, v in sorted(vars(args).items()))
        raise RuntimeError(f"{e} | args: {args_str}") from e


if __name__ == "__main__":
    main()
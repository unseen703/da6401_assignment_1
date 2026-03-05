"""
train.py - Training script for the MLP.
"""

import argparse
import json
import wandb
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_utils import load_dataset, preprocess
from utils.metrics import compute_metrics


#  CLI 
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a NumPy MLP on MNIST or Fashion-MNIST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-d",  "--dataset",       type=str,   default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",  "--epochs",        type=int,   default=35)
    parser.add_argument("-b",  "--batch_size",    type=int,   default=64)
    parser.add_argument("-l",  "--loss",          type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mean_squared_error"])
    parser.add_argument("-o",  "--optimizer",     type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay",  type=float, default=0.0)
    parser.add_argument("-nhl","--num_layers",    type=int,   default=3)
    parser.add_argument("-sz", "--hidden_size",   type=int,   default=128)
    parser.add_argument("-a",  "--activation",    type=str,   default="relu",
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",
                        choices=["random", "xavier", "zeros"])
    parser.add_argument("--save_model",  type=str, default="best_model.npy")
    parser.add_argument("--save_config", type=str, default="best_config.json")

    parser.add_argument("-w_p", "--wandb_project", type=str, default=None,
                        help="Weights & Biases project name.")

    parser.add_argument("--tags",            type=str,  default=None)
    parser.add_argument("--log_images",      action="store_true")
    parser.add_argument("--log_gradients",   action="store_true")
    parser.add_argument("--log_activations", action="store_true")
    parser.add_argument("--log_conf_matrix", action="store_true")
    parser.add_argument("--grad_log_steps",  type=int,  default=50)
    return parser


#  W&B logging 
def log_sample_images(X_raw, y_raw, dataset_name):
    """W&B Table with 5 images per class."""
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
        log_dict[f"layer0_neuron{n} - neuron_grad"] = float(np.linalg.norm(gW0[:, n]))
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
    """confusion matrix + per-class accuracy bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    FASHION_NAMES = ["T-shirt","Trouser","Pullover","Dress","Coat",
                     "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    MNIST_NAMES   = [str(i) for i in range(10)]
    names  = FASHION_NAMES if "fashion" in dataset_name else MNIST_NAMES
    y_true = np.argmax(y_true_oh,    axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm     = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title("Confusion Matrix – Best Model (Test Set)")
    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bars = ax2.bar(names, per_class_acc, color=plt.cm.RdYlGn(per_class_acc))
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Per-class Accuracy (Green = Good, Red = Problematic)")
    ax2.set_xticklabels(names, rotation=45, ha="right")
    for bar, acc in zip(bars, per_class_acc):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height()+0.01, f"{acc:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    wandb.log({"per_class_accuracy": wandb.Image(fig2)})
    plt.close(fig2)
    print("  [W&B] Logged confusion matrix.")


# ── Main ────────────
def main():
    parser = build_parser()
    args   = parser.parse_args()

    #  W&B init 
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

    # instanstiate Model 
    model = NeuralNetwork(args)
    print(f"  {model}")

    #  Training loop 
    best_val_f1  = -1.0
    best_weights = None
    global_step  = 0

    for epoch in range(args.epochs):
        N   = X_tr.shape[0]
        idx = np.random.permutation(N)
        X_sh, y_sh = X_tr[idx], y_tr[idx]
        epoch_loss, num_batches = 0.0, 0

        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            Xb, yb = X_sh[start:end], y_sh[start:end]

            # Forward — returns logits; predict_proba applies softmax
            logits = model.forward(Xb)
            probs  = model.predict_proba(Xb)

            epoch_loss  += model.loss_fn(yb, probs)
            num_batches += 1

            model.backward(yb, probs)

            if wandb_run and args.log_gradients:
                if global_step % args.grad_log_steps == 0:
                    log_gradient_norms(model, step=global_step)

            model.update_weights()
            global_step += 1

        if wandb_run and args.log_activations and epoch % 2 == 0:
            log_activation_stats(model, act_sample, step=global_step)

        # ── Epoch metrics ─────────────────────────────────────────────────
        avg_loss    = epoch_loss / num_batches
        train_acc   = model.evaluate(X_tr,  y_tr)
        val_probs   = model.predict_proba(X_val)
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

        # Save best weights in memory (by val F1)
        if val_metrics["f1"] > best_val_f1 :
            best_val_f1  = val_metrics["f1"]
            best_weights = model.get_weights()

    # ── Final test evaluation ─────────────────────────────────────────────
    if not args.tags:
        print("\nRestoring best model weights for test evaluation...")
        model.set_weights(best_weights)
        np.save(args.save_model, best_weights)
        print(f"  Best model saved → '{args.save_model}'")

    test_probs   = model.predict_proba(X_te)
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


    # Save config
    if not args.tags:
        config = {
            **vars(args),
            "best_val_f1": best_val_f1,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }

        with open(args.save_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved → '{args.save_config}'")


if __name__ == "__main__":
    main()
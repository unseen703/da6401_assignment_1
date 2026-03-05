"""
experiments.py - Run all W&B experiments.

Each function is named after its experiment description.

Run all:            python experiments.py --project wandb_report --all
Run one experiment: python experiments.py --project wandb_report --exp optimizer_showdown

Available experiments:
  sample_image_table
  optimizer_showdown
  vanishing_gradient
  dead_neuron_investigation
  loss_function_comparison
  confusion_matrix_analysis
  weight_init_symmetry
  fashion_mnist_transfer
"""

import argparse
import subprocess
import sys


BASE_ARGS = [sys.executable, "train.py"]

def run(project: str, extra_flags: list, label: str = "", tags: list = None):
    cmd = BASE_ARGS + extra_flags + ["-w_p", project]
    if tags:
        cmd += ["--tags", ",".join(tags)]
    if label:
        print(f"\n{'─'*60}")
        print(f"  Running: {label}")
        print(f"{'─'*60}")
    subprocess.run(cmd, check=True)


def base_flags(optimizer="rmsprop", activation="relu", num_layers=3,
               hidden_size=128, epochs=10, lr=0.001, batch_size=64,
               loss="cross_entropy", weight_init="xavier", dataset="mnist"):
    return [
        "-d",   dataset,
        "-e",   str(epochs),
        "-b",   str(batch_size),
        "-l",   loss,
        "-o",   optimizer,
        "-lr",  str(lr),
        "-wd",  "0.0",
        "-nhl", str(num_layers),
        "-sz",  str(hidden_size),
        "-a",   activation,
        "-w_i",  weight_init,
    ]


# Log 5 sample images per class to a W&B Table
def sample_image_table(project):
    """
    Logs a W&B Table with 5 images from each of the 10 classes.
    Run this once to populate the Data Exploration section of the report.
    """
    run(project,
        base_flags(epochs=1) + ["--log_images"],
        label="Sample Images per Class",
        tags=["sample_image_table"])


# Compare all 4 optimizers under identical conditions

def optimizer_showdown(project):
    """
    Trains 4 separate runs — one per optimizer (SGD, Momentum, NAG, RMSProp)
    with identical architecture: 3 hidden layers, 128 neurons, ReLU, lr=0.001.
    Compare train_loss curves over first 5 epoches.
    """
    for opt in ["sgd", "momentum", "nag", "rmsprop"]:
        run(project,
            base_flags(optimizer=opt, num_layers=3, hidden_size=128,
                       activation="relu", epochs=10, lr=0.001),
            label=f"Optimizer Showdown: {opt.upper()}",
            tags=["optimizer_showdown", opt])


# Vanishing gradient: Sigmoid vs ReLU, optimizer = RMSProp

def vanishing_gradient(project):
    """
    Optimizer fixed to RMSProp. Compares Sigmoid vs ReLU at 2-layer
    and 4-layer depths. Logs grad_norm/layer_0_W every 50 steps.
    """
    configs = [
        ("sigmoid", 2, "Sigmoid-2layers"),
        ("sigmoid", 4, "Sigmoid-4layers"),
        ("relu",    2, "ReLU-2layers"),
        ("relu",    4, "ReLU-4layers"),
    ]
    for act, nhl, label in configs:
        run(project,
            base_flags(optimizer="rmsprop", activation=act, num_layers=nhl,
                       hidden_size=128, epochs=10, lr=0.001)
            + ["--log_gradients", "--grad_log_steps", "50"],
            label=f"Vanishing Gradient: {label}",
            tags=["vanishing_gradient", act])


# Dead neuron detection: ReLU high LR vs Tanh, RMSProp

def dead_neuron_investigation(project):
    """
    Optimizer fixed to RMSProp. Uses ReLU + high LR (0.1) to trigger
    dead neurons. Compares:
      - ReLU at lr=0.1  (dead neurons expected)
      - ReLU at lr=0.01 (healthy baseline)
      - Tanh at lr=0.1  (no dead neurons — tanh never outputs exactly 0)
    Logs activations/layer_N_dead_frac every 2 epochs.
    """
    configs = [
        ("relu", 0.1,  "ReLU-HighLR-0.1"),
        ("relu", 0.01, "ReLU-NormalLR-0.01"),
        ("tanh", 0.1,  "Tanh-HighLR-0.1"),
    ]
    for act, lr, label in configs:
        run(project,
            base_flags(optimizer="rmsprop", activation=act, num_layers=3,
                       hidden_size=128, epochs=10, lr=lr)
            + ["--log_activations", "--log_gradients"],
            label=f"Dead Neuron Investigation: {label}",
            tags=["dead_neuron_investigation", act])


 # Cross-Entropy vs MSE, optimizer = RMSProp

def loss_function_comparison(project):
    """
    Train two identical models (Same architecture, same LR, same optimizer (RMSProp)).
    Only the loss function differs.
    """
    for loss in ["cross_entropy", "mean_squared_error"]:
        run(project,
            base_flags(loss=loss, optimizer="rmsprop", activation="relu",
                       num_layers=3, hidden_size=128, epochs=10, lr=0.001),
            label=f"Loss Comparison: {loss}",
            tags=["loss_function_comparison", loss])

#  Best model error analysis, confusion matrix

def confusion_matrix_analysis(project):
    """
    Runs the best-performing configuration for 15 epochs and logs:
      - Confusion matrix heatmap
      - Per-class accuracy bar chart (red → green coloured)
    Optimizer: RMSProp.
    """
    run(project,
        base_flags(optimizer="rmsprop", activation="relu", num_layers=3,
                   hidden_size=128, epochs=15, lr=0.001, weight_init="xavier")
        + ["--log_conf_matrix"],
        label="Confusion Matrix & Per-class Accuracy",
        tags=["confusion_matrix_analysis"])


# Zeros vs Xavier weight initialisation: symmetry breaking

def weight_init_symmetry(project):
    """
    Compares zeros init vs Xavier init. Optimizer fixed to RMSProp.
    Logs neuron_grad/layer0_neuron0-4 at EVERY step (grad_log_steps=1).
    Zeros run: all 5 neuron gradient lines overlap perfectly — symmetry problem.
    Xavier run: 5 distinct lines — symmetry broken, each neuron learns differently.
    """
    for init in ["zeros", "xavier"]:
        run(project,
            base_flags(optimizer="rmsprop", activation="relu", num_layers=3,
                       hidden_size=128, epochs=50, lr=0.001, weight_init=init)
            + ["--log_gradients", "--grad_log_steps", "1"],
            label=f"Weight Init Symmetry: {init}",
            tags=["weight_init_symmetry", init])


# Fashion-MNIST transfer: top 3 configs from MNIST learnings

def fashion_mnist_transfer(project):
    """
    Applies the top 3 configurations discovered from MNIST experiments
    to Fashion-MNIST with a limited compute budget (3 runs only).

    Config choices based on MNIST learnings:
      1. RMSProp + ReLU + Xavier 3L — best overall on MNIST
      2. RMSProp + ReLU + Xavier 4L — deeper for harder dataset
      3. RMSProp + Tanh + Xavier 3L — Tanh avoids dead neurons
    """
    configs = [
        dict(opt="nag", act="relu", nhl=4, sz=128, lr=0.001, w_i="xavier",
             label="NAG-ReLU-Xavier-4L-128-fashion-mnist"),
        dict(opt="rmsprop", act="relu", nhl=4, sz=128, lr=0.001, w_i="xavier",
             label="RMSProp-ReLU-Xavier-4L-128-fashion-mnist"),
        dict(opt="rmsprop", act="relu", nhl=3, sz=128, lr=0.001, w_i="xavier",
             label="RMSProp-ReLU-Xavier-3L-128-fashion-mnist")
    ]
    for c in configs:
        run(project,
            base_flags(optimizer=c["opt"], activation=c["act"],
                       num_layers=c["nhl"], hidden_size=c["sz"],
                       lr=c["lr"], weight_init=c["w_i"],
                       dataset="fashion_mnist", epochs=35, batch_size=c.get("b", 64)),
            label=f"Fashion-MNIST Transfer: {c['label']}",
            tags=["fashion_mnist_transfer", c["opt"]])


# ─────────────────────────────────────────────────────────────────
# Registry & entry point
# ─────────────────────────────────────────────────────────────────

ALL_EXPERIMENTS = {
    "sample_image_table":        sample_image_table,
    "optimizer_showdown":        optimizer_showdown,
    "vanishing_gradient":        vanishing_gradient,
    "dead_neuron_investigation": dead_neuron_investigation,
    "loss_function_comparison":  loss_function_comparison,
    "confusion_matrix_analysis": confusion_matrix_analysis,
    "weight_init_symmetry":      weight_init_symmetry,
    "fashion_mnist_transfer":    fashion_mnist_transfer,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run W&B experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", type=str, required=True,
                        help="W&B project name (e.g. da6401_a1)")
    parser.add_argument("--exp",     type=str, default=None,
                        choices=list(ALL_EXPERIMENTS.keys()),
                        metavar="EXPERIMENT",
                        help="Name of experiment to run.")
    parser.add_argument("--all",     action="store_true",
                        help="Run all experiments sequentially.")
    args = parser.parse_args()

    if args.all:
        for name, fn in ALL_EXPERIMENTS.items():
            print(f"\n{'='*60}")
            print(f"  EXPERIMENT: {name}")
            print(f"{'='*60}")
            fn(args.project)
    elif args.exp:
        ALL_EXPERIMENTS[args.exp](args.project)
    else:
        parser.print_help()
        print("\nAvailable experiments:")
        for name in ALL_EXPERIMENTS:
            print(f"  --exp {name}")


if __name__ == "__main__":
    main()
# DA6401 Assignment 1 — Multi-Layer Perceptron for Image Classification

A modular, configurable **NumPy-only** MLP implementation for classifying MNIST and Fashion-MNIST images.


🔗 **W&B Report**[https://wandb.ai/dipakkanzariya702-iitmaana/wandb_report/reports/DA6401-Assignment--VmlldzoxNjEwMjY0OQ?accessToken=jfks70tv55nzef88fk50u36qc14hedu95xs2wadli6umkv2yfww4bvwahj1xs941]
🔗 **GitHub Repository**[https://github.com/unseen703/da6401_assignment_1/tree/main]

---

## Project Structure

```
da6401_assignment_1/
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py         # Sigmoid, Tanh, ReLU, Softmax + derivatives
│   │   ├── neural_layer.py        # NeuralLayer class (forward / backward / gradients)
│   │   ├── neural_network.py      # NeuralNetwork class (train / evaluate / save / load)
│   │   ├── objective_functions.py # Cross-entropy and MSE losses + gradients
│   │   └── optimizers.py          # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   └── utils/
│   |    ├── __init__.py
│   |   ├── data_utils.py          # Dataset loading, normalization, one-hot encoding
│   |   └── metrics.py             # Accuracy, Precision, Recall, F1-score
|   |
|   ├── train.py                       # Training entry point (argparse CLI)
|   ├── inference.py                   # Inference entry point (loads .npy weights)
|   ├── best_model.npy                 # Saved best model weights (generated after training)
|   └── best_config.json               # Best model configuration (generated after training)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/unseen703/da6401_assignment_1/tree/main
cd da6401_assignment_1
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### 4. Login to Weights & Biases

```bash
wandb login 
```

---

## Training

Run `train.py` with CLI arguments. All arguments have defaults matching the best found configuration.

### Full CLI reference

| Flag | Long form | Default | Description |
|------|-----------|---------|-------------|
| `-d` | `--dataset` | `mnist` | `mnist` or `fashion_mnist` |
| `-e` | `--epochs` | `10` | Number of training epochs |
| `-b` | `--batch_size` | `64` | Mini-batch size |
| `-l` | `--loss` | `cross_entropy` | `cross_entropy` or `mean_squared_error` |
| `-o` | `--optimizer` | `rmsprop` | `sgd`, `momentum`, `nag`, `rmsprop` |
| `-lr` | `--learning_rate` | `0.001` | Learning rate |
| `-wd` | `--weight_decay` | `0.0` | L2 regularization coefficient |
| `-nhl` | `--num_layers` | `3` | Number of hidden layers |
| `-sz` | `--hidden_size` | `128` | Neurons per hidden layer |
| `-a` | `--activation` | `relu` | `sigmoid`, `tanh`, `relu` |
| `-w_i` | `--weight_init` | `xavier` | `random`, `xavier`, `zeros` |
| `-w_p` | `--wandb_project` | `None` | W&B project name |

### Basic training run

```bash
python train.py \
  -d mnist \
  -e 10 \
  -b 64 \
  -l cross_entropy \
  -o rmsprop \
  -lr 0.001 \
  -wd 0.0 \
  -nhl 3 \
  -sz 128 \
  -a relu \
  -w_i xavier \
  -w_p your_wandb_project
```

### Training with W&B logging enabled

```bash
# Log sample images (Data Exploration)
python train.py -d mnist -e 1 -w_p your_wandb_project --log_images

# Log gradient norms every 50 steps (Vanishing Gradient analysis)
python train.py -d mnist -e 10 -w_p your_wandb_project --log_gradients --grad_log_steps 50

# Log dead neuron fraction every 2 epochs
python train.py -d mnist -e 10 -o rmsprop -lr 0.1 -a relu -w_p your_wandb_project --log_activations

# Log confusion matrix and per-class accuracy on test set
python train.py -d mnist -e 15 -w_p your_wandb_project --log_conf_matrix
```

### Training on Fashion-MNIST

```bash
python train.py \
  -d fashion_mnist \
  -e 15 \
  -o rmsprop \
  -lr 0.001 \
  -nhl 3 \
  -sz 128 \
  -a relu \
  -w_i xavier \
  -w_p your_wandb_project
```

> **Note:** The best model weights are saved to `src/best_model.npy` and configuration to `src/best_config.json`. These files are only overwritten if a new run achieves a higher validation F1-score than the existing saved model.

---

## Inference

`inference.py` automatically loads `src/best_config.json` to reconstruct the exact architecture used during training, then loads `src/best_model.npy` weights.

```bash
# Run on MNIST using saved best model (default)
python inference.py

# Run on Fashion-MNIST
python inference.py -d fashion_mnist

# Custom model path
python inference.py --model_path src/best_model.npy
```

**Output:**
```
Accuracy  : 0.9800
Precision : 0.9799
Recall    : 0.9798
F1-score  : 0.9799
```

---

## Hyperparameter Sweep

Runs W&B sweep over 100+ configurations varying optimizer, learning rate, batch size, depth, width, activation, and weight initialisation.

```bash
python sweep.py --project your_wandb_project --count 100
```

> **Tip:** This takes 1–3 hours. Run it first so it executes in the background while you run other experiments.

**Parameters swept:**

| Parameter | Values |
|-----------|--------|
| `epochs` | 25, 35 |
| `optimizer` | sgd, momentum, nag, rmsprop |
| `learning_rate` | log-uniform [1e-4, 1e-1] |
| `weight_decay` | log-uniform [1e-5, 1e-2] |
| `batch_size` | 32, 64, 128 |
| `num_layers` | 1, 2, 3, 4, 5, 6 |
| `hidden_size` | 32, 64, 128 |
| `activation` | sigmoid, tanh, relu |
| `weight_init` | random, xavier |

---

## W&B Report Experiments

`experiments.py` runs all experiments needed for the W&B report. Each experiment is tagged automatically so you can filter runs cleanly in the report.

### Run a single experiment

```bash
python experiments.py --project your_wandb_project --exp <experiment_name>
```

### Run all experiments sequentially

```bash
python experiments.py --project your_wandb_project --all
```

### Available experiments

| Experiment name | What it does | W&B tag to filter by |
|-----------------|--------------|----------------------|
| `sample_image_table` | Logs W&B Table with 5 images per class | `sample_image_table` |
| `optimizer_showdown` | Compares SGD, Momentum, NAG, RMSProp | `optimizer_showdown` |
| `vanishing_gradient` | Sigmoid vs ReLU at 2 and 4 layers, logs grad norms | `vanishing_gradient` |
| `dead_neuron_investigation` | ReLU high LR vs ReLU normal LR vs Tanh | `dead_neuron_investigation` |
| `loss_function_comparison` | Cross-Entropy vs MSE convergence | `loss_function_comparison` |
| `confusion_matrix_analysis` | Best model confusion matrix + per-class accuracy | `confusion_matrix_analysis` |
| `weight_init_symmetry` | Zeros vs Xavier, logs per-neuron gradients | `weight_init_symmetry` |
| `fashion_mnist_transfer` | Top 3 MNIST configs applied to Fashion-MNIST | `fashion_mnist_transfer` |

### Individual experiment commands

```bash
# Sample images table
python experiments.py --project your_wandb_project --exp sample_image_table

# Optimizer comparison (4 runs)
python experiments.py --project your_wandb_project --exp optimizer_showdown

# Vanishing gradient (4 runs: Sigmoid/ReLU × 2/4 layers)
python experiments.py --project your_wandb_project --exp vanishing_gradient

# Dead neuron investigation (3 runs)
python experiments.py --project your_wandb_project --exp dead_neuron_investigation

# Loss function comparison (2 runs)
python experiments.py --project your_wandb_project --exp loss_function_comparison

# Confusion matrix on best model
python experiments.py --project your_wandb_project --exp confusion_matrix_analysis

# Weight initialisation symmetry (2 runs)
python experiments.py --project your_wandb_project --exp weight_init_symmetry

# Fashion-MNIST transfer (3 runs)
python experiments.py --project your_wandb_project --exp fashion_mnist_transfer
```

---

## Recommended Execution Order

```bash
# Step 1 — Login
wandb login

# Step 2 — Quick sanity check (1 epoch, no W&B)
python train.py -e 1

# Step 3 — Start sweep first (takes longest)
python sweep.py --project your_wandb_project --count 100

# Step 4 — Run all experiments while sweep is running
python experiments.py --project your_wandb_project --exp sample_image_table
python experiments.py --project your_wandb_project --exp optimizer_showdown
python experiments.py --project your_wandb_project --exp vanishing_gradient
python experiments.py --project your_wandb_project --exp dead_neuron_investigation
python experiments.py --project your_wandb_project --exp loss_function_comparison
python experiments.py --project your_wandb_project --exp confusion_matrix_analysis
python experiments.py --project your_wandb_project --exp weight_init_symmetry
python experiments.py --project your_wandb_project --exp fashion_mnist_transfer

# Step 5 — Evaluate best saved model
python inference.py
```

---
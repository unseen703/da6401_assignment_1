"""
utils/data_utils.py - Data loading, preprocessing, and splitting utilities.

Loads MNIST or Fashion-MNIST via keras.datasets, normalizes pixel values,
one-hot encodes labels, and performs a train/validation split.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(name: str):
    """
    Load MNIST or Fashion-MNIST dataset.

    Args:
        name: 'mnist' or 'fashion_mnist'.

    Returns:
        (X_train, y_train), (X_test, y_test)
        X arrays are float32, y arrays are int32.
    """
    name = name.lower().replace("-", "_")
    if name == "mnist":
        # New
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Use 'mnist' or 'fashion_mnist'.")

    return (X_train, y_train), (X_test, y_test)


def preprocess(X_train, y_train, X_test, y_test,
               num_classes: int = 10, val_size: float = 0.1,
               random_state: int = 42):
    """
    Normalize images, flatten, one-hot encode labels, and split train/val.

    Args:
        X_train     : Raw training images, shape (N, 28, 28), uint8.
        y_train     : Integer training labels, shape (N,).
        X_test      : Raw test images,     shape (M, 28, 28), uint8.
        y_test      : Integer test labels,  shape (M,).
        num_classes : Number of output classes (default 10).
        val_size    : Fraction of training data reserved for validation.
        random_state: Random seed for reproducibility.

    Returns:
        X_tr, y_tr, X_val, y_val, X_te, y_te
        All X arrays have shape (N, 784), normalized to [0, 1].
        All y arrays are one-hot encoded, shape (N, num_classes).
    """
    # Flatten and normalize
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_test  = X_test.reshape(X_test.shape[0],   -1).astype(np.float32) / 255.0

    # Train / validation split
    X_tr, X_val, y_tr_int, y_val_int = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train
    )

    # One-hot encode
    y_tr  = one_hot(y_tr_int,  num_classes)
    y_val = one_hot(y_val_int, num_classes)
    y_te  = one_hot(y_test,    num_classes)

    return X_tr, y_tr, X_val, y_val, X_test, y_te


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer class labels to one-hot encoding.

    Args:
        y          : Integer labels, shape (N,).
        num_classes: Number of classes.

    Returns:
        One-hot matrix, shape (N, num_classes).
    """
    N = y.shape[0]
    oh = np.zeros((N, num_classes), dtype=np.float32)
    oh[np.arange(N), y] = 1.0
    return oh

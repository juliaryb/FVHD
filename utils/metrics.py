import ssl
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def compute_cf_nn(X_low_dim: np.ndarray, y: np.ndarray, nn: int) -> float:
    """
    Compute cf_nn for a given number of neighbors (nn).

    Parameters:
    - X_low_dim: np.array of shape (M, d), the reduced-dimensional embedding.
    - y: np.array of shape (M,), the class labels.
    - nn: int, number of nearest neighbors to consider.

    Returns:
    - cf_nn: float, class fidelity for the given nn.
    """
    M = X_low_dim.shape[0]
    neigh = NearestNeighbors(n_neighbors=nn + 1)
    neigh.fit(X_low_dim)
    _, indices = neigh.kneighbors(X_low_dim)

    # Ignore the first column (the point itself)
    indices = indices[:, 1:]

    same_class_counts = np.sum(y[indices] == y[:, None], axis=1)
    cf_nn = np.sum(same_class_counts) / (nn * M)
    return cf_nn

def compute_cf(X_low_dim: np.ndarray, y: np.ndarray, nn_max: int = 30) -> tuple[list[float], float]:
    """
    Compute overall class fidelity (cf) by averaging cf_nn from nn=1 to nn_max.

    Parameters:
    - X_low_dim: np.array of shape (M, d)
    - y: np.array of shape (M,)
    - nn_max: int

    Returns:
    - cf_values: list of cf_nn values
    - cf: float, average class fidelity score
    """
    cf_values = [compute_cf_nn(X_low_dim, y, nn) for nn in range(1, nn_max + 1)]
    return cf_values, float(np.mean(cf_values))

def visualize_cf(cf_values, method_name=""):
    """
    Plot the cf_nn values over the range of neighbors.

    Parameters:
    - cf_values: list of float, each entry is cf_nn
    - method_name: str, label for the title
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cf_values) + 1), cf_values, marker="o")
    plt.xlabel("Number of Nearest Neighbors (nn)")
    plt.ylabel("Class Fidelity $cf_{nn}$")
    plt.title(f"Class Fidelity Curve – {method_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualise_cf_scores(x: np.ndarray, y: torch.Tensor, dataset_name: str, nn_max: int = 30) -> Tuple[list, float]:
    """
    Computes and plots class fidelity curve from the 2D embedding and labels.

    Parameters:
    - x: np.ndarray (n_samples, 2) — low-dim embedding
    - y: torch.Tensor or np.ndarray — class labels
    - dataset_name: str — used for title
    - nn_max: int — maximum number of neighbours to test

    Returns:
    - float — average class fidelity score
    """
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    cf_values, cf_avg = compute_cf(x, y, nn_max=nn_max)
    visualize_cf(cf_values, method_name=f"{dataset_name} (avg CF={cf_avg:.2f})")
    return cf_values, cf_avg

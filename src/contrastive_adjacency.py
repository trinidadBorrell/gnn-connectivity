"""Anatomical weighted-KNN adjacency for the contrastive graph encoder.

Builds a dense (N, N) symmetric-normalised adjacency from electrode (x, y, z)
coordinates only — no wSMI information leaks into the graph structure.

Edge weight: exp(-d_ij^2 / (2 sigma^2)) for the top-K nearest neighbours of
each node (sigma defaults to the median nearest-neighbour distance).
Symmetric normalisation with self-loops (standard GCN convention):

    A_hat = (A + I)
    D = diag(sum_j A_hat_ij)
    A_norm = D^{-1/2} * A_hat * D^{-1/2}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def load_coords(coords_file: str) -> np.ndarray:
    """Read the GSN-HydroCel-257 layout file -> (256, 3) XYZ array."""
    df = pd.read_csv(coords_file, sep=r'\s+', header=None,
                     names=['label', 'x', 'y', 'z'])
    return df[['x', 'y', 'z']].to_numpy(dtype=np.float32)


def weighted_knn_adjacency(coords: np.ndarray, k: int = 10,
                           sigma: float | None = None) -> torch.Tensor:
    """Return the (N, N) symmetric-normalised adjacency for the K-NN graph.

    Args:
        coords:  (N, 3) electrode XYZ.
        k:       neighbourhood size.
        sigma:   RBF kernel width. If None, set to the median k-nearest
                 distance across all nodes.
    Returns:
        (N, N) float32 torch tensor, symmetric, ready to use as the dense
        GCN propagation matrix.
    """
    n = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]            # (N, N, 3)
    D = np.linalg.norm(diff, axis=-1)                         # (N, N)
    np.fill_diagonal(D, np.inf)

    # Top-K neighbours per node + their distances
    nn_idx = np.argsort(D, axis=1)[:, :k]                     # (N, K)
    nn_dist = np.take_along_axis(D, nn_idx, axis=1)           # (N, K)
    if sigma is None:
        sigma = float(np.median(nn_dist))

    # Sparse A with RBF weights, symmetrised
    A = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.flatten()
    w = np.exp(-nn_dist.flatten() ** 2 / (2 * sigma ** 2))
    A[rows, cols] = w
    A = np.maximum(A, A.T)                                    # union, keeps max weight

    # Add self-loops, symmetric normalise (standard GCN)
    A = A + np.eye(n, dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
    A_norm = A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    return torch.tensor(A_norm, dtype=torch.float32)

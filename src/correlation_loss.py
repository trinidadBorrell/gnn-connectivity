"""
CORRELATION-PRESERVING REGULARIZER
==================================
Differentiable Pearson correlation matrix and the corr-MSE loss used as a
regularizer in time-series mode. Pure PyTorch, runs on the same device as the
input tensors.

Used by `train.train_one_epoch` when `corr_lambda > 0` so the reconstruction
is pushed to preserve channel-by-channel covariance structure, not just
per-sample MSE.
"""
import torch
import torch.nn.functional as F


def pearson_correlation_matrix(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-graph Pearson correlation between channels.

    Args:
        x: shape (..., n_channels, n_timepoints).
        eps: floor on per-channel std to avoid divide-by-zero on flat channels.

    Returns:
        corr: shape (..., n_channels, n_channels).
    """
    x_centered = x - x.mean(dim=-1, keepdim=True)
    std = x_centered.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    x_norm = x_centered / std
    T = x.shape[-1]
    return torch.matmul(x_norm, x_norm.transpose(-1, -2)) / T


def corr_loss(x: torch.Tensor, x_recon: torch.Tensor,
              num_graphs: int) -> torch.Tensor:
    """MSE between Pearson correlation matrices of `x` and `x_recon`.

    Assumes a PyG-style batched tensor where all graphs have the same node
    count, i.e. `x.shape == (num_graphs * n_channels, n_timepoints)`. The
    biosemi64 dataset satisfies this (every graph has 64 nodes in fixed
    order), so the reshape is unambiguous.
    """
    n_per = x.shape[0] // num_graphs
    x_r = x.view(num_graphs, n_per, -1)
    xr_r = x_recon.view(num_graphs, n_per, -1)
    c_target = pearson_correlation_matrix(x_r)
    c_recon = pearson_correlation_matrix(xr_r)
    return F.mse_loss(c_recon, c_target)

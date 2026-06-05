"""wSMI augmentation pipeline for contrastive pretraining.

Input convention: x is a tensor of node-feature matrices,
shape (B, N, F) or (N, F), where:
  N = 256 nodes (electrodes)
  F = 256 features per node (each node's row of the wSMI matrix)

All augmentations operate elementwise on x and preserve the matrix's
symmetry where it matters (Gaussian noise, edge mask). Self-loops on the
graph are decoupled — they live in the adjacency, not in x.
"""
from __future__ import annotations
import torch


def _symmetrise(t: torch.Tensor) -> torch.Tensor:
    """Make the last-two dims symmetric: (.., N, N) -> mean with its transpose."""
    return 0.5 * (t + t.transpose(-1, -2))


def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """Add symmetric Gaussian noise. Preserves wSMI symmetry."""
    noise = _symmetrise(torch.randn_like(x) * std)
    return x + noise


def edge_mask(x: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out p fraction of off-diagonal entries (symmetric mask)."""
    mask = (torch.rand_like(x) > p).float()
    mask = (_symmetrise(mask) > 0.5).float()                # symmetrise + threshold
    return x * mask


def node_mask(x: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out p fraction of entire node-feature rows AND their column
    counterparts (so the matrix stays symmetric)."""
    if x.dim() == 3:
        B, N, _ = x.shape
        keep = (torch.rand(B, N, device=x.device) > p).float()   # (B, N)
    else:
        N = x.shape[0]
        keep = (torch.rand(N, device=x.device) > p).float()
        keep = keep.unsqueeze(0)                                  # (1, N)
    # Outer product: keep[i,j] = keep_i * keep_j
    row = keep.unsqueeze(-1)                                      # (..., N, 1)
    col = keep.unsqueeze(-2)                                      # (..., 1, N)
    return x * (row * col).squeeze(0) if x.dim() == 2 else x * (row * col)


def amplitude_scale(x: torch.Tensor, max_pct: float) -> torch.Tensor:
    """Multiply each example by 1 + delta with delta ~ U(-max_pct, max_pct)."""
    if x.dim() == 3:
        s = 1.0 + (torch.rand(x.shape[0], 1, 1, device=x.device) - 0.5) * 2 * max_pct
    else:
        s = 1.0 + (torch.rand(1, device=x.device) - 0.5) * 2 * max_pct
    return x * s


def augment(x: torch.Tensor,
            noise_std: float = 0.02,
            edge_p: float = 0.05,
            node_p: float = 0.10,
            scale_pct: float = 0.05) -> torch.Tensor:
    """The default contrastive augmentation pipeline. Apply with two
    independent random draws to get the two views of each batch."""
    x = gaussian_noise(x, noise_std)
    x = edge_mask(x, edge_p)
    x = node_mask(x, node_p)
    x = amplitude_scale(x, scale_pct)
    return x

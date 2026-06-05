"""Minimal MoCo-v2 wrapper for graph-encoded wSMI.

Encoder = stack of dense GCN layers operating on the fixed (N, N) anatomical
adjacency. Output is one scalar per node → 256-D embedding per epoch when
flattened. Projector is an MLP used during training only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- encoder
class DenseGCNLayer(nn.Module):
    """Standard GCN propagation step on a dense, fixed, symmetric adjacency:
        y = A_norm @ (x @ W) + b
    A_norm is a buffer (no grad), shared across all examples in the batch.
    """
    def __init__(self, in_dim: int, out_dim: int, A_norm: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.register_buffer('A_norm', A_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim)
        x = self.linear(x)               # (B, N, out_dim)
        x = self.A_norm @ x              # message passing on every example
        return x


class Encoder(nn.Module):
    """SAGEConv-style stack ending at out_dim features per node.
    Outputs (B, N) when out_dim=1, (B, N, out_dim) otherwise."""
    def __init__(self, A_norm: torch.Tensor,
                 in_dim: int = 256,
                 hidden_dims: tuple = (128, 64, 32, 16),
                 out_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        self.convs = nn.ModuleList([
            DenseGCNLayer(dims[i], dims[i + 1], A_norm) for i in range(len(dims) - 1)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(d) for d in hidden_dims])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            x = self.norms[i](x.reshape(B * N, -1)).reshape(B, N, -1)
            x = F.leaky_relu(x, 0.1)
            x = self.dropout(x)
        x = self.convs[-1](x)                # (B, N, out_dim)
        if self.out_dim == 1:
            x = x.squeeze(-1)                # (B, N)
        return x


# ------------------------------------------------------------------- projector
class Projector(nn.Module):
    """MLP head for InfoNCE: 256 → 128 → 128, L2-normalised output."""
    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ------------------------------------------------------------------ MoCo wrapper
class MoCo(nn.Module):
    def __init__(self,
                 A_norm: torch.Tensor,
                 in_dim: int = 256,
                 hidden_dims: tuple = (128, 64, 32, 16),
                 latent_per_node: int = 1,
                 proj_hidden: int = 128,
                 proj_out: int = 128,
                 queue_size: int = 8192,
                 momentum: float = 0.999,
                 temperature: float = 0.07,
                 dropout: float = 0.2):
        super().__init__()
        self.K = queue_size
        self.m = momentum
        self.T = temperature

        # Per-graph embedding dim after flatten = N * latent_per_node
        # (default 256 * 1 = 256)
        N = A_norm.shape[0]
        embed_dim = N * latent_per_node

        self.encoder_q = Encoder(A_norm, in_dim, hidden_dims, latent_per_node, dropout)
        self.encoder_k = Encoder(A_norm, in_dim, hidden_dims, latent_per_node, dropout)
        self.projector_q = Projector(embed_dim, proj_hidden, proj_out)
        self.projector_k = Projector(embed_dim, proj_hidden, proj_out)

        for p, q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            p.data.copy_(q.data); p.requires_grad = False
        for p, q in zip(self.projector_k.parameters(), self.projector_q.parameters()):
            p.data.copy_(q.data); p.requires_grad = False

        self.register_buffer('queue', F.normalize(torch.randn(self.K, proj_out), dim=1))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for p, q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            p.data.mul_(self.m).add_(q.data, alpha=1 - self.m)
        for p, q in zip(self.projector_k.parameters(), self.projector_q.parameters()):
            p.data.mul_(self.m).add_(q.data, alpha=1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        B = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        # Wrap-around enqueue
        if ptr + B <= self.K:
            self.queue[ptr:ptr + B] = keys
        else:
            tail = self.K - ptr
            self.queue[ptr:] = keys[:tail]
            self.queue[:B - tail] = keys[tail:]
        self.queue_ptr[0] = (ptr + B) % self.K

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened per-graph embedding (B, N * latent_per_node)."""
        h = self.encoder_q(x)
        return h.reshape(h.shape[0], -1)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor):
        """Compute InfoNCE loss for a pair of augmented views.

        x_q, x_k: (B, N, F) — two augmentations of the same batch.
        Returns:
            loss        : scalar InfoNCE
            q_embed     : (B, embed_dim) — pre-projection backbone output
                          (for downstream diagnostics)
        """
        # Query side
        h_q = self.encoder_q(x_q)                       # (B, N) when latent=1
        q_embed = h_q.reshape(h_q.shape[0], -1)         # (B, N*latent)
        q = self.projector_q(q_embed)                   # (B, proj_out)

        # Key side: no grad, momentum-updated
        with torch.no_grad():
            self._momentum_update()
            h_k = self.encoder_k(x_k)
            k_embed = h_k.reshape(h_k.shape[0], -1)
            k = self.projector_k(k_embed)               # (B, proj_out)

        # InfoNCE: positive = (q, k), negatives = queue.
        # Clone the queue: otherwise the in-place _dequeue_and_enqueue
        # below mutates the storage that autograd kept for the matmul.
        l_pos = (q * k).sum(-1, keepdim=True)           # (B, 1)
        l_neg = q @ self.queue.clone().T                # (B, K) — clone is critical
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)

        self._dequeue_and_enqueue(k.detach())
        return loss, q_embed

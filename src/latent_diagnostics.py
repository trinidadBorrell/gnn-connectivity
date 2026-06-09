"""
Reconstruction + latent-space diagnostics for trained GAE / VGAE models.

Complements src/cluster_analysis.py (which is about cluster structure on the
graph-level latents). The functions here are about the model itself:

- per_edge_reconstruction_error: where in the (64, 64) connectivity matrix is
  reconstruction worst? Returns a 64x64 mean-squared-error tensor across graphs.
- per_subject_recon_mse: MSE per (subject, diagnosis_group) — surfaces
  outliers and group-level recon gaps.
- VGAE-only:
  - kl_per_dim: mean KL contribution of each latent dimension across the
    dataset. Useful for spotting dead/inactive dims.
  - posterior_collapse_fraction: fraction of latent dims with mean KL < tol.
  - mu_logvar_stats: per-dim summary of mu and log_var (sanity check).

All functions expect:
  model: trained GAE or VGAE (from src.model)
  graphs: list of torch_geometric.data.Data (with .x, .edge_index, plus the
          metadata attached by wsmi_loader)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

# Local import resolves once src/ is on sys.path (as in cookbook/run_wsmi_pipeline.py).
from model import GAEVAE, VGAE, kl_divergence  # noqa: E402

# Both VGAE and GAEVAE expose .encode() returning (mu, log_var) and forward()
# returning (x_recon, z, mu, log_var). Any function operating on those outputs
# accepts either class.
VARIATIONAL_CLASSES = (VGAE, GAEVAE)


@torch.no_grad()
def per_edge_reconstruction_error(
    model: torch.nn.Module,
    graphs: Sequence,
    device: Optional[torch.device] = None,
    batch_size: int = 64,  # kept for API compatibility; iteration is per-graph
) -> np.ndarray:
    """Average squared error per (node, feature) over all graphs.

    Returns (n_nodes, n_features) float64 array. For wSMI graphs where x is the
    64x64 connectivity matrix and each row/column is a node, this is the
    per-edge reconstruction error.

    Iterates one graph at a time to avoid PyG-DataLoader device-collation
    issues when training leaves graphs in mixed device state.
    """
    device = device or torch.device("cpu")
    model.eval()

    sums: Optional[np.ndarray] = None
    counts = 0
    for g in graphs:
        x = g.x.to(device)
        edge_index = g.edge_index.to(device)
        out = model(x, edge_index)
        recon = out[0]
        # MPS doesn't support float64; cast on CPU.
        se = ((recon - x) ** 2).detach().cpu().double().numpy()
        sums = se if sums is None else sums + se
        counts += 1

    if sums is None or counts == 0:
        return np.zeros((0, 0))
    return sums / counts


@torch.no_grad()
def per_subject_recon_mse(
    model: torch.nn.Module,
    graphs: Sequence,
    device: Optional[torch.device] = None,
    batch_size: int = 64,  # kept for API compatibility
    diagnosis_attr: str = "diagnosis_group",
) -> pd.DataFrame:
    """MSE per (subject, diagnosis_group), then aggregated.

    Returns a DataFrame with one row per subject and columns:
      subject_id, diagnosis_group, n_graphs, mse_mean, mse_std.
    """
    device = device or torch.device("cpu")
    model.eval()

    per_graph_mse: List[float] = []
    for g in graphs:
        x = g.x.to(device)
        edge_index = g.edge_index.to(device)
        out = model(x, edge_index)
        recon = out[0]
        mse = float(((recon - x) ** 2).mean().detach().cpu())
        per_graph_mse.append(mse)

    rows = []
    per_subject: Dict[str, List[float]] = defaultdict(list)
    diag_lookup: Dict[str, str] = {}
    for g, mse in zip(graphs, per_graph_mse):
        sub = getattr(g, "subject_id", "UNK")
        per_subject[sub].append(mse)
        diag_lookup[sub] = getattr(g, diagnosis_attr, "UNK")

    for sub, vals in per_subject.items():
        arr = np.asarray(vals, dtype=np.float64)
        rows.append({
            "subject_id": sub,
            "diagnosis_group": diag_lookup[sub],
            "n_graphs": int(arr.size),
            "mse_mean": float(arr.mean()),
            "mse_std": float(arr.std(ddof=0)),
        })
    return pd.DataFrame(rows).sort_values(["diagnosis_group", "subject_id"]).reset_index(drop=True)


@torch.no_grad()
def _vgae_mu_logvar(model: VGAE, graphs: Sequence, device: torch.device, batch_size: int = 64):
    """Per-graph (mu, log_var) as (N, latent_dim) numpy arrays.

    Pools node-level outputs to a single graph-level vector via mean — matching
    cluster_analysis.extract_latents_from_graphs(aggregate="mean"). Iterates
    one graph at a time to avoid PyG-DataLoader device-collation issues.
    """
    model.eval()
    mus: List[np.ndarray] = []
    logvars: List[np.ndarray] = []
    for g in graphs:
        x = g.x.to(device)
        edge_index = g.edge_index.to(device)
        mu, log_var = model.encode(x, edge_index)
        mus.append(mu.detach().cpu().numpy().mean(axis=0))
        logvars.append(log_var.detach().cpu().numpy().mean(axis=0))
    return np.stack(mus, axis=0), np.stack(logvars, axis=0)


def kl_per_dim(
    model: torch.nn.Module,
    graphs: Sequence,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Per-dim mean KL contribution KL(q(z_d | x) || N(0, 1)).

    KL_d = -0.5 * mean(1 + log_var_d - mu_d^2 - exp(log_var_d))
    Returns a 1D array of length latent_dim.
    """
    if not isinstance(model, VARIATIONAL_CLASSES):
        raise ValueError("kl_per_dim requires a variational model (VGAE/GAEVAE); got "
                         + type(model).__name__)
    device = device or torch.device("cpu")
    mu, log_var = _vgae_mu_logvar(model, graphs, device, batch_size=batch_size)
    # Per-graph, per-dim KL
    per_dim = -0.5 * (1.0 + log_var - mu ** 2 - np.exp(log_var))
    return per_dim.mean(axis=0)


def posterior_collapse_fraction(
    model: torch.nn.Module,
    graphs: Sequence,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
    tol: float = 1e-2,
) -> Dict[str, float]:
    """Fraction of latent dims whose mean KL is below `tol`.

    A high fraction (e.g. > 0.5) is a red flag: the posterior is collapsing
    toward the prior and the latent isn't being used. See `kl_per_dim`.
    """
    per_dim = kl_per_dim(model, graphs, device=device, batch_size=batch_size)
    fraction = float((per_dim < tol).mean())
    return {
        "tol": float(tol),
        "latent_dim": int(per_dim.size),
        "n_collapsed": int((per_dim < tol).sum()),
        "fraction_collapsed": fraction,
        "kl_per_dim_min": float(per_dim.min()),
        "kl_per_dim_max": float(per_dim.max()),
        "kl_per_dim_mean": float(per_dim.mean()),
    }


def mu_logvar_stats(
    model: torch.nn.Module,
    graphs: Sequence,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Per-dim mean/std of mu and mean of log_var across graphs."""
    if not isinstance(model, VARIATIONAL_CLASSES):
        raise ValueError("mu_logvar_stats requires a variational model (VGAE/GAEVAE); got "
                         + type(model).__name__)
    device = device or torch.device("cpu")
    mu, log_var = _vgae_mu_logvar(model, graphs, device, batch_size=batch_size)
    rows = []
    for d in range(mu.shape[1]):
        rows.append({
            "latent_dim": d,
            "mu_mean": float(mu[:, d].mean()),
            "mu_std": float(mu[:, d].std(ddof=0)),
            "log_var_mean": float(log_var[:, d].mean()),
            "sigma_mean": float(np.exp(0.5 * log_var[:, d]).mean()),
        })
    return pd.DataFrame(rows)

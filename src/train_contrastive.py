"""MoCo-v2 contrastive pretraining on per-epoch wSMI matrices.

- Reads the held-out split (data/holdout_subjects.json) and pretrains only
  on epochs from subjects NOT in the held-out pool.
- Two independent augmentations per epoch -> InfoNCE.
- Saves encoder weights + per-step loss curve.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from preprocessing import EEGtoGraph
from contrastive_adjacency import load_coords, weighted_knn_adjacency
from augmentations import augment
from contrastive_model import MoCo


# ---------------------------------------------------------------- data loading
class WsmiEpochs(Dataset):
    """In-memory wSMI epoch dataset.

    Loads every allowed subject's .npz once at init into a single
    contiguous (n_epochs, 256, 256) float32 array. Per-item __getitem__
    is then just a slice + normalise — no I/O on the hot path.

    Memory cost: ~26 GB for the full 115-subject train pool. The sbatch
    allocates 64 GB so this is comfortable.
    """
    def __init__(self, data_dir: str, allowed_subjects: set,
                 n_nodes: int = 256, normalise: bool = True):
        sessions = EEGtoGraph.enumerate_matrix_sessions(data_dir)
        # Filter to allowed subjects and npz sources
        sessions = [(sid, snum, src) for sid, snum, src in sessions
                    if src['kind'] == 'npz' and sid in allowed_subjects]
        # Load all into one contiguous array
        chunks = []
        for sid, snum, source in sessions:
            with np.load(source['path']) as d:
                chunks.append(d['data'].astype(np.float32, copy=False))
        if not chunks:
            raise RuntimeError(f"No data loaded for {len(allowed_subjects)} subjects")
        self.data = np.concatenate(chunks, axis=0)               # (n_total, N, N)
        # Compute normalisation stats from the loaded array
        if normalise:
            self.x_min = float(self.data.min())
            self.x_max = float(self.data.max())
            self.x_range = self.x_max - self.x_min
        else:
            self.x_min, self.x_max, self.x_range = 0.0, 1.0, 1.0
        print(f"  WsmiEpochs: {len(self.data):,} epochs across "
              f"{len(allowed_subjects)} subjects, "
              f"mem={self.data.nbytes/1e9:.1f} GB")
        if normalise:
            print(f"  norm stats: x_min={self.x_min:.4f}, x_max={self.x_max:.4f}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        mat = self.data[idx]                                     # (N, N) view
        if self.x_range > 0:
            mat = 2.0 * (mat - self.x_min) / self.x_range - 1.0
        return torch.from_numpy(mat.astype(np.float32, copy=False))


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--coords_file', default='data_scalp/GSN-HydroCel-257.txt')
    ap.add_argument('--holdout_json', default='data/holdout_subjects.json')
    ap.add_argument('--output_dir', default='output/contrastive')
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--queue_size', type=int, default=8192)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--momentum', type=float, default=0.999)
    ap.add_argument('--latent_per_node', type=int, default=1)
    ap.add_argument('--k_neighbours', type=int, default=10)
    ap.add_argument('--noise_std', type=float, default=0.02)
    ap.add_argument('--edge_p', type=float, default=0.05)
    ap.add_argument('--node_p', type=float, default=0.10)
    ap.add_argument('--scale_pct', type=float, default=0.05)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--smoke', action='store_true',
                    help='Use 10% of train_pool for a 2-epoch sanity run.')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device: {device}")

    # Held-out / train pool
    with open(args.holdout_json) as f:
        split = json.load(f)
    train_pool = set(split['train_pool'])
    if args.smoke:
        train_pool = set(sorted(train_pool)[:10])
        print(f"  SMOKE mode: training on {len(train_pool)} subjects; "
              f"--epochs respected ({args.epochs})")

    # Dataset + loader
    ds = WsmiEpochs(args.data_dir, allowed_subjects=train_pool)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=(args.num_workers > 0))

    # Adjacency
    coords = load_coords(args.coords_file)
    A_norm = weighted_knn_adjacency(coords, k=args.k_neighbours).to(device)
    print(f"  adjacency: {A_norm.shape}, nnz frac = "
          f"{(A_norm != 0).float().mean().item():.3f}")

    # Model
    model = MoCo(
        A_norm=A_norm,
        in_dim=256,
        hidden_dims=(128, 64, 32, 16),
        latent_per_node=args.latent_per_node,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs * max(1, len(loader)))

    # Train
    loss_hist = []
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        for x in loader:                                  # (B, N, F)
            x = x.to(device, non_blocking=True)
            x_q = augment(x, args.noise_std, args.edge_p, args.node_p, args.scale_pct)
            x_k = augment(x, args.noise_std, args.edge_p, args.node_p, args.scale_pct)
            loss, _ = model(x_q, x_k)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            scheduler.step()
            loss_hist.append(float(loss.item()))
            step += 1
            if step % 50 == 0 or step == 1:
                lr_now = scheduler.get_last_lr()[0]
                elapsed = (time.time() - t0) / 60
                print(f"    epoch {epoch+1}/{args.epochs}  step {step}  "
                      f"loss {loss.item():.4f}  lr {lr_now:.2e}  "
                      f"elapsed {elapsed:.1f} min")
        print(f"  --- epoch {epoch+1}/{args.epochs} done, "
              f"mean loss {np.mean(loss_hist[-len(loader):]):.4f}")

    # Save
    ckpt_path = os.path.join(args.output_dir, 'encoder_q.pt')
    torch.save({
        'encoder_q_state': model.encoder_q.state_dict(),
        'config': vars(args),
        'A_norm_path_hint': args.coords_file,
        'loss_hist': loss_hist,
    }, ckpt_path)
    print(f"\n  saved encoder weights -> {ckpt_path}")

    # Quick collapse check: embed a small batch and compute embedding std
    model.eval()
    with torch.no_grad():
        sample = torch.stack([ds[i] for i in range(min(256, len(ds)))]).to(device)
        emb = model._embed(sample)   # (B, embed_dim)
        emb_std = emb.std(dim=0).mean().item()
        emb_mean_norm = emb.norm(dim=1).mean().item()
    print(f"  sanity: embedding mean L2 norm = {emb_mean_norm:.3f}, "
          f"mean-per-dim std = {emb_std:.4f}  (collapsed if very small)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Per-cluster wSMI centroid matrices for the V=0.252 winning scenario
(last_100_per_session × balanced_by_diagnosis × GMM K=4).

Refits the GMM (deterministic with random_state=42), assigns clusters to every
last_100 epoch (not just the balanced subset — we want max statistical power
for the centroid plots), then streams the original .npz session files and
averages the (256, 256) wSMI matrices per cluster.

Outputs:
  output/centroid_last_100_balanced/
    centroids_wsmi.npy                   (4, 256, 256) float32
    centroids_counts.npy                 (4,) int — number of epochs per cluster
    centroids_absolute.png               raw mean per cluster, shared colourbar
    centroids_vs_grand_mean.png          difference vs grand-mean wSMI, RdBu_r
    labels_last_100.npy                  per-epoch cluster id for every last_100 epoch
    contingency_last_100_balanced.csv    cluster × diagnosis count table on balanced fit
    summary.json                         metrics + per-cluster diagnosis enrichment
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import EEGtoGraph


DX_ORDER = ['control', 'UWS', 'MCS-', 'MCS+', 'EMCS', 'COMA']


def take_last_n_per_session_idx(df, n):
    out = []
    for _, gdf in df.groupby(['subject_id', 'session_num'], sort=False):
        out.append(gdf.iloc[-n:].index.to_numpy() if len(gdf) >= n
                   else gdf.index.to_numpy())
    return np.concatenate(out)


def balance_by_diagnosis_indices(df, in_idx, rng):
    sub = df.loc[in_idx]
    counts = sub['diagnosis'].value_counts()
    # exclude any diagnoses not in DX_ORDER when deciding the cap
    cap_pool = counts[counts.index.isin(DX_ORDER)]
    target = int(cap_pool.min())
    out = []
    for dx, gdf in sub.groupby('diagnosis', sort=False):
        if dx not in DX_ORDER:
            continue
        if len(gdf) <= target:
            out.append(gdf.index.to_numpy())
        else:
            out.append(rng.choice(gdf.index.to_numpy(), size=target, replace=False))
    return np.concatenate(out), target


def contingency(labels, diagnoses, dx_order):
    K = int(np.max(labels)) + 1
    M = np.zeros((K, len(dx_order)), dtype=int)
    for c, d in zip(labels, diagnoses):
        if d in dx_order:
            M[c, dx_order.index(d)] += 1
    chi2, p, dof, _ = chi2_contingency(M + 1e-9)
    n = int(M.sum())
    cv = float(np.sqrt(chi2 / (n * (min(M.shape) - 1)))) if min(M.shape) > 1 else float('nan')
    return M, {'chi2': float(chi2), 'p': float(p), 'cramers_v': cv, 'n': n}


def compute_and_plot_centroids(labels, df_last, npz_index, out_dir, K):
    """Stream npz files, accumulate per-cluster sum of original wSMI matrices."""
    accum = np.zeros((K, 256, 256), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)

    # bucket meta entries by (subject, session) so each npz is opened once
    bucket: dict = {}
    for (row_pos, m), c in zip(df_last.iterrows(), labels):
        key = (m['subject_id'], m['session_num'])
        bucket.setdefault(key, []).append((int(m['matrix_idx']), int(c)))

    print(f"  computing centroids across {len(bucket)} sessions ...")
    t0 = time.time()
    for i, ((sid, snum), entries) in enumerate(bucket.items()):
        path = npz_index.get((sid, snum))
        if path is None:
            print(f"  WARNING: no npz for sub-{sid}/ses-{snum}")
            continue
        with np.load(path) as d:
            arr = d['data']
            mi = np.array([m for m, _ in entries], dtype=np.int64)
            cs = np.array([c for _, c in entries], dtype=np.int64)
            for k in range(K):
                sel = mi[cs == k]
                if sel.size:
                    accum[k] += arr[sel].sum(axis=0)
                    counts[k] += sel.size
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(bucket)}  ({(time.time()-t0)/60:.1f} min)")

    centroids = accum / counts[:, None, None].clip(min=1)
    np.save(os.path.join(out_dir, 'centroids_wsmi.npy'), centroids.astype(np.float32))
    np.save(os.path.join(out_dir, 'centroids_counts.npy'), counts)
    print(f"  counts per cluster: {counts.tolist()}")

    grand_mean = (accum.sum(axis=0) / counts.sum()).astype(np.float32) if counts.sum() else None

    # Pick a grid layout that's compact for any K (single-row for K ≤ 4,
    # otherwise 2 rows). K=3 → 1×3; K=4 → 2×2; K=6 → 2×3; etc.
    def _grid(k):
        if k <= 4:
            return 1, k
        cols = int(np.ceil(k / 2))
        return 2, cols

    # absolute heatmaps
    vmin = float(np.percentile(centroids, 2))
    vmax = float(np.percentile(centroids, 98))
    rows, cols = _grid(K)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.5))
    axes = np.atleast_1d(axes).ravel()
    for c in range(K):
        ax = axes[c]
        im = ax.imshow(centroids[c], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'cluster {c} centroid wSMI  (n={counts[c]:,} epochs)')
        ax.set_xlabel('electrode j'); ax.set_ylabel('electrode i')
        plt.colorbar(im, ax=ax, fraction=0.046)
    for c in range(K, len(axes)):
        axes[c].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'centroids_absolute.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # diff vs grand mean
    if grand_mean is not None:
        diffs = centroids - grand_mean
        vlim = float(np.percentile(np.abs(diffs), 98))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.5))
        axes = np.atleast_1d(axes).ravel()
        for c in range(K):
            ax = axes[c]
            im = ax.imshow(diffs[c], cmap='RdBu_r', vmin=-vlim, vmax=vlim)
            ax.set_title(f'cluster {c} − grand mean  (n={counts[c]:,})')
            ax.set_xlabel('electrode j'); ax.set_ylabel('electrode i')
            plt.colorbar(im, ax=ax, fraction=0.046)
        for c in range(K, len(axes)):
            axes[c].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'centroids_vs_grand_mean.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    return centroids, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--cluster_dir', default='output/full_cluster')
    ap.add_argument('--output_dir', default='output/centroid_last_100_balanced')
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--last_n', type=int, default=100)
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    print("[1/5] loading X_pca + meta")
    X_pca = np.load(os.path.join(args.cluster_dir, 'X_pca.npy'))
    with open(os.path.join(args.cluster_dir, 'meta.json')) as f:
        meta = json.load(f)
    df = pd.DataFrame(meta)
    df['matrix_idx'] = df['matrix_idx'].astype(int)
    df = df.sort_values(['subject_id', 'session_num', 'matrix_idx']).reset_index()
    df = df.rename(columns={'index': 'orig_index'})
    print(f"  X_pca: {X_pca.shape}, meta sorted: {len(df)}")

    # Rebuild npz lookup so we can re-open original session files for centroids
    sessions = EEGtoGraph.enumerate_matrix_sessions(args.data_dir)
    npz_index = {(sid, snum): source['path']
                 for sid, snum, source in sessions if source['kind'] == 'npz'}

    print(f"[2/5] taking last {args.last_n} epochs per session")
    last_idx_sorted = take_last_n_per_session_idx(df, args.last_n)
    df_last = df.loc[last_idx_sorted].reset_index(drop=True)
    last_orig_rows = df_last['orig_index'].to_numpy()
    print(f"  last_{args.last_n} subset: {len(df_last)} epochs across "
          f"{df_last['subject_id'].nunique()} subjects "
          f"({df_last.groupby(['subject_id', 'session_num']).ngroups} sessions)")

    print(f"[3/5] balancing across diagnoses for GMM fit")
    bal_local_idx, target = balance_by_diagnosis_indices(
        df_last, df_last.index.to_numpy(), rng)
    bal_orig_rows = df_last.loc[bal_local_idx, 'orig_index'].to_numpy()
    bal_dx = df_last.loc[bal_local_idx, 'diagnosis'].tolist()
    print(f"  per-dx cap = {target} → {len(bal_orig_rows)} epochs used for GMM fit")
    print(f"  bal dx counts: {Counter(bal_dx)}")

    print(f"[4/5] fitting GMM K={args.K} on the balanced subset, predicting on full last_N")
    gm = GaussianMixture(n_components=args.K, covariance_type='full',
                         random_state=args.random_state, n_init=3,
                         max_iter=200, reg_covar=1e-3)
    gm.fit(X_pca[bal_orig_rows])
    # Within-fit (balanced) labels — for the reproducibility V check
    bal_lbl = gm.predict(X_pca[bal_orig_rows])
    M_bal, stats_bal = contingency(bal_lbl, bal_dx, DX_ORDER)
    print(f"  balanced-fit V={stats_bal['cramers_v']:.3f}, chi2={stats_bal['chi2']:.1f}")
    np.savetxt(os.path.join(args.output_dir, 'contingency_last_100_balanced.csv'),
               M_bal, fmt='%d', delimiter=',',
               header=','.join(DX_ORDER), comments='')

    # Predict cluster for EVERY last_N epoch (not just balanced) — used for centroids
    full_last_lbl = gm.predict(X_pca[last_orig_rows])
    full_last_dx = df_last['diagnosis'].tolist()
    M_full, stats_full = contingency(full_last_lbl, full_last_dx, DX_ORDER)
    print(f"  full-last predictions V={stats_full['cramers_v']:.3f}, "
          f"chi2={stats_full['chi2']:.1f}  (over n={stats_full['n']:,} epochs)")
    np.save(os.path.join(args.output_dir, 'labels_last_100.npy'), full_last_lbl)

    print(f"[5/5] computing & plotting wSMI centroids")
    centroids, counts = compute_and_plot_centroids(
        full_last_lbl, df_last, npz_index, args.output_dir, args.K)

    # Compute per-cluster diagnosis enrichment for the summary
    col_pct = M_full / M_full.sum(axis=0, keepdims=True).clip(min=1)
    enrichment = {}
    for c in range(args.K):
        row = {dx: float(col_pct[c, j]) for j, dx in enumerate(DX_ORDER)}
        enrichment[c] = row

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump({
            'config': vars(args),
            'balanced_fit_V': stats_bal['cramers_v'],
            'balanced_fit_chi2': stats_bal['chi2'],
            'balanced_fit_n': stats_bal['n'],
            'balanced_fit_per_dx_cap': target,
            'full_last_V': stats_full['cramers_v'],
            'full_last_chi2': stats_full['chi2'],
            'full_last_n': stats_full['n'],
            'cluster_counts_full_last': counts.tolist(),
            'per_cluster_dx_enrichment_pct': enrichment,
        }, f, indent=2)

    print(f"\n  centroids → {args.output_dir}/centroids_absolute.png + _vs_grand_mean.png")
    print(f"  cluster diagnosis enrichment (P(diagnosis | cluster)):")
    print(f"           {'  '.join(f'{d:>7s}' for d in DX_ORDER)}")
    for c in range(args.K):
        row_pct = M_full[c] / M_full[c].sum() if M_full[c].sum() else np.zeros(len(DX_ORDER))
        print(f"  cl_{c}    {'  '.join(f'{v:6.1%} ' for v in row_pct)}  (n={counts[c]:,})")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Compare GMM-clustering quality across different epoch-subsampling schemes.

Question: does HOW we subsample the per-session epochs change which clusters
we find and how clinically meaningful they are? Specifically:
  - is early/middle/late part of the recording more informative?
  - is per-session equal weighting better than per-epoch (proportional to
    session length)?
  - is the previous V=0.207 subsample baseline reproducible?

Uses the already-saved X_pca + meta + diagnoses from output/full_cluster/, so
the PCA fit is held constant across all subsamples — isolating the clustering
effect from PCA-fit drift.
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


# ----------------------------------------------------------- partition helpers
def scheme_all(df, k=None, rng=None):
    return df.index.to_numpy()

def scheme_first_k(df, k, rng=None):
    """First k epochs (sorted by matrix_idx) per session."""
    return df.groupby(['subject_id', 'session_num'], sort=False) \
             .head(k).index.to_numpy()

def scheme_last_k(df, k, rng=None):
    return df.groupby(['subject_id', 'session_num'], sort=False) \
             .tail(k).index.to_numpy()

def scheme_middle_k(df, k, rng=None):
    """k contiguous epochs centred in each session."""
    out = []
    for (_, _), g in df.groupby(['subject_id', 'session_num'], sort=False):
        n = len(g)
        if n <= k:
            out.append(g.index.to_numpy())
        else:
            start = (n - k) // 2
            out.append(g.iloc[start:start + k].index.to_numpy())
    return np.concatenate(out)

def scheme_random_k(df, k, rng):
    out = []
    for (_, _), g in df.groupby(['subject_id', 'session_num'], sort=False):
        n = len(g)
        if n <= k:
            out.append(g.index.to_numpy())
        else:
            picks = rng.choice(g.index.to_numpy(), size=k, replace=False)
            picks.sort()
            out.append(picks)
    return np.concatenate(out)

def scheme_balanced_by_diagnosis(df, k=None, rng=None):
    """Downsample so every diagnosis has the same number of epochs (= the
    smallest diagnosis count). Tests whether equal class weighting changes
    the V relative to the natural imbalance.
    """
    counts = df['diagnosis'].value_counts()
    target = int(counts.min())
    out = []
    for dx, gdf in df.groupby('diagnosis', sort=False):
        if len(gdf) <= target:
            out.append(gdf.index.to_numpy())
        else:
            picks = rng.choice(gdf.index.to_numpy(), size=target, replace=False)
            out.append(picks)
    return np.concatenate(out)


SCHEMES = {
    # name              : (callable, k_arg)
    'all'                       : (scheme_all,                   None),
    'random_100_per_session'    : (scheme_random_k,              100),
    'random_300_per_session'    : (scheme_random_k,              300),
    'first_100_per_session'     : (scheme_first_k,               100),
    'last_100_per_session'      : (scheme_last_k,                100),
    'middle_100_per_session'    : (scheme_middle_k,              100),
    'balanced_by_diagnosis'     : (scheme_balanced_by_diagnosis, None),
}


# ------------------------------------------------------------ stats / plotting
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


def normalised_heatmap(M, dx_order, title, out_path):
    colsum = M.sum(axis=0, keepdims=True).clip(min=1)
    norm = M / colsum
    fig, ax = plt.subplots(figsize=(8, max(3.5, M.shape[0] * 0.5)))
    im = ax.imshow(norm, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(dx_order)))
    ax.set_xticklabels(dx_order, rotation=30, ha='right')
    ax.set_yticks(range(M.shape[0]))
    ax.set_xlabel('diagnosis'); ax.set_ylabel('cluster')
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(len(dx_order)):
            v = norm[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color='white' if v < 0.5 else 'black', fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster_dir', default='output/full_cluster')
    ap.add_argument('--output_dir', default='output/partition_search')
    ap.add_argument('--K', type=int, default=4, help='Number of GMM components')
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/4] loading X_pca + meta from {args.cluster_dir}")
    X_pca = np.load(os.path.join(args.cluster_dir, 'X_pca.npy'))
    with open(os.path.join(args.cluster_dir, 'meta.json')) as f:
        meta = json.load(f)
    print(f"  X_pca: {X_pca.shape}, meta: {len(meta)} entries")
    df = pd.DataFrame(meta)
    df = df.reset_index(drop=True)
    # diagnosis is already present from earlier attach_diagnoses step
    df['matrix_idx'] = df['matrix_idx'].astype(int)
    df = df.sort_values(['subject_id', 'session_num', 'matrix_idx']) \
           .reset_index().rename(columns={'index': 'orig_index'})
    # Sorted by session-major-then-time so head/tail/middle have intuitive meaning.

    dx_order_all = ['control', 'UWS', 'MCS-', 'MCS+', 'EMCS', 'COMA', 'unknown', 'n/a']
    seen = set(df['diagnosis'])
    dx_order = [d for d in dx_order_all if d in seen]
    print(f"  diagnoses (ordered): {dx_order}")
    print(f"  overall counts: {Counter(df['diagnosis'])}")

    print(f"[2/4] running GMM K={args.K} on each partition")
    rng = np.random.default_rng(args.random_state)
    summary = []
    for name, (fn, karg) in SCHEMES.items():
        t0 = time.time()
        idx_in_sorted = fn(df, k=karg, rng=rng)
        # Map back to original index in X_pca
        orig_idx = df.loc[idx_in_sorted, 'orig_index'].to_numpy()
        Xs = X_pca[orig_idx]
        diag_s = df.loc[idx_in_sorted, 'diagnosis'].tolist()
        gm = GaussianMixture(n_components=args.K, covariance_type='full',
                             random_state=args.random_state, n_init=3,
                             max_iter=200, reg_covar=1e-4)
        gm.fit(Xs)
        lbl = gm.predict(Xs)
        M, stats = contingency(lbl, diag_s, dx_order)
        dx_counts = Counter(diag_s)
        ctrl_share = dx_counts.get('control', 0) / max(stats['n'], 1)
        elapsed = time.time() - t0
        print(f"  {name:>26s}  n={stats['n']:6d}  ctrl_share={ctrl_share:5.1%}  "
              f"V={stats['cramers_v']:.3f}  chi2={stats['chi2']:8.1f}  ({elapsed:.0f}s)")
        normalised_heatmap(M, dx_order,
                           title=f'GMM K={args.K} | scheme={name} | V={stats["cramers_v"]:.3f}',
                           out_path=os.path.join(args.output_dir, f'heatmap_{name}.png'))
        np.savetxt(os.path.join(args.output_dir, f'contingency_{name}.csv'),
                   M, fmt='%d', delimiter=',', header=','.join(dx_order), comments='')
        summary.append({
            'scheme': name, 'n_epochs': stats['n'],
            'control_share': ctrl_share, 'cramers_v': stats['cramers_v'],
            'chi2': stats['chi2'], 'bic': float(gm.bic(Xs)),
            'cluster_sizes': np.bincount(lbl).tolist(),
            'diagnosis_counts': dict(dx_counts),
        })

    print("[3/4] cross-scheme summary")
    df_sum = pd.DataFrame(summary).sort_values('cramers_v', ascending=False)
    print()
    print(df_sum[['scheme', 'n_epochs', 'control_share', 'cramers_v', 'chi2', 'bic']]
          .to_string(index=False))
    print()

    df_sum.to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump({
            'config': vars(args),
            'results': summary,
        }, f, indent=2, default=str)

    print(f"[4/4] artefacts in {args.output_dir}/")


if __name__ == '__main__':
    main()

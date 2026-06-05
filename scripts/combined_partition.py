#!/usr/bin/env python3
"""
Stack the two best levers: last_N epochs per session AND diagnosis balancing.

From scripts/partition_search.py we know:
  * balanced_by_diagnosis → V = 0.227 (uses all epochs, equal dx weighting)
  * last_100_per_session  → V = 0.219 (best natural temporal window)

This script composes them: first take last_N epochs from each session, then
downsample each diagnosis class to a common cap (`balance_to` either the
smallest diagnosis count or a fixed target). We also sweep N ∈ {50,100,200}
and K ∈ {4,6,8} so we can see whether richer partitions need more components
to fit cleanly.
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


# --------------------------------------------------------------------- helpers
def take_last_n_per_session(df, n):
    """Return row indices for the last `n` epochs of each session."""
    out = []
    for _, gdf in df.groupby(['subject_id', 'session_num'], sort=False):
        if len(gdf) < 1:
            continue
        out.append(gdf.iloc[-n:].index.to_numpy() if len(gdf) >= n
                   else gdf.index.to_numpy())
    return np.concatenate(out)


def balance_by_diagnosis(df, idx, target_per_dx, rng):
    """From the given subset (idx), downsample so every diagnosis has at most
    `target_per_dx` epochs. If target_per_dx is None, use the smallest dx count.
    """
    sub = df.loc[idx]
    counts = sub['diagnosis'].value_counts()
    target = int(counts.min()) if target_per_dx is None else int(target_per_dx)
    out = []
    for dx, gdf in sub.groupby('diagnosis', sort=False):
        if len(gdf) <= target:
            out.append(gdf.index.to_numpy())
        else:
            picks = rng.choice(gdf.index.to_numpy(), size=target, replace=False)
            out.append(picks)
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


def normalised_heatmap(M, dx_order, title, out_path):
    colsum = M.sum(axis=0, keepdims=True).clip(min=1)
    norm = M / colsum
    fig, ax = plt.subplots(figsize=(7, max(3.5, M.shape[0] * 0.5)))
    im = ax.imshow(norm, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(dx_order)))
    ax.set_xticklabels(dx_order, rotation=30, ha='right')
    ax.set_yticks(range(M.shape[0]))
    ax.set_xlabel('diagnosis'); ax.set_ylabel('cluster')
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(len(dx_order)):
            v = norm[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color='white' if v < 0.5 else 'black', fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def fit_and_score(X_pca, idx_sorted, df, dx_order, K, random_state, name, out_dir,
                  reg_covar=1e-3):
    """Fit GMM, score, save heatmap + contingency, return summary dict.

    Returns None on numerical failure (rare with reg_covar=1e-3 but still
    possible for very small / collinear samples — we'd rather skip + log than
    crash the whole sweep).
    """
    orig_idx = df.loc[idx_sorted, 'orig_index'].to_numpy()
    Xs = X_pca[orig_idx]
    diag_s = df.loc[idx_sorted, 'diagnosis'].tolist()
    try:
        gm = GaussianMixture(n_components=K, covariance_type='full',
                             random_state=random_state, n_init=3,
                             max_iter=200, reg_covar=reg_covar)
        gm.fit(Xs)
    except ValueError as e:
        # Common cause: singleton component / ill-defined covariance at small n.
        # Retry with much stronger regularisation; if that also fails, skip.
        try:
            gm = GaussianMixture(n_components=K, covariance_type='full',
                                 random_state=random_state, n_init=1,
                                 max_iter=200, reg_covar=1e-2)
            gm.fit(Xs)
        except ValueError as e2:
            print(f"  ! GMM failed for {name} K={K}: {e2}")
            return None
    lbl = gm.predict(Xs)
    M, stats = contingency(lbl, diag_s, dx_order)
    dx_counts = Counter(diag_s)
    ctrl_share = dx_counts.get('control', 0) / max(stats['n'], 1)
    normalised_heatmap(M, dx_order,
                       title=f'{name} | K={K} | V={stats["cramers_v"]:.3f}',
                       out_path=os.path.join(out_dir, f'heatmap_{name}.png'))
    np.savetxt(os.path.join(out_dir, f'contingency_{name}.csv'),
               M, fmt='%d', delimiter=',', header=','.join(dx_order), comments='')
    return {
        'scheme': name, 'K': K, 'n_epochs': stats['n'],
        'control_share': ctrl_share, 'cramers_v': stats['cramers_v'],
        'chi2': stats['chi2'], 'bic': float(gm.bic(Xs)),
        'dx_counts': dict(dx_counts),
    }


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster_dir', default='output/full_cluster')
    ap.add_argument('--output_dir', default='output/combined_partition')
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--ns', type=int, nargs='+', default=[50, 100, 200])
    ap.add_argument('--ks', type=int, nargs='+', default=[4, 6, 8])
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    print(f"[1/4] loading X_pca + meta from {args.cluster_dir}")
    X_pca = np.load(os.path.join(args.cluster_dir, 'X_pca.npy'))
    with open(os.path.join(args.cluster_dir, 'meta.json')) as f:
        meta = json.load(f)
    df = pd.DataFrame(meta)
    df['matrix_idx'] = df['matrix_idx'].astype(int)
    df = df.sort_values(['subject_id', 'session_num', 'matrix_idx']).reset_index()
    df = df.rename(columns={'index': 'orig_index'})
    print(f"  X_pca: {X_pca.shape}, meta sorted: {len(df)}")

    dx_order_all = ['control', 'UWS', 'MCS-', 'MCS+', 'EMCS', 'COMA']
    seen = set(df['diagnosis'])
    dx_order = [d for d in dx_order_all if d in seen]
    print(f"  diagnoses (ordered): {dx_order}")

    print("[2/4] running combined schemes")
    summary = []

    def record(s, kind, N):
        if s is None:
            return
        s['scheme_kind'] = kind
        s['N'] = N
        summary.append(s)
        print(f"  {s['scheme']:>26s}  K={s['K']}  n={s['n_epochs']:5d}  "
              f"ctrl={s['control_share']:5.1%}  V={s['cramers_v']:.3f}  "
              f"chi2={s['chi2']:7.1f}")

    # Baseline A: balanced_by_diagnosis on all epochs
    for K in args.ks:
        idx_all = df.index.to_numpy()
        idx_bal, target = balance_by_diagnosis(df, idx_all, target_per_dx=None, rng=rng)
        s = fit_and_score(X_pca, idx_bal, df, dx_order, K, args.random_state,
                          name=f'all_balanced_K{K}', out_dir=args.output_dir)
        record(s, 'all_balanced', None)

    # Baseline B: last_N only (no balancing)
    for N in args.ns:
        idx_last = take_last_n_per_session(df, N)
        for K in args.ks:
            s = fit_and_score(X_pca, idx_last, df, dx_order, K, args.random_state,
                              name=f'last_{N}_K{K}', out_dir=args.output_dir)
            record(s, 'last_only', N)

    # COMBINED: last_N THEN balanced_by_diagnosis
    for N in args.ns:
        idx_last = take_last_n_per_session(df, N)
        idx_lb, tgt = balance_by_diagnosis(df, idx_last, target_per_dx=None, rng=rng)
        for K in args.ks:
            s = fit_and_score(X_pca, idx_lb, df, dx_order, K, args.random_state,
                              name=f'last_{N}_balanced_K{K}', out_dir=args.output_dir)
            record(s, 'last_balanced', N)

    print()
    print("[3/4] summary (sorted by V desc)")
    df_sum = pd.DataFrame(summary).sort_values('cramers_v', ascending=False)
    print(df_sum[['scheme', 'scheme_kind', 'N', 'K', 'n_epochs', 'control_share',
                  'cramers_v', 'chi2', 'bic']].to_string(index=False))
    print()
    df_sum.to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump({'config': vars(args), 'results': summary}, f, indent=2, default=str)

    # Comparison plot: V vs (N, K) for last_only vs last_balanced
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for kind, marker, ls in [('last_only', 'o', '--'), ('last_balanced', 's', '-')]:
        for K in args.ks:
            sub = df_sum[(df_sum['scheme_kind'] == kind) & (df_sum['K'] == K)] \
                  .sort_values('N')
            if len(sub):
                ax.plot(sub['N'], sub['cramers_v'], marker=marker, ls=ls,
                        markersize=9, linewidth=2,
                        label=f'{kind}, K={K}')
    # Horizontal reference for all_balanced (best across K)
    ab = df_sum[df_sum['scheme_kind'] == 'all_balanced']
    if len(ab):
        best_ab = ab['cramers_v'].max()
        ax.axhline(best_ab, color='grey', ls=':', label=f'all_balanced best (K varies)  V={best_ab:.3f}')
    ax.set_xlabel('N (last_N epochs per session)')
    ax.set_ylabel("Cramer's V")
    ax.set_title('Composing partitions: last_N × diagnosis-balancing × K')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'combined_sweep.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[4/4] artefacts in {args.output_dir}/")


if __name__ == '__main__':
    main()

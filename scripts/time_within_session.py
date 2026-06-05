#!/usr/bin/env python3
"""
Drill into the time-within-session effect on GMM clustering quality.

Builds on scripts/partition_search.py's finding that
  last_100 > first_100 > random_100 > middle_100
and asks two follow-up questions:

  (Q1) Where exactly in the recording is the signal strongest?
       → 5 quintile schemes, each = first N epochs of its 1/5-of-session bin.

  (Q2) What's the right window size?
       → first_N and last_N for N ∈ {50, 100, 200, 300}.

All schemes use the same precomputed PCA(50) features in X_pca.npy + the same
GMM hyperparameters (K=4, full covariance, random_state=42, n_init=3) so the
only thing changing between rows of the result table is the epoch selection.
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


# ---------------------------------------------------------- partition helpers
def take_window(df_sess, start_frac, window):
    """From a session's dataframe (already sorted by matrix_idx), take `window`
    epochs starting at position `start_frac` (in [0, 1)) of the session.
    Returns the row-index numpy array (into the global df). Skips sessions
    whose length is less than start_frac * n + window — the few short ones —
    so every scheme sees roughly the same per-session contribution.
    """
    n = len(df_sess)
    start = int(start_frac * n)
    end = start + window
    if end > n:
        return None
    return df_sess.iloc[start:end].index.to_numpy()


def take_last(df_sess, window):
    n = len(df_sess)
    if n < window:
        return None
    return df_sess.iloc[-window:].index.to_numpy()


def build_scheme(df, scheme_kind, **kwargs):
    """Apply a per-session scheme; concatenate the returned indices.
    `df` should already be sorted (subject_id, session_num, matrix_idx).
    """
    out = []
    skipped = 0
    for _, gdf in df.groupby(['subject_id', 'session_num'], sort=False):
        if scheme_kind == 'first':
            idx = take_window(gdf, 0.0, kwargs['window'])
        elif scheme_kind == 'last':
            idx = take_last(gdf, kwargs['window'])
        elif scheme_kind == 'quintile':
            start_frac = kwargs['q'] / 5.0
            idx = take_window(gdf, start_frac, kwargs['window'])
        else:
            raise ValueError(scheme_kind)
        if idx is None:
            skipped += 1
            continue
        out.append(idx)
    if not out:
        return np.array([], dtype=int), skipped
    return np.concatenate(out), skipped


# ------------------------------------------------------------------- analysis
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


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster_dir', default='output/full_cluster')
    ap.add_argument('--output_dir', default='output/time_within_session')
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    # Build the full scheme list
    schemes = []
    # Quintiles at window=100 — answers Q1 (when in the recording is signal strongest?)
    for q in range(5):
        schemes.append((f'quintile_{q+1}_w100', 'quintile', {'q': q, 'window': 100}))
    # First/Last at multiple windows — answers Q2 (sweet-spot window size?)
    for W in (50, 100, 200, 300):
        schemes.append((f'first_{W}', 'first', {'window': W}))
        schemes.append((f'last_{W}', 'last', {'window': W}))

    print(f"[2/4] running GMM K={args.K} on {len(schemes)} schemes "
          f"(quintile sweep + first/last window sweep)")
    rng_unused = np.random.default_rng(args.random_state)  # not used; all schemes deterministic
    summary = []
    for (name, kind, kwargs) in schemes:
        t0 = time.time()
        idx_sorted, skipped = build_scheme(df, kind, **kwargs)
        if idx_sorted.size == 0:
            print(f"  {name:>22s}  SKIPPED ({skipped} sessions too short)")
            continue
        orig_idx = df.loc[idx_sorted, 'orig_index'].to_numpy()
        Xs = X_pca[orig_idx]
        diag_s = df.loc[idx_sorted, 'diagnosis'].tolist()
        gm = GaussianMixture(n_components=args.K, covariance_type='full',
                             random_state=args.random_state, n_init=3,
                             max_iter=200, reg_covar=1e-4)
        gm.fit(Xs)
        lbl = gm.predict(Xs)
        M, stats = contingency(lbl, diag_s, dx_order)
        dx_counts = Counter(diag_s)
        ctrl_share = dx_counts.get('control', 0) / max(stats['n'], 1)
        elapsed = time.time() - t0
        print(f"  {name:>22s}  n={stats['n']:6d}  skipped={skipped:3d}  "
              f"ctrl_share={ctrl_share:5.1%}  V={stats['cramers_v']:.3f}  "
              f"chi2={stats['chi2']:8.1f}  ({elapsed:.0f}s)")
        normalised_heatmap(M, dx_order,
                           title=f'GMM K={args.K} | {name} | V={stats["cramers_v"]:.3f}',
                           out_path=os.path.join(args.output_dir, f'heatmap_{name}.png'))
        np.savetxt(os.path.join(args.output_dir, f'contingency_{name}.csv'),
                   M, fmt='%d', delimiter=',', header=','.join(dx_order), comments='')
        summary.append({
            'scheme': name, 'kind': kind, 'window': kwargs.get('window'),
            'q': kwargs.get('q'), 'n_epochs': stats['n'], 'skipped_sessions': skipped,
            'control_share': ctrl_share, 'cramers_v': stats['cramers_v'],
            'chi2': stats['chi2'], 'bic': float(gm.bic(Xs)),
        })

    print()
    print("[3/4] summary table (sorted by Cramer's V desc)")
    df_sum = pd.DataFrame(summary).sort_values('cramers_v', ascending=False)
    print(df_sum[['scheme', 'n_epochs', 'skipped_sessions', 'control_share',
                  'cramers_v', 'chi2']].to_string(index=False))
    print()

    df_sum.to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump({'config': vars(args), 'results': summary}, f, indent=2, default=str)

    # Two summary plots
    quintile_df = df_sum[df_sum['kind'] == 'quintile'].sort_values('q')
    if not quintile_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(quintile_df['q'].astype(int) + 1, quintile_df['cramers_v'],
                'o-', linewidth=2, markersize=10)
        ax.set_xlabel('quintile (1 = first 1/5, 5 = last 1/5)')
        ax.set_ylabel("Cramer's V")
        ax.set_title('V across recording-time quintiles (window=100)')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3, 4, 5])
        for _, r in quintile_df.iterrows():
            ax.annotate(f"V={r['cramers_v']:.3f}", (int(r['q'])+1, r['cramers_v']),
                        textcoords='offset points', xytext=(8, 6), fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'quintile_sweep.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    win_df = df_sum[df_sum['kind'].isin(['first', 'last'])]
    if not win_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for side, marker, color in [('first', 'o', 'tab:blue'), ('last', 's', 'tab:red')]:
            sub = win_df[win_df['kind'] == side].sort_values('window')
            ax.plot(sub['window'], sub['cramers_v'],
                    marker=marker, color=color, linewidth=2,
                    markersize=10, label=f'{side}_N (start at session {side})')
        ax.set_xlabel('window size N (epochs)')
        ax.set_ylabel("Cramer's V")
        ax.set_title('V vs window size — first vs last N epochs per session')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'window_size_sweep.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[4/4] artefacts in {args.output_dir}/")


if __name__ == '__main__':
    main()

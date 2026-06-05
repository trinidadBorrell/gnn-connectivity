#!/usr/bin/env python3
"""
RAW wSMI clustering sanity check
=================================
Cluster directly on per-epoch wSMI matrices (no GAE involvement) and ask:
do the resulting clusters preferentially capture UWS, MCS-, MCS+, EMCS, COMA
or controls? If so, the dataset already carries diagnostic signal and the GAE
has something to learn. If not, the raw signal is weak.

Pipeline:
  - enumerate sessions via src.preprocessing.EEGtoGraph.enumerate_matrix_sessions
  - subsample EPOCHS_PER_SESSION random epochs per session (stratifies by session)
  - flatten each 256x256 wSMI to its upper triangle (32,640 dims)
  - StandardScaler then PCA(N_PCA)
  - run four clusterings: KMeans, GMM, Spectral (nystroem), Louvain on a kNN graph
  - per method: cluster x diagnosis contingency, chi^2 + Cramer's V, stacked bar
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from scipy.stats import chi2_contingency
import networkx as nx
import community as community_louvain  # python-louvain

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import EEGtoGraph


def load_subsampled_features(data_dir, epochs_per_session, random_state):
    """Walk sessions, return (X, meta) where X is (n, 32640) and meta is a list
    of dicts {subject_id, session_num, cohort, matrix_idx}.
    """
    rng = np.random.default_rng(random_state)
    sessions = EEGtoGraph.enumerate_matrix_sessions(data_dir)
    print(f"  enumerated {len(sessions)} sessions")
    feats = []
    meta = []
    n_nodes = 256
    iu = np.triu_indices(n_nodes, k=1)
    for i, (sid, snum, source) in enumerate(sessions):
        if source['kind'] != 'npz':
            print(f"  WARNING: skipping non-npz session sub-{sid}/ses-{snum}")
            continue
        with np.load(source['path']) as d:
            arr = d['data']  # (n_eps, 256, 256)
            n_eps = arr.shape[0]
            k = min(epochs_per_session, n_eps)
            idx = rng.choice(n_eps, size=k, replace=False)
            idx.sort()
            sub = arr[idx]  # (k, 256, 256)
        # flatten upper triangle
        flat = sub[:, iu[0], iu[1]]  # (k, 32640)
        feats.append(flat)
        for ix in idx:
            meta.append({'subject_id': sid, 'session_num': snum,
                         'cohort': source.get('cohort'), 'matrix_idx': int(ix)})
        if (i + 1) % 25 == 0:
            print(f"  loaded {i+1}/{len(sessions)} sessions, total {sum(f.shape[0] for f in feats)} epochs")
    X = np.concatenate(feats, axis=0).astype(np.float32)
    print(f"  final X: shape={X.shape}, dtype={X.dtype}, mem={X.nbytes/1e9:.1f} GB")
    return X, meta


def attach_diagnoses(meta, labels_csv):
    """Add 'diagnosis' to each meta entry by looking up patient_labels.csv.
    Controls (cohort=='control') get diagnosis='control'.
    Missing DOC entries get diagnosis='unknown'.
    """
    df = pd.read_csv(labels_csv, dtype=str)
    df['session_z'] = df['session'].str.zfill(2)
    lookup = {(r['subject'], r['session_z']): r['diagnostic_crs_final']
              for _, r in df.iterrows()}
    for m in meta:
        if m['cohort'] == 'control':
            m['diagnosis'] = 'control'
        else:
            key = (m['subject_id'], str(m['session_num']).zfill(2))
            m['diagnosis'] = lookup.get(key, 'unknown')
    counter = Counter(m['diagnosis'] for m in meta)
    print(f"  diagnosis distribution at epoch level: {dict(counter)}")
    return meta


def contingency_and_stats(labels, diagnoses, dx_order):
    """Build (n_clusters, n_diagnoses) count matrix and run chi-square."""
    K = int(labels.max()) + 1
    M = np.zeros((K, len(dx_order)), dtype=int)
    for c, d in zip(labels, diagnoses):
        if d in dx_order:
            M[c, dx_order.index(d)] += 1
    chi2, pval, dof, expected = chi2_contingency(M + 1e-9)  # smooth zeros
    n = M.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(M.shape) - 1)))) if min(M.shape) > 1 else float('nan')
    return M, {'chi2': float(chi2), 'p': float(pval), 'dof': int(dof),
               'cramers_v': cramers_v, 'n': int(n)}


def save_plots(M, dx_order, method, k_label, out_dir):
    """Per-cluster stacked bar (% of each diagnosis) + raw count heatmap."""
    K = M.shape[0]
    rowsum = M.sum(axis=1, keepdims=True).clip(min=1)
    pct = M / rowsum

    # stacked bar
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(K)
    colors = plt.cm.tab10.colors[:len(dx_order)]
    for j, dx in enumerate(dx_order):
        ax.bar(np.arange(K), pct[:, j], bottom=bottom, label=dx, color=colors[j])
        bottom += pct[:, j]
    ax.set_xlabel('cluster')
    ax.set_ylabel('fraction of epochs')
    ax.set_title(f'{method} K={k_label}: diagnosis mix per cluster')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_xticks(np.arange(K))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'stacked_bar_{method}_K{k_label}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # heatmap of normalised columns (so column sums to 1: % of each dx that lands in cluster c)
    colsum = M.sum(axis=0, keepdims=True).clip(min=1)
    norm_by_dx = M / colsum
    fig, ax = plt.subplots(figsize=(8, max(4, K * 0.4)))
    im = ax.imshow(norm_by_dx, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(dx_order)))
    ax.set_xticklabels(dx_order, rotation=30, ha='right')
    ax.set_yticks(range(K))
    ax.set_xlabel('diagnosis')
    ax.set_ylabel('cluster')
    ax.set_title(f'{method} K={k_label}: P(cluster | diagnosis)')
    for i in range(K):
        for j in range(len(dx_order)):
            ax.text(j, i, f'{norm_by_dx[i, j]:.2f}', ha='center', va='center',
                    color='white' if norm_by_dx[i, j] < 0.5 else 'black', fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'heatmap_{method}_K{k_label}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def run_kmeans(X, ks, dx, diagnoses, dx_order, out_dir, random_state):
    summary = []
    for K in ks:
        km = KMeans(n_clusters=K, n_init=10, random_state=random_state)
        lbl = km.fit_predict(X)
        M, stats = contingency_and_stats(lbl, diagnoses, dx_order)
        print(f"  KMeans K={K}: chi2={stats['chi2']:.1f}, p={stats['p']:.2e}, V={stats['cramers_v']:.3f}")
        save_plots(M, dx_order, 'kmeans', K, out_dir)
        np.savetxt(os.path.join(out_dir, f'contingency_kmeans_K{K}.csv'), M,
                   fmt='%d', delimiter=',', header=','.join(dx_order), comments='')
        summary.append({'method': 'kmeans', 'K': K, **stats})
    return summary


def run_gmm(X, ks, dx, diagnoses, dx_order, out_dir, random_state):
    summary = []
    for K in ks:
        gm = GaussianMixture(n_components=K, covariance_type='full',
                             random_state=random_state, n_init=3, max_iter=200,
                             reg_covar=1e-4)
        gm.fit(X)
        lbl = gm.predict(X)
        M, stats = contingency_and_stats(lbl, diagnoses, dx_order)
        stats['bic'] = float(gm.bic(X))
        stats['aic'] = float(gm.aic(X))
        print(f"  GMM K={K}: chi2={stats['chi2']:.1f}, p={stats['p']:.2e}, V={stats['cramers_v']:.3f}, BIC={stats['bic']:.0f}")
        save_plots(M, dx_order, 'gmm', K, out_dir)
        np.savetxt(os.path.join(out_dir, f'contingency_gmm_K{K}.csv'), M,
                   fmt='%d', delimiter=',', header=','.join(dx_order), comments='')
        summary.append({'method': 'gmm', 'K': K, **stats})
    return summary


def run_spectral(X, ks, dx, diagnoses, dx_order, out_dir, random_state, max_n=8000):
    """Spectral clustering is O(N^2) memory; subsample if too large."""
    summary = []
    if X.shape[0] > max_n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        Xs = X[idx]
        diag_s = [diagnoses[i] for i in idx]
        print(f"  spectral: subsampled to {max_n} points")
    else:
        Xs = X
        diag_s = diagnoses
    for K in ks:
        sc = SpectralClustering(n_clusters=K, affinity='nearest_neighbors',
                                n_neighbors=15, assign_labels='kmeans',
                                random_state=random_state, n_jobs=-1)
        lbl = sc.fit_predict(Xs)
        M, stats = contingency_and_stats(lbl, diag_s, dx_order)
        print(f"  Spectral K={K}: chi2={stats['chi2']:.1f}, p={stats['p']:.2e}, V={stats['cramers_v']:.3f}")
        save_plots(M, dx_order, 'spectral', K, out_dir)
        np.savetxt(os.path.join(out_dir, f'contingency_spectral_K{K}.csv'), M,
                   fmt='%d', delimiter=',', header=','.join(dx_order), comments='')
        summary.append({'method': 'spectral', 'K': K, **stats})
    return summary


def run_louvain(X, dx, diagnoses, dx_order, out_dir, random_state, knn=15, max_n=12000):
    if X.shape[0] > max_n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        Xs = X[idx]
        diag_s = [diagnoses[i] for i in idx]
        print(f"  louvain: subsampled to {max_n} points")
    else:
        Xs = X
        diag_s = diagnoses
    A = kneighbors_graph(Xs, n_neighbors=knn, mode='distance', include_self=False, n_jobs=-1)
    # convert distance to similarity (Gaussian kernel with median bandwidth)
    nz = A.data
    bw = np.median(nz[nz > 0])
    A.data = np.exp(-(A.data ** 2) / (2 * bw ** 2))
    A = 0.5 * (A + A.T)  # symmetrise
    G = nx.from_scipy_sparse_array(A)
    partition = community_louvain.best_partition(G, random_state=random_state)
    lbl = np.array([partition[i] for i in range(len(Xs))])
    K = int(lbl.max()) + 1
    M, stats = contingency_and_stats(lbl, diag_s, dx_order)
    stats['K'] = K
    print(f"  Louvain (auto K={K}): chi2={stats['chi2']:.1f}, p={stats['p']:.2e}, V={stats['cramers_v']:.3f}")
    save_plots(M, dx_order, 'louvain', K, out_dir)
    np.savetxt(os.path.join(out_dir, f'contingency_louvain_K{K}.csv'), M,
               fmt='%d', delimiter=',', header=','.join(dx_order), comments='')
    return [{'method': 'louvain', **stats}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--labels_csv',
                    default='data/wsmi_res/256_electrodes-20260526T091011Z-3-001/256_electrodes/wsmi_res_DOC/patient_labels.csv')
    ap.add_argument('--output_dir', default='output/raw_cluster_sanity')
    ap.add_argument('--epochs_per_session', type=int, default=100)
    ap.add_argument('--pca_dim', type=int, default=50)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--ks', type=int, nargs='+', default=[4, 6, 8])
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[1/5] loading + subsampling ({args.epochs_per_session} epochs/session)")
    X, meta = load_subsampled_features(args.data_dir, args.epochs_per_session, args.random_state)

    print("[2/5] attaching diagnoses from patient_labels.csv")
    meta = attach_diagnoses(meta, args.labels_csv)

    dx_order = ['control', 'UWS', 'MCS-', 'MCS+', 'EMCS', 'COMA', 'unknown', 'n/a']
    diagnoses = [m['diagnosis'] for m in meta]
    # drop diagnoses not in dx_order? keep all but stats will ignore not-in-order
    seen = set(diagnoses)
    dx_order = [d for d in dx_order if d in seen]
    print(f"  diagnoses present (ordered): {dx_order}")

    print("[3/5] StandardScaler + PCA")
    scaler = StandardScaler(copy=False)
    X = scaler.fit_transform(X)
    print(f"  scaled. fitting PCA({args.pca_dim})")
    pca = PCA(n_components=args.pca_dim, random_state=args.random_state)
    X_pca = pca.fit_transform(X)
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance: {explained:.3f}")
    del X, scaler  # free memory

    print(f"[4/5] clustering. KMeans+GMM on full {X_pca.shape[0]} pts; spectral/louvain subsampled")
    summary = []
    summary += run_kmeans(X_pca, args.ks, None, diagnoses, dx_order,
                          args.output_dir, args.random_state)
    summary += run_gmm(X_pca, args.ks, None, diagnoses, dx_order,
                       args.output_dir, args.random_state)
    summary += run_spectral(X_pca, args.ks, None, diagnoses, dx_order,
                            args.output_dir, args.random_state)
    summary += run_louvain(X_pca, None, diagnoses, dx_order,
                           args.output_dir, args.random_state)

    print("[5/5] saving summary")
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump({
            'config': vars(args),
            'pca_explained_variance': explained,
            'diagnosis_counts': dict(Counter(diagnoses)),
            'methods': summary,
        }, f, indent=2)

    print("\n=== summary ===")
    for s in summary:
        K = s.get('K', s.get('K', '?'))
        print(f"  {s['method']:>9s}  K={K}  chi2={s['chi2']:8.1f}  p={s['p']:.2e}  Cramers_V={s['cramers_v']:.3f}")
    print(f"\nartifacts: {args.output_dir}/")


if __name__ == '__main__':
    main()

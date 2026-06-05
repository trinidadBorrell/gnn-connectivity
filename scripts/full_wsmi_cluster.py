#!/usr/bin/env python3
"""
FULL-DATA raw wSMI clustering + cluster-centroid plotting.

Differences from scripts/raw_wsmi_cluster_sanity.py:
  - uses *every* epoch of every session (no per-session subsampling cap)
  - persists the fitted (StandardScaler, PCA, GMM, KMeans) models to disk
    so leave-N-subjects-out validation can reuse them without retraining
  - dumps per-method per-epoch labels for downstream analysis
  - extra step: for the GMM K=4 fit, computes the per-cluster mean wSMI
    256x256 matrix (the "cluster centroid in wSMI space", as opposed to the
    PCA-space centroid the GMM actually stores) and renders 4 + (diff) plots

Reproducibility:
  - random_state threaded everywhere (StandardScaler is deterministic;
    PCA, KMeans, GMM all take random_state)
  - sessions sorted deterministically by (subject_id, session_num)
  - epochs taken in their natural on-disk order (no random subsampling)
  - the exact CLI args + scikit-learn versions are dumped to summary.json
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

import joblib
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import EEGtoGraph


# --------------------------------------------------------------------- loading
def load_full_features(data_dir):
    """Return (X_flat, meta, npz_index) where:
      X_flat   : (n_total_epochs, 32640) float32 — upper-triangle of every wSMI matrix
      meta     : list of dicts, one per row of X_flat, with subject_id / session_num /
                 cohort / matrix_idx / global_index
      npz_index: dict {(subject_id, session_num): npz_path} for the centroid pass

    Single pass: load each session's .npz, extract upper-triangle, append to a
    list, then np.concatenate at the end. Peaks ~2× the final X size during
    concat — fine inside our 128 GB SLURM allocation. (Earlier attempt at a
    header-only "peek then preallocate" used `np.load(..., mmap_mode='r')` which
    silently doesn't mmap .npz files; it returns an NpzFile wrapper.)
    """
    sessions = EEGtoGraph.enumerate_matrix_sessions(data_dir)
    print(f"  enumerated {len(sessions)} sessions")
    n_nodes = 256
    iu = np.triu_indices(n_nodes, k=1)
    n_triu = len(iu[0])

    feats = []
    meta = []
    npz_index = {}
    cursor = 0
    t0 = time.time()
    for i, (sid, snum, source) in enumerate(sessions):
        if source['kind'] != 'npz':
            print(f"  WARNING: skipping non-npz session sub-{sid}/ses-{snum}")
            continue
        npz_index[(sid, snum)] = source['path']
        with np.load(source['path']) as d:
            arr = d['data']  # (n_eps, 256, 256)
            n_eps = int(arr.shape[0])
            # vectorised upper-triangle extraction → (n_eps, n_triu) float32
            flat = arr[:, iu[0], iu[1]].astype(np.float32, copy=False)
        feats.append(flat)
        cohort = source.get('cohort')
        for j in range(n_eps):
            meta.append({'subject_id': sid, 'session_num': snum, 'cohort': cohort,
                         'matrix_idx': j, 'global_index': cursor + j})
        cursor += n_eps
        if (i + 1) % 25 == 0:
            print(f"  loaded {i+1}/{len(sessions)} sessions, {cursor:,} epochs, "
                  f"{(time.time()-t0)/60:.1f} min elapsed")

    print(f"  concatenating {len(feats)} session arrays ...")
    X = np.concatenate(feats, axis=0)
    del feats  # release the per-session list refs (large)
    print(f"  final X: shape={X.shape}, dtype={X.dtype}, mem={X.nbytes/1e9:.1f} GB")
    return X, meta, npz_index


def attach_diagnoses(meta, labels_csv):
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
    print(f"  diagnosis distribution (epoch-level): {dict(counter)}")
    return meta


# ----------------------------------------------------------------------- stats
def contingency_and_stats(labels, diagnoses, dx_order):
    K = int(np.max(labels)) + 1
    M = np.zeros((K, len(dx_order)), dtype=int)
    for c, d in zip(labels, diagnoses):
        if d in dx_order:
            M[c, dx_order.index(d)] += 1
    chi2, p, dof, _ = chi2_contingency(M + 1e-9)
    n = int(M.sum())
    cv = float(np.sqrt(chi2 / (n * (min(M.shape) - 1)))) if min(M.shape) > 1 else float('nan')
    return M, {'chi2': float(chi2), 'p': float(p), 'dof': int(dof),
               'cramers_v': cv, 'n': n}


def save_diagnostic_plots(M, dx_order, method, K, out_dir):
    """Stacked bar (P(dx | cluster)) and heatmap (P(cluster | dx))."""
    rowsum = M.sum(axis=1, keepdims=True).clip(min=1)
    pct = M / rowsum
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(M.shape[0])
    colors = plt.cm.tab10.colors[:len(dx_order)]
    for j, dx in enumerate(dx_order):
        ax.bar(np.arange(M.shape[0]), pct[:, j], bottom=bottom, label=dx, color=colors[j])
        bottom += pct[:, j]
    ax.set_xlabel('cluster')
    ax.set_ylabel('fraction of epochs')
    ax.set_title(f'{method} K={K}: diagnosis mix per cluster')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_xticks(np.arange(M.shape[0]))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'stacked_bar_{method}_K{K}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    colsum = M.sum(axis=0, keepdims=True).clip(min=1)
    norm_by_dx = M / colsum
    fig, ax = plt.subplots(figsize=(8, max(4, M.shape[0] * 0.4)))
    im = ax.imshow(norm_by_dx, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(dx_order)))
    ax.set_xticklabels(dx_order, rotation=30, ha='right')
    ax.set_yticks(range(M.shape[0]))
    ax.set_xlabel('diagnosis')
    ax.set_ylabel('cluster')
    ax.set_title(f'{method} K={K}: P(cluster | diagnosis)')
    for i in range(M.shape[0]):
        for j in range(len(dx_order)):
            v = norm_by_dx[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color='white' if v < 0.5 else 'black', fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'heatmap_{method}_K{K}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ----------------------------------------------------------- centroid plotting
def compute_and_plot_centroids(labels, meta, npz_index, out_dir, method_tag):
    """For each cluster, average the *original* (256,256) wSMI matrices of the
    epochs assigned to that cluster, then plot.

    Streamed by session — only one session's npz is in RAM at a time.
    """
    K = int(np.max(labels)) + 1
    n_nodes = 256
    accum = np.zeros((K, n_nodes, n_nodes), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)

    # Group meta entries by (subject, session) so we open each npz once.
    bucket = {}
    for m, c in zip(meta, labels):
        key = (m['subject_id'], m['session_num'])
        bucket.setdefault(key, []).append((m['matrix_idx'], int(c)))

    print(f"  computing centroids for {K} clusters across {len(bucket)} sessions ...")
    t0 = time.time()
    for i, ((sid, snum), entries) in enumerate(bucket.items()):
        path = npz_index.get((sid, snum))
        if path is None:
            print(f"  WARNING: no npz path for sub-{sid}/ses-{snum}, skipping")
            continue
        arr = np.load(path, mmap_mode='r')  # (n_eps, 256, 256)
        for mi, c in entries:
            accum[c] += arr[mi]
            counts[c] += 1
        del arr
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(bucket)} sessions processed, "
                  f"{(time.time()-t0)/60:.1f} min")

    centroids = accum / counts[:, None, None].clip(min=1)
    np.save(os.path.join(out_dir, f'centroids_{method_tag}_wsmi.npy'), centroids.astype(np.float32))
    np.save(os.path.join(out_dir, f'centroids_{method_tag}_counts.npy'), counts)

    # Grand mean for reference
    grand_mean = (accum.sum(axis=0) / counts.sum()).astype(np.float32) if counts.sum() else None

    # 1) Per-cluster centroid heatmaps (absolute)
    vmin = float(np.percentile(centroids, 2))
    vmax = float(np.percentile(centroids, 98))
    cols = 2
    rows = int(np.ceil(K / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    axes = np.atleast_2d(axes).ravel()
    for c in range(K):
        ax = axes[c]
        im = ax.imshow(centroids[c], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'cluster {c} centroid wSMI  (n={counts[c]:,} epochs)')
        ax.set_xlabel('electrode j'); ax.set_ylabel('electrode i')
        plt.colorbar(im, ax=ax, fraction=0.046)
    for c in range(K, len(axes)):
        axes[c].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'centroids_{method_tag}_absolute.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 2) Centroid minus grand-mean — emphasises what makes each cluster unique
    if grand_mean is not None:
        diffs = centroids - grand_mean
        vlim = float(np.percentile(np.abs(diffs), 98))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
        axes = np.atleast_2d(axes).ravel()
        for c in range(K):
            ax = axes[c]
            im = ax.imshow(diffs[c], cmap='RdBu_r', vmin=-vlim, vmax=vlim)
            ax.set_title(f'cluster {c} − grand mean  (n={counts[c]:,})')
            ax.set_xlabel('electrode j'); ax.set_ylabel('electrode i')
            plt.colorbar(im, ax=ax, fraction=0.046)
        for c in range(K, len(axes)):
            axes[c].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'centroids_{method_tag}_vs_grand_mean.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  centroids saved (absolute + diff). counts per cluster: {counts.tolist()}")
    return centroids, counts


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--labels_csv',
                    default='data/wsmi_res/256_electrodes-20260526T091011Z-3-001/256_electrodes/wsmi_res_DOC/patient_labels.csv')
    ap.add_argument('--output_dir', default='output/full_cluster')
    ap.add_argument('--pca_dim', type=int, default=50)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--ks', type=int, nargs='+', default=[4, 6, 8])
    ap.add_argument('--centroid_for_K', type=int, default=4,
                    help='K value for which to compute & plot wSMI-space centroids (GMM)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/6] loading ALL epochs from disk")
    X_flat, meta, npz_index = load_full_features(args.data_dir)

    print("[2/6] attaching diagnoses")
    meta = attach_diagnoses(meta, args.labels_csv)
    dx_order_all = ['control', 'UWS', 'MCS-', 'MCS+', 'EMCS', 'COMA', 'unknown', 'n/a']
    diagnoses = [m['diagnosis'] for m in meta]
    seen = set(diagnoses)
    dx_order = [d for d in dx_order_all if d in seen]
    print(f"  diagnoses present (ordered): {dx_order}")

    print("[3/6] StandardScaler + PCA (full data)")
    t0 = time.time()
    scaler = StandardScaler(copy=False)
    X_flat = scaler.fit_transform(X_flat)
    print(f"  StandardScaler done in {(time.time()-t0)/60:.1f} min")
    t0 = time.time()
    pca = PCA(n_components=args.pca_dim, random_state=args.random_state, svd_solver='randomized')
    X_pca = pca.fit_transform(X_flat).astype(np.float32)
    print(f"  PCA done in {(time.time()-t0)/60:.1f} min")
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA({args.pca_dim}) explained variance: {explained:.4f}")
    # Free memory: X_flat is no longer needed for clustering
    del X_flat

    # Persist preprocessors and projected features
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(args.output_dir, 'pca.joblib'))
    np.save(os.path.join(args.output_dir, 'X_pca.npy'), X_pca)
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    print("[4/6] KMeans across K-values")
    km_summary = []
    for K in args.ks:
        t0 = time.time()
        km = KMeans(n_clusters=K, n_init=10, random_state=args.random_state)
        lbl = km.fit_predict(X_pca)
        M, stats = contingency_and_stats(lbl, diagnoses, dx_order)
        save_diagnostic_plots(M, dx_order, 'kmeans', K, args.output_dir)
        np.savetxt(os.path.join(args.output_dir, f'contingency_kmeans_K{K}.csv'),
                   M, fmt='%d', delimiter=',', header=','.join(dx_order), comments='')
        np.save(os.path.join(args.output_dir, f'labels_kmeans_K{K}.npy'), lbl)
        joblib.dump(km, os.path.join(args.output_dir, f'model_kmeans_K{K}.joblib'))
        print(f"  KMeans K={K}: chi2={stats['chi2']:.1f}, p={stats['p']:.2e}, "
              f"V={stats['cramers_v']:.3f}, fit_time={(time.time()-t0):.0f}s")
        km_summary.append({'method': 'kmeans', 'K': K, **stats})

    print("[5/6] GMM across K-values")
    gmm_summary = []
    gmm_models = {}
    gmm_labels = {}
    for K in args.ks:
        t0 = time.time()
        gm = GaussianMixture(n_components=K, covariance_type='full',
                             random_state=args.random_state, n_init=3,
                             max_iter=200, reg_covar=1e-4)
        gm.fit(X_pca)
        lbl = gm.predict(X_pca)
        M, stats = contingency_and_stats(lbl, diagnoses, dx_order)
        stats['bic'] = float(gm.bic(X_pca))
        stats['aic'] = float(gm.aic(X_pca))
        save_diagnostic_plots(M, dx_order, 'gmm', K, args.output_dir)
        np.savetxt(os.path.join(args.output_dir, f'contingency_gmm_K{K}.csv'),
                   M, fmt='%d', delimiter=',', header=','.join(dx_order), comments='')
        np.save(os.path.join(args.output_dir, f'labels_gmm_K{K}.npy'), lbl)
        joblib.dump(gm, os.path.join(args.output_dir, f'model_gmm_K{K}.joblib'))
        gmm_models[K] = gm
        gmm_labels[K] = lbl
        print(f"  GMM K={K}: chi2={stats['chi2']:.1f}, p={stats['p']:.2e}, "
              f"V={stats['cramers_v']:.3f}, BIC={stats['bic']:.0f}, "
              f"fit_time={(time.time()-t0):.0f}s")
        gmm_summary.append({'method': 'gmm', 'K': K, **stats})

    print(f"[6/6] computing per-cluster wSMI centroids for GMM K={args.centroid_for_K}")
    centroids, counts = compute_and_plot_centroids(
        gmm_labels[args.centroid_for_K], meta, npz_index,
        args.output_dir, method_tag=f'gmm_K{args.centroid_for_K}',
    )

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump({
            'config': vars(args),
            'sklearn_version': sklearn.__version__,
            'n_total_epochs': int(X_pca.shape[0]),
            'n_subjects_total': len({m['subject_id'] for m in meta}),
            'n_sessions_total': len(npz_index),
            'pca_explained_variance': explained,
            'diagnosis_counts': dict(Counter(diagnoses)),
            'methods': km_summary + gmm_summary,
            'centroid_for_K': args.centroid_for_K,
            'centroid_cluster_counts': counts.tolist(),
        }, f, indent=2)

    print("\n=== summary ===")
    for s in km_summary + gmm_summary:
        bic = f", BIC={s.get('bic'):.0f}" if 'bic' in s else ""
        print(f"  {s['method']:>7s}  K={s['K']}  chi2={s['chi2']:8.1f}  "
              f"p={s['p']:.2e}  V={s['cramers_v']:.3f}{bic}")
    print(f"\nartifacts: {args.output_dir}/")


if __name__ == '__main__':
    main()

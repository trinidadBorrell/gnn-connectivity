#!/usr/bin/env python3
"""
Compute per-cluster wSMI centroid matrices from already-saved cluster labels.

This is the standalone version of the centroid step in scripts/full_wsmi_cluster.py
— useful when the heavy steps (loading, PCA, GMM fitting) have already run and
you only need to redo or recompute centroids.

Inputs (all in --cluster_dir):
  labels_<method>_K<K>.npy   (n_total_epochs,) int — per-epoch cluster id
  meta.json                  list of {subject_id, session_num, cohort, matrix_idx, ...}

Output (in same dir):
  centroids_<method>_K<K>_wsmi.npy           (K, 256, 256) float32
  centroids_<method>_K<K>_counts.npy         (K,) int
  centroids_<method>_K<K>_absolute.png       raw mean wSMI per cluster
  centroids_<method>_K<K>_vs_grand_mean.png  centroid minus grand mean
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import EEGtoGraph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--cluster_dir', default='output/full_cluster')
    ap.add_argument('--labels_file', default='labels_gmm_K4.npy',
                    help='Filename of the labels array inside --cluster_dir')
    ap.add_argument('--method_tag', default='gmm_K4',
                    help='Tag used for output filenames')
    args = ap.parse_args()

    print(f"[1/3] loading labels + meta from {args.cluster_dir}")
    labels = np.load(os.path.join(args.cluster_dir, args.labels_file))
    with open(os.path.join(args.cluster_dir, 'meta.json')) as f:
        meta = json.load(f)
    assert len(labels) == len(meta), f"length mismatch: {len(labels)} vs {len(meta)}"
    K = int(labels.max()) + 1
    print(f"  {len(labels):,} epochs, K={K}, cluster counts={np.bincount(labels).tolist()}")

    # Re-enumerate to get the npz path for each (sub, ses) — same order as during clustering
    sessions = EEGtoGraph.enumerate_matrix_sessions(args.data_dir)
    npz_index = {(sid, snum): source['path']
                 for sid, snum, source in sessions
                 if source['kind'] == 'npz'}

    print(f"[2/3] streaming sessions, accumulating per-cluster sums")
    accum = np.zeros((K, 256, 256), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)

    # Group meta entries by (subject, session) so we open each npz exactly once
    bucket: dict = {}
    for i, m in enumerate(meta):
        key = (m['subject_id'], m['session_num'])
        bucket.setdefault(key, []).append((m['matrix_idx'], int(labels[i])))

    t0 = time.time()
    for i, ((sid, snum), entries) in enumerate(bucket.items()):
        path = npz_index.get((sid, snum))
        if path is None:
            print(f"  WARNING: no npz path for sub-{sid}/ses-{snum}, skipping")
            continue
        # Load the full session array (np.load on .npz returns NpzFile; access
        # the 'data' key to materialise the (n_eps, 256, 256) ndarray).
        with np.load(path) as d:
            arr = d['data']
            # vectorised per-cluster accumulation: for each unique cluster id
            # present in this session, sum the matrices with that id in one go
            mi_arr = np.array([mi for mi, _ in entries], dtype=np.int64)
            c_arr = np.array([c for _, c in entries], dtype=np.int64)
            for c in range(K):
                sel = mi_arr[c_arr == c]
                if sel.size:
                    accum[c] += arr[sel].sum(axis=0)
                    counts[c] += sel.size
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(bucket)} sessions processed, {(time.time()-t0)/60:.1f} min")

    centroids = accum / counts[:, None, None].clip(min=1)
    np.save(os.path.join(args.cluster_dir, f'centroids_{args.method_tag}_wsmi.npy'),
            centroids.astype(np.float32))
    np.save(os.path.join(args.cluster_dir, f'centroids_{args.method_tag}_counts.npy'), counts)
    print(f"  centroids: shape={centroids.shape}, counts={counts.tolist()}")

    print(f"[3/3] plotting")
    grand_mean = (accum.sum(axis=0) / counts.sum()).astype(np.float32) if counts.sum() else None

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
    plt.savefig(os.path.join(args.cluster_dir, f'centroids_{args.method_tag}_absolute.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

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
        plt.savefig(os.path.join(args.cluster_dir, f'centroids_{args.method_tag}_vs_grand_mean.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  centroid plots saved to {args.cluster_dir}/")
    print(f"  files: centroids_{args.method_tag}_absolute.png, "
          f"centroids_{args.method_tag}_vs_grand_mean.png")


if __name__ == '__main__':
    main()

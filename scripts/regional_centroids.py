#!/usr/bin/env python3
"""
Regional decomposition of the per-cluster wSMI centroid matrices.

For each cluster's (256, 256) centroid matrix:
  (A) Assign each electrode to one of 5 anatomical regions (F/C/P/T/O) based
      on its (x, y, z) coordinate.
  (B) Reorder rows and columns by region; render the 256x256 matrix with
      grid lines + region labels at the boundaries.
  (C) Compute the (5, 5) region-level mean wSMI per (region_i, region_j)
      pair and render as a compact heatmap — much easier to interpret
      ("how is the average F-O coupling in cluster k?").

Also produces a sanity-check scalp plot showing the region assignment per
electrode (top-down view).

Inputs:
  --coords_file   electrode XYZ file (default data_scalp/GSN-HydroCel-257.txt)
  --centroid_dir  output of centroid_last_100_balanced.py
                  (must contain centroids_wsmi.npy + centroids_counts.npy)
  --output_dir    where to dump plots

Output filenames are tagged with the centroid_dir's basename so K=3 and K=4
runs don't collide.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ----------------------------------------------------------- region assignment
REGION_ORDER = ['F', 'C', 'P', 'T', 'O']
REGION_LONG = {
    'F': 'Frontal',
    'C': 'Central',
    'P': 'Parietal',
    'T': 'Temporal',
    'O': 'Occipital',
}
REGION_COLOUR = {
    'F': '#e57373',  # red
    'C': '#fdd835',  # yellow
    'P': '#81c784',  # green
    'T': '#64b5f6',  # blue
    'O': '#ba68c8',  # purple
}


def assign_region(row):
    """Coordinate-based 5-region assignment on the GSN HydroCel 256-channel
    montage. Boundaries chosen by inspection of the actual XYZ distribution:
      x: -8..+8  (left/right)
      y: -10..+11 (posterior/anterior)
      z: -10..+9  (inferior/superior)
    Regions:
      F  — Frontal: anterior, including frontal poles
      C  — Central: midline / superior, around the vertex
      P  — Parietal: superior-posterior
      T  — Temporal: lateral and inferior (the temporal "shelf")
      O  — Occipital: most posterior
    """
    x, y, z = row['x'], row['y'], row['z']
    # Temporal: lateral + low (the side / under-ear electrodes)
    if abs(x) > 5.5 and z < -1.5:
        return 'T'
    # Occipital: most-posterior pole
    if y < -6.0:
        return 'O'
    # Frontal: anterior half (above y=3 approx)
    if y > 3.0:
        return 'F'
    # Parietal: posterior superior (negative y, above midline)
    if y < -1.0 and z > 0.0:
        return 'P'
    # Default = Central
    return 'C'


def reorder_by_region(df_coords):
    """Return permutation (length 256) that orders electrodes by region
    (F → C → P → T → O), then by 'y' (anterior to posterior) within each
    region for a stable secondary order.
    """
    df = df_coords.copy()
    df['region'] = df.apply(assign_region, axis=1)
    df['orig_idx'] = np.arange(len(df))
    region_rank = {r: i for i, r in enumerate(REGION_ORDER)}
    df['region_rank'] = df['region'].map(region_rank)
    df_sorted = df.sort_values(['region_rank', 'y'], ascending=[True, False])
    perm = df_sorted['orig_idx'].to_numpy()
    region_assignment = df['region'].to_numpy()
    return perm, region_assignment


def region_boundaries(region_assignment_sorted):
    """Given the sorted-by-region array of region labels (length 256), return
    a dict {region: (start_idx, end_idx)} giving the contiguous block where
    each region lives in the sorted order.
    """
    bounds = {}
    cur = region_assignment_sorted[0]
    start = 0
    for i, r in enumerate(region_assignment_sorted):
        if r != cur:
            bounds[cur] = (start, i)
            cur = r
            start = i
    bounds[cur] = (start, len(region_assignment_sorted))
    return bounds


# ---------------------------------------------------------------- plotting
def plot_sanity_check_scalp(df_coords, out_path):
    """Top-down scatter of electrodes, coloured by assigned region — verifies
    the boundary heuristics are reasonable.
    """
    df = df_coords.copy()
    df['region'] = df.apply(assign_region, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    # Top-down: x-y plane
    ax = axes[0]
    for r in REGION_ORDER:
        m = df['region'] == r
        ax.scatter(df.loc[m, 'x'], df.loc[m, 'y'], c=REGION_COLOUR[r], s=40,
                   label=f"{r} – {REGION_LONG[r]}  (n={m.sum()})", edgecolor='black', linewidth=0.3)
    ax.set_xlabel('x (left → right)')
    ax.set_ylabel('y (posterior → anterior)')
    ax.set_title('Top-down view  (z ignored)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
    # Side view: x-z plane (head viewed from the right)
    ax = axes[1]
    for r in REGION_ORDER:
        m = df['region'] == r
        ax.scatter(df.loc[m, 'y'], df.loc[m, 'z'], c=REGION_COLOUR[r], s=40,
                   edgecolor='black', linewidth=0.3)
    ax.set_xlabel('y (posterior → anterior)')
    ax.set_ylabel('z (inferior → superior)')
    ax.set_title('Side view  (x ignored)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reordered_matrices(centroids, counts, perm, bounds, out_path,
                            mode='absolute', grand_mean=None):
    """Plot per-cluster centroid matrices, rows/cols reordered by region,
    with region boundaries drawn as grid lines + labels at the edges.

    mode='absolute' → viridis, value range from data percentiles
    mode='diff'     → RdBu_r, symmetric range from |diff| percentile
    """
    K = centroids.shape[0]
    # Reorder all cluster matrices
    reord = centroids[:, perm, :][:, :, perm]
    if mode == 'diff':
        if grand_mean is None:
            raise ValueError("diff mode requires grand_mean")
        reord = reord - grand_mean[perm, :][:, perm]
        cmap = 'RdBu_r'
        vmax = float(np.percentile(np.abs(reord), 98))
        vmin = -vmax
        cb_label = 'wSMI − grand mean'
    else:
        cmap = 'viridis'
        vmin = float(np.percentile(reord, 2))
        vmax = float(np.percentile(reord, 98))
        cb_label = 'mean wSMI'

    rows, cols = (1, K) if K <= 4 else (2, int(np.ceil(K / 2)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.7))
    axes = np.atleast_1d(axes).ravel()
    for c in range(K):
        ax = axes[c]
        im = ax.imshow(reord[c], cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        # Region grid lines
        for r, (s, e) in bounds.items():
            ax.axhline(s - 0.5, color='black', lw=1.0, alpha=0.6)
            ax.axvline(s - 0.5, color='black', lw=1.0, alpha=0.6)
        # Final boundary lines
        n = reord.shape[-1]
        # Labels at midpoints of each region block (on both axes)
        for r, (s, e) in bounds.items():
            mid = (s + e) / 2
            ax.text(mid, -8, r, ha='center', va='top', fontsize=12,
                    fontweight='bold', color=REGION_COLOUR[r])
            ax.text(-8, mid, r, ha='right', va='center', fontsize=12,
                    fontweight='bold', color=REGION_COLOUR[r], rotation=90)
        title = (f"cluster {c}  centroid wSMI  (n={counts[c]:,})"
                 if mode == 'absolute'
                 else f"cluster {c} − grand mean  (n={counts[c]:,})")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, label=cb_label)
    for c in range(K, len(axes)):
        axes[c].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_region_summary(centroids, counts, region_assignment, out_path,
                        mode='absolute', grand_mean=None):
    """Compute the (n_regions, n_regions) mean-wSMI matrix per cluster and
    plot as a compact heatmap with numeric annotations.
    """
    K = centroids.shape[0]
    R = len(REGION_ORDER)
    # Build per-electrode → region index
    region_idx = np.array([REGION_ORDER.index(r) for r in region_assignment])
    # Mean wSMI in each (region_i, region_j) block
    region_mat = np.zeros((K, R, R), dtype=np.float64)
    for k in range(K):
        for ri in range(R):
            for rj in range(R):
                m_i = region_idx == ri
                m_j = region_idx == rj
                if m_i.any() and m_j.any():
                    block = centroids[k][np.ix_(m_i, m_j)]
                    # exclude diagonal when ri==rj (self-wSMI is 0)
                    if ri == rj:
                        idx_i, idx_j = np.where(m_i)[0], np.where(m_j)[0]
                        sub = centroids[k][np.ix_(idx_i, idx_j)]
                        # mask the diagonal
                        mask = ~np.eye(sub.shape[0], dtype=bool)
                        region_mat[k, ri, rj] = sub[mask].mean()
                    else:
                        region_mat[k, ri, rj] = block.mean()

    if mode == 'diff':
        if grand_mean is None:
            raise ValueError
        gm_region = np.zeros((R, R), dtype=np.float64)
        for ri in range(R):
            for rj in range(R):
                m_i = region_idx == ri
                m_j = region_idx == rj
                if m_i.any() and m_j.any():
                    block = grand_mean[np.ix_(m_i, m_j)]
                    if ri == rj:
                        idx_i, idx_j = np.where(m_i)[0], np.where(m_j)[0]
                        sub = grand_mean[np.ix_(idx_i, idx_j)]
                        mask = ~np.eye(sub.shape[0], dtype=bool)
                        gm_region[ri, rj] = sub[mask].mean()
                    else:
                        gm_region[ri, rj] = block.mean()
        plot_mat = region_mat - gm_region
        cmap = 'RdBu_r'
        vmax = float(np.max(np.abs(plot_mat)))
        vmin = -vmax
        cb_label = 'mean wSMI − grand mean'
    else:
        plot_mat = region_mat
        cmap = 'viridis'
        vmin = float(plot_mat.min())
        vmax = float(plot_mat.max())
        cb_label = 'mean wSMI'

    rows, cols = (1, K) if K <= 4 else (2, int(np.ceil(K / 2)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.0))
    axes = np.atleast_1d(axes).ravel()
    for c in range(K):
        ax = axes[c]
        im = ax.imshow(plot_mat[c], cmap=cmap, vmin=vmin, vmax=vmax)
        for ri in range(R):
            for rj in range(R):
                v = plot_mat[c, ri, rj]
                ax.text(rj, ri, f'{v:+.3f}' if mode == 'diff' else f'{v:.3f}',
                        ha='center', va='center', fontsize=10,
                        color='white' if (cmap == 'viridis' and v < (vmin + vmax) / 2)
                              or (cmap == 'RdBu_r' and abs(v) > 0.66 * vmax)
                              else 'black')
        ax.set_xticks(range(R)); ax.set_yticks(range(R))
        ax.set_xticklabels(REGION_ORDER); ax.set_yticklabels(REGION_ORDER)
        title = (f"cluster {c}  (n={counts[c]:,})"
                 if mode == 'absolute'
                 else f"cluster {c} − grand mean  (n={counts[c]:,})")
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, label=cb_label)
    for c in range(K, len(axes)):
        axes[c].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return region_mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coords_file', default='data_scalp/GSN-HydroCel-257.txt')
    ap.add_argument('--centroid_dir', required=True,
                    help='Dir containing centroids_wsmi.npy + centroids_counts.npy')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--tag', default=None,
                    help='Suffix for output filenames (default: basename of centroid_dir)')
    args = ap.parse_args()

    tag = args.tag or os.path.basename(os.path.normpath(args.centroid_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    # Load coords
    df_coords = pd.read_csv(args.coords_file, sep=r'\s+', header=None,
                            names=['label', 'x', 'y', 'z'])
    print(f"[1/4] loaded {len(df_coords)} electrodes from {args.coords_file}")
    perm, region_assignment = reorder_by_region(df_coords)
    region_assignment_sorted = region_assignment[perm]
    bounds = region_boundaries(region_assignment_sorted)
    print(f"  region assignment:")
    for r in REGION_ORDER:
        s, e = bounds[r]
        print(f"    {r} ({REGION_LONG[r]}): {e - s} electrodes")

    # Sanity-check scatter
    plot_sanity_check_scalp(
        df_coords, os.path.join(args.output_dir, f'region_assignment_{tag}.png'))

    # Load centroids
    print(f"[2/4] loading centroids from {args.centroid_dir}")
    centroids = np.load(os.path.join(args.centroid_dir, 'centroids_wsmi.npy'))
    counts = np.load(os.path.join(args.centroid_dir, 'centroids_counts.npy'))
    K = centroids.shape[0]
    print(f"  centroids: {centroids.shape}, K={K}, counts={counts.tolist()}")

    # Grand mean weighted by counts
    grand_mean = (centroids * counts[:, None, None]).sum(axis=0) / counts.sum()

    # Plot reordered 256x256 matrices
    print(f"[3/4] plotting reordered 256x256 matrices")
    plot_reordered_matrices(
        centroids, counts, perm, bounds,
        os.path.join(args.output_dir, f'centroids_regional_absolute_{tag}.png'),
        mode='absolute')
    plot_reordered_matrices(
        centroids, counts, perm, bounds,
        os.path.join(args.output_dir, f'centroids_regional_vs_grand_mean_{tag}.png'),
        mode='diff', grand_mean=grand_mean)

    # Plot 5x5 region summary
    print(f"[4/4] computing & plotting 5x5 region-level summary")
    region_mat_abs = plot_region_summary(
        centroids, counts, region_assignment,
        os.path.join(args.output_dir, f'region_summary_absolute_{tag}.png'),
        mode='absolute')
    region_mat_diff = plot_region_summary(
        centroids, counts, region_assignment,
        os.path.join(args.output_dir, f'region_summary_vs_grand_mean_{tag}.png'),
        mode='diff', grand_mean=grand_mean)

    # Dump per-cluster region summaries as JSON for downstream eyeballing
    summary = {
        'config': vars(args),
        'tag': tag,
        'K': int(K),
        'cluster_counts': counts.tolist(),
        'region_order': REGION_ORDER,
        'region_long': REGION_LONG,
        'region_sizes': {r: int(bounds[r][1] - bounds[r][0]) for r in REGION_ORDER},
        'region_matrix_absolute': region_mat_abs.tolist(),
        'region_matrix_vs_grand_mean': region_mat_diff.tolist(),
    }
    with open(os.path.join(args.output_dir, f'region_summary_{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  artefacts in {args.output_dir}/  with tag '{tag}'")
    print(f"  region sanity check:    region_assignment_{tag}.png")
    print(f"  256x256 reordered:      centroids_regional_(absolute|vs_grand_mean)_{tag}.png")
    print(f"  5x5 region summary:     region_summary_(absolute|vs_grand_mean)_{tag}.png")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Manifold visualisations of the wSMI clustering result.

Goal: see whether the unsupervised clusters live on a 1-D-or-2-D continuum
that aligns with the clinical consciousness gradient
(control → EMCS → MCS+ → MCS- → UWS → COMA), and whether MCS patients sit
literally between controls and UWS in the manifold.

Renders (in --output_dir):
  pca2_clusters.png           PCA(2) of X_pca, points coloured by GMM cluster id,
                              with GMM Gaussians projected as 2σ ellipses.
  pca2_diagnosis.png          PCA(2), points coloured by diagnosis with an ordinal
                              palette (control = light, COMA = dark) + per-diagnosis
                              centroid arrows showing the average-position trajectory.
  umap2_clusters.png          UMAP(2) of X_pca, points coloured by cluster.
  umap2_diagnosis.png         UMAP(2), points coloured by diagnosis + trajectory.
  pca2_subjects.png           Per-subject AVERAGE position in PCA(2), one point per
                              subject. Shows whether the patient-level geometry is
                              clean (less noisy than per-epoch).
  umap2_subjects.png          Same in UMAP(2).
  consciousness_axis.png      A 1-D distillation: each subject's projection on the
                              line connecting the control centroid to the UWS centroid
                              in PCA space, histogrammed by diagnosis.
  summary.json                Per-diagnosis centroids, axis correlations, etc.
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
from matplotlib.patches import Ellipse
from matplotlib import cm

from sklearn.decomposition import PCA
import umap


DX_ORDER = ['control', 'EMCS', 'MCS+', 'MCS-', 'UWS', 'COMA']  # ordinal consciousness gradient
# Sequential palette: light (conscious) -> dark (unconscious)
def dx_palette():
    cmap = cm.get_cmap('viridis_r', len(DX_ORDER))
    return {dx: cmap(i) for i, dx in enumerate(DX_ORDER)}


def take_last_n_per_session_idx(df, n):
    out = []
    for _, gdf in df.groupby(['subject_id', 'session_num'], sort=False):
        out.append(gdf.iloc[-n:].index.to_numpy() if len(gdf) >= n
                   else gdf.index.to_numpy())
    return np.concatenate(out)


def project_gmm_to_pca2(gmm_means_50d, gmm_covs_50d, pca2_components):
    """Project the 50-D GMM Gaussians to 2D PCA space.
    pca2_components: (2, 50) — the rows are the top-2 principal axes.
    Returns (means_2d, covs_2d) for each component.
    """
    W = pca2_components  # (2, 50)
    means_2d = gmm_means_50d @ W.T  # (K, 2)
    covs_2d = []
    for cov in gmm_covs_50d:
        cov2 = W @ cov @ W.T  # (2, 2)
        covs_2d.append(cov2)
    return means_2d, np.stack(covs_2d, axis=0)


def plot_gaussian_ellipse(ax, mean, cov, color, n_std=2.0, **kw):
    """Plot a 2D Gaussian as an n_std confidence ellipse."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    e = Ellipse(xy=mean, width=width, height=height, angle=theta,
                edgecolor=color, facecolor='none', lw=2.5, **kw)
    ax.add_patch(e)


def plot_dx_trajectory(ax, dx_centroids, palette):
    """Draw arrows connecting consecutive diagnoses in clinical order to
    show whether the manifold has a monotone gradient."""
    pts = np.array([dx_centroids[dx] for dx in DX_ORDER])
    # Draw lines
    ax.plot(pts[:, 0], pts[:, 1], color='black', lw=2.0, alpha=0.5, zorder=3)
    # Big markers for diagnosis centroids
    for i, dx in enumerate(DX_ORDER):
        ax.scatter(*pts[i], s=380, color=palette[dx], edgecolor='black',
                   linewidth=2, zorder=4, label=f'{dx} (centroid)')
        ax.annotate(dx, pts[i], xytext=(8, 8), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85))


def scatter_by_dx(ax, X2, dx_arr, palette, alpha=0.25, size=4, max_per_dx=None,
                  rng=None):
    """Scatter epochs in 2D, coloured by diagnosis. Optionally subsample per
    diagnosis to keep the plot legible.
    """
    for dx in DX_ORDER:
        mask = (dx_arr == dx)
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        if max_per_dx is not None and idx.size > max_per_dx:
            idx = rng.choice(idx, size=max_per_dx, replace=False)
        ax.scatter(X2[idx, 0], X2[idx, 1], s=size, color=palette[dx],
                   alpha=alpha, label=f'{dx}  (n={mask.sum()})')


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--cluster_dir', default='output/full_cluster')
    ap.add_argument('--centroid_dir', default='output/centroid_last_100_balanced')
    ap.add_argument('--output_dir', default='output/manifold_viz')
    ap.add_argument('--last_n', type=int, default=100)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--max_epochs_per_dx', type=int, default=2000,
                    help='Cap per-diagnosis points in scatter plots (legibility).')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_state)
    palette = dx_palette()

    print("[1/6] loading X_pca, meta, and last_100 cluster labels")
    X_pca_full = np.load(os.path.join(args.cluster_dir, 'X_pca.npy'))
    with open(os.path.join(args.cluster_dir, 'meta.json')) as f:
        meta = json.load(f)
    df = pd.DataFrame(meta)
    df['matrix_idx'] = df['matrix_idx'].astype(int)
    df = df.sort_values(['subject_id', 'session_num', 'matrix_idx']).reset_index()
    df = df.rename(columns={'index': 'orig_index'})

    # Build the last_N subset in the same order used by centroid_last_100_balanced.py
    last_idx_sorted = take_last_n_per_session_idx(df, args.last_n)
    df_last = df.loc[last_idx_sorted].reset_index(drop=True)
    last_orig_rows = df_last['orig_index'].to_numpy()
    X = X_pca_full[last_orig_rows]  # (~17500, 50)
    labels = np.load(os.path.join(args.centroid_dir, 'labels_last_100.npy'))
    assert len(labels) == len(df_last), \
        f"label count {len(labels)} != last_N count {len(df_last)}"
    K = int(labels.max()) + 1
    dx_arr = df_last['diagnosis'].to_numpy()
    print(f"  X: {X.shape}, K={K}, dx counts: {Counter(dx_arr.tolist())}")

    print(f"[2/6] refitting GMM (deterministic) to recover means + covariances for ellipses")
    from sklearn.mixture import GaussianMixture
    # We need access to means_ and covariances_; the saved model is in summary.json only.
    # Recompute by re-fitting on the balanced subset.
    counts_per_dx = Counter(dx_arr.tolist())
    target = min(counts_per_dx[d] for d in DX_ORDER if d in counts_per_dx)
    bal_idx = []
    for dx in DX_ORDER:
        m = (dx_arr == dx)
        idxs = np.where(m)[0]
        if idxs.size <= target:
            bal_idx.append(idxs)
        else:
            bal_idx.append(rng.choice(idxs, size=target, replace=False))
    bal_idx = np.concatenate(bal_idx)
    gm = GaussianMixture(n_components=K, covariance_type='full',
                         random_state=args.random_state, n_init=3,
                         max_iter=200, reg_covar=1e-3)
    gm.fit(X[bal_idx])
    print(f"  GMM refit: means {gm.means_.shape}, covariances {gm.covariances_.shape}")

    print("[3/6] PCA(2) of X_pca")
    pca2 = PCA(n_components=2, random_state=args.random_state).fit(X)
    X_pca2 = pca2.transform(X)
    var_explained = pca2.explained_variance_ratio_
    print(f"  PCA(2) variance explained: {var_explained.tolist()} "
          f"(sum {var_explained.sum():.3f})")
    means_2d, covs_2d = project_gmm_to_pca2(gm.means_, gm.covariances_, pca2.components_)

    print("[4/6] UMAP(2) of X_pca (~1 min for 17.5k points)")
    t0 = time.time()
    um = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                   metric='euclidean', random_state=args.random_state, verbose=False)
    X_umap = um.fit_transform(X)
    print(f"  UMAP done in {(time.time()-t0):.1f}s")

    # --------------- PCA(2): clusters + Gaussians
    print("[5/6] rendering plots")
    fig, ax = plt.subplots(figsize=(9, 8))
    cluster_palette = cm.get_cmap('tab10', K)
    for c in range(K):
        m = (labels == c)
        sub = X_pca2[m]
        if sub.shape[0] > args.max_epochs_per_dx * 2:
            keep = rng.choice(sub.shape[0], size=args.max_epochs_per_dx * 2, replace=False)
            sub = sub[keep]
        ax.scatter(sub[:, 0], sub[:, 1], s=4, color=cluster_palette(c),
                   alpha=0.25, label=f'cluster {c} (n={m.sum()})')
        plot_gaussian_ellipse(ax, means_2d[c], covs_2d[c], cluster_palette(c), n_std=2.0)
        ax.scatter(*means_2d[c], s=200, color=cluster_palette(c),
                   edgecolor='black', linewidth=2, zorder=5)
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
    ax.set_title(f'PCA(2) — epochs by GMM cluster + 2σ Gaussians')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'pca2_clusters.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # PCA(2) by diagnosis + trajectory
    dx_centroids_pca = {dx: X_pca2[dx_arr == dx].mean(axis=0)
                        for dx in DX_ORDER if (dx_arr == dx).any()}
    fig, ax = plt.subplots(figsize=(9, 8))
    scatter_by_dx(ax, X_pca2, dx_arr, palette, alpha=0.25, size=4,
                  max_per_dx=args.max_epochs_per_dx, rng=rng)
    plot_dx_trajectory(ax, dx_centroids_pca, palette)
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
    ax.set_title('PCA(2) — epochs by diagnosis (ordinal palette) + diagnosis trajectory')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'pca2_diagnosis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # UMAP(2) — clusters
    fig, ax = plt.subplots(figsize=(9, 8))
    for c in range(K):
        m = (labels == c)
        sub = X_umap[m]
        if sub.shape[0] > args.max_epochs_per_dx * 2:
            keep = rng.choice(sub.shape[0], size=args.max_epochs_per_dx * 2, replace=False)
            sub = sub[keep]
        ax.scatter(sub[:, 0], sub[:, 1], s=4, color=cluster_palette(c),
                   alpha=0.3, label=f'cluster {c} (n={m.sum()})')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.set_title('UMAP(2) of X_pca — by GMM cluster')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'umap2_clusters.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # UMAP(2) — diagnosis + trajectory
    dx_centroids_umap = {dx: X_umap[dx_arr == dx].mean(axis=0)
                         for dx in DX_ORDER if (dx_arr == dx).any()}
    fig, ax = plt.subplots(figsize=(9, 8))
    scatter_by_dx(ax, X_umap, dx_arr, palette, alpha=0.3, size=4,
                  max_per_dx=args.max_epochs_per_dx, rng=rng)
    plot_dx_trajectory(ax, dx_centroids_umap, palette)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.set_title('UMAP(2) of X_pca — by diagnosis + trajectory')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'umap2_diagnosis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Per-subject means
    df_last['pca_x'] = X_pca2[:, 0]; df_last['pca_y'] = X_pca2[:, 1]
    df_last['umap_x'] = X_umap[:, 0]; df_last['umap_y'] = X_umap[:, 1]
    subj_mean = df_last.groupby('subject_id').agg(
        pca_x=('pca_x', 'mean'), pca_y=('pca_y', 'mean'),
        umap_x=('umap_x', 'mean'), umap_y=('umap_y', 'mean'),
        diagnosis=('diagnosis', lambda s: Counter(s).most_common(1)[0][0])
    ).reset_index()

    # Subject-level PCA(2)
    fig, ax = plt.subplots(figsize=(9, 8))
    for dx in DX_ORDER:
        m = (subj_mean['diagnosis'] == dx).to_numpy()
        if not m.any(): continue
        ax.scatter(subj_mean.loc[m, 'pca_x'], subj_mean.loc[m, 'pca_y'],
                   s=80, color=palette[dx], edgecolor='black',
                   linewidth=0.5, label=f'{dx} (n={m.sum()})')
    plot_dx_trajectory(ax, dx_centroids_pca, palette)
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
    ax.set_title('PCA(2) — per-subject average position + trajectory')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'pca2_subjects.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Subject-level UMAP(2)
    fig, ax = plt.subplots(figsize=(9, 8))
    for dx in DX_ORDER:
        m = (subj_mean['diagnosis'] == dx).to_numpy()
        if not m.any(): continue
        ax.scatter(subj_mean.loc[m, 'umap_x'], subj_mean.loc[m, 'umap_y'],
                   s=80, color=palette[dx], edgecolor='black',
                   linewidth=0.5, label=f'{dx} (n={m.sum()})')
    plot_dx_trajectory(ax, dx_centroids_umap, palette)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.set_title('UMAP(2) — per-subject average position + trajectory')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'umap2_subjects.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --------------- Consciousness axis (1-D distillation)
    print("[6/6] computing 'consciousness axis' (control–UWS line in PCA space)")
    if 'control' in dx_centroids_pca and 'UWS' in dx_centroids_pca:
        c_ctrl = dx_centroids_pca['control']
        c_uws = dx_centroids_pca['UWS']
        axis = c_uws - c_ctrl
        axis_norm = axis / (np.linalg.norm(axis) + 1e-12)
        # Project each subject onto this axis (signed distance from control centroid)
        subj_pos = subj_mean[['pca_x', 'pca_y']].to_numpy() - c_ctrl
        subj_axis = subj_pos @ axis_norm
        subj_mean['consciousness_axis'] = subj_axis  # control side ≈ 0, UWS side ≈ ||axis||

        # Histogram by diagnosis
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for dx in DX_ORDER:
            m = (subj_mean['diagnosis'] == dx).to_numpy()
            if not m.any(): continue
            ax.hist(subj_mean.loc[m, 'consciousness_axis'], bins=18,
                    color=palette[dx], alpha=0.5, edgecolor='black', linewidth=0.5,
                    label=f'{dx} (n={m.sum()})')
        ax.axvline(0, color='black', lw=1, ls=':', alpha=0.5)
        ax.set_xlabel('position on control → UWS axis in PCA(2)')
        ax.set_ylabel('# subjects')
        ax.set_title('Subjects projected onto the control–UWS axis (1-D consciousness manifold)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'consciousness_axis.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    else:
        subj_axis = None

    subj_mean.to_csv(os.path.join(args.output_dir, 'subject_positions.csv'),
                     index=False)

    # Summary
    summary = {
        'config': vars(args),
        'pca2_var_explained': var_explained.tolist(),
        'gmm_means_2d': means_2d.tolist(),
        'dx_centroids_pca': {dx: c.tolist() for dx, c in dx_centroids_pca.items()},
        'dx_centroids_umap': {dx: c.tolist() for dx, c in dx_centroids_umap.items()},
        'cluster_sizes': np.bincount(labels).tolist(),
        'diagnosis_counts_per_subject': dict(Counter(subj_mean['diagnosis'])),
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  PCA(2) variance explained: {var_explained[0]:.1%} + {var_explained[1]:.1%} "
          f"= {var_explained.sum():.1%}")
    print(f"\n  diagnosis centroids in PCA(2):")
    for dx in DX_ORDER:
        if dx in dx_centroids_pca:
            print(f"    {dx:>7s}: ({dx_centroids_pca[dx][0]:6.3f}, {dx_centroids_pca[dx][1]:6.3f})")
    print(f"\n  artefacts in {args.output_dir}/")


if __name__ == '__main__':
    main()

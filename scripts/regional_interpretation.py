#!/usr/bin/env python3
"""
Enhanced region-pair interpretation plots for the K=3 clustering.

Builds three new views on top of the basic 5×5 region summary from
scripts/regional_centroids.py:

  (1) **Consciousness sensitivity**: per region pair, the spread of mean wSMI
      across the 3 clusters (max − min). Highlights which couplings vary
      most across the consciousness gradient — the candidate biomarkers.

  (2) **Differential (cl_1 − cl_0)**: the *direct* contrast between the
      control-rich cluster and the UWS-dominated cluster. The cleanest
      "this is what consciousness looks like in wSMI" plot.

  (3) **Top-N region-pair table + barplot**: region pairs ranked by
      sensitivity, highlighting which to focus on for downstream GAE work.

Inputs: the centroid wSMI + per-cluster diagnosis identity from a previous
regional_centroids.py run.

Outputs (in --output_dir, with --tag suffix):
    consciousness_sensitivity_<tag>.png    5×5 max−min heatmap
    cl1_minus_cl0_<tag>.png                control-rich minus UWS-rich diff
    pair_ranking_<tag>.png                 sorted barplot of sensitivity
    region_pair_summary_<tag>.json         numerical table for the chapter
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


REGION_ORDER = ['F', 'C', 'P', 'T', 'O']
REGION_LONG = {
    'F': 'Frontal', 'C': 'Central', 'P': 'Parietal',
    'T': 'Temporal', 'O': 'Occipital',
}
REGION_COLOUR = {
    'F': '#e57373', 'C': '#fdd835', 'P': '#81c784',
    'T': '#64b5f6', 'O': '#ba68c8',
}


def assign_region(row):
    """Match scripts/regional_centroids.py exactly."""
    x, y, z = row['x'], row['y'], row['z']
    if abs(x) > 5.5 and z < -1.5:
        return 'T'
    if y < -6.0:
        return 'O'
    if y > 3.0:
        return 'F'
    if y < -1.0 and z > 0.0:
        return 'P'
    return 'C'


def compute_region_matrices(centroids, region_idx):
    """For each cluster, mean wSMI per (region_i, region_j). Diagonal cells
    exclude the electrode-to-itself zero on the wSMI matrix's diagonal.
    """
    K = centroids.shape[0]
    R = len(REGION_ORDER)
    out = np.zeros((K, R, R), dtype=np.float64)
    for k in range(K):
        for ri in range(R):
            m_i = region_idx == ri
            i_idx = np.where(m_i)[0]
            for rj in range(R):
                m_j = region_idx == rj
                j_idx = np.where(m_j)[0]
                if i_idx.size == 0 or j_idx.size == 0:
                    continue
                sub = centroids[k][np.ix_(i_idx, j_idx)]
                if ri == rj:
                    # Exclude the within-region diagonal (zero by construction)
                    mask = ~np.eye(sub.shape[0], dtype=bool)
                    out[k, ri, rj] = sub[mask].mean()
                else:
                    out[k, ri, rj] = sub.mean()
    return out


def plot_consciousness_sensitivity(region_mat, out_path):
    """5×5 heatmap of max − min across clusters per cell."""
    R = region_mat.shape[1]
    sens = region_mat.max(axis=0) - region_mat.min(axis=0)
    vmax = float(sens.max())
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sens, cmap='magma', vmin=0, vmax=vmax)
    for i in range(R):
        for j in range(R):
            v = sens[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white' if v < 0.5 * vmax else 'black')
    ax.set_xticks(range(R))
    ax.set_yticks(range(R))
    # Coloured region labels at axis ticks
    for i, r in enumerate(REGION_ORDER):
        ax.get_xticklabels()
        ax.get_yticklabels()
    ax.set_xticklabels([REGION_LONG[r] for r in REGION_ORDER], rotation=30, ha='right')
    ax.set_yticklabels([REGION_LONG[r] for r in REGION_ORDER])
    # Tint the tick labels by region
    for tick, r in zip(ax.get_xticklabels(), REGION_ORDER):
        tick.set_color(REGION_COLOUR[r]); tick.set_fontweight('bold')
    for tick, r in zip(ax.get_yticklabels(), REGION_ORDER):
        tick.set_color(REGION_COLOUR[r]); tick.set_fontweight('bold')
    ax.set_title("Consciousness sensitivity per region pair\n"
                 "(max − min of mean wSMI across the 3 K=3 clusters)",
                 fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('|cluster spread|  (higher = more discriminative)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    return sens


def plot_cl1_minus_cl0(region_mat, cluster_identities, out_path):
    """Direct contrast: control-rich cluster minus UWS-rich cluster."""
    ctrl_idx = cluster_identities['control_rich']
    uws_idx = cluster_identities['uws_rich']
    diff = region_mat[ctrl_idx] - region_mat[uws_idx]
    R = diff.shape[0]
    vmax = float(np.max(np.abs(diff)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    for i in range(R):
        for j in range(R):
            v = diff[i, j]
            ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white' if abs(v) > 0.66 * vmax else 'black')
    ax.set_xticks(range(R))
    ax.set_yticks(range(R))
    ax.set_xticklabels([REGION_LONG[r] for r in REGION_ORDER], rotation=30, ha='right')
    ax.set_yticklabels([REGION_LONG[r] for r in REGION_ORDER])
    for tick, r in zip(ax.get_xticklabels(), REGION_ORDER):
        tick.set_color(REGION_COLOUR[r]); tick.set_fontweight('bold')
    for tick, r in zip(ax.get_yticklabels(), REGION_ORDER):
        tick.set_color(REGION_COLOUR[r]); tick.set_fontweight('bold')
    ax.set_title(f"Control-rich cluster (cl_{ctrl_idx}) − UWS-rich cluster (cl_{uws_idx})\n"
                 "(red = stronger in conscious, blue = stronger in unconscious)",
                 fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('Δ mean wSMI  (control − UWS)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    return diff


def plot_pair_ranking(region_mat, cluster_identities, out_path):
    """Horizontal bar chart of all 15 unique region pairs ranked by
    consciousness sensitivity (max − min across clusters), annotated
    with the control-vs-UWS direction.
    """
    R = region_mat.shape[1]
    ctrl_idx = cluster_identities['control_rich']
    uws_idx = cluster_identities['uws_rich']

    rows = []
    for i in range(R):
        for j in range(i, R):
            pair = f'{REGION_ORDER[i]}-{REGION_ORDER[j]}'
            sens = region_mat[:, i, j].max() - region_mat[:, i, j].min()
            ctrl_minus_uws = region_mat[ctrl_idx, i, j] - region_mat[uws_idx, i, j]
            rows.append({'pair': pair, 'sensitivity': sens,
                         'ctrl_minus_uws': ctrl_minus_uws,
                         'is_long_range': abs(REGION_ORDER.index(REGION_ORDER[i]) -
                                              REGION_ORDER.index(REGION_ORDER[j])) >= 2 or i == j
                                          and False,
                         'is_within': i == j})
    df = pd.DataFrame(rows).sort_values('sensitivity', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colours = ['#1976d2' if r.ctrl_minus_uws < 0
               else ('#d32f2f' if r.ctrl_minus_uws > 0 else 'gray')
               for r in df.itertuples()]
    ax.barh(df['pair'], df['sensitivity'], color=colours, edgecolor='black', linewidth=0.4)
    for i, r in enumerate(df.itertuples()):
        ax.text(r.sensitivity + 0.0005, i,
                f' Δ(C−UWS) = {r.ctrl_minus_uws:+.3f}',
                va='center', fontsize=9)
    ax.set_xlabel('Cluster spread (max − min)  — higher = more consciousness-discriminative')
    ax.set_title('Region pairs ranked by consciousness sensitivity (K=3)\n'
                 'Red = control coupling > UWS coupling.  Blue = opposite (rare).',
                 fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coords_file', default='data_scalp/GSN-HydroCel-257.txt')
    ap.add_argument('--centroid_dir', default='output/centroid_last_100_balanced_K3')
    ap.add_argument('--output_dir', default='output/regional_centroids')
    ap.add_argument('--tag', default='K3')
    # Identify which K=3 cluster is the "control-rich" vs "UWS-rich" one
    # (from the cluster_diagnosis_enrichment in our K=3 centroid run):
    #   cl_0: 48.8% UWS, 3.7% control  → UWS-rich
    #   cl_1: 17.0% UWS, 28.6% control → control-rich
    #   cl_2: 29.4% UWS, 7.9% control  → intermediate
    ap.add_argument('--control_rich_cluster', type=int, default=1)
    ap.add_argument('--uws_rich_cluster',     type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load + re-derive the region assignment
    df_coords = pd.read_csv(args.coords_file, sep=r'\s+', header=None,
                            names=['label', 'x', 'y', 'z'])
    df_coords['region'] = df_coords.apply(assign_region, axis=1)
    region_idx = np.array([REGION_ORDER.index(r) for r in df_coords['region']])
    print(f"  {len(df_coords)} electrodes")
    for r in REGION_ORDER:
        print(f"    {r} ({REGION_LONG[r]}): {(df_coords['region'] == r).sum()} electrodes")

    centroids = np.load(os.path.join(args.centroid_dir, 'centroids_wsmi.npy'))
    counts = np.load(os.path.join(args.centroid_dir, 'centroids_counts.npy'))
    K = centroids.shape[0]
    print(f"  centroids: {centroids.shape}, counts={counts.tolist()}")

    cluster_identities = {
        'control_rich': args.control_rich_cluster,
        'uws_rich': args.uws_rich_cluster,
    }
    region_mat = compute_region_matrices(centroids, region_idx)

    print(f"  plotting consciousness-sensitivity heatmap")
    sens = plot_consciousness_sensitivity(
        region_mat, os.path.join(args.output_dir, f'consciousness_sensitivity_{args.tag}.png'))

    print(f"  plotting control-rich − UWS-rich differential")
    diff = plot_cl1_minus_cl0(
        region_mat, cluster_identities,
        os.path.join(args.output_dir, f'cl1_minus_cl0_{args.tag}.png'))

    print(f"  plotting region-pair ranking barplot")
    df_pairs = plot_pair_ranking(
        region_mat, cluster_identities,
        os.path.join(args.output_dir, f'pair_ranking_{args.tag}.png'))

    summary = {
        'config': vars(args),
        'cluster_counts': counts.tolist(),
        'region_order': REGION_ORDER,
        'cluster_identities': cluster_identities,
        'sensitivity_per_pair': sens.tolist(),
        'differential_ctrl_minus_uws_per_pair': diff.tolist(),
        'sorted_pair_ranking': df_pairs.sort_values(
            'sensitivity', ascending=False).to_dict('records'),
    }
    with open(os.path.join(args.output_dir, f'region_pair_summary_{args.tag}.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  artefacts in {args.output_dir}/  with tag '{args.tag}'")
    print(f"\n  top-5 most-consciousness-sensitive region pairs (K=3):")
    top5 = df_pairs.sort_values('sensitivity', ascending=False).head(5)
    for r in top5.itertuples():
        direction = 'CONTROL stronger' if r.ctrl_minus_uws > 0 else 'UWS stronger'
        print(f"    {r.pair:6s}  Δ_spread = {r.sensitivity:.4f}  "
              f"Δ(C−UWS) = {r.ctrl_minus_uws:+.4f}  →  {direction}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Decompose PC1 of the wSMI feature space:
  - Which electrode pairs have the largest |loading|?
  - Which 5×5 region pairs contribute most, and in which sign?
  - Does the sign convention match "high PC1 = more conscious"?

We re-fit the same PCA(50) used in the baseline (`last_100 × balanced` × all
subjects), pull `pca.components_[0]` (32,640-D loading vector), then:
  (a) flip its sign so that PC1(control) > PC1(COMA);
  (b) map each entry back to (electrode_i, electrode_j);
  (c) aggregate by (region_i, region_j) ∈ F/C/P/T/O;
  (d) rank top-N electrode pairs.

Outputs:
  output/pc1_interpretation/
    pc1_loadings.npy                       (32640,) signed loading vector
    pc1_loadings_256x256.npy               (256, 256) symmetric matrix
    pc1_region_loadings.csv                5×5 region-pair signed mean loadings
    pc1_topN_electrode_pairs.csv           top-100 electrode pairs by |loading|
    pc1_per_diagnosis_score.csv            mean PC1 per subject + diagnosis
    pc1_diagnosis_boxplot.png              per-diagnosis PC1 distribution
    pc1_region_loadings.png                5×5 heatmap (signed)
    pc1_topN_electrode_pairs.png           horizontal barplot of top-20 pairs
    pc1_scalp_top_edges.png                scalp layout with strongest edges drawn
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import EEGtoGraph


DX_TO_COARSE = {
    'control': 'control',
    'UWS': 'low_doc', 'COMA': 'low_doc',
    'MCS-': 'high_doc', 'MCS+': 'high_doc', 'EMCS': 'high_doc',
}
DX_ORDER = ['control', 'EMCS', 'MCS+', 'MCS-', 'UWS', 'COMA']
REGION_ORDER = ['F', 'C', 'P', 'T', 'O']
REGION_LONG = {'F': 'Frontal', 'C': 'Central', 'P': 'Parietal',
               'T': 'Temporal', 'O': 'Occipital'}


def assign_region(row):
    x, y, z = row['x'], row['y'], row['z']
    if abs(x) > 5.5 and z < -1.5: return 'T'
    if y < -6.0: return 'O'
    if y > 3.0:  return 'F'
    if y < -1.0 and z > 0.0: return 'P'
    return 'C'


def load_coords_df(coords_file):
    df = pd.read_csv(coords_file, sep=r'\s+', header=None,
                     names=['name', 'x', 'y', 'z'])
    df['region'] = df.apply(assign_region, axis=1)
    return df


def stream_features(data_dir, labels_csv):
    sessions = EEGtoGraph.enumerate_matrix_sessions(data_dir)
    lab = pd.read_csv(labels_csv, dtype=str)
    lab['session_z'] = lab['session'].str.zfill(2)
    lookup = {(r['subject'], r['session_z']): r['diagnostic_crs_final']
              for _, r in lab.iterrows()}

    iu, ju = np.triu_indices(256, k=1)
    chunks, meta = [], []
    for sid, snum, src in sessions:
        if src['kind'] != 'npz':
            continue
        cohort = src.get('cohort')
        dx = 'control' if cohort == 'control' \
            else lookup.get((sid, str(snum).zfill(2)), 'unknown')
        with np.load(src['path']) as d:
            arr = d['data'].astype(np.float32, copy=False)
        chunks.append(arr[:, iu, ju])
        for i in range(len(arr)):
            meta.append({'subject_id': sid, 'session_num': snum,
                         'matrix_idx': i, 'diagnosis': dx})
    X = np.concatenate(chunks, axis=0)
    df = pd.DataFrame(meta)
    return X, df, iu, ju


def take_last_n(df, n):
    out = []
    for _, g in df.groupby(['subject_id', 'session_num'], sort=False):
        out.append(g.iloc[-n:].index.to_numpy() if len(g) >= n
                   else g.index.to_numpy())
    return np.concatenate(out)


def balance_by_diagnosis(df, idx, rng):
    sub = df.loc[idx]
    counts = sub['diagnosis'].value_counts()
    cap = counts[counts.index.isin(DX_ORDER)]
    target = int(cap.min())
    out = []
    for dx, g in sub.groupby('diagnosis', sort=False):
        if dx not in DX_ORDER:
            continue
        rows = g.index.to_numpy()
        out.append(rows if len(rows) <= target
                   else rng.choice(rows, size=target, replace=False))
    return np.concatenate(out), target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--coords_file', default='data_scalp/GSN-HydroCel-257.txt')
    ap.add_argument('--labels_csv',
                    default='data/wsmi_res/256_electrodes-20260526T091011Z-3-001/256_electrodes/wsmi_res_DOC/patient_labels.csv')
    ap.add_argument('--output_dir', default='output/pc1_interpretation')
    ap.add_argument('--last_n', type=int, default=100)
    ap.add_argument('--n_pca', type=int, default=50)
    ap.add_argument('--top_n_pairs', type=int, default=100)
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    print(f"[1/6] streaming features")
    X_full, df_all, iu, ju = stream_features(args.data_dir, args.labels_csv)
    keep = df_all['diagnosis'].isin(DX_ORDER).to_numpy()
    df_all = df_all[keep].reset_index(drop=True)
    df_all['orig_index'] = np.arange(len(df_all))
    X_full = X_full[keep]
    print(f"  X_full: {X_full.shape}, "
          f"{df_all['subject_id'].nunique()} subjects, "
          f"counts={dict(Counter(df_all['diagnosis']))}")

    print(f"[2/6] last_{args.last_n} × balanced → PCA fit set")
    last_idx = take_last_n(df_all, args.last_n)
    df_last = df_all.loc[last_idx].reset_index(drop=True)
    df_last['global_row'] = df_all.loc[last_idx, 'orig_index'].to_numpy()
    bal_idx, target = balance_by_diagnosis(df_last,
                                           df_last.index.to_numpy(), rng)
    bal_global = df_last.loc[bal_idx, 'global_row'].to_numpy()
    print(f"  last_{args.last_n} rows: {len(df_last):,}; "
          f"balanced for PCA fit: {len(bal_global)} (cap={target})")

    print(f"[3/6] StandardScaler + PCA({args.n_pca}) on balanced set")
    sc = StandardScaler()
    X_bal_std = sc.fit_transform(X_full[bal_global])
    pca = PCA(n_components=args.n_pca, random_state=args.random_state)
    pca.fit(X_bal_std)
    # Per-epoch PC1 scores on the *full* last_n
    X_all_std = sc.transform(X_full[df_last['global_row'].to_numpy()])
    pc1_scores = pca.transform(X_all_std)[:, 0]
    df_last['pc1'] = pc1_scores
    print(f"  PC1 explained variance: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"  PC1-5 cum: "
          f"{pca.explained_variance_ratio_[:5].cumsum()[-1]*100:.2f}%")

    # ---------------------------------------------------------- sign convention
    print(f"[4/6] sign convention (high PC1 should be more conscious)")
    per_dx_mean = df_last.groupby('diagnosis')['pc1'].mean()
    pc1_control = float(per_dx_mean.get('control', np.nan))
    pc1_coma = float(per_dx_mean.get('COMA', per_dx_mean.get('UWS', np.nan)))
    flip = pc1_control < pc1_coma
    if flip:
        print(f"  flipping PC1 sign: control={pc1_control:.3f} < "
              f"COMA/UWS={pc1_coma:.3f}, want positive = conscious")
        pca.components_[0] *= -1
        pc1_scores *= -1
        df_last['pc1'] = pc1_scores
    else:
        print(f"  PC1 already oriented: control={pc1_control:.3f} > "
              f"COMA/UWS={pc1_coma:.3f}")
    per_dx_mean = df_last.groupby('diagnosis')['pc1'].mean().reindex(DX_ORDER)
    print(f"  mean PC1 per dx (descending consciousness):")
    print(per_dx_mean.to_string())

    # Subject-level PC1
    per_subj = (df_last.groupby('subject_id')
                .agg(diagnosis=('diagnosis', 'first'),
                     pc1_mean=('pc1', 'mean'),
                     n_epochs=('pc1', 'count'))
                .reset_index())
    per_subj.to_csv(os.path.join(args.output_dir,
                                 'pc1_per_diagnosis_score.csv'), index=False)

    # ----------------------------------- decompose PC1 by electrode + region
    print(f"[5/6] decomposing PC1 loadings")
    pc1_load = pca.components_[0]                              # (32640,)
    np.save(os.path.join(args.output_dir, 'pc1_loadings.npy'), pc1_load)

    # Build symmetric (256, 256) matrix
    L = np.zeros((256, 256), dtype=np.float64)
    L[iu, ju] = pc1_load
    L = L + L.T
    np.save(os.path.join(args.output_dir, 'pc1_loadings_256x256.npy'), L)

    # Region assignment
    coords_df = load_coords_df(args.coords_file)
    region_idx = np.array([REGION_ORDER.index(r) for r in coords_df['region']])

    # 5×5 region-pair: mean signed loading + sum |loading| + count
    R = len(REGION_ORDER)
    region_signed = np.zeros((R, R))
    region_abs = np.zeros((R, R))
    region_count = np.zeros((R, R), dtype=int)
    for k in range(len(pc1_load)):
        i, j = iu[k], ju[k]
        ri, rj = region_idx[i], region_idx[j]
        a, b = min(ri, rj), max(ri, rj)
        region_signed[a, b] += pc1_load[k]
        region_abs[a, b] += abs(pc1_load[k])
        region_count[a, b] += 1
    region_signed_mean = np.where(region_count > 0,
                                  region_signed / np.clip(region_count, 1, None),
                                  0.0)
    region_abs_mean = np.where(region_count > 0,
                               region_abs / np.clip(region_count, 1, None),
                               0.0)
    # Symmetrise for display
    region_signed_mean = region_signed_mean + region_signed_mean.T \
                        - np.diag(region_signed_mean.diagonal())
    region_abs_mean = region_abs_mean + region_abs_mean.T \
                     - np.diag(region_abs_mean.diagonal())

    region_rows = []
    for a in range(R):
        for b in range(a, R):
            region_rows.append({
                'region_pair': f'{REGION_ORDER[a]}-{REGION_ORDER[b]}',
                'n_electrode_pairs': int(region_count[a, b]),
                'mean_signed_loading': float(region_signed_mean[a, b]),
                'mean_abs_loading': float(region_abs_mean[a, b]),
                'sum_signed_loading': float(region_signed[a, b]),
            })
    df_region = pd.DataFrame(region_rows).sort_values(
        'mean_abs_loading', ascending=False)
    df_region.to_csv(os.path.join(args.output_dir,
                                  'pc1_region_loadings.csv'), index=False)
    print(df_region.head(10).to_string(index=False))

    # Top-N electrode pairs by |loading|
    order = np.argsort(-np.abs(pc1_load))[:args.top_n_pairs]
    top_rows = []
    for k in order:
        i, j = int(iu[k]), int(ju[k])
        ri, rj = region_idx[i], region_idx[j]
        top_rows.append({
            'electrode_i': coords_df['name'].iloc[i],
            'electrode_j': coords_df['name'].iloc[j],
            'region_i': REGION_ORDER[ri],
            'region_j': REGION_ORDER[rj],
            'region_pair': f'{REGION_ORDER[min(ri,rj)]}-{REGION_ORDER[max(ri,rj)]}',
            'loading': float(pc1_load[k]),
            'abs_loading': float(abs(pc1_load[k])),
            'i_idx': i, 'j_idx': j,
        })
    df_top = pd.DataFrame(top_rows)
    df_top.to_csv(os.path.join(args.output_dir,
                               'pc1_topN_electrode_pairs.csv'), index=False)
    print(f"  top-20 electrode pairs:")
    print(df_top.head(20)[['electrode_i','electrode_j','region_pair',
                           'loading']].to_string(index=False))

    # ---------------------------------------------------------------- plots
    print(f"[6/6] plotting")
    # (a) per-diagnosis boxplot of subject-level PC1
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [per_subj.loc[per_subj['diagnosis'] == d, 'pc1_mean'].to_numpy()
            for d in DX_ORDER]
    bp = ax.boxplot(data, labels=DX_ORDER, patch_artist=True, showfliers=False)
    palette = ['#2e7d32', '#7cb342', '#fdd835', '#fb8c00', '#e53935', '#6d4c41']
    for patch, c in zip(bp['boxes'], palette):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    # Overlay individual subjects
    for i, arr in enumerate(data):
        x = np.full(len(arr), i + 1) + (rng.random(len(arr)) - 0.5) * 0.18
        ax.scatter(x, arr, color='black', s=12, alpha=0.6, zorder=3)
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Diagnosis (descending consciousness)')
    ax.set_ylabel('Subject-level mean PC1 score')
    ax.set_title('PC1 distribution per diagnosis '
                 '(sign flipped so positive = more conscious)')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'pc1_diagnosis_boxplot.png'),
                dpi=140); plt.close(fig)

    # (b) 5×5 region-pair signed loading heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title, cmap in [
            (axes[0], region_signed_mean,
             'PC1 mean SIGNED loading by region pair\n(red = '
             'higher wSMI here → more conscious)', 'RdBu_r'),
            (axes[1], region_abs_mean,
             'PC1 mean |LOADING| by region pair\n'
             '(brightness = how informative this region pair is)', 'viridis')]:
        vmax = float(np.max(np.abs(mat)))
        if cmap == 'RdBu_r':
            im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(R)); ax.set_yticks(range(R))
        ax.set_xticklabels([REGION_LONG[r] for r in REGION_ORDER], rotation=30, ha='right')
        ax.set_yticklabels([REGION_LONG[r] for r in REGION_ORDER])
        ax.set_title(title, fontsize=10)
        for i in range(R):
            for j in range(R):
                ax.text(j, i, f'{mat[i,j]:+.2e}' if cmap=='RdBu_r' else f'{mat[i,j]:.2e}',
                        ha='center', va='center', fontsize=8,
                        color='white' if (cmap=='viridis' and mat[i,j]>vmax*0.5)
                        else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'pc1_region_loadings.png'),
                dpi=140); plt.close(fig)

    # (c) Top-20 electrode pairs bar
    top20 = df_top.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    labels = [f"{r.electrode_i} — {r.electrode_j} ({r.region_pair})"
              for r in top20.itertuples()]
    colors = ['#e53935' if r.loading > 0 else '#1e88e5'
              for r in top20.itertuples()]
    ax.barh(range(len(top20)), top20['loading'].to_numpy(), color=colors)
    ax.set_yticks(range(len(top20))); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('PC1 loading (red = higher wSMI → more conscious, '
                  'blue = lower wSMI → more conscious)')
    ax.set_title(f'Top 20 electrode pairs by |PC1 loading|')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir,
                             'pc1_topN_electrode_pairs.png'),
                dpi=140); plt.close(fig)

    # (d) Scalp layout with top edges drawn
    fig, ax = plt.subplots(figsize=(8, 8))
    region_color = {'F': '#e57373', 'C': '#fdd835', 'P': '#81c784',
                    'T': '#64b5f6', 'O': '#ba68c8'}
    for r in REGION_ORDER:
        m = coords_df['region'] == r
        ax.scatter(coords_df.loc[m, 'x'], coords_df.loc[m, 'y'],
                   s=40, color=region_color[r], edgecolor='black',
                   linewidth=0.5, label=REGION_LONG[r], zorder=3)
    # Draw the top-50 edges, alpha by |loading|
    top50 = df_top.head(50)
    max_abs = top50['abs_loading'].max() if len(top50) else 1.0
    segs, cols, lws = [], [], []
    for r in top50.itertuples():
        x0, y0 = coords_df.iloc[r.i_idx][['x', 'y']]
        x1, y1 = coords_df.iloc[r.j_idx][['x', 'y']]
        segs.append([(x0, y0), (x1, y1)])
        cols.append('#e53935' if r.loading > 0 else '#1e88e5')
        lws.append(0.5 + 3.0 * r.abs_loading / max_abs)
    lc = LineCollection(segs, colors=cols, linewidths=lws, alpha=0.55, zorder=2)
    ax.add_collection(lc)
    ax.set_aspect('equal')
    ax.set_xlabel('x (anterior →)'); ax.set_ylabel('y (right →)')
    ax.set_title('Top 50 PC1-loading edges drawn on scalp\n'
                 '(red = higher wSMI → more conscious, '
                 'blue = lower wSMI → more conscious)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'pc1_scalp_top_edges.png'),
                dpi=140); plt.close(fig)

    # Summary JSON
    summary = {
        'config': vars(args),
        'pc1_explained_variance': float(pca.explained_variance_ratio_[0]),
        'first_5_explained_variance': pca.explained_variance_ratio_[:5].tolist(),
        'flipped_sign': bool(flip),
        'per_dx_mean_pc1': {d: float(per_dx_mean[d]) for d in DX_ORDER
                            if d in per_dx_mean.index},
        'top_region_pair_by_abs_loading':
            df_region.iloc[0]['region_pair'],
        'top_region_pair_loading_mean':
            float(df_region.iloc[0]['mean_signed_loading']),
        'top5_region_pairs': df_region.head(5).to_dict('records'),
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  summary → {args.output_dir}/summary.json")


if __name__ == '__main__':
    main()

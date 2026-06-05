#!/usr/bin/env python3
"""
Cluster wSMI epochs using ONLY a subset of region-pair entries.

Motivation: chapter 4 showed P-O / C-P / F-P pairs carry the strongest
consciousness signal. Question: does restricting the feature vector to those
pairs (zeroing out the rest) tighten the cluster x diagnosis association?

Pipeline mirrors scripts/centroid_last_100_balanced.py exactly, except features
are the masked upper-triangle of each (256,256) wSMI matrix.

  --pair_set ∈ {full, P-O, top3, long_range}
  --K, --last_n, --random_state  (matching the V=0.252 / K=3 baseline)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import EEGtoGraph


DX_ORDER = ['control', 'UWS', 'MCS-', 'MCS+', 'EMCS', 'COMA']
REGION_ORDER = ['F', 'C', 'P', 'T', 'O']


def assign_region(row):
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


def load_region_idx(coords_file: str) -> np.ndarray:
    df = pd.read_csv(coords_file, sep=r'\s+', header=None,
                     names=['name', 'x', 'y', 'z'])
    df['region'] = df.apply(assign_region, axis=1)
    return np.array([REGION_ORDER.index(r) for r in df['region']])


def build_mask(region_idx: np.ndarray, pair_set: str) -> np.ndarray:
    """Boolean (256, 256), symmetric, diagonal excluded."""
    N = len(region_idx)
    if pair_set == 'full':
        m = np.ones((N, N), dtype=bool)
    elif pair_set == 'P-O':
        pairs = {('P', 'O')}
        m = _pairs_to_mask(region_idx, pairs)
    elif pair_set == 'top3':
        pairs = {('P', 'O'), ('C', 'P'), ('F', 'P')}
        m = _pairs_to_mask(region_idx, pairs)
    elif pair_set == 'long_range':
        # |r_i - r_j| >= 2 in REGION_ORDER (i.e. F-P, F-T, F-O, C-T, C-O, P-O)
        ri = region_idx[:, None]
        rj = region_idx[None, :]
        m = np.abs(ri - rj) >= 2
    else:
        raise ValueError(f"unknown pair_set: {pair_set}")
    np.fill_diagonal(m, False)
    return m


def _pairs_to_mask(region_idx: np.ndarray, pairs: set) -> np.ndarray:
    N = len(region_idx)
    m = np.zeros((N, N), dtype=bool)
    for ra, rb in pairs:
        ia = REGION_ORDER.index(ra)
        ib = REGION_ORDER.index(rb)
        a_nodes = (region_idx == ia)
        b_nodes = (region_idx == ib)
        m |= np.outer(a_nodes, b_nodes)
        m |= np.outer(b_nodes, a_nodes)
    return m


def upper_tri_idx(mask: np.ndarray) -> np.ndarray:
    """Linear indices of the masked upper triangle (no diagonal)."""
    N = mask.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    keep = mask[iu, ju]
    return iu[keep], ju[keep]


def stream_features(sessions, last_n: int,
                    iu: np.ndarray, ju: np.ndarray, labels_csv: str):
    """Build (n_epochs, n_features) + meta in one streaming pass."""
    Xs, meta = [], []
    for sid, snum, src in sessions:
        if src['kind'] != 'npz':
            continue
        with np.load(src['path']) as d:
            arr = d['data'].astype(np.float32, copy=False)
        n_keep = min(last_n, len(arr))
        arr = arr[-n_keep:]
        Xs.append(arr[:, iu, ju])
        for i in range(n_keep):
            meta.append({'subject_id': sid, 'session_num': snum,
                         'matrix_idx': len(arr) - n_keep + i,
                         'cohort': src.get('cohort')})
    X = np.concatenate(Xs, axis=0)
    df = pd.DataFrame(meta)

    # Attach diagnoses from labels CSV (same as scripts/full_wsmi_cluster.py)
    lab = pd.read_csv(labels_csv, dtype=str)
    lab['session_z'] = lab['session'].str.zfill(2)
    lookup = {(r['subject'], r['session_z']): r['diagnostic_crs_final']
              for _, r in lab.iterrows()}
    df['session_z'] = df['session_num'].astype(str).str.zfill(2)
    df['diagnosis'] = np.where(
        df['cohort'] == 'control',
        'control',
        df.apply(lambda r: lookup.get((r['subject_id'], r['session_z']), 'unknown'),
                 axis=1))
    return X, df


def balance_by_diagnosis_indices(df, in_idx, rng):
    sub = df.loc[in_idx]
    counts = sub['diagnosis'].value_counts()
    cap_pool = counts[counts.index.isin(DX_ORDER)]
    target = int(cap_pool.min())
    out = []
    for dx, gdf in sub.groupby('diagnosis', sort=False):
        if dx not in DX_ORDER:
            continue
        if len(gdf) <= target:
            out.append(gdf.index.to_numpy())
        else:
            out.append(rng.choice(gdf.index.to_numpy(), size=target, replace=False))
    return np.concatenate(out), target


def contingency(labels, diagnoses, dx_order):
    K = int(np.max(labels)) + 1
    M = np.zeros((K, len(dx_order)), dtype=int)
    for c, d in zip(labels, diagnoses):
        if d in dx_order:
            M[c, dx_order.index(d)] += 1
    chi2, p, _, _ = chi2_contingency(M + 1e-9)
    n = int(M.sum())
    cv = float(np.sqrt(chi2 / (n * (min(M.shape) - 1)))) if min(M.shape) > 1 else float('nan')
    return M, {'chi2': float(chi2), 'p': float(p), 'cramers_v': cv, 'n': n}


def subject_level_dx(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('subject_id', sort=False)['diagnosis'].first().reset_index()


def loocv_classify(X_pca, df, K, random_state):
    """Subject-level LOOCV: GMM(K) on the train fold's subjects, then aggregate
    cluster posteriors to subject-level features and predict diagnosis."""
    subj_df = subject_level_dx(df)
    subj_df = subj_df[subj_df['diagnosis'].isin(DX_ORDER)].reset_index(drop=True)
    subj_ids = subj_df['subject_id'].to_numpy()
    subj_y = subj_df['diagnosis'].to_numpy()

    keep_mask = df['diagnosis'].isin(DX_ORDER).to_numpy()
    X = X_pca[keep_mask]
    grp = df.loc[keep_mask, 'subject_id'].to_numpy()
    dxv = df.loc[keep_mask, 'diagnosis'].to_numpy()

    logo = LeaveOneGroupOut()
    y_true, y_pred = [], []
    for tr, te in logo.split(X, dxv, groups=grp):
        gm = GaussianMixture(n_components=K, covariance_type='full',
                             random_state=random_state, n_init=2, max_iter=200,
                             reg_covar=1e-3)
        gm.fit(X[tr])
        # Subject-level features = mean cluster posterior over the subject's epochs
        post = gm.predict_proba(X)
        feat_subj = []
        y_subj = []
        for sid in np.unique(grp):
            m = grp == sid
            feat_subj.append(post[m].mean(axis=0))
            y_subj.append(dxv[m][0])
        feat_subj = np.array(feat_subj)
        y_subj = np.array(y_subj)
        # Hold-out test subject(s) for this LOO fold
        held_id = grp[te][0]
        tr_subj = (np.array([sid for sid in np.unique(grp)]) != held_id)
        te_subj = ~tr_subj
        clf = LogisticRegression(max_iter=2000, class_weight='balanced')
        clf.fit(feat_subj[tr_subj], y_subj[tr_subj])
        y_pred.append(clf.predict(feat_subj[te_subj])[0])
        y_true.append(y_subj[te_subj][0])
    return balanced_accuracy_score(y_true, y_pred), y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--coords_file', default='data_scalp/GSN-HydroCel-257.txt')
    ap.add_argument('--labels_csv',
                    default='data/wsmi_res/256_electrodes-20260526T091011Z-3-001/256_electrodes/wsmi_res_DOC/patient_labels.csv')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--pair_set', required=True,
                    choices=['full', 'P-O', 'top3', 'long_range'])
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--last_n', type=int, default=100)
    ap.add_argument('--n_pca', type=int, default=50)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--skip_loocv', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    print(f"[1/6] building mask for pair_set='{args.pair_set}'")
    region_idx = load_region_idx(args.coords_file)
    mask = build_mask(region_idx, args.pair_set)
    iu, ju = upper_tri_idx(mask)
    n_feat = len(iu)
    print(f"  electrodes per region: "
          f"{dict(Counter([REGION_ORDER[r] for r in region_idx]))}")
    print(f"  retained off-diagonal entries: {n_feat:,} / "
          f"{256*255//2:,} = {100*n_feat/(256*255//2):.2f}%")

    print(f"[2/6] enumerating sessions + streaming last_{args.last_n} features")
    sessions = EEGtoGraph.enumerate_matrix_sessions(args.data_dir)
    X_raw, df = stream_features(sessions, args.last_n, iu, ju, args.labels_csv)
    print(f"  X_raw: {X_raw.shape}, {df['subject_id'].nunique()} subjects, "
          f"{df.groupby(['subject_id','session_num']).ngroups} sessions")

    n_pca = min(args.n_pca, X_raw.shape[1], X_raw.shape[0])
    print(f"[3/6] PCA → {n_pca}D")
    pca = PCA(n_components=n_pca, random_state=args.random_state)
    X_pca = pca.fit_transform(X_raw.astype(np.float32))
    var = pca.explained_variance_ratio_.sum()
    print(f"  cumulative explained variance: {var*100:.2f}%")

    print(f"[4/6] balancing by diagnosis for GMM fit")
    df_keep = df[df['diagnosis'].isin(DX_ORDER)].copy()
    df_keep = df_keep.reset_index(drop=True)
    X_keep = X_pca[df['diagnosis'].isin(DX_ORDER).to_numpy()]
    # Map keep-row index back to X_keep row index — they are aligned by construction
    bal_idx, target = balance_by_diagnosis_indices(
        df_keep, df_keep.index.to_numpy(), rng)
    print(f"  per-dx cap = {target} → {len(bal_idx)} epochs for GMM fit")

    print(f"[5/6] GMM K={args.K} on the balanced subset, V on full last_n")
    gm = GaussianMixture(n_components=args.K, covariance_type='full',
                         random_state=args.random_state, n_init=3,
                         max_iter=200, reg_covar=1e-3)
    gm.fit(X_keep[bal_idx])
    bal_lbl = gm.predict(X_keep[bal_idx])
    bal_dx = df_keep.loc[bal_idx, 'diagnosis'].tolist()
    _, stats_bal = contingency(bal_lbl, bal_dx, DX_ORDER)
    full_lbl = gm.predict(X_keep)
    full_dx = df_keep['diagnosis'].tolist()
    _, stats_full = contingency(full_lbl, full_dx, DX_ORDER)
    print(f"  balanced-fit V = {stats_bal['cramers_v']:.4f}")
    print(f"  full last_n V  = {stats_full['cramers_v']:.4f}  (n={stats_full['n']:,})")

    summary = {
        'config': vars(args),
        'n_features_after_mask': int(n_feat),
        'n_features_full_uppertri': 256 * 255 // 2,
        'pca_cum_var': float(var),
        'balanced_fit_V': stats_bal['cramers_v'],
        'full_last_V': stats_full['cramers_v'],
        'full_last_n': stats_full['n'],
        'balanced_fit_n': stats_bal['n'],
    }

    if not args.skip_loocv:
        print(f"[6/6] subject-level LOOCV (GMM posteriors → LogReg)")
        bal_acc, y_true, y_pred = loocv_classify(
            X_keep, df_keep, args.K, args.random_state)
        print(f"  LOOCV balanced accuracy = {bal_acc:.4f} "
              f"(over {len(y_true)} subjects)")
        summary['loocv_bal_acc'] = float(bal_acc)
        summary['loocv_n_subjects'] = len(y_true)

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  summary → {args.output_dir}/summary.json")


if __name__ == '__main__':
    main()

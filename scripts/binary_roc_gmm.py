#!/usr/bin/env python3
"""
5-fold (repeatable) ROC AUC for raw PCA + GMM K=k baseline.

Subject-level GroupKFold with optional --n_reps repetitions using different
fold seeds. Per fold and per binary task, fit GMM (on the train fold) + LogReg
(on subject-level posteriors) and compute ROC AUC on the held-out fold.

Output:
  output/{out}/per_fold_aucs.csv     rep, fold, task, auc, n_pos, n_neg
  output/{out}/per_subject_proba.csv subject_id, true_dx_coarse, p_*, fold, rep
  output/{out}/roc_summary.json      mean/std per task across folds*reps
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

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import EEGtoGraph


DX_TO_COARSE = {
    'control': 'control',
    'UWS': 'low_doc', 'COMA': 'low_doc',
    'MCS-': 'high_doc', 'MCS+': 'high_doc', 'EMCS': 'high_doc',
}
COARSE_ORDER = ['control', 'low_doc', 'high_doc']


def take_last_n(df, n):
    out = []
    for _, g in df.groupby(['subject_id', 'session_num'], sort=False):
        out.append(g.iloc[-n:].index.to_numpy() if len(g) >= n
                   else g.index.to_numpy())
    return np.concatenate(out)


def balance_by_diagnosis(df, idx, rng):
    sub = df.loc[idx]
    counts = sub['diagnosis'].value_counts()
    cap = counts[counts.index.isin(DX_TO_COARSE.keys())]
    if cap.empty:
        return idx, 0
    target = int(cap.min())
    out = []
    for dx, g in sub.groupby('diagnosis', sort=False):
        if dx not in DX_TO_COARSE:
            continue
        rows = g.index.to_numpy()
        out.append(rows if len(rows) <= target
                   else rng.choice(rows, size=target, replace=False))
    return np.concatenate(out), target


def stream_features(data_dir, labels_csv):
    """Stream all wSMI epochs as flat upper-triangle features + meta."""
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
    return X, df


def fit_one_fold(X_pca, df_last, train_subj, test_subj, K, random_state, rng):
    """Returns dict mapping subject_id (in test_subj) -> (true_dx_coarse, p_control, p_low_doc, p_high_doc)."""
    train_mask = df_last['subject_id'].isin(train_subj)
    test_mask = df_last['subject_id'].isin(test_subj)
    df_train = df_last[train_mask].reset_index(drop=True)
    df_test = df_last[test_mask].reset_index(drop=True)

    # Balance training epochs by diagnosis for the GMM fit
    bal_idx, _ = balance_by_diagnosis(df_train, df_train.index.to_numpy(), rng)
    train_bal_global = df_train.loc[bal_idx, 'global_row'].to_numpy()

    gm = GaussianMixture(n_components=K, covariance_type='full',
                         random_state=random_state, n_init=2,
                         max_iter=200, reg_covar=1e-3)
    gm.fit(X_pca[train_bal_global])

    # Subject-level soft fingerprint on TRAIN_last (not balanced)
    train_post = gm.predict_proba(X_pca[df_train['global_row'].to_numpy()])
    post = pd.DataFrame(train_post, columns=[f'p{c}' for c in range(K)])
    post['subject_id'] = df_train['subject_id'].to_numpy()
    train_soft = post.groupby('subject_id').mean()
    train_subj_list = train_soft.index.tolist()
    Xtr = train_soft.to_numpy()
    ytr = (df_train.groupby('subject_id')['diagnosis']
           .apply(lambda s: Counter(s).most_common(1)[0][0])
           .reindex(train_subj_list).map(DX_TO_COARSE).tolist())

    # Test fingerprints
    test_post = gm.predict_proba(X_pca[df_test['global_row'].to_numpy()])
    tpost = pd.DataFrame(test_post, columns=[f'p{c}' for c in range(K)])
    tpost['subject_id'] = df_test['subject_id'].to_numpy()
    test_soft = tpost.groupby('subject_id').mean()
    test_subj_list = test_soft.index.tolist()
    Xte = test_soft.to_numpy()

    # LogReg with predict_proba on coarse labels
    clf = LogisticRegression(max_iter=2000, class_weight='balanced',
                             solver='lbfgs', random_state=random_state)
    clf.fit(Xtr, ytr)
    probs = clf.predict_proba(Xte)

    rows = []
    for s, p in zip(test_subj_list, probs):
        row = {'subject_id': s}
        dx = df_test[df_test['subject_id'] == s]['diagnosis'].iloc[0]
        row['true_dx_coarse'] = DX_TO_COARSE[dx]
        for cls, pp in zip(clf.classes_, p):
            row[f'p_{cls}'] = float(pp)
        for cls in COARSE_ORDER:
            row.setdefault(f'p_{cls}', 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_fold_aucs(df_subj):
    """Per-fold ROC AUC for the two binary tasks. Returns NaN if one class is missing."""
    # Task A: control vs rest
    ya = (df_subj['true_dx_coarse'] == 'control').astype(int).to_numpy()
    sa = df_subj['p_control'].to_numpy()
    auc_a = float(roc_auc_score(ya, sa)) if len(set(ya)) > 1 else float('nan')

    # Task B: MCS vs UWS (exclude controls)
    sub = df_subj[df_subj['true_dx_coarse'].isin(['low_doc', 'high_doc'])]
    yb = (sub['true_dx_coarse'] == 'high_doc').astype(int).to_numpy()
    denom = (sub['p_high_doc'] + sub['p_low_doc']).clip(lower=1e-12)
    sb = (sub['p_high_doc'] / denom).to_numpy()
    auc_b = float(roc_auc_score(yb, sb)) if len(set(yb)) > 1 else float('nan')
    return auc_a, int(ya.sum()), int((1 - ya).sum()), \
           auc_b, int(yb.sum()), int((1 - yb).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--labels_csv',
                    default='data/wsmi_res/256_electrodes-20260526T091011Z-3-001/256_electrodes/wsmi_res_DOC/patient_labels.csv')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--last_n', type=int, default=100)
    ap.add_argument('--pca_dim', type=int, default=50)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--n_reps', type=int, default=5)
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    print(f"[1/4] streaming features")
    X_full, df = stream_features(args.data_dir, args.labels_csv)
    keep = df['diagnosis'].isin(DX_TO_COARSE).to_numpy()
    df = df[keep].reset_index(drop=True)
    df['orig_index'] = np.arange(len(df))
    X_full = X_full[keep]
    print(f"  X: {X_full.shape}, {df['subject_id'].nunique()} subjects, "
          f"{df.groupby(['subject_id','session_num']).ngroups} sessions")

    print(f"[2/4] PCA({args.pca_dim})")
    pca = PCA(n_components=args.pca_dim, random_state=args.random_state)
    X_pca = pca.fit_transform(X_full)
    print(f"  cum var = {pca.explained_variance_ratio_.sum()*100:.2f}%")

    last_idx = take_last_n(df, args.last_n)
    df_last = df.loc[last_idx].reset_index(drop=True)
    df_last['global_row'] = df.loc[last_idx, 'orig_index'].to_numpy()
    subjects = sorted(df_last['subject_id'].unique())
    subj_arr = df_last['subject_id'].to_numpy()
    print(f"  last_{args.last_n}: {len(df_last):,} epochs, {len(subjects)} subjects")

    print(f"[3/4] {args.n_folds}-fold CV x {args.n_reps} reps "
          f"(= {args.n_folds * args.n_reps} fits)")
    fold_rows = []
    proba_rows = []
    for rep in range(args.n_reps):
        # Shuffle subjects per rep so folds differ across reps
        rng_local = np.random.default_rng(args.random_state + rep)
        # Each subject's group label = subject_id; we want GroupKFold splits.
        # GroupKFold doesn't randomize; shuffle by re-ordering groups input.
        order = rng_local.permutation(len(df_last))
        groups_shuffled = subj_arr[order]
        rows_shuffled = df_last.index.to_numpy()[order]
        # Stratify-ish: GroupKFold ensures subject-disjoint splits
        gkf = GroupKFold(n_splits=args.n_folds)
        # Run GroupKFold on shuffled order
        for fold_i, (tr_pos, te_pos) in enumerate(
                gkf.split(np.zeros(len(df_last)), None, groups=groups_shuffled)):
            tr_subj = set(groups_shuffled[tr_pos])
            te_subj = set(groups_shuffled[te_pos])
            df_subj = fit_one_fold(X_pca, df_last, tr_subj, te_subj,
                                   args.K, args.random_state + rep * 100 + fold_i,
                                   rng_local)
            df_subj['fold'] = fold_i + 1
            df_subj['rep'] = rep + 1
            proba_rows.append(df_subj)

            auc_a, na_pos, na_neg, auc_b, nb_pos, nb_neg = compute_fold_aucs(df_subj)
            fold_rows.append({'rep': rep + 1, 'fold': fold_i + 1,
                              'task': 'control_vs_rest', 'auc': auc_a,
                              'n_pos': na_pos, 'n_neg': na_neg})
            fold_rows.append({'rep': rep + 1, 'fold': fold_i + 1,
                              'task': 'high_vs_low_doc', 'auc': auc_b,
                              'n_pos': nb_pos, 'n_neg': nb_neg})
            print(f"  rep {rep+1} fold {fold_i+1}: "
                  f"control={auc_a:.3f} (n_pos={na_pos}/{na_neg}), "
                  f"MCS-vs-UWS={auc_b:.3f} (n_pos={nb_pos}/{nb_neg})")

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(os.path.join(args.output_dir, 'per_fold_aucs.csv'),
                    index=False)
    df_proba = pd.concat(proba_rows, ignore_index=True)
    df_proba.to_csv(os.path.join(args.output_dir, 'per_subject_proba.csv'),
                    index=False)

    print(f"\n[4/4] summary across {args.n_folds * args.n_reps} folds")
    summary = {'config': vars(args)}
    for task in ['control_vs_rest', 'high_vs_low_doc']:
        sub = df_folds[df_folds['task'] == task]['auc'].dropna()
        summary[task] = {
            'mean_auc': float(sub.mean()),
            'std_auc': float(sub.std(ddof=1)),
            'min_auc': float(sub.min()),
            'max_auc': float(sub.max()),
            'n_folds': int(len(sub)),
            'all_aucs': sub.tolist(),
        }
        print(f"  {task}: {sub.mean():.3f} +/- {sub.std(ddof=1):.3f}  "
              f"(n={len(sub)}, range [{sub.min():.3f}, {sub.max():.3f}])")

    summary['elapsed_min'] = (time.time() - t0) / 60
    with open(os.path.join(args.output_dir, 'roc_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()

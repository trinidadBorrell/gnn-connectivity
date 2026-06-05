#!/usr/bin/env python3
"""
Leave-N-subjects-out validation of the raw-wSMI clustering pipeline.

Tests whether the V=0.252 baseline (last_100_balanced_K4) generalises to
held-out patients. For each of N_FOLDS:

  1. Split subjects into train / test (GroupKFold).
  2. Refit StandardScaler + PCA(50) on train-subject epochs ONLY (no leakage).
  3. Take last_N epochs of each train session; balance across diagnoses.
  4. Fit GMM K=4 on that balanced-train set.
  5. For each test subject, predict cluster for their last_N epochs.
  6. Score three ways:
     (A) Within-fold V (sanity check vs all-data V).
     (B) Per-subject modal-cluster → diagnosis prediction via training-fold
         cluster-to-diagnosis majority mapping. Balanced accuracy.
     (C) Per-subject cluster-fingerprint (hard) and posterior-fingerprint
         (soft) → logistic regression on training-fold subjects → predict
         test-fold diagnoses. Balanced accuracy + confusion matrix.

Also reports collapsed-class metrics: {control, lowDOC=UWS+COMA, highDOC=
MCS-/MCS+/EMCS} as the coarse 3-class problem the clinic actually cares about.
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, classification_report
)
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import EEGtoGraph


DX_ORDER_FULL = ['control', 'UWS', 'MCS-', 'MCS+', 'EMCS', 'COMA']
DX_TO_COARSE = {
    'control': 'control',
    'UWS': 'low_doc',
    'COMA': 'low_doc',
    'MCS-': 'high_doc',
    'MCS+': 'high_doc',
    'EMCS': 'high_doc',
}
COARSE_ORDER = ['control', 'low_doc', 'high_doc']


# --------------------------------------------------------------------- loading
def load_raw_features(data_dir):
    """Load the (n_total_epochs, 32640) upper-triangle features + meta."""
    sessions = EEGtoGraph.enumerate_matrix_sessions(data_dir)
    iu = np.triu_indices(256, k=1)
    feats, meta = [], []
    cursor = 0
    t0 = time.time()
    for i, (sid, snum, source) in enumerate(sessions):
        if source['kind'] != 'npz':
            continue
        with np.load(source['path']) as d:
            arr = d['data']
            n_eps = int(arr.shape[0])
            flat = arr[:, iu[0], iu[1]].astype(np.float32, copy=False)
        feats.append(flat)
        cohort = source.get('cohort')
        for j in range(n_eps):
            meta.append({'subject_id': sid, 'session_num': snum, 'cohort': cohort,
                         'matrix_idx': j, 'global_index': cursor + j})
        cursor += n_eps
        if (i + 1) % 25 == 0:
            print(f"  loaded {i+1}/{len(sessions)} sessions, {cursor:,} epochs, "
                  f"{(time.time()-t0)/60:.1f} min")
    X = np.concatenate(feats, axis=0)
    del feats
    print(f"  X: shape={X.shape}, dtype={X.dtype}, mem={X.nbytes/1e9:.1f} GB")
    return X, meta


def attach_diagnoses(df, labels_csv):
    labels = pd.read_csv(labels_csv, dtype=str)
    labels['session_z'] = labels['session'].str.zfill(2)
    lookup = {(r['subject'], r['session_z']): r['diagnostic_crs_final']
              for _, r in labels.iterrows()}
    def get_dx(row):
        if row['cohort'] == 'control':
            return 'control'
        key = (row['subject_id'], str(row['session_num']).zfill(2))
        return lookup.get(key, 'unknown')
    df['diagnosis'] = df.apply(get_dx, axis=1)
    return df


# ------------------------------------------------------------ partition utils
def last_n_per_session_mask(df, n):
    """Boolean mask: True for the last n epochs of each session."""
    mask = np.zeros(len(df), dtype=bool)
    for _, gdf in df.groupby(['subject_id', 'session_num'], sort=False):
        if len(gdf) >= n:
            mask[gdf.iloc[-n:].index.to_numpy()] = True
        else:
            mask[gdf.index.to_numpy()] = True
    return mask


def balance_by_diagnosis_indices(df, in_idx, rng):
    """Within in_idx, downsample each diagnosis to the smallest dx count."""
    sub = df.loc[in_idx]
    counts = sub['diagnosis'].value_counts()
    target = int(counts.min())
    out = []
    for dx, gdf in sub.groupby('diagnosis', sort=False):
        if len(gdf) <= target:
            out.append(gdf.index.to_numpy())
        else:
            out.append(rng.choice(gdf.index.to_numpy(), size=target, replace=False))
    return np.concatenate(out), target


# --------------------------------------------------------- metrics + plotting
def cluster_to_dx_majority(labels, diagnoses, dx_order):
    """For each cluster id, return which diagnosis is the modal class in that
    cluster across the training data.
    """
    K = int(np.max(labels)) + 1
    mapping = {}
    for c in range(K):
        sel = [d for d, lb in zip(diagnoses, labels) if lb == c]
        if not sel:
            mapping[c] = None
            continue
        cnt = Counter(sel)
        # Restrict to known dx_order classes (skip unknown / nan)
        for dx, _ in cnt.most_common():
            if dx in dx_order:
                mapping[c] = dx
                break
        else:
            mapping[c] = None
    return mapping


def per_subject_fingerprints(df, labels_full, gmm, X_pca_full, K):
    """For each subject, compute hard and soft cluster fingerprints (K-dim).
    Returns dict subject_id → {'hard': vec, 'soft': vec, 'diagnosis': str}.
    """
    out = {}
    for sid, gdf in df.groupby('subject_id', sort=False):
        idx = gdf.index.to_numpy()
        lbls = labels_full[idx]
        hard = np.bincount(lbls, minlength=K).astype(float)
        hard /= hard.sum() if hard.sum() > 0 else 1
        # Soft posterior — average per-epoch P(cluster | epoch)
        post = gmm.predict_proba(X_pca_full[idx])
        soft = post.mean(axis=0)
        # Subject-level diagnosis: take the modal diagnosis across their epochs
        # (all of one subject's epochs share the same dx in our metadata,
        # but be defensive)
        dx = Counter(gdf['diagnosis']).most_common(1)[0][0]
        out[sid] = {'hard': hard, 'soft': soft, 'diagnosis': dx}
    return out


def safe_balanced_accuracy(y_true, y_pred):
    if len(set(y_true)) < 2:
        return float('nan')
    return float(balanced_accuracy_score(y_true, y_pred))


def confusion_plot(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('predicted'); ax.set_ylabel('true')
    ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.2f})',
                    ha='center', va='center', fontsize=8,
                    color='white' if cm_norm[i, j] < 0.5 else 'black')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def contingency(labels, diagnoses, dx_order):
    K = int(np.max(labels)) + 1
    M = np.zeros((K, len(dx_order)), dtype=int)
    for c, d in zip(labels, diagnoses):
        if d in dx_order:
            M[c, dx_order.index(d)] += 1
    chi2, p, dof, _ = chi2_contingency(M + 1e-9)
    n = int(M.sum())
    cv = float(np.sqrt(chi2 / (n * (min(M.shape) - 1)))) if min(M.shape) > 1 else float('nan')
    return M, cv, float(chi2)


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--labels_csv',
                    default='data/wsmi_res/256_electrodes-20260526T091011Z-3-001/256_electrodes/wsmi_res_DOC/patient_labels.csv')
    ap.add_argument('--output_dir', default='output/holdout')
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--pca_dim', type=int, default=50)
    ap.add_argument('--last_n', type=int, default=100,
                    help='Take last N epochs of each session (train + test).')
    ap.add_argument('--reg_covar', type=float, default=1e-3,
                    help='GMM covariance regulariser. Bumping to 1e-2 suppresses '
                         'tiny singleton clusters at the cost of slightly less '
                         'fitted detail.')
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    print(f"[1/4] loading raw features")
    X_raw, meta = load_raw_features(args.data_dir)
    df = pd.DataFrame(meta)
    df['matrix_idx'] = df['matrix_idx'].astype(int)
    df = df.sort_values(['subject_id', 'session_num', 'matrix_idx']).reset_index(drop=True)
    df = attach_diagnoses(df, args.labels_csv)
    # Drop unknown diagnoses from the analysis
    valid_dx = set(DX_ORDER_FULL)
    drop_mask = ~df['diagnosis'].isin(valid_dx)
    if drop_mask.any():
        print(f"  dropping {drop_mask.sum()} epochs with unknown/missing diagnoses")
    keep_idx = df.index[~drop_mask].to_numpy()
    df = df.loc[keep_idx].reset_index(drop=True)
    # Build a corresponding raw feature view (same row order).
    # We need the original X_raw rows aligned with our (post-sort, post-filter) df.
    # 'global_index' in meta corresponds to original X_raw rows; we kept rows in
    # global-index order via the sort above.
    X_raw = X_raw[df['global_index'].to_numpy()]
    df['orig_row'] = np.arange(len(df))  # row index into X_raw after filtering
    print(f"  after filtering: X_raw={X_raw.shape}, df={len(df)} rows")

    # Subject-level grouping (one diagnosis per subject — take their modal)
    subj_meta = df.groupby('subject_id').agg(
        diagnosis=('diagnosis', lambda s: Counter(s).most_common(1)[0][0])
    ).reset_index()
    print(f"  subjects: {len(subj_meta)}, dx distribution: {Counter(subj_meta['diagnosis'])}")
    subj_meta['diagnosis_coarse'] = subj_meta['diagnosis'].map(DX_TO_COARSE)

    # Clamp n_folds to the number of unique subjects (GroupKFold can't have
    # more splits than groups). When n_folds >= n_subjects we're doing LOO.
    n_subjects = df['subject_id'].nunique()
    if args.n_folds > n_subjects:
        print(f"  NOTE: clamping --n_folds {args.n_folds} to n_subjects={n_subjects} "
              f"(GroupKFold can't have more splits than groups)")
        args.n_folds = n_subjects
    print(f"[2/4] {args.n_folds}-fold GroupKFold cross-validation"
          + (" (LOO: one subject per test fold)" if args.n_folds == n_subjects else ""))
    gkf = GroupKFold(n_splits=args.n_folds)
    subject_ids = df['subject_id'].to_numpy()

    folds_results = []
    all_hard_preds, all_soft_preds, all_modal_preds = [], [], []
    all_true_dx, all_true_coarse = [], []
    all_subj = []

    for fold_i, (train_row_idx, test_row_idx) in enumerate(
            gkf.split(np.arange(len(df)), groups=subject_ids)):
        t0 = time.time()
        train_subjects = set(df.iloc[train_row_idx]['subject_id'])
        test_subjects = set(df.iloc[test_row_idx]['subject_id'])
        print(f"\n--- fold {fold_i+1}/{args.n_folds}: "
              f"{len(train_subjects)} train subjects, {len(test_subjects)} test subjects ---")

        # 2a. Scaler + PCA on TRAIN rows only
        X_train_raw = X_raw[train_row_idx]
        scaler = StandardScaler(copy=False).fit(X_train_raw)
        X_train_scaled = scaler.transform(X_train_raw)
        pca = PCA(n_components=args.pca_dim, random_state=args.random_state,
                  svd_solver='randomized').fit(X_train_scaled)
        del X_train_scaled
        # Project full X_raw — we keep them all (we need test rows for prediction)
        X_pca_full = pca.transform(scaler.transform(X_raw)).astype(np.float32)
        print(f"  pca explained={pca.explained_variance_ratio_.sum():.4f}, "
              f"X_pca: {X_pca_full.shape}")

        # 2b. Partition: last_N of each TRAIN session + balance across diagnoses
        df_train = df.iloc[train_row_idx].reset_index(drop=False).rename(
            columns={'index': 'global_row'})  # keep mapping to X_pca_full
        last_mask = last_n_per_session_mask(df_train, args.last_n)
        df_train_last = df_train[last_mask].reset_index(drop=True)
        train_last_global = df_train_last['global_row'].to_numpy()
        # Balance: pick subset of train_last_global by diagnosis
        # (rng draws inside this function)
        bal_local_idx, target = balance_by_diagnosis_indices(
            df_train_last, df_train_last.index.to_numpy(), rng)
        # bal_local_idx is into df_train_last; map to global rows
        train_bal_global = df_train_last.loc[bal_local_idx, 'global_row'].to_numpy()
        train_bal_dx = df_train_last.loc[bal_local_idx, 'diagnosis'].tolist()
        print(f"  train_last_n: {len(df_train_last)} epochs; "
              f"balanced cap={target} → {len(train_bal_global)} epochs")

        # 2c. Fit GMM on balanced training-last subset
        gm = GaussianMixture(n_components=args.K, covariance_type='full',
                             random_state=args.random_state, n_init=3,
                             max_iter=200, reg_covar=args.reg_covar)
        gm.fit(X_pca_full[train_bal_global])
        train_lbl = gm.predict(X_pca_full[train_bal_global])
        M_train, V_train, chi2_train = contingency(train_lbl, train_bal_dx, DX_ORDER_FULL)
        print(f"  train within-fold V={V_train:.3f}, chi2={chi2_train:.1f}")

        # Cluster → majority dx mapping (within-fold)
        c2dx = cluster_to_dx_majority(train_lbl, train_bal_dx, DX_ORDER_FULL)
        print(f"  cluster→majority_dx: {c2dx}")

        # 2d. Predict on TEST subjects' last_N epochs
        df_test = df.iloc[test_row_idx].reset_index(drop=False).rename(
            columns={'index': 'global_row'})
        last_mask_test = last_n_per_session_mask(df_test, args.last_n)
        df_test_last = df_test[last_mask_test].reset_index(drop=True)
        test_last_global = df_test_last['global_row'].to_numpy()
        test_lbl = gm.predict(X_pca_full[test_last_global])
        # Also compute soft posteriors for test_last
        test_post = gm.predict_proba(X_pca_full[test_last_global])

        # 2e. Metric A: per-subject modal cluster → predicted dx
        df_test_last['cluster'] = test_lbl
        modal_per_subj = df_test_last.groupby('subject_id').agg(
            modal_cluster=('cluster', lambda s: Counter(s).most_common(1)[0][0]),
            true_dx=('diagnosis', lambda s: Counter(s).most_common(1)[0][0]),
        ).reset_index()
        modal_per_subj['pred_dx'] = modal_per_subj['modal_cluster'].map(c2dx)
        modal_per_subj['pred_dx_coarse'] = modal_per_subj['pred_dx'].map(DX_TO_COARSE)
        modal_per_subj['true_dx_coarse'] = modal_per_subj['true_dx'].map(DX_TO_COARSE)
        # Drop subjects whose modal cluster mapped to None
        ok = modal_per_subj['pred_dx'].notna()
        ba_modal_fine = safe_balanced_accuracy(modal_per_subj.loc[ok, 'true_dx'],
                                               modal_per_subj.loc[ok, 'pred_dx'])
        ba_modal_coarse = safe_balanced_accuracy(modal_per_subj.loc[ok, 'true_dx_coarse'],
                                                 modal_per_subj.loc[ok, 'pred_dx_coarse'])
        print(f"  Metric A (modal-cluster→dx): bal_acc 6-class={ba_modal_fine:.3f}, "
              f"3-class={ba_modal_coarse:.3f}")

        # 2f. Metric B+C: subject-level fingerprints → logistic regression
        # Train fingerprints (computed on train-fold last_N, NOT the balanced subset
        # — we want a per-subject summary; balancing was just for clustering)
        df_train_last['cluster'] = gm.predict(X_pca_full[train_last_global])
        train_post_last = gm.predict_proba(X_pca_full[train_last_global])
        # Hard fingerprint per subject (cluster occupancy)
        def hard_fp(group_df):
            v = np.bincount(group_df['cluster'].to_numpy(), minlength=args.K).astype(float)
            return v / v.sum() if v.sum() > 0 else v
        train_hard = df_train_last.groupby('subject_id').apply(hard_fp).to_dict()
        train_subj_dx = df_train_last.groupby('subject_id')['diagnosis'].apply(
            lambda s: Counter(s).most_common(1)[0][0]).to_dict()
        # Soft fingerprint per subject (mean posterior)
        train_post_df = pd.DataFrame(train_post_last,
                                     columns=[f'p{c}' for c in range(args.K)])
        train_post_df['subject_id'] = df_train_last['subject_id'].to_numpy()
        train_soft = train_post_df.groupby('subject_id').mean().to_dict('index')

        train_subj_list = sorted(train_hard.keys())
        Xtr_hard = np.array([train_hard[s] for s in train_subj_list])
        Xtr_soft = np.array([[train_soft[s][f'p{c}'] for c in range(args.K)]
                             for s in train_subj_list])
        ytr = [train_subj_dx[s] for s in train_subj_list]
        ytr_coarse = [DX_TO_COARSE[d] for d in ytr]

        # Test subjects' fingerprints
        df_test_last['cluster'] = test_lbl
        test_hard = df_test_last.groupby('subject_id').apply(hard_fp).to_dict()
        test_post_df = pd.DataFrame(test_post,
                                    columns=[f'p{c}' for c in range(args.K)])
        test_post_df['subject_id'] = df_test_last['subject_id'].to_numpy()
        test_soft = test_post_df.groupby('subject_id').mean().to_dict('index')
        test_subj_list = sorted(test_hard.keys())
        Xte_hard = np.array([test_hard[s] for s in test_subj_list])
        Xte_soft = np.array([[test_soft[s][f'p{c}'] for c in range(args.K)]
                             for s in test_subj_list])
        yte = []
        for s in test_subj_list:
            dxs = df_test_last[df_test_last['subject_id'] == s]['diagnosis']
            yte.append(Counter(dxs).most_common(1)[0][0])
        yte_coarse = [DX_TO_COARSE[d] for d in yte]

        def fit_predict(Xtr, ytr, Xte):
            # sklearn >=1.7 removed the multi_class kwarg; the lbfgs solver
            # defaults to multinomial when given multi-class labels, which is
            # what we want here.
            clf = LogisticRegression(max_iter=2000, class_weight='balanced',
                                     solver='lbfgs',
                                     random_state=args.random_state)
            clf.fit(Xtr, ytr)
            return clf.predict(Xte)

        # Metric B (hard fingerprint)
        pred_hard_fine = fit_predict(Xtr_hard, ytr, Xte_hard)
        pred_hard_coarse = fit_predict(Xtr_hard, ytr_coarse, Xte_hard)
        ba_hard_fine = safe_balanced_accuracy(yte, pred_hard_fine)
        ba_hard_coarse = safe_balanced_accuracy(yte_coarse, pred_hard_coarse)
        # Metric C (soft fingerprint)
        pred_soft_fine = fit_predict(Xtr_soft, ytr, Xte_soft)
        pred_soft_coarse = fit_predict(Xtr_soft, ytr_coarse, Xte_soft)
        ba_soft_fine = safe_balanced_accuracy(yte, pred_soft_fine)
        ba_soft_coarse = safe_balanced_accuracy(yte_coarse, pred_soft_coarse)
        print(f"  Metric B (hard FP→logreg): bal_acc 6={ba_hard_fine:.3f}, 3={ba_hard_coarse:.3f}")
        print(f"  Metric C (soft FP→logreg): bal_acc 6={ba_soft_fine:.3f}, 3={ba_soft_coarse:.3f}")

        # Save per-subject predictions
        for s, t_fine, t_coarse, h_fine, h_coarse, sf_fine, sf_coarse, \
                modal_pred_fine, modal_pred_coarse \
                in zip(test_subj_list, yte, yte_coarse,
                       pred_hard_fine, pred_hard_coarse,
                       pred_soft_fine, pred_soft_coarse,
                       modal_per_subj['pred_dx'].fillna('?'),
                       modal_per_subj['pred_dx_coarse'].fillna('?')):
            all_subj.append({'fold': fold_i+1, 'subject_id': s,
                             'true_dx': t_fine, 'true_dx_coarse': t_coarse,
                             'modal_pred_fine': modal_pred_fine,
                             'modal_pred_coarse': modal_pred_coarse,
                             'hard_pred_fine': h_fine, 'hard_pred_coarse': h_coarse,
                             'soft_pred_fine': sf_fine, 'soft_pred_coarse': sf_coarse})

        # Append for global confusion matrix aggregation (Metric C, coarse)
        all_soft_preds.extend(pred_soft_coarse)
        all_hard_preds.extend(pred_hard_coarse)
        all_modal_preds.extend(modal_per_subj.loc[ok, 'pred_dx_coarse'].fillna('?').tolist())
        all_true_dx.extend(yte)
        all_true_coarse.extend(yte_coarse)

        folds_results.append({
            'fold': fold_i + 1,
            'n_train_subjects': len(train_subjects),
            'n_test_subjects': len(test_subjects),
            'n_train_epochs': int(len(train_bal_global)),
            'n_test_epochs': int(len(test_last_global)),
            'within_fold_V': V_train,
            'within_fold_chi2': chi2_train,
            'cluster_to_dx': {int(k): v for k, v in c2dx.items()},
            'modal_bal_acc_6class': ba_modal_fine,
            'modal_bal_acc_3class': ba_modal_coarse,
            'hard_bal_acc_6class': ba_hard_fine,
            'hard_bal_acc_3class': ba_hard_coarse,
            'soft_bal_acc_6class': ba_soft_fine,
            'soft_bal_acc_3class': ba_soft_coarse,
            'fold_elapsed_s': time.time() - t0,
        })
        print(f"  fold {fold_i+1} done in {(time.time()-t0):.0f}s")

    # -------------------------------------------------------------- aggregate
    print(f"\n[3/4] aggregating across folds")
    df_folds = pd.DataFrame(folds_results)
    print(df_folds.to_string(index=False))

    df_subj_all = pd.DataFrame(all_subj)
    df_subj_all.to_csv(os.path.join(args.output_dir, 'per_subject_predictions.csv'),
                       index=False)

    # Global confusion matrices (coarse 3-class)
    confusion_plot(all_true_coarse, all_soft_preds, COARSE_ORDER,
                   f'Soft FP→logreg | 3-class | bal_acc={safe_balanced_accuracy(all_true_coarse, all_soft_preds):.3f}',
                   os.path.join(args.output_dir, 'confusion_soft_3class.png'))
    confusion_plot(all_true_coarse, all_hard_preds, COARSE_ORDER,
                   f'Hard FP→logreg | 3-class | bal_acc={safe_balanced_accuracy(all_true_coarse, all_hard_preds):.3f}',
                   os.path.join(args.output_dir, 'confusion_hard_3class.png'))
    confusion_plot(all_true_coarse, all_modal_preds, COARSE_ORDER + ['?'],
                   f'Modal-cluster→dx | 3-class | bal_acc={safe_balanced_accuracy(all_true_coarse, all_modal_preds):.3f}',
                   os.path.join(args.output_dir, 'confusion_modal_3class.png'))

    # Aggregate metrics
    agg = {
        'config': vars(args),
        'aggregate': {
            'mean_within_fold_V': float(df_folds['within_fold_V'].mean()),
            'std_within_fold_V': float(df_folds['within_fold_V'].std()),
            'mean_soft_bal_acc_6class': float(df_folds['soft_bal_acc_6class'].mean()),
            'mean_soft_bal_acc_3class': float(df_folds['soft_bal_acc_3class'].mean()),
            'mean_hard_bal_acc_6class': float(df_folds['hard_bal_acc_6class'].mean()),
            'mean_hard_bal_acc_3class': float(df_folds['hard_bal_acc_3class'].mean()),
            'mean_modal_bal_acc_6class': float(df_folds['modal_bal_acc_6class'].mean()),
            'mean_modal_bal_acc_3class': float(df_folds['modal_bal_acc_3class'].mean()),
            'global_soft_bal_acc_3class': safe_balanced_accuracy(all_true_coarse, all_soft_preds),
            'global_hard_bal_acc_3class': safe_balanced_accuracy(all_true_coarse, all_hard_preds),
            'global_modal_bal_acc_3class': safe_balanced_accuracy(all_true_coarse, all_modal_preds),
        },
        'per_fold': folds_results,
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(agg, f, indent=2, default=str)

    print("\n=== aggregate metrics ===")
    a = agg['aggregate']
    print(f"  within-fold V (mean ± std): {a['mean_within_fold_V']:.3f} ± {a['std_within_fold_V']:.3f}")
    print(f"  modal-cluster bal_acc 3-class:  mean {a['mean_modal_bal_acc_3class']:.3f}, global {a['global_modal_bal_acc_3class']:.3f}")
    print(f"  hard-FP    bal_acc 3-class:  mean {a['mean_hard_bal_acc_3class']:.3f}, global {a['global_hard_bal_acc_3class']:.3f}")
    print(f"  soft-FP    bal_acc 3-class:  mean {a['mean_soft_bal_acc_3class']:.3f}, global {a['global_soft_bal_acc_3class']:.3f}")
    print(f"  random-baseline 3-class:     0.333")
    print(f"\n  modal-cluster bal_acc 6-class:  mean {a['mean_modal_bal_acc_6class']:.3f}")
    print(f"  hard-FP    bal_acc 6-class:  mean {a['mean_hard_bal_acc_6class']:.3f}")
    print(f"  soft-FP    bal_acc 6-class:  mean {a['mean_soft_bal_acc_6class']:.3f}")
    print(f"  random-baseline 6-class:     0.167")

    print(f"\n[4/4] artefacts in {args.output_dir}/")


if __name__ == '__main__':
    main()

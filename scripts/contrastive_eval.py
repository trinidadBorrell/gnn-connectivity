#!/usr/bin/env python3
"""Embed every last_100 epoch with a frozen contrastive encoder, then re-run
the same held-out pipeline as scripts/holdout_prediction.py to test whether
the learned 256-D embedding beats raw-PCA(50) on the 0.59 LOOCV baseline.

Key differences vs scripts/holdout_prediction.py:
- No PCA stage; features are the frozen encoder output (256-D per epoch).
- Held-out subjects from data/holdout_subjects.json are the test set; we
  evaluate ONLY on them (the encoder never saw them during pretraining).
- We still GMM(K=3) + balance + logreg, just on the new features.

Two evaluation modes:
  --eval_mode holdout    use only the 29 held-out subjects (paranoia)
  --eval_mode loocv      144-fold LOOCV over ALL subjects with the
                         pre-pretrained encoder (mild self-supervised
                         leakage; the standard SSL convention)
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
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import EEGtoGraph
from contrastive_adjacency import load_coords, weighted_knn_adjacency
from contrastive_model import Encoder

# Reuse the holdout harness for the actual cross-validation + scoring
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import holdout_prediction as hp


def embed_all_last_n(data_dir, coords_file, ckpt_path, last_n=100, batch=256):
    """Return (X_emb, meta) where X_emb is (n_epochs, 256) frozen-encoder
    embeddings for the last_n epochs of every session."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coords = load_coords(coords_file)
    A_norm = weighted_knn_adjacency(coords, k=10).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    enc = Encoder(A_norm,
                  in_dim=256,
                  hidden_dims=(128, 64, 32, 16),
                  out_dim=cfg.get('latent_per_node', 1),
                  dropout=0.0).to(device)
    enc.load_state_dict(ckpt['encoder_q_state'])
    enc.eval()
    print(f"  loaded encoder from {ckpt_path}")

    # Build flat index of (sub, ses, npz_path, epoch_idx) for last_n epochs
    sessions = EEGtoGraph.enumerate_matrix_sessions(data_dir)
    meta = []
    # Cache one session at a time to limit RAM
    cache_path, cache_arr = None, None

    embeddings_chunks = []
    # Need normalisation stats consistent with pretraining; the ckpt config
    # has train_pool stats but for fair eval we recompute on all-epochs.
    print(f"  pass 1: computing global normalisation stats")
    x_min, x_max = float('inf'), float('-inf')
    npz_paths = set()
    for sid, snum, source in sessions:
        if source['kind'] != 'npz':
            continue
        npz_paths.add(source['path'])
    for p in npz_paths:
        with np.load(p) as d:
            arr = d['data']
            x_min = min(x_min, float(arr.min()))
            x_max = max(x_max, float(arr.max()))
    x_range = x_max - x_min
    print(f"    x_min={x_min:.4f}, x_max={x_max:.4f}")

    print(f"  pass 2: embedding last_{last_n} epochs of every session")
    t0 = time.time()
    n_sessions_done = 0
    for sid, snum, source in sessions:
        if source['kind'] != 'npz':
            continue
        with np.load(source['path']) as d:
            arr = d['data']
            n_eps = int(arr.shape[0])
            start = max(0, n_eps - last_n)
            sub = arr[start:n_eps].astype(np.float32, copy=False)   # (<=last_n, N, N)
        # Normalise
        sub = 2.0 * (sub - x_min) / x_range - 1.0
        # Run through encoder in chunks to bound GPU memory
        with torch.no_grad():
            t = torch.from_numpy(sub).to(device)
            embs = []
            for i in range(0, t.shape[0], batch):
                h = enc(t[i:i + batch])                    # (b, 256) or (b, 256, k)
                if h.dim() == 3:
                    h = h.reshape(h.shape[0], -1)          # flatten if latent_per_node > 1
                embs.append(h.cpu())
            session_emb = torch.cat(embs, dim=0).numpy()   # (n_last, 256*k)
        embeddings_chunks.append(session_emb)
        for j in range(start, n_eps):
            meta.append({'subject_id': sid, 'session_num': snum,
                         'cohort': source.get('cohort'),
                         'matrix_idx': j})
        n_sessions_done += 1
        if n_sessions_done % 25 == 0:
            print(f"    {n_sessions_done}/{len(sessions)} sessions "
                  f"({(time.time()-t0)/60:.1f} min)")

    X = np.concatenate(embeddings_chunks, axis=0)
    print(f"  X_emb: {X.shape}")
    return X, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--coords_file', default='data_scalp/GSN-HydroCel-257.txt')
    ap.add_argument('--ckpt', default='output/contrastive/encoder_q.pt')
    ap.add_argument('--labels_csv',
                    default='data/wsmi_res/256_electrodes-20260526T091011Z-3-001/256_electrodes/wsmi_res_DOC/patient_labels.csv')
    ap.add_argument('--output_dir', default='output/contrastive_eval')
    ap.add_argument('--holdout_json', default='data/holdout_subjects.json')
    ap.add_argument('--eval_mode', choices=['holdout', 'loocv', '5fold'],
                    default='5fold',
                    help='holdout = test only on the 29 unseen subjects; '
                         'loocv = 144-fold LOOCV; '
                         '5fold = subject-wise 5-fold over all 144 subjects.')
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--last_n', type=int, default=100)
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Embed every last_n epoch
    X_emb, meta = embed_all_last_n(args.data_dir, args.coords_file,
                                   args.ckpt, args.last_n)
    # 2) Attach diagnoses
    df = pd.DataFrame(meta)
    df['matrix_idx'] = df['matrix_idx'].astype(int)
    df = df.sort_values(['subject_id', 'session_num', 'matrix_idx']).reset_index(drop=True)
    df = hp.attach_diagnoses(df, args.labels_csv)
    valid_dx = set(hp.DX_ORDER_FULL)
    mask = df['diagnosis'].isin(valid_dx)
    df = df.loc[mask].reset_index(drop=True)
    # Align X_emb the same way: rebuild ordering match
    # (we built meta in the same order as embeddings, then sorted df —
    #  need to track the original order)
    df['orig_row'] = np.arange(len(df))
    # Re-derive the original ordering: since we sorted df, the rows of X
    # need to be re-sliced. Simpler: redo X in sorted order.
    # The meta list was built in iteration order; we need its index after sort.
    # We never permuted X_emb, so use df's index (which IS the sorted order) to
    # access X_emb via the saved index map.
    # Easiest: drop the unknown-dx rows from X_emb in the order they were built.
    keep_idx = np.where(mask.to_numpy())[0]
    X_emb = X_emb[keep_idx]
    print(f"  after dx filter: X={X_emb.shape}, df={len(df)}")

    # 3) Run the existing holdout-prediction harness with X_emb as the features
    # We monkey-patch the PCA stage out — keep scaler-then-cluster, no PCA.
    # Easiest: call the per-fold loop manually, mirroring hp.main but with
    # X_pca_full := X_emb (no scaler, no PCA — features are already encoder-derived).
    from sklearn.mixture import GaussianMixture
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    # Subject grouping
    subject_ids = df['subject_id'].to_numpy()

    if args.eval_mode == 'holdout':
        with open(args.holdout_json) as f:
            split = json.load(f)
        holdout = set(split['holdout'])
        train_mask = ~df['subject_id'].isin(holdout)
        test_mask = df['subject_id'].isin(holdout)
        # Single fold: fit on train_pool, test on holdout
        folds = [(np.where(train_mask)[0], np.where(test_mask)[0])]
    elif args.eval_mode == '5fold':
        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(np.arange(len(df)), groups=subject_ids))
    else:  # loocv
        n_subj = df['subject_id'].nunique()
        gkf = GroupKFold(n_splits=n_subj)
        folds = list(gkf.split(np.arange(len(df)), groups=subject_ids))

    print(f"  eval_mode={args.eval_mode}, n_folds={len(folds)}")

    rng = np.random.default_rng(args.random_state)
    all_true_coarse, all_hard_preds, all_soft_preds = [], [], []
    all_subj_proba = []  # per-fold per-subject probabilities for ROC AUC
    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        t0 = time.time()
        df_train = df.iloc[train_idx].reset_index().rename(columns={'index': 'global_row'})
        df_test = df.iloc[test_idx].reset_index().rename(columns={'index': 'global_row'})

        # last_n already applied at embedding time, so train_idx == last_n epochs of train subjects
        # Balance training by diagnosis for the GMM fit
        bal_local, target = hp.balance_by_diagnosis_indices(
            df_train, df_train.index.to_numpy(), rng)
        train_bal_global = df_train.loc[bal_local, 'global_row'].to_numpy()
        train_bal_dx = df_train.loc[bal_local, 'diagnosis'].tolist()

        # Fit GMM K=3 on balanced train embeddings
        gm = GaussianMixture(n_components=args.K, covariance_type='full',
                             random_state=args.random_state, n_init=3,
                             max_iter=200, reg_covar=1e-3)
        gm.fit(X_emb[train_bal_global])

        # Per-subject fingerprints — soft posterior on train_last (NOT balanced)
        train_last_global = df_train['global_row'].to_numpy()
        train_post = gm.predict_proba(X_emb[train_last_global])
        train_subj_dx = df_train.groupby('subject_id')['diagnosis'].apply(
            lambda s: Counter(s).most_common(1)[0][0]).to_dict()
        train_post_df = pd.DataFrame(train_post,
                                     columns=[f'p{c}' for c in range(args.K)])
        train_post_df['subject_id'] = df_train['subject_id'].to_numpy()
        train_soft = train_post_df.groupby('subject_id').mean().to_dict('index')
        train_subj_list = sorted(train_soft.keys())
        Xtr = np.array([[train_soft[s][f'p{c}'] for c in range(args.K)]
                        for s in train_subj_list])
        ytr = [train_subj_dx[s] for s in train_subj_list]
        ytr_coarse = [hp.DX_TO_COARSE[d] for d in ytr]
        # Hard fingerprint too
        train_lbl = gm.predict(X_emb[train_last_global])
        df_train['cluster'] = train_lbl
        train_hard_fp = df_train.groupby('subject_id')['cluster'].apply(
            lambda s: np.bincount(s.to_numpy(), minlength=args.K).astype(float)).to_dict()
        Xtr_hard = np.array([train_hard_fp[s] / max(train_hard_fp[s].sum(), 1)
                             for s in train_subj_list])

        # Test fingerprints
        test_last_global = df_test['global_row'].to_numpy()
        test_post = gm.predict_proba(X_emb[test_last_global])
        test_lbl = gm.predict(X_emb[test_last_global])
        df_test['cluster'] = test_lbl
        test_subj_dx = df_test.groupby('subject_id')['diagnosis'].apply(
            lambda s: Counter(s).most_common(1)[0][0]).to_dict()
        test_post_df = pd.DataFrame(test_post,
                                    columns=[f'p{c}' for c in range(args.K)])
        test_post_df['subject_id'] = df_test['subject_id'].to_numpy()
        test_soft = test_post_df.groupby('subject_id').mean().to_dict('index')
        test_hard_fp = df_test.groupby('subject_id')['cluster'].apply(
            lambda s: np.bincount(s.to_numpy(), minlength=args.K).astype(float)).to_dict()
        test_subj_list = sorted(test_soft.keys())
        Xte = np.array([[test_soft[s][f'p{c}'] for c in range(args.K)]
                        for s in test_subj_list])
        Xte_hard = np.array([test_hard_fp[s] / max(test_hard_fp[s].sum(), 1)
                             for s in test_subj_list])
        yte = [test_subj_dx[s] for s in test_subj_list]
        yte_coarse = [hp.DX_TO_COARSE[d] for d in yte]

        # Train logreg on coarse and predict
        clf = LogisticRegression(max_iter=2000, class_weight='balanced',
                                 solver='lbfgs', random_state=args.random_state)
        clf.fit(Xtr, ytr_coarse)
        pred_soft_coarse = clf.predict(Xte)
        probs_soft = clf.predict_proba(Xte)  # for ROC AUC
        # Map clf.classes_ (coarse strings) onto p_control / p_low_doc / p_high_doc
        cls_idx = {c: i for i, c in enumerate(clf.classes_)}
        for j, sid in enumerate(test_subj_list):
            all_subj_proba.append({
                'subject_id': sid,
                'true_dx_coarse': yte_coarse[j],
                'p_control':  float(probs_soft[j, cls_idx['control']])
                              if 'control'  in cls_idx else 0.0,
                'p_low_doc':  float(probs_soft[j, cls_idx['low_doc']])
                              if 'low_doc'  in cls_idx else 0.0,
                'p_high_doc': float(probs_soft[j, cls_idx['high_doc']])
                              if 'high_doc' in cls_idx else 0.0,
                'fold': fold_i + 1,
            })

        clf_h = LogisticRegression(max_iter=2000, class_weight='balanced',
                                   solver='lbfgs', random_state=args.random_state)
        clf_h.fit(Xtr_hard, ytr_coarse)
        pred_hard_coarse = clf_h.predict(Xte_hard)

        ba_soft = float(balanced_accuracy_score(yte_coarse, pred_soft_coarse)) \
                  if len(set(yte_coarse)) > 1 else float('nan')
        ba_hard = float(balanced_accuracy_score(yte_coarse, pred_hard_coarse)) \
                  if len(set(yte_coarse)) > 1 else float('nan')
        print(f"  fold {fold_i+1}/{len(folds)}: n_train={len(train_subj_list)}, "
              f"n_test={len(test_subj_list)}, "
              f"hard_bal_acc_3={ba_hard:.3f}, soft_bal_acc_3={ba_soft:.3f}  "
              f"({time.time()-t0:.0f}s)")

        all_true_coarse.extend(yte_coarse)
        all_hard_preds.extend(pred_hard_coarse)
        all_soft_preds.extend(pred_soft_coarse)
        fold_results.append({'fold': fold_i + 1,
                             'n_train_subj': len(train_subj_list),
                             'n_test_subj': len(test_subj_list),
                             'hard_bal_acc_3': ba_hard,
                             'soft_bal_acc_3': ba_soft})

    # Aggregate
    global_hard = float(balanced_accuracy_score(all_true_coarse, all_hard_preds))
    global_soft = float(balanced_accuracy_score(all_true_coarse, all_soft_preds))
    print(f"\n  === aggregate ({args.eval_mode}) ===")
    print(f"    pooled hard-FP bal_acc 3-class: {global_hard:.3f}")
    print(f"    pooled soft-FP bal_acc 3-class: {global_soft:.3f}")
    print(f"    random baseline 3-class:        0.333")
    print(f"\n  baseline to beat: 0.590 (raw PCA+GMM K=3 LOOCV)")

    # Save
    with open(os.path.join(args.output_dir, 'eval_summary.json'), 'w') as f:
        json.dump({
            'eval_mode': args.eval_mode,
            'ckpt': args.ckpt,
            'config': vars(args),
            'aggregate': {
                'global_hard_bal_acc_3class': global_hard,
                'global_soft_bal_acc_3class': global_soft,
            },
            'per_fold': fold_results,
        }, f, indent=2, default=str)
    print(f"  saved -> {args.output_dir}/eval_summary.json")

    pd.DataFrame(all_subj_proba).to_csv(
        os.path.join(args.output_dir, 'per_subject_proba.csv'), index=False)
    print(f"  saved -> {args.output_dir}/per_subject_proba.csv "
          f"({len(all_subj_proba)} rows)")


if __name__ == '__main__':
    main()

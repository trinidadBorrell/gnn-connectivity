#!/usr/bin/env python3
"""Score a trained `src/model.py` GNN on the SAME held-out protocol as the
raw-wSMI baseline (scripts/holdout_prediction.py), so learned-representation
numbers are head-to-head comparable with the 0.61 3-class / 0.82 AUC floor.

This is the `src/model.py` analogue of scripts/contrastive_eval.py (which only
bridges the 256-node contrastive_model.Encoder). It works for BOTH the
reconstruction autoencoders (gae/vgae/gae_vae/gat_vae) and the encoder-only
CEBRA models (enc_gae_fc/enc_gat_fc), because both go through the shared
LatentBundle interface in src/cluster_analysis.py.

Pipeline (mirrors contrastive_eval.py's fold loop, which is verified working):
  1. load output/<run>/splits/graphs.pt  (train+val+test graphs pooled)
  2. load output/<run>/models/<tag>/model.pt  ({model_state_dict, config})
  3. rebuild + load the model, extract ONE latent vector per epoch
  4. keep last_n epochs/session, map diagnoses -> {control, low_doc, high_doc}
  5. GroupKFold (5fold / loocv) -> balance train by dx -> GMM(K) on latents
     -> soft/hard per-subject fingerprint -> LogisticRegression -> bal_acc
  6. dump eval_summary.json + per_subject_proba.csv (feeds compare_roc.py)

DRAFT — not yet smoke-tested. Run via slurm/gae_latent_eval.sbatch on a
compute node (never the login node). See docs/lab-notebook/chapter_07.

Caveat: graphs.pt was min-max normalised in-place using the pipeline's OWN
70/15/15 train split, which overlaps our GroupKFold folds -> a mild
normalisation leak. Acceptable for a first pass; for the paranoid version,
re-run stage_load per fold. Noted in chapter 7.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
sys.path.insert(0, _HERE)

from train import build_decoder_model, _is_variational  # noqa: E402
from cluster_analysis import (  # noqa: E402
    extract_latents_from_graphs, extract_embeddings_encoder,
)
import holdout_prediction as hp  # noqa: E402

ENCODER_KINDS = {"enc_gae_fc", "enc_gat_fc"}


def to_coarse(dx: str, dgroup: str):
    """Map a raw diagnosis / diagnosis_group onto {control, low_doc, high_doc}.

    Tries the notebook's fine map first, then falls back to the pipeline's
    coarse diagnosis_group. Returns None for classes we drop (EMCS/COMA in
    some granularities, unknowns)."""
    d = (dx or '').strip()
    if d in hp.DX_TO_COARSE:
        return hp.DX_TO_COARSE[d]
    g = (dgroup or '').strip().upper()
    if g in ('CONTROL', 'HC', 'CTRL'):
        return 'control'
    if g in ('UWS', 'VS'):
        return 'low_doc'
    if g in ('MCS', 'MCS-', 'MCS+', 'MCS_MINUS', 'MCS_PLUS', 'EMCS'):
        return 'high_doc'
    return None


def load_model(model_path, device):
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model_kind = cfg.get("model_kind") or cfg.get("model") or "gae"
    in_channels = cfg.get("in_channels")
    if in_channels is None:
        raise ValueError("checkpoint config lacks in_channels")
    model = build_decoder_model(model_kind, in_channels, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  loaded {model_kind} (in_channels={in_channels}, "
          f"latent_dim={cfg.get('latent_dim')}) from {model_path}")
    return model, model_kind, cfg


def embed_run(out_root, model_kind, aggregate, device):
    """Return (X_emb, df) with one latent row per pooled epoch + metadata."""
    graphs_path = os.path.join(out_root, "splits", "graphs.pt")
    payload = torch.load(graphs_path, weights_only=False)
    graphs, splits = [], []
    for tag in ("train", "val", "test"):
        for g in payload.get(tag, []):
            graphs.append(g)
            splits.append(tag)
    print(f"  pooled {len(graphs)} graphs from {graphs_path}")

    model_path = os.path.join(out_root, "models", model_kind, "model.pt")
    model, kind, _cfg = load_model(model_path, device)

    if kind in ENCODER_KINDS:
        bundle = extract_embeddings_encoder(model, graphs, splits, device=device)
    else:
        bundle = extract_latents_from_graphs(
            model, graphs, splits, device=device, aggregate=aggregate)

    df = pd.DataFrame({
        'subject_id': bundle.subject_ids,
        'session_num': bundle.sessions,
        'matrix_idx': [int(e) for e in bundle.epochs],
        'diagnosis': bundle.diagnoses,
        'diagnosis_group': bundle.diagnosis_groups,
    })
    return bundle.embeds.astype(np.float32), df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_name', required=True,
                    help='output/<run_name> dir with splits/ + models/')
    ap.add_argument('--output_root', default='output')
    ap.add_argument('--model_kind', required=True,
                    help='gae|vgae|gae_vae|gat_vae|enc_gae_fc|enc_gat_fc '
                         '(the models/<tag> subfolder)')
    ap.add_argument('--graph_latent_agg', choices=['mean', 'flatten'],
                    default='flatten',
                    help='node-latent aggregation for autoencoders')
    ap.add_argument('--eval_mode', choices=['5fold', 'loocv'], default='5fold')
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--last_n', type=int, default=100)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    from sklearn.mixture import GaussianMixture
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score

    device = torch.device('cpu') if args.cpu else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    out_root = os.path.join(args.output_root, args.run_name)
    eval_dir = os.path.join(out_root, f"gae_eval_{args.model_kind}")
    os.makedirs(eval_dir, exist_ok=True)

    # 1) embed every epoch
    X_emb, df = embed_run(out_root, args.model_kind, args.graph_latent_agg, device)

    # 2) coarse labels + drop unmapped classes
    df['coarse'] = [to_coarse(d, g) for d, g in
                    zip(df['diagnosis'], df['diagnosis_group'])]
    keep = df['coarse'].notna().to_numpy()
    X_emb, df = X_emb[keep], df.loc[keep].reset_index(drop=True)

    # 3) last_n epochs per session
    df = df.sort_values(['subject_id', 'session_num', 'matrix_idx'])
    order = df.index.to_numpy()
    X_emb = X_emb[order]
    df = df.reset_index(drop=True)
    last_mask = np.asarray(hp.last_n_per_session_mask(df, args.last_n))
    X_emb, df = X_emb[last_mask], df.loc[last_mask].reset_index(drop=True)
    print(f"  after coarse-filter + last_{args.last_n}: X={X_emb.shape}, "
          f"subjects={df['subject_id'].nunique()}")

    subject_ids = df['subject_id'].to_numpy()
    if args.eval_mode == '5fold':
        folds = list(GroupKFold(n_splits=5).split(np.arange(len(df)),
                                                  groups=subject_ids))
    else:
        n_subj = df['subject_id'].nunique()
        folds = list(GroupKFold(n_splits=n_subj).split(np.arange(len(df)),
                                                       groups=subject_ids))
    print(f"  eval_mode={args.eval_mode}, n_folds={len(folds)}")

    rng = np.random.default_rng(args.random_state)
    all_true, all_hard, all_soft, all_proba, fold_rows = [], [], [], [], []

    for fi, (tr, te) in enumerate(folds):
        t0 = time.time()
        df_tr = df.iloc[tr].reset_index().rename(columns={'index': 'grow'})
        df_te = df.iloc[te].reset_index().rename(columns={'index': 'grow'})

        bal_local, _ = hp.balance_by_diagnosis_indices(
            df_tr.assign(diagnosis=df_tr['coarse']), df_tr.index.to_numpy(), rng)
        tr_bal = df_tr.loc[bal_local, 'grow'].to_numpy()

        gm = GaussianMixture(n_components=args.K, covariance_type='full',
                             random_state=args.random_state, n_init=3,
                             max_iter=200, reg_covar=1e-3)
        gm.fit(X_emb[tr_bal])

        # per-subject soft + hard fingerprints on train (unbalanced last_n)
        tr_glob = df_tr['grow'].to_numpy()
        tr_post = gm.predict_proba(X_emb[tr_glob])
        tr_lbl = gm.predict(X_emb[tr_glob])
        df_tr['cluster'] = tr_lbl
        subj_dx_tr = df_tr.groupby('subject_id')['coarse'].apply(
            lambda s: Counter(s).most_common(1)[0][0]).to_dict()
        post_df = pd.DataFrame(tr_post, columns=[f'p{c}' for c in range(args.K)])
        post_df['subject_id'] = df_tr['subject_id'].to_numpy()
        soft_tr = post_df.groupby('subject_id').mean()
        subj_list = sorted(soft_tr.index)
        Xtr = soft_tr.loc[subj_list].to_numpy()
        ytr = [subj_dx_tr[s] for s in subj_list]
        hard_tr = df_tr.groupby('subject_id')['cluster'].apply(
            lambda s: np.bincount(s.to_numpy(), minlength=args.K).astype(float)).to_dict()
        Xtr_h = np.array([hard_tr[s] / max(hard_tr[s].sum(), 1) for s in subj_list])

        te_glob = df_te['grow'].to_numpy()
        te_post = gm.predict_proba(X_emb[te_glob])
        df_te['cluster'] = gm.predict(X_emb[te_glob])
        subj_dx_te = df_te.groupby('subject_id')['coarse'].apply(
            lambda s: Counter(s).most_common(1)[0][0]).to_dict()
        post_df_te = pd.DataFrame(te_post, columns=[f'p{c}' for c in range(args.K)])
        post_df_te['subject_id'] = df_te['subject_id'].to_numpy()
        soft_te = post_df_te.groupby('subject_id').mean()
        te_list = sorted(soft_te.index)
        Xte = soft_te.loc[te_list].to_numpy()
        hard_te = df_te.groupby('subject_id')['cluster'].apply(
            lambda s: np.bincount(s.to_numpy(), minlength=args.K).astype(float)).to_dict()
        Xte_h = np.array([hard_te[s] / max(hard_te[s].sum(), 1) for s in te_list])
        yte = [subj_dx_te[s] for s in te_list]

        clf = LogisticRegression(max_iter=2000, class_weight='balanced',
                                 solver='lbfgs', random_state=args.random_state)
        clf.fit(Xtr, ytr)
        pred_soft = clf.predict(Xte)
        probs = clf.predict_proba(Xte)
        ci = {c: i for i, c in enumerate(clf.classes_)}
        for j, sid in enumerate(te_list):
            all_proba.append({
                'subject_id': sid, 'true_dx_coarse': yte[j],
                'p_control': float(probs[j, ci['control']]) if 'control' in ci else 0.0,
                'p_low_doc': float(probs[j, ci['low_doc']]) if 'low_doc' in ci else 0.0,
                'p_high_doc': float(probs[j, ci['high_doc']]) if 'high_doc' in ci else 0.0,
                'fold': fi + 1})

        clf_h = LogisticRegression(max_iter=2000, class_weight='balanced',
                                   solver='lbfgs', random_state=args.random_state)
        clf_h.fit(Xtr_h, ytr)
        pred_hard = clf_h.predict(Xte_h)

        ba_s = float(balanced_accuracy_score(yte, pred_soft)) if len(set(yte)) > 1 else float('nan')
        ba_h = float(balanced_accuracy_score(yte, pred_hard)) if len(set(yte)) > 1 else float('nan')
        print(f"  fold {fi+1}/{len(folds)}: n_te={len(te_list)}, "
              f"soft={ba_s:.3f}, hard={ba_h:.3f} ({time.time()-t0:.0f}s)")
        all_true += yte; all_hard += list(pred_hard); all_soft += list(pred_soft)
        fold_rows.append({'fold': fi + 1, 'n_test_subj': len(te_list),
                          'soft_bal_acc_3': ba_s, 'hard_bal_acc_3': ba_h})

    g_hard = float(balanced_accuracy_score(all_true, all_hard))
    g_soft = float(balanced_accuracy_score(all_true, all_soft))
    per_fold = pd.DataFrame(fold_rows)
    print(f"\n  === {args.model_kind} ({args.eval_mode}) ===")
    print(f"    pooled soft-FP 3-class bal_acc: {g_soft:.3f}")
    print(f"    pooled hard-FP 3-class bal_acc: {g_hard:.3f}")
    if args.eval_mode == '5fold':
        print(f"    per-fold soft mean±std: {per_fold['soft_bal_acc_3'].mean():.3f} "
              f"± {per_fold['soft_bal_acc_3'].std():.3f}")
    print(f"    baseline to beat: 0.611 ± 0.131 (raw PCA+GMM K=3 5-fold)")

    with open(os.path.join(eval_dir, 'eval_summary.json'), 'w') as f:
        json.dump({'model_kind': args.model_kind, 'eval_mode': args.eval_mode,
                   'args': vars(args),
                   'aggregate': {'pooled_soft_bal_acc_3class': g_soft,
                                 'pooled_hard_bal_acc_3class': g_hard},
                   'per_fold': fold_rows}, f, indent=2, default=str)
    pd.DataFrame(all_proba).to_csv(
        os.path.join(eval_dir, 'per_subject_proba.csv'), index=False)
    print(f"  saved -> {eval_dir}/{{eval_summary.json,per_subject_proba.csv}}")


if __name__ == '__main__':
    main()

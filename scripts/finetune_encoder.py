#!/usr/bin/env python3
"""Supervised classification heads on a trained CEBRA GNNEncoder — the direct
alternative to the GMM-fingerprint readout in scripts/gae_latent_eval.py.

Two modes (both trained with class-weighted cross-entropy on the 3-class coarse
target control / low_doc / high_doc):

  --mode full    encoder + head, ALL weights trained end-to-end (fine-tune)
  --mode frozen  encoder FROZEN, only the head trained (linear/MLP probe)

  --head linear  Linear(latent_dim -> 3)  + softmax
  --head mlp     Linear(latent -> hidden) -> ReLU -> Dropout -> Linear(hidden -> 3)

Same leakage-safe protocol as gae_latent_eval / the baseline: pooled last_n
epochs/session, subject-disjoint 5-fold GroupKFold, a FRESH encoder+head rebuilt
from the checkpoint per fold (so the test subjects never leak). Per-subject
prediction = mean softmax over the subject's epochs. Writes 3-class bal_acc +
per_subject_proba.csv (feeds scripts/compare_roc.py for the binary AUCs).

Run on a GPU node (see slurm/finetune_encoder.sbatch). DRAFT — smoke on cluster.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
sys.path.insert(0, _HERE)

from train import build_decoder_model  # noqa: E402  (handles enc_* AND autoencoders)
from torch_geometric.nn import global_mean_pool  # noqa: E402
import holdout_prediction as hp  # noqa: E402

COARSE = {'control': 0, 'low_doc': 1, 'high_doc': 2}
INV = {0: 'control', 1: 'low_doc', 2: 'high_doc'}
ENCODER_KINDS = {'enc_gae_fc', 'enc_gat_fc'}


def to_coarse(dx, dgroup):
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


class Classifier(nn.Module):
    """Backbone (encoder-only OR autoencoder) + classification head.

    - encoder-only (enc_gae_fc/enc_gat_fc): backbone returns one L2-normed
      graph embedding directly.
    - autoencoder (gae/vgae/gae_vae/gat_vae): use `model.encode` -> node latents
      (VGAE/GAEVAE return (mu, logvar) -> mu), then global-mean-pool to one vector
      per graph. Head input dim = latent_dim in every case.
    """
    def __init__(self, model, model_kind, latent_dim, head='linear',
                 hidden=32, dropout=0.3, n_classes=3):
        super().__init__()
        self.model = model
        self.is_encoder = model_kind in ENCODER_KINDS
        if head == 'linear':
            self.clf = nn.Linear(latent_dim, n_classes)
        else:
            self.clf = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def embed(self, x, edge_index, batch):
        if self.is_encoder:
            return self.model(x, edge_index, batch)      # (G, latent), L2-normed
        enc = self.model.encode(x, edge_index)           # node latents (or (mu,..))
        z_nodes = enc[0] if isinstance(enc, tuple) else enc
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        return global_mean_pool(z_nodes, batch)          # (G, latent)

    def forward(self, x, edge_index, batch):
        return self.clf(self.embed(x, edge_index, batch))


def build_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck['config']
    model_kind = cfg.get('model_kind') or cfg.get('model') or 'enc_gae_fc'
    model = build_decoder_model(model_kind, cfg['in_channels'], cfg)
    model.load_state_dict(ck['model_state_dict'])
    return model.to(device), model_kind, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--output_root', default='output')
    ap.add_argument('--model_kind', default='enc_gae_fc')
    ap.add_argument('--mode', choices=['full', 'frozen'], required=True)
    ap.add_argument('--head', choices=['linear', 'mlp'], default='linear')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--last_n', type=int, default=100)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--task', choices=['coarse3', 'binary_doc'], default='coarse3',
                    help='coarse3 = control/low_doc/high_doc (3-class); '
                         'binary_doc = low_doc(UWS/COMA) vs high_doc(MCS/EMCS), '
                         'controls dropped -> a dedicated head for the within-DOC '
                         '(MCS-vs-UWS) boundary')
    args = ap.parse_args()

    # task-specific label maps
    if args.task == 'binary_doc':
        LMAP = {'low_doc': 0, 'high_doc': 1}
        INV_T = {0: 'low_doc', 1: 'high_doc'}
        NCLS, KEEP = 2, {'low_doc', 'high_doc'}
    else:
        LMAP, INV_T, NCLS, KEEP = COARSE, INV, 3, {'control', 'low_doc', 'high_doc'}

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_root = os.path.join(args.output_root, args.run_name)
    tag = args.tag or f"{args.mode}_{args.head}"
    eval_dir = os.path.join(out_root, f"finetune_{tag}")
    os.makedirs(eval_dir, exist_ok=True)
    ckpt_path = os.path.join(out_root, 'models', args.model_kind, 'model.pt')

    # ---- load pooled graphs + coarse labels + last_n mask ----
    payload = torch.load(os.path.join(out_root, 'splits', 'graphs.pt'),
                         weights_only=False)
    graphs = []
    for sp in ('train', 'val', 'test'):
        graphs += list(payload.get(sp, []))
    rows = []
    for i, g in enumerate(graphs):
        c = to_coarse(getattr(g, 'diagnosis', None),
                      getattr(g, 'diagnosis_group', None))
        rows.append({'idx': i, 'subject_id': str(getattr(g, 'subject_id', '?')),
                     'session_num': str(getattr(g, 'session_num', '?')),
                     'matrix_idx': int(getattr(g, 'matrix_idx', -1)), 'coarse': c})
    df = pd.DataFrame(rows)
    df = df[df['coarse'].isin(KEEP)].sort_values(
        ['subject_id', 'session_num', 'matrix_idx']).reset_index(drop=True)
    mask = np.asarray(hp.last_n_per_session_mask(df, args.last_n))
    df = df[mask].reset_index(drop=True)
    y_all = df['coarse'].map(LMAP).to_numpy()
    idx_all = df['idx'].to_numpy()
    subj_all = df['subject_id'].to_numpy()
    print(f"  {len(df)} epochs, {df['subject_id'].nunique()} subjects, "
          f"task={args.task} ({NCLS}-class), mode={args.mode}, head={args.head}, "
          f"device={device}")

    folds = list(GroupKFold(5).split(np.arange(len(df)), groups=subj_all))
    all_proba, fold_rows = [], []

    for fi, (tr, te) in enumerate(folds):
        t0 = time.time()
        model, model_kind, cfg = build_model(ckpt_path, device)
        clf = Classifier(model, model_kind, cfg['latent_dim'], head=args.head,
                         n_classes=NCLS).to(device)
        if args.mode == 'frozen':
            for p in clf.model.parameters():
                p.requires_grad = False
            params = list(clf.clf.parameters())
        else:
            params = list(clf.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        ytr = y_all[tr]
        cnt = np.bincount(ytr, minlength=NCLS).astype(float)
        cnt[cnt == 0] = 1.0
        w = torch.tensor(len(ytr) / (NCLS * cnt), dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=w)

        train_graphs = []
        for j in tr:
            g = graphs[idx_all[j]]
            g.y = torch.tensor([int(y_all[j])], dtype=torch.long)
            train_graphs.append(g)
        loader = PyGLoader(train_graphs, batch_size=args.batch_size, shuffle=True)

        for ep in range(args.epochs):
            if args.mode == 'full':
                clf.train()
            else:
                clf.model.eval()
                clf.clf.train()
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                logits = clf(batch.x, batch.edge_index, batch.batch)
                loss = crit(logits, batch.y.view(-1))
                loss.backward()
                opt.step()

        # ---- predict test epochs -> per-subject mean softmax ----
        clf.eval()
        te_graphs = [graphs[idx_all[j]] for j in te]
        te_loader = PyGLoader(te_graphs, batch_size=256, shuffle=False)
        probs = []
        with torch.no_grad():
            for batch in te_loader:
                batch = batch.to(device)
                p = torch.softmax(clf(batch.x, batch.edge_index, batch.batch), dim=1)
                probs.append(p.cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        pcols = [f'p{k}' for k in range(NCLS)]
        te_df = df.iloc[te].reset_index(drop=True).copy()
        te_df[pcols] = probs

        f_true, f_pred = [], []
        for sid, sub in te_df.groupby('subject_id'):
            mp = sub[pcols].mean().to_numpy()
            true = sub['coarse'].iloc[0]
            pred = INV_T[int(mp.argmax())]
            f_true.append(true)
            f_pred.append(pred)
            # write proba in the p_control/p_low_doc/p_high_doc schema compare_roc
            # expects; binary_doc has no control column -> 0.
            if args.task == 'binary_doc':
                rec = {'p_control': 0.0, 'p_low_doc': float(mp[0]),
                       'p_high_doc': float(mp[1])}
            else:
                rec = {'p_control': float(mp[0]), 'p_low_doc': float(mp[1]),
                       'p_high_doc': float(mp[2])}
            all_proba.append({'subject_id': sid, 'true_dx_coarse': true,
                              **rec, 'fold': fi + 1})
        ba = float(balanced_accuracy_score(f_true, f_pred)) if len(set(f_true)) > 1 else float('nan')
        fold_rows.append({'fold': fi + 1, 'n_test_subj': len(f_true), 'bal_acc': ba})
        print(f"  fold {fi+1}/5: n_te={len(f_true)}, bal_acc={ba:.3f} "
              f"({time.time()-t0:.0f}s)")

    proba_df = pd.DataFrame(all_proba)
    per_fold = pd.DataFrame(fold_rows)
    mean_ba, std_ba = per_fold['bal_acc'].mean(), per_fold['bal_acc'].std()
    print(f"\n  === {args.run_name} finetune {tag} (task={args.task}) ===")
    print(f"    per-fold bal_acc: {mean_ba:.3f} ± {std_ba:.3f}")
    auc = None
    if args.task == 'binary_doc':
        from sklearn.metrics import roc_auc_score
        y = (proba_df['true_dx_coarse'] == 'high_doc').astype(int)
        auc = float(roc_auc_score(y, proba_df['p_high_doc'])) if y.nunique() > 1 else float('nan')
        pooled_pred = ['high_doc' if p > 0.5 else 'low_doc' for p in proba_df['p_high_doc']]
        pooled = float(balanced_accuracy_score(proba_df['true_dx_coarse'], pooled_pred))
        print(f"    pooled  bal_acc: {pooled:.3f}")
        print(f"    MCS-vs-UWS pooled ROC AUC: {auc:.3f}  (ceiling to break: ~0.71)")
    else:
        pooled_pred = [INV_T[i] for i in
                       proba_df[['p_control', 'p_low_doc', 'p_high_doc']].to_numpy().argmax(1)]
        pooled = float(balanced_accuracy_score(proba_df['true_dx_coarse'], pooled_pred))
        print(f"    pooled  bal_acc: {pooled:.3f}")
        print(f"    baseline to beat: 0.611 (raw PCA+GMM K=3 3-class)")

    with open(os.path.join(eval_dir, 'eval_summary.json'), 'w') as f:
        json.dump({'run_name': args.run_name, 'mode': args.mode, 'head': args.head,
                   'args': vars(args),
                   'task': args.task,
                   'aggregate': {'per_fold_mean': mean_ba, 'per_fold_std': std_ba,
                                 'pooled': pooled, 'mcs_vs_uws_auc': auc},
                   'per_fold': fold_rows}, f, indent=2, default=str)
    proba_df.to_csv(os.path.join(eval_dir, 'per_subject_proba.csv'), index=False)
    print(f"  saved -> {eval_dir}/{{eval_summary.json,per_subject_proba.csv}}")


if __name__ == '__main__':
    main()

"""Supervised end-to-end training of the GCN encoder on 3-class DOC.

Same encoder architecture as the contrastive pipeline (DenseGCN stack on the
anatomical adjacency). Adds a 256 -> 64 -> 3 classification head and trains
end-to-end with cross-entropy. Augmentations from contrastive variant A are
used as regularization only (not as an invariance objective).

Evaluation: 5-fold subject-level GroupKFold on the train_pool. Per-fold,
predict per-epoch on the held-out subjects, aggregate to per-subject by
mean softmax, then compute 3-class balanced accuracy. Pool predictions
across folds for the global metric.

This is the head-to-head against raw-PCA + GMM K=3 LOOCV bal_acc 0.590.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from preprocessing import EEGtoGraph
from contrastive_adjacency import load_coords, weighted_knn_adjacency
from contrastive_model import Encoder
from augmentations import augment


DX_TO_COARSE = {
    'control': 'control',
    'UWS': 'low_doc', 'COMA': 'low_doc',
    'MCS-': 'high_doc', 'MCS+': 'high_doc', 'EMCS': 'high_doc',
}
COARSE_ORDER = ['control', 'low_doc', 'high_doc']
COARSE_TO_IDX = {c: i for i, c in enumerate(COARSE_ORDER)}


# --------------------------------------------------------------------- dataset
class WsmiSupervised(Dataset):
    """Same preload pattern as WsmiEpochs; also returns the per-epoch coarse label."""
    def __init__(self, data_dir: str, labels_csv: str, allowed_subjects: set):
        sessions = EEGtoGraph.enumerate_matrix_sessions(data_dir)
        sessions = [(sid, snum, src) for sid, snum, src in sessions
                    if src['kind'] == 'npz' and sid in allowed_subjects]
        # Attach diagnoses from labels CSV
        lab = pd.read_csv(labels_csv, dtype=str)
        lab['session_z'] = lab['session'].str.zfill(2)
        lookup = {(r['subject'], r['session_z']): r['diagnostic_crs_final']
                  for _, r in lab.iterrows()}

        chunks, labels, subj = [], [], []
        for sid, snum, src in sessions:
            cohort = src.get('cohort')
            if cohort == 'control':
                dx = 'control'
            else:
                dx = lookup.get((sid, str(snum).zfill(2)), 'unknown')
            if dx not in DX_TO_COARSE:
                continue
            coarse_idx = COARSE_TO_IDX[DX_TO_COARSE[dx]]
            with np.load(src['path']) as d:
                arr = d['data'].astype(np.float32, copy=False)
            chunks.append(arr)
            labels.extend([coarse_idx] * len(arr))
            subj.extend([sid] * len(arr))
        self.data = np.concatenate(chunks, axis=0)
        self.labels = np.array(labels, dtype=np.int64)
        self.subjects = np.array(subj)
        self.x_min = float(self.data.min())
        self.x_max = float(self.data.max())
        self.x_range = self.x_max - self.x_min
        print(f"  WsmiSupervised: {len(self.data):,} epochs, "
              f"{len(set(self.subjects))} subjects, "
              f"mem={self.data.nbytes/1e9:.1f} GB")
        print(f"  per-class counts (epochs): "
              f"{dict(Counter([COARSE_ORDER[c] for c in self.labels]))}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mat = self.data[idx]
        if self.x_range > 0:
            mat = 2.0 * (mat - self.x_min) / self.x_range - 1.0
        return torch.from_numpy(mat.astype(np.float32, copy=False)), int(self.labels[idx])


# ----------------------------------------------------------------------- model
class SupervisedModel(nn.Module):
    def __init__(self, A_norm: torch.Tensor, dropout: float = 0.2,
                 head_hidden: int = 64, n_classes: int = 3):
        super().__init__()
        self.encoder = Encoder(A_norm, in_dim=256,
                               hidden_dims=(128, 64, 32, 16),
                               out_dim=1, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(256, head_hidden), nn.BatchNorm1d(head_hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def embed(self, x):
        h = self.encoder(x)
        return h.reshape(h.shape[0], -1)

    def forward(self, x):
        return self.head(self.embed(x))


# ------------------------------------------------------------------ train loop
def train_fold(model, train_loader, val_loader, class_weights, device,
               epochs, lr, weight_decay, aug_kwargs, log_prefix):
    model.to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * max(1, len(train_loader)))
    best_val = -1.0
    for ep in range(epochs):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = augment(x, **aug_kwargs)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item()
            n_batches += 1
        val_acc = evaluate_epoch_acc(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
        print(f"    {log_prefix} ep {ep+1}/{epochs}  "
              f"train_loss {loss_sum/max(1,n_batches):.4f}  "
              f"val_epoch_acc {val_acc:.3f}  best {best_val:.3f}")
    return best_val


@torch.no_grad()
def evaluate_epoch_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        p = F.softmax(model(x), dim=1).cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(ys, axis=0)


# ----------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--coords_file', default='data_scalp/GSN-HydroCel-257.txt')
    ap.add_argument('--labels_csv',
                    default='data/wsmi_res/256_electrodes-20260526T091011Z-3-001/256_electrodes/wsmi_res_DOC/patient_labels.csv')
    ap.add_argument('--holdout_json', default='data/holdout_subjects.json')
    ap.add_argument('--output_dir', default='output/supervised')
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--k_neighbours', type=int, default=10)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--head_hidden', type=int, default=64)
    ap.add_argument('--noise_std', type=float, default=0.02)
    ap.add_argument('--edge_p', type=float, default=0.05)
    ap.add_argument('--node_p', type=float, default=0.05)
    ap.add_argument('--scale_pct', type=float, default=0.05)
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device: {device}")

    with open(args.holdout_json) as f:
        split = json.load(f)
    train_pool = set(split['train_pool'])
    print(f"  train_pool: {len(train_pool)} subjects")

    ds = WsmiSupervised(args.data_dir, args.labels_csv, train_pool)
    coords = load_coords(args.coords_file)
    A_norm = weighted_knn_adjacency(coords, k=args.k_neighbours).to(device)

    aug_kwargs = dict(noise_std=args.noise_std, edge_p=args.edge_p,
                      node_p=args.node_p, scale_pct=args.scale_pct)

    # Subject-level GroupKFold
    gkf = GroupKFold(n_splits=args.n_folds)
    subjects = ds.subjects
    labels = ds.labels
    folds = list(gkf.split(np.arange(len(ds)), labels, groups=subjects))

    fold_results = []
    all_true_subj, all_pred_subj_soft, all_pred_subj_hard = [], [], []
    all_subj_proba = []   # list of dicts {subject_id, true_dx_coarse, p_control, p_low_doc, p_high_doc}

    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        print(f"\n  === fold {fold_i+1}/{args.n_folds} ===")
        # Train/val split inside training fold by subject (held-out 1 subject per class for val)
        tr_subjects = np.unique(subjects[tr_idx])
        val_subjects = []
        rng = np.random.default_rng(args.random_state + fold_i)
        for c in range(3):
            in_class = [s for s in tr_subjects
                        if labels[(subjects == s)][0] == c]
            if in_class:
                val_subjects.append(rng.choice(in_class))
        val_mask = np.isin(subjects, val_subjects)
        train_mask = ~val_mask & np.isin(np.arange(len(ds)), tr_idx)
        val_mask = val_mask & np.isin(np.arange(len(ds)), tr_idx)

        tr_loader = DataLoader(Subset(ds, np.where(train_mask)[0]),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(Subset(ds, np.where(val_mask)[0]),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        te_loader = DataLoader(Subset(ds, te_idx),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=0, pin_memory=True)

        # Class weights from training epochs only
        cw = compute_class_weight('balanced', classes=np.arange(3),
                                  y=labels[train_mask])
        class_weights = torch.tensor(cw, dtype=torch.float32)
        print(f"    train epochs={int(train_mask.sum())}  "
              f"val epochs={int(val_mask.sum())}  "
              f"test epochs={len(te_idx)}  test subjects={len(np.unique(subjects[te_idx]))}")
        print(f"    class weights: {cw}")

        model = SupervisedModel(A_norm, dropout=args.dropout,
                                head_hidden=args.head_hidden)
        best_val = train_fold(model, tr_loader, val_loader, class_weights,
                              device, args.epochs, args.lr, args.weight_decay,
                              aug_kwargs, log_prefix=f"f{fold_i+1}")
        torch.save({'state_dict': model.state_dict(),
                    'fold': fold_i + 1,
                    'test_subjects': sorted(set(subjects[te_idx].tolist())),
                    'config': vars(args)},
                   os.path.join(args.output_dir, f'model_fold{fold_i+1}.pt'))

        # Per-epoch predictions on test fold
        probs, ys = predict_proba(model, te_loader, device)
        te_subjects = subjects[te_idx]
        # Aggregate to per-subject (mean softmax + majority)
        df = pd.DataFrame({'subject_id': te_subjects, 'y': ys})
        for c in range(3):
            df[f'p{c}'] = probs[:, c]
        soft = df.groupby('subject_id').mean(numeric_only=True)
        pred_soft = soft[[f'p{c}' for c in range(3)]].to_numpy().argmax(axis=1)
        pred_hard = df.assign(pred=probs.argmax(axis=1)) \
                      .groupby('subject_id')['pred'] \
                      .agg(lambda s: Counter(s).most_common(1)[0][0]).to_numpy()
        y_subj = soft['y'].astype(int).to_numpy()
        ba_soft = balanced_accuracy_score(y_subj, pred_soft)
        ba_hard = balanced_accuracy_score(y_subj, pred_hard)
        print(f"    fold {fold_i+1}: ba_soft={ba_soft:.3f}  ba_hard={ba_hard:.3f} "
              f"({len(y_subj)} test subjects)")

        all_true_subj.extend(y_subj.tolist())
        all_pred_subj_soft.extend(pred_soft.tolist())
        all_pred_subj_hard.extend(pred_hard.tolist())

        # Per-subject probability snapshot for downstream ROC AUC
        for sid in soft.index:
            row = {
                'subject_id': sid,
                'true_dx_coarse': COARSE_ORDER[int(soft.loc[sid, 'y'])],
                'p_control': float(soft.loc[sid, 'p0']),
                'p_low_doc': float(soft.loc[sid, 'p1']),
                'p_high_doc': float(soft.loc[sid, 'p2']),
                'fold': fold_i + 1,
            }
            all_subj_proba.append(row)
        fold_results.append({'fold': fold_i + 1, 'ba_soft': float(ba_soft),
                             'ba_hard': float(ba_hard), 'best_val': float(best_val),
                             'n_test_subjects': int(len(y_subj))})

    pooled_soft = balanced_accuracy_score(all_true_subj, all_pred_subj_soft)
    pooled_hard = balanced_accuracy_score(all_true_subj, all_pred_subj_hard)
    cm_soft = confusion_matrix(all_true_subj, all_pred_subj_soft, labels=[0, 1, 2])

    print(f"\n  POOLED across {args.n_folds} folds ({len(all_true_subj)} subjects):")
    print(f"    soft bal_acc 3-class: {pooled_soft:.4f}")
    print(f"    hard bal_acc 3-class: {pooled_hard:.4f}")
    print(f"    random baseline:      0.333")
    print(f"    baseline to beat:     0.590 (raw PCA+GMM K=3 LOOCV)")
    print(f"    confusion (soft, rows=true, cols=pred):\n{cm_soft}")

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump({
            'config': vars(args),
            'fold_results': fold_results,
            'pooled_soft_bal_acc_3': float(pooled_soft),
            'pooled_hard_bal_acc_3': float(pooled_hard),
            'n_pooled_subjects': len(all_true_subj),
            'confusion_soft': cm_soft.tolist(),
        }, f, indent=2)
    pd.DataFrame(all_subj_proba).to_csv(
        os.path.join(args.output_dir, 'per_subject_proba.csv'), index=False)
    print(f"\n  summary -> {args.output_dir}/summary.json")
    print(f"  per_subject_proba -> {args.output_dir}/per_subject_proba.csv")


if __name__ == '__main__':
    main()

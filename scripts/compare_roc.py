#!/usr/bin/env python3
"""
Compare ROC AUC of the 3 best models on the 2 binary tasks, reporting:
  - 5 fold-level point AUCs → mean ± std across folds (between-fold variance)
  - Per fold: bootstrap mean ± std on that fold's subjects (within-fold variance)
  - Pooled ROC curve (concatenating all folds' predictions, point AUC)

All inputs read `per_subject_proba.csv` with columns:
  subject_id, true_dx_coarse, p_control, p_low_doc, p_high_doc, fold
(supervised CSV has these; the GMM 5-fold CSVs add `rep` which we collapse).

Usage:
  python scripts/compare_roc.py \
      --gmm_k3 output/roc_gmm_K3_5fold \
      --gmm_k4 output/roc_gmm_K4_5fold \
      --supervised output/supervised \
      --output_dir output/roc_compare
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score


def task_scores(df, task):
    if task == 'control_vs_rest':
        y = (df['true_dx_coarse'] == 'control').astype(int).to_numpy()
        s = df['p_control'].to_numpy()
    else:
        sub = df[df['true_dx_coarse'].isin(['low_doc', 'high_doc'])]
        y = (sub['true_dx_coarse'] == 'high_doc').astype(int).to_numpy()
        denom = (sub['p_high_doc'] + sub['p_low_doc']).clip(lower=1e-12)
        s = (sub['p_high_doc'] / denom).to_numpy()
    return y, s


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float('nan')


def bootstrap_auc(y, s, n_boot, rng):
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if len(set(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
    if not aucs:
        return float('nan'), float('nan'), float('nan'), float('nan')
    a = np.asarray(aucs)
    return (float(a.mean()), float(a.std(ddof=1)),
            float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def analyse_model(name, mdir, n_boot, random_state):
    """Returns {task: {fold_aucs, fold_mean, fold_std, fold_bootstrap[...]}}"""
    path = os.path.join(mdir, 'per_subject_proba.csv')
    raw = pd.read_csv(path)
    fold_col = 'fold'
    if fold_col not in raw.columns:
        raise ValueError(f"{path} has no `fold` column; can't compute per-fold AUC")
    rng = np.random.default_rng(random_state)
    out = {'n_subjects': int(raw['subject_id'].nunique()),
           'n_folds': int(raw[fold_col].nunique())}

    for task in ('control_vs_rest', 'high_vs_low_doc'):
        fold_records = []
        for fold_i in sorted(raw[fold_col].unique()):
            sub = raw[raw[fold_col] == fold_i]
            # If a subject appears in multiple reps within the same fold, average
            if 'rep' in sub.columns:
                sub = (sub.groupby(['subject_id', 'true_dx_coarse'])
                          [['p_control', 'p_low_doc', 'p_high_doc']]
                          .mean().reset_index())
            y, s = task_scores(sub, task)
            point = safe_auc(y, s)
            bs_mean, bs_std, bs_lo, bs_hi = bootstrap_auc(
                y, s, n_boot=n_boot, rng=rng)
            fold_records.append({'fold': int(fold_i), 'point_auc': point,
                                 'bs_mean': bs_mean, 'bs_std': bs_std,
                                 'bs_ci_lo': bs_lo, 'bs_ci_hi': bs_hi,
                                 'n_pos': int(y.sum()),
                                 'n_neg': int((1 - y).sum())})
        df_f = pd.DataFrame(fold_records)
        point_aucs = df_f['point_auc'].dropna().to_numpy()
        out[task] = {
            'per_fold': fold_records,
            'fold_mean_auc': float(point_aucs.mean()) if len(point_aucs) else float('nan'),
            'fold_std_auc': float(point_aucs.std(ddof=1)) if len(point_aucs) > 1 else 0.0,
            'fold_min_auc': float(point_aucs.min()) if len(point_aucs) else float('nan'),
            'fold_max_auc': float(point_aucs.max()) if len(point_aucs) else float('nan'),
        }

    # 3-class balanced accuracy per fold (mean ± std)
    ba_per_fold = []
    coarse_to_idx = {'control': 0, 'low_doc': 1, 'high_doc': 2}
    for fold_i in sorted(raw[fold_col].unique()):
        sub = raw[raw[fold_col] == fold_i]
        if 'rep' in sub.columns:
            sub = (sub.groupby(['subject_id', 'true_dx_coarse'])
                      [['p_control', 'p_low_doc', 'p_high_doc']]
                      .mean().reset_index())
        probs = sub[['p_control', 'p_low_doc', 'p_high_doc']].to_numpy()
        y_true = sub['true_dx_coarse'].map(coarse_to_idx).to_numpy()
        y_pred = probs.argmax(axis=1)
        ba_per_fold.append({'fold': int(fold_i),
                            'bal_acc_3class': float(balanced_accuracy_score(y_true, y_pred)),
                            'n_subjects': int(len(y_true))})
    df_ba = pd.DataFrame(ba_per_fold)['bal_acc_3class'].to_numpy()
    out['bal_acc_3class'] = {
        'per_fold': ba_per_fold,
        'mean': float(df_ba.mean()),
        'std': float(df_ba.std(ddof=1)) if len(df_ba) > 1 else 0.0,
        'min': float(df_ba.min()),
        'max': float(df_ba.max()),
    }
    return out, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gmm_k3', required=True)
    ap.add_argument('--gmm_k4', required=True)
    ap.add_argument('--supervised', required=True)
    ap.add_argument('--moco', default=None,
                    help='Path to MoCo (contrastive) per_subject_proba.csv dir; optional')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models = [
        ('GMM K=3', args.gmm_k3, '#1f77b4'),
        ('GMM K=4', args.gmm_k4, '#ff7f0e'),
        ('Supervised GCN', args.supervised, '#2ca02c'),
    ]
    if args.moco is not None:
        models.append(('MoCo GCN (wave 1)', args.moco, '#d62728'))

    summary = {}
    raw_by_model = {}
    for name, mdir, _ in models:
        out, raw = analyse_model(name, mdir, args.n_boot, args.random_state)
        summary[name] = out
        raw_by_model[name] = raw

    tasks = [('control_vs_rest', 'Control vs any other'),
             ('high_vs_low_doc', 'MCS (high_doc) vs UWS (low_doc)')]

    # --- print to console -------------------------------------------------
    for name, _, _ in models:
        print(f"\n=== {name} (n={summary[name]['n_subjects']} subjects, "
              f"{summary[name]['n_folds']} folds) ===")
        ba = summary[name]['bal_acc_3class']
        per_fold_ba = ", ".join(f"{r['bal_acc_3class']:.3f}" for r in ba['per_fold'])
        print(f"  3-class bal_acc: mean={ba['mean']:.3f} ± {ba['std']:.3f}  "
              f"[{per_fold_ba}]")
        for task, label in tasks:
            s = summary[name][task]
            print(f"  {label}")
            print(f"    Across {len(s['per_fold'])} folds: "
                  f"mean={s['fold_mean_auc']:.3f}  std={s['fold_std_auc']:.3f}  "
                  f"[min={s['fold_min_auc']:.3f}, max={s['fold_max_auc']:.3f}]")
            for r in s['per_fold']:
                print(f"      fold {r['fold']}: point={r['point_auc']:.3f}  "
                      f"bootstrap={r['bs_mean']:.3f}±{r['bs_std']:.3f}  "
                      f"95% CI [{r['bs_ci_lo']:.3f}, {r['bs_ci_hi']:.3f}]  "
                      f"(n_pos={r['n_pos']}, n_neg={r['n_neg']})")

    # --- plot -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, (task, label) in enumerate(tasks):
        ax = axes[0, col]
        names = [m[0] for m in models]
        colors = [m[2] for m in models]
        x = np.arange(len(names))
        means = [summary[n][task]['fold_mean_auc'] for n in names]
        stds = [summary[n][task]['fold_std_auc'] for n in names]
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.55,
               capsize=8, edgecolor='black', linewidth=0.5,
               label='Mean ± std (across folds)')
        # Overlay each fold's point AUC + bootstrap CI as error bar
        for i, n in enumerate(names):
            recs = summary[n][task]['per_fold']
            xs = np.full(len(recs), x[i])
            jit = (np.random.default_rng(i).random(len(recs)) - 0.5) * 0.18
            points = np.array([r['point_auc'] for r in recs])
            bs_lo = np.array([r['bs_ci_lo'] for r in recs])
            bs_hi = np.array([r['bs_ci_hi'] for r in recs])
            ax.errorbar(xs + jit, points,
                        yerr=[points - bs_lo, bs_hi - points],
                        fmt='o', color='black', alpha=0.6, ms=4,
                        elinewidth=1, capsize=2, zorder=4,
                        label='Per-fold point AUC ± bootstrap 95% CI'
                        if (col == 0 and i == 0) else None)
        ax.axhline(0.5, color='grey', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=10, ha='right', fontsize=9)
        ax.set_ylabel('ROC AUC')
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}\nmean ± std across folds (bars), "
                     f"per-fold point + bootstrap CI (black)")
        ax.grid(axis='y', alpha=0.3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(x[i], m + s + 0.04, f'{m:.3f}±{s:.3f}',
                    ha='center', fontsize=9, fontweight='bold')
        if col == 0:
            ax.legend(loc='lower right', fontsize=8)

    # Pooled ROC (one curve per model, concatenating all folds' predictions)
    for col, (task, label) in enumerate(tasks):
        ax = axes[1, col]
        for name, _, color in models:
            raw = raw_by_model[name]
            if 'rep' in raw.columns:
                raw = (raw.groupby(['subject_id', 'true_dx_coarse'])
                          [['p_control', 'p_low_doc', 'p_high_doc']]
                          .mean().reset_index())
            y, s = task_scores(raw, task)
            fpr, tpr, _ = roc_curve(y, s)
            pooled_auc = safe_auc(y, s)
            ax.plot(fpr, tpr, lw=2, color=color,
                    label=f'{name}  pooled AUC={pooled_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title(f"{label}\npooled ROC across folds")
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Binary discrimination — 3 models × 2 tasks', y=1.00)
    fig.tight_layout()
    out_png = os.path.join(args.output_dir, 'roc_compare.png')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    with open(os.path.join(args.output_dir, 'roc_compare.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  plot -> {out_png}")
    print(f"  json -> {os.path.join(args.output_dir, 'roc_compare.json')}")


if __name__ == '__main__':
    main()

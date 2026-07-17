"""
Leave-One-Subject-Out diagnosis decoder on (V)GAE-derived features.

Mirrors the LOSO logistic-regression decoder in clustering-wsmi/scripts/
run_raw_state_experiments.py:264-317, with one addition specific to this
project: a "latent_means" feature set that uses each subject's mean (V)GAE
latent vector across their graphs.

Inputs:
- per_recording_df from state_dynamics.per_recording_dynamics, containing
  `subject_id`, `diagnosis_group`, occupancy_p_*, transition_*_*,
  occupancy_entropy_bits, entropy_rate_bits, weighted_entropy.
- (optional) bundle from cluster_analysis.extract_latents_from_graphs.

Outputs:
- decoder_metrics_df with one row per feature_set: macro_auc_ovr, accuracy,
  macro_f1, n_subjects.
- predictions_df with per-subject predictions and class probabilities.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


def _safe_auc(y_true: np.ndarray, prob_matrix: np.ndarray, class_order: List[str]) -> float:
    if len(class_order) < 2:
        return float("nan")
    if len(class_order) == 2:
        positive = class_order[1]
        idx = class_order.index(positive)
        return float(roc_auc_score((y_true == positive).astype(int), prob_matrix[:, idx]))
    y_bin = label_binarize(y_true, classes=class_order)
    present = y_bin.sum(axis=0) > 0
    if present.sum() < 2:
        return float("nan")
    return float(roc_auc_score(
        y_bin[:, present], prob_matrix[:, present], average="macro", multi_class="ovr",
    ))


def aggregate_latents_per_subject(
    embeds: np.ndarray, subject_ids: Sequence[str], diagnosis_groups: Sequence[str],
) -> pd.DataFrame:
    """Mean latent vector per subject; returns columns latent_0..latent_{D-1}
    plus subject_id and diagnosis_group.
    """
    bag: Dict[str, List[int]] = defaultdict(list)
    diag: Dict[str, str] = {}
    for i, (s, dg) in enumerate(zip(subject_ids, diagnosis_groups)):
        bag[s].append(i)
        diag[s] = dg
    rows = []
    for s, idxs in bag.items():
        mean = embeds[idxs].mean(axis=0)
        row = {"subject_id": s, "diagnosis_group": diag[s]}
        for d, v in enumerate(mean):
            row[f"latent_{d}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def build_subject_feature_table(
    per_recording_df: pd.DataFrame,
    latent_per_subject_df: Optional[pd.DataFrame] = None,
    label_col: str = "diagnosis_group",
) -> pd.DataFrame:
    """Average numeric per-recording features to subject level, optionally
    merging per-subject latent means.

    Output keeps `subject_id`, `<label_col>`, plus numeric feature columns.
    """
    if per_recording_df.empty:
        return per_recording_df

    num_cols = per_recording_df.select_dtypes(include="number").columns.tolist()
    subj = (per_recording_df
            .groupby(["subject_id", label_col], as_index=False)[num_cols]
            .mean())

    if latent_per_subject_df is not None and not latent_per_subject_df.empty:
        merge_cols = ["subject_id"]
        if label_col in latent_per_subject_df.columns:
            merge_cols.append(label_col)
        subj = subj.merge(latent_per_subject_df, on=merge_cols, how="inner")
    return subj


def _feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    state_prob_cols = [c for c in df.columns if c.startswith("occupancy_p_")]
    trans_cols = [c for c in df.columns if c.startswith("transition_")]
    latent_cols = [c for c in df.columns if c.startswith("latent_")]
    entropy_cols = [c for c in ("weighted_entropy", "occupancy_entropy_bits", "entropy_rate_bits") if c in df.columns]

    feature_sets: Dict[str, List[str]] = {}
    if state_prob_cols:
        feature_sets["state_probabilities"] = state_prob_cols
    if trans_cols:
        feature_sets["transitions"] = trans_cols
    if latent_cols:
        feature_sets["latent_means"] = latent_cols
    combined = state_prob_cols + trans_cols + entropy_cols
    if combined:
        feature_sets["state_combined"] = combined
    all_cols = combined + latent_cols
    if latent_cols and (state_prob_cols or trans_cols):
        feature_sets["latent_plus_state"] = all_cols
    return feature_sets


def loso_decoder(
    subject_features: pd.DataFrame,
    label_col: str = "diagnosis_group",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run LOSO logistic regression for each available feature set.

    Returns (metrics_df, predictions_df).
    """
    feature_sets = _feature_columns(subject_features)
    if not feature_sets:
        raise ValueError("No usable feature columns found in subject_features")

    labels = subject_features[label_col].to_numpy()
    groups = subject_features["subject_id"].to_numpy()
    class_order = sorted(set(str(x) for x in labels))

    logo = LeaveOneGroupOut()
    metric_rows = []
    all_preds = []

    for fname, cols in feature_sets.items():
        pred_rows = []
        for train_idx, test_idx in logo.split(subject_features, labels, groups):
            train = subject_features.iloc[train_idx]
            test = subject_features.iloc[test_idx]
            X_train = train[cols].to_numpy(dtype=float)
            X_test = test[cols].to_numpy(dtype=float)
            y_train = train[label_col].to_numpy()

            # Skip degenerate folds where the train set has fewer than 2 classes.
            if len(set(y_train)) < 2:
                continue

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=5000, class_weight="balanced",
                    solver="lbfgs", random_state=random_state,
                ),
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)
            preds = model.predict(X_test)
            for i, idx in enumerate(test.index):
                row = {
                    "feature_set": fname,
                    "subject_id": subject_features.loc[idx, "subject_id"],
                    "y_true": subject_features.loc[idx, label_col],
                    "y_pred": preds[i],
                }
                for cls, p in zip(model.classes_, probs[i]):
                    row[f"prob_{cls}"] = float(p)
                pred_rows.append(row)

        if not pred_rows:
            metric_rows.append({
                "feature_set": fname, "macro_auc_ovr": float("nan"),
                "accuracy": float("nan"), "macro_f1": float("nan"),
                "n_subjects": 0, "n_features": len(cols),
            })
            continue

        pred_df = pd.DataFrame(pred_rows)
        prob_matrix = np.column_stack([
            pred_df.get(f"prob_{cls}", pd.Series(np.zeros(len(pred_df)))).fillna(0.0).to_numpy(dtype=float)
            for cls in class_order
        ])
        auc = _safe_auc(pred_df["y_true"].to_numpy(), prob_matrix, class_order)
        metric_rows.append({
            "feature_set": fname,
            "macro_auc_ovr": float(auc),
            "accuracy": float(accuracy_score(pred_df["y_true"], pred_df["y_pred"])),
            "macro_f1": float(f1_score(pred_df["y_true"], pred_df["y_pred"], average="macro")),
            "n_subjects": int(pred_df["subject_id"].nunique()),
            "n_features": len(cols),
        })
        all_preds.append(pred_df)

    metrics_df = pd.DataFrame(metric_rows).sort_values("macro_auc_ovr", ascending=False).reset_index(drop=True)
    predictions_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return metrics_df, predictions_df


def plot_decoder_performance(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    out_dir: str,
    title_prefix: str = "",
) -> List[str]:
    """Render LOSO-decoder performance figures next to the TSV tables.

    Writes:
      - decoder_metrics_bars.png : grouped bar chart of macro_auc_ovr / accuracy /
        macro_f1 per feature set (chance AUC=0.5 line), feature sets ordered by AUC.
      - decoder_confusion.png    : one normalized confusion matrix per feature set
        (rows = true, normalized to sum 1), shared class order.

    Returns the list of files written. Safe to call with empty inputs.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    if metrics_df is None or len(metrics_df) == 0:
        return written

    pre = f"{title_prefix}: " if title_prefix else ""

    # ---- (1) grouped metric bars ----
    md = metrics_df.sort_values("macro_auc_ovr", ascending=False)
    fsets = md["feature_set"].tolist()
    metrics = [("macro_auc_ovr", "macro AUC (OvR)"),
               ("accuracy", "accuracy"),
               ("macro_f1", "macro F1")]
    x = np.arange(len(fsets))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(fsets) + 3), 5))
    for i, (col, lab) in enumerate(metrics):
        vals = md[col].to_numpy(dtype=float) if col in md else np.full(len(fsets), np.nan)
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, ls="--", color="grey", lw=1, label="AUC chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(fsets, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"{pre}LOSO decoder performance by feature set")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    p = os.path.join(out_dir, "decoder_metrics_bars.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    written.append(p)

    # ---- (2) per-feature-set confusion matrices ----
    if predictions_df is not None and len(predictions_df) > 0:
        classes = sorted(set(predictions_df["y_true"].astype(str)) |
                         set(predictions_df["y_pred"].astype(str)))
        ci = {c: i for i, c in enumerate(classes)}
        fs_present = [f for f in fsets if f in set(predictions_df["feature_set"])]
        n = len(fs_present)
        if n:
            ncol = min(3, n)
            nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow),
                                     squeeze=False)
            axes = axes.flatten()
            for ax, fs in zip(axes, fs_present):
                sub = predictions_df[predictions_df["feature_set"] == fs]
                cm = np.zeros((len(classes), len(classes)), dtype=float)
                for yt, yp in zip(sub["y_true"].astype(str), sub["y_pred"].astype(str)):
                    cm[ci[yt], ci[yp]] += 1
                row = cm.sum(axis=1, keepdims=True)
                cmn = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
                im = ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
                ax.set_title(fs, fontsize=9)
                ax.set_xticks(range(len(classes)))
                ax.set_yticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(classes, fontsize=7)
                ax.set_xlabel("predicted", fontsize=8)
                ax.set_ylabel("true", fontsize=8)
                for r in range(len(classes)):
                    for c in range(len(classes)):
                        ax.text(c, r, f"{cmn[r, c]:.2f}\n({int(cm[r, c])})",
                                ha="center", va="center", fontsize=7,
                                color="white" if cmn[r, c] > 0.5 else "black")
                fig.colorbar(im, ax=ax, fraction=0.046)
            for ax in axes[n:]:
                ax.axis("off")
            plt.suptitle(f"{pre}LOSO confusion matrices (row-normalized)", y=1.02)
            plt.tight_layout()
            p = os.path.join(out_dir, "decoder_confusion.png")
            plt.savefig(p, dpi=200, bbox_inches="tight")
            plt.close()
            written.append(p)
    return written

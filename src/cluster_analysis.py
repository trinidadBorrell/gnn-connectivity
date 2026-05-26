"""
CLUSTER ANALYSIS (wSMI pipeline)
================================
Cluster graph-level latents from the trained GAE/VGAE and produce the three
deliverables requested:

  (a) Per-cluster diagnosis prevalence (stacked-bar + CSV, graph-level and
      subject-level).
  (b) Per-cluster mean of the underlying raw wSMI 64x64 matrices (heatmap + .npy).
  (c) Cluster occupancy entropy: Shannon entropy over cluster proportions,
      variance-weighted entropy, and a per-subject Shannon entropy
      (how scattered a subject's epochs are across clusters), with
      diagnosis-group box plot.

Clusterers: K-Means (silhouette-best k in 2..K_MAX), GMM (BIC-best k),
HDBSCAN (validity-index across min_cluster_size choices). All three are
applied to the SAME graph-level latents.

Inputs are produced by `extract_latents_from_graphs` below — call that on a
fitted model (GAE or VGAE) plus the full graph list with raw_matrix /
diagnosis metadata attached (see wsmi_loader.load_wsmi_dataset).
"""
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


try:
    import hdbscan as _hdbscan
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False


DIAG_GROUP_ORDER = ["CONTROL", "UWS", "MCS"]
DIAG_GROUP_COLORS = {
    "CONTROL": "#2ca02c",
    "UWS": "#d62728",
    "MCS": "#ff7f0e",
}


@dataclass
class LatentBundle:
    """Container for latents + parallel metadata used by all clusterers."""
    embeds: np.ndarray           # (N, D)
    subject_ids: List[str]
    sessions: List[str]
    epochs: List[int]
    diagnoses: List[str]
    diagnosis_groups: List[str]
    raw_matrices: List[np.ndarray]  # parallel list of 64x64 wSMI matrices
    splits: List[str]            # 'train' | 'val' | 'test' per row

    def __len__(self):
        return self.embeds.shape[0]


def extract_latents_from_graphs(
    model,
    graphs: Sequence,
    splits: Sequence[str],
    device: Optional[torch.device] = None,
    aggregate: str = "mean",
) -> LatentBundle:
    """
    Run encoder on every graph and aggregate node latents to one vector per graph.

    `splits` is a parallel iterable labelling each graph as 'train'/'val'/'test'.
    For VGAE, uses mu deterministically.
    """
    if len(graphs) != len(splits):
        raise ValueError("graphs and splits must have equal length")
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    embeds = []
    subject_ids, sessions, epochs = [], [], []
    diagnoses, diagnosis_groups = [], []
    raw_matrices = []
    splits_out = []
    with torch.no_grad():
        for g, split in zip(graphs, splits):
            x = g.x.to(device)
            edge_index = g.edge_index.to(device)
            enc_out = model.encode(x, edge_index)
            z = enc_out[0] if isinstance(enc_out, tuple) else enc_out  # VGAE: mu
            z_np = z.detach().cpu().numpy()
            if aggregate == "mean":
                vec = z_np.mean(axis=0)
            elif aggregate == "flatten":
                vec = z_np.reshape(-1)
            else:
                raise ValueError(f"Unknown aggregate: {aggregate}")
            embeds.append(vec)
            subject_ids.append(str(getattr(g, "subject_id", "unknown")))
            sessions.append(str(getattr(g, "session_num", "unknown")))
            epochs.append(int(getattr(g, "matrix_idx", -1)))
            diagnoses.append(str(getattr(g, "diagnosis", "UNK")))
            diagnosis_groups.append(str(getattr(g, "diagnosis_group", "UNK")))
            raw_matrices.append(np.asarray(getattr(g, "raw_matrix", None)))
            splits_out.append(split)
    return LatentBundle(
        embeds=np.stack(embeds, axis=0).astype(np.float32),
        subject_ids=subject_ids,
        sessions=sessions,
        epochs=epochs,
        diagnoses=diagnoses,
        diagnosis_groups=diagnosis_groups,
        raw_matrices=raw_matrices,
        splits=splits_out,
    )


def save_latent_bundle(bundle: LatentBundle, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "graph_latents.npy"), bundle.embeds)
    meta = [
        {
            "subject_id": s, "session": ses, "epoch": ep,
            "diagnosis": d, "diagnosis_group": dg, "split": sp,
        }
        for s, ses, ep, d, dg, sp in zip(
            bundle.subject_ids, bundle.sessions, bundle.epochs,
            bundle.diagnoses, bundle.diagnosis_groups, bundle.splits,
        )
    ]
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# --------- clusterers (return labels, n_clusters, metric_curve dict) ---------


def kmeans_best_k(X: np.ndarray, k_range=range(2, 11), random_state: int = 42):
    sils, fitted = [], {}
    for k in k_range:
        if k >= X.shape[0]:
            sils.append(-1.0)
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            sils.append(-1.0)
            continue
        sils.append(float(silhouette_score(X, labels)))
        fitted[k] = (km, labels)
    best_k = list(k_range)[int(np.argmax(sils))]
    km, labels = fitted[best_k]
    return labels, best_k, {"k": list(k_range), "silhouette": sils, "best_k": best_k}


def gmm_best_k(X: np.ndarray, k_range=range(2, 11), random_state: int = 42):
    bics, fitted = [], {}
    for k in k_range:
        if k >= X.shape[0]:
            bics.append(np.inf)
            continue
        try:
            gmm = GaussianMixture(
                n_components=k, covariance_type="full",
                random_state=random_state, n_init=3, max_iter=200,
            )
            gmm.fit(X)
            bics.append(float(gmm.bic(X)))
            fitted[k] = (gmm, gmm.predict(X))
        except Exception:
            bics.append(np.inf)
    best_k = list(k_range)[int(np.argmin(bics))]
    gmm, labels = fitted[best_k]
    return labels, best_k, {"k": list(k_range), "bic": bics, "best_k": best_k}


def hdbscan_best(X: np.ndarray, min_sizes=(5, 10, 20, 50, 100)):
    """Run HDBSCAN across `min_sizes` (with a lowered min_samples for each)
    and return the best result by validity index.

    Falls back gracefully if no setting yields >=2 clusters: returns the one
    with the most non-noise points; if all settings produce pure noise, marks
    every point as cluster 0 so downstream consumers still work.
    """
    if not _HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan not installed. pip install hdbscan")
    Xs = StandardScaler().fit_transform(X) if X.shape[1] > 1 else X.astype(np.float64)

    fitted = {}     # ms -> (labels, validity_score, n_clusters, n_noise)
    for ms in min_sizes:
        if ms >= X.shape[0]:
            continue
        # min_samples controls how conservative the density estimate is.
        # Lower than min_cluster_size encourages splitting dense regions.
        ms_samples = max(2, ms // 2)
        try:
            clusterer = _hdbscan.HDBSCAN(
                min_cluster_size=ms, min_samples=ms_samples,
                prediction_data=False,
            )
            labels = clusterer.fit_predict(Xs.astype(np.float64))
        except Exception as e:
            print(f"  [hdbscan ms={ms}] failed: {e}")
            continue
        non_noise = labels[labels >= 0]
        n_clusters = int(len(np.unique(non_noise)))
        n_noise = int((labels == -1).sum())
        validity = -np.inf
        if n_clusters >= 2:
            try:
                validity = float(_hdbscan.validity.validity_index(
                    Xs.astype(np.float64), labels))
            except Exception as e:
                print(f"  [hdbscan ms={ms}] validity_index failed: {e}")
        fitted[ms] = (labels, validity, n_clusters, n_noise)
        print(f"  [hdbscan ms={ms}] n_clusters={n_clusters} "
              f"n_noise={n_noise}/{len(labels)} validity={validity:.4f}")

    if not fitted:
        # Every config crashed outright -> single pseudo-cluster.
        print("  [hdbscan] all settings failed; treating all points as one cluster")
        labels = np.zeros(X.shape[0], dtype=int)
        return labels, int(min_sizes[0]), {
            "min_cluster_size": list(min_sizes), "validity": [-np.inf] * len(min_sizes),
            "best_min_cluster_size": int(min_sizes[0]),
            "note": "all settings failed; single pseudo-cluster",
        }

    # Pick best run.
    multi = {ms: v for ms, v in fitted.items() if v[2] >= 2}
    if multi:
        best_ms = max(multi.keys(), key=lambda m: multi[m][1])
        note = "best validity among >=2-cluster runs"
    else:
        best_ms = max(fitted.keys(), key=lambda m: -fitted[m][3])  # least noise
        note = "no setting produced >=2 clusters; chose run with least noise"
        print(f"  [hdbscan] {note}")

    labels = fitted[best_ms][0]
    return labels, int(best_ms), {
        "min_cluster_size": list(min_sizes),
        "validity": [fitted.get(m, (None, -np.inf, 0, 0))[1] for m in min_sizes],
        "n_clusters_per_setting": [fitted.get(m, (None, -np.inf, 0, 0))[2] for m in min_sizes],
        "n_noise_per_setting": [fitted.get(m, (None, -np.inf, 0, 0))[3] for m in min_sizes],
        "best_min_cluster_size": int(best_ms),
        "note": note,
    }


# --------- analysis: prevalence, mean matrices, entropy ---------


def per_cluster_diagnosis_counts(
    labels: np.ndarray, diagnosis_groups: Sequence[str],
) -> pd.DataFrame:
    """Wide DataFrame: cluster x diagnosis_group counts (rows include noise=-1)."""
    rows = defaultdict(Counter)
    for c, dg in zip(labels, diagnosis_groups):
        rows[int(c)][dg] += 1
    clusters = sorted(rows.keys())
    cols = [g for g in DIAG_GROUP_ORDER if any(g in r for r in rows.values())]
    df = pd.DataFrame(
        [[rows[c].get(g, 0) for g in cols] for c in clusters],
        index=[f"cluster_{c}" for c in clusters],
        columns=cols,
    )
    df.index.name = "cluster"
    return df


def per_subject_modal_cluster(
    labels: np.ndarray, subject_ids: Sequence[str], diagnosis_groups: Sequence[str],
) -> pd.DataFrame:
    """For each subject, the modal cluster across their epochs."""
    bag: Dict[str, Counter] = defaultdict(Counter)
    sg: Dict[str, str] = {}
    for s, c, dg in zip(subject_ids, labels, diagnosis_groups):
        bag[s][int(c)] += 1
        sg[s] = dg
    rows = []
    for s, ctr in bag.items():
        modal, modal_n = ctr.most_common(1)[0]
        rows.append({
            "subject_id": s, "diagnosis_group": sg[s],
            "modal_cluster": modal, "modal_count": modal_n,
            "n_epochs": sum(ctr.values()),
        })
    return pd.DataFrame(rows).sort_values(["diagnosis_group", "subject_id"]).reset_index(drop=True)


def plot_prevalence(df: pd.DataFrame, title: str, out_path: str, normalize: bool = False):
    if normalize:
        plot_df = df.div(df.sum(axis=1).replace(0, 1), axis=0)
        ylabel = "Proportion"
    else:
        plot_df = df
        ylabel = "Count"
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(plot_df)), 5))
    bottom = np.zeros(len(plot_df))
    for g in plot_df.columns:
        ax.bar(plot_df.index, plot_df[g].values, bottom=bottom,
               label=g, color=DIAG_GROUP_COLORS.get(g, "#999999"))
        bottom += plot_df[g].values
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Cluster")
    ax.legend(loc="upper right", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def cluster_mean_matrices(
    labels: np.ndarray, raw_matrices: Sequence[np.ndarray],
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for c in sorted(set(int(l) for l in labels)):
        idxs = np.where(labels == c)[0]
        mats = [raw_matrices[i] for i in idxs if raw_matrices[i] is not None]
        if not mats:
            continue
        out[c] = np.mean(np.stack(mats, axis=0), axis=0)
    return out


def save_mean_matrices(
    means: Dict[int, np.ndarray], electrode_labels: Optional[Sequence[str]],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if not means:
        return
    for c, m in means.items():
        np.save(os.path.join(out_dir, f"mean_matrix_cluster_{c}.npy"), m)

    # vmin/vmax fixed across panels for comparability
    vmax = max(np.abs(m).max() for m in means.values())
    vmin = -vmax
    n = len(means)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (c, m) in zip(axes, sorted(means.items())):
        im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_title(f"Cluster {c}  (n={m.shape[0]}x{m.shape[1]})", fontsize=10)
        if electrode_labels is not None and len(electrode_labels) <= 80:
            ax.set_xticks(range(len(electrode_labels)))
            ax.set_yticks(range(len(electrode_labels)))
            ax.set_xticklabels(electrode_labels, rotation=90, fontsize=4)
            ax.set_yticklabels(electrode_labels, fontsize=4)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes[len(means):]:
        ax.axis("off")
    plt.suptitle("Per-cluster mean wSMI matrix", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_matrices_grid.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def entropy_metrics(
    labels: np.ndarray, raw_matrices: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, float]:
    eps = 1e-12
    unique = sorted(set(int(l) for l in labels))
    counts = np.array([(labels == c).sum() for c in unique], dtype=float)
    p = counts / max(1.0, counts.sum())
    H = float(-np.sum(p * np.log(p + eps)))
    out = {
        "shannon": H,
        "cluster_ids": unique,
        "cluster_proportions": p.tolist(),
        "cluster_counts": counts.tolist(),
    }
    if raw_matrices is not None and len(raw_matrices) == len(labels):
        per_cluster_var = []
        for c in unique:
            idxs = np.where(labels == c)[0]
            mats = [raw_matrices[i] for i in idxs if raw_matrices[i] is not None]
            if len(mats) > 1:
                per_cluster_var.append(float(np.var(np.stack(mats, 0), axis=0).mean()))
            else:
                per_cluster_var.append(0.0)
        per_cluster_var = np.array(per_cluster_var)
        out["per_cluster_variance"] = per_cluster_var.tolist()
        out["variance_weighted_entropy"] = float(np.sum(p * per_cluster_var))
    return out


# --------- diagnosis-aware clustering quality metrics ---------


def diagnosis_clustering_metrics(
    labels: np.ndarray, diagnosis_groups: Sequence[str],
) -> Dict[str, float]:
    """Compare cluster assignments against ground-truth diagnosis labels.

    Returns ARI, AMI, homogeneity, completeness, V-measure, plus cluster
    purity (mean over clusters of the fraction held by the modal class).

    Rows with a non-informative diagnosis ('UNK') are dropped before scoring,
    because including them deflates ARI/AMI without telling us anything useful.
    """
    labels = np.asarray(labels)
    diagnoses = np.asarray(diagnosis_groups)
    keep = (diagnoses != "UNK") & (labels != -1)
    labels_k = labels[keep]
    diag_k = diagnoses[keep]

    if labels_k.size == 0 or len(set(diag_k)) < 2 or len(set(labels_k)) < 2:
        return {
            "ari": float("nan"), "ami": float("nan"),
            "homogeneity": float("nan"), "completeness": float("nan"),
            "v_measure": float("nan"), "purity": float("nan"),
            "n_scored": int(labels_k.size), "n_dropped_unk_or_noise": int((~keep).sum()),
        }

    ari = float(adjusted_rand_score(diag_k, labels_k))
    ami = float(adjusted_mutual_info_score(diag_k, labels_k))
    homo, comp, vmes = homogeneity_completeness_v_measure(diag_k, labels_k)

    # Purity: for each cluster, fraction of the modal diagnosis; averaged.
    purities = []
    for c in np.unique(labels_k):
        idx = labels_k == c
        if idx.sum() == 0:
            continue
        ctr = Counter(diag_k[idx])
        modal_n = max(ctr.values())
        purities.append(modal_n / idx.sum())
    purity = float(np.mean(purities)) if purities else float("nan")

    return {
        "ari": ari, "ami": ami,
        "homogeneity": float(homo), "completeness": float(comp), "v_measure": float(vmes),
        "purity": purity,
        "n_scored": int(labels_k.size),
        "n_dropped_unk_or_noise": int((~keep).sum()),
    }


def per_subject_entropy(
    labels: np.ndarray, subject_ids: Sequence[str], diagnosis_groups: Sequence[str],
) -> pd.DataFrame:
    """Per-subject Shannon entropy of their epoch-level cluster assignments."""
    eps = 1e-12
    bag: Dict[str, Counter] = defaultdict(Counter)
    sg: Dict[str, str] = {}
    for s, c, dg in zip(subject_ids, labels, diagnosis_groups):
        bag[s][int(c)] += 1
        sg[s] = dg
    rows = []
    for s, ctr in bag.items():
        n = sum(ctr.values())
        p = np.array(list(ctr.values()), dtype=float) / n
        H = float(-np.sum(p * np.log(p + eps)))
        rows.append({
            "subject_id": s, "diagnosis_group": sg[s],
            "n_epochs": int(n), "n_clusters_used": int(len(ctr)),
            "shannon_entropy": H,
        })
    return pd.DataFrame(rows).sort_values(["diagnosis_group", "subject_id"]).reset_index(drop=True)


def plot_per_subject_entropy(df: pd.DataFrame, out_path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [g for g in DIAG_GROUP_ORDER if g in df["diagnosis_group"].unique()]
    data = [df[df["diagnosis_group"] == g]["shannon_entropy"].values for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True, showmeans=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(DIAG_GROUP_COLORS.get(g, "#cccccc"))
        patch.set_alpha(0.6)
    # overlay raw points
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.random.normal(i, 0.05, size=len(vals)), vals,
                   s=10, alpha=0.6, color="black")
    ax.set_ylabel("Shannon entropy of cluster occupancy")
    ax.set_xlabel("Diagnosis group")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# --------- top-level driver ---------


def save_metric_curve(curve: dict, out_path: str, kind: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    if kind == "silhouette":
        ax.plot(curve["k"], curve["silhouette"], marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette score")
        ax.axvline(curve["best_k"], linestyle="--", color="red",
                   label=f"best k={curve['best_k']}")
    elif kind == "bic":
        ax.plot(curve["k"], curve["bic"], marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("BIC (lower is better)")
        ax.axvline(curve["best_k"], linestyle="--", color="red",
                   label=f"best k={curve['best_k']}")
    elif kind == "hdbscan":
        ax.plot(curve["min_cluster_size"], curve["validity"], marker="o")
        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel("validity index")
        ax.axvline(curve["best_min_cluster_size"], linestyle="--",
                   color="red",
                   label=f"best mcs={curve['best_min_cluster_size']}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def run_one_clusterer(
    name: str, labels: np.ndarray, bundle: LatentBundle, out_dir: str,
    electrode_labels: Optional[Sequence[str]], metric_curve: dict,
):
    os.makedirs(out_dir, exist_ok=True)

    # Cluster counts / assignments
    assignments = pd.DataFrame({
        "subject_id": bundle.subject_ids,
        "session": bundle.sessions,
        "epoch": bundle.epochs,
        "diagnosis": bundle.diagnoses,
        "diagnosis_group": bundle.diagnosis_groups,
        "split": bundle.splits,
        "cluster": labels.astype(int),
    })
    assignments.to_csv(os.path.join(out_dir, "assignments.csv"), index=False)

    # (a) prevalence
    counts = per_cluster_diagnosis_counts(labels, bundle.diagnosis_groups)
    counts.to_csv(os.path.join(out_dir, "prevalence_counts.csv"))
    plot_prevalence(counts, f"{name}: diagnosis_group per cluster (counts)",
                    os.path.join(out_dir, "prevalence_counts.png"), normalize=False)
    plot_prevalence(counts, f"{name}: diagnosis_group per cluster (proportions)",
                    os.path.join(out_dir, "prevalence_proportions.png"), normalize=True)

    subj_modal = per_subject_modal_cluster(
        labels, bundle.subject_ids, bundle.diagnosis_groups)
    subj_modal.to_csv(os.path.join(out_dir, "subject_modal_cluster.csv"), index=False)

    # (b) per-cluster mean matrices
    means = cluster_mean_matrices(labels, bundle.raw_matrices)
    save_mean_matrices(means, electrode_labels,
                       os.path.join(out_dir, "mean_matrices"))

    # (c) entropy
    H = entropy_metrics(labels, bundle.raw_matrices)
    with open(os.path.join(out_dir, "entropy.json"), "w") as f:
        json.dump(H, f, indent=2, default=float)
    subj_H = per_subject_entropy(labels, bundle.subject_ids, bundle.diagnosis_groups)
    subj_H.to_csv(os.path.join(out_dir, "per_subject_entropy.csv"), index=False)
    plot_per_subject_entropy(
        subj_H, os.path.join(out_dir, "per_subject_entropy_box.png"),
        title=f"{name}: per-subject cluster-occupancy entropy",
    )

    # metric curve
    curve_kind = {"kmeans": "silhouette", "gmm": "bic", "hdbscan": "hdbscan"}[name]
    save_metric_curve(metric_curve, os.path.join(out_dir, "metric_curve.png"),
                      kind=curve_kind)
    with open(os.path.join(out_dir, "metric_curve.json"), "w") as f:
        json.dump(metric_curve, f, indent=2, default=float)

    # latent scatter (PCA-2)
    try:
        from sklearn.decomposition import PCA
        X = bundle.embeds
        if X.shape[1] > 2:
            X2 = PCA(n_components=2, random_state=0).fit_transform(X)
        else:
            X2 = np.column_stack([X, np.zeros(len(X))]) if X.shape[1] == 1 else X
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
        ax.set_title(f"{name}: latent (PCA-2) coloured by cluster")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax, label="cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "latent_scatter.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [warn] scatter failed: {e}")


def run_all_clusterers(
    bundle: LatentBundle, out_dir: str,
    electrode_labels: Optional[Sequence[str]] = None,
    k_range=range(2, 11),
    hdbscan_min_sizes=(10, 20, 50, 100),
    random_state: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    save_latent_bundle(bundle, out_dir)
    X = bundle.embeds

    print(f"\n[K-Means] sweeping k in {list(k_range)} on X={X.shape}")
    labels, best_k, curve = kmeans_best_k(X, k_range, random_state=random_state)
    print(f"  best k={best_k}, silhouette={max(curve['silhouette']):.4f}")
    run_one_clusterer("kmeans", labels, bundle,
                      os.path.join(out_dir, "kmeans"), electrode_labels, curve)

    print(f"\n[GMM] sweeping k in {list(k_range)}")
    labels, best_k, curve = gmm_best_k(X, k_range, random_state=random_state)
    print(f"  best k={best_k}, BIC={min(curve['bic']):.2f}")
    run_one_clusterer("gmm", labels, bundle,
                      os.path.join(out_dir, "gmm"), electrode_labels, curve)

    if _HDBSCAN_AVAILABLE:
        print(f"\n[HDBSCAN] sweeping min_cluster_size in {list(hdbscan_min_sizes)}")
        labels, best_ms, curve = hdbscan_best(X, hdbscan_min_sizes)
        n_noise = int((labels == -1).sum())
        print(f"  best min_cluster_size={best_ms}, noise={n_noise}/{len(labels)}")
        run_one_clusterer("hdbscan", labels, bundle,
                          os.path.join(out_dir, "hdbscan"),
                          electrode_labels, curve)
    else:
        print("\n[HDBSCAN] skipped (pip install hdbscan to enable)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True,
                        help="path to graph_latents.npy")
    parser.add_argument("--meta", required=True, help="path to meta.json")
    parser.add_argument("--matrices", required=True,
                        help="path to raw_matrices.npy (parallel to latents)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    X = np.load(args.latents)
    with open(args.meta) as f:
        meta = json.load(f)
    raw = np.load(args.matrices, allow_pickle=True)
    bundle = LatentBundle(
        embeds=X,
        subject_ids=[m["subject_id"] for m in meta],
        sessions=[m["session"] for m in meta],
        epochs=[m["epoch"] for m in meta],
        diagnoses=[m["diagnosis"] for m in meta],
        diagnosis_groups=[m["diagnosis_group"] for m in meta],
        raw_matrices=list(raw),
        splits=[m["split"] for m in meta],
    )
    run_all_clusterers(bundle, args.out_dir)

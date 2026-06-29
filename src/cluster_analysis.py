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

# Preferred clinical ordering (coarse + fine labels), most-impaired -> recovered.
PREFERRED_GROUP_ORDER = ["CONTROL", "COMA", "UWS", "MCS-", "MCS", "MCS+", "EMCS"]
# Fixed hues for known labels; unknown labels get a tab10 colour at resolve time.
KNOWN_GROUP_COLORS = {
    "CONTROL": "#2ca02c",  # green
    "COMA": "#7f7f7f",     # grey
    "UWS": "#d62728",      # red
    "MCS-": "#ff9896",     # light orange/red
    "MCS": "#ff7f0e",      # orange
    "MCS+": "#c49c94",     # brown-orange
    "EMCS": "#9467bd",     # purple
}


def resolve_group_order(groups_present: Sequence[str]) -> List[str]:
    """Order the diagnosis groups present: known clinical order first, then any
    extras alphabetically. 'UNK' is dropped (not plotted)."""
    present = list(dict.fromkeys(str(g) for g in groups_present))
    ordered = [g for g in PREFERRED_GROUP_ORDER if g in present]
    extras = sorted(g for g in present if g not in PREFERRED_GROUP_ORDER and g != "UNK")
    return ordered + extras


def resolve_group_colors(order: Sequence[str]) -> Dict[str, str]:
    """Map each group to a colour: fixed hues for known labels, tab10 for the rest."""
    import matplotlib.cm as cm
    colors: Dict[str, str] = {}
    tab = cm.get_cmap("tab10")
    unknown_i = 0
    for g in order:
        if g in KNOWN_GROUP_COLORS:
            colors[g] = KNOWN_GROUP_COLORS[g]
        else:
            colors[g] = tab(unknown_i % 10)
            unknown_i += 1
    return colors


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


def _raw_matrix_for(g, x_np: Optional[np.ndarray] = None, eps: float = 1e-8) -> np.ndarray:
    """Return the graph's connectivity matrix for cluster summaries.

    Uses the stored `raw_matrix` when present (wSMI mode). For time-series graphs
    it is omitted to save RAM, so recompute the per-channel Pearson correlation
    from the node features (invariant to the global affine normalization).
    """
    rm = getattr(g, "raw_matrix", None)
    if rm is not None:
        return np.asarray(rm)
    if x_np is None:
        x_np = g.x.detach().cpu().numpy()
    centered = x_np - x_np.mean(axis=1, keepdims=True)
    std = centered.std(axis=1, keepdims=True).clip(min=eps)
    normed = centered / std
    return ((normed @ normed.T) / x_np.shape[1]).astype(np.float32)


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
            raw_matrices.append(_raw_matrix_for(g))
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


def extract_embeddings_encoder(
    model,
    graphs: Sequence,
    splits: Sequence[str],
    device: Optional[torch.device] = None,
) -> LatentBundle:
    """Latent extraction for the encoder-only `enc_gae_fc` (GNNEncoder).

    Unlike the autoencoders, this model already returns ONE embedding per graph
    (it pools over nodes internally), so there is no node-level aggregation.
    Packages the same LatentBundle fields so the clustering / dynamics / decoder
    tools run unchanged.
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
            z = model(x, edge_index, batch=None)  # (1, latent_dim)
            embeds.append(z.squeeze(0).detach().cpu().numpy())
            subject_ids.append(str(getattr(g, "subject_id", "unknown")))
            sessions.append(str(getattr(g, "session_num", "unknown")))
            epochs.append(int(getattr(g, "matrix_idx", -1)))
            diagnoses.append(str(getattr(g, "diagnosis", "UNK")))
            diagnosis_groups.append(str(getattr(g, "diagnosis_group", "UNK")))
            raw_matrices.append(_raw_matrix_for(g))
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
    ks = list(k_range)
    sils, fitted = [], {}
    for k in ks:
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
    if not fitted:
        # Degenerate case (e.g. fewer samples than 2): one cluster, no crash.
        return (np.zeros(X.shape[0], dtype=int), 1,
                {"k": ks, "silhouette": sils, "best_k": 1})
    # Pick the best k AMONG successfully fitted ones (argmax over all k can land
    # on a skipped/failed k that isn't in `fitted`).
    best_k = max(fitted, key=lambda k: sils[ks.index(k)])
    _, labels = fitted[best_k]
    return labels, best_k, {"k": ks, "silhouette": sils, "best_k": best_k}


def gmm_best_k(X: np.ndarray, k_range=range(2, 11), random_state: int = 42):
    ks = list(k_range)
    # 'full' covariance needs a D×D matrix per component, which is singular /
    # explodes when D is large (e.g. 64*latent_dim with --graph_latent_agg flatten).
    # Use 'diag' for high-dim embeddings; keep 'full' for compact ones.
    cov_type = "full" if X.shape[1] <= 32 else "diag"
    bics, fitted = [], {}
    for k in ks:
        if k >= X.shape[0]:
            bics.append(np.inf)
            continue
        try:
            gmm = GaussianMixture(
                n_components=k, covariance_type=cov_type,
                random_state=random_state, n_init=3, max_iter=200,
            )
            gmm.fit(X)
            bics.append(float(gmm.bic(X)))
            fitted[k] = (gmm, gmm.predict(X))
        except Exception:
            bics.append(np.inf)
    if not fitted:
        return (np.zeros(X.shape[0], dtype=int), 1,
                {"k": ks, "bic": bics, "best_k": 1, "covariance_type": cov_type})
    # Best k AMONG fitted ones (guards KeyError when the global argmin is a
    # skipped/failed k).
    best_k = min(fitted, key=lambda k: bics[ks.index(k)])
    _, labels = fitted[best_k]
    return labels, best_k, {"k": ks, "bic": bics, "best_k": best_k,
                            "covariance_type": cov_type}


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
    group_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Wide DataFrame: cluster x diagnosis_group counts (rows include noise=-1)."""
    rows = defaultdict(Counter)
    for c, dg in zip(labels, diagnosis_groups):
        rows[int(c)][dg] += 1
    clusters = sorted(rows.keys())
    if group_order is None:
        group_order = resolve_group_order(diagnosis_groups)
    cols = [g for g in group_order if any(g in r for r in rows.values())]
    df = pd.DataFrame(
        [[rows[c].get(g, 0) for g in cols] for c in clusters],
        index=[f"cluster_{c}" for c in clusters],
        columns=cols,
    )
    df.index.name = "cluster"
    return df


def per_diagnosis_cluster_occupancy(
    labels: np.ndarray, diagnosis_groups: Sequence[str],
    group_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """MATRIX-LEVEL state occupancy P(cluster | diagnosis) — the Della-Bella view.

    Counts every matrix (epoch), ignoring subjects. For each diagnosis group g and
    cluster c: occupancy[c, g] = (#matrices of g in c) / (#matrices of g over ALL
    clusters). So each **column (diagnosis) sums to 1 across clusters** — "of all UWS
    matrices, what fraction lands in each state".

    Toy: cluster_0 has 3 UWS, cluster_1 has 1 UWS -> UWS column = [0.75, 0.25].
    """
    counts = per_cluster_diagnosis_counts(labels, diagnosis_groups, group_order)
    occ = counts.div(counts.sum(axis=0).replace(0, 1), axis=1)
    return occ


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


def per_subject_cluster_proportions(
    labels: np.ndarray, subject_ids: Sequence[str], diagnosis_groups: Sequence[str],
    clusters: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Per-subject cluster-occupancy proportions (long form).

    For each subject, proportion(c) = #epochs in cluster c / #subject epochs, so a
    subject's proportions sum to 1 across clusters (0 for clusters never visited).

    Returns columns: subject_id, diagnosis_group, cluster, proportion.
    """
    if clusters is None:
        clusters = sorted(set(int(l) for l in labels))
    # Key by (subject, diagnosis_group): in fine mode a subject's diagnosis can
    # differ across sessions, so each (subject, group) is its own unit (matches
    # state_dynamics.per_subject_aggregate). In coarse mode this reduces to one
    # unit per subject.
    bag: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for s, c, dg in zip(subject_ids, labels, diagnosis_groups):
        bag[(s, dg)][int(c)] += 1
    rows = []
    for (s, dg), ctr in bag.items():
        n = sum(ctr.values())
        for c in clusters:
            rows.append({
                "subject_id": s, "diagnosis_group": dg, "cluster": int(c),
                "proportion": (ctr.get(int(c), 0) / n) if n > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def per_subject_within_cluster_share(
    labels: np.ndarray, subject_ids: Sequence[str], diagnosis_groups: Sequence[str],
    clusters: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Per-subject share of each cluster (column / within-cluster normalization).

    For each (subject, diagnosis_group) unit, share(c) = #unit epochs in cluster c
    / #total epochs in cluster c (across all units). Within a fixed cluster the
    shares sum to 1 (it's normalized by the cluster's occupancy, not by the
    subject's). This is the column-wise complement of
    `per_subject_cluster_proportions` (which normalizes within each subject).

    Returns columns: subject_id, diagnosis_group, cluster, share.
    """
    if clusters is None:
        clusters = sorted(set(int(l) for l in labels))
    cluster_total = {int(c): int((np.asarray(labels) == c).sum()) for c in clusters}
    bag: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for s, c, dg in zip(subject_ids, labels, diagnosis_groups):
        bag[(s, dg)][int(c)] += 1
    rows = []
    for (s, dg), ctr in bag.items():
        for c in clusters:
            tot = cluster_total[int(c)]
            rows.append({
                "subject_id": s, "diagnosis_group": dg, "cluster": int(c),
                "share": (ctr.get(int(c), 0) / tot) if tot > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def group_modal_cluster_fractions(
    labels: np.ndarray, subject_ids: Sequence[str], diagnosis_groups: Sequence[str],
    clusters: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Fraction of each diagnosis group's subjects whose MODAL cluster is c.

    For each subject, take their modal cluster (most-occupied). Then for each
    diagnosis group, fraction(group, c) = #subjects-of-group with modal cluster c
    / #subjects-of-group. These sum to 1 across clusters per group.

    Returns columns: diagnosis_group, cluster, fraction, n_subjects.
    """
    if clusters is None:
        clusters = sorted(set(int(l) for l in labels))
    # Unit = (subject, diagnosis_group) so a session-varying diagnosis counts in
    # each group it occurred in (see per_subject_cluster_proportions).
    bag: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for s, c, dg in zip(subject_ids, labels, diagnosis_groups):
        bag[(s, dg)][int(c)] += 1
    # unit -> modal cluster
    modal = {key: ctr.most_common(1)[0][0] for key, ctr in bag.items()}
    # per group: count units by modal cluster
    group_counts: Dict[str, Counter] = defaultdict(Counter)
    group_n: Counter = Counter()
    for (s, dg), mc in modal.items():
        group_counts[dg][mc] += 1
        group_n[dg] += 1
    rows = []
    for g in group_counts:
        for c in clusters:
            rows.append({
                "diagnosis_group": g, "cluster": int(c),
                "fraction": group_counts[g].get(int(c), 0) / max(1, group_n[g]),
                "n_subjects": int(group_n[g]),
            })
    return pd.DataFrame(rows)


def _grouped_cluster_positions(n_clusters: int, n_groups: int, width: float = 0.8):
    """Return (cluster_centers, per-group x-offsets, box_width) for grouped plots."""
    centers = np.arange(n_clusters)
    box_w = width / max(1, n_groups)
    offsets = [(-width / 2) + box_w * (i + 0.5) for i in range(n_groups)]
    return centers, offsets, box_w


def plot_cluster_proportion_boxplots(
    prop_df: pd.DataFrame, clusters: Sequence[int], out_path: str, title: str = "",
    group_order: Optional[Sequence[str]] = None,
    group_colors: Optional[Dict[str, str]] = None,
    value_col: str = "proportion",
    ylabel: str = "Per-subject proportion of epochs (sums to 1 over clusters)",
):
    """Grouped boxplots: x=cluster, one box per diagnosis group, y=`value_col`.
    Subjects overlaid as jittered points. `value_col` is `proportion` (per-subject
    normalization) or `share` (within-cluster normalization)."""
    if group_order is None:
        group_order = resolve_group_order(prop_df["diagnosis_group"].tolist())
    if group_colors is None:
        group_colors = resolve_group_colors(group_order)
    groups = [g for g in group_order if g in set(prop_df["diagnosis_group"])]
    clusters = list(clusters)
    centers, offsets, box_w = _grouped_cluster_positions(len(clusters), len(groups))

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(clusters)), 5))
    for gi, g in enumerate(groups):
        gdf = prop_df[prop_df["diagnosis_group"] == g]
        data = [gdf[gdf["cluster"] == c][value_col].values for c in clusters]
        positions = [centers[ci] + offsets[gi] for ci in range(len(clusters))]
        bp = ax.boxplot(data, positions=positions, widths=box_w * 0.9,
                        patch_artist=True, showmeans=True, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(group_colors.get(g, "#cccccc"))
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        # jittered points
        for ci, c in enumerate(clusters):
            vals = data[ci]
            if len(vals):
                jitter = np.random.normal(0, box_w * 0.12, size=len(vals))
                ax.scatter(np.full(len(vals), positions[ci]) + jitter, vals,
                           s=8, color="black", alpha=0.5, zorder=3)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(c) for c in clusters])
    ax.set_xlabel("Cluster")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=group_colors.get(g, "#cccccc"),
                             alpha=0.55) for g in groups]
    ax.legend(handles, groups, title="diagnosis", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_prevalence_proportion_boxplots(
    prop_df: pd.DataFrame, clusters: Sequence[int],
    out_path: str, title: str = "",
    group_order: Optional[Sequence[str]] = None,
    group_colors: Optional[Dict[str, str]] = None,
    value_col: str = "proportion",
):
    """Per-subject cluster proportions with the group MEAN marked as a ★.

    For each (cluster c, diagnosis group g) this boxplots the per-subject
    proportions — proportion[s, c] = (#epochs of subject s in cluster c) /
    (#epochs of subject s) — one point per subject, each in [0, 1]. The ★ sits at
    the group MEAN of those per-subject proportions, so the points spread AROUND
    the marker (unlike a sum, which the points could never surround).

    `prop_df`: columns subject_id, diagnosis_group, cluster, `value_col`
        (`proportion`, from per_subject_cluster_proportions). The overall
        cluster composition (the prevalence numbers) is shown separately in
        prevalence_proportions.png.
    """
    if group_order is None:
        group_order = resolve_group_order(prop_df["diagnosis_group"].tolist())
    if group_colors is None:
        group_colors = resolve_group_colors(group_order)
    groups = [g for g in group_order if g in set(prop_df["diagnosis_group"])]
    clusters = list(clusters)
    centers, offsets, box_w = _grouped_cluster_positions(len(clusters), len(groups))

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(clusters)), 5))
    for gi, g in enumerate(groups):
        gdf = prop_df[prop_df["diagnosis_group"] == g]
        data = [gdf[gdf["cluster"] == c][value_col].values for c in clusters]
        positions = [centers[ci] + offsets[gi] for ci in range(len(clusters))]
        bp = ax.boxplot(data, positions=positions, widths=box_w * 0.9,
                        patch_artist=True, showmeans=False, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(group_colors.get(g, "#cccccc"))
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        for ci, c in enumerate(clusters):
            vals = data[ci]
            if len(vals):
                jitter = np.random.normal(0, box_w * 0.12, size=len(vals))
                ax.scatter(np.full(len(vals), positions[ci]) + jitter, vals,
                           s=8, color="black", alpha=0.4, zorder=3)
                # ★ at the group MEAN of the per-subject proportions
                ax.scatter([positions[ci]], [float(np.mean(vals))],
                           marker="*", s=170, color=group_colors.get(g, "#cccccc"),
                           edgecolor="black", linewidth=0.6, zorder=5)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(c) for c in clusters])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("per-subject proportion of epochs in cluster  (★ = group mean)")
    if title:
        ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=group_colors.get(g, "#cccccc"),
                             alpha=0.55) for g in groups]
    ax.legend(handles, groups, title="diagnosis", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_state_occupancy_by_diagnosis(
    occ: pd.DataFrame, out_path: str, title: str = "",
    group_colors: Optional[Dict[str, str]] = None,
):
    """Grouped bars of MATRIX-LEVEL occupancy P(cluster | diagnosis) (the Della-Bella
    figure). x = cluster/state, one bar per diagnosis; each diagnosis's bars sum to 1
    across clusters. `occ` is the output of `per_diagnosis_cluster_occupancy`
    (rows = 'cluster_<c>', columns = diagnosis groups, each COLUMN sums to 1)."""
    clusters = list(occ.index)
    groups = list(occ.columns)
    if group_colors is None:
        group_colors = resolve_group_colors(groups)
    centers, offsets, box_w = _grouped_cluster_positions(len(clusters), len(groups))
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(clusters)), 5))
    for gi, g in enumerate(groups):
        positions = [centers[ci] + offsets[gi] for ci in range(len(clusters))]
        heights = [float(occ.loc[c, g]) for c in clusters]
        ax.bar(positions, heights, width=box_w * 0.9,
               color=group_colors.get(g, "#cccccc"), alpha=0.85, label=g)
    ax.set_xticks(centers)
    ax.set_xticklabels([c.replace("cluster_", "") for c in clusters])
    ax.set_xlabel("Cluster / state")
    ax.set_ylabel("P(cluster | diagnosis)  — each diagnosis sums to 1 across clusters")
    if title:
        ax.set_title(title)
    ax.legend(title="diagnosis", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_group_modal_cluster_bars(
    frac_df: pd.DataFrame, clusters: Sequence[int], out_path: str, title: str = "",
    group_order: Optional[Sequence[str]] = None,
    group_colors: Optional[Dict[str, str]] = None,
):
    """Grouped bar chart: x=cluster, one bar per diagnosis group, height=fraction
    of that group's subjects whose modal cluster is c (sums to 1 across clusters
    per group)."""
    if group_order is None:
        group_order = resolve_group_order(frac_df["diagnosis_group"].tolist())
    if group_colors is None:
        group_colors = resolve_group_colors(group_order)
    groups = [g for g in group_order if g in set(frac_df["diagnosis_group"])]
    clusters = list(clusters)
    centers, offsets, bar_w = _grouped_cluster_positions(len(clusters), len(groups))

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(clusters)), 5))
    for gi, g in enumerate(groups):
        gdf = frac_df[frac_df["diagnosis_group"] == g].set_index("cluster")
        heights = [float(gdf.loc[c, "fraction"]) if c in gdf.index else 0.0
                   for c in clusters]
        positions = [centers[ci] + offsets[gi] for ci in range(len(clusters))]
        ax.bar(positions, heights, width=bar_w * 0.9,
               color=group_colors.get(g, "#cccccc"), alpha=0.85, label=g)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(c) for c in clusters])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Fraction of group's subjects (modal cluster; sums to 1)")
    if title:
        ax.set_title(title)
    ax.legend(title="diagnosis", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_prevalence(df: pd.DataFrame, title: str, out_path: str, normalize: bool = False,
                    group_colors: Optional[Dict[str, str]] = None):
    if normalize:
        plot_df = df.div(df.sum(axis=1).replace(0, 1), axis=0)
        ylabel = "Proportion"
    else:
        plot_df = df
        ylabel = "Count"
    if group_colors is None:
        group_colors = resolve_group_colors(list(plot_df.columns))
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(plot_df)), 5))
    bottom = np.zeros(len(plot_df))
    for g in plot_df.columns:
        ax.bar(plot_df.index, plot_df[g].values, bottom=bottom,
               label=g, color=group_colors.get(g, "#999999"))
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


def plot_per_subject_entropy(df: pd.DataFrame, out_path: str, title: str = "",
                             group_order: Optional[Sequence[str]] = None,
                             group_colors: Optional[Dict[str, str]] = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    if group_order is None:
        group_order = resolve_group_order(df["diagnosis_group"].tolist())
    if group_colors is None:
        group_colors = resolve_group_colors(group_order)
    groups = [g for g in group_order if g in df["diagnosis_group"].unique()]
    data = [df[df["diagnosis_group"] == g]["shannon_entropy"].values for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True, showmeans=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(group_colors.get(g, "#cccccc"))
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


def plot_latent_scatter(embeds: np.ndarray, labels: np.ndarray, name: str,
                        out_dir: str) -> None:
    """Save a 2D (latent_scatter.png) and a 3D (latent_scatter_3d.png) PCA scatter
    of the latents, coloured by cluster. PCA is fit once to 3 components and the
    2D plot reuses its first two, so both views share the same projection.
    """
    from sklearn.decomposition import PCA
    X = np.asarray(embeds)
    n_comp = min(3, X.shape[1])
    Xp = PCA(n_components=n_comp, random_state=0).fit_transform(X) if X.shape[1] > 1 else X
    # pad to >=3 cols so 2D/3D indexing is uniform for tiny latent dims
    if Xp.shape[1] < 3:
        Xp = np.column_stack([Xp, np.zeros((len(Xp), 3 - Xp.shape[1]))])

    # 2D
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Xp[:, 0], Xp[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    ax.set_title(f"{name}: latent (PCA-2) coloured by cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latent_scatter.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # 3D
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], c=labels, cmap="tab10",
                    s=8, alpha=0.7, depthshade=True)
    ax.set_title(f"{name}: latent (PCA-3) coloured by cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.colorbar(sc, ax=ax, label="cluster", shrink=0.6, pad=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latent_scatter_3d.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def run_one_clusterer(
    name: str, labels: np.ndarray, bundle: LatentBundle, out_dir: str,
    electrode_labels: Optional[Sequence[str]], metric_curve: dict,
):
    os.makedirs(out_dir, exist_ok=True)

    # Resolve diagnosis-group ordering + colours once (works for coarse + fine).
    group_order = resolve_group_order(bundle.diagnosis_groups)
    group_colors = resolve_group_colors(group_order)
    clusters = sorted(set(int(l) for l in labels))

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
    counts = per_cluster_diagnosis_counts(labels, bundle.diagnosis_groups, group_order)
    counts.to_csv(os.path.join(out_dir, "prevalence_counts.csv"))
    plot_prevalence(counts, f"{name}: diagnosis_group per cluster (counts)",
                    os.path.join(out_dir, "prevalence_counts.png"), normalize=False,
                    group_colors=group_colors)
    plot_prevalence(counts, f"{name}: diagnosis_group per cluster (proportions)",
                    os.path.join(out_dir, "prevalence_proportions.png"), normalize=True,
                    group_colors=group_colors)
    # (a1) MATRIX-LEVEL state occupancy P(cluster|diagnosis): each diagnosis sums to 1
    # across clusters (Della-Bella view). Column-normalized complement of prevalence.
    occ = per_diagnosis_cluster_occupancy(labels, bundle.diagnosis_groups, group_order)
    occ.to_csv(os.path.join(out_dir, "state_occupancy_by_diagnosis.csv"))
    plot_state_occupancy_by_diagnosis(
        occ, os.path.join(out_dir, "state_occupancy_by_diagnosis.png"),
        title=f"{name}: state occupancy P(cluster|diagnosis)", group_colors=group_colors)

    subj_modal = per_subject_modal_cluster(
        labels, bundle.subject_ids, bundle.diagnosis_groups)
    subj_modal.to_csv(os.path.join(out_dir, "subject_modal_cluster.csv"), index=False)

    # (a2) per-cluster occupancy plots grouped by diagnosis.
    #   Plot A: per-subject proportions (sum to 1 across clusters per subject).
    prop_df = per_subject_cluster_proportions(
        labels, bundle.subject_ids, bundle.diagnosis_groups, clusters)
    prop_df.to_csv(os.path.join(out_dir, "per_subject_cluster_proportions.csv"), index=False)
    plot_cluster_proportion_boxplots(
        prop_df, clusters, os.path.join(out_dir, "cluster_proportion_boxplots.png"),
        title=f"{name}: per-subject cluster occupancy by diagnosis",
        group_order=group_order, group_colors=group_colors)
    #   Plot A2: same boxplots, normalized WITHIN each cluster (shares sum to 1
    #   over subjects in a cluster) — i.e. normalized by the cluster's occupancy.
    share_df = per_subject_within_cluster_share(
        labels, bundle.subject_ids, bundle.diagnosis_groups, clusters)
    share_df.to_csv(os.path.join(out_dir, "per_subject_within_cluster_share.csv"), index=False)
    plot_cluster_proportion_boxplots(
        share_df, clusters,
        os.path.join(out_dir, "cluster_proportion_boxplots_within_cluster.png"),
        title=f"{name}: within-cluster subject shares by diagnosis",
        group_order=group_order, group_colors=group_colors,
        value_col="share",
        ylabel="Per-subject share of the cluster (sums to 1 within a cluster)")
    #   Plot A3: per-subject cluster proportions with the group MEAN starred, so
    #   the per-subject points spread around the ★ (the overall prevalence numbers
    #   live in prevalence_proportions.png).
    plot_prevalence_proportion_boxplots(
        prop_df, clusters,
        os.path.join(out_dir, "prevalence_proportion_boxplots.png"),
        title=f"{name}: per-subject cluster proportions (★ = group mean) by diagnosis",
        group_order=group_order, group_colors=group_colors)
    #   Plot B: per-group modal-cluster subject fractions (sum to 1 per group).
    frac_df = group_modal_cluster_fractions(
        labels, bundle.subject_ids, bundle.diagnosis_groups, clusters)
    frac_df.to_csv(os.path.join(out_dir, "group_modal_cluster_fractions.csv"), index=False)
    plot_group_modal_cluster_bars(
        frac_df, clusters, os.path.join(out_dir, "group_modal_cluster_bars.png"),
        title=f"{name}: modal-cluster subject fractions by diagnosis",
        group_order=group_order, group_colors=group_colors)

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
        group_order=group_order, group_colors=group_colors,
    )

    # metric curve
    curve_kind = {"kmeans": "silhouette", "gmm": "bic", "hdbscan": "hdbscan"}[name]
    save_metric_curve(metric_curve, os.path.join(out_dir, "metric_curve.png"),
                      kind=curve_kind)
    with open(os.path.join(out_dir, "metric_curve.json"), "w") as f:
        json.dump(metric_curve, f, indent=2, default=float)

    # latent scatter (PCA-2 and PCA-3)
    try:
        plot_latent_scatter(bundle.embeds, labels, name, out_dir)
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

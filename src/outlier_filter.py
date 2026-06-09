"""
MAHALANOBIS 2-SIGMA OUTLIER FILTER
==================================
Drop epoch-graphs whose wSMI connectivity matrix is an outlier under a
multivariate-Gaussian model fit on the TRAIN split only, then apply the same
model unchanged to val/test (so the decision boundary never sees held-out data).

Design
------
Each epoch's wSMI matrix is reduced to its off-diagonal upper triangle, i.e. an
n = C*(C-1)/2 dimensional point (n = 2016 for biosemi64). We treat the train
points as samples of a diagonal n-D Gaussian: estimate a per-dimension sigma and
flag a point as an outlier when it lies beyond the n-D analogue of the "two
sigma" boundary.

  - standardize each dimension by its own train sigma:  z_j = (x_j - mu_j)/sigma_j
  - squared (diagonal) Mahalanobis distance  d2(x) = sum_j z_j^2
  - keep iff d2 <= d2_max.

A *diagonal* covariance is used on purpose: the full 2016x2016 covariance is
rank-deficient whenever the train split has fewer epochs than dimensions (the
common case here), which makes out-of-sample Mahalanobis distances explode.
Per-dimension sigmas need no matrix inversion and stay well-conditioned, and they
match the "estimator of sigma" framing directly.

The 2-sigma boundary (`d2_max`) is set per `--outlier_threshold`:
  - "empirical" (default): the conf-quantile of the TRAIN d2 distribution, so the
    boundary is read straight off the samples and ~ (1 - conf) of train is flagged
    regardless of feature correlation. conf = 2*Phi(n_sigma) - 1
    (n_sigma = 2 -> conf = 0.9545).
  - "chi2": the theoretical chi2.ppf(conf, df=n), exact only if the standardized
    features were independent Gaussians.

The basis matrix is read from each graph's `wsmi_matrix` attribute, set by the
loaders (wsmi_loader sets it to the wSMI matrix itself; the time-series loader
attaches the matched wSMI matrix), so the filter always tests the real wSMI and
never a Pearson proxy.
"""
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import chi2, norm


def _upper_tri_vectors(graphs: Sequence) -> np.ndarray:
    """Stack the off-diagonal upper triangle of each graph's wSMI matrix.

    Returns an array of shape (n_graphs, C*(C-1)/2). Raises if any graph is
    missing the `wsmi_matrix` attribute (the loaders must attach it).
    """
    vecs = []
    iu = None
    for g in graphs:
        # wSMI mode keeps a single copy under `raw_matrix`; time-series mode
        # attaches a matched `wsmi_matrix`. Prefer the latter, fall back to the
        # former so we never store a duplicate.
        mat = getattr(g, "wsmi_matrix", None)
        if mat is None:
            mat = getattr(g, "raw_matrix", None)
        if mat is None:
            raise ValueError(
                "outlier filter requires graphs to carry a `wsmi_matrix` (or "
                "`raw_matrix`) attribute; none found. Ensure the loader attaches it."
            )
        mat = np.asarray(mat, dtype=np.float32)
        if iu is None:
            iu = np.triu_indices(mat.shape[0], k=1)
        vecs.append(mat[iu])
    return np.stack(vecs, axis=0)


def _diag_mahalanobis(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Squared diagonal Mahalanobis distance per row: sum_j ((x_j - mu_j)/sigma_j)^2."""
    z = (X - mean) / std
    return np.einsum("ij,ij->i", z, z)


def fit_outlier_model(train_graphs: Sequence, n_sigma: float = 2.0,
                      threshold: str = "empirical", eps: float = 1e-8) -> Dict:
    """Fit the diagonal-Gaussian outlier model on TRAIN graphs only.

    Args:
        train_graphs: graphs whose `wsmi_matrix` defines the n-D points.
        n_sigma: the 1-D sigma multiple whose two-tailed mass sets the cutoff.
        threshold: "empirical" (conf-quantile of train d2) or "chi2"
            (chi2.ppf(conf, df=n)).
        eps: floor on per-dimension std (guards flat dimensions).

    Returns a JSON-friendly-ish dict with mean/std and the d2 threshold.
    """
    X = _upper_tri_vectors(train_graphs)
    n_dims = X.shape[1]
    mean = X.mean(axis=0)
    std = X.std(axis=0).clip(min=eps)
    conf = float(2.0 * norm.cdf(n_sigma) - 1.0)  # n_sigma=2 -> 0.9545
    train_d2 = _diag_mahalanobis(X, mean, std)
    chi2_d2_max = float(chi2.ppf(conf, df=n_dims))
    emp_d2_max = float(np.quantile(train_d2, conf))
    d2_max = emp_d2_max if threshold == "empirical" else chi2_d2_max
    return {
        "mean": mean,
        "std": std,
        "d2_max": float(d2_max),
        "threshold": str(threshold),
        "empirical_d2_max": emp_d2_max,
        "chi2_d2_max": chi2_d2_max,
        "conf": conf,
        "n_sigma": float(n_sigma),
        "n_dims": int(n_dims),
        "n_train": int(X.shape[0]),
        "n_train_dropped": int((train_d2 > d2_max).sum()),
    }


def apply_outlier_model(graphs: Sequence, model: Dict
                        ) -> Tuple[List, np.ndarray, np.ndarray]:
    """Apply a fitted model to a graph list.

    Returns (kept_graphs, keep_mask, d2) where keep_mask is True for inliers.
    """
    if len(graphs) == 0:
        return [], np.zeros(0, dtype=bool), np.zeros(0)
    X = _upper_tri_vectors(graphs)
    d2 = _diag_mahalanobis(X, model["mean"], model["std"])
    keep = d2 <= model["d2_max"]
    kept = [g for g, k in zip(graphs, keep) if k]
    return kept, keep, d2


def plot_outlier_diagnostics(
    model: Dict,
    pre_graphs: Dict[str, Sequence],
    masks: Dict[str, np.ndarray],
    d2: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Visual sanity checks for the outlier filter (saved as one PNG).

    Three panels:
      A) PCA(2) of the standardized TRAIN wSMI vectors — inliers vs flagged
         outliers. PCA is fit on TRAIN only; val/test are projected into the same
         axes so you can eyeball whether the boundary generalizes.
      B) Histogram of the squared diagonal-Mahalanobis distance d2 per split, with
         the d2_max cutoff drawn as a vertical line. Inliers fall left of it.
      C) Drop fraction per diagnosis_group (all splits pooled) — checks the filter
         is not disproportionately removing one clinical group.

    Requires the graphs to still carry `wsmi_matrix`/`raw_matrix` (call before the
    pipeline frees them). No-op (prints a warning) if matplotlib is unavailable or
    TRAIN is too small.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  [outlier-plot] skipped (matplotlib unavailable: {e})")
        return

    mean, std = model["mean"], model["std"]
    d2_max = model["d2_max"]

    # --- standardized vectors per split (reuse the filter's mean/std) ---
    z = {}
    for name, graphs in pre_graphs.items():
        if len(graphs) == 0:
            continue
        z[name] = (_upper_tri_vectors(graphs) - mean) / std
    if "train" not in z or z["train"].shape[0] < 3:
        print("  [outlier-plot] skipped (TRAIN too small for PCA)")
        return

    # --- PCA fit on TRAIN (numpy SVD, no sklearn dependency) ---
    ztr = z["train"]
    ztr_mean = ztr.mean(axis=0)
    _, S, Vt = np.linalg.svd(ztr - ztr_mean, full_matrices=False)
    comps = Vt[:2]
    evr = (S ** 2) / (S ** 2).sum()

    def project(zz):
        return (zz - ztr_mean) @ comps.T

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: TRAIN PCA scatter, inliers vs outliers
    keep = masks["train"]
    p = project(ztr)
    axes[0].scatter(p[keep, 0], p[keep, 1], s=8, c="#2c7fb8", alpha=0.5,
                    label=f"inlier (n={int(keep.sum())})")
    axes[0].scatter(p[~keep, 0], p[~keep, 1], s=30, c="#d62728", marker="x",
                    label=f"outlier (n={int((~keep).sum())})")
    axes[0].set_title("TRAIN epochs — PCA of standardized wSMI")
    axes[0].set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    axes[0].legend(loc="best", fontsize=8)

    # Panel B: d2 histograms per split + threshold
    colors = {"train": "#2c7fb8", "val": "#fdae61", "test": "#7570b3"}
    for name in ("train", "val", "test"):
        if name in d2 and len(d2[name]) > 0:
            axes[1].hist(d2[name], bins=40, histtype="step", linewidth=1.5,
                         color=colors[name], label=f"{name} (n={len(d2[name])})")
    axes[1].axvline(d2_max, color="#d62728", lw=2, ls="--",
                    label=f"d2_max={d2_max:.0f}")
    axes[1].set_title(f"Squared diag-Mahalanobis d2 (conf={model['conf']:.4f})")
    axes[1].set_xlabel("d2 (sum of squared z-scores)")
    axes[1].set_ylabel("count")
    axes[1].legend(loc="best", fontsize=8)

    # Panel C: drop fraction by diagnosis_group (all splits pooled)
    tot, drop = Counter(), Counter()
    for name, graphs in pre_graphs.items():
        m = masks.get(name)
        if m is None:
            continue
        for g, k in zip(graphs, m):
            grp = str(getattr(g, "diagnosis_group", "UNK"))
            tot[grp] += 1
            if not k:
                drop[grp] += 1
    groups = sorted(tot)
    fracs = [drop[g] / tot[g] if tot[g] else 0.0 for g in groups]
    axes[2].bar(groups, fracs, color="#d62728", alpha=0.8)
    for i, g in enumerate(groups):
        axes[2].text(i, fracs[i], f"{drop[g]}/{tot[g]}", ha="center",
                     va="bottom", fontsize=8)
    axes[2].set_title("Outlier drop fraction by diagnosis_group")
    axes[2].set_ylabel("fraction dropped")
    axes[2].set_ylim(0, max(fracs + [0.05]) * 1.25)

    fig.suptitle("Outlier filter diagnostics (model fit on TRAIN only)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [outlier-plot] wrote {output_path}")


def summarize(model: Dict, split_results: Dict[str, np.ndarray],
              split_graphs: Dict[str, Sequence]) -> Dict:
    """Build a JSON-serializable report of drop counts per split / diagnosis.

    Args:
        model: the dict from `fit_outlier_model` (estimator is dropped here).
        split_results: {split_name: keep_mask} from `apply_outlier_model`.
        split_graphs: {split_name: original graph list} (pre-filter), used to
            break drops down by diagnosis_group.
    """
    report = {k: v for k, v in model.items() if k not in ("mean", "std")}
    per_split = {}
    for name, keep in split_results.items():
        graphs = split_graphs[name]
        dropped_groups = Counter(
            str(getattr(g, "diagnosis_group", "UNK"))
            for g, k in zip(graphs, keep) if not k
        )
        per_split[name] = {
            "n_total": int(len(keep)),
            "n_kept": int(keep.sum()),
            "n_dropped": int((~keep).sum()),
            "dropped_by_diagnosis_group": dict(dropped_groups),
        }
    report["per_split"] = per_split
    return report

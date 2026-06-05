"""
LATENT CLUSTER ANALYSIS
=======================
Purpose: Post-training analysis on the latent embeddings saved by train.py.

Pipeline Position: AFTER TRAINING
- Input: saved latent JSONs (`{subset}_latent_{experiment}.json`) + raw feature matrices
- Output: cluster assignments, per-cluster mean matrices, within-session Markov transitions,
  Shannon entropy and variance-weighted entropy.

Workflow:
1. Load latent embeddings (node-level) per graph, aggregate to graph-level via mean across nodes.
2. K-Means on pooled train+val+test graph embeddings.
3. For each cluster, fetch the raw feature matrices (via DEFAULT_MATRIX_PATH_PATTERN or a
   user-supplied pattern), compute the mean matrix and a scalar intra-cluster variance.
4. Build a row-normalized K x K Markov transition matrix from within-session sequential
   cluster pairs (matrix_idx i -> i+1).
5. Compute Shannon entropy of cluster occupancy and variance-weighted entropy.
"""
import os
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


DEFAULT_MATRIX_PATTERN = '{matrix_dir}/sub-{subject_id}/ses-{session_num}/matrix_{matrix_idx}.npy'


class LatentClusterAnalysis:
    """Cluster graph-level latent embeddings and analyze the resulting state space.

    Supports two clustering methods (selected via `method`):
      - 'kmeans': sklearn KMeans (hard assignments)
      - 'gmm':    sklearn GaussianMixture with full covariance (soft → argmax for
                  cluster labels; BIC/AIC reported for model selection)

    Quality metrics (`entropy_metrics`) include Shannon entropy of cluster
    occupancy, silhouette score, variance-weighted entropy, and — when
    method='gmm' — BIC/AIC. Use `sweep_k()` to compare across K.
    """

    def __init__(self, latent_dir, experiment_name, matrix_dir,
                 matrix_path_pattern=None, n_clusters=8,
                 subsets=('train', 'val', 'test'), random_state=42,
                 method='kmeans'):
        self.latent_dir = latent_dir
        self.experiment_name = experiment_name
        self.matrix_dir = matrix_dir
        self.matrix_path_pattern = matrix_path_pattern or DEFAULT_MATRIX_PATTERN
        self.n_clusters = n_clusters
        self.subsets = subsets
        self.random_state = random_state
        self.method = method.lower()
        if self.method not in ('kmeans', 'gmm'):
            raise ValueError(f"method must be 'kmeans' or 'gmm', got {method!r}")

        self.embeds = None        # [N, latent_dim] graph-level (mean across nodes)
        self.meta = None          # list of {subset, subject_id, session_num, matrix_idx}
        self.labels = None        # [N] cluster id
        self.electrodes = None
        self.estimator_ = None    # fitted sklearn estimator (KMeans or GaussianMixture)
        self.bic_ = None
        self.aic_ = None

    def _find_latent_path(self, subset):
        """Locate {subset}_latent_{experiment}.json, with a glob fallback."""
        candidate = os.path.join(self.latent_dir, f'{subset}_latent_{self.experiment_name}.json')
        if os.path.exists(candidate):
            return candidate
        matches = glob.glob(os.path.join(self.latent_dir, f'{subset}_latent_*.json'))
        if not matches:
            raise FileNotFoundError(f"No latent file found for subset '{subset}' in {self.latent_dir}")
        if len(matches) > 1:
            print(f"WARNING: multiple {subset} latent files found; using {matches[0]}")
        return matches[0]

    def load_and_aggregate(self):
        """Parse latent JSONs and produce graph-level embeddings + metadata."""
        embeds = []
        meta = []
        for subset in self.subsets:
            path = self._find_latent_path(subset)
            with open(path) as f:
                blob = json.load(f)
            if self.electrodes is None:
                self.electrodes = blob.get('electrodes')

            for subject_id, sessions in blob['data'].items():
                for session_num, matrices in sessions.items():
                    for matrix_idx, node_latents in matrices.items():
                        arr = np.asarray(node_latents)  # [n_nodes, latent_dim]
                        embeds.append(arr.mean(axis=0))
                        meta.append({
                            'subset': subset,
                            'subject_id': subject_id,
                            'session_num': session_num,
                            'matrix_idx': int(matrix_idx),
                        })

        self.embeds = np.stack(embeds, axis=0)
        self.meta = meta
        print(f"Loaded {len(meta)} graph embeddings of dim {self.embeds.shape[1]}")
        return self.embeds, self.meta

    def cluster(self):
        if self.embeds is None:
            self.load_and_aggregate()
        assert self.embeds.shape[0] >= self.n_clusters, \
            f"n_clusters={self.n_clusters} exceeds N={self.embeds.shape[0]}"

        if self.method == 'kmeans':
            est = KMeans(n_clusters=self.n_clusters, n_init=10,
                         random_state=self.random_state)
            self.labels = est.fit_predict(self.embeds)
            self.bic_ = None
            self.aic_ = None
        else:  # gmm
            est = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type='full',
                random_state=self.random_state,
                n_init=3,
                max_iter=200,
                reg_covar=1e-6,
            )
            est.fit(self.embeds)
            self.labels = est.predict(self.embeds)
            # BIC / AIC are nominally for model selection across K.
            self.bic_ = float(est.bic(self.embeds))
            self.aic_ = float(est.aic(self.embeds))

        self.estimator_ = est
        counts = np.bincount(self.labels, minlength=self.n_clusters)
        print(f"{self.method.upper()} done (K={self.n_clusters}). "
              f"Cluster sizes: {counts.tolist()}")
        return self.labels

    def silhouette(self, sample_size=None):
        """Silhouette score on the latent embeddings.

        For large N silhouette is O(N^2); pass `sample_size` to subsample.
        """
        if self.labels is None:
            self.cluster()
        unique = np.unique(self.labels)
        if len(unique) < 2:
            return float('nan')
        N = self.embeds.shape[0]
        if sample_size is not None and sample_size < N:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(N, size=sample_size, replace=False)
            return float(silhouette_score(self.embeds[idx], self.labels[idx]))
        return float(silhouette_score(self.embeds, self.labels))

    def sweep_k(self, k_range, method=None, silhouette_sample=20000):
        """Fit the chosen method across a range of K and return a metrics dict.

        Returns:
            {
                'method': 'kmeans' | 'gmm',
                'k': [...], 'silhouette': [...], 'inertia': [...],
                'bic': [...], 'aic': [...],   # only for GMM (NaN otherwise)
            }
        """
        method = (method or self.method).lower()
        ks, sil, inertia, bic, aic = [], [], [], [], []
        for k in k_range:
            if self.embeds.shape[0] < k:
                continue
            if method == 'kmeans':
                est = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
                labels = est.fit_predict(self.embeds)
                inertia.append(float(est.inertia_))
                bic.append(float('nan'))
                aic.append(float('nan'))
            else:
                est = GaussianMixture(
                    n_components=k, covariance_type='full',
                    random_state=self.random_state, n_init=3, max_iter=200,
                    reg_covar=1e-6,
                )
                est.fit(self.embeds)
                labels = est.predict(self.embeds)
                inertia.append(float('nan'))
                bic.append(float(est.bic(self.embeds)))
                aic.append(float(est.aic(self.embeds)))

            if len(np.unique(labels)) < 2:
                sil.append(float('nan'))
            else:
                N = self.embeds.shape[0]
                if silhouette_sample is not None and silhouette_sample < N:
                    rng = np.random.default_rng(self.random_state)
                    sidx = rng.choice(N, size=silhouette_sample, replace=False)
                    sil.append(float(silhouette_score(self.embeds[sidx], labels[sidx])))
                else:
                    sil.append(float(silhouette_score(self.embeds, labels)))
            ks.append(int(k))
            print(f"  {method.upper()} K={k}: silhouette={sil[-1]:.4f}"
                  + (f", BIC={bic[-1]:.1f}, AIC={aic[-1]:.1f}" if method == 'gmm'
                     else f", inertia={inertia[-1]:.1f}"))
        return {
            'method': method, 'k': ks, 'silhouette': sil,
            'inertia': inertia, 'bic': bic, 'aic': aic,
        }

    def _load_matrix(self, m):
        """Load the per-epoch feature matrix referenced by one metadata entry.

        Two on-disk formats are supported:
          (A) per-epoch  — pattern contains {matrix_idx}, e.g. `.../matrix_{matrix_idx}.npy`.
              The .npy file holds a single (n_nodes, n_nodes) matrix.
          (B) per-session — pattern has no {matrix_idx}, e.g. `.../data.npy`.
              The .npy holds (n_epochs, n_nodes, n_nodes); we mmap and slice
              the requested epoch. This matches the lazy-dataset cache layout
              produced by training (output/.../wsmi_npy_cache/sub-X/ses-Y/data.npy).
        """
        per_epoch = '{matrix_idx}' in self.matrix_path_pattern
        path = self.matrix_path_pattern.format(
            matrix_dir=self.matrix_dir,
            subject_id=m['subject_id'],
            session_num=m['session_num'],
            matrix_idx=m.get('matrix_idx', 0) if per_epoch else 0,
        )
        if per_epoch:
            return np.load(path)
        # Per-session: mmap once, then slice. Repeated calls on the same
        # session will re-open via mmap_mode='r' — cheap (header-only read).
        arr = np.load(path, mmap_mode='r')
        # Copy out of mmap so the caller can freely stack / aggregate.
        return np.array(arr[m['matrix_idx']], copy=True)

    def cluster_mean_matrices(self):
        """For each cluster, average the underlying raw feature matrices."""
        means = {}
        for c in range(self.n_clusters):
            idxs = np.where(self.labels == c)[0]
            mats = []
            for i in idxs:
                try:
                    mats.append(self._load_matrix(self.meta[i]))
                except FileNotFoundError as e:
                    print(f"WARNING: missing matrix for cluster {c}: {e}")
            if mats:
                means[c] = np.mean(np.stack(mats, axis=0), axis=0)
            else:
                means[c] = None
        return means

    def cluster_intra_variance(self):
        """Mean elementwise variance of raw matrices within each cluster (scalar per cluster)."""
        variances = {}
        for c in range(self.n_clusters):
            idxs = np.where(self.labels == c)[0]
            mats = []
            for i in idxs:
                try:
                    mats.append(self._load_matrix(self.meta[i]))
                except FileNotFoundError:
                    continue
            if len(mats) > 1:
                variances[c] = float(np.var(np.stack(mats, axis=0), axis=0).mean())
            else:
                variances[c] = 0.0
        return variances

    def transition_matrix(self):
        """Within-session sequential Markov transitions, row-normalized."""
        K = self.n_clusters
        T = np.zeros((K, K), dtype=float)

        # Group entries by (subject, session)
        groups = defaultdict(list)
        for i, m in enumerate(self.meta):
            groups[(m['subject_id'], m['session_num'])].append(i)

        for key, idxs in groups.items():
            # Sort by matrix_idx within the session
            idxs_sorted = sorted(idxs, key=lambda i: self.meta[i]['matrix_idx'])
            seq = [self.labels[i] for i in idxs_sorted]
            for a, b in zip(seq[:-1], seq[1:]):
                T[a, b] += 1.0

        row_sums = T.sum(axis=1, keepdims=True)
        P = T / np.clip(row_sums, a_min=1.0, a_max=None)
        return P

    def entropy_metrics(self, variances=None, include_silhouette=True,
                        silhouette_sample=20000):
        """Shannon entropy + variance-weighted variants + silhouette.

        Also reports BIC/AIC when the active method is GMM.
        """
        eps = 1e-12
        counts = np.bincount(self.labels, minlength=self.n_clusters).astype(float)
        p = counts / counts.sum()
        H = float(-np.sum(p * np.log(p + eps)))

        out = {
            'method': self.method,
            'n_clusters': int(self.n_clusters),
            'shannon': H,
            'cluster_proportions': p.tolist(),
        }
        if include_silhouette:
            out['silhouette'] = self.silhouette(sample_size=silhouette_sample)
        if self.method == 'gmm':
            out['bic'] = self.bic_
            out['aic'] = self.aic_
        if variances is not None:
            var_arr = np.array([variances.get(c, 0.0) for c in range(self.n_clusters)])
            out['variance_weighted_sum'] = float(np.sum(p * var_arr))   # E[var | cluster] under p
            out['shannon_times_mean_variance'] = float(H * var_arr.mean())
            out['per_cluster_variance'] = var_arr.tolist()
        return out

    def save_all(self, output_dir, sweep_k_range=None, silhouette_sample=20000):
        """Run the full analysis and persist all artifacts under output_dir.

        If `sweep_k_range` is provided (iterable of K values), also runs the
        same clustering method across those K values, saves a JSON of metrics
        and a silhouette-vs-K (and BIC/AIC-vs-K for GMM) plot. The chosen
        `self.n_clusters` model remains the one used for cluster_assignments,
        transition matrix, and entropy metrics.
        """
        if self.labels is None:
            self.cluster()

        out_dir = os.path.join(output_dir, 'analysis', self.experiment_name)
        out_dir = os.path.join(out_dir, self.method)  # split kmeans/ vs gmm/
        os.makedirs(out_dir, exist_ok=True)

        # Cluster assignments
        assignments = [
            {**m, 'cluster': int(self.labels[i])} for i, m in enumerate(self.meta)
        ]
        with open(os.path.join(out_dir, 'cluster_assignments.json'), 'w') as f:
            json.dump(assignments, f, indent=2)

        # Mean matrices + variances
        means = self.cluster_mean_matrices()
        variances = self.cluster_intra_variance()
        for c, mat in means.items():
            if mat is not None:
                np.save(os.path.join(out_dir, f'mean_matrix_cluster_{c}.npy'), mat)

        # Transition matrix + heatmap
        P = self.transition_matrix()
        np.save(os.path.join(out_dir, 'transition_matrix.npy'), P)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(P, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_title('Within-session cluster transition probabilities P(j | i)')
        ax.set_xlabel('Cluster j (next)')
        ax.set_ylabel('Cluster i (current)')
        ax.set_xticks(range(self.n_clusters))
        ax.set_yticks(range(self.n_clusters))
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'transition_matrix.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # Entropy + silhouette (+ BIC/AIC for GMM) metrics
        metrics = self.entropy_metrics(variances=variances,
                                       silhouette_sample=silhouette_sample)
        with open(os.path.join(out_dir, 'entropy_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # K-sweep: silhouette + (BIC/AIC if GMM, inertia if KMeans) vs K
        if sweep_k_range is not None:
            print(f"\nSweeping K over {list(sweep_k_range)} with method={self.method}...")
            sweep = self.sweep_k(sweep_k_range, silhouette_sample=silhouette_sample)
            with open(os.path.join(out_dir, 'sweep_k.json'), 'w') as f:
                json.dump(sweep, f, indent=2)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(sweep['k'], sweep['silhouette'], 'o-', color='tab:blue')
            axes[0].set_xlabel('K (n_clusters)')
            axes[0].set_ylabel('Silhouette score')
            axes[0].set_title(f'Silhouette vs K ({self.method.upper()})')
            axes[0].grid(True, alpha=0.3)

            if self.method == 'gmm':
                axes[1].plot(sweep['k'], sweep['bic'], 'o-', label='BIC', color='tab:red')
                axes[1].plot(sweep['k'], sweep['aic'], 's--', label='AIC', color='tab:orange')
                axes[1].set_ylabel('BIC / AIC (lower is better)')
                axes[1].set_title('GMM information criteria')
                axes[1].legend()
            else:
                axes[1].plot(sweep['k'], sweep['inertia'], 'o-', color='tab:red')
                axes[1].set_ylabel('Inertia (within-cluster SS)')
                axes[1].set_title('KMeans elbow')
            axes[1].set_xlabel('K (n_clusters)')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'sweep_k.png'), dpi=200, bbox_inches='tight')
            plt.close()

        # 2D scatter if latent_dim >= 2
        if self.embeds.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 7))
            sc = ax.scatter(self.embeds[:, 0], self.embeds[:, 1], c=self.labels,
                            cmap='tab10', s=12, alpha=0.7)
            ax.set_title(f'Graph-level latent space (n_clusters={self.n_clusters})')
            ax.set_xlabel('Latent dim 0')
            ax.set_ylabel('Latent dim 1')
            plt.colorbar(sc, ax=ax, label='Cluster')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'cluster_scatter.png'), dpi=200, bbox_inches='tight')
            plt.close()

        print(f"Analysis artifacts saved to: {out_dir}")
        return out_dir

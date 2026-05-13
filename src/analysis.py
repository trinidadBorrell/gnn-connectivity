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


DEFAULT_MATRIX_PATTERN = '{matrix_dir}/sub-{subject_id}/ses-{session_num}/matrix_{matrix_idx}.npy'


class LatentClusterAnalysis:
    """Cluster graph-level latent embeddings and analyze the resulting state space."""

    def __init__(self, latent_dir, experiment_name, matrix_dir,
                 matrix_path_pattern=None, n_clusters=8,
                 subsets=('train', 'val', 'test'), random_state=42):
        self.latent_dir = latent_dir
        self.experiment_name = experiment_name
        self.matrix_dir = matrix_dir
        self.matrix_path_pattern = matrix_path_pattern or DEFAULT_MATRIX_PATTERN
        self.n_clusters = n_clusters
        self.subsets = subsets
        self.random_state = random_state

        self.embeds = None        # [N, latent_dim] graph-level (mean across nodes)
        self.meta = None          # list of {subset, subject_id, session_num, matrix_idx}
        self.labels = None        # [N] cluster id
        self.electrodes = None

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
        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        self.labels = km.fit_predict(self.embeds)
        counts = np.bincount(self.labels, minlength=self.n_clusters)
        print(f"K-Means done. Cluster sizes: {counts.tolist()}")
        return self.labels

    def _load_matrix(self, m):
        """Load the raw feature matrix referenced by one metadata entry."""
        path = self.matrix_path_pattern.format(
            matrix_dir=self.matrix_dir,
            subject_id=m['subject_id'],
            session_num=m['session_num'],
            matrix_idx=m['matrix_idx'],
        )
        return np.load(path)

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

    def entropy_metrics(self, variances=None):
        """Shannon entropy + variance-weighted variants."""
        eps = 1e-12
        counts = np.bincount(self.labels, minlength=self.n_clusters).astype(float)
        p = counts / counts.sum()
        H = float(-np.sum(p * np.log(p + eps)))

        out = {
            'shannon': H,
            'cluster_proportions': p.tolist(),
        }
        if variances is not None:
            var_arr = np.array([variances.get(c, 0.0) for c in range(self.n_clusters)])
            out['variance_weighted_sum'] = float(np.sum(p * var_arr))   # E[var | cluster] under p
            out['shannon_times_mean_variance'] = float(H * var_arr.mean())
            out['per_cluster_variance'] = var_arr.tolist()
        return out

    def save_all(self, output_dir):
        """Run the full analysis and persist all artifacts under output_dir."""
        if self.labels is None:
            self.cluster()

        out_dir = os.path.join(output_dir, 'analysis', self.experiment_name)
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

        # Entropy metrics
        metrics = self.entropy_metrics(variances=variances)
        with open(os.path.join(out_dir, 'entropy_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

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

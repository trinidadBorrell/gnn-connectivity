#!/usr/bin/env python3
"""
COOKBOOK: Post-training cluster analysis
========================================

Takes the latent JSONs saved by train.py and the raw feature matrices, then:
- Clusters graph-level latents (mean across nodes) with K-Means
- Computes per-cluster mean matrices and intra-cluster variance
- Builds the within-session Markov transition matrix
- Reports Shannon entropy and variance-weighted entropy

Example (matrix mode — preferred for round-tripping raw matrices):
    python cookbook/run_analysis.py \\
        --latent_dir output/training/latent_space \\
        --experiment_name 20260513_145721 \\
        --matrix_dir /path/to/precomputed/matrices \\
        --n_clusters 8

Example (EEG mode — point at the saved feature matrices):
    python cookbook/run_analysis.py \\
        --latent_dir output/training/latent_space \\
        --experiment_name 20260513_145721 \\
        --matrix_dir output/preprocessing/data/feature_matrix \\
        --matrix_path_pattern '{matrix_dir}/feature_matrix_sub{subject_id}_session_{session_num}_epoch{matrix_idx}_connectivity.npy' \\
        --n_clusters 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis import LatentClusterAnalysis


def main():
    parser = argparse.ArgumentParser(
        description='Cluster latent embeddings and analyze state-space dynamics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--latent_dir', type=str, required=True,
                        help='Directory containing {subset}_latent_{experiment}.json files')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Experiment name suffix used when training saved the latents')
    parser.add_argument('--matrix_dir', type=str, required=True,
                        help='Base directory containing the raw feature matrices')
    parser.add_argument('--matrix_path_pattern', type=str, default=None,
                        help='Custom path pattern; placeholders {matrix_dir} {subject_id} {session_num} {matrix_idx}')
    parser.add_argument('--n_clusters', type=int, default=8,
                        help='Number of clusters for the primary fit')
    parser.add_argument('--method', type=str, default='kmeans',
                        choices=['kmeans', 'gmm'],
                        help="Clustering method: 'kmeans' (hard assignments) or "
                             "'gmm' (Gaussian Mixture; also reports BIC/AIC)")
    parser.add_argument('--sweep_k_min', type=int, default=None,
                        help='If set with --sweep_k_max, sweeps K over '
                             '[sweep_k_min, sweep_k_max] and saves silhouette + '
                             'inertia/BIC/AIC curves')
    parser.add_argument('--sweep_k_max', type=int, default=None,
                        help='Upper bound (inclusive) for the K sweep')
    parser.add_argument('--silhouette_sample', type=int, default=20000,
                        help='Random subsample size for silhouette '
                             '(silhouette is O(N^2); set None/-1 to disable subsampling)')
    parser.add_argument('--subsets', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Which subsets to pool for clustering')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for KMeans / GMM')
    parser.add_argument('--output_dir', type=str, default='output/training',
                        help='Root output directory; results land under '
                             '{output_dir}/analysis/{experiment_name}/{method}/')
    args = parser.parse_args()

    analysis = LatentClusterAnalysis(
        latent_dir=args.latent_dir,
        experiment_name=args.experiment_name,
        matrix_dir=args.matrix_dir,
        matrix_path_pattern=args.matrix_path_pattern,
        n_clusters=args.n_clusters,
        subsets=tuple(args.subsets),
        random_state=args.random_state,
        method=args.method,
    )

    analysis.load_and_aggregate()
    analysis.cluster()

    sweep_k_range = None
    if args.sweep_k_min is not None and args.sweep_k_max is not None:
        sweep_k_range = range(args.sweep_k_min, args.sweep_k_max + 1)
    silhouette_sample = None if (args.silhouette_sample is None or args.silhouette_sample <= 0) \
        else args.silhouette_sample
    out_dir = analysis.save_all(args.output_dir, sweep_k_range=sweep_k_range,
                                silhouette_sample=silhouette_sample)

    print("\nDone. Inspect:")
    print(f"  {out_dir}/cluster_assignments.json")
    print(f"  {out_dir}/transition_matrix.png")
    print(f"  {out_dir}/entropy_metrics.json   (includes silhouette"
          + (", BIC, AIC" if args.method == 'gmm' else "") + ")")
    print(f"  {out_dir}/mean_matrix_cluster_*.npy")
    if sweep_k_range is not None:
        print(f"  {out_dir}/sweep_k.json + sweep_k.png")
    return 0


if __name__ == '__main__':
    exit(main())

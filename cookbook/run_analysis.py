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
                        help='Number of K-Means clusters')
    parser.add_argument('--subsets', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Which subsets to pool for clustering')
    parser.add_argument('--random_state', type=int, default=42,
                        help='K-Means random_state')
    parser.add_argument('--output_dir', type=str, default='output/training',
                        help='Root output directory; results land under {output_dir}/analysis/{experiment_name}/')
    args = parser.parse_args()

    analysis = LatentClusterAnalysis(
        latent_dir=args.latent_dir,
        experiment_name=args.experiment_name,
        matrix_dir=args.matrix_dir,
        matrix_path_pattern=args.matrix_path_pattern,
        n_clusters=args.n_clusters,
        subsets=tuple(args.subsets),
        random_state=args.random_state,
    )

    analysis.load_and_aggregate()
    analysis.cluster()
    out_dir = analysis.save_all(args.output_dir)

    print("\nDone. Inspect:")
    print(f"  {out_dir}/cluster_assignments.json")
    print(f"  {out_dir}/transition_matrix.png")
    print(f"  {out_dir}/entropy_metrics.json")
    print(f"  {out_dir}/mean_matrix_cluster_*.npy")
    return 0


if __name__ == '__main__':
    exit(main())

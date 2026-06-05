#!/bin/bash
#SBATCH --job-name=gnn-combined-part
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/combined_part_%j.out
#SBATCH --error=slurm/logs/combined_part_%j.err
#
# Combine the two best partition levers identified by partition_search +
# time_within_session: last_N epochs per session × diagnosis balancing.
# Sweep N ∈ {50,100,200} and K ∈ {4,6,8}.
#
# Reuses precomputed PCA from job 297759 (output/full_cluster/X_pca.npy) so
# only the GMM is refit per scheme — fast (~3 min total).
#
# Submit:  sbatch slurm/combined_partition.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"

# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/combined_partition.py \
    --cluster_dir output/full_cluster \
    --output_dir output/combined_partition \
    --random_state 42 \
    --ns 50 100 200 \
    --ks 4 6 8

echo "[$(date -Iseconds)] done"

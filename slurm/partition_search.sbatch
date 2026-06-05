#!/bin/bash
#SBATCH --job-name=gnn-partition-search
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/partition_search_%j.out
#SBATCH --error=slurm/logs/partition_search_%j.err
#
# Compare GMM K=4 clustering quality across multiple epoch-subsampling
# schemes (first/last/middle/random/all/balanced) on the already-saved X_pca
# from job 297759. Holds PCA fixed; only the GMM is refit per scheme.
#
# Submit:  sbatch slurm/partition_search.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"

# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/partition_search.py \
    --cluster_dir output/full_cluster \
    --output_dir output/partition_search \
    --K 4 \
    --random_state 42

echo "[$(date -Iseconds)] done"

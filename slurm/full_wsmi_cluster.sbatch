#!/bin/bash
#SBATCH --job-name=gnn-full-cluster
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/full_cluster_%j.out
#SBATCH --error=slurm/logs/full_cluster_%j.err
#
# Full-data raw wSMI clustering: KMeans + GMM on ALL ~100k epochs, plus
# per-cluster centroid (256x256 wSMI matrix) plotting for GMM K=4.
# Persists fitted scaler/PCA/GMM/KMeans so leave-N-subjects-out validation
# can reuse them without retraining.
#
# Submit:  sbatch slurm/full_wsmi_cluster.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"

# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/full_wsmi_cluster.py \
    --data_dir data/wsmi_res \
    --output_dir output/full_cluster \
    --pca_dim 50 \
    --random_state 42 \
    --ks 4 6 8 \
    --centroid_for_K 4

echo "[$(date -Iseconds)] done"

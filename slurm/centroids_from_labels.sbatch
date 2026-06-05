#!/bin/bash
#SBATCH --job-name=gnn-centroids
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/centroids_%j.out
#SBATCH --error=slurm/logs/centroids_%j.err
#
# Compute per-cluster wSMI centroid matrices from the cluster labels saved by
# job 297759 (or any other run of full_wsmi_cluster.py).
#
# Submit:  sbatch slurm/centroids_from_labels.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"

# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/centroids_from_labels.py \
    --data_dir data/wsmi_res \
    --cluster_dir output/full_cluster \
    --labels_file labels_gmm_K4.npy \
    --method_tag gmm_K4

echo "[$(date -Iseconds)] done"

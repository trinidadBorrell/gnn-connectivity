#!/bin/bash
#SBATCH --job-name=gnn-regional
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/regional_%j.out
#SBATCH --error=slurm/logs/regional_%j.err
#SBATCH --exclude=marg036,marg039,marg040,marg041,marg046
#
# Regional decomposition of the centroid wSMI matrices for K=3 and K=4.
# Reorders 256 electrodes into 5 anatomical regions (F/C/P/T/O), plots
# the reordered 256x256 matrices with region grid lines, AND a compact
# 5x5 region-level summary heatmap per cluster.
#
# Submit:  sbatch slurm/regional_centroids.sbatch
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== K=3 ==="
python -u scripts/regional_centroids.py \
    --coords_file data_scalp/GSN-HydroCel-257.txt \
    --centroid_dir output/centroid_last_100_balanced_K3 \
    --output_dir output/regional_centroids \
    --tag K3

echo
echo "=== K=4 ==="
python -u scripts/regional_centroids.py \
    --coords_file data_scalp/GSN-HydroCel-257.txt \
    --centroid_dir output/centroid_last_100_balanced \
    --output_dir output/regional_centroids \
    --tag K4

echo "[$(date -Iseconds)] done"

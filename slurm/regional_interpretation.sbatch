#!/bin/bash
#SBATCH --job-name=gnn-regional-interp
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/regional_interp_%j.out
#SBATCH --error=slurm/logs/regional_interp_%j.err
#SBATCH --exclude=marg036,marg039,marg040,marg041,marg046
#
# Enhanced regional interpretation plots: consciousness-sensitivity heatmap,
# control-vs-UWS direct differential, and ranked region-pair barplot.
# Reads K=3 centroids from output/centroid_last_100_balanced_K3/.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u scripts/regional_interpretation.py \
    --coords_file data_scalp/GSN-HydroCel-257.txt \
    --centroid_dir output/centroid_last_100_balanced_K3 \
    --output_dir output/regional_centroids \
    --tag K3 \
    --control_rich_cluster 1 \
    --uws_rich_cluster 0
echo "[$(date -Iseconds)] done"

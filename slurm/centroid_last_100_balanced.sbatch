#!/bin/bash
#SBATCH --job-name=gnn-centroid-best
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/centroid_best_%j.out
#SBATCH --error=slurm/logs/centroid_best_%j.err
#SBATCH --exclude=marg036,marg039,marg040,marg041,marg046
#
# Per-cluster wSMI centroid matrices for the V=0.252 winning partition
# (last_100 × balanced × GMM K=4). Reuses precomputed X_pca + meta from
# job 297759; refits GMM with random_state=42; streams .npz files to compute
# per-cluster mean (256x256) matrix.
#
# Fast: ~2-3 min total. Submit:
#   sbatch slurm/centroid_last_100_balanced.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"

# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONUNBUFFERED=1
python -u scripts/centroid_last_100_balanced.py \
    --data_dir data/wsmi_res \
    --cluster_dir output/full_cluster \
    --output_dir output/centroid_last_100_balanced \
    --K 4 \
    --last_n 100 \
    --random_state 42

echo "[$(date -Iseconds)] done"

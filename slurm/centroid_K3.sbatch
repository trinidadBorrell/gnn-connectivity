#!/bin/bash
#SBATCH --job-name=gnn-centroid-K3
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/centroid_K3_%j.out
#SBATCH --error=slurm/logs/centroid_K3_%j.err
#SBATCH --exclude=marg036,marg039,marg040,marg041,marg046
#
# Re-do the centroid plots from chapter 2 but with K=3, the clearer winner
# from chapter 3's held-out validation. Same scenario otherwise:
# last_100_per_session × balanced_by_diagnosis × GMM full covariance.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u scripts/centroid_last_100_balanced.py \
    --data_dir data/wsmi_res \
    --cluster_dir output/full_cluster \
    --output_dir output/centroid_last_100_balanced_K3 \
    --K 3 \
    --last_n 100 \
    --random_state 42
echo "[$(date -Iseconds)] done"

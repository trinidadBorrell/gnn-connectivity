#!/bin/bash
#SBATCH --job-name=gnn-holdout-K3
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/holdout_K3_%j.out
#SBATCH --error=slurm/logs/holdout_K3_%j.err
#SBATCH --exclude=marg036,marg039,marg040,marg041,marg046
#
# Held-out validation with K=3 instead of K=4. Probing the hypothesis that
# the data naturally wants 3 clusters + 1 noise sliver; K=3 should give
# cleaner centroids and similar predictive power.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u scripts/holdout_prediction.py \
    --data_dir data/wsmi_res \
    --output_dir output/holdout_K3 \
    --n_folds 5 \
    --K 3 \
    --pca_dim 50 \
    --last_n 100 \
    --reg_covar 1e-3 \
    --random_state 42
echo "[$(date -Iseconds)] done"

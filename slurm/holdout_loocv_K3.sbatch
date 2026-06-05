#!/bin/bash
#SBATCH --job-name=gnn-loocv-K3
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/holdout_loocv_K3_%j.out
#SBATCH --error=slurm/logs/holdout_loocv_K3_%j.err
#SBATCH --exclude=marg036,marg039,marg040,marg041,marg046
#
# Leave-One-Subject-Out CV with K=3 (the winner from chapter 3).
# Symmetric run alongside the K=4 LOOCV (job 298259) so the paper has
# tightest-possible held-out CIs for both K values.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "free memory at start: $(free -h | awk 'NR==2 {print $7}')"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u scripts/holdout_prediction.py \
    --data_dir data/wsmi_res \
    --output_dir output/holdout_loocv_K3 \
    --n_folds 146 \
    --K 3 \
    --pca_dim 50 \
    --last_n 100 \
    --reg_covar 1e-3 \
    --random_state 42
echo "[$(date -Iseconds)] done"

#!/bin/bash
#SBATCH --job-name=roc-gmm
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/roc_gmm_%j.out
#SBATCH --error=slurm/logs/roc_gmm_%j.err
#
# LOOCV ROC AUC for raw-PCA + GMM K=k baseline.
# Pass --K and --output_dir:
#   sbatch slurm/binary_roc_gmm.sbatch --K 3 --output_dir output/binary_roc_gmm_K3
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -u scripts/binary_roc_gmm.py \
    --data_dir data/wsmi_res \
    --last_n 100 \
    --pca_dim 50 \
    --random_state 42 \
    "$@"

echo "[$(date -Iseconds)] done"

#!/bin/bash
#SBATCH --job-name=roc-compare
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/roc_compare_%j.out
#SBATCH --error=slurm/logs/roc_compare_%j.err
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -u scripts/compare_roc.py \
    --gmm_k3 output/roc_gmm_K3_5fold \
    --gmm_k4 output/roc_gmm_K4_5fold \
    --supervised output/supervised \
    --moco output/contrastive_eval \
    --output_dir output/roc_compare \
    --n_boot 1000 \
    --random_state 42

echo "[$(date -Iseconds)] done"

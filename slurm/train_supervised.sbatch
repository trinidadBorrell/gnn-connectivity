#!/bin/bash
#SBATCH --job-name=gnn-supervised
#SBATCH --partition=parietal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/supervised_%j.out
#SBATCH --error=slurm/logs/supervised_%j.err
#
# End-to-end supervised training of the GCN encoder + 3-class head.
# Head-to-head vs raw-PCA+GMM K=3 LOOCV 0.590 baseline.
#
# Usage:
#   sbatch slurm/train_supervised.sbatch                # 5-fold default
#   sbatch slurm/train_supervised.sbatch --epochs 50    # override
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi || true
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -u src/train_supervised.py \
    --data_dir data/wsmi_res \
    --coords_file data_scalp/GSN-HydroCel-257.txt \
    --holdout_json data/holdout_subjects.json \
    --output_dir output/supervised \
    --batch_size 512 \
    --epochs 30 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --n_folds 5 \
    --k_neighbours 10 \
    --dropout 0.3 \
    --noise_std 0.02 \
    --edge_p 0.05 \
    --node_p 0.05 \
    --random_state 42 \
    "$@"

echo "[$(date -Iseconds)] done"

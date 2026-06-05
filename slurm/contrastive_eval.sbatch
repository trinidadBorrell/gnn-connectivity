#!/bin/bash
#SBATCH --job-name=gnn-cont-eval
#SBATCH --partition=parietal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/contrastive_eval_%j.out
#SBATCH --error=slurm/logs/contrastive_eval_%j.err
#
# Freeze the contrastive encoder + embed every last_100 epoch + run our
# standard GroupKFold + GMM(K=3) + LogReg harness with the 256-D embeddings.
# Defaults to 5-fold subject-wise CV (apples-to-apples vs raw-PCA baseline).
#
# Usage:
#   sbatch slurm/contrastive_eval.sbatch                # 5-fold
#   sbatch slurm/contrastive_eval.sbatch --eval_mode loocv
#   sbatch slurm/contrastive_eval.sbatch --eval_mode holdout  # only 29 unseen
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi || true
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -u scripts/contrastive_eval.py \
    --data_dir data/wsmi_res \
    --coords_file data_scalp/GSN-HydroCel-257.txt \
    --ckpt output/contrastive/encoder_q.pt \
    --output_dir output/contrastive_eval \
    --holdout_json data/holdout_subjects.json \
    --K 3 \
    --last_n 100 \
    --random_state 42 \
    "$@"

echo "[$(date -Iseconds)] done"

#!/bin/bash
#SBATCH --job-name=gnn-contrastive
#SBATCH --partition=parietal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/contrastive_%j.out
#SBATCH --error=slurm/logs/contrastive_%j.err
#
# MoCo-v2 contrastive pretraining of a graph encoder on per-epoch wSMI.
# Pretraining cohort = subjects NOT in data/holdout_subjects.json.
# Outputs encoder weights to output/contrastive/encoder_q.pt.
#
# Smoke run first (~5 min on GPU):
#     sbatch --time=00:30:00 slurm/contrastive_pretrain.sbatch --smoke
# Full run (~30 min - 2 h depending on data + epochs):
#     sbatch slurm/contrastive_pretrain.sbatch
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi || true
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -u src/train_contrastive.py \
    --data_dir data/wsmi_res \
    --coords_file data_scalp/GSN-HydroCel-257.txt \
    --holdout_json data/holdout_subjects.json \
    --output_dir output/contrastive \
    --batch_size 512 \
    --num_workers 0 \
    --epochs 50 \
    --lr 1e-3 \
    --queue_size 8192 \
    --temperature 0.10 \
    --momentum 0.99 \
    --latent_per_node 1 \
    --k_neighbours 10 \
    --noise_std 0.05 \
    --edge_p 0.10 \
    --node_p 0.15 \
    --scale_pct 0.05 \
    --random_state 42 \
    "$@"

echo "[$(date -Iseconds)] done"

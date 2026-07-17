#!/bin/bash
#SBATCH --job-name=finetune-enc
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/finetune_%j.out
#SBATCH --error=slurm/logs/finetune_%j.err
#
# Supervised classification head on a trained CEBRA encoder (see
# scripts/finetune_encoder.py). Forward --run_name / --mode / --head, e.g.:
#   sbatch slurm/finetune_encoder.sbatch --run_name wsmi256_cebra_t10_d32 --mode full   --head linear
#   sbatch slurm/finetune_encoder.sbatch --run_name wsmi256_cebra_t10_d32 --mode frozen --head mlp
#
# gpu-best has free ampere capacity; parietal is saturated. All I/O on store3.

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi || true
# shellcheck disable=SC1091
source .venv/bin/activate
python scripts/finetune_encoder.py "$@"
echo "[$(date -Iseconds)] done"

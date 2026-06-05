#!/bin/bash
#SBATCH --job-name=pc1-interp
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/pc1_interp_%j.out
#SBATCH --error=slurm/logs/pc1_interp_%j.err
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -u scripts/pc1_interpretation.py \
    --data_dir data/wsmi_res \
    --coords_file data_scalp/GSN-HydroCel-257.txt \
    --output_dir output/pc1_interpretation \
    --last_n 100 \
    --n_pca 50 \
    --top_n_pairs 100 \
    --random_state 42

echo "[$(date -Iseconds)] done"

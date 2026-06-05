#!/bin/bash
#SBATCH --job-name=gnn-holdout-loocv
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/holdout_loocv_%j.out
#SBATCH --error=slurm/logs/holdout_loocv_%j.err
#SBATCH --exclude=marg036,marg039,marg040,marg041,marg046
#
# Leave-One-Subject-Out cross-validation (144 folds — one per subject after
# dropping the 2 subjects with unknown diagnoses).
#
# Per-fold cost ~80–100s. 144 folds → ~4 hours wall + one-time ~6 min data
# load. Allocated 8 h for safety.
#
# Notes from previous failed run (298194):
#   - Bumped --mem 128G → 200G to dodge memory-pressured nodes.
#   - Excluded marg036/039/040/041/046 which had <50 GB free physical memory
#     at scheduling time (SLURM only tracks AllocMem, not actual free RAM,
#     so without the exclude it can land on a thrashed node and stall on I/O).
#   - python -u forces unbuffered stdout so progress lines flush immediately.
#
# Submit:  sbatch slurm/holdout_loocv.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "free memory at start: $(free -h | awk 'NR==2 {print $7}')"

# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONUNBUFFERED=1
python -u scripts/holdout_prediction.py \
    --data_dir data/wsmi_res \
    --output_dir output/holdout_loocv \
    --n_folds 146 \
    --K 4 \
    --pca_dim 50 \
    --last_n 100 \
    --random_state 42

echo "[$(date -Iseconds)] done"

#!/bin/bash
#SBATCH --job-name=gnn-holdout
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/holdout_%j.out
#SBATCH --error=slurm/logs/holdout_%j.err
#
# Leave-N-subjects-out validation of the raw-wSMI clustering pipeline.
# Refits scaler+PCA+GMM inside each fold (no test-fold leakage). Scores three
# diagnosis-prediction strategies (modal-cluster, hard fingerprint→logreg,
# soft posterior fingerprint→logreg) at 6-class and collapsed 3-class.
#
# Submit:  sbatch slurm/holdout_prediction.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"

# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/holdout_prediction.py \
    --data_dir data/wsmi_res \
    --output_dir output/holdout \
    --n_folds 5 \
    --K 4 \
    --pca_dim 50 \
    --last_n 100 \
    --random_state 42

echo "[$(date -Iseconds)] done"

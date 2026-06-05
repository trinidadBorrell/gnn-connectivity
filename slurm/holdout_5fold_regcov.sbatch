#!/bin/bash
#SBATCH --job-name=gnn-holdout-regcov
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/holdout_regcov_%j.out
#SBATCH --error=slurm/logs/holdout_regcov_%j.err
#SBATCH --exclude=marg036,marg039,marg040,marg041,marg046
#
# Held-out validation with K=4 but reg_covar bumped 1e-3 → 1e-2. Probes
# whether stronger covariance regularisation suppresses tiny outlier
# clusters (the 53-epoch cl_3 we saw in the centroid run) while keeping
# the four substantive modes.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u scripts/holdout_prediction.py \
    --data_dir data/wsmi_res \
    --output_dir output/holdout_regcov \
    --n_folds 5 \
    --K 4 \
    --pca_dim 50 \
    --last_n 100 \
    --reg_covar 1e-2 \
    --random_state 42
echo "[$(date -Iseconds)] done"

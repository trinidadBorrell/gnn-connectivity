#!/bin/bash
#SBATCH --job-name=gnn-time-sweep
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/time_sweep_%j.out
#SBATCH --error=slurm/logs/time_sweep_%j.err
#
# Drill into the time-within-session effect found by partition_search:
#   * 5 quintiles (where in the recording does discriminability peak?)
#   * window sizes 50/100/200/300 (first_N vs last_N — sweet spot?)
#
# Uses the precomputed PCA(50) from job 297759 — fast (~1 min).
# Submit:  sbatch slurm/time_within_session.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"

# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/time_within_session.py \
    --cluster_dir output/full_cluster \
    --output_dir output/time_within_session \
    --K 4 \
    --random_state 42

echo "[$(date -Iseconds)] done"

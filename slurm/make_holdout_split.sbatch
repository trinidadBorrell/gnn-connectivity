#!/bin/bash
#SBATCH --job-name=gnn-holdout-split
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/holdout_split_%j.out
#SBATCH --error=slurm/logs/holdout_split_%j.err
#SBATCH --exclude=marg036,marg039,marg040,marg041,marg046
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u scripts/make_holdout_split.py --frac 0.20 --random_state 42

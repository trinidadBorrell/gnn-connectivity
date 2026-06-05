#!/bin/bash
#SBATCH --job-name=gnn-gridsearch
#SBATCH --partition=parietal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/gridsearch_%j.out
#SBATCH --error=slurm/logs/gridsearch_%j.err
#
# Hyperparameter search for the GAE/VGAE on per-epoch wSMI matrices.
# Run from the repo root:
#     mkdir -p slurm/logs
#     sbatch slurm/gridsearch.sbatch [extra args forwarded to run_pipeline.py]
#
# If the parietal partition is full, retry with:  --partition=gpu-best
# Or for a quick interactive test on a single A100/A40:
#     srun --partition=parietal --gres=gpu:1 --cpus-per-task=8 --mem=64G \
#          --pty bash slurm/gridsearch.sbatch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local} cwd=$(pwd)"
nvidia-smi || true

# Activate the project's uv venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Defaults — override on the sbatch command line, e.g.
#     sbatch slurm/gridsearch.sbatch --num_samples 40 --tune_epochs 200
# Anything passed after the script name is forwarded to run_pipeline.py.
DEFAULTS=(
    --matrix_dir          data/wsmi_res
    --coordinates_file    data_scalp/GSN-HydroCel-257.txt
    --coordinate_system   file
    --no_plot_neighbors
    --tuning
    --num_samples         20
    --tune_epochs         150
    --tune_grace_period   20
    --n_epochs            200
    --batch_size          128
    --patience            30
    --output_dir          output/gridsearch
    --experiment_name     "gs_$(date +%Y%m%d_%H%M%S)"
)

python cookbook/run_pipeline.py "${DEFAULTS[@]}" "$@"

echo "[$(date -Iseconds)] done"

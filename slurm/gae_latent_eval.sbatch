#!/bin/bash
#SBATCH --job-name=gae-latent-eval
#SBATCH --partition=parietal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/gae_latent_eval_%j.out
#SBATCH --error=slurm/logs/gae_latent_eval_%j.err
#
# Score a trained src/model.py GNN on the raw-wSMI baseline's held-out
# protocol (last_n x balanced x GroupKFold x GMM-K3) so its bal_acc / AUC is
# head-to-head comparable with the 0.611 floor. See docs/lab-notebook/chapter_07.
#
# CPU-only: latent extraction on 64-node graphs + GMM/logreg is cheap. Add
# --gres=gpu:ampere:1 only if you point it at a heavy time-series model.
#
# Run from the repo root (NOT the login node — submit with sbatch):
#     mkdir -p slurm/logs
#     sbatch slurm/gae_latent_eval.sbatch \
#         --run_name wsmi_full --model_kind gae --eval_mode 5fold
#
# Anything after the script name is forwarded to gae_latent_eval.py.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local} cwd=$(pwd)"

# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/gae_latent_eval.py --cpu "$@"

echo "[$(date -Iseconds)] done"

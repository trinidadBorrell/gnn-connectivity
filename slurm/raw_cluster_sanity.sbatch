#!/bin/bash
#SBATCH --job-name=gnn-raw-sanity
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/raw_sanity_%j.out
#SBATCH --error=slurm/logs/raw_sanity_%j.err
#
# Raw wSMI clustering sanity check: does the unprocessed wSMI matrix already
# carry diagnostic signal (UWS / MCS- / MCS+ / EMCS / control)?
# Runs KMeans, GMM, Spectral, Louvain on the PCA-reduced flattened wSMI of
# ~17,800 subsampled epochs. CPU-only, ~30 min.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"

# shellcheck disable=SC1091
source .venv/bin/activate

# Force OMP / BLAS to stay within the sbatch CPU allotment
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python scripts/raw_wsmi_cluster_sanity.py \
    --data_dir data/wsmi_res \
    --output_dir output/raw_cluster_sanity \
    --epochs_per_session 100 \
    --pca_dim 50 \
    --ks 4 6 8

echo "[$(date -Iseconds)] done"

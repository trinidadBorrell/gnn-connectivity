#!/bin/bash
#SBATCH --job-name=gnn-gmm-smoke
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/gmm_smoke_%j.out
#SBATCH --error=slurm/logs/gmm_smoke_%j.err
#
# Validate the new GMM + silhouette path in src/analysis.py end-to-end.
# Runs the analysis twice on the smoke-test latents:
#   1. KMeans  (no sweep, K=4)
#   2. GMM     (with K-sweep 2..8 + BIC/AIC)
# CPU-only; no GPU needed. Targets the normal-best partition.
#
# Submit:  sbatch slurm/analysis_gmm_smoke.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"

# shellcheck disable=SC1091
source .venv/bin/activate

LATENT_DIR=output/smoke_new/training/latent_space
MATRIX_DIR=output/smoke_new/preprocessing/wsmi_npy_cache
EXP=smoke_new
OUT=output/smoke_new/training

echo "=== KMeans (K=4) ==="
python cookbook/run_analysis.py \
    --latent_dir "$LATENT_DIR" \
    --experiment_name "$EXP" \
    --matrix_dir "$MATRIX_DIR" \
    --matrix_path_pattern '{matrix_dir}/sub-{subject_id}/ses-{session_num}/data.npy' \
    --n_clusters 4 \
    --method kmeans \
    --silhouette_sample 5000 \
    --output_dir "$OUT"

echo
echo "=== GMM with K-sweep 2..8 ==="
python cookbook/run_analysis.py \
    --latent_dir "$LATENT_DIR" \
    --experiment_name "$EXP" \
    --matrix_dir "$MATRIX_DIR" \
    --matrix_path_pattern '{matrix_dir}/sub-{subject_id}/ses-{session_num}/data.npy' \
    --n_clusters 6 \
    --method gmm \
    --sweep_k_min 2 --sweep_k_max 8 \
    --silhouette_sample 5000 \
    --output_dir "$OUT"

echo "[$(date -Iseconds)] done"

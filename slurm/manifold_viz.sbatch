#!/bin/bash
#SBATCH --job-name=gnn-manifold-viz
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/manifold_viz_%j.out
#SBATCH --error=slurm/logs/manifold_viz_%j.err
#SBATCH --exclude=marg036,marg039,marg040,marg041,marg046
#
# Manifold visualisations: PCA(2) + UMAP(2) of the 17.5k last_100 epochs,
# coloured by cluster and by diagnosis. Overlays the GMM Gaussians (in PCA)
# and the diagnosis-centroid trajectory (control→EMCS→MCS+→MCS−→UWS→COMA)
# in both projections. Per-subject summary + 1-D "consciousness axis"
# projection from control to UWS, histogrammed by diagnosis.
#
# Submit:  sbatch slurm/manifold_viz.sbatch
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u scripts/manifold_viz.py \
    --data_dir data/wsmi_res \
    --cluster_dir output/full_cluster \
    --centroid_dir output/centroid_last_100_balanced \
    --output_dir output/manifold_viz \
    --last_n 100 \
    --random_state 42
echo "[$(date -Iseconds)] done"

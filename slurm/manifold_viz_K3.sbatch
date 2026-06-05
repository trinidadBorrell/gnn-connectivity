#!/bin/bash
#SBATCH --job-name=gnn-manifold-K3
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/manifold_K3_%j.out
#SBATCH --error=slurm/logs/manifold_K3_%j.err
#SBATCH --exclude=marg036,marg039,marg040,marg041,marg046
#SBATCH --dependency=singleton-or-replace
#
# K=3 versions of pca2_diagnosis, pca2_clusters, umap2_*, consciousness_axis.
# Depends on labels written by the K=3 centroid job (which writes
# output/centroid_last_100_balanced_K3/labels_last_100.npy).
#
# The manifold_viz.py script reads K from those labels (K = labels.max()+1)
# so no code change needed; only the --centroid_dir + --output_dir differ.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u scripts/manifold_viz.py \
    --data_dir data/wsmi_res \
    --cluster_dir output/full_cluster \
    --centroid_dir output/centroid_last_100_balanced_K3 \
    --output_dir output/manifold_viz_K3 \
    --last_n 100 \
    --random_state 42
echo "[$(date -Iseconds)] done"

#!/bin/bash
#SBATCH --job-name=wsmi256-train
#SBATCH --partition=parietal
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/wsmi256_train_%j.out
#SBATCH --error=slurm/logs/wsmi256_train_%j.err
#
# Train GNN encoders on the 256-electrode wSMI data (the SAME data as the
# raw-wSMI baseline) so their latents are head-to-head comparable via
# scripts/gae_latent_eval.py. See docs/lab-notebook/chapter_07.
#
# Node features = 256x256 wSMI rows; k-NN graph on GSN-HydroCel-257 coords.
# --max_epochs_per_recording 100 keeps the last 100 epochs/session (matches the
# baseline's last_100 window AND bounds RAM: the full 256 set is ~68 GB).
#
# Run from the repo root; forward --models / --loss / --run_name, e.g.:
#   mkdir -p slurm/logs
#   sbatch slurm/train_wsmi256.sbatch --models gae vgae --loss mse --run_name wsmi256_recon_a
#   sbatch slurm/train_wsmi256.sbatch --models enc_gae_fc --loss cebra --run_name wsmi256_cebra
#
# If parietal is full, retry with --partition=gpu-best (may queue for hours).

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local} cwd=$(pwd)"
nvidia-smi || true

# shellcheck disable=SC1091
source .venv/bin/activate

LABELS=data/wsmi_res/256_electrodes-20260526T091011Z-3-001/256_electrodes/wsmi_res_DOC/patient_labels.csv

DEFAULTS=(
    --stage             all
    --input_mode        wsmi
    --wsmi_format       npz
    --coords_file       data_scalp/GSN-HydroCel-257.txt
    --patient_dir       data/wsmi_res
    --control_dir       data/wsmi_res
    --diagnosis_csv     "$LABELS"
    --max_epochs_per_recording 100
    --num_trials        12
    --tune_epochs       40
    --cpus_per_trial    4
    --train_epochs      120
    --train_patience    20
    --output_root       output
)

python cookbook/run_wsmi_pipeline.py "${DEFAULTS[@]}" "$@"

echo "[$(date -Iseconds)] done"

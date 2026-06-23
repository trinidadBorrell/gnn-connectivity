#!/usr/bin/env bash
# Sequential experiment matrix for the GNN-connectivity pipeline.
#
#   GAE_VAE (full Ray Tune), 2-sigma outlier filter on:
#     wsmi  rs  patients+controls  mse        coarse,fine
#     wsmi  rs  patients-only      mse        coarse,fine   (NEW)
#     wsmi  lg  patients-only      mse        coarse,fine
#     wsmi  lg  patients+controls  mse        coarse,fine   (control-inclusive lg)
#     ts    rs  patients-only      mse_corr   coarse,fine
#     ts    lg  patients-only      mse_corr   coarse,fine
#     ts    lg  patients+controls  mse_corr   coarse,fine   (control-inclusive lg)
#   enc_gae_fc / CEBRA (small grid tune), outlier filter on:
#     ts    lg  patients-only      cebra      coarse,fine
#     ts    lg  patients+controls  cebra      coarse,fine   (NEW)
#     ts    rs  patients-only      cebra      coarse,fine   (NEW)
#     ts    rs  patients+controls  cebra      coarse,fine   (NEW)
#
# Control-inclusive runs drop --patients_only so the pipeline auto-resolves the
# control dirs from DATA_DEFAULTS[type_data] in run_wsmi_pipeline.py. Control lg/rs
# use healthy-control data recombined to biosemi64 (lg also resampled to 100 Hz);
# see preprocessing/01_recombine_egi256_to_biosemi64.py + 04_run_wsmi_theta.py.
# Timeseries window: both rs and lg crop to -0.2..0.6 s (0.8 s); it depends only on
# the data type, not the model (handled in run_wsmi_pipeline.py, not per-run flags).
#
# Runs sequentially, each logged to output/<run>/console.log, and is RESUMABLE:
# a run whose final test_report.json already exists is skipped.
#
# Usage:
#   bash gnn_connectivity/cookbook/run_experiments.sh   # run everything (from repo root)
#   DRYRUN=1   bash ... run_experiments.sh   # just print the commands, run nothing
#   CPU=1      bash ... run_experiments.sh   # force CPU (default: MPS if available)
#
# MEMORY (important on <=32 GB machines — each run is a separate process, so RAM is
# released between runs; the risk is the PER-RUN peak on the big lg datasets):
#   MAX_CONCURRENT=N  cap simultaneous Ray Tune trials (DEFAULT 2). Biggest lever:
#                     peak RAM ~ dataset*(1+N). Set 1 for the tightest memory.
#   MAX_EPOCHS=N      cap epochs/recording (subsamples lg; 150-300 is plenty for tuning).
#   NUM_TRIALS=N      fewer ASHA samples (default 30) -> less time, slightly less RAM.
#   CPUS_PER_TRIAL=N  more cpus/trial -> fewer parallel trials (another way to serialize).
#   THREADS=N         cap BLAS/OMP/torch threads per process so the CPU is not maxed
#                     (e.g. THREADS=6 on a 12-core machine leaves ~half the cores free).
#   TRIM_GRAPHS=1     delete each run's splits/graphs.pt after it completes, so disk
#                     stays bounded across the matrix (lg graphs.pt is ~1.4 GB each).
#                     Re-running a trimmed run's cluster/latents stage needs a reload.
#
#   # Recommended memory-safe full run on a 24 GB Mac (all data, just slower):
#   MAX_CONCURRENT=1 MAX_EPOCHS=200 bash gnn_connectivity/cookbook/run_experiments.sh
#
# Env vars combine, e.g.:  DRYRUN=1 MAX_CONCURRENT=1 MAX_EPOCHS=200 bash run_experiments.sh
# The script activates conda env `gnn_env` and cd's to the repo root itself, so it
# can be launched from anywhere. Re-running resumes: completed runs are skipped.
set -uo pipefail

# --- locate repo root (two levels up from this script) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
PIPE="gnn_connectivity/cookbook/run_wsmi_pipeline.py"
OUT_ROOT="gnn_connectivity/output"

# --- activate conda env gnn_env ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gnn_env

DRYRUN="${DRYRUN:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-}"             # empty = use all epochs (full lg)
# --- memory controls (24 GB machine: keep concurrency low) ---
# Each concurrent Ray Tune trial deserializes its OWN copy of the dataset, so
# peak RAM ~ dataset*(1 + MAX_CONCURRENT). Default 2 -> ~3x dataset, not ~7x.
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"   # cap simultaneous tuning trials (0/empty = unlimited)
NUM_TRIALS="${NUM_TRIALS:-}"            # empty = pipeline default (30)
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-}"    # empty = pipeline default (2)
# --- CPU headroom: cap BLAS/OMP/torch threads per process so the machine stays
# usable. Empty = use all cores. With MAX_CONCURRENT=1, total busy cores ~= THREADS.
THREADS="${THREADS:-}"
if [ -n "$THREADS" ]; then
  export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
         OPENBLAS_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS" \
         NUMEXPR_NUM_THREADS="$THREADS"
fi

CPU_FLAG=""
[ "${CPU:-0}" = "1" ] && CPU_FLAG="--cpu"
CAP_FLAG=""
[ -n "$MAX_EPOCHS" ] && CAP_FLAG="--max_epochs_per_recording $MAX_EPOCHS"
MEM_FLAGS=""
[ -n "$MAX_CONCURRENT" ] && [ "$MAX_CONCURRENT" != "0" ] && MEM_FLAGS="$MEM_FLAGS --max_concurrent_trials $MAX_CONCURRENT"
[ -n "$NUM_TRIALS" ]     && MEM_FLAGS="$MEM_FLAGS --num_trials $NUM_TRIALS"
[ -n "$CPUS_PER_TRIAL" ] && MEM_FLAGS="$MEM_FLAGS --cpus_per_trial $CPUS_PER_TRIAL"

# run <name> <model> <test_subdir> <extra args...>
run () {
  local name="$1"; shift
  local test_report="$OUT_ROOT/$name/final_test/$1/test_report.json"; shift
  local logdir="$OUT_ROOT/$name"
  if [ -f "$test_report" ]; then
    echo "[skip] $name (already complete: $test_report)"
    return 0
  fi
  mkdir -p "$logdir"
  local cmd="python $PIPE --stage all --filter_outliers --run_name $name $CPU_FLAG $CAP_FLAG $MEM_FLAGS $*"
  echo "=================================================================="
  echo "[run ] $name"
  echo "       $cmd"
  if [ "$DRYRUN" = "1" ]; then return 0; fi
  # shellcheck disable=SC2086
  $cmd > "$logdir/console.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] $name (exit $rc) — see $logdir/console.log"
  else
    echo "[done] $name"
    # Keep disk bounded across the matrix: the cached dataset (splits/graphs.pt) is
    # only needed within a run; drop it once the run finished successfully.
    [ "${TRIM_GRAPHS:-0}" = "1" ] && rm -f "$logdir/splits/graphs.pt"
  fi
  return 0   # keep going to the next run regardless
}

GV="--models gae_vae"            # decoder autoencoder
CB="--models enc_gae_fc --loss cebra --cebra_tune"

# ---- GAE_VAE : MSE on wSMI ----
run gaevae_rs_wsmi_mse_coarse gae_vae  $GV --input_mode wsmi --type_data rs --loss mse --diagnosis_granularity coarse
run gaevae_rs_wsmi_mse_fine   gae_vae  $GV --input_mode wsmi --type_data rs --loss mse --diagnosis_granularity fine
run gaevae_rs_wsmi_mse_coarse_ponly gae_vae  $GV --input_mode wsmi --type_data rs --loss mse --diagnosis_granularity coarse --patients_only
run gaevae_rs_wsmi_mse_fine_ponly   gae_vae  $GV --input_mode wsmi --type_data rs --loss mse --diagnosis_granularity fine   --patients_only
run gaevae_lg_wsmi_mse_coarse gae_vae  $GV --input_mode wsmi --type_data lg --loss mse --diagnosis_granularity coarse --patients_only
run gaevae_lg_wsmi_mse_fine   gae_vae  $GV --input_mode wsmi --type_data lg --loss mse --diagnosis_granularity fine   --patients_only

# ---- GAE_VAE : MSE on wSMI, local-global WITH controls (drop --patients_only) ----
run gaevae_lg_wsmi_mse_coarse_withctrl gae_vae  $GV --input_mode wsmi --type_data lg --loss mse --diagnosis_granularity coarse
run gaevae_lg_wsmi_mse_fine_withctrl   gae_vae  $GV --input_mode wsmi --type_data lg --loss mse --diagnosis_granularity fine

# ---- GAE_VAE : MSE + corr on time-series (patients-only) ----
run gaevae_rs_ts_corr_coarse  gae_vae  $GV --input_mode timeseries --type_data rs --loss mse_corr --diagnosis_granularity coarse --patients_only
run gaevae_rs_ts_corr_fine    gae_vae  $GV --input_mode timeseries --type_data rs --loss mse_corr --diagnosis_granularity fine   --patients_only
run gaevae_lg_ts_corr_coarse  gae_vae  $GV --input_mode timeseries --type_data lg --loss mse_corr --diagnosis_granularity coarse --patients_only
run gaevae_lg_ts_corr_fine    gae_vae  $GV --input_mode timeseries --type_data lg --loss mse_corr --diagnosis_granularity fine   --patients_only

# ---- GAE_VAE : time-series, local-global WITH controls (drop --patients_only) ----
run gaevae_lg_ts_corr_coarse_withctrl gae_vae  $GV --input_mode timeseries --type_data lg --loss mse_corr --diagnosis_granularity coarse
run gaevae_lg_ts_corr_fine_withctrl   gae_vae  $GV --input_mode timeseries --type_data lg --loss mse_corr --diagnosis_granularity fine

# ---- enc_gae_fc / CEBRA : local-global, patients-only ----
run cebra_lg_coarse  enc_gae_fc  $CB --input_mode timeseries --type_data lg --diagnosis_granularity coarse --patients_only
run cebra_lg_fine    enc_gae_fc  $CB --input_mode timeseries --type_data lg --diagnosis_granularity fine   --patients_only

# ---- enc_gae_fc / CEBRA : local-global WITH controls (drop --patients_only) ----
run cebra_lg_coarse_withctrl  enc_gae_fc  $CB --input_mode timeseries --type_data lg --diagnosis_granularity coarse
run cebra_lg_fine_withctrl    enc_gae_fc  $CB --input_mode timeseries --type_data lg --diagnosis_granularity fine

# ---- enc_gae_fc / CEBRA : resting-state, patients-only ----
# rs & lg both crop to -0.2..0.6 s (0.8 s window) in run_wsmi_pipeline.py — no
# per-run window override needed; the window depends only on the data type.
run cebra_rs_coarse_ponly  enc_gae_fc  $CB --input_mode timeseries --type_data rs --diagnosis_granularity coarse --patients_only
run cebra_rs_fine_ponly    enc_gae_fc  $CB --input_mode timeseries --type_data rs --diagnosis_granularity fine   --patients_only

# ---- enc_gae_fc / CEBRA : resting-state WITH controls (drop --patients_only) ----
run cebra_rs_coarse_withctrl  enc_gae_fc  $CB --input_mode timeseries --type_data rs --diagnosis_granularity coarse
run cebra_rs_fine_withctrl    enc_gae_fc  $CB --input_mode timeseries --type_data rs --diagnosis_granularity fine

echo "=================================================================="
echo "All runs dispatched. Outputs under $OUT_ROOT/<run_name>/"

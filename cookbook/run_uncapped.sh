#!/usr/bin/env bash
# Full experiment matrix, UNCAPPED — every recording uses ALL its epochs.
#
# Background (2026-06-12): the matrix was previously launched with MAX_EPOCHS=100,
# which set --max_epochs_per_recording 100 and randomly SUBSAMPLED each recording
# to 100 epochs. That cap was applied inconsistently: only 4 runs carry it
# (gaevae_rs_wsmi_mse_coarse, gaevae_lg_wsmi_mse_coarse, gaevae_lg_wsmi_mse_fine,
# gaevae_lg_wsmi_mse_coarse_withctrl); the rest were already run with all epochs.
#
# This queue re-runs EVERYTHING with no cap, but is smart + resumable:
#   * SKIP a run only if it is complete AND its args.json shows NO cap
#     (max_epochs_per_recording == null) -> already a valid uncapped result.
#   * Otherwise WIPE its output dir (capped result or killed/partial run) and
#     re-run it fresh uncapped.
# So a first invocation re-runs the 4 capped runs + the 9 never-run lg runs and
# skips the 9 already-uncapped ones; later invocations skip everything done.
#
# NO --max_epochs_per_recording flag is passed (default None = all epochs). This
# raises peak RAM/runtime on the lg datasets, so concurrency is pinned to 1
# (peak RAM ~ 2x dataset). Each run is a separate process, so RAM is released
# between runs.
#
# Usage:
#   bash gnn_connectivity/cookbook/run_uncapped.sh
#   DRYRUN=1 bash gnn_connectivity/cookbook/run_uncapped.sh   # print, run nothing
#
# Tunables (env vars, all optional):
#   MAX_CONCURRENT=N  simultaneous Ray Tune trials (DEFAULT 1; peak RAM ~ ds*(1+N))
#   THREADS=N         BLAS/OMP/torch threads per process (DEFAULT 6)
#   CPU=1             force CPU instead of MPS
#   MAX_EPOCHS=N      re-introduce a cap (DEFAULT empty = use ALL epochs)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
PIPE="gnn_connectivity/cookbook/run_wsmi_pipeline.py"
OUT_ROOT="gnn_connectivity/output"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gnn_env

DRYRUN="${DRYRUN:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-}"            # empty = use ALL epochs (no subsampling)
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"  # 1 = tightest memory (~2x dataset peak)
THREADS="${THREADS:-6}"
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
[ -n "$MAX_CONCURRENT" ] && [ "$MAX_CONCURRENT" != "0" ] && MEM_FLAGS="--max_concurrent_trials $MAX_CONCURRENT"

# Is <dir> a COMPLETE result produced with NO epoch cap?
uncapped_done () {
  local dir="$1"
  ls "$dir"/final_test/*/test_report.json >/dev/null 2>&1 || return 1
  [ -f "$dir/args.json" ] || return 1
  local cap
  cap=$(python -c "import json;print(json.load(open('$dir/args.json')).get('max_epochs_per_recording'))" 2>/dev/null)
  [ "$cap" = "None" ]
}

# run <name> <model> <test_subdir> <extra args...>
run () {
  local name="$1"; shift
  shift                                   # <test_subdir> no longer needed for skip
  local logdir="$OUT_ROOT/$name"
  if uncapped_done "$logdir"; then
    echo "[skip] $name (already complete, uncapped)"
    return 0
  fi
  if [ -d "$logdir" ]; then
    echo "[clean] $name (removing capped/partial result)"
    [ "$DRYRUN" = "1" ] || rm -rf "$logdir"
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
  fi
  return 0
}

GV="--models gae_vae"
CB="--models enc_gae_fc --loss cebra --cebra_tune"

# ---- GAE_VAE : MSE on wSMI ----
run gaevae_rs_wsmi_mse_coarse gae_vae  $GV --input_mode wsmi --type_data rs --loss mse --diagnosis_granularity coarse
run gaevae_rs_wsmi_mse_fine   gae_vae  $GV --input_mode wsmi --type_data rs --loss mse --diagnosis_granularity fine
run gaevae_rs_wsmi_mse_coarse_ponly gae_vae  $GV --input_mode wsmi --type_data rs --loss mse --diagnosis_granularity coarse --patients_only
run gaevae_rs_wsmi_mse_fine_ponly   gae_vae  $GV --input_mode wsmi --type_data rs --loss mse --diagnosis_granularity fine   --patients_only
run gaevae_lg_wsmi_mse_coarse gae_vae  $GV --input_mode wsmi --type_data lg --loss mse --diagnosis_granularity coarse --patients_only
run gaevae_lg_wsmi_mse_fine   gae_vae  $GV --input_mode wsmi --type_data lg --loss mse --diagnosis_granularity fine   --patients_only

# ---- GAE_VAE : MSE on wSMI, local-global WITH controls ----
run gaevae_lg_wsmi_mse_coarse_withctrl gae_vae  $GV --input_mode wsmi --type_data lg --loss mse --diagnosis_granularity coarse
run gaevae_lg_wsmi_mse_fine_withctrl   gae_vae  $GV --input_mode wsmi --type_data lg --loss mse --diagnosis_granularity fine

# ---- GAE_VAE : MSE + corr on time-series (patients-only) ----
run gaevae_rs_ts_corr_coarse  gae_vae  $GV --input_mode timeseries --type_data rs --loss mse_corr --diagnosis_granularity coarse --patients_only
run gaevae_rs_ts_corr_fine    gae_vae  $GV --input_mode timeseries --type_data rs --loss mse_corr --diagnosis_granularity fine   --patients_only
run gaevae_lg_ts_corr_coarse  gae_vae  $GV --input_mode timeseries --type_data lg --loss mse_corr --diagnosis_granularity coarse --patients_only
run gaevae_lg_ts_corr_fine    gae_vae  $GV --input_mode timeseries --type_data lg --loss mse_corr --diagnosis_granularity fine   --patients_only

# ---- GAE_VAE : time-series, local-global WITH controls ----
run gaevae_lg_ts_corr_coarse_withctrl gae_vae  $GV --input_mode timeseries --type_data lg --loss mse_corr --diagnosis_granularity coarse
run gaevae_lg_ts_corr_fine_withctrl   gae_vae  $GV --input_mode timeseries --type_data lg --loss mse_corr --diagnosis_granularity fine

# ---- enc_gae_fc / CEBRA : local-global, patients-only ----
run cebra_lg_coarse  enc_gae_fc  $CB --input_mode timeseries --type_data lg --diagnosis_granularity coarse --patients_only
run cebra_lg_fine    enc_gae_fc  $CB --input_mode timeseries --type_data lg --diagnosis_granularity fine   --patients_only

# ---- enc_gae_fc / CEBRA : local-global WITH controls ----
run cebra_lg_coarse_withctrl  enc_gae_fc  $CB --input_mode timeseries --type_data lg --diagnosis_granularity coarse
run cebra_lg_fine_withctrl    enc_gae_fc  $CB --input_mode timeseries --type_data lg --diagnosis_granularity fine

# ---- enc_gae_fc / CEBRA : resting-state, patients-only ----
run cebra_rs_coarse_ponly  enc_gae_fc  $CB --input_mode timeseries --type_data rs --diagnosis_granularity coarse --patients_only
run cebra_rs_fine_ponly    enc_gae_fc  $CB --input_mode timeseries --type_data rs --diagnosis_granularity fine   --patients_only

# ---- enc_gae_fc / CEBRA : resting-state WITH controls ----
run cebra_rs_coarse_withctrl  enc_gae_fc  $CB --input_mode timeseries --type_data rs --diagnosis_granularity coarse
run cebra_rs_fine_withctrl    enc_gae_fc  $CB --input_mode timeseries --type_data rs --diagnosis_granularity fine

echo "=================================================================="
echo "All uncapped runs dispatched. Outputs under $OUT_ROOT/<run_name>/"

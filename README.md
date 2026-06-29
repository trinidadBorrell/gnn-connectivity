# GNN Connectivity Pipeline

A Graph Neural Network (GNN) pipeline for EEG / connectivity analysis. Trains
a graph autoencoder (or a contrastive encoder) on per-epoch biosemi64 graphs,
supports two input modalities (pre-computed **wSMI** connectivity matrices or
**raw per-electrode time-series**), four model architectures (**GAE**, **VGAE**,
**GAEVAE**, **enc_gae_fc**), three training objectives (**mse**, **mse_corr**,
**cebra**/InfoNCE), **Ray Tune ASHA** hyperparameter search, **stratified
subject-level splits**, and a full downstream suite of clustering, LOSO decoding,
and latent-space diagnostics — all from a single entry point.

The canonical entry point is [cookbook/run_wsmi_pipeline.py](cookbook/run_wsmi_pipeline.py).
The older [cookbook/run_pipeline.py](cookbook/run_pipeline.py) is kept for
backward compatibility; new work should use `run_wsmi_pipeline.py`.

## Installation

```bash
pip install -r requirements.txt
```

## Modes & flags overview

Every dimension of variation lives behind one CLI flag. Combine them freely.

| Dimension | Flag | Choices | Default |
|---|---|---|---|
| Input modality | `--input_mode` | `wsmi`, `timeseries` | `wsmi` |
| Task | `--type_data` | `rs`, `lg` | `rs` |
| RS window | `--rs_duration` | `0.8s`, `16s` (rs only; lg is always 0.8 s) | `0.8s` |
| Model architecture | `--models` | any subset of `gae`, `vgae`, `gae_vae`; **or** `enc_gae_fc` (alone, with `--loss cebra`) | `gae vgae` |
| Training objective | `--loss` | `mse`, `mse_corr`, `cebra` | `mse` |
| Label granularity | `--diagnosis_granularity` | `coarse` (CONTROL/UWS/MCS), `fine` (raw `diagnostic_crs_final`) | `coarse` |
| Cohort | `--patients_only` | flag (omit → controls included, **lg supported**) | off |
| Data root | `--data_root` | prefix joined onto the per-variant `DATA_DEFAULTS` subpaths | `data` |
| Correlation regularizer | `--lambda_corr` | float, `None` = use tuned value | `None` |
| Time-series window | `--window_sec` | seconds | `16.0` |
| Time-series sampling rate | `--sfreq` | Hz | `100.0` |
| Stage | `--stage` | `load`, `tune`, `train`, `test`, `latents`, `cluster`, `all` | `all` |

`--loss` is coupled to `--models`: `mse`/`mse_corr` need a **decoder**
(`gae`/`vgae`/`gae_vae`); `cebra` needs the **encoder-only** `enc_gae_fc`. Both
`mse_corr` and `cebra` run in either input modality (wsmi or timeseries).

## Run recipes

All commands are runnable from the **repository root** (the directory that
contains both `gnn_connectivity/` and `data/`). Substitute `python` for
`gnn_connectivity/.venv/bin/python` if you're using a different env. On the
cluster, point `--data_root` / `--output_root` at the shared data trees instead
of the repo-relative defaults (see the Condor section).

**1. wSMI matrices, all subjects, GAE + VGAE** (most common):
```bash
gnn_connectivity/.venv/bin/python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --stage all --models gae vgae \
    --num_trials 30 --tune_epochs 80 \
    --train_epochs 200 --train_patience 25 \
    --run_name wsmi_full
```

**2. wSMI matrices, DoC patients only** (no controls — UWS vs MCS only):
```bash
gnn_connectivity/.venv/bin/python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --stage all --models gae vgae \
    --patients_only \
    --run_name wsmi_patients_only
```

**3. wSMI matrices, all three architectures** (adds GAEVAE):
```bash
gnn_connectivity/.venv/bin/python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --stage all --models gae vgae gae_vae \
    --run_name wsmi_all_models
```

**4. Time-series mode, all subjects** (raw per-electrode signal as node feature):
```bash
gnn_connectivity/.venv/bin/python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --stage all --models gae vgae \
    --input_mode timeseries \
    --timeseries_patient_dir /path/to/patient/fif/root \
    --timeseries_control_dir /path/to/control/fif/root \
    --sfreq 100 --window_sec 16 \
    --run_name ts_full
```

**5. Time-series mode, patients only**:
```bash
gnn_connectivity/.venv/bin/python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --stage all --models gae_vae \
    --input_mode timeseries \
    --timeseries_patient_dir /path/to/patient/fif/root \
    --patients_only \
    --sfreq 100 --window_sec 16 \
    --run_name ts_patients_only
```

**6. Local-global, controls + patients** (`--type_data lg`, controls auto-resolved):
```bash
gnn_connectivity/.venv/bin/python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --stage all --models gae_vae \
    --input_mode wsmi --type_data lg \
    --run_name wsmi_lg_withctrl
```
Add `--patients_only` to exclude controls. Omitting `--patient_dir`/`--control_dir` lets the pipeline
resolve them from `DATA_DEFAULTS["lg"]` (the `wsmi_theta_lg/` patient + control trees).

**7. Smoke test** (~30 s; tiny config, plumbing check only):
```bash
gnn_connectivity/.venv/bin/python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --smoke --models gae --cpus_per_trial 1 \
    --run_name wsmi_smoke
```

### Batch experiment matrix

[cookbook/run_experiments.sh](cookbook/run_experiments.sh) runs the full sequential matrix
(GAE_VAE + CEBRA across rs/lg × wsmi/timeseries × coarse/fine, including the new
control-inclusive `*_withctrl` lg runs). It activates conda env `gnn_env`, cd's to the repo root,
logs each run to `output/<run>/console.log`, and is **resumable** (completed runs are skipped):
```bash
bash gnn_connectivity/cookbook/run_experiments.sh            # run everything
DRYRUN=1 bash gnn_connectivity/cookbook/run_experiments.sh   # print commands only
MAX_EPOCHS=300 bash gnn_connectivity/cookbook/run_experiments.sh   # cap epochs/recording (lg memory/speed)
CPU=1 bash gnn_connectivity/cookbook/run_experiments.sh      # force CPU
```

[cookbook/run_uncapped.sh](cookbook/run_uncapped.sh) is a variant that runs the
matrix with no epoch cap (all windows of every recording).

### Condor: one job per combination (168 jobs)

For the cluster, [cookbook/condor/](cookbook/condor/) fans the **full** grid out
to **one HTCondor job per combination** — architecture × data-variant
(`rs08`/`rs16`/`lg`) × labels × cohort × input × loss, with the impossible
`cebra ⟺ enc_gae_fc` couplings pruned (168 jobs). Generate the manifest, then
submit:

```bash
python gnn_connectivity/cookbook/condor/gen_job_matrix.py
condor_submit -dry-run:- gnn_connectivity/cookbook/condor/gnn_experiments.submit  # verify
condor_submit gnn_connectivity/cookbook/condor/gnn_experiments.submit
```

Re-submission is resumable (a run with its `final_test/<arch>/test_report.json`
is skipped). See [cookbook/condor/README.md](cookbook/condor/README.md).

**7. Re-cluster an already-trained model** (skips load/tune/train):
```bash
gnn_connectivity/.venv/bin/python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --stage cluster --models gae vgae \
    --run_name wsmi_full
```

**8. Run individual stages**:
```bash
# (a) load + split
... --stage load --run_name wsmi_full
# (b) hyperparameter search
... --stage tune --models gae vgae --num_trials 30 --tune_epochs 80 --run_name wsmi_full
# (c) final training
... --stage train --models gae vgae --train_epochs 200 --run_name wsmi_full
# (d) held-out test
... --stage test --models gae vgae --run_name wsmi_full
# (e) clustering + downstream metrics
... --stage cluster --models gae vgae --run_name wsmi_full
```

## What the stages do

| `--stage` | What runs | What lands on disk |
|---|---|---|
| `load` | Reads the patient (and, unless `--patients_only`, control) folders, builds one Data graph per epoch, looks up `diagnostic_crs_final` from the CSV, runs the **stratified 70/15/15 subject-level split**, computes train-only normalization stats and applies them in place. | `splits/{train,val,test}_subjects.json`, `splits/graphs.pt`, `splits/normalization.json` |
| `tune` | **Ray Tune ASHA** sweep on `(train, val)`. Search space: `latent_dim ∈ {2,4,8,16,32}`, `hidden_dims`, `lr` log-uniform 1e-4..5e-3, `dropout` uniform 0..0.4, `batch_size ∈ {32,64}`, `weight_decay` log-uniform 1e-6..1e-3. Variational models (VGAE/GAEVAE): `beta_kl ∈ {0.001,0.01,0.1,1.0}`, `kl_warmup_epochs ∈ {0,10,30}`. Time-series mode: `corr_lambda` log-uniform 1e-2..1e1. | `tuning/<tag>/best_config.json`, `trials.csv`, `sweep_plot.png`, Ray logs under `tuning/<tag>/ray_results/` |
| `train` | Final training on `train` (early-stopped on `val`) using `best_config.json`. The test split is NOT touched. | `models/<tag>/model.pt`, `config.json`, `loss_curve.png`, `val_recon_grid.png`, (variational) `kl_curve.png` |
| `test` | **One-shot** evaluation on the held-out test split. | `final_test/<tag>/test_report.json`, `test_recon_grid.png` |
| `cluster` | KMeans, GMM, HDBSCAN on graph-level latents. For each clusterer: per-cluster diagnosis prevalence, per-cluster mean 64×64 matrix, Shannon entropy, per-subject entropy. Also runs the diagnosis-aware / state-dynamics / LOSO decoder / latent diagnostics suite. | `clustering/<tag>/{kmeans,gmm,hdbscan}/...`, `metrics/<tag>/*.tsv`, `*.npy` |
| `latents` | Re-extract latents and re-cluster without re-training (reads `models/<tag>/model.pt`). | same as `cluster` |
| `all` | `load → tune → train → test → cluster` end-to-end. |

`<tag>` is the model kind (`gae`, `vgae`, `gae_vae`, `enc_gae_fc`) — runs that
include multiple architectures get one subfolder per model.

> **Tuning backend:** the decoder `tune` stage uses Ray Tune ASHA when Ray is
> importable, and otherwise falls back automatically to a **Ray-free random
> search** over the same search space (`random_search_wsmi` in
> [src/train.py](src/train.py)) — useful on environments without Ray wheels
> (e.g. ppc64le). The `enc_gae_fc`/`cebra` path uses its own lightweight grid
> (`--cebra_tune`) and never needs Ray.

All artifacts for one invocation land under `output/<run_name>/`.

## Model variants

- **GAE** — deterministic graph autoencoder. SAGEConv encoder stack
  (`in_channels → hidden_dims → latent_dim`) and mirror decoder. Pure MSE
  reconstruction loss (plus `λ_corr·corr_loss` in time-series mode).
- **VGAE** — same SAGEConv stack, but the **last conv** outputs
  `2·latent_dim` (mu, log_var). Message-passing all the way to the
  bottleneck. KL warm-up controlled by `beta_kl` and `kl_warmup_epochs`.
- **GAEVAE** — plain SAGEConv encoder/decoder stacks **plus** a separate
  per-node MLP bottleneck `(mu, log_var)` between them. Cleanly separates
  "graph encoder" from "VAE encoder". Same loss as VGAE. See
  [src/model.py](src/model.py).
- **enc_gae_fc** — **encoder-only** `GNNEncoder` (SAGEConv stack + FC head),
  no decoder and no reconstruction. Emits one L2-normalized embedding per
  epoch-graph, trained **contrastively** with the CEBRA/InfoNCE loss
  (`--loss cebra`, the only loss it accepts). Selected by passing
  `--models enc_gae_fc` alone. See [src/model.py](src/model.py) and
  [src/cebra_loss.py](src/cebra_loss.py).

## Loss reference

Reconstruction (always):
```
L_recon = MSE(x_recon, x)            [+ β · KL(q(z|x) || N(0,I))  if VGAE / GAEVAE]
```

`--loss mse_corr` adds a **Pearson-correlation-preservation regularizer**:
```
L = L_recon + λ_corr · MSE( corr(x_recon), corr(x) )
```
where `corr(·)` is the per-graph channel × channel Pearson correlation matrix
(see [src/correlation_loss.py](src/correlation_loss.py)). **Why this exists**: on
raw time-series, pure MSE burns capacity on the ~90 % of the variance that is
noise; the correlation term forces cross-electrode covariance structure to be
preserved. It also runs on **wsmi** input (it then preserves the row-correlation
structure of the 64×64 matrix). Ray Tune searches `λ_corr ∈ [1e-2, 1e1]` whenever
`--loss mse_corr`; override at train stage with `--lambda_corr` (`0` disables it).

`--loss cebra` (encoder-only `enc_gae_fc`) trains a **time-contrastive
CEBRA/InfoNCE** objective instead of reconstruction (see
[src/cebra_loss.py](src/cebra_loss.py)): the positive of epoch *i* is epoch *i+1*
in the **same recording** (ordered by `matrix_idx`), negatives are the other
positives in the batch, similarity is cosine over L2-normalized embeddings, and
the temperature is optionally learnable. Tuned by a lightweight grid
(`--cebra_tune`), not Ray Tune.

## Required data layout

Patient/control folders are resolved from `DATA_DEFAULTS` in
[cookbook/run_wsmi_pipeline.py](cookbook/run_wsmi_pipeline.py), keyed by **data
variant** (`--type_data` + `--rs_duration`): `rs_0.8s`, `rs_16s`, `lg`. The
subpaths are joined onto `--data_root` (default `data`); explicit `--patient_dir`
/ `--control_dir` / `--timeseries_patient_dir` / `--timeseries_control_dir`
override them. lg is always 0.8 s, so `--rs_duration` is ignored there.

### wSMI matrices (`--input_mode wsmi`), subpaths under `<data_root>/`

| variant | patient | control (omit with `--patients_only`) |
|---|---|---|
| `rs_0.8s` | `markers/wsmi_theta/nice_epochs_sfreq-100Hz_recombine-biosemi64` | `markers/wsmi_theta/control_bids_biosemi64_tau10` |
| `rs_16s` | `markers/wsmi_theta/nice_epochs_sfreq-100Hz_recombine-biosemi64_dur-16s` | `markers/wsmi_theta/control_bids_biosemi64_dur-16_tau10` |
| `lg` | `markers/wsmi_theta_lg/nice_epochs_sfreq-100Hz_recombine-biosemi64_lg` | `markers/wsmi_theta_lg/control_bids_sfreq-100Hz_recombine-biosemi64-lg` |

Each tree: `sub-<ID>/ses-<NN>/eeg/sub-<ID>_ses-<NN>_acq-<NN>_desc-wsmi_connectivity.pkl`.
Labels: `metadata/DoC_metadata/metadata_patient_labels.csv` (`diagnostic_crs_final`).
Control lg matrices are recombined from the EGI256 `controls-lg` source to
biosemi64 at 100 Hz / tau=4 — see `preprocessing/01_recombine_egi256_to_biosemi64.py --task lg`
and `preprocessing/04_run_wsmi_theta.py`.

### Time-series (`--input_mode timeseries`), subpaths under `<data_root>/`

| variant | patient | control | window |
|---|---|---|---|
| `rs_0.8s` | `fif/pic-nic/nice_epochs_sfreq-100Hz_recombine-biosemi64` (task=rs) | `fif/pic-nic/control_bids_biosemi64-rs` | crop -0.2..0.6 s (80) |
| `rs_16s` | `fif/pic-nic/nice_epochs_sfreq-100Hz_recombine-biosemi64_dur-16s` (task=rs) | `fif/pic-nic/control_bids_biosemi64_dur-16s-rs` | 16 s (1600) |
| `lg` | `fif/pic-nic/nice_epochs_sfreq-100Hz_recombine-biosemi64` (task=lg) | `fif/pic-nic/control_bids_sfreq-100Hz_recombine-biosemi64-lg` | crop -0.2..0.6 s (80) |

Each tree: `sub-<ID>/ses-<NN>/eeg/sub-<ID>_ses-<NN>_task-<TASK>_acq-<NN>_epo.fif`.
The plain `nice_epochs_..._biosemi64` tree holds **both** rs and lg epochs — the
loader's `task=` filter (driven by `--type_data`) selects the right ones. Each
`.fif` is read with MNE, resampled to `--sfreq`, and either cropped to
`[--crop_tmin, --crop_tmax]` (0.8 s variants) or sliced to the first
`--window_sec * sfreq` samples (`rs_16s`). `Data.x` is `(64, 80)` for 0.8 s and
`(64, 1600)` for 16 s.

### Subject / diagnosis mapping

Numeric subject IDs (`sub-001`) are treated as **healthy controls** and get
`diagnosis_group = "CONTROL"`. Non-numeric IDs are patients whose
`diagnostic_crs_final` is mapped via `DIAGNOSIS_GROUP_MAP`:

| `diagnostic_crs_final` | `diagnosis_group` |
|---|---|
| `UWS`, `VS` | `UWS` |
| `MCS-`, `MCS`, `MCS+` | `MCS` |
| `EMCS`, `COMA`, missing | dropped at load time |

Edit `DIAGNOSIS_GROUP_MAP` in [src/wsmi_loader.py](src/wsmi_loader.py) to
change the mapping. With `--patients_only`, `CONTROL` does not appear and
the stratified split runs over `{UWS, MCS}`.

## Key CLI flags (full reference)

| Flag | Default | Meaning |
|---|---|---|
| `--stage` | `all` | Which stage to run (`load`/`tune`/`train`/`test`/`latents`/`cluster`/`all`). |
| `--models` | `gae vgae` | Subset of `{gae, vgae, gae_vae}` (decoders, with `mse`/`mse_corr`), **or** `enc_gae_fc` alone (with `--loss cebra`). One tag per architecture. |
| `--loss` | `mse` | Training objective: `mse`, `mse_corr` (decoders), or `cebra` (enc_gae_fc). |
| `--input_mode` | `wsmi` | `wsmi` (pre-computed matrices) or `timeseries` (raw .fif). |
| `--type_data` | `rs` | Task: `rs` (resting-state) or `lg` (local-global). Selects default dirs + task filter. |
| `--rs_duration` | `0.8s` | RS window: `0.8s` (crop -0.2..0.6 s) or `16s` (full window). Ignored for `lg`. |
| `--diagnosis_granularity` | `coarse` | `coarse` → CONTROL/UWS/MCS; `fine` → raw `diagnostic_crs_final` labels. |
| `--patients_only` | off | Skip the control directory; train on DoC patients only. |
| `--data_root` | `data` | Prefix joined onto the per-variant `DATA_DEFAULTS` subpaths. |
| `--patient_dir` / `--control_dir` | — | Explicit wSMI roots (override `DATA_DEFAULTS`). |
| `--timeseries_patient_dir` / `--timeseries_control_dir` | — | Explicit `.fif` roots (required for `timeseries` unless resolved from `DATA_DEFAULTS`; control optional with `--patients_only`). |
| `--coords_file` | `…/biosemi64.txt` | Electrode coordinates file (k-NN adjacency). |
| `--k` | `6` | k for the k-NN electrode adjacency graph. |
| `--graph_latent_agg` | `flatten` | Per-graph latent: `flatten` (concat electrodes) or `mean` (average). Autoencoders only. |
| `--lambda_corr` | `None` | Override the `mse_corr` weight at train stage. `None` → use Ray Tune's value. `0` → disable. |
| `--crop_tmin` / `--crop_tmax` | `-0.2` / `0.6` | Timeseries crop window (s) for the 0.8 s variants. |
| `--sfreq` | `100.0` | Target sampling rate (Hz) for time-series mode. |
| `--window_sec` | `16.0` | Per-graph window length (s) used when crop is disabled (`rs_16s`). |
| `--filter_outliers` | off | Drop epochs that are Mahalanobis 2-sigma outliers under a Gaussian fit on TRAIN. |
| `--outlier_n_sigma` / `--outlier_threshold` | `2.0` / `empirical` | Outlier boundary; `empirical` (TRAIN quantile) or `chi2`. |
| `--subject_filter` | `None` | Restrict the load to a subset of subjects. |
| `--max_epochs_per_recording` | `None` | Subsample epochs/recording (caps memory on big lg / rs_16s sets). |
| `--num_trials` | `30` | Ray Tune ASHA samples per architecture. |
| `--tune_epochs` | `80` | Max epochs per ASHA trial (`max_t`). |
| `--cpus_per_trial` | `2` | CPU cores per Ray trial (tune stage is CPU-only). |
| `--max_concurrent_trials` | `None` | Cap simultaneous Ray trials (peak RAM ≈ dataset·(1+N)). |
| `--cebra_tune` | off | Run the lightweight enc_gae_fc/cebra grid (temperature/lr/latent_dim) before training. |
| `--cebra_*` | see `--help` | enc_gae_fc/cebra hyperparameters (`--cebra_latent_dim`, `--cebra_lr`, `--cebra_temperature`, `--cebra_tune_*`, …). |
| `--train_epochs` | `150` | Max epochs for final training. |
| `--train_patience` | `25` | Early-stopping patience on val MSE. |
| `--test_frac` / `--val_frac` | `0.15` / `0.15` | Subject-level split fractions (stratified on `diagnosis_group`). |
| `--seed` | `42` | Seeds the split, Ray sampling, and torch RNG. |
| `--smoke` | off | Tiny config (`num_trials=2`, `tune_epochs=3`, `train_epochs=5`). |
| `--cpu` | off | Force CPU even if CUDA / MPS is available. |
| `--run_name` | `wsmi_run_<timestamp>` | Output subfolder under `--output_root` (default `gnn_connectivity/output`). |

## Adaptability to other scalp configurations

Designed to work with any EEG configuration. Create a coordinates file in
`data_scalp/` listing electrode labels (one per line, matching the
`eeg_positions` 10-05 library), then pass it via `--coords_file`. Ships with
[data_scalp/biosemi64.txt](data_scalp/biosemi64.txt) (64-ch Biosemi) and
[data_scalp/GSN-HydroCel-257.txt](data_scalp/GSN-HydroCel-257.txt) (257-ch EGI).

## Data leakage prevention

All splits are **subject-level** (`split_by_subject_stratified` in
[src/data_loaders.py](src/data_loaders.py)) stratified on diagnosis_group.
Every graph from a given subject lands in exactly one of train/val/test.
The pipeline asserts non-overlap before returning.

## Diagnosis-aware metrics (run automatically inside `cluster`)

During the `cluster` stage the pipeline fits a KMeans on the latents
(silhouette-best k) and computes the following, dumped under
`output/<run>/metrics/<tag>/`:

- **`diagnosis_clustering_quality.tsv`** — ARI, AMI, homogeneity, completeness, V-measure, cluster purity. See [src/cluster_analysis.py](src/cluster_analysis.py) → `diagnosis_clustering_metrics`.
- **`state_dynamics_{per_recording,per_subject,group_summary}.tsv`** — state sequence, transition matrix, occupancy entropy, entropy rate, weighted entropy. See [src/state_dynamics.py](src/state_dynamics.py).
- **`decoder_metrics.tsv` / `decoder_predictions.tsv`** — Leave-One-Subject-Out logistic regression over `latent_means`, `state_probabilities`, `transitions`, `state_combined`, `latent_plus_state`; macro-AUC (OvR), accuracy, macro-F1. See [src/decoder_eval.py](src/decoder_eval.py). Skipped if fewer than 2 diagnosis_groups are present.
- **`per_edge_mse.npy` + `per_subject_recon_mse.tsv`** — reconstruction error per (node, feature) and per subject. See [src/latent_diagnostics.py](src/latent_diagnostics.py).
- **Variational only** (`vgae`, `gae_vae`): `posterior_collapse.json`, `kl_per_dim.npy`, `mu_logvar_stats.tsv` — flags dead latent dims.

## Companion scripts in `scripts/`

| Script | What it does |
|---|---|
| [scripts/build_wsmi_manifest.py](scripts/build_wsmi_manifest.py) | Build a 5-column TSV manifest from junifer `.pkl` outputs + `metadata_patient_labels.csv`. Drops EMCS/COMA by default; `--keep-all` to retain them. |
| [scripts/render_wsmi_run_report.py](scripts/render_wsmi_run_report.py) | Render one combined HTML + TSV report for an `output/<run>/` directory: hyperparameter sweep, final test metrics, (model × clusterer) table, brain-state dynamics by diagnosis, LOSO decoder, reconstruction & latent diagnostics. |
| [scripts/render_clustered_mean_matrices.py](scripts/render_clustered_mean_matrices.py) | Re-render each cluster's mean 64×64 wSMI matrix as an electrode-by-electrode heatmap in native electrode order. |
| [scripts/convert_wsmi_pkl_to_npz.py](scripts/convert_wsmi_pkl_to_npz.py) | Optional `.pkl` → `.npz` converter. Not required; kept as utility. |
| [scripts/add_prevalence_boxplots.py](scripts/add_prevalence_boxplots.py) | Re-render `prevalence_proportion_boxplots.png` (per-subject cluster proportions, ★ = group mean) from saved CSVs for existing runs — no retraining. |

### Typical run-then-report workflow

```bash
# 1. Train + cluster + dump per-model metrics
gnn_connectivity/.venv/bin/python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --models gae vgae \
    --patient_dir data/markers/wsmi_theta/nice_epochs_sfreq-100Hz_recombine-biosemi64_dur-16s \
    --control_dir data/markers/wsmi_theta/control_bids_biosemi64_dur-16s \
    --diagnosis_csv metadata/DoC_metadata/metadata_patient_labels.csv \
    --coords_file  gnn_connectivity/data_scalp/biosemi64.txt \
    --output_root  gnn_connectivity/output \
    --run_name     wsmi_full_combined

# 2. Render the combined HTML report
gnn_connectivity/.venv/bin/python gnn_connectivity/scripts/render_wsmi_run_report.py \
    --run-dir    gnn_connectivity/output/wsmi_full_combined \
    --output-html gnn_connectivity/output/wsmi_full_combined/report.html \
    --output-tsv  gnn_connectivity/output/wsmi_full_combined/report.tsv

# 3. (Optional) Re-render per-cluster mean matrices (native electrode order)
for combo in gae/kmeans gae/gmm vgae/kmeans vgae/gmm; do
  gnn_connectivity/.venv/bin/python gnn_connectivity/scripts/render_clustered_mean_matrices.py \
      --mean-matrices-dir gnn_connectivity/output/wsmi_full_combined/clustering/$combo/mean_matrices \
      --coords-file       gnn_connectivity/data_scalp/biosemi64.txt \
      --output-png        gnn_connectivity/output/wsmi_full_combined/clustering/$combo/mean_matrices/mean_matrices_grid.png
done
```

## Module overview

| Module | Purpose |
|---|---|
| [src/model.py](src/model.py) | GAE, VGAE, GAEVAE autoencoders + `GNNEncoder` (enc_gae_fc) contrastive encoder. |
| [src/train.py](src/train.py) | Training loop, MSE evaluation, Ray Tune ASHA wrapper. Honors `corr_lambda` and `model_kind`. |
| [src/cebra_loss.py](src/cebra_loss.py) | CEBRA-style InfoNCE criterion + `CebraPairDataset` (temporal ref/pos pairs) for enc_gae_fc. |
| [src/outlier_filter.py](src/outlier_filter.py) | Mahalanobis 2-sigma outlier model (fit on TRAIN, applied to all splits) for `--filter_outliers`. |
| [src/analysis.py](src/analysis.py) | Standalone post-training latent cluster analysis (KMeans, per-cluster mean matrices, within-session Markov transitions, occupancy entropy). |
| [src/wsmi_loader.py](src/wsmi_loader.py) | Read junifer wSMI `.pkl` outputs into per-epoch graphs; applies the CONTROL/UWS/MCS mapping. |
| [src/timeseries_loader.py](src/timeseries_loader.py) | Read `.fif` epoch files into per-epoch graphs with raw time-series node features. |
| [src/correlation_loss.py](src/correlation_loss.py) | Differentiable per-graph Pearson correlation matrix + `corr_loss`. |
| [src/data_loaders.py](src/data_loaders.py) | Subject-level stratified splits + DataLoader wrappers. |
| [src/preprocessing.py](src/preprocessing.py) | `EEGtoGraph` adjacency builder (k-NN over electrode coords). |
| [src/cluster_analysis.py](src/cluster_analysis.py) | KMeans / GMM / HDBSCAN over latents; ARI / AMI / V-measure / purity vs diagnosis. |
| [src/state_dynamics.py](src/state_dynamics.py) | Per-recording state sequences, transition matrices, occupancy / weighted entropy, entropy rate. |
| [src/decoder_eval.py](src/decoder_eval.py) | LOSO logistic-regression decoder over latent-mean / state-prob / transition / combined feature sets. |
| [src/latent_diagnostics.py](src/latent_diagnostics.py) | Per-edge / per-subject reconstruction MSE; variational-only `kl_per_dim`, `posterior_collapse_fraction`, `mu_logvar_stats`. |
| [src/latent_space_study.py](src/latent_space_study.py) | Post-hoc latent-space clustering, PCA, silhouette scoring. |
| [src/inference.py](src/inference.py) | Run a trained model on new data. |

## License

MIT License

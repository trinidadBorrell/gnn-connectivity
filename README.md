# GNN Connectivity Pipeline

A Graph Neural Network (GNN) based pipeline for analyzing EEG connectivity data using Graph Autoencoders (GAE).

## Features

- **Flexible Data Input**: Process raw EEG `.fif` files OR load pre-computed connectivity matrices
- **Subject-Level Data Splitting**: GroupKFold ensures no data leakage between train/val/test sets
- **GPU/MPS Support**: Auto-detects CUDA, Apple Silicon (MPS), or CPU
- **Configurable Architecture**: Progressive dimension reduction (e.g., 64→64→32→16→2)
- **Learning Rate Scheduler**: ReduceLROnPlateau for optimal convergence
- **Multi-Electrode Support**: Works with any EEG electrode configuration
- **Experiment Tracking**: Optional wandb integration for real-time metrics visualization

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Full Pipeline (EEG Mode)

```bash
python cookbook/run_pipeline.py \
    --main_path /path/to/eeg_data \
    --coordinates_file data_scalp/biosemi64.txt \
    --feature_type connectivity
```

### Option 2: Full Pipeline (Matrix Mode)

```bash
python cookbook/run_pipeline.py \
    --matrix_dir /path/to/matrices \
    --coordinates_file data_scalp/biosemi64.txt
```

### Option 3: Skip Preprocessing (use saved datasets)

```bash
python cookbook/run_pipeline.py \
    --skip_preprocessing \
    --datasets_dir output/preprocessing/datasets \
    --coordinates_file data_scalp/biosemi64.txt
```

### Option 4: Custom Training Configuration

```bash
python cookbook/run_pipeline.py \
    --main_path /path/to/eeg_data \
    --hidden_dims "64,64,32,16" \
    --latent_dim 2 \
    --batch_size 64 \
    --n_epochs 200 \
    --dropout 0.2 \
    --lr 0.001
```

### Option 5: With Wandb Logging

```bash
# First time: authenticate with wandb
wandb login

# Run with wandb enabled
python cookbook/run_pipeline.py \
    --main_path /path/to/eeg_data \
    --coordinates_file data_scalp/biosemi64.txt \
    --wandb \
    --wandb_project my-gnn-project
```

## wSMI Pipeline (DoC patients + healthy controls)

End-to-end pipeline for the wSMI-theta connectivity matrices: trains GAE and
VGAE on per-epoch biosemi64 graphs and clusters the resulting latent space.
Entry point: [cookbook/run_wsmi_pipeline.py](cookbook/run_wsmi_pipeline.py).

### What the stages do

| `--stage` | What runs | What lands on disk |
|-----------|-----------|--------------------|
| `load`    | Reads the patient + control wSMI folders, builds one Data graph per epoch, looks up `diagnostic_crs_final` from the CSV, runs the **stratified 70/15/15 subject-level split**, computes train-only normalization stats and applies them in place. | `splits/{train,val,test}_subjects.json`, `splits/graphs.pt`, `splits/normalization.json` |
| `tune`    | **Ray Tune ASHA** hyperparameter search on `(train, val)`. Search space: `latent_dim ∈ {2,4,8,16,32}`, `hidden_dims`, `lr` log-uniform 1e-4..5e-3, `dropout` uniform 0..0.4, `batch_size ∈ {32,64}`, `weight_decay` log-uniform 1e-6..1e-3. VGAE only: `beta_kl ∈ {0.001,0.01,0.1,1.0}`, `kl_warmup_epochs ∈ {0,10,30}`. `--num_trials` = how many independent configurations ASHA samples (each is one trial; bad trials are pruned early by the scheduler). | `tuning/{gae,vgae}/best_config.json`, `tuning/{gae,vgae}/trials.csv`, `tuning/{gae,vgae}/sweep_plot.png`, full Ray logs under `tuning/{gae,vgae}/ray_results/` |
| `train`   | Final model training on `train` (early-stopped on `val`) using `best_config.json`. The test split is NOT touched. | `models/{gae,vgae}/model.pt`, `models/{gae,vgae}/config.json`, `loss_curve.png`, `val_recon_grid.png`, VGAE also `kl_curve.png` |
| `test`    | **One-shot** evaluation on the held-out test split. After this, do not re-tune. | `final_test/{gae,vgae}/test_report.json`, `test_recon_grid.png` |
| `cluster` | Extracts graph-level latents for every graph, then runs **K-Means** (silhouette over k=2..10), **GMM** (BIC over k=2..10), and **HDBSCAN** (validity index over min_cluster_size ∈ {10,20,50,100}). For each clusterer it saves: per-cluster diagnosis prevalence (counts CSV + stacked-bar PNG, counts and proportions), per-cluster mean 64×64 wSMI matrix, Shannon + variance-weighted cluster-occupancy entropy, and per-subject entropy with a box plot grouped by diagnosis_group. **Also runs the diagnosis-aware / dynamics / decoder / latent-diagnostics suite** (see "Diagnosis-aware metrics" below). | `clustering/{gae,vgae}/{kmeans,gmm,hdbscan}/{assignments.csv, prevalence_counts.csv, prevalence_counts.png, prevalence_proportions.png, subject_modal_cluster.csv, mean_matrices/, entropy.json, per_subject_entropy.csv, per_subject_entropy_box.png, metric_curve.png, latent_scatter.png}` plus `metrics/{gae,vgae}/{diagnosis_clustering_quality.tsv, state_dynamics_per_recording.tsv, state_dynamics_per_subject.tsv, state_dynamics_group_summary.tsv, decoder_metrics.tsv, decoder_predictions.tsv, per_edge_mse.npy, per_subject_recon_mse.tsv}` and VGAE only `metrics/vgae/{posterior_collapse.json, kl_per_dim.npy, mu_logvar_stats.tsv}` |
| `all`     | Runs `load → tune → train → test → cluster` end to end. |
| `latents` | Re-extract latents and re-cluster without re-training (uses the saved `model.pt`). |

All artifacts for one invocation land under
`output/<run_name>/` (default `run_name` is `wsmi_run_<timestamp>`).

### Step-by-step commands

Run from the **repository root** (`Documents/Work/PhD/Proyects/GNNs`).
Default paths in the script already point at the patient/control wSMI folders,
the metadata CSV, and `gnn_connectivity/data_scalp/biosemi64.txt`; override
with `--patient_dir / --control_dir / --diagnosis_csv / --coords_file` if you
move things around.

**1. Smoke test (~30 seconds; tiny config to confirm the wiring is fine):**
```bash
python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --smoke --models gae --cpus_per_trial 1 \
    --run_name wsmi_smoke
```

**2. Full training (Ray Tune sweep + final fit + held-out test) for BOTH GAE and VGAE.**
Pick a `--run_name` so re-running stages later finds the right folder:
```bash
python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --stage all --models gae vgae \
    --num_trials 30 --tune_epochs 80 --cpus_per_trial 2 \
    --train_epochs 200 --train_patience 25 \
    --run_name wsmi_run_main
```

**3. Train just GAE (or just VGAE):**
```bash
python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --stage all --models gae \
    --num_trials 30 --tune_epochs 80 \
    --run_name wsmi_gae_only
```

**4. Re-cluster an already-trained model** (skips loading, tuning, and training; reads `models/<tag>/model.pt` and `splits/graphs.pt` from the same `--run_name`):
```bash
python gnn_connectivity/cookbook/run_wsmi_pipeline.py \
    --stage cluster --models gae vgae \
    --run_name wsmi_run_main
```

**5. Run individual stages (chain them as you go):**
```bash
# (a) load + split
python gnn_connectivity/cookbook/run_wsmi_pipeline.py --stage load --run_name wsmi_run_main
# (b) hyperparameter search
python gnn_connectivity/cookbook/run_wsmi_pipeline.py --stage tune --models gae vgae --num_trials 30 --tune_epochs 80 --run_name wsmi_run_main
# (c) final training with the best config
python gnn_connectivity/cookbook/run_wsmi_pipeline.py --stage train --models gae vgae --train_epochs 200 --run_name wsmi_run_main
# (d) one-shot held-out test evaluation
python gnn_connectivity/cookbook/run_wsmi_pipeline.py --stage test --models gae vgae --run_name wsmi_run_main
# (e) cluster the latent space
python gnn_connectivity/cookbook/run_wsmi_pipeline.py --stage cluster --models gae vgae --run_name wsmi_run_main
```

### Key wSMI-pipeline CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--stage` | `all` | Which stage to run (`load`/`tune`/`train`/`test`/`latents`/`cluster`/`all`). |
| `--models` | `gae vgae` | Which models to train. Pass a subset to skip one. |
| `--num_trials` | `30` | How many configurations Ray Tune samples (ASHA prunes losers early). Higher = more thorough sweep, linear cost. |
| `--tune_epochs` | `80` | Maximum epochs per ASHA trial (`max_t`). |
| `--cpus_per_trial` | `2` | CPU cores reserved per Ray trial. Lower it if you have fewer cores. |
| `--train_epochs` | `150` | Max epochs for the final post-tuning training. |
| `--train_patience` | `25` | Early-stopping patience on val MSE during final training. |
| `--test_frac` / `--val_frac` | `0.15` / `0.15` | Subject-level split fractions (stratified on `diagnosis_group`). |
| `--seed` | `42` | Seeds the split, Ray sampling, and torch RNG. |
| `--smoke` | off | Tiny config (`num_trials=2`, `tune_epochs=3`, `train_epochs=5`) — for plumbing checks only. |
| `--cpu` | off | Force CPU even if CUDA / MPS is available. |
| `--run_name` | `wsmi_run_<timestamp>` | Output subfolder under `--output_root` (default `gnn_connectivity/output`). |

### Required data layout

```
data/markers/wsmi_theta/nice_epochs_sfreq-100Hz_recombine-biosemi64_dur-16s/
└── sub-<ID>/ses-<NN>/eeg/sub-<ID>_ses-<NN>_acq-<NN>_desc-wsmi_connectivity.pkl

data/markers/wsmi_theta/control_bids_biosemi64_dur-16s/
└── sub-<NNN>/ses-<NN>/eeg/sub-<NNN>_ses-<NN>_acq-<NN>_desc-wsmi_connectivity.pkl

metadata/DoC_metadata/metadata_patient_labels.csv   # has column `diagnostic_crs_final`
```

Numeric subject IDs (`sub-001`) are treated as **healthy controls** and get `diagnosis_group = "CONTROL"`. Non-numeric IDs are patients whose `diagnostic_crs_final` is mapped to `diagnosis_group ∈ {UWS, MCS}`:

| `diagnostic_crs_final` | `diagnosis_group` |
|------------------------|-------------------|
| `UWS`, `VS`            | `UWS`             |
| `MCS-`, `MCS`, `MCS+`  | `MCS`             |
| `EMCS`, `COMA`, missing| dropped at load time (filtered out before splitting) |

Edit `DIAGNOSIS_GROUP_MAP` in [src/wsmi_loader.py](src/wsmi_loader.py) to change the mapping.

### Diagnosis-aware metrics (run automatically inside `cluster`)

During the `cluster` stage the pipeline also fits a KMeans on the latents (silhouette-best k) and then computes the following, dumped under `output/<run>/metrics/{gae,vgae}/`:

- **`diagnosis_clustering_quality.tsv`** — ARI, AMI, homogeneity, completeness, V-measure, and cluster purity vs `CONTROL/UWS/MCS`. See [src/cluster_analysis.py](src/cluster_analysis.py) → `diagnosis_clustering_metrics`.
- **`state_dynamics_per_recording.tsv` / `_per_subject.tsv` / `_group_summary.tsv`** — per-recording state sequence, occupancy probabilities, transition matrix, occupancy entropy, entropy rate, weighted entropy; aggregated to subject and diagnosis-group level. See [src/state_dynamics.py](src/state_dynamics.py).
- **`decoder_metrics.tsv` / `decoder_predictions.tsv`** — Leave-One-Subject-Out logistic regression over five feature sets (`latent_means`, `state_probabilities`, `transitions`, `state_combined`, `latent_plus_state`); reports macro-AUC (OvR), accuracy, macro-F1. See [src/decoder_eval.py](src/decoder_eval.py).
- **`per_edge_mse.npy` + `per_subject_recon_mse.tsv`** — reconstruction error per (node, feature) and per subject. See [src/latent_diagnostics.py](src/latent_diagnostics.py).
- **VGAE only**: `posterior_collapse.json`, `kl_per_dim.npy`, `mu_logvar_stats.tsv` — flags dead latent dims.

### Companion scripts in `scripts/`

| Script | What it does |
|--------|--------------|
| [scripts/build_wsmi_manifest.py](scripts/build_wsmi_manifest.py) | Build a 5-column TSV manifest (`sample_id, subject, session, path, diagnosis_canonical`) from the junifer `.pkl` outputs + `metadata_patient_labels.csv`, suitable for the sibling `clustering-wsmi` pipeline. Drops EMCS/COMA by default; `--keep-all` to retain them. |
| [scripts/render_wsmi_run_report.py](scripts/render_wsmi_run_report.py) | Render one combined HTML + TSV report for a `output/<run>/` directory: run metadata, pretraining hyperparameter sweep, final test metrics, the (model × clusterer) table with silhouette/BIC/ARI/AMI/V-measure/purity, brain-state dynamics by diagnosis class, LOSO decoder breakdown, reconstruction & latent diagnostics. Gracefully handles single-model runs. |
| [scripts/render_clustered_mean_matrices.py](scripts/render_clustered_mean_matrices.py) | Re-render each cluster's mean 64×64 wSMI matrix with electrodes grouped by functional network (AUD / DMN / FP / MOT / SAL / VIS). The mapping is heuristic; edit `NETWORK_LAYOUT` at the top of the script to customize. Writes a JSON sidecar with the resolved permutation. |
| [scripts/convert_wsmi_pkl_to_npz.py](scripts/convert_wsmi_pkl_to_npz.py) | Optional `.pkl` → `.npz` converter. **Not required** for clustering-wsmi (which reads `.pkl` natively); kept as a utility for downstream tools that demand `.npz`. |

#### Typical run-then-report workflow

```bash
# 1. Train + cluster + dump per-model metrics (all subjects, default 16s data)
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

# 3. (Optional) Re-render per-cluster mean matrices grouped by brain network
for combo in gae/kmeans gae/gmm vgae/kmeans vgae/gmm; do
  gnn_connectivity/.venv/bin/python gnn_connectivity/scripts/render_clustered_mean_matrices.py \
      --mean-matrices-dir gnn_connectivity/output/wsmi_full_combined/clustering/$combo/mean_matrices \
      --coords-file       gnn_connectivity/data_scalp/biosemi64.txt \
      --output-png        gnn_connectivity/output/wsmi_full_combined/clustering/$combo/mean_matrices/mean_matrices_grid_by_network.png
done
```

## Directory Structure

### EEG Data (EEG Mode)
```
{main_path}/
├── sub-{ID}/
│   ├── ses-{num}/
│   │   └── eeg/
│   │       └── sub-{ID}_ses-{num}_epo.fif
```

### Pre-computed Matrices (Matrix Mode)
```
{matrix_dir}/
├── sub-{ID}/
│   ├── ses-{num}/
│   │   ├── matrix_0.npy
│   │   ├── matrix_1.npy
│   │   └── ...
```

### Output Structure
```
output/
├── preprocessing/
│   ├── data/
│   │   └── adjacency_matrix_*.npy
│   ├── datasets/
│   │   ├── train_dataset.pt
│   │   ├── val_dataset.pt
│   │   └── test_dataset.pt
│   └── images/
└── training/
    ├── models/
    │   └── gae_model_*.pt
    └── plots/
```

## Adaptability to Other Scalp Configurations

This pipeline is designed to work with **any EEG electrode configuration**:

### Step 1: Create a Coordinates File

Create a text file in `data_scalp/` listing your electrode labels (one per line):

```
# data_scalp/my_system.txt
Fp1
Fp2
F3
F4
...
```

The electrode labels must match those in the `eeg_positions` library (10-05 system).

### Step 2: Run the Pipeline

```bash
python cookbook/run_pipeline.py \
    --coordinates_file data_scalp/my_system.txt \
    --main_path /path/to/your/data
```

### Available Scalp Configurations

- `data_scalp/biosemi64.txt` - 64-channel Biosemi system
- `data_scalp/GSN-HydroCel-257.txt` - 257-channel EGI system

### Custom Path Patterns

If your data follows a different naming convention, use custom path patterns:

```bash
# For EEG files with custom naming
python cookbook/run_pipeline.py \
    --main_path /path/to/data \
    --path_pattern "{main_path}/sub-{subject_id}/ses-{session_num}/eeg/custom_name.fif" \
    --coordinates_file data_scalp/biosemi64.txt

# For pre-computed matrices with custom naming
python cookbook/run_pipeline.py \
    --matrix_dir /path/to/matrices \
    --matrix_path_pattern "{matrix_dir}/subject_{subject_id}/session_{session_num}/conn_{matrix_idx}.npy" \
    --coordinates_file data_scalp/biosemi64.txt
```

**Available placeholders:**
- EEG patterns: `{main_path}`, `{subject_id}`, `{session_num}`
- Matrix patterns: `{matrix_dir}`, `{subject_id}`, `{session_num}`, `{matrix_idx}`

## Data Leakage Prevention

The pipeline uses **subject-level splitting** via `GroupKFold` to prevent data leakage:

- All matrices/epochs from a single subject stay in the same split (train OR val OR test)
- This prevents the model from learning subject-specific patterns during training
- The `verify_no_data_leakage()` function automatically checks for overlapping subjects

```python
from data_loaders import verify_no_data_leakage

# This will raise an error if any subjects appear in multiple splits
verify_no_data_leakage(train_dataset, val_dataset, test_dataset)
```

## Key Parameters

### Preprocessing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--feature_type` | `connectivity` | `temporal` (raw time window) or `connectivity` (Pearson correlation) |
| `--window_points` | `64` | Number of time points per window (EEG mode only) |
| `--k` | `6` | Number of nearest neighbors for adjacency matrix |
| `--n_splits` | `5` | Number of folds for GroupKFold |
| `--test_fold` | `0` | Which fold to use as test set |
| `--no_plot_neighbors` | `False` | Skip k-nearest neighbors visualization |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hidden_dims` | `64,64,32,16` | Comma-separated layer dimensions |
| `--latent_dim` | `2` | Latent space dimension |
| `--dropout` | `0.2` | Dropout probability |
| `--batch_size` | `64` | Batch size for training |
| `--n_epochs` | `200` | Training epochs |
| `--lr` | `0.001` | Learning rate |
| `--weight_decay` | `1e-5` | Adam weight decay |
| `--wandb` | `False` | Enable wandb logging |
| `--wandb_project` | `gnn-connectivity` | Wandb project name |

## Module Overview

| Module | Purpose |
|--------|---------|
| `preprocessing.py` | Convert EEG/matrices to PyTorch Geometric graphs |
| `data_loaders.py` | Create train/val/test loaders with subject-level splits |
| `wsmi_loader.py` | Read junifer wSMI `.pkl` outputs into per-epoch graphs; applies the CONTROL/UWS/MCS mapping |
| `model.py` | GAE + VGAE architectures (GraphSAGE-based) |
| `train.py` | Training loop, MSE evaluation, Ray Tune ASHA wrapper |
| `inference.py` | Run trained model on new data |
| `cluster_analysis.py` | KMeans / GMM / HDBSCAN over latents; per-cluster prevalence, mean matrices, entropy; **ARI / AMI / V-measure / purity** vs diagnosis |
| `state_dynamics.py` | Per-recording state sequences, transition matrices, occupancy entropy, entropy rate, weighted entropy |
| `decoder_eval.py` | LOSO logistic-regression decoder over latent-mean / state-prob / transition / combined feature sets |
| `latent_diagnostics.py` | Per-edge / per-subject reconstruction MSE; VGAE-only `kl_per_dim`, `posterior_collapse_fraction`, `mu_logvar_stats` |
| `analysis.py` | Legacy raw-EEG latent cluster analysis (`LatentClusterAnalysis`) used by `cookbook/run_analysis.py` |

## Running Individual Scripts

### Preprocessing Only

```python
import sys
sys.path.insert(0, 'src')

from preprocessing import EEGtoGraph
import numpy as np
from eeg_positions import get_elec_coords

# Load electrode coordinates
labels = np.loadtxt('data_scalp/biosemi64.txt', usecols=(0,), dtype=str)
coords_data = get_elec_coords(system='1005', as_mne_montage=False)
coords_df = coords_data[coords_data['label'].isin(labels)].copy()

# Create datasets
train_ds, val_ds, test_ds = EEGtoGraph.create_graph_dataset(
    coords_df=coords_df,
    matrix_dir='/path/to/matrices',  # OR main_path for EEG mode
    k=6,
    n_splits=5,
    test_fold=0,
    plot_neighbors=True
)
```

### Training Only (with existing datasets)

```python
import torch
from train import TrainGAE, normalize_graph_features

# Load saved datasets
train_ds = torch.load('output/preprocessing/datasets/train_dataset.pt', weights_only=False)
val_ds = torch.load('output/preprocessing/datasets/val_dataset.pt', weights_only=False)
test_ds = torch.load('output/preprocessing/datasets/test_dataset.pt', weights_only=False)

# Get graphs and normalize
train_graphs, val_graphs, test_graphs = train_ds.data, val_ds.data, test_ds.data
all_graphs = train_graphs + val_graphs + test_graphs
normalized_all, _, _, _ = normalize_graph_features(all_graphs)

# Split back
train_graphs = normalized_all[:len(train_ds.data)]
val_graphs = normalized_all[len(train_ds.data):len(train_ds.data)+len(val_ds.data)]
test_graphs = normalized_all[len(train_ds.data)+len(val_ds.data):]

# Train
in_channels = train_graphs[0].x.shape[1]
trainer = TrainGAE(train_graphs, val_graphs, test_graphs, in_channels)

config = {
    'in_channels': in_channels,
    'hidden_dims': [64, 64, 32, 16],  # 64 -> 64 -> 32 -> 16 -> latent
    'latent_dim': 2,
    'dropout': 0.2,
    'batch_size': 64,
    'n_epochs': 200,
    'lr': 0.001,
    'weight_decay': 1e-5,
}

model = trainer.train_final(config, save_results=True, output_dir='output/training')

# With wandb logging
model = trainer.train_final(config, save_results=True, output_dir='output/training', use_wandb=True)
```

### Inference (with trained model)

```bash
python src/inference.py \
    --model_path output/training/models/gae_model_*.pt \
    --main_path /path/to/eeg_data \
    --subject_id 01 \
    --session_num 01
```

## License

MIT License

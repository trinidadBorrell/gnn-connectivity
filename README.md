# GNN Connectivity Pipeline

A Graph Neural Network (GNN) based pipeline for analyzing EEG connectivity data using Graph Autoencoders (GAE).

## Features

- **Flexible Data Input**: Process raw EEG `.fif` files OR load pre-computed connectivity matrices
- **Subject-Level Data Splitting**: GroupKFold ensures no data leakage between train/val/test sets
- **GPU/MPS Support**: Auto-detects CUDA, Apple Silicon (MPS), or CPU
- **Configurable Architecture**: Progressive dimension reduction (e.g., 64→64→32→16→2)
- **Learning Rate Scheduler**: ReduceLROnPlateau for optimal convergence
- **Multi-Electrode Support**: Works with any EEG electrode configuration

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

## Module Overview

| Module | Purpose |
|--------|---------|
| `preprocessing.py` | Convert EEG/matrices to PyTorch Geometric graphs |
| `data_loaders.py` | Create train/val/test loaders with subject-level splits |
| `model.py` | Graph Autoencoder architecture (GraphSAGE-based) |
| `train.py` | Training loop and model saving |
| `inference.py` | Run trained model on new data |

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

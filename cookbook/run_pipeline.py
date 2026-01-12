#!/usr/bin/env python3
"""
COOKBOOK: Complete Pipeline Example
====================================
This script demonstrates how to run the complete GNN Connectivity pipeline:
1. Preprocessing: Convert EEG data OR load pre-computed matrices to graph representations
2. Training: Train the Graph Autoencoder model with subject-level data splits

Data Sources:
- EEG mode: Load .fif files and compute features
- Matrix mode: Load pre-computed matrices from folder

Usage Examples:
    # EEG mode - compute features from .fif files
    python cookbook/run_pipeline.py --main_path /path/to/eeg_data \\
        --coordinates_file data_scalp/biosemi64.txt
    
    # Matrix mode - load pre-computed matrices
    python cookbook/run_pipeline.py --matrix_dir /path/to/matrices \\
        --coordinates_file data_scalp/biosemi64.txt
    
    # Skip preprocessing and use existing datasets
    python cookbook/run_pipeline.py --skip_preprocessing \\
        --datasets_dir output/preprocessing/datasets \\
        --coordinates_file data_scalp/biosemi64.txt

For more options:
    python cookbook/run_pipeline.py --help

Adaptability to Other Scalp Configurations:
-------------------------------------------
This pipeline is designed to work with any EEG electrode configuration:

1. Create a new coordinates file in data_scalp/ listing electrode labels
   (e.g., GSN-HydroCel-257.txt for 257-channel EGI system)

2. Run with your coordinates file:
   python cookbook/run_pipeline.py --coordinates_file data_scalp/your_system.txt ...

3. For pre-computed matrices, ensure:
   - Matrix rows match the number of electrodes in your coordinates file
   - Folder structure follows: {matrix_dir}/sub-{ID}/ses-{num}/matrix_{i}.npy
   - Or customize with --matrix_path_pattern
"""

import argparse
import os
import sys
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import EEGtoGraph
from train import TrainGAE, normalize_graph_features
from data_loaders import verify_no_data_leakage, save_datasets


def load_electrode_coordinates(coordinates_file, channels=None):
    """
    Load electrode coordinates from file.
    
    Args:
        coordinates_file: Path to file with electrode labels
        channels: Optional list of specific channels to use
        
    Returns:
        coords_df: DataFrame with electrode coordinates
    """
    from eeg_positions import get_elec_coords
    import numpy as np
    
    labels = np.loadtxt(coordinates_file, usecols=(0,), dtype=str)
    coords_data = get_elec_coords(system='1005', as_mne_montage=False)
    coords_df = coords_data[coords_data['label'].isin(labels)].copy()
    
    if channels is not None:
        coords_df = coords_df[coords_df['label'].isin(channels)].copy()
    
    return coords_df


def run_preprocessing(args, coords_df):
    """
    Run preprocessing pipeline to create graph datasets.
    
    Supports two modes:
    - EEG mode: Load .fif files and compute features (when main_path is set)
    - Matrix mode: Load pre-computed matrices (when matrix_dir is set)
    """
    print("\n" + "="*70)
    print("STEP 1: PREPROCESSING")
    print("="*70)
    
    print(f"Using {len(coords_df)} electrodes")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    preprocessing_dir = os.path.join(args.output_dir, 'preprocessing')
    os.makedirs(preprocessing_dir, exist_ok=True)
    
    # Determine mode
    use_matrix_mode = args.matrix_dir is not None
    
    if use_matrix_mode:
        print(f"\nMatrix mode: Loading pre-computed matrices from {args.matrix_dir}")
        print(f"Matrix path pattern: {args.matrix_path_pattern or 'default'}")
    else:
        print(f"\nEEG mode: Computing features from {args.main_path}")
        print(f"Feature type: {args.feature_type}")
        print(f"Path pattern: {args.path_pattern or 'default'}")
    
    # Create graph dataset
    dataset_train, dataset_val, dataset_test = EEGtoGraph.create_graph_dataset(
        coords_df=coords_df,
        main_path=args.main_path,
        matrix_dir=args.matrix_dir,
        window_points=args.window_points,
        k=args.k,
        output_dir=preprocessing_dir,
        feature_type=args.feature_type,
        channels=args.channels,
        path_pattern=args.path_pattern,
        matrix_path_pattern=args.matrix_path_pattern,
        save=True,
        plot_neighbors=not args.no_plot_neighbors,
        n_splits=args.n_splits,
        test_fold=args.test_fold
    )
    
    # Verify no data leakage
    print("\nVerifying data integrity...")
    verify_no_data_leakage(dataset_train, dataset_val, dataset_test)
    
    # Save datasets
    datasets_dir = save_datasets(dataset_train, dataset_val, dataset_test, preprocessing_dir)
    
    return dataset_train, dataset_val, dataset_test, datasets_dir


def run_training(args, dataset_train, dataset_val, dataset_test):
    """Run training pipeline."""
    print("\n" + "="*70)
    print("STEP 2: TRAINING")
    print("="*70)
    
    # Extract underlying graph lists
    train_graphs = dataset_train.data
    val_graphs = dataset_val.data
    test_graphs = dataset_test.data
    
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs:   {len(val_graphs)}")
    print(f"Test graphs:  {len(test_graphs)}")
    
    # Normalize all graphs
    all_graphs = train_graphs + val_graphs + test_graphs
    normalized_all, x_min, x_max, x_range = normalize_graph_features(all_graphs)
    
    train_graphs = normalized_all[:len(train_graphs)]
    val_graphs = normalized_all[len(train_graphs):len(train_graphs)+len(val_graphs)]
    test_graphs = normalized_all[len(train_graphs)+len(val_graphs):]
    
    # Get in_channels from data
    in_channels = train_graphs[0].x.shape[1]
    print(f"Input channels: {in_channels}")
    
    # Parse hidden_dims from string or use default
    if args.hidden_dims:
        hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    else:
        hidden_dims = [64, 64, 32, 16]  # Default: 64 -> 64 -> 32 -> 16 -> latent
    
    # Training configuration
    config = {
        'in_channels': in_channels,
        'hidden_dims': hidden_dims,
        'latent_dim': args.latent_dim,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
    }
    
    print("\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create output directory for training
    training_dir = os.path.join(args.output_dir, 'training')
    os.makedirs(training_dir, exist_ok=True)
    
    # Train model
    trainer = TrainGAE(train_graphs, val_graphs, test_graphs, in_channels)
    
    experiment_name = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    final_model = trainer.train_final(
        config=config,
        save_results=True,
        output_dir=training_dir,
        experiment_name=experiment_name,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project
    )
    
    return final_model, training_dir


def main():
    parser = argparse.ArgumentParser(
        description='Run complete GNN Connectivity pipeline (preprocessing + training)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data source arguments (mutually exclusive unless skip_preprocessing)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--main_path', type=str, default = '/Users/trinidad.borrell/Documents/Work/PhD/Proyects/VGAE/gnn_connectivity/data',
                            help='Path to EEG data directory (for EEG mode)')
    data_group.add_argument('--matrix_dir', type=str,
                            help='Path to pre-computed matrices directory (for matrix mode)')
    
    # Required
    parser.add_argument('--coordinates_file', type=str, default = '/Users/trinidad.borrell/Documents/Work/PhD/Proyects/VGAE/gnn_connectivity/data_scalp/biosemi64.txt',
                        help='Path to electrode coordinates file (e.g., data_scalp/biosemi64.txt)')
    
    # Path patterns for flexibility
    parser.add_argument('--path_pattern', type=str, default=None,
                        help='Custom EEG file path pattern. Placeholders: {main_path}, {subject_id}, {session_num}')
    parser.add_argument('--matrix_path_pattern', type=str, default=None,
                        help='Custom matrix path pattern. Placeholders: {matrix_dir}, {subject_id}, {session_num}, {matrix_idx}')
    
    # Preprocessing options
    parser.add_argument('--window_points', type=int, default=64,
                        help='Number of time points in sliding window (EEG mode only)')
    parser.add_argument('--k', type=int, default=6,
                        help='Number of nearest neighbors for adjacency matrix')
    parser.add_argument('--feature_type', type=str, default='connectivity',
                        choices=['temporal', 'connectivity'],
                        help="Feature type: 'temporal' or 'connectivity' (EEG mode only)")
    parser.add_argument('--channels', type=str, nargs='*', default=None,
                        help='List of channels to use (default: all)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for GroupKFold (subject-level split)')
    parser.add_argument('--test_fold', type=int, default=0,
                        help='Which fold to use as test set')
    parser.add_argument('--no_plot_neighbors', action='store_true',
                        help='Skip k-nearest neighbors visualization (shown by default)')
    
    # Training options
    parser.add_argument('--hidden_dims', type=str, default=None,
                        help='Comma-separated hidden dims (e.g., "64,64,32,16"). Default: 64,64,32,16')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='Latent dimension size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for Adam optimizer')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save all outputs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment (defaults to timestamp)')
    
    # Experiment tracking
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging for training metrics')
    parser.add_argument('--wandb_project', type=str, default='gnn-connectivity',
                        help='wandb project name (default: gnn-connectivity)')
    
    # Pipeline control
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='Skip preprocessing (use existing datasets)')
    parser.add_argument('--datasets_dir', type=str, default=None,
                        help='Path to existing datasets (required if --skip_preprocessing)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.skip_preprocessing and args.main_path is None and args.matrix_dir is None:
        parser.error("Either --main_path, --matrix_dir, or --skip_preprocessing is required")
    
    if args.skip_preprocessing and args.datasets_dir is None:
        parser.error("--datasets_dir required when using --skip_preprocessing")
    
    print("\n" + "="*70)
    print("GNN CONNECTIVITY PIPELINE")
    print("="*70)
    
    if args.skip_preprocessing:
        print("Mode: Loading existing datasets")
        print(f"Datasets: {args.datasets_dir}")
    elif args.matrix_dir:
        print(f"Mode: Matrix (pre-computed)")
        print(f"Matrix dir: {args.matrix_dir}")
    else:
        print(f"Mode: EEG")
        print(f"Data path: {args.main_path}")
        print(f"Feature type: {args.feature_type}")
    
    print(f"Output dir: {args.output_dir}")
    print(f"Coordinates: {args.coordinates_file}")
    if args.channels:
        print(f"Channels: {args.channels}")
    else:
        print("Channels: all")
    
    # Load electrode coordinates
    coords_df = load_electrode_coordinates(args.coordinates_file, args.channels)
    print(f"Loaded {len(coords_df)} electrode coordinates")
    
    # Run preprocessing or load existing datasets
    if args.skip_preprocessing:
        print(f"\nLoading existing datasets from: {args.datasets_dir}")
        dataset_train = torch.load(os.path.join(args.datasets_dir, 'train_dataset.pt'), weights_only=False)
        dataset_val = torch.load(os.path.join(args.datasets_dir, 'val_dataset.pt'), weights_only=False)
        dataset_test = torch.load(os.path.join(args.datasets_dir, 'test_dataset.pt'), weights_only=False)
        datasets_dir = args.datasets_dir
        
        # Verify no data leakage even for loaded datasets
        verify_no_data_leakage(dataset_train, dataset_val, dataset_test)
    else:
        dataset_train, dataset_val, dataset_test, datasets_dir = run_preprocessing(args, coords_df)
    
    # Run training
    final_model, training_dir = run_training(args, dataset_train, dataset_val, dataset_test)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nOutputs saved to:")
    print(f"  Datasets: {datasets_dir}")
    print(f"  Models & plots: {training_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())

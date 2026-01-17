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

1. For standard 10-20/10-10/10-05 systems:
   - Create a file listing electrode labels (one per line or with coordinates)
   - Run with --coordinate_system 1020|1010|1005 to use get_elec_coords

2. For custom electrode systems (e.g., EGI GSN-HydroCel-257):
   - Create a coordinates file with format: label x y z (space-separated)
   - Run with --coordinate_system file to read coordinates directly from file:
   python cookbook/run_pipeline.py --coordinates_file data_scalp/GSN-HydroCel-257.txt \
       --coordinate_system file ...

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
from train import TrainGAE, compute_normalization_stats, apply_normalization
from model import GAE
from data_loaders import verify_no_data_leakage, save_datasets


def load_electrode_coordinates(coordinates_file, channels=None, coordinate_system='1005'):
    """
    Load electrode coordinates from file.
    
    Args:
        coordinates_file: Path to file with electrode labels (and optionally coordinates)
        channels: Optional list of specific channels to use
        coordinate_system: One of '1020', '1010', '1005' to use standard systems via get_elec_coords,
                          or 'file' to read coordinates directly from the coordinates_file
                          (expects format: label x y z per line)
        
    Returns:
        coords_df: DataFrame with electrode coordinates
    """
    import numpy as np
    import pandas as pd
    
    if coordinate_system == 'file':
        # Read coordinates directly from file (format: label x y z)
        data = np.loadtxt(coordinates_file, dtype=str)
        coords_df = pd.DataFrame({
            'label': data[:, 0],
            'x': data[:, 1].astype(float),
            'y': data[:, 2].astype(float),
            'z': data[:, 3].astype(float)
        })
    else:
        # Use standard electrode positioning system
        from eeg_positions import get_elec_coords
        labels = np.loadtxt(coordinates_file, usecols=(0,), dtype=str)
        coords_data = get_elec_coords(system=coordinate_system, as_mne_montage=False)
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
    
    if args.subject_filter:
        print(f"Subject filter: {args.subject_filter.split(',')} (only processing listed subjects)")
    
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
        test_fold=args.test_fold,
        subject_filter=args.subject_filter
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
    
    # Normalize using TRAIN-ONLY statistics (prevents data leakage)
    print("\nComputing normalization stats from TRAIN set only (no data leakage):")
    x_min, x_max, x_range = compute_normalization_stats(train_graphs)
    
    print("Applying normalization IN-PLACE to all splits...")
    apply_normalization(train_graphs, x_min, x_max, x_range, inplace=True)
    apply_normalization(val_graphs, x_min, x_max, x_range, inplace=True)
    apply_normalization(test_graphs, x_min, x_max, x_range, inplace=True)
    
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


def run_inference(args, dataset_train, dataset_val, dataset_test):
    """Run inference using a pre-trained checkpoint.
    
    Loads the model from checkpoint, evaluates on all sets, generates visualizations,
    and saves latent spaces.
    """
    print("\n" + "="*70)
    print("INFERENCE MODE")
    print("="*70)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    config = checkpoint['config']
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A'))}")
    print(f"  Config: {config}")
    
    # Extract underlying graph lists
    train_graphs = dataset_train.data
    val_graphs = dataset_val.data
    test_graphs = dataset_test.data
    
    print(f"\nDatasets loaded:")
    print(f"  Train graphs: {len(train_graphs)}")
    print(f"  Val graphs:   {len(val_graphs)}")
    print(f"  Test graphs:  {len(test_graphs)}")
    
    # Normalize using TRAIN-ONLY statistics (same as training)
    print("\nComputing normalization stats from TRAIN set only:")
    x_min, x_max, x_range = compute_normalization_stats(train_graphs)
    
    print("Applying normalization IN-PLACE to all splits...")
    apply_normalization(train_graphs, x_min, x_max, x_range, inplace=True)
    apply_normalization(val_graphs, x_min, x_max, x_range, inplace=True)
    apply_normalization(test_graphs, x_min, x_max, x_range, inplace=True)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"\n✓ CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"\n⚠ CUDA not available, using: {device}")
    
    # Initialize model with config from checkpoint
    hidden_dims = config.get('hidden_dims', [64, 64, 32, 16])
    model = GAE(
        config['in_channels'],
        hidden_dims=hidden_dims,
        latent_dim=config.get('latent_dim', 2),
        dropout=config.get('dropout', 0.2)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\nModel loaded: {config['in_channels']} -> {hidden_dims} -> {config.get('latent_dim', 2)}")
    
    # Compute MSE on all sets
    from train import compute_mse_on_graphs
    
    print("\n" + "-"*50)
    print("Evaluating model on all sets...")
    print("-"*50)
    
    train_mse = compute_mse_on_graphs(model, train_graphs, device=device)
    val_mse = compute_mse_on_graphs(model, val_graphs, device=device)
    test_mse = compute_mse_on_graphs(model, test_graphs, device=device)
    
    print(f"\nReconstruction MSE:")
    print(f"  Train: {train_mse:.6f}")
    print(f"  Val:   {val_mse:.6f}")
    print(f"  Test:  {test_mse:.6f}")
    
    # Create output directory
    inference_dir = os.path.join(args.output_dir, 'inference')
    os.makedirs(inference_dir, exist_ok=True)
    
    experiment_name = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S") + "_inference"
    
    # Create trainer object for saving utilities
    in_channels = config['in_channels']
    trainer = TrainGAE(train_graphs, val_graphs, test_graphs, in_channels)
    
    # Save latent spaces for all sets
    print("\n" + "-"*50)
    print("Saving latent space representations...")
    print("-"*50)
    trainer.save_latent_spaces(model, inference_dir, experiment_name, device)
    
    # Create visualizations
    print("\n" + "-"*50)
    print("Creating visualizations...")
    print("-"*50)
    trainer._create_subset_visualizations(model, config, inference_dir, experiment_name, device)
    
    # Save evaluation results
    results = {
        'checkpoint': args.checkpoint,
        'epoch': checkpoint.get('epoch', 'N/A'),
        'train_mse': train_mse,
        'val_mse': val_mse,
        'test_mse': test_mse,
        'config': config,
        'n_train_graphs': len(train_graphs),
        'n_val_graphs': len(val_graphs),
        'n_test_graphs': len(test_graphs),
    }
    
    import json
    results_path = os.path.join(inference_dir, f'evaluation_results_{experiment_name}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nEvaluation results saved to: {results_path}")
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETED")
    print("="*70)
    print(f"\nOutputs saved to: {inference_dir}")
    
    return inference_dir


def main():
    parser = argparse.ArgumentParser(
        description='Run complete GNN Connectivity pipeline (preprocessing + training)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data source arguments (mutually exclusive unless skip_preprocessing)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--main_path', type=str, default = '/data/project/eeg_foundation/data/data_250Hz_EGI256/nice_epochs_from_cohen_2/nice_epochs/nice_epochs2',
                            help='Path to EEG data directory (for EEG mode)')
    data_group.add_argument('--matrix_dir', type=str,
                            help='Path to pre-computed matrices directory (for matrix mode)')
    
    # Required
    parser.add_argument('--coordinates_file', type=str, default = '/home/triniborrell/home/projects/gnn_connectivity/data_scalp/GSN-HydroCel-257.txt',
                        help='Path to electrode coordinates file (e.g., data_scalp/biosemi64.txt)')
    parser.add_argument('--coordinate_system', type=str, default='file',
                        choices=['1020', '1010', '1005', 'file'],
                        help="Coordinate system: '1020', '1010', '1005' for standard systems via get_elec_coords, "
                             "or 'file' to read coordinates directly from coordinates_file (format: label x y z)")
    
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
    parser.add_argument('--subject_filter', type=str, default=None,
                        help='Comma-separated list of subject IDs to process (e.g., "AA164,BB178,AL012")')
    parser.add_argument('--no_plot_neighbors', action='store_true',
                        help='Skip k-nearest neighbors visualization (shown by default)')
    
    # Training options
    parser.add_argument('--hidden_dims', type=str, default=None,
                        help='Comma-separated hidden dims (e.g., "64,64,32,16"). Default: 64,64,32,16')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='Latent dimension size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training (larger = faster on GPU)')
    parser.add_argument('--n_epochs', type=int, default=100,
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
                        help='Path to existing datasets (required if --skip_preprocessing or --inference)')
    
    # Inference mode
    parser.add_argument('--inference', action='store_true',
                        help='Run inference only (requires --checkpoint and --datasets_dir)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for inference (e.g., output/training/checkpoints/best_model.pt)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.inference:
        if args.checkpoint is None:
            parser.error("--checkpoint required when using --inference")
        if args.datasets_dir is None:
            parser.error("--datasets_dir required when using --inference")
    elif not args.skip_preprocessing and args.main_path is None and args.matrix_dir is None:
        parser.error("Either --main_path, --matrix_dir, --skip_preprocessing, or --inference is required")
    
    if args.skip_preprocessing and args.datasets_dir is None:
        parser.error("--datasets_dir required when using --skip_preprocessing")
    
    print("\n" + "="*70)
    print("GNN CONNECTIVITY PIPELINE")
    print("="*70)
    
    if args.inference:
        print("Mode: INFERENCE (load checkpoint, evaluate, save latent spaces)")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Datasets: {args.datasets_dir}")
    elif args.skip_preprocessing:
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
    print(f"Coordinates: {args.coordinates_file} (system: {args.coordinate_system})")
    if args.channels:
        print(f"Channels: {args.channels}")
    else:
        print("Channels: all")
    
    # Load electrode coordinates
    coords_df = load_electrode_coordinates(args.coordinates_file, args.channels, args.coordinate_system)
    print(f"Loaded {len(coords_df)} electrode coordinates (system: {args.coordinate_system})")
    
    # Run preprocessing or load existing datasets
    if args.inference or args.skip_preprocessing:
        print(f"\nLoading existing datasets from: {args.datasets_dir}")
        dataset_train = torch.load(os.path.join(args.datasets_dir, 'train_dataset.pt'), weights_only=False)
        dataset_val = torch.load(os.path.join(args.datasets_dir, 'val_dataset.pt'), weights_only=False)
        dataset_test = torch.load(os.path.join(args.datasets_dir, 'test_dataset.pt'), weights_only=False)
        datasets_dir = args.datasets_dir
        
        # Verify no data leakage even for loaded datasets
        verify_no_data_leakage(dataset_train, dataset_val, dataset_test)
    else:
        dataset_train, dataset_val, dataset_test, datasets_dir = run_preprocessing(args, coords_df)
    
    # Run inference or training
    if args.inference:
        inference_dir = run_inference(args, dataset_train, dataset_val, dataset_test)
        
        print("\n" + "="*70)
        print("INFERENCE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nOutputs saved to:")
        print(f"  Datasets: {datasets_dir}")
        print(f"  Inference results: {inference_dir}")
    else:
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

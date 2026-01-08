"""
INFERENCE & EVALUATION PIPELINE
================================
Purpose: Final model evaluation and predictions on unseen data.

Pipeline Position: FINAL STEP
- Input: Trained model, test_loader (or new data)
- Output: Performance metrics, predictions

Key Operations:
1. evaluate_model(): Run model on test_loader ONCE for final metrics
2. predict(): Make predictions on new unseen data
3. visualize_results(): Generate plots and analysis

Critical: This is the ONLY place where test_loader is used. 
Test set provides unbiased estimate of real-world performance.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from model import GAE
from preprocessing import EEGtoGraph

# Plotting style parameters
COLOR = "black"
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (14, 9),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "legend.fontsize": "medium",
        "legend.title_fontsize": 18,
        "axes.titlesize": 18,
        "axes.labelsize": "large",
        "ytick.labelsize": 12,
        "xtick.labelsize": 12,
        # colour-consistent theme
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "grid.color": COLOR,
    }
)
plt.rcParams["text.latex.preamble"] = r"\usepackage[version=3]{mhchem}"


class InferenceGAE:
    def __init__(self, model_path, data):
        """
        Initialize inference with a trained model
        
        Args:
            model_path: Path to the saved model checkpoint
            data: PyTorch Geometric Data object with test data
        """
        self.model_path = model_path
        self.data = data
        self.data_raw = data.x.clone()
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, weights_only=False)
        
        # Extract config - handle both old and new checkpoint formats
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.in_channels = config['in_channels']
            self.hidden_channels = config['hidden_channels']
            self.latent_dim = config['latent_dim']
            self.num_layers = config.get('num_layers', 4)
            self.dropout = config.get('dropout', 0.1)
        else:
            # Legacy format
            self.in_channels = checkpoint.get('in_channels', data.x.shape[1])
            self.hidden_channels = checkpoint.get('hidden_channels', 64)
            self.latent_dim = checkpoint.get('latent_dim', 2)
            self.num_layers = checkpoint.get('num_layers', 4)
            self.dropout = checkpoint.get('dropout', 0.1)
        
        # Normalize input data to [-1, 1] (same as training)
        self.x_min = data.x.min()
        self.x_max = data.x.max()
        self.x_range = self.x_max - self.x_min
        
        if self.x_range > 0:
            self.data.x = 2 * (data.x - self.x_min) / self.x_range - 1
        else:
            self.data.x = torch.zeros_like(data.x)
        
        # Initialize and load model
        self.model = GAE(
            self.in_channels,
            self.hidden_channels,
            self.latent_dim,
            self.num_layers,
            self.dropout
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Architecture: in={self.in_channels}, hidden={self.hidden_channels}, latent={self.latent_dim}, layers={self.num_layers}, dropout={self.dropout}")
    
    def denormalize(self, x_normalized):
        """Convert from [-1, 1] back to original scale"""
        if self.x_range > 0:
            return (x_normalized + 1) / 2 * self.x_range + self.x_min
        else:
            return x_normalized + self.x_min
    
    def run_inference(self):
        """Run inference on the data"""
        print("\n" + "="*60)
        print("Running Inference")
        print("="*60 + "\n")
        
        # Print data statistics
        print("Original (raw) data statistics:")
        print(f"  Shape: {self.data_raw.shape}")
        print(f"  Range: [{self.data_raw.min():.6f}, {self.data_raw.max():.6f}]")
        print(f"  Mean: {self.data_raw.mean():.6f}, Std: {self.data_raw.std():.6f}")
        
        print("\nNormalized data statistics (used for inference):")
        print(f"  Shape: {self.data.x.shape}")
        print(f"  Range: [{self.data.x.min():.6f}, {self.data.x.max():.6f}]")
        print(f"  Mean: {self.data.x.mean():.6f}, Std: {self.data.x.std():.6f}")
        print()
        
        # Run forward pass
        with torch.no_grad():
            x_reconstructed, z = self.model(self.data.x, self.data.edge_index)
        
        # Calculate metrics in normalized space
        mse_norm = F.mse_loss(x_reconstructed, self.data.x).item()
        mae_norm = torch.mean(torch.abs(x_reconstructed - self.data.x)).item()
        
        # Calculate metrics in original scale
        original_denorm = self.denormalize(self.data.x)
        reconstruction_denorm = self.denormalize(x_reconstructed)
        mse_orig = F.mse_loss(reconstruction_denorm, original_denorm).item()
        mae_orig = torch.mean(torch.abs(reconstruction_denorm - original_denorm)).item()
        
        print("\n" + "="*60)
        print("Inference Results")
        print("="*60)
        print(f"Original features shape: {self.data.x.shape}")
        print(f"Latent representation shape: {z.shape}")
        print(f"Reconstructed features shape: {x_reconstructed.shape}")
        
        print("\nReconstruction Error:")
        print(f"  Normalized space [-1, 1]: MSE={mse_norm:.6f}, MAE={mae_norm:.6f}")
        print(f"  Original scale: MSE={mse_orig:.10f}, MAE={mae_orig:.10f}")
        
        print("\nLatent space values (first 10 nodes):")
        print(z[:10].squeeze())
        print("="*60 + "\n")
        
        return x_reconstructed, z
    
    def visualize_results(self, output_dir='../output/inference', experiment_name=None):
        """
        Create visualization of inference results
        
        Args:
            output_dir: Directory to save outputs
            experiment_name: Name for this experiment (defaults to timestamp)
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate predictions
        with torch.no_grad():
            x_reconstructed, z = self.model(self.data.x, self.data.edge_index)
            
            # Calculate reconstruction error in normalized space
            error_normalized = torch.abs(self.data.x - x_reconstructed)
            
            # Convert to numpy for plotting (use normalized values)
            original = self.data.x.cpu().numpy()
            reconstruction = x_reconstructed.cpu().numpy()
            error_np = error_normalized.cpu().numpy()
            latent = z.cpu().numpy()
        
        # Create 4-column visualization
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # Plot 1: Original Feature Matrix (normalized)
        im1 = axes[0].imshow(original, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[0].set_title('Original Feature Matrix\n(Normalized [-1, 1])', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Nodes')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Reconstructed Feature Matrix (normalized)
        im2 = axes[1].imshow(reconstruction, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[1].set_title('Reconstruction\n(Normalized [-1, 1])', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Nodes')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot 3: Reconstruction Error (in normalized space)
        im3 = axes[2].imshow(error_np, aspect='auto', cmap='Reds')
        axes[2].set_title('Reconstruction Error\n(Normalized Space)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Features')
        axes[2].set_ylabel('Nodes')
        plt.colorbar(im3, ax=axes[2])
        
        # Plot 4: Latent Space Representation
        if self.latent_dim == 1:
            # For 1D latent space, plot as a heatmap
            im4 = axes[3].imshow(latent, aspect='auto', cmap='coolwarm')
            axes[3].set_title('Latent Space (1D)', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Latent Dimension')
            axes[3].set_ylabel('Nodes')
            plt.colorbar(im4, ax=axes[3])
        elif self.latent_dim == 2:
            # For 2D latent space, scatter plot
            axes[3].scatter(latent[:, 0], latent[:, 1], alpha=0.6, c=range(len(latent)), cmap='coolwarm')
            axes[3].set_title('Latent Space (2D)', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Latent Dim 1')
            axes[3].set_ylabel('Latent Dim 2')
            axes[3].grid(True, alpha=0.3)
        else:
            # For higher dimensions, show as heatmap
            im4 = axes[3].imshow(latent, aspect='auto', cmap='coolwarm')
            axes[3].set_title(f'Latent Space ({self.latent_dim}D)', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Latent Dimensions')
            axes[3].set_ylabel('Nodes')
            plt.colorbar(im4, ax=axes[3])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_dir, f'gae_inference_{experiment_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")
        plt.close()
        
        return plot_path


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with a trained GAE model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--main_path',
        type=str,
        required=True,
        help='Path to the main EEG data directory'
    )
    parser.add_argument(
        '--coordinates_file',
        type=str,
        required=True,
        help='Path to biosemi64.txt file with electrode labels'
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--subject_id',
        type=str,
        default='01',
        help='Subject ID for inference'
    )
    parser.add_argument(
        '--session_num',
        type=str,
        default='01',
        help='Session number for inference'
    )
    parser.add_argument(
        '--path_pattern',
        type=str,
        default=None,
        help='Custom EEG file path pattern. Placeholders: {main_path}, {subject_id}, {session_num}'
    )
    parser.add_argument(
        '--window_points',
        type=int,
        default=64,
        help='Number of time points in the window'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=0,
        help='Epoch number to process'
    )
    parser.add_argument(
        '--k_neighbors',
        type=int,
        default=6,
        help='Number of nearest neighbors for adjacency matrix'
    )
    parser.add_argument(
        '--feature_type',
        type=str,
        default='connectivity',
        choices=['temporal', 'connectivity'],
        help="Feature type: 'temporal' (raw time window) or 'connectivity' (Pearson correlation)"
    )
    parser.add_argument(
        '--channels',
        type=str,
        nargs='*',
        default=None,
        help='List of channel names to select (e.g., --channels Fp1 Fp2 F3)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../output/inference',
        help='Directory to save inference outputs'
    )
    parser.add_argument(
        '--preprocessing_output_dir',
        type=str,
        default='../output/preprocessing',
        help='Directory where preprocessing outputs are stored'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Name for this inference run (defaults to timestamp)'
    )
    
    args = parser.parse_args()
    
    # Load coordinates
    try:
        from eeg_positions import get_elec_coords
        
        # Load labels from file
        labels = np.loadtxt(args.coordinates_file, usecols=(0,), dtype=str)
        
        # Get electrode coordinates
        coords_data = get_elec_coords(system='1005', as_mne_montage=False)
        
        # Filter to only include biosemi64 electrodes
        coords_df = coords_data[coords_data['label'].isin(labels)].copy()
        
        print(f"Loaded {len(coords_df)} electrode coordinates")
        
    except Exception as e:
        print(f"\nERROR loading coordinates: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create graph from test data
    print("\n" + "="*70)
    print("INFERENCE PIPELINE")
    print("="*70)
    print(f"\nSubject: {args.subject_id}")
    print(f"Session: {args.session_num}")
    print(f"Model: {args.model_path}")
    
    try:
        print("\nLoading test data...")
        data, adjacency, features, labels_list, distance_matrix = EEGtoGraph.create_graph(
            coords_df=coords_df,
            main_path=args.main_path,
            subject_id=args.subject_id,
            session_num=args.session_num,
            window_points=args.window_points,
            epoch=args.epoch,
            k=args.k_neighbors,
            output_dir=args.preprocessing_output_dir,
            feature_type=args.feature_type,
            channels=args.channels,
            path_pattern=args.path_pattern,
            save=False,  # Don't save during inference
            plot_neighbors=False
        )
        
        print("\nInitializing inference...")
        inference = InferenceGAE(args.model_path, data)
        
        # Run inference
        x_reconstructed, z = inference.run_inference()
        
        # Create visualization
        print("\nGenerating visualization...")
        inference.visualize_results(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )
        
        print("\n" + "="*70)
        print("INFERENCE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nOutputs saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

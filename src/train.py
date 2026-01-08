"""
TRAINING PIPELINE
=================
Purpose: Hyperparameter optimization with Ray Tune and final model training.

Pipeline Position: SECOND STEP
- Input: train_dataset, val_dataset, test_dataset from preprocessing.py
- Output: Trained model with optimized hyperparameters

1. optimize_hyperparameters(): 
   - Uses Ray Tune to find best hyperparameters (lr, batch_size, hidden_dims, etc.)

2. train_final_model():
   - Train model on train_dataset with best hyperparameters
   - Use test_dataset for early stopping/monitoring only
   - Return fully trained model

Critical: test_dataset is NEVER used here. 
Validation set guides both hyperparameter selection and early stopping, but doesn't update model weights.
"""

import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from datetime import datetime
from model import GAE
#import ray
#from ray import tune
#from ray.tune.schedulers import ASHAScheduler

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
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "grid.color": COLOR,
    }
)
plt.rcParams["text.latex.preamble"] = r"\usepackage[version=3]{mhchem}"


def normalize_graph_features(graph_list):
    """Normalize all graphs in a list to [-1, 1] range."""
    all_features = torch.cat([g.x for g in graph_list], dim=0)
    x_min = all_features.min()
    x_max = all_features.max()
    x_range = x_max - x_min
    
    normalized_graphs = []
    for g in graph_list:
        g_copy = g.clone()
        if x_range > 0:
            g_copy.x = 2 * (g.x - x_min) / x_range - 1
        else:
            g_copy.x = torch.zeros_like(g.x)
        normalized_graphs.append(g_copy)
    
    return normalized_graphs, x_min, x_max, x_range


def compute_mse_on_graphs(model, graph_list, device=None):
    """Compute average MSE reconstruction error on a list of graphs."""
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    num_batches = 0
    
    loader = PyGDataLoader(graph_list, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_recon, _ = model(batch.x, batch.edge_index)
            loss = criterion(x_recon, batch.x)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_one_epoch(model, optimizer, criterion, graph_list, batch_size=64, device=None):
    """Train model for one epoch with mini-batching."""
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    if device is None:
        device = torch.device('cpu')
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Use PyG DataLoader for efficient batching
    loader = PyGDataLoader(graph_list, batch_size=batch_size, shuffle=True, num_workers=0)
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        x_recon, _ = model(batch.x, batch.edge_index)
        loss = criterion(x_recon, batch.x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def ray_trainable(config, train_graphs=None, val_graphs=None):
    """Ray Tune trainable function. Trains on train_graphs, reports val MSE."""
    from ray.air import session
    
    in_channels = config['in_channels']
    hidden_channels = config['hidden_channels']
    latent_dim = config['latent_dim']
    num_layers = config.get('num_layers', 4)
    dropout = config.get('dropout', 0.1)
    lr = config['lr']
    n_epochs = config['n_epochs']
    
    model = GAE(in_channels, hidden_channels, latent_dim, num_layers, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, train_graphs)
        val_mse = compute_mse_on_graphs(model, val_graphs)
        
        session.report({"val_mse": val_mse, "train_loss": train_loss, "epoch": epoch})


class TrainGAE:
    """Wrapper for final model training after hyperparameter tuning."""
    
    def __init__(self, train_graphs, val_graphs, test_graphs, in_channels):
        self.train_graphs = train_graphs
        self.val_graphs = val_graphs
        self.test_graphs = test_graphs
        self.in_channels = in_channels

    def save_model_and_visualizations(self, model, config, loss_history=None, output_dir='../output/training', experiment_name=None):
        """
        Save the trained model and create 4-column visualization plot plus loss plot
        
        Args:
            model: Trained GAE model
            config: Configuration dict with hyperparameters
            loss_history: List of loss values across epochs
            output_dir: Directory to save outputs
            experiment_name: Name for this experiment (defaults to timestamp)
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = os.path.join(output_dir, 'models')
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f'gae_model_{experiment_name}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Plot loss history if available
        if loss_history is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(loss_history, linewidth=2)
            ax.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            loss_plot_path = os.path.join(plots_dir, f'loss_history_{experiment_name}.png')
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            print(f"Loss history plot saved to: {loss_plot_path}")
            plt.close()
        
        # Create GAE visualization with 5 random validation graphs
        self._create_gae_visualization(model, config, plots_dir, experiment_name)
        
        return model_path
    
    def _create_gae_visualization(self, model, config, plots_dir, experiment_name, device=None):
        """Create 5x4 grid visualization of GAE reconstructions from validation set."""
        import numpy as np
        
        if device is None:
            device = next(model.parameters()).device
        
        # Select 5 random validation graphs
        num_samples = min(5, len(self.val_graphs))
        indices = np.random.choice(len(self.val_graphs), num_samples, replace=False)
        sample_graphs = [self.val_graphs[i] for i in indices]
        
        model.eval()
        
        # Create figure with 5 rows, 4 columns
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        latent_dim = config.get('latent_dim', 2)
        
        with torch.no_grad():
            for row_idx, graph in enumerate(sample_graphs):
                # Move graph to device
                graph = graph.to(device)
                # Forward pass
                x_recon, z = model(graph.x, graph.edge_index)
                
                # Calculate error in normalized space
                error_normalized = torch.abs(graph.x - x_recon)
                
                # Convert to numpy
                original = graph.x.cpu().numpy()
                reconstruction = x_recon.cpu().numpy()
                error_np = error_normalized.cpu().numpy()
                latent = z.cpu().numpy()
                
                # Column 1: Original Feature Matrix (normalized [-1, 1])
                im1 = axes[row_idx, 0].imshow(original, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
                axes[row_idx, 0].set_title(f'Graph {indices[row_idx]} - Original\n(Normalized [-1, 1])', fontsize=12, fontweight='bold')
                axes[row_idx, 0].set_xlabel('Features')
                axes[row_idx, 0].set_ylabel('Nodes')
                plt.colorbar(im1, ax=axes[row_idx, 0])
                
                # Column 2: Reconstructed Feature Matrix (normalized [-1, 1])
                im2 = axes[row_idx, 1].imshow(reconstruction, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
                axes[row_idx, 1].set_title(f'Reconstruction\n(Normalized [-1, 1])', fontsize=12, fontweight='bold')
                axes[row_idx, 1].set_xlabel('Features')
                axes[row_idx, 1].set_ylabel('Nodes')
                plt.colorbar(im2, ax=axes[row_idx, 1])
                
                # Column 3: Reconstruction Error
                im3 = axes[row_idx, 2].imshow(error_np, aspect='auto', cmap='Reds')
                axes[row_idx, 2].set_title(f'Reconstruction Error\n(MAE: {error_np.mean():.6f})', fontsize=12, fontweight='bold')
                axes[row_idx, 2].set_xlabel('Features')
                axes[row_idx, 2].set_ylabel('Nodes')
                plt.colorbar(im3, ax=axes[row_idx, 2])
                
                # Column 4: Latent Space Representation
                if latent_dim == 1:
                    # For 1D latent space, plot as a heatmap
                    im4 = axes[row_idx, 3].imshow(latent, aspect='auto', cmap='coolwarm')
                    axes[row_idx, 3].set_title('Latent Space (1D)', fontsize=12, fontweight='bold')
                    axes[row_idx, 3].set_xlabel('Latent Dimension')
                    axes[row_idx, 3].set_ylabel('Nodes')
                    plt.colorbar(im4, ax=axes[row_idx, 3])
                elif latent_dim == 2:
                    # For 2D latent space, scatter plot
                    scatter = axes[row_idx, 3].scatter(latent[:, 0], latent[:, 1], alpha=0.6, 
                                                       c=range(len(latent)), cmap='coolwarm')
                    axes[row_idx, 3].set_title('Latent Space (2D)', fontsize=12, fontweight='bold')
                    axes[row_idx, 3].set_xlabel('Latent Dim 1')
                    axes[row_idx, 3].set_ylabel('Latent Dim 2')
                    axes[row_idx, 3].grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=axes[row_idx, 3])
                else:
                    # For higher dimensions, show as heatmap
                    im4 = axes[row_idx, 3].imshow(latent, aspect='auto', cmap='coolwarm')
                    axes[row_idx, 3].set_title(f'Latent Space ({latent_dim}D)', fontsize=12, fontweight='bold')
                    axes[row_idx, 3].set_xlabel('Latent Dimensions')
                    axes[row_idx, 3].set_ylabel('Nodes')
                    plt.colorbar(im4, ax=axes[row_idx, 3])
        
        plt.tight_layout()
        
        # Save plot
        viz_path = os.path.join(plots_dir, f'gae_visualization_{experiment_name}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"GAE visualization saved to: {viz_path}")
        plt.close()

    def train_final(self, config, save_results=False, output_dir='../output/training', experiment_name=None):
        """Train final model on train+val with best config, evaluate on test."""
        print(f"\n{'='*60}")
        print("Training Final Model on Train+Val")
        print(f"{'='*60}\n")
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Combine train and val for final training
        combined_graphs = self.train_graphs + self.val_graphs
        batch_size = config.get('batch_size', 64)
        
        # Get hidden_dims from config, or build from hidden_channels
        hidden_dims = config.get('hidden_dims', [64, 64, 32, 16])
        
        model = GAE(
            config['in_channels'], 
            hidden_dims=hidden_dims,
            latent_dim=config.get('latent_dim', 2),
            dropout=config.get('dropout', 0.2)
        )
        model = model.to(device)
        
        # Optimizer with weight decay
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.get('lr', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler - reduce on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        
        criterion = torch.nn.MSELoss()
        
        loss_history = []
        best_loss = float('inf')
        
        for epoch in range(config['n_epochs']):
            train_loss = train_one_epoch(model, optimizer, criterion, combined_graphs, 
                                         batch_size=batch_size, device=device)
            loss_history.append(train_loss)
            
            # Step scheduler
            scheduler.step(train_loss)
            
            if train_loss < best_loss:
                best_loss = train_loss
            
            if epoch % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch:03d}, Loss: {train_loss:.6f}, LR: {lr:.6f}')
        
        # Final evaluation on test set
        test_mse = compute_mse_on_graphs(model, self.test_graphs, device=device)
        
        print(f"\n{'='*60}")
        print("Final Results")
        print(f"{'='*60}")
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Best Train Loss: {best_loss:.6f}")
        print(f"{'='*60}\n")
        
        if save_results:
            self.save_model_and_visualizations(model, config, loss_history, output_dir, experiment_name)
        
        return model

def main():
    """
    Main training function
    
    Loads train/val/test datasets, optionally runs Ray Tune, then trains final model.
    """
    parser = argparse.ArgumentParser(description='Train Graph Autoencoder with Ray Tune')

    parser.add_argument('--train_dataset', type=str, required=True,
                        help='Path to train_dataset.pt')
    parser.add_argument('--val_dataset', type=str, required=True,
                        help='Path to val_dataset.pt')
    parser.add_argument('--test_dataset', type=str, required=True,
                        help='Path to test_dataset.pt')
    parser.add_argument('--tuning', action='store_true',
                        help='Enable hyperparameter tuning with Ray Tune')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='Number of input channels (default: 3)')
    parser.add_argument('--hidden_channels', type=int, default=32,
                        help='Number of hidden channels (default: 64)')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='Latent dimension size (default: 2)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of SAGE layers in encoder/decoder (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability (default: 0.1)')
    parser.add_argument('--n_epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--save', action='store_true',
                        help='Save model and visualizations')
    parser.add_argument('--output_dir', type=str, default='../output/training',
                        help='Directory to save outputs (default: ../output/training)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment (defaults to timestamp)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of Ray Tune samples (default: 10)')
    
    args = parser.parse_args()

    # Load datasets
    print("\n" + "="*70)
    print("Loading datasets...")
    print("="*70)
    train_dataset = torch.load(args.train_dataset, weights_only=False)
    val_dataset = torch.load(args.val_dataset, weights_only=False)
    test_dataset = torch.load(args.test_dataset, weights_only=False)
    
    # Extract underlying graph lists from GraphAutoencoderDataset
    train_graphs = train_dataset.data
    val_graphs = val_dataset.data
    test_graphs = test_dataset.data
    
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs:   {len(val_graphs)}")
    print(f"Test graphs:  {len(test_graphs)}")
    
    # Normalize all graphs
    all_graphs = train_graphs + val_graphs + test_graphs
    normalized_all, x_min, x_max, x_range = normalize_graph_features(all_graphs)
    
    train_graphs = normalized_all[:len(train_graphs)]
    val_graphs = normalized_all[len(train_graphs):len(train_graphs)+len(val_graphs)]
    test_graphs = normalized_all[len(train_graphs)+len(val_graphs):]
    
    # Get in_channels from first graph or use provided value
    data_in_channels = train_graphs[0].x.shape[1]
    in_channels = args.in_channels if args.in_channels != data_in_channels else data_in_channels
    print(f"Input channels (from data): {data_in_channels}")
    print(f"Input channels (used): {in_channels}")
    
    # Warn if mismatch
    if in_channels != data_in_channels:
        print(f"WARNING: Provided in_channels ({in_channels}) differs from data ({data_in_channels}). Using data value.")
        in_channels = data_in_channels
    
    # Hyperparameter optimization with Ray Tune
    '''
    if args.tuning:
        print("\n" + "="*70)
        print("Running Ray Tune hyperparameter optimization")
        print("="*70)
        
        # ray.init(ignore_reinit_error=True)
        
        search_space = {
            'in_channels': in_channels,
            'hidden_channels': tune.choice([32, 64, 128]),
            'latent_dim': tune.choice([1, 2, 4, 8]),
            'n_epochs': tune.choice([200, 500, 1000]),
            'lr': tune.loguniform(1e-4, 1e-2),
        }
        
        scheduler = ASHAScheduler(
            max_t=1000,
            grace_period=100,
            reduction_factor=2
        )
        
        analysis = tune.run(
            tune.with_parameters(ray_trainable, train_graphs=train_graphs, val_graphs=val_graphs),
            config=search_space,
            num_samples=args.num_samples,
            scheduler=scheduler,
            metric="val_mse",
            mode="min",
            resources_per_trial={"cpu": 2},
            verbose=1
        )
        
        best_config = analysis.best_config
        print("\nBest config:", best_config)
        print("Best val MSE:", analysis.best_result["val_mse"])
        
        ray.shutdown()
    '''
   # else:
    best_config = {
            'in_channels': in_channels,
            'hidden_channels': args.hidden_channels,
            'latent_dim': args.latent_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'n_epochs': args.n_epochs,
            'lr': args.lr,
        }
    
    # Train final model on train+val with best config
    print("\n" + "="*70)
    print("Training final model with best hyperparameters")
    print("="*70)
    
    trainer = TrainGAE(train_graphs, val_graphs, test_graphs, in_channels)
    final_model = trainer.train_final(
        config=best_config,
        save_results=args.save,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )
    
    return final_model, trainer

if __name__ == '__main__':
    main()
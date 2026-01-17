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
import gc
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from datetime import datetime
from model import GAE

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")


def print_resource_usage(label=""):
    """Print current GPU and CPU resource usage for debugging.
    
    Args:
        label: String label to identify where in the code this is called
    """
    print(f"\n{'─'*60}")
    print(f"📊 RESOURCE USAGE: {label}")
    print(f"{'─'*60}")
    
    # GPU metrics
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i} ({torch.cuda.get_device_name(i)}):")
            print(f"    Allocated: {allocated:.3f} GB")
            print(f"    Reserved:  {reserved:.3f} GB")
            print(f"    Max Alloc: {max_allocated:.3f} GB")
            print(f"    Total:     {total:.1f} GB")
            print(f"    Usage:     {100*allocated/total:.1f}%")
    else:
        print("  GPU: Not available (CUDA not detected)")
    
    # CPU/RAM metrics
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        print(f"  CPU:")
        print(f"    Process CPU:  {cpu_percent:.1f}%")
        print(f"    Process RAM:  {mem_info.rss / 1024**3:.3f} GB")
        print(f"    System RAM:   {psutil.virtual_memory().percent:.1f}% used")
    else:
        print("  CPU: psutil not available (pip install psutil)")
    
    print(f"{'─'*60}\n")
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


def compute_normalization_stats(graph_list):
    """Compute normalization statistics from a list of graphs (train set only).
    
    Args:
        graph_list: List of graphs to compute stats from (should be training set only)
        
    Returns:
        x_min, x_max, x_range: Normalization parameters
    """
    all_features = torch.cat([g.x for g in graph_list], dim=0)
    x_min = all_features.min()
    x_max = all_features.max()
    x_range = x_max - x_min
    print(f"  Normalization stats computed from {len(graph_list)} graphs")
    print(f"  x_min: {x_min:.6f}, x_max: {x_max:.6f}, x_range: {x_range:.6f}")
    return x_min, x_max, x_range


def apply_normalization(graph_list, x_min, x_max, x_range, inplace=True):
    """Apply normalization to graphs using pre-computed stats.
    
    Args:
        graph_list: List of graphs to normalize
        x_min, x_max, x_range: Pre-computed normalization parameters (from train set)
        inplace: If True, modify graphs in-place (saves memory). Default: True
        
    Returns:
        List of normalized graphs (same objects if inplace=True)
    """
    if inplace:
        for g in graph_list:
            if x_range > 0:
                g.x = 2 * (g.x - x_min) / x_range - 1
            else:
                g.x = torch.zeros_like(g.x)
        return graph_list
    else:
        normalized_graphs = []
        for g in graph_list:
            g_copy = g.clone()
            if x_range > 0:
                g_copy.x = 2 * (g.x - x_min) / x_range - 1
            else:
                g_copy.x = torch.zeros_like(g.x)
            normalized_graphs.append(g_copy)
        return normalized_graphs


def normalize_graph_features(graph_list):
    """Normalize all graphs in a list to [-1, 1] range.
    
    DEPRECATED: Use compute_normalization_stats + apply_normalization instead
    to avoid data leakage. This function computes stats from all provided graphs.
    """
    x_min, x_max, x_range = compute_normalization_stats(graph_list)
    normalized_graphs = apply_normalization(graph_list, x_min, x_max, x_range)
    return normalized_graphs, x_min, x_max, x_range


def compute_mse_on_graphs(model, graph_list, device=None, batch_size=64):
    """Compute average MSE reconstruction error on a list of graphs."""
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    num_batches = 0
    
    # Larger batch size for evaluation (no gradients = less memory)
    loader = PyGDataLoader(graph_list, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)  # Async transfer
            x_recon, _ = model(batch.x, batch.edge_index)
            loss = criterion(x_recon, batch.x)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_one_epoch(model, optimizer, criterion, loader, device=None):
    """Train model for one epoch with mini-batching.
    
    Args:
        model: The model to train
        optimizer: Optimizer
        criterion: Loss function
        loader: Pre-created PyG DataLoader (created once, reused every epoch)
        device: Device to use
    """
    if device is None:
        device = torch.device('cpu')
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device, non_blocking=True)  # Async CPU->GPU transfer
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
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
            
            # Check if loss_history is a dict with train/val/test keys
            if isinstance(loss_history, dict):
                if 'train' in loss_history:
                    ax.plot(loss_history['train'], linewidth=2, label='Train', color='blue')
                if 'val' in loss_history:
                    ax.plot(loss_history['val'], linewidth=2, label='Validation', color='orange')
                if 'test' in loss_history:
                    ax.plot(loss_history['test'], linewidth=2, label='Test', color='green', linestyle='--')
                ax.legend(fontsize=12)
            else:
                # Backwards compatibility: single list
                ax.plot(loss_history, linewidth=2, label='Train')
            
            ax.set_title('Loss Over Epochs (Overfitting Check)', fontsize=14, fontweight='bold')
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

    def train_final(self, config, save_results=False, output_dir='../output/training', experiment_name=None, use_wandb=True, wandb_project='gnn-connectivity'):
        """Train final model on train+val with best config, evaluate on test.
        
        Args:
            config: Configuration dict with hyperparameters
            save_results: Whether to save model and visualizations
            output_dir: Directory to save outputs
            experiment_name: Name for this experiment
            use_wandb: Whether to use wandb for logging (default: True)
            wandb_project: wandb project name (default: 'gnn-connectivity')
        """
        print(f"\n{'='*60}")
        print("Training Final Model on Train+Val")
        print(f"{'='*60}\n")
        
        # Device setup - force GPU if available
        if torch.cuda.is_available():
            device = torch.device('cuda:0')   # Use first GPU
            print(f"✓ CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"⚠ CUDA not available, using: {device}")
     #   device = torch.device('cpu')
        print(f"Final device: {device}")
        
        # Initialize wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=wandb_project,
                name=experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S"),
                config=config,
                tags=["GAE", "training"]
            )
            wandb.config.update({"device": str(device)})
            print("wandb initialized successfully")
        elif use_wandb and not WANDB_AVAILABLE:
            print("wandb requested but not available. Continuing without wandb.")
            use_wandb = False
        
        # Import PyG DataLoader once
        from torch_geometric.loader import DataLoader as PyGDataLoader
        
        # Combine train and val for final training
        combined_graphs = self.train_graphs + self.val_graphs
        batch_size = config.get('batch_size', 64)
        
        # Create DataLoader ONCE (not every epoch - major speedup)
        # System says max 2 workers - respect that limit
        num_workers = 2
        train_loader = PyGDataLoader(
            combined_graphs, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"DataLoader: {len(train_loader)} batches, {num_workers} workers, batch_size={batch_size}")
        
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
        
        loss_history = {'train': [], 'val': []}
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        
        # Compute validation loss every N epochs (balance speed vs monitoring)
        eval_interval = max(1, config['n_epochs'] // 50)  # ~50 eval points
        
        # Setup checkpoint directory for best model
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Resource monitoring before training
        print_resource_usage("BEFORE TRAINING (model loaded to device)")
        
        # Reset GPU memory stats for accurate tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for epoch in range(config['n_epochs']):
            train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device=device)
            loss_history['train'].append(train_loss)
            
            # Compute val loss periodically (not every epoch for speed)
            # NOTE: Test set is held out - only evaluated once at end with best model
            if epoch % eval_interval == 0 or epoch == config['n_epochs'] - 1:
                val_loss = compute_mse_on_graphs(model, self.val_graphs, device=device)
                loss_history['val'].append(val_loss)
                
                # Track best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    
                    # Save best model checkpoint
                    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'config': config
                    }, best_checkpoint_path)
                    print(f"  🏆 New best model saved (val_loss={val_loss:.6f}) at epoch {epoch}")
            else:
                # Interpolate for plotting (use last known value)
                if loss_history['val']:
                    loss_history['val'].append(loss_history['val'][-1])
                else:
                    loss_history['val'].append(train_loss)
            
            # Step scheduler
            scheduler.step(train_loss)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": loss_history['val'][-1],
                    "best_val_loss": best_val_loss,
                    "learning_rate": current_lr
                })
            
            if epoch % 10 == 0:
                # Monitor GPU memory usage
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_cached = torch.cuda.memory_reserved(0) / 1024**3
                    print(f'Epoch {epoch:03d}, Train: {train_loss:.6f}, Val: {loss_history["val"][-1]:.6f}, Best Val: {best_val_loss:.6f}, LR: {current_lr:.6f}, GPU: {gpu_memory:.3f}GB/{gpu_cached:.3f}GB')
                else:
                    print(f'Epoch {epoch:03d}, Train: {train_loss:.6f}, Val: {loss_history["val"][-1]:.6f}, Best Val: {best_val_loss:.6f}, LR: {current_lr:.6f}')
            
            # Full resource dump at key epochs (first, middle, last)
            if epoch == 0 or epoch == config['n_epochs'] // 2 or epoch == config['n_epochs'] - 1:
                print_resource_usage(f"DURING TRAINING (epoch {epoch}/{config['n_epochs']})")
        
        # Resource monitoring after training loop
        print_resource_usage("AFTER TRAINING LOOP (before final eval)")
        
        # Load best model for final evaluation on held-out test set
        print(f"\n{'='*60}")
        print(f"Loading best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
        print(f"{'='*60}")
        model.load_state_dict(best_model_state)
        model = model.to(device)
        
        # Final evaluation on held-out test set (only done once with best model)
        test_mse = compute_mse_on_graphs(model, self.test_graphs, device=device)
        
        print(f"\n{'='*60}")
        print("Final Results (Best Model on Held-Out Test Set)")
        print(f"{'='*60}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Val Loss: {best_val_loss:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        print(f"{'='*60}\n")
        
        # Add test loss to history for final point only
        loss_history['test'] = [None] * (len(loss_history['train']) - 1) + [test_mse]
        
        # Log final metrics to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "test_mse": test_mse,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch
            })
            wandb.summary["test_mse"] = test_mse
            wandb.summary["best_val_loss"] = best_val_loss
            wandb.summary["best_epoch"] = best_epoch
        
        if save_results:
            self.save_model_and_visualizations(model, config, loss_history, output_dir, experiment_name)
            # Save latent space representations for all subsets
            self.save_latent_spaces(model, output_dir, experiment_name, device)
            # Create visualizations for each subset
            self._create_subset_visualizations(model, config, output_dir, experiment_name, device)
        
        # Finish wandb run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        return model
    
    def save_latent_spaces(self, model, output_dir, experiment_name, device=None):
        """
        Save latent space representations for train, val, and test sets.
        
        Saves a JSON file for each subset with structure:
        {
            "electrodes": ["Fp1", "Fp2", ...],
            "data": {
                "subject_id": {
                    "session_num": {
                        "matrix_idx": [[latent_0, latent_1], [latent_0, latent_1], ...]
                    }
                }
            }
        }
        
        Also saves the raw latent tensors as .pt files.
        """
        import json
        
        # Use CPU for latent space extraction to avoid GPU memory leak
        # (PyTorch Geometric SAGEConv has memory issues with many sequential graphs)
        cpu_device = torch.device('cpu')
        original_device = next(model.parameters()).device
        model = model.to(cpu_device)
        print(f"Moving model to CPU for latent space extraction (avoids GPU memory leak)")
        
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        latent_dir = os.path.join(output_dir, 'latent_space')
        os.makedirs(latent_dir, exist_ok=True)
        
        model.eval()
        
        subsets = {
            'train': self.train_graphs,
            'val': self.val_graphs,
            'test': self.test_graphs
        }
        
        for subset_name, graphs in subsets.items():
            print(f"Processing {subset_name} set ({len(graphs)} graphs)...")
            latent_json = {"electrodes": None, "data": {}}
            latent_tensors = []
            
            num_graphs = len(graphs)
            
            with torch.no_grad():
                for i, graph in enumerate(graphs):
                    if i % 5000 == 0:
                        print(f"  Progress: {i}/{num_graphs} ({100*i/num_graphs:.1f}%)")
                    
                    # Encode on CPU (graph is already on CPU)
                    z = model.encode(graph.x, graph.edge_index)
                    
                    # Get metadata from graph (stored during preprocessing)
                    subject_id = str(getattr(graph, 'subject_id', 'unknown'))
                    session_num = str(getattr(graph, 'session_num', 'unknown'))
                    matrix_idx = str(getattr(graph, 'matrix_idx', -1))
                    electrode_labels = getattr(graph, 'electrode_labels', None)
                    
                    # Store electrode labels (only need to do once)
                    if latent_json["electrodes"] is None and electrode_labels is not None:
                        latent_json["electrodes"] = electrode_labels
                    
                    # Store full node-level latent vectors (no aggregation)
                    z_np = z.numpy().tolist()  # Shape: [n_nodes, latent_dim]
                    latent_tensors.append(z.clone())
                    
                    # Build nested structure: subject -> session -> matrix
                    if subject_id not in latent_json["data"]:
                        latent_json["data"][subject_id] = {}
                    if session_num not in latent_json["data"][subject_id]:
                        latent_json["data"][subject_id][session_num] = {}
                    
                    latent_json["data"][subject_id][session_num][matrix_idx] = z_np
                    
                    # Periodic garbage collection
                    if i % 10000 == 0 and i > 0:
                        gc.collect()
            
            # Save as JSON
            json_path = os.path.join(latent_dir, f'{subset_name}_latent_{experiment_name}.json')
            with open(json_path, 'w') as f:
                json.dump(latent_json, f, indent=2)
            print(f"Latent space ({subset_name}) saved to: {json_path}")
            
            # Save raw tensors
            pt_path = os.path.join(latent_dir, f'{subset_name}_latent_tensors_{experiment_name}.pt')
            torch.save(latent_tensors, pt_path)
        
        print(f"\nLatent space files saved to: {latent_dir}")
        
        # Move model back to original device
        model = model.to(original_device)
        print(f"Model moved back to {original_device}")
    
    def _create_subset_visualizations(self, model, config, output_dir, experiment_name, device=None):
        """Create GAE visualization for each subset (train, val, test)."""
        if device is None:
            device = next(model.parameters()).device
        
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        subsets = {
            'train': self.train_graphs,
            'val': self.val_graphs,
            'test': self.test_graphs
        }
        
        for subset_name, graphs in subsets.items():
            self._create_gae_visualization_for_subset(
                model, config, graphs, subset_name, plots_dir, experiment_name, device
            )
    
    def _create_gae_visualization_for_subset(self, model, config, graphs, subset_name, plots_dir, experiment_name, device):
        """Create 5x4 grid visualization for a specific subset."""
        import numpy as np
        
        if len(graphs) == 0:
            print(f"No graphs in {subset_name} subset, skipping visualization")
            return
        
        num_samples = min(5, len(graphs))
        indices = np.random.choice(len(graphs), num_samples, replace=False)
        sample_graphs = [graphs[i] for i in indices]
        
        model.eval()
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        latent_dim = config.get('latent_dim', 2)
        
        with torch.no_grad():
            for row_idx, graph in enumerate(sample_graphs):
                graph = graph.to(device)
                x_recon, z = model(graph.x, graph.edge_index)
                
                # Get metadata
                subject_id = getattr(graph, 'subject_id', 'unknown')
                matrix_idx = getattr(graph, 'matrix_idx', indices[row_idx])
                
                error_normalized = torch.abs(graph.x - x_recon)
                
                original = graph.x.cpu().numpy()
                reconstruction = x_recon.cpu().numpy()
                error_np = error_normalized.cpu().numpy()
                latent = z.cpu().numpy()
                
                # Column 1: Original
                im1 = axes[row_idx, 0].imshow(original, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
                axes[row_idx, 0].set_title(f'Sub-{subject_id} Mat-{matrix_idx}\nOriginal', fontsize=11, fontweight='bold')
                axes[row_idx, 0].set_xlabel('Features')
                axes[row_idx, 0].set_ylabel('Nodes')
                plt.colorbar(im1, ax=axes[row_idx, 0])
                
                # Column 2: Reconstructed
                im2 = axes[row_idx, 1].imshow(reconstruction, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
                axes[row_idx, 1].set_title('Reconstruction', fontsize=11, fontweight='bold')
                axes[row_idx, 1].set_xlabel('Features')
                axes[row_idx, 1].set_ylabel('Nodes')
                plt.colorbar(im2, ax=axes[row_idx, 1])
                
                # Column 3: Error
                im3 = axes[row_idx, 2].imshow(error_np, aspect='auto', cmap='Reds')
                axes[row_idx, 2].set_title(f'Error (MAE: {error_np.mean():.4f})', fontsize=11, fontweight='bold')
                axes[row_idx, 2].set_xlabel('Features')
                axes[row_idx, 2].set_ylabel('Nodes')
                plt.colorbar(im3, ax=axes[row_idx, 2])
                
                # Column 4: Latent Space
                if latent_dim >= 2:
                    scatter = axes[row_idx, 3].scatter(latent[:, 0], latent[:, 1], c=range(len(latent)), 
                                                        cmap='viridis', s=30, alpha=0.7)
                    axes[row_idx, 3].set_xlabel('Latent Dim 0')
                    axes[row_idx, 3].set_ylabel('Latent Dim 1')
                    cbar = plt.colorbar(scatter, ax=axes[row_idx, 3])
                    cbar.set_label('Node Index')
                else:
                    axes[row_idx, 3].hist(latent[:, 0], bins=20, alpha=0.7)
                    axes[row_idx, 3].set_xlabel('Latent Dim 0')
                    axes[row_idx, 3].set_ylabel('Count')
                axes[row_idx, 3].set_title('Latent Space (nodes)', fontsize=11, fontweight='bold')
        
        plt.suptitle(f'GAE Visualization - {subset_name.upper()} Set', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        viz_path = os.path.join(plots_dir, f'gae_visualization_{subset_name}_{experiment_name}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"GAE visualization ({subset_name}) saved to: {viz_path}")
        plt.close()

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
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging for training metrics')
    parser.add_argument('--wandb_project', type=str, default='gnn-connectivity',
                        help='wandb project name (default: gnn-connectivity)')
    
    args = parser.parse_args()

    # Initial resource check - critical for debugging GPU issues
    print("\n" + "="*70)
    print("INITIAL SYSTEM CHECK")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute: {props.major}.{props.minor}")
    else:
        print("⚠ NO GPU DETECTED - Check CUDA installation")
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    print_resource_usage("AT STARTUP (before data loading)")
    
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
    
    # Free memory: delete dataset wrappers (graphs are now referenced directly)
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs:   {len(val_graphs)}")
    print(f"Test graphs:  {len(test_graphs)}")
    print_resource_usage("AFTER LOADING (datasets deleted)")
    
    # Normalize graphs using TRAIN-ONLY statistics (prevents data leakage)
    # Using inplace=True to avoid cloning (saves ~50% memory)
    print("\nComputing normalization stats from TRAIN set only (no data leakage):")
    x_min, x_max, x_range = compute_normalization_stats(train_graphs)
    
    print("Applying normalization IN-PLACE to all splits...")
    apply_normalization(train_graphs, x_min, x_max, x_range, inplace=True)
    apply_normalization(val_graphs, x_min, x_max, x_range, inplace=True)
    apply_normalization(test_graphs, x_min, x_max, x_range, inplace=True)
    print(f"  Normalized {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test graphs")
    gc.collect()

    print_resource_usage("AFTER NORMALIZATION (in-place, no copies)")

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
        experiment_name=args.experiment_name,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project
    )
    
    return final_model, trainer

if __name__ == '__main__':
    main()
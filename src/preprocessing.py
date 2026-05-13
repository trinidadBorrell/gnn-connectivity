"""
PREPROCESSING PIPELINE
======================
Purpose: Transform raw .fif files into PyTorch Geometric Data objects and create dataloaders.

Pipeline Position: FIRST STEP
- Input: Raw .fif files (EEG/MEG recordings)
- Output: train_loader, val_loader, test_loader

Key Operations:
1. Load .fif files and convert to PyTorch Geometric Data objects
2. Split data into train/test/val before any model sees it (70%/15%/15%)
3. Create DataLoader objects for each split
4. Apply any necessary normalization/preprocessing
"""

import torch
from torch_geometric.data import Data     
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
import mne
import numpy as np
import argparse
from typing import Dict, List, Tuple
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os
import pandas as pd

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

class GraphAutoencoderDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample, sample

# Default file path pattern - can be customized
# Available placeholders: {main_path}, {subject_id}, {session_num}
DEFAULT_EEG_PATH_PATTERN = '{main_path}/sub-{subject_id}/ses-{session_num}/eeg/sub-{subject_id}_ses-{session_num}_task-lg_acq-01_epo.fif'

# Default matrix path pattern for loading pre-computed matrices
# Available placeholders: {matrix_dir}, {subject_id}, {session_num}, {matrix_idx}
DEFAULT_MATRIX_PATH_PATTERN = '{matrix_dir}/sub-{subject_id}/ses-{session_num}/matrix_{matrix_idx}.npy'


class EEGtoGraph:
    # Read results single subject
    @staticmethod
    def load_epochs(main_path: str, subject_id: str, session_num: str, channels: List[str] = None,
                    path_pattern: str = None):
        """
        Load epochs from a .fif file.
        
        Args:
            main_path: Path to data directory
            subject_id: Subject ID
            session_num: Session number
            channels: List of channel names to select. If None, all channels are used.
            path_pattern: Custom path pattern. If None, uses DEFAULT_EEG_PATH_PATTERN.
                         Available placeholders: {main_path}, {subject_id}, {session_num}
            
        Returns:
            Epoch data array (n_epochs, n_channels, n_times)
        """
        if path_pattern is None:
            path_pattern = DEFAULT_EEG_PATH_PATTERN
        
        path = path_pattern.format(
            main_path=main_path,
            subject_id=subject_id,
            session_num=session_num
        )
        epochs = mne.read_epochs(path, verbose=False)
        
        if channels is not None:
            epochs = epochs.pick_channels(channels, ordered=True)
        
        return epochs.get_data()  # (n_epochs, n_channels, n_times)
    
    @staticmethod
    def load_precomputed_matrices(matrix_dir: str, subject_id: str, session_num: str,
                                   n_nodes: int = None, path_pattern: str = None) -> List[np.ndarray]:
        """
        Load pre-computed feature matrices from a folder.
        
        Args:
            matrix_dir: Base directory containing the matrices
            subject_id: Subject ID
            session_num: Session number
            n_nodes: Expected number of nodes (for validation). If None, no validation.
            path_pattern: Custom path pattern. If None, uses DEFAULT_MATRIX_PATH_PATTERN.
                         Available placeholders: {matrix_dir}, {subject_id}, {session_num}, {matrix_idx}
            
        Returns:
            List of feature matrices (each n_nodes x n_features)
        """
        if path_pattern is None:
            path_pattern = DEFAULT_MATRIX_PATH_PATTERN
        
        matrices = []
        matrix_idx = 0
        
        while True:
            matrix_path = path_pattern.format(
                matrix_dir=matrix_dir,
                subject_id=subject_id,
                session_num=session_num,
                matrix_idx=matrix_idx
            )
            
            if not os.path.exists(matrix_path):
                break
            
            matrix = np.load(matrix_path)
            
            # Validate node count if specified
            if n_nodes is not None and matrix.shape[0] != n_nodes:
                raise ValueError(
                    f"Matrix at {matrix_path} has {matrix.shape[0]} nodes, "
                    f"expected {n_nodes} to match adjacency matrix"
                )
            
            matrices.append(matrix)
            matrix_idx += 1
        
        if len(matrices) == 0:
            raise FileNotFoundError(
                f"No matrices found for sub-{subject_id}/ses-{session_num} in {matrix_dir}"
            )
        
        return matrices

    @staticmethod
    def cartesian_distance(x1: float, y1: float, x2: float, y2: float, z1: float = None, z2: float = None) -> float:
        """Calculate Euclidean distance between two points in 2D or 3D Cartesian coordinates."""
        if z1 is not None and z2 is not None:
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        else:
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance
    
    @staticmethod
    def create_torch_geometric_data(f, a):
        a_coo = a.tocoo()
        edge_array = np.array([a_coo.row, a_coo.col])
        edge_index = torch.tensor(edge_array, dtype=torch.long)
        x = torch.tensor(f, dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index)
        return data

    @staticmethod
    def find_k_nearest_sensors(
        coords_df: pd.DataFrame,
        k: int = 6
    ) -> Tuple[Dict[str, List[Tuple[str, float]]], np.ndarray]:
        """
        Find k nearest neighbors for each sensor using Cartesian distance.
        
        Args:
            coords_df: DataFrame with columns ['label', 'x', 'y'] or ['label', 'x', 'y', 'z']
            k: Number of nearest neighbors
            
        Returns:
            k_nearest: Dictionary mapping sensor names to list of (neighbor_name, distance) tuples
            distance_matrix: Full pairwise distance matrix
        """
        n_sensors = len(coords_df)
        sensor_names = coords_df['label'].values
        X_array = coords_df['x'].values
        Y_array = coords_df['y'].values
        has_z = 'z' in coords_df.columns
        Z_array = coords_df['z'].values if has_z else None
        
        # Validate inputs
        if k >= n_sensors:
            raise ValueError(f"k ({k}) must be less than the number of sensors ({n_sensors})")
        
        # Calculate pairwise distances
        distance_matrix = np.zeros((n_sensors, n_sensors))
        
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                x1, y1 = X_array[i], Y_array[i]
                x2, y2 = X_array[j], Y_array[j]
                z1 = Z_array[i] if has_z else None
                z2 = Z_array[j] if has_z else None
                
                dist = EEGtoGraph.cartesian_distance(x1, y1, x2, y2, z1, z2)
                
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetric
        
        # Find k nearest neighbors for each sensor
        k_nearest = {}
        
        for i, sensor_name in enumerate(sensor_names):
            # Get distances from this sensor to all others
            distances = distance_matrix[i, :]
            
            # Get indices of k nearest neighbors (excluding itself)
            # argsort gives indices in ascending order
            nearest_indices = np.argsort(distances)[1:k+1]  # Skip index 0 (self)
            
            # Create list of (neighbor_name, distance) tuples
            neighbors = [(sensor_names[idx], distances[idx])
                for idx in nearest_indices]
            
            k_nearest[sensor_name] = neighbors
        
        return k_nearest, distance_matrix

    @staticmethod
    def plot_k_nearest_positions(
        coords_df: pd.DataFrame,
        distance_matrix: np.ndarray,
        k: int = 6,
        output_dir: str = None,
        save: bool = True,
        n_samples: int = None,
        random_seed: int = 42
    ):
        """
        Plot the k nearest neighbors for each sensor.
        
        For 2D coordinates (from eeg_positions): plots all sensors in a grid.
        For 3D coordinates (from file): plots a random sample with x-y and x-z projections.
        
        Args:
            coords_df: DataFrame with columns ['label', 'x', 'y'] or ['label', 'x', 'y', 'z']
            distance_matrix: Pairwise distance matrix
            k: Number of nearest neighbors to highlight
            output_dir: Directory to save plots (if save=True)
            save: Whether to save the plots
            n_samples: Number of random electrodes to plot (default: all for 2D, 16 for 3D)
            random_seed: Random seed for reproducibility when sampling
        """
        n_sensors = len(coords_df)
        has_z = 'z' in coords_df.columns
        
        # Determine number of samples
        if n_samples is None:
            n_samples = 16 if has_z else n_sensors
        n_samples = min(n_samples, n_sensors)
        
        # Select electrode indices to plot
        if n_samples < n_sensors:
            np.random.seed(random_seed)
            sample_indices = np.random.choice(n_sensors, size=n_samples, replace=False)
            sample_indices = np.sort(sample_indices)
        else:
            sample_indices = np.arange(n_sensors)
        
        if has_z:
            # 3D coordinates: create side-by-side x-y and x-z projections for each sampled electrode
            n_rows = int(np.ceil(np.sqrt(n_samples)))
            n_cols = int(np.ceil(n_samples / n_rows))
            
            fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(24, 12))
            axes = axes.reshape(-1, 2) if n_samples > 1 else [axes]
            
            for plot_idx, sensor_idx in enumerate(sample_indices):
                ax_xy = axes[plot_idx][0]
                ax_xz = axes[plot_idx][1]
                row = distance_matrix[sensor_idx, :]
                
                # Get indices of k smallest values (excluding self)
                smallest_indices = np.argpartition(row, k+1)[:k+1]
                smallest_indices = smallest_indices[row[smallest_indices] > 0][:k]
                
                df_neighbours = coords_df.iloc[smallest_indices]
                df_sensor = coords_df.iloc[sensor_idx]
                
                # X-Y projection
                ax_xy.plot(coords_df['x'], coords_df['y'], 'o', color='black', markersize=3, alpha=0.2)
                ax_xy.plot(df_neighbours['x'], df_neighbours['y'], 'o', color='red', markersize=5)
                ax_xy.plot(df_sensor['x'], df_sensor['y'], 'o', color='green', markersize=7)
                ax_xy.set_title(f"{df_sensor['label']} (X-Y)", fontsize=9, fontweight='bold')
                ax_xy.set_xlabel('X')
                ax_xy.set_ylabel('Y')
                ax_xy.set_aspect('equal')
                ax_xy.grid(True, alpha=0.3)
                
                # X-Z projection
                ax_xz.plot(coords_df['x'], coords_df['z'], 'o', color='black', markersize=3, alpha=0.2)
                ax_xz.plot(df_neighbours['x'], df_neighbours['z'], 'o', color='red', markersize=5)
                ax_xz.plot(df_sensor['x'], df_sensor['z'], 'o', color='green', markersize=7)
                ax_xz.set_title(f"{df_sensor['label']} (X-Z)", fontsize=9, fontweight='bold')
                ax_xz.set_xlabel('X')
                ax_xz.set_ylabel('Z')
                ax_xz.set_aspect('equal')
                ax_xz.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for plot_idx in range(n_samples, len(axes)):
                axes[plot_idx][0].axis('off')
                axes[plot_idx][1].axis('off')
        else:
            # 2D coordinates: original behavior with grid layout
            n_rows = int(np.ceil(np.sqrt(n_samples)))
            n_cols = int(np.ceil(n_samples / n_rows))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
            axes = axes.flatten() if n_samples > 1 else [axes]
            
            for plot_idx, sensor_idx in enumerate(sample_indices):
                ax = axes[plot_idx]
                row = distance_matrix[sensor_idx, :]
                
                # Get indices of k smallest values (excluding self)
                smallest_indices = np.argpartition(row, k+1)[:k+1]
                smallest_indices = smallest_indices[row[smallest_indices] > 0][:k]
                
                # Plot all sensors in black
                ax.plot(coords_df['x'], coords_df['y'], 'o', color='black', markersize=4, alpha=0.3)
                
                # Plot k nearest neighbors in red
                df_neighbours = coords_df.iloc[smallest_indices]
                ax.plot(df_neighbours['x'], df_neighbours['y'], 'o', color='red', markersize=6)
                
                # Plot current sensor in green
                df_sensor = coords_df.iloc[sensor_idx]
                ax.plot(df_sensor['x'], df_sensor['y'], 'o', color='green', markersize=8)
                
                # Annotate labels
                for idx, label in enumerate(coords_df['label']):
                    ax.annotate(label, (coords_df['x'].iloc[idx], coords_df['y'].iloc[idx]), 
                               fontsize=6, alpha=0.7)
                
                ax.set_title(f"{df_sensor['label']}", fontsize=10, fontweight='bold')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for plot_idx in range(n_samples, len(axes)):
                axes[plot_idx].axis('off')
        
        plt.tight_layout()
        
        if save and output_dir:
            num_electrodes = len(coords_df)
            if not os.path.exists(f'{output_dir}/images'):
                os.makedirs(f'{output_dir}/images')
            suffix = f'_sample{n_samples}' if n_samples < n_sensors else ''
            output_path = os.path.join(output_dir, 'images', f'k_nearest_neighbours_k{k}_{num_electrodes}_electrodes{suffix}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved k-nearest neighbors plot to: {output_path}")
            plt.close()
        else:
            plt.show()

    # Create adjacency matrix
    @staticmethod
    def adjacency_matrix(
        coords_df: pd.DataFrame,
        k: int,
        output_dir: str,
        save: bool = True,
        plot_neighbors: bool = True
    ):
        """
        Create a sparse adjacency matrix based on k-nearest neighbors in Cartesian coordinates.
        
        Args:
            coords_df: DataFrame with columns ['label', 'x', 'y'] or ['label', 'x', 'y', 'z']
            k: Number of nearest neighbors
            output_dir: Directory to save outputs
            save: Whether to save the adjacency matrix
            plot_neighbors: Whether to plot the k-nearest neighbors visualization
            
        Returns:
            Sparse adjacency matrix (scipy.sparse.csr_matrix)
            labels: Array of sensor names
            distance_matrix: Full pairwise distance matrix
        """
        labels = coords_df['label'].values
        n_sensors = len(labels)
        
        # Check if adjacency matrix already exists
        if save:
            adjacency_npy_path = f'{output_dir}/data/adjacency_matrix_{n_sensors}_electrodes.npy'
            if os.path.exists(adjacency_npy_path):
                print(f"Loading existing adjacency matrix from {adjacency_npy_path}")
                adjacency = np.load(adjacency_npy_path, allow_pickle=True).item()
                # Still need distance matrix for potential neighbor plotting
                _, distance_matrix = EEGtoGraph.find_k_nearest_sensors(coords_df, k)
                return adjacency, labels, distance_matrix
        
        # Find k nearest neighbors
        k_nearest, distance_matrix = EEGtoGraph.find_k_nearest_sensors(coords_df, k)
        
        # Optional: Plot k-nearest neighbors
        if plot_neighbors:
            EEGtoGraph.plot_k_nearest_positions(
                coords_df, distance_matrix, k, output_dir, save
            )
        
        # Create sparse adjacency matrix
        row_indices = []
        col_indices = []
        
        for i, label in enumerate(labels):
            neighbors = k_nearest[label]
            for neighbor_name, _ in neighbors:
                # Find the index of the neighbor
                j = np.where(labels == neighbor_name)[0][0]
                row_indices.append(i)
                col_indices.append(j)
        
        # Create sparse matrix with ones for edges
        data = np.ones(len(row_indices))
        adjacency = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_sensors, n_sensors))
        
        # Make symmetric (if not already)
        adjacency = adjacency + adjacency.T
        adjacency = (adjacency > 0).astype(float)

        if save:
            if not os.path.exists(f'{output_dir}/data'):
                os.makedirs(f'{output_dir}/data')
            np.save(f'{output_dir}/data/adjacency_matrix_{n_sensors}_electrodes.npy', adjacency)
        
        return adjacency, labels, distance_matrix

    
    # Create feature matrix
    @staticmethod
    def feature_matrix(main_path: str,
        subject_id: str,
        session_num: str,
        output_dir: str, 
        window_points: int = 152,
        epoch_num: int = 0,
        feature_type: str = 'temporal',
        channels: List[str] = None,
        path_pattern: str = None,
        save: bool = True):

        """
        Create feature matrix from windowed data.
        
        Args:
            main_path: str to data
            subject_id: Subject ID
            session_num: Session number
            output_dir: Directory to save outputs
            window_points: Points from temporal window
            epoch_num: Epoch selected as a graph
            feature_type: 'temporal' (raw time window) or 'connectivity' (Pearson correlation row)
            channels: List of channel names to select. If None, all channels are used.
            path_pattern: Custom EEG file path pattern (see load_epochs)
            save: Whether to save the feature matrix
            
        Returns:
            Feature matrix (n_channels, n_features)
        """
        data = EEGtoGraph.load_epochs(main_path, subject_id, session_num, channels, path_pattern)  # (n_epochs, n_channels, n_times)
        
        if feature_type == 'temporal':
            # Original behavior: raw time window as features
            F = data[epoch_num, :, :window_points]
        elif feature_type == 'connectivity':
            # New: Pearson correlation matrix - each node's features are its connectivity to all other nodes
            epoch_data = data[epoch_num, :, :window_points]  # (n_channels, window_points)
            # Compute Pearson correlation matrix
            F = np.corrcoef(epoch_data)  # (n_channels, n_channels)
            # Handle NaN values (can occur if a channel has constant signal)
            F = np.nan_to_num(F, nan=0.0)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}. Use 'temporal' or 'connectivity'.")

        print(f'Shape of feature matrix: {F.shape}')
        if save:
            os.makedirs(f'{output_dir}/data/feature_matrix', exist_ok=True)
            np.save(f'{output_dir}/data/feature_matrix/feature_matrix_sub{subject_id}_session_{session_num}_epoch{epoch_num}_{feature_type}.npy', F)
            loaded_F = np.load(f'{output_dir}/data/feature_matrix/feature_matrix_sub{subject_id}_session_{session_num}_epoch{epoch_num}_{feature_type}.npy')
            print(f'Shape of loaded feature matrix: {loaded_F.shape}')
        return F
    
    @staticmethod
    def feature_matrix_all_epochs(main_path: str,
        subject_id: str,
        session_num: str,
        output_dir: str, 
        window_points: int = 152,
        feature_type: str = 'temporal',
        channels: List[str] = None,
        path_pattern: str = None,
        save: bool = True):
        """
        Create feature matrices for ALL epochs in a session.
        
        Args:
            main_path: str to data
            subject_id: Subject ID
            session_num: Session number
            output_dir: Directory to save outputs
            window_points: Points from temporal window
            feature_type: 'temporal' (raw time window) or 'connectivity' (Pearson correlation row)
            channels: List of channel names to select. If None, all channels are used.
            path_pattern: Custom EEG file path pattern (see load_epochs)
            save: Whether to save the feature matrices
            
        Returns:
            List of feature matrices, one per epoch
        """
        data = EEGtoGraph.load_epochs(main_path, subject_id, session_num, channels, path_pattern)  # (n_epochs, n_channels, n_times)
        n_epochs = data.shape[0]
        
        feature_matrices = []
        for epoch_num in range(n_epochs):
            if feature_type == 'temporal':
                F = data[epoch_num, :, :window_points]
            elif feature_type == 'connectivity':
                epoch_data = data[epoch_num, :, :window_points]
                F = np.corrcoef(epoch_data)
                F = np.nan_to_num(F, nan=0.0)
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}. Use 'temporal' or 'connectivity'.")
            feature_matrices.append(F)
        
        print(f'Created {n_epochs} feature matrices of shape {feature_matrices[0].shape}')
        
        if save:
            os.makedirs(f'{output_dir}/data/feature_matrix', exist_ok=True)
            for epoch_num, F in enumerate(feature_matrices):
                np.save(f'{output_dir}/data/feature_matrix/feature_matrix_sub{subject_id}_session_{session_num}_epoch{epoch_num}_{feature_type}.npy', F)
        
        return feature_matrices
    

    # Plot and save matrices
    @staticmethod
    def plot_matrix(
        matrix,
        title: str,
        output_dir: str,
        filename: str,
        cmap: str = 'viridis',
        figsize: tuple = (10, 8),
        labels_: list = [],
        matrix_type: str = 'feature'
    ):
        """
        Plot a matrix and save it to a file.
        
        Args:
            matrix: Matrix to plot (can be sparse or dense)
            title: Plot title
            output_dir: Directory to save the plot
            filename: Name of the output file
            cmap: Colormap to use
            figsize: Figure size (width, height)
        """        
        # Convert sparse matrix to dense if necessary
        if sp.issparse(matrix):
            matrix_dense = matrix.toarray()
        else:
            matrix_dense = matrix
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix_dense, cmap=cmap, aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Node Index', fontsize=12)
        if matrix_type == 'adjacency':
            ax.set_ylabel('Node Index', fontsize=12)
            ax.set_xticks(range(len(labels_)), labels=labels_, rotation=45, fontsize=6)

        else:
            ax.set_ylabel('Feature Index', fontsize=12)
            ax.set_xticks(range(int(matrix_dense.shape[1])), labels=np.linspace(-0.2, 1.34, int(matrix_dense.shape[1])), fontsize=6)

        ax.set_yticks(range(len(labels_)), labels=labels_, rotation=45, fontsize=6)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value', fontsize=12)
        
        # Save figure
        output_path = os.path.join(output_dir, 'images', filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot to: {output_path}")

    # Create graph
    @staticmethod
    def create_graph(
        coords_df: pd.DataFrame,
        main_path: str,
        subject_id: str,
        session_num: str,
        window_points: int = 154,
        epoch: int = 0,
        k: int = 6, 
        output_dir: str = '../output/preprocessing',
        corr_type: str = 'pearson',
        feature_type: str = 'temporal',
        channels: List[str] = None,
        path_pattern: str = None,
        save: bool = True,
        plot_neighbors: bool = False
    ):
        """
        Create adjacency and feature matrices and plot them.
        
        Args:
            coords_df: DataFrame with columns ['label', 'x', 'y']
            main_path: Path to the main data directory
            subject_id: Subject ID
            session_num: Session number
            window_points: Number of time points in the window
            epoch: Epoch number to process
            k: Number of nearest neighbors for adjacency matrix
            output_dir: Directory to save output plots
            corr_type: Type of correlation for feature matrix
            feature_type: 'temporal' or 'connectivity'
            channels: List of channel names to select
            path_pattern: Custom EEG file path pattern
            save: Whether to save outputs
            plot_neighbors: Whether to plot k-nearest neighbors visualization
            
        Returns:
            data, adjacency_matrix, feature_matrix, labels, distance_matrix
        """
        if save:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f'{output_dir}/data', exist_ok=True)
            os.makedirs(f'{output_dir}/images', exist_ok=True)

        print(f"Processing subject {subject_id}, session {session_num}, epoch {epoch}")
        print(f"Creating graph with k={k} nearest neighbors")
        
        num_electrodes = len(coords_df)
        
        # Check if adjacency matrix plot already exists
        adjacency_plot_path = os.path.join(output_dir, 'images', f'adjacency_matrix_{num_electrodes}.png')
        adjacency_npy_path = f'{output_dir}/data/adjacency_matrix_{num_electrodes}_electrodes.npy'
        
        if save and os.path.exists(adjacency_plot_path) and os.path.exists(adjacency_npy_path):
            print("\nAdjacency matrix plot and data already exist. Skipping computation.")
            adjacency = np.load(adjacency_npy_path, allow_pickle=True).item()
            labels = coords_df['label'].values
            _, distance_matrix = EEGtoGraph.find_k_nearest_sensors(coords_df, k)
        else:
            # Create adjacency matrix
            print("\nCreating adjacency matrix...")
            adjacency, labels, distance_matrix = EEGtoGraph.adjacency_matrix(
                coords_df, k, output_dir, save, plot_neighbors
            )
            print(f"Adjacency matrix shape: {adjacency.shape}")
            print(f"Number of edges: {adjacency.nnz // 2}")  # Divide by 2 because it's symmetric
            print(f"Sparsity: {1 - adjacency.nnz / (adjacency.shape[0] ** 2):.4f}")
            
            # Plot adjacency matrix
            print("\nPlotting adjacency matrix...")
            EEGtoGraph.plot_matrix(
                adjacency,
                f'Adjacency Matrix (k={k})',
                output_dir,
                f'adjacency_matrix_{num_electrodes}.png',
                cmap='binary',
                labels_=labels, 
                matrix_type='adjacency'
            )
        
        
        # Create feature matrix
        print("\nCreating feature matrix...")
        feature_mat = EEGtoGraph.feature_matrix(
            main_path, subject_id, session_num, output_dir,
            window_points=window_points, epoch_num=epoch,
            feature_type=feature_type, channels=channels,
            path_pattern=path_pattern, save=save
        )
        print(f"Feature matrix shape: {feature_mat.shape}")
        
        # Plot feature matrix
        print("\nPlotting feature matrix...")
        EEGtoGraph.plot_matrix(
            feature_mat,
            f'Feature Matrix ({corr_type.capitalize()} Correlation)',
            output_dir,
            f'feature_matrix_sub-{subject_id}_ses-{session_num}_epoch-{epoch}.png',
            cmap='RdBu_r',
            labels_=labels
        )

        print("\nCreate Torch Geometric Data Structure...")
        data = EEGtoGraph.create_torch_geometric_data(feature_mat, adjacency)
        
        print("\nGraph creation completed successfully!")
        
        return data, adjacency, feature_mat, labels, distance_matrix
    
    @staticmethod
    def create_graph_dataset(
        coords_df,
        main_path: str = None,
        matrix_dir: str = None,
        window_points: int = 64, 
        k: int = 6, 
        output_dir: str = 'output/preprocessing', 
        feature_type: str = 'temporal',
        channels: List[str] = None,
        path_pattern: str = None,
        matrix_path_pattern: str = None,
        save: bool = True, 
        plot_neighbors: bool = False,
        n_splits: int = 5,
        test_fold: int = 0,
        subject_filter: str = None
    ):
        """
        Create graph dataset using ALL epochs/matrices per subject with GroupKFold splitting.
        
        This ensures no data leakage by keeping all data from a subject in the same split.
        
        Supports two modes:
        1. EEG mode: Load .fif files and compute features (set main_path)
        2. Matrix mode: Load pre-computed matrices from folder (set matrix_dir)
        
        Args:
            coords_df: DataFrame with electrode coordinates
            main_path: Path to EEG data directory (for EEG mode)
            matrix_dir: Path to pre-computed matrices (for matrix mode)
            window_points: Number of time points in window (EEG mode only)
            k: Number of nearest neighbors for adjacency
            output_dir: Directory to save outputs
            feature_type: 'temporal' or 'connectivity' (EEG mode only)
            channels: List of channel names to select (EEG mode only)
            path_pattern: Custom EEG file path pattern
            matrix_path_pattern: Custom matrix path pattern
            save: Whether to save outputs
            plot_neighbors: Whether to plot k-nearest neighbors
            n_splits: Number of folds for GroupKFold
            test_fold: Which fold to use as test set (0 to n_splits-1)
            subject_filter: Comma-separated list of subject IDs to process (e.g., "AA164,BB178,AL012")
            
        Returns:
            dataset_train, dataset_val, dataset_test as GraphAutoencoderDataset objects
        """
        if main_path is None and matrix_dir is None:
            raise ValueError("Either main_path (EEG mode) or matrix_dir (matrix mode) must be provided")
        
        if main_path is not None and matrix_dir is not None:
            raise ValueError("Only one of main_path or matrix_dir should be provided")
        
        use_precomputed = matrix_dir is not None
        data_path = matrix_dir if use_precomputed else main_path
        
        dataset = []
        subject_ids = []  # Track subject ID for each graph (for GroupKFold)
        
        # First, create adjacency matrix (shared across all graphs)
        adjacency = None
        labels = None
        n_nodes = len(coords_df)
        
        for subject_folder in sorted(os.listdir(data_path)):
            if not subject_folder.startswith('sub-'):
                continue
            subject_id = subject_folder.split('-')[1]
            
            # Apply subject filter if provided (comma-separated list of IDs)
            if subject_filter is not None:
                allowed_ids = [s.strip() for s in subject_filter.split(',')]
                if subject_id not in allowed_ids:
                    continue
            subject_path = os.path.join(data_path, subject_folder)
            
            for session_folder in sorted(os.listdir(subject_path)):
                if not session_folder.startswith('ses-'):
                    continue
                session_num = session_folder.split('-')[1]
                session_path = os.path.join(subject_path, session_folder)
                
                # For EEG mode, check if eeg folder exists
                if not use_precomputed:
                    eeg_path = os.path.join(session_path, 'eeg')
                    if not os.path.exists(eeg_path):
                        continue
                
                # Create adjacency matrix once (same for all graphs)
                if adjacency is None:
                    adjacency, labels, _ = EEGtoGraph.adjacency_matrix(
                        coords_df, k, output_dir, save, plot_neighbors
                    )
                
                # Get all feature matrices for this subject/session
                try:
                    if use_precomputed:
                        # Load pre-computed matrices
                        feature_matrices = EEGtoGraph.load_precomputed_matrices(
                            matrix_dir=matrix_dir,
                            subject_id=subject_id,
                            session_num=session_num,
                            n_nodes=n_nodes,
                            path_pattern=matrix_path_pattern
                        )
                    else:
                        # Compute from EEG data
                        feature_matrices = EEGtoGraph.feature_matrix_all_epochs(
                            main_path=main_path,
                            subject_id=subject_id,
                            session_num=session_num,
                            output_dir=output_dir,
                            window_points=window_points,
                            feature_type=feature_type,
                            channels=channels,
                            path_pattern=path_pattern,
                            save=False
                        )
                    
                    # Create a graph for each matrix
                    for matrix_idx, feature_mat in enumerate(feature_matrices):
                        data = EEGtoGraph.create_torch_geometric_data(feature_mat, adjacency)
                        # Store metadata for later use
                        data.subject_id = subject_id
                        data.session_num = session_num
                        data.matrix_idx = matrix_idx
                        data.electrode_labels = list(coords_df['label'].values)
                        dataset.append(data)
                        subject_ids.append(subject_id)
                        
                except Exception as e:
                    print(f"Warning: Could not process sub-{subject_id}/ses-{session_num}: {e}")
                    continue
        
        print(f"\nTotal graphs created: {len(dataset)}")
        print(f"Unique subjects: {len(set(subject_ids))}")
        
        if len(dataset) == 0:
            raise ValueError("No graphs were created. Check your data paths.")
        
        # Use GroupKFold to split by subject (no data leakage)
        # Keep as list to avoid Data object conversion issues
        subject_ids_np = np.array(subject_ids)
        group_kfold = GroupKFold(n_splits=n_splits)
        
        # Get all fold indices
        folds = list(group_kfold.split(dataset, groups=subject_ids_np))
        
        # Use test_fold as test, next fold as val, rest as train
        test_idx = folds[test_fold][1]
        val_fold = (test_fold + 1) % n_splits
        val_idx = folds[val_fold][1]
        
        # Train is everything else
        train_idx = np.concatenate([folds[i][1] for i in range(n_splits) if i != test_fold and i != val_fold])
        
        # Extract datasets directly from list
        dataset_train = [dataset[i] for i in train_idx]
        dataset_val = [dataset[i] for i in val_idx]
        dataset_test = [dataset[i] for i in test_idx]
        
        print(f"Train graphs: {len(dataset_train)} (subjects: {len(set(subject_ids_np[train_idx]))})")
        print(f"Val graphs: {len(dataset_val)} (subjects: {len(set(subject_ids_np[val_idx]))})")
        print(f"Test graphs: {len(dataset_test)} (subjects: {len(set(subject_ids_np[test_idx]))})")
        
        dataset_train = GraphAutoencoderDataset(dataset_train)
        dataset_val = GraphAutoencoderDataset(dataset_val)
        dataset_test = GraphAutoencoderDataset(dataset_test)

        return dataset_train, dataset_val, dataset_test


def main():
    parser = argparse.ArgumentParser(
        description='Create graph representations from EEG data or pre-computed matrices',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data source arguments (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--main_path',
        type=str,
        help='Path to EEG data directory (for EEG mode)'
    )
    data_group.add_argument(
        '--matrix_dir',
        type=str,
        help='Path to pre-computed matrices directory (for matrix mode)'
    )

    parser.add_argument(
        '--coordinates_file',
        type=str,
        required=True,
        help='Path to electrode coordinates file (e.g., biosemi64.txt)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--one_subject',
        action='store_true',
        help='Process only one subject (EEG mode only)'
    )
    parser.add_argument(
        '--subject_id',
        type=str,
        help='Subject ID (e.g., "01")'
    )
    parser.add_argument(
        '--session_num',
        type=str,
        help='Session number (e.g., "01")'
    )
    parser.add_argument(
        '--path_pattern',
        type=str,
        default=None,
        help='Custom EEG file path pattern. Placeholders: {main_path}, {subject_id}, {session_num}'
    )
    parser.add_argument(
        '--matrix_path_pattern',
        type=str,
        default=None,
        help='Custom matrix path pattern. Placeholders: {matrix_dir}, {subject_id}, {session_num}, {matrix_idx}'
    )
    parser.add_argument(
        '--window_points',
        type=int,
        default=64,
        help='Number of time points in the sliding window (EEG mode only)'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=0,
        help='Epoch number to process (single subject mode only)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=6,
        help='Number of nearest neighbors for adjacency matrix'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/preprocessing',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--feature_type',
        type=str,
        default='connectivity',
        choices=['temporal', 'connectivity'],
        help="Feature type: 'temporal' (raw time window) or 'connectivity' (Pearson correlation)"
    )
    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help='Number of folds for GroupKFold cross-validation'
    )
    parser.add_argument(
        '--test_fold',
        type=int,
        default=0,
        help='Which fold to use as test set (0 to n_splits-1)'
    )
    parser.add_argument(
        '--channels',
        type=str,
        nargs='*',
        default=None,
        help='List of channel names to select (e.g., --channels Fp1 Fp2 F3)'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Disable saving outputs'
    )
    parser.add_argument(
        '--plot_neighbors',
        action='store_true',
        help='Plot k-nearest neighbors visualization'
    )
    
    args = parser.parse_args()
    
    # Load coordinates — auto-detect format (biosemi label-theta-phi vs GSN label-x-y-z)
    try:
        with open(args.coordinates_file) as f:
            first_line = f.readline().split()
        n_cols = len(first_line)

        if n_cols >= 4:
            # GSN/HydroCel-style: label x y z
            coords_df = pd.read_csv(
                args.coordinates_file, sep=r"\s+", header=None,
                names=["label", "x", "y", "z"], usecols=[0, 1, 2, 3]
            )
            coords_df = coords_df[~coords_df['label'].str.startswith('Fid')].reset_index(drop=True)
        else:
            from eeg_positions import get_elec_coords
            labels = np.loadtxt(args.coordinates_file, usecols=(0,), dtype=str)
            coords_data = get_elec_coords(system='1005', as_mne_montage=False)
            coords_df = coords_data[coords_data['label'].isin(labels)].copy()

        print(f"Loaded {len(coords_df)} electrode coordinates")

    except Exception as e:
        print(f"\nERROR loading coordinates: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    save = not args.no_save
    
    if args.one_subject:
        if args.main_path is None:
            print("ERROR: --one_subject requires --main_path (EEG mode)")
            return 1
        
        data, adjacency, feature_mat, labels, distance_matrix = EEGtoGraph.create_graph(
            coords_df=coords_df,
            main_path=args.main_path,
            subject_id=args.subject_id,
            session_num=args.session_num,
            window_points=args.window_points,
            epoch=args.epoch,
            k=args.k,
            output_dir=args.output_dir,
            feature_type=args.feature_type,
            channels=args.channels,
            path_pattern=args.path_pattern,
            save=save,
            plot_neighbors=args.plot_neighbors
        )
        
        print("\n" + "="*60)
        print("SUCCESS: All operations completed!")
        print("="*60)

        return data, adjacency, feature_mat, labels, distance_matrix

    else:
        data_train, data_val, data_test = EEGtoGraph.create_graph_dataset(
            coords_df=coords_df,
            main_path=args.main_path,
            matrix_dir=args.matrix_dir,
            window_points=args.window_points,
            k=args.k,
            output_dir=args.output_dir,
            feature_type=args.feature_type,
            channels=args.channels,
            path_pattern=args.path_pattern,
            matrix_path_pattern=args.matrix_path_pattern,
            save=save,
            plot_neighbors=args.plot_neighbors,
            n_splits=args.n_splits,
            test_fold=args.test_fold
        )
        
        print("\n" + "="*60)
        print("SUCCESS: All operations completed!")
        print("="*60)
        
        return data_train, data_val, data_test


if __name__ == '__main__':
    exit(main())
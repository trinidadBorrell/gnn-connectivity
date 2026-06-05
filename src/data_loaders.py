"""
DATA LOADERS MODULE
===================
Purpose: Create train/val/test data loaders with subject-level splits to prevent data leakage.

This module provides utilities for:
1. Loading pre-saved datasets
2. Creating PyTorch DataLoaders with proper batching
3. Subject-level splitting to ensure no data leakage

Usage:
    from data_loaders import create_data_loaders, load_datasets
    
    # Load existing datasets
    train_dataset, val_dataset, test_dataset = load_datasets('path/to/datasets')
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=32
    )
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
# PyTorch Geometric DataLoader available if needed for batched graph processing
# from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import GroupKFold
from typing import Tuple, List, Optional
import os


def collate_graphs(batch):
    """
    Custom collate function for graph autoencoder training.
    Returns (graphs, graphs) for reconstruction loss.
    """
    graphs = [item[0] for item in batch]
    return graphs, graphs


def create_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from datasets.
    
    Args:
        train_dataset: Training dataset (GraphAutoencoderDataset)
        val_dataset: Validation dataset (GraphAutoencoderDataset)
        test_dataset: Test dataset (GraphAutoencoderDataset)
        batch_size: Batch size for data loaders
        shuffle_train: Whether to shuffle training data
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_graphs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graphs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graphs
    )
    
    return train_loader, val_loader, test_loader


def load_datasets(
    datasets_dir: str
) -> Tuple:
    """
    Load pre-saved train/val/test datasets.
    
    Args:
        datasets_dir: Directory containing train_dataset.pt, val_dataset.pt, test_dataset.pt
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    train_path = os.path.join(datasets_dir, 'train_dataset.pt')
    val_path = os.path.join(datasets_dir, 'val_dataset.pt')
    test_path = os.path.join(datasets_dir, 'test_dataset.pt')
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(f"Dataset files not found in {datasets_dir}")
    
    train_dataset = torch.load(train_path, weights_only=False)
    val_dataset = torch.load(val_path, weights_only=False)
    test_dataset = torch.load(test_path, weights_only=False)
    
    print(f"Loaded datasets from {datasets_dir}")
    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val:   {len(val_dataset)} graphs")
    print(f"  Test:  {len(test_dataset)} graphs")
    
    return train_dataset, val_dataset, test_dataset


def save_datasets(
    train_dataset,
    val_dataset,
    test_dataset,
    output_dir: str
) -> str:
    """
    Save datasets to disk.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        output_dir: Directory to save datasets
        
    Returns:
        Path to datasets directory
    """
    datasets_dir = os.path.join(output_dir, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)
    
    torch.save(train_dataset, os.path.join(datasets_dir, 'train_dataset.pt'))
    torch.save(val_dataset, os.path.join(datasets_dir, 'val_dataset.pt'))
    torch.save(test_dataset, os.path.join(datasets_dir, 'test_dataset.pt'))
    
    print(f"Datasets saved to: {datasets_dir}")
    
    return datasets_dir


def split_by_subject(
    graphs: List,
    subject_ids: List[str],
    n_splits: int = 5,
    test_fold: int = 0,
    val_fold: Optional[int] = None
) -> Tuple[List, List, List, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split graphs by subject using GroupKFold to prevent data leakage.
    
    All graphs from a single subject will be in the same split (train, val, or test).
    This prevents data leakage where the model could learn subject-specific patterns
    during training and exploit them during evaluation.
    
    Args:
        graphs: List of graph objects
        subject_ids: List of subject IDs corresponding to each graph
        n_splits: Number of folds for GroupKFold
        test_fold: Which fold to use as test set (0 to n_splits-1)
        val_fold: Which fold to use as val set. If None, uses (test_fold + 1) % n_splits
        
    Returns:
        train_graphs, val_graphs, test_graphs, train_idx, val_idx, test_idx
    """
    graphs = np.array(graphs, dtype=object)
    subject_ids = np.array(subject_ids)
    
    group_kfold = GroupKFold(n_splits=n_splits)
    folds = list(group_kfold.split(graphs, groups=subject_ids))
    
    # Determine fold indices
    test_idx = folds[test_fold][1]
    
    if val_fold is None:
        val_fold = (test_fold + 1) % n_splits
    val_idx = folds[val_fold][1]
    
    # Train is everything else
    train_idx = np.concatenate([
        folds[i][1] for i in range(n_splits) 
        if i != test_fold and i != val_fold
    ])
    
    train_graphs = list(graphs[train_idx])
    val_graphs = list(graphs[val_idx])
    test_graphs = list(graphs[test_idx])
    
    # Print split info
    train_subjects = set(subject_ids[train_idx])
    val_subjects = set(subject_ids[val_idx])
    test_subjects = set(subject_ids[test_idx])
    
    print("\nSubject-level split summary:")
    print(f"  Train: {len(train_graphs)} graphs from {len(train_subjects)} subjects")
    print(f"  Val:   {len(val_graphs)} graphs from {len(val_subjects)} subjects")
    print(f"  Test:  {len(test_graphs)} graphs from {len(test_subjects)} subjects")
    
    # Verify no overlap
    assert len(train_subjects & val_subjects) == 0, "Data leakage: train/val subjects overlap!"
    assert len(train_subjects & test_subjects) == 0, "Data leakage: train/test subjects overlap!"
    assert len(val_subjects & test_subjects) == 0, "Data leakage: val/test subjects overlap!"
    print("  No subject overlap between splits (no data leakage)")
    
    return train_graphs, val_graphs, test_graphs, train_idx, val_idx, test_idx


def get_subject_ids_from_graphs(graphs: List) -> List[str]:
    """
    Extract subject IDs from graph metadata.
    
    Args:
        graphs: List of graph objects with subject_id attribute
        
    Returns:
        List of subject IDs
    """
    subject_ids = []
    for g in graphs:
        if hasattr(g, 'subject_id'):
            subject_ids.append(g.subject_id)
        else:
            raise ValueError("Graph does not have subject_id attribute. "
                           "Ensure graphs were created with create_graph_dataset.")
    return subject_ids


def verify_no_data_leakage(train_dataset, val_dataset, test_dataset) -> bool:
    """
    Verify that there is no subject overlap between train/val/test datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        
    Returns:
        True if no data leakage, raises AssertionError otherwise
    """
    def get_subjects(dataset):
        # Fast path: lazy on-disk dataset exposes a per-item subject id list
        # without touching the actual data.
        if hasattr(dataset, 'subject_ids'):
            return set(dataset.subject_ids)
        if hasattr(dataset, 'data') and hasattr(dataset.data, 'subject_ids'):
            return set(dataset.data.subject_ids)
        # Fallback: iterate and pull from each graph
        subjects = set()
        data = dataset.data if hasattr(dataset, 'data') else dataset
        for g in data:
            if hasattr(g, 'subject_id'):
                subjects.add(g.subject_id)
        return subjects
    
    train_subjects = get_subjects(train_dataset)
    val_subjects = get_subjects(val_dataset)
    test_subjects = get_subjects(test_dataset)
    
    overlap_train_val = train_subjects & val_subjects
    overlap_train_test = train_subjects & test_subjects
    overlap_val_test = val_subjects & test_subjects
    
    if overlap_train_val:
        raise AssertionError(f"Data leakage: Train/Val share subjects: {overlap_train_val}")
    if overlap_train_test:
        raise AssertionError(f"Data leakage: Train/Test share subjects: {overlap_train_test}")
    if overlap_val_test:
        raise AssertionError(f"Data leakage: Val/Test share subjects: {overlap_val_test}")
    
    print("Data leakage check passed: No subject overlap between splits")
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Val subjects:   {len(val_subjects)}")
    print(f"  Test subjects:  {len(test_subjects)}")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data loader utilities')
    parser.add_argument('--datasets_dir', type=str, required=True,
                        help='Directory containing saved datasets')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loaders')
    parser.add_argument('--verify_leakage', action='store_true',
                        help='Verify no data leakage in datasets')
    
    args = parser.parse_args()
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets(args.datasets_dir)
    
    # Verify no data leakage
    if args.verify_leakage:
        verify_no_data_leakage(train_dataset, val_dataset, test_dataset)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size
    )
    
    print("\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

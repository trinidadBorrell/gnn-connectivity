"""
LATENT SPACE STUDY v2
=====================
Analyzes latent space representations from trained GAE models.

NEW APPROACH:
- Each matrix (n_electrodes × latent_dim) is FLATTENED to a single point
- E.g., 256 electrodes × 2 latent dims = 512-dimensional point
- Combines train/val/test for unified clustering
- Calculates cluster prevalence per diagnosis

Usage:
    python latent_space_study_v2.py --latent_dir /path/to/latent_space --labels_csv /path/to/labels.csv
"""

import argparse
import json
import os
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Plotting style
COLOR = "black"
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.figsize": (16, 12),
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "legend.fontsize": "medium",
    "legend.title_fontsize": 14,
    "axes.titlesize": 14,
    "axes.labelsize": "large",
    "ytick.labelsize": 10,
    "xtick.labelsize": 10,
    "text.color": COLOR,
    "axes.labelcolor": COLOR,
    "xtick.color": COLOR,
    "ytick.color": COLOR,
    "grid.color": COLOR,
})


class LatentSpaceStudy:
    """Analyzes latent space representations from GAE models."""
    
    def __init__(self, latent_dir, labels_csv, output_dir=None, only_two_subject_types=False):
        self.latent_dir = latent_dir
        self.labels_csv = labels_csv
        self.output_dir = output_dir or os.path.join(latent_dir, 'analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.labels_df = pd.read_csv(labels_csv)
        print(f"Loaded {len(self.labels_df)} patient records from {labels_csv}")
        
        self.only_two_subject_types = only_two_subject_types
        if only_two_subject_types:
            print("Filtering to only UWS and MCS (MCS+/MCS-) subjects")
        
        self.latent_data = {}
        self._load_latent_data()
        
        self.electrodes = None
        self.n_electrodes = None
        self.latent_dim = None
        
    def _load_latent_data(self):
        """Load latent space JSON files."""
        for subset in ['train', 'val', 'test']:
            json_files = [f for f in os.listdir(self.latent_dir) 
                         if f.startswith(f'{subset}_latent_') and f.endswith('.json')]
            
            if json_files:
                json_path = os.path.join(self.latent_dir, json_files[0])
                print(f"Loading {subset}: {json_files[0]}...")
                with open(json_path, 'r') as f:
                    self.latent_data[subset] = json.load(f)
    
    def get_subject_diagnosis(self, subject_id, session_num=None):
        """Get diagnosis for a subject."""
        mask = self.labels_df['subject'] == subject_id
        if session_num is not None:
            session_mask = self.labels_df['session'] == int(session_num)
            if (mask & session_mask).any():
                mask = mask & session_mask
        
        if mask.any():
            return self.labels_df.loc[mask, 'diagnostic_crs_final'].iloc[0]
        return 'Unknown'
    
    def _is_valid_diagnosis(self, diagnosis):
        """Check if diagnosis is valid for current filter settings."""
        if not self.only_two_subject_types:
            return True
        return diagnosis in ['UWS', 'MCS+', 'MCS-']
    
    def _get_simplified_diagnosis(self, diagnosis):
        """Get simplified diagnosis (MCS+ and MCS- -> MCS)."""
        if diagnosis in ['MCS+', 'MCS-']:
            return 'MCS'
        return diagnosis
    
    def extract_flattened_matrices(self, subsets=None):
        """
        Extract flattened matrices from all subsets.
        Each matrix [n_electrodes, latent_dim] becomes a single flattened point.
        
        Returns:
            flattened_points: np.array [n_matrices_total, n_electrodes * latent_dim]
            metadata: List of dicts with subject_id, session, matrix_idx, diagnosis, subset
        """
        if subsets is None:
            subsets = list(self.latent_data.keys())
        
        flattened_points = []
        metadata = []
        
        for subset in subsets:
            if subset not in self.latent_data:
                continue
                
            data = self.latent_data[subset]['data']
            electrodes = self.latent_data[subset].get('electrodes', [])
            
            if self.electrodes is None and electrodes:
                self.electrodes = electrodes
                self.n_electrodes = len(electrodes)
            
            for subject_id, sessions in data.items():
                diagnosis = self.get_subject_diagnosis(subject_id)
                
                if not self._is_valid_diagnosis(diagnosis):
                    continue
                
                for session_num, matrices in sessions.items():
                    for matrix_idx, latent_vectors in matrices.items():
                        latent_arr = np.array(latent_vectors)
                        
                        if self.latent_dim is None:
                            self.latent_dim = latent_arr.shape[1]
                            self.n_electrodes = latent_arr.shape[0]
                        
                        flattened = latent_arr.flatten()
                        flattened_points.append(flattened)
                        
                        metadata.append({
                            'subject_id': subject_id,
                            'session': session_num,
                            'matrix_idx': matrix_idx,
                            'diagnosis': diagnosis,
                            'subset': subset,
                            'latent_original': latent_arr
                        })
        
        flattened_points = np.array(flattened_points)
        print(f"\nExtracted {len(flattened_points)} flattened points")
        print(f"  Dimensionality: {flattened_points.shape[1]} ({self.n_electrodes} electrodes × {self.latent_dim} latent dims)")
        
        return flattened_points, metadata
    
    def visualize_latent_space_grid(self, subset='test', n_subjects=4, n_matrices=4, save=True):
        """Create a grid visualization of latent space."""
        if subset not in self.latent_data:
            raise ValueError(f"Subset '{subset}' not loaded")
        
        data = self.latent_data[subset]['data']
        
        subjects = []
        for subject_id, sessions in data.items():
            total_matrices = sum(len(m) for m in sessions.values())
            if total_matrices >= n_matrices:
                subjects.append(subject_id)
            if len(subjects) >= n_subjects:
                break
        
        if len(subjects) < n_subjects:
            print(f"Warning: Only found {len(subjects)} subjects with >= {n_matrices} matrices")
            n_subjects = len(subjects)
        
        if n_subjects == 0:
            print("No subjects with enough matrices found!")
            return
        
        fig, axes = plt.subplots(n_subjects, n_matrices, figsize=(4*n_matrices, 4*n_subjects))
        if n_subjects == 1:
            axes = axes.reshape(1, -1)
        if n_matrices == 1:
            axes = axes.reshape(-1, 1)
        
        cmap = plt.cm.viridis
        
        for row_idx, subject_id in enumerate(subjects[:n_subjects]):
            diagnosis = self.get_subject_diagnosis(subject_id)
            sessions = data[subject_id]
            
            all_matrices = []
            for session_num, matrices in sessions.items():
                for matrix_idx, latent_vectors in matrices.items():
                    all_matrices.append({
                        'session': session_num,
                        'matrix_idx': matrix_idx,
                        'latent': np.array(latent_vectors)
                    })
            
            for col_idx in range(n_matrices):
                ax = axes[row_idx, col_idx]
                
                if col_idx < len(all_matrices):
                    matrix_data = all_matrices[col_idx]
                    latent = matrix_data['latent']
                    
                    if latent.shape[1] == 2:
                        latent_2d = latent
                    elif latent.shape[1] > 2:
                        pca = PCA(n_components=2)
                        latent_2d = pca.fit_transform(latent)
                    else:
                        latent_2d = np.column_stack([latent, np.zeros(len(latent))])
                    
                    ax.scatter(
                        latent_2d[:, 0], latent_2d[:, 1],
                        c=np.arange(len(latent_2d)), cmap=cmap,
                        alpha=0.7, s=30, edgecolors='white', linewidths=0.5
                    )
                    ax.set_title(f"S{matrix_data['session']}-M{matrix_data['matrix_idx']}", fontsize=10)
                else:
                    ax.set_visible(False)
                
                if col_idx == 0:
                    ax.set_ylabel(f"{subject_id}\n({diagnosis})", fontsize=10, fontweight='bold')
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Latent Space per Matrix ({subset.upper()})', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'latent_grid_{subset}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def cluster_flattened(self, n_clusters=5, eps=0.5, min_samples=10):
        """Cluster flattened matrices across all subsets."""
        print("\n" + "="*60)
        print("CLUSTERING FLATTENED MATRICES")
        print("="*60)
        
        flattened_points, metadata = self.extract_flattened_matrices()
        
        diagnoses = [m['diagnosis'] for m in metadata]
        subjects = [m['subject_id'] for m in metadata]
        unique_diagnoses = sorted(set(diagnoses))
        unique_subjects = set(subjects)
        
        print(f"\nData summary:")
        print(f"  Total matrices (points): {len(flattened_points)}")
        print(f"  Unique subjects: {len(unique_subjects)}")
        print(f"  Diagnoses: {unique_diagnoses}")
        for diag in unique_diagnoses:
            count = diagnoses.count(diag)
            print(f"    {diag}: {count} matrices ({100*count/len(diagnoses):.1f}%)")
        
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(flattened_points)
        
        print("\nReducing dimensionality with PCA...")
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points_scaled)
        print(f"  Explained variance: {sum(pca.explained_variance_ratio_):.2%}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_points': len(flattened_points),
            'n_subjects': len(unique_subjects),
            'point_dim': flattened_points.shape[1],
            'diagnoses': unique_diagnoses,
            'pca_explained_variance': float(sum(pca.explained_variance_ratio_)),
            'clustering': {}
        }
        
        # KMeans
        print(f"\nRunning KMeans (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(points_scaled)
        kmeans_sil = silhouette_score(points_scaled, kmeans_labels)
        print(f"  Silhouette score: {kmeans_sil:.3f}")
        
        results['clustering']['kmeans'] = {
            'n_clusters': n_clusters,
            'silhouette': float(kmeans_sil),
        }
        
        # DBSCAN
        print(f"\nRunning DBSCAN (eps={eps}, min_samples={min_samples})...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(points_scaled)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = (dbscan_labels == -1).sum()
        print(f"  Found {n_clusters_dbscan} clusters, {n_noise} noise points")
        
        dbscan_sil = None
        if n_clusters_dbscan > 1:
            valid_mask = dbscan_labels != -1
            if valid_mask.sum() > 1:
                dbscan_sil = silhouette_score(points_scaled[valid_mask], dbscan_labels[valid_mask])
                print(f"  Silhouette score: {dbscan_sil:.3f}")
        
        results['clustering']['dbscan'] = {
            'n_clusters': n_clusters_dbscan,
            'n_noise': int(n_noise),
            'silhouette': float(dbscan_sil) if dbscan_sil else None,
        }
        
        # Prevalence
        print("\n--- Cluster Prevalence by Diagnosis (KMeans) ---")
        prevalence_kmeans = self._calculate_prevalence(kmeans_labels, metadata, unique_diagnoses)
        results['prevalence_kmeans'] = prevalence_kmeans
        
        if n_clusters_dbscan > 0:
            print("\n--- Cluster Prevalence by Diagnosis (DBSCAN) ---")
            prevalence_dbscan = self._calculate_prevalence(dbscan_labels, metadata, unique_diagnoses)
            results['prevalence_dbscan'] = prevalence_dbscan
        
        self._cluster_results = {
            'points_2d': points_2d,
            'metadata': metadata,
            'kmeans_labels': kmeans_labels,
            'dbscan_labels': dbscan_labels,
            'flattened_points': flattened_points
        }
        
        json_path = os.path.join(self.output_dir, 'clustering_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")
        
        return results
    
    def _calculate_prevalence(self, cluster_labels, metadata, diagnoses):
        """
        Calculate cluster prevalence per diagnosis using per-subject averaging.
        
        For each subject, calculate the fraction of their matrices in each cluster,
        then average across subjects within each diagnosis group.
        This avoids bias from subjects with more matrices.
        """
        unique_clusters = sorted(set(cluster_labels))
        
        subject_to_data = defaultdict(lambda: {'diagnosis': None, 'labels': []})
        for label, meta in zip(cluster_labels, metadata):
            subj = meta['subject_id']
            subject_to_data[subj]['diagnosis'] = meta['diagnosis']
            subject_to_data[subj]['labels'].append(label)
        
        prevalence = {diag: {} for diag in diagnoses}
        subject_prevalences = {diag: {c: [] for c in unique_clusters} for diag in diagnoses}
        
        for subject_id, data in subject_to_data.items():
            diag = data['diagnosis']
            labels = data['labels']
            total = len(labels)
            
            for cluster_id in unique_clusters:
                count = labels.count(cluster_id)
                pct = 100 * count / total
                subject_prevalences[diag][cluster_id].append(pct)
        
        for diag in diagnoses:
            n_subjects = len(subject_prevalences[diag][unique_clusters[0]]) if unique_clusters else 0
            if n_subjects == 0:
                continue
            
            print(f"\n{diag} (n={n_subjects} subjects):")
            
            for cluster_id in unique_clusters:
                pcts = subject_prevalences[diag][cluster_id]
                mean_pct = np.mean(pcts) if pcts else 0
                std_pct = np.std(pcts) if pcts else 0
                cluster_name = f"cluster_{cluster_id}" if cluster_id >= 0 else "noise"
                prevalence[diag][cluster_name] = {
                    'mean_percentage': round(mean_pct, 2),
                    'std_percentage': round(std_pct, 2),
                    'n_subjects': n_subjects,
                    'per_subject_percentages': [round(p, 2) for p in pcts]
                }
                if mean_pct > 0:
                    print(f"  {cluster_name}: {mean_pct:.1f}% ± {std_pct:.1f}%")
        
        return prevalence
    
    def visualize_clusters(self, method='kmeans', save=True):
        """Visualize clusters in 2D PCA space."""
        if not hasattr(self, '_cluster_results'):
            print("Run cluster_flattened() first!")
            return
        
        points_2d = self._cluster_results['points_2d']
        metadata = self._cluster_results['metadata']
        
        labels = self._cluster_results[f'{method}_labels']
        diagnoses = [m['diagnosis'] for m in metadata]
        unique_diagnoses = sorted(set(diagnoses))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Clusters
        scatter = axes[0].scatter(
            points_2d[:, 0], points_2d[:, 1],
            c=labels, cmap='tab10', alpha=0.6, s=20
        )
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        axes[0].set_title(f'{method.upper()} Clusters (n={n_clusters})', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('PCA 1')
        axes[0].set_ylabel('PCA 2')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Cluster')
        
        # Diagnosis
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_diagnoses)))
        color_map = {diag: colors[i] for i, diag in enumerate(unique_diagnoses)}
        
        for diag in unique_diagnoses:
            mask = np.array(diagnoses) == diag
            axes[1].scatter(
                points_2d[mask, 0], points_2d[mask, 1],
                c=[color_map[diag]], label=diag, alpha=0.6, s=20
            )
        axes[1].set_title('By Diagnosis', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('PCA 1')
        axes[1].set_ylabel('PCA 2')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Flattened Matrix Clustering (PCA)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'cluster_visualization_{method}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def visualize_cluster_latent_means(self, method='kmeans', n_samples=3, save=True):
        """Visualize mean latent space per cluster and sample matrices."""
        if not hasattr(self, '_cluster_results'):
            print("Run cluster_flattened() first!")
            return
        
        metadata = self._cluster_results['metadata']
        labels = self._cluster_results[f'{method}_labels']
        
        unique_clusters = sorted([c for c in set(labels) if c >= 0])
        n_clusters = len(unique_clusters)
        
        if n_clusters == 0:
            print("No valid clusters found!")
            return
        
        n_cols = 1 + n_samples
        fig, axes = plt.subplots(n_clusters, n_cols, figsize=(4*n_cols, 4*n_clusters))
        
        if n_clusters == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, cluster_id in enumerate(unique_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            cluster_latents = []
            cluster_diagnoses = []
            for idx in cluster_indices:
                cluster_latents.append(metadata[idx]['latent_original'])
                cluster_diagnoses.append(metadata[idx]['diagnosis'])
            
            cluster_latents = np.array(cluster_latents)
            mean_latent = cluster_latents.mean(axis=0)
            
            diag_counts = {}
            for d in cluster_diagnoses:
                diag_counts[d] = diag_counts.get(d, 0) + 1
            diag_str = ", ".join([f"{k}:{v}" for k, v in sorted(diag_counts.items())])
            
            # Mean plot
            ax = axes[row_idx, 0]
            if mean_latent.shape[1] == 2:
                ax.scatter(
                    mean_latent[:, 0], mean_latent[:, 1],
                    c=np.arange(len(mean_latent)), cmap='viridis',
                    s=50, alpha=0.8, edgecolors='white', linewidths=0.5
                )
                ax.set_title(f'Cluster {cluster_id} Mean\n(n={len(cluster_indices)})', fontsize=10, fontweight='bold')
            else:
                ax.imshow(mean_latent, aspect='auto', cmap='coolwarm')
                ax.set_title(f'Cluster {cluster_id} Mean', fontsize=10, fontweight='bold')
            
            ax.set_xlabel(f'{diag_str}', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Samples
            sample_indices = np.random.choice(len(cluster_latents), min(n_samples, len(cluster_latents)), replace=False)
            for col_idx, sample_idx in enumerate(sample_indices):
                ax = axes[row_idx, 1 + col_idx]
                sample_latent = cluster_latents[sample_idx]
                sample_diag = cluster_diagnoses[sample_idx]
                
                if sample_latent.shape[1] == 2:
                    ax.scatter(
                        sample_latent[:, 0], sample_latent[:, 1],
                        c=np.arange(len(sample_latent)), cmap='viridis',
                        s=30, alpha=0.7, edgecolors='white', linewidths=0.3
                    )
                else:
                    ax.imshow(sample_latent, aspect='auto', cmap='coolwarm')
                
                ax.set_title(f'Sample ({sample_diag})', fontsize=9)
                ax.grid(True, alpha=0.3)
            
            for col_idx in range(len(sample_indices), n_samples):
                axes[row_idx, 1 + col_idx].set_visible(False)
        
        plt.suptitle(f'Cluster Latent Means & Samples ({method.upper()})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'cluster_latent_means_{method}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def visualize_cluster_prevalence_boxplots(self, method='kmeans', save=True):
        """
        Create boxplots showing per-subject cluster prevalence for UWS vs MCS.
        
        One subplot per cluster, x-axis = diagnosis type (UWS, MCS),
        y-axis = prevalence of that cluster in each subject.
        """
        if not hasattr(self, '_cluster_results'):
            print("Run cluster_flattened() first!")
            return
        
        metadata = self._cluster_results['metadata']
        labels = self._cluster_results[f'{method}_labels']
        
        unique_clusters = sorted([c for c in set(labels) if c >= 0])
        n_clusters = len(unique_clusters)
        
        if n_clusters == 0:
            print("No valid clusters found!")
            return
        
        subject_to_data = defaultdict(lambda: {'diagnosis': None, 'labels': []})
        for label, meta in zip(labels, metadata):
            subj = meta['subject_id']
            subject_to_data[subj]['diagnosis'] = meta['diagnosis']
            subject_to_data[subj]['labels'].append(label)
        
        subject_cluster_prevalence = defaultdict(lambda: {c: 0.0 for c in unique_clusters})
        subject_diagnosis_simplified = {}
        
        for subject_id, data in subject_to_data.items():
            diag = data['diagnosis']
            simplified_diag = self._get_simplified_diagnosis(diag)
            
            if simplified_diag not in ['UWS', 'MCS']:
                continue
            
            subject_diagnosis_simplified[subject_id] = simplified_diag
            labels_subj = data['labels']
            total = len(labels_subj)
            
            for cluster_id in unique_clusters:
                count = labels_subj.count(cluster_id)
                pct = 100 * count / total
                subject_cluster_prevalence[subject_id][cluster_id] = pct
        
        if not subject_cluster_prevalence:
            print("No UWS or MCS subjects found for boxplot!")
            return
        
        n_cols = min(4, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        
        if n_clusters == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        colors = {'UWS': '#E74C3C', 'MCS': '#3498DB'}
        
        for idx, cluster_id in enumerate(unique_clusters):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            uws_prevalences = []
            mcs_prevalences = []
            
            for subject_id, prevalences in subject_cluster_prevalence.items():
                diag = subject_diagnosis_simplified[subject_id]
                pct = prevalences[cluster_id]
                if diag == 'UWS':
                    uws_prevalences.append(pct)
                elif diag == 'MCS':
                    mcs_prevalences.append(pct)
            
            box_data = [uws_prevalences, mcs_prevalences]
            positions = [0, 1]
            
            bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], [colors['UWS'], colors['MCS']]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            for i, (data, pos, diag) in enumerate(zip(box_data, positions, ['UWS', 'MCS'])):
                jitter = np.random.uniform(-0.15, 0.15, len(data))
                ax.scatter([pos + j for j in jitter], data, 
                          c=colors[diag], alpha=0.7, s=40, edgecolors='white', linewidths=0.5,
                          zorder=5)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(['UWS', 'MCS'])
            ax.set_ylabel('Cluster Prevalence (%)')
            ax.set_title(f'Cluster {cluster_id}\n(UWS: n={len(uws_prevalences)}, MCS: n={len(mcs_prevalences)})', 
                        fontsize=11, fontweight='bold')
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for idx in range(n_clusters, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'Cluster Prevalence by Diagnosis ({method.upper()})\nPer-Subject Distribution', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'cluster_prevalence_boxplot_{method}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def run_full_analysis(self, n_clusters=5):
        """Run complete analysis."""
        print("="*60)
        print("LATENT SPACE STUDY - FULL ANALYSIS")
        print("="*60)
        
        for subset in self.latent_data.keys():
            print(f"\nCreating latent grid for {subset}...")
            self.visualize_latent_space_grid(subset=subset)
        
        results = self.cluster_flattened(n_clusters=n_clusters)
        
        print("\nCreating visualizations...")
        self.visualize_clusters(method='kmeans')
        self.visualize_clusters(method='dbscan')
        self.visualize_cluster_latent_means(method='kmeans')
        self.visualize_cluster_prevalence_boxplots(method='kmeans')
        
        if results['clustering']['dbscan']['n_clusters'] > 0:
            self.visualize_cluster_latent_means(method='dbscan')
            self.visualize_cluster_prevalence_boxplots(method='dbscan')
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print(f"Output: {self.output_dir}")
        print("="*60)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze latent space representations from GAE models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--latent_dir', type=str,
        default='/home/triniborrell/home/projects/gnn_connectivity/output/inference/latent_space',
        help='Directory containing latent space JSON files')
    parser.add_argument('--labels_csv', type=str,
        default='/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv',
        help='Path to CSV with patient diagnoses')
    parser.add_argument('--output_dir', type=str, default=None,
        help='Output directory (default: latent_dir/analysis)')
    parser.add_argument('--n_clusters', type=int, default=5,
        help='Number of clusters for KMeans')
    parser.add_argument('--eps', type=float, default=0.5,
        help='DBSCAN eps parameter')
    parser.add_argument('--min_samples', type=int, default=10,
        help='DBSCAN min_samples parameter')
    parser.add_argument('--only-two-subject-types', action='store_true',
        help='Only include UWS and MCS (MCS+/MCS-) subjects in analysis')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LATENT SPACE STUDY v2")
    print("="*60)
    print(f"Latent dir: {args.latent_dir}")
    print(f"Labels CSV: {args.labels_csv}")
    print()
    
    study = LatentSpaceStudy(
        latent_dir=args.latent_dir,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        only_two_subject_types=getattr(args, 'only_two_subject_types', False)
    )
    
    study.run_full_analysis(n_clusters=args.n_clusters)
    
    return 0


if __name__ == '__main__':
    exit(main())

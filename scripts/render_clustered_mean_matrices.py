"""
Re-render the per-cluster mean wSMI matrices with electrodes grouped by
functional network (AUD, DMN, FP, MOT, SAL, VIS).

This is a heuristic scalp-to-network mapping based on standard 10-10
electrode positions:
  - DMN  (default mode):       medial prefrontal pole, midline frontal, medial parietal
  - FP   (frontoparietal):     lateral DLPFC, lateral parietal/IPS, P9/P10
  - SAL  (salience):           frontocentral (insular/dACC projection)
  - MOT  (sensorimotor):       central + centroparietal
  - AUD  (auditory):           temporal + frontotemporal + temporoparietal
  - VIS  (visual):             occipital + parieto-occipital

The mapping is in NETWORK_LAYOUT below. Edit it if you disagree with any
assignment. The script writes a JSON sidecar with the resolved mapping
next to the output figure, so you can always audit what was used.

Usage:
  .venv/bin/python scripts/render_clustered_mean_matrices.py \\
      --mean-matrices-dir gnn_connectivity/output/smoke_combined/clustering/vgae/gmm/mean_matrices \\
      --coords-file gnn_connectivity/data_scalp/biosemi64.txt \\
      --output-png  gnn_connectivity/output/smoke_combined/clustering/vgae/gmm/mean_matrices/mean_matrices_grid_by_network.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Network -> ordered list of electrodes (display order within the network).
# 64 electrodes total. Each electrode appears in exactly one network.
NETWORK_LAYOUT: Dict[str, List[str]] = {
    "AUD": ["FT7", "T7", "TP7", "TP8", "T8", "FT8"],
    "DMN": ["Fp1", "AF7", "AF3", "Fpz", "AFz", "Fp2", "AF4", "AF8",
            "F1", "Fz", "F2", "P1", "Pz", "P2"],
    "FP":  ["F7", "F5", "F3", "F4", "F6", "F8",
            "P9", "P7", "P5", "P3", "P4", "P6", "P8", "P10"],
    "MOT": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "SAL": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6"],
    "VIS": ["PO7", "PO3", "O1", "Iz", "Oz", "POz", "O2", "PO4", "PO8"],
}

# Display network order (left-to-right / top-to-bottom in the figure)
NETWORK_ORDER = ["AUD", "DMN", "FP", "MOT", "SAL", "VIS"]

NETWORK_COLORS = {
    "AUD": "#1f77b4",
    "DMN": "#2ca02c",
    "FP":  "#ff7f0e",
    "MOT": "#d62728",
    "SAL": "#9467bd",
    "VIS": "#7f7f7f",
}


def load_electrode_labels(coords_file: Path) -> List[str]:
    """Read biosemi64 labels in file order (matches the order in the wSMI matrices)."""
    labels: List[str] = []
    fid_re = re.compile(r"^(Fid|LPA|RPA|Nz)", re.IGNORECASE)
    for line in coords_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        label = parts[0]
        if fid_re.match(label):
            continue
        labels.append(label)
    return labels


def build_permutation(
    labels: List[str],
    layout: Dict[str, List[str]] = NETWORK_LAYOUT,
    order: List[str] = NETWORK_ORDER,
) -> Tuple[np.ndarray, List[str], List[Tuple[str, int, int]]]:
    """Return (permutation_indices, reordered_labels, network_spans).

    network_spans is a list of (network_name, start_idx, stop_idx_exclusive)
    over the reordered axis.
    """
    label_to_idx = {l: i for i, l in enumerate(labels)}
    perm: List[int] = []
    reordered: List[str] = []
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
    used = set()
    for net in order:
        net_labels = layout[net]
        start = cursor
        for lab in net_labels:
            if lab not in label_to_idx:
                raise ValueError(f"Electrode '{lab}' in network '{net}' not found in labels file")
            if lab in used:
                raise ValueError(f"Electrode '{lab}' assigned to more than one network")
            perm.append(label_to_idx[lab])
            reordered.append(lab)
            used.add(lab)
            cursor += 1
        spans.append((net, start, cursor))
    missing = [l for l in labels if l not in used]
    if missing:
        raise ValueError(f"Electrodes not assigned to any network: {missing}")
    return np.array(perm, dtype=int), reordered, spans


def reorder_matrix(M: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return M[np.ix_(perm, perm)]


def render_grid(
    matrices_by_cluster: Dict[int, np.ndarray],
    reordered_labels: List[str],
    spans: List[Tuple[str, int, int]],
    output_path: Path,
    title_prefix: str = "Cluster",
) -> None:
    n = len(matrices_by_cluster)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.8 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    vmax = max(np.abs(M).max() for M in matrices_by_cluster.values())
    vmin = -vmax

    for ax, (c, M) in zip(axes, sorted(matrices_by_cluster.items())):
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="equal")
        ax.set_title(f"{title_prefix} {c}  (n=64x64)", fontsize=11)

        # Draw network boundary lines + colored ticks
        for _, start, stop in spans[:-1]:
            ax.axhline(stop - 0.5, color="black", lw=0.8)
            ax.axvline(stop - 0.5, color="black", lw=0.8)

        ax.set_xticks(range(len(reordered_labels)))
        ax.set_yticks(range(len(reordered_labels)))
        ax.set_xticklabels(reordered_labels, rotation=90, fontsize=4.5)
        ax.set_yticklabels(reordered_labels, fontsize=4.5)

        # Color each tick label by its network
        idx_to_net = {}
        for net, start, stop in spans:
            for i in range(start, stop):
                idx_to_net[i] = net
        for i, lbl in enumerate(ax.get_xticklabels()):
            lbl.set_color(NETWORK_COLORS[idx_to_net[i]])
        for i, lbl in enumerate(ax.get_yticklabels()):
            lbl.set_color(NETWORK_COLORS[idx_to_net[i]])

        # Network band annotations across the top
        for net, start, stop in spans:
            mid = (start + stop - 1) / 2.0
            ax.text(mid, -3.0, net, ha="center", va="bottom",
                    fontsize=9, color=NETWORK_COLORS[net], fontweight="bold",
                    clip_on=False)

        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes[len(matrices_by_cluster):]:
        ax.axis("off")

    plt.suptitle(
        "Per-cluster mean wSMI matrix (electrodes grouped by functional network)",
        y=1.02, fontsize=13,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mean-matrices-dir", required=True, type=Path,
                        help="Directory containing mean_matrix_cluster_*.npy")
    parser.add_argument("--coords-file", required=True, type=Path,
                        help="biosemi64.txt with labels in matrix order")
    parser.add_argument("--output-png", required=True, type=Path,
                        help="Output figure path (a .json sidecar is written alongside)")
    args = parser.parse_args()

    labels = load_electrode_labels(args.coords_file)
    if len(labels) != 64:
        raise SystemExit(f"Expected 64 biosemi64 electrodes, got {len(labels)}")

    perm, reordered, spans = build_permutation(labels)

    # Load all per-cluster matrices and reorder.
    cluster_re = re.compile(r"mean_matrix_cluster_(\d+)\.npy$")
    matrices: Dict[int, np.ndarray] = {}
    for npy in sorted(args.mean_matrices_dir.glob("mean_matrix_cluster_*.npy")):
        m = cluster_re.search(npy.name)
        if not m:
            continue
        c = int(m.group(1))
        arr = np.load(npy)
        if arr.shape != (64, 64):
            print(f"  SKIP {npy.name}: unexpected shape {arr.shape}")
            continue
        matrices[c] = reorder_matrix(arr, perm)

    if not matrices:
        raise SystemExit(f"No mean_matrix_cluster_*.npy files found under {args.mean_matrices_dir}")

    print(f"Loaded {len(matrices)} cluster matrices from {args.mean_matrices_dir}")
    for net, start, stop in spans:
        print(f"  {net}: indices [{start}:{stop}], {stop - start} electrodes")

    render_grid(matrices, reordered, spans, args.output_png)
    print(f"\nWrote figure: {args.output_png}")

    # Sidecar JSON: the resolved mapping + the permutation
    sidecar = args.output_png.with_suffix(".mapping.json")
    sidecar.write_text(json.dumps({
        "network_layout": NETWORK_LAYOUT,
        "network_order": NETWORK_ORDER,
        "input_label_order": labels,
        "reordered_labels": reordered,
        "permutation_indices": perm.tolist(),
        "spans": [{"network": n, "start": s, "stop": e} for (n, s, e) in spans],
    }, indent=2))
    print(f"Wrote mapping sidecar: {sidecar}")


if __name__ == "__main__":
    main()

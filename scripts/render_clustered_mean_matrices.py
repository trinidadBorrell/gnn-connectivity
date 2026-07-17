"""
Re-render the per-cluster mean wSMI matrices as 64x64 electrode-by-electrode
heatmaps, in the native electrode order of the coordinates file (the same order
as the wSMI matrices themselves).

NOTE: electrodes are NOT grouped into functional networks. This is scalp EEG;
applying fMRI resting-state network labels (DMN/FP/SAL/...) to individual
electrodes is not meaningful, so no such grouping is done here.

Usage:
  .venv/bin/python scripts/render_clustered_mean_matrices.py \\
      --mean-matrices-dir gnn_connectivity/output/smoke_combined/clustering/vgae/gmm/mean_matrices \\
      --coords-file gnn_connectivity/data_scalp/biosemi64.txt \\
      --output-png  gnn_connectivity/output/smoke_combined/clustering/vgae/gmm/mean_matrices/mean_matrices_grid.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


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


_DIVERGING_CMAPS = {
    "RdBu", "RdBu_r", "RdYlBu", "RdYlBu_r", "RdYlGn", "RdYlGn_r",
    "seismic", "bwr", "PuOr", "PuOr_r", "BrBG", "BrBG_r", "coolwarm",
    "PiYG", "PRGn", "Spectral", "Spectral_r",
}


def render_grid(
    matrices_by_cluster: Dict[int, np.ndarray],
    labels: List[str],
    output_path: Path,
    title_prefix: str = "Cluster",
    cmap: str = "viridis",
) -> None:
    n = len(matrices_by_cluster)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.8 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    if cmap in _DIVERGING_CMAPS:
        # Diverging map: pin zero to the colorbar center.
        vmax = max(np.abs(M).max() for M in matrices_by_cluster.values())
        vmin = -vmax
    else:
        # Sequential map (viridis, etc.): use the data's natural range so the
        # full colormap is utilized.
        vmin = min(M.min() for M in matrices_by_cluster.values())
        vmax = max(M.max() for M in matrices_by_cluster.values())

    for ax, (c, M) in zip(axes, sorted(matrices_by_cluster.items())):
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
        ax.set_title(f"{title_prefix} {c}  (n=64x64)", fontsize=11)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=4.5)
        ax.set_yticklabels(labels, fontsize=4.5)

        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes[len(matrices_by_cluster):]:
        ax.axis("off")

    plt.suptitle(
        "Per-cluster mean wSMI matrix (native electrode order)",
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
    parser.add_argument("--cmap", default="viridis",
                        help="Matplotlib colormap name. Default 'viridis' (blue->yellow). "
                             "Diverging maps (RdBu_r, seismic, ...) use a symmetric ±vmax "
                             "range; sequential maps use the data's natural min/max.")
    args = parser.parse_args()

    labels = load_electrode_labels(args.coords_file)
    if len(labels) != 64:
        raise SystemExit(f"Expected 64 biosemi64 electrodes, got {len(labels)}")

    # Load all per-cluster matrices (native electrode order, no reordering).
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
        matrices[c] = arr

    if not matrices:
        raise SystemExit(f"No mean_matrix_cluster_*.npy files found under {args.mean_matrices_dir}")

    print(f"Loaded {len(matrices)} cluster matrices from {args.mean_matrices_dir}")

    render_grid(matrices, labels, args.output_png, cmap=args.cmap)
    print(f"\nWrote figure: {args.output_png}")

    # Sidecar JSON: the electrode order used (matrix axis order).
    sidecar = args.output_png.with_suffix(".mapping.json")
    sidecar.write_text(json.dumps({
        "electrode_order": labels,
    }, indent=2))
    print(f"Wrote electrode-order sidecar: {sidecar}")


if __name__ == "__main__":
    main()

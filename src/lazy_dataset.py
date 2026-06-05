"""
LAZY ON-DISK WSMI DATASET
=========================
Memory-efficient replacement for the materialized list of PyG `Data` objects
used in matrix mode. Each session's compressed .npz is converted once to an
uncompressed mmap-friendly .npy cache; __getitem__ then slices a single
(n_nodes, n_nodes) epoch matrix out of the mmap region.

Resident RAM is O(batch_size * n_nodes * n_nodes), independent of dataset
size. The full 164-session dataset (~105k epochs) becomes tractable.

Compatible with the existing pipeline:
- subclass of torch.utils.data.Dataset; works directly with PyG DataLoader
- exposes `.subject_ids` for GroupKFold splits (avoids iterating __getitem__)
- exposes `.compute_train_stats()` / `.set_normalization()` so normalization
  can be computed streaming and applied in __getitem__ rather than materialized
- pickles cleanly via torch.save (mmap dict is excluded from state)
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data


class WsmiOnDiskDataset(torch.utils.data.Dataset):
    """Lazy dataset of per-epoch wSMI matrices backed by mmapped .npy files."""

    def __init__(
        self,
        sessions: List[dict],
        edge_index: torch.Tensor,
        electrode_labels,
        cache_dir: str,
        npz_key: str = 'data',
        n_nodes: Optional[int] = None,
    ):
        """
        Args:
            sessions: list of dicts with keys
                {'subject_id', 'session_num', 'npz_path', 'n_epochs' (optional)}.
                If 'n_epochs' is missing it's filled in by inspecting the .npy
                cache (created here if it doesn't exist yet).
            edge_index: shared adjacency edge_index (long tensor, shape [2, E]).
            electrode_labels: list of electrode names, stored per graph item.
            cache_dir: directory holding the mmap-friendly .npy caches.
            npz_key: key inside the source .npz holding the (n_eps, n, n) tensor.
            n_nodes: optional validation hint; raises if a cached array
                disagrees with this.
        """
        super().__init__()
        self.npz_key = npz_key
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Materialize the .npy caches and the per-session metadata.
        self.sessions = []
        for s in sessions:
            npy_path = self._ensure_npy_cache(s)
            # peek at shape via mmap (touches only the .npy header)
            arr_mmap = np.load(npy_path, mmap_mode='r')
            shape = arr_mmap.shape
            del arr_mmap  # release the mmap; we'll re-open it lazily in _get_mmap
            if len(shape) != 3 or shape[1] != shape[2]:
                raise ValueError(f"{npy_path}: expected (n_epochs, n, n), got {shape}")
            if n_nodes is not None and shape[1] != n_nodes:
                raise ValueError(
                    f"{npy_path}: has {shape[1]} nodes, expected {n_nodes}"
                )
            entry = dict(s)
            entry['npy_path'] = npy_path
            entry['n_epochs'] = int(shape[0])
            entry['n_nodes'] = int(shape[1])
            self.sessions.append(entry)

        # Global flat index: i -> (session_idx, epoch_idx)
        self._index: List[tuple] = []
        for si, sess in enumerate(self.sessions):
            for ei in range(sess['n_epochs']):
                self._index.append((si, ei))

        # Pre-compute the per-item subject id list for GroupKFold (fast, no I/O).
        self._subject_ids = [self.sessions[si]['subject_id'] for si, _ in self._index]

        self.edge_index = edge_index
        self.electrode_labels = list(electrode_labels)
        self.n_nodes = self.sessions[0]['n_nodes'] if self.sessions else None

        # Normalization parameters; populated via set_normalization().
        self.x_min: Optional[float] = None
        self.x_max: Optional[float] = None
        self.x_range: Optional[float] = None

        # Per-process mmap cache (NOT pickled). DataLoader workers each rebuild
        # their own on first __getitem__.
        self._mmaps: dict = {}

    # ------------------------------------------------------------------ utils
    def _ensure_npy_cache(self, s: dict) -> str:
        target = os.path.join(
            self.cache_dir,
            f"sub-{s['subject_id']}",
            f"ses-{s['session_num']}",
            'data.npy',
        )
        if os.path.exists(target):
            return target
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with np.load(s['npz_path']) as d:
            if self.npz_key not in d:
                raise KeyError(f"{s['npz_path']} missing key '{self.npz_key}'")
            arr = d[self.npz_key].astype(np.float32, copy=False)
        # Write to a temp path then rename: avoids leaving a half-written file
        # if we get interrupted mid-conversion. np.save would auto-append .npy
        # to a bare path; using a file handle bypasses that behavior.
        tmp = target + '.tmp'
        with open(tmp, 'wb') as f:
            np.save(f, arr, allow_pickle=False)
        os.replace(tmp, target)
        return target

    def _get_mmap(self, si: int) -> np.ndarray:
        path = self.sessions[si]['npy_path']
        arr = self._mmaps.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode='r')
            self._mmaps[path] = arr
        return arr

    # ----------------------------------------------------------- normalization
    def compute_train_stats(self) -> tuple:
        """Exact min/max across every epoch matrix (streaming via mmap).

        Operates per-session — each mmap is a flat sequential read of the .npy
        file, so this is O(disk-throughput) rather than O(RAM).
        """
        x_min = float('inf')
        x_max = float('-inf')
        for si in range(len(self.sessions)):
            arr = self._get_mmap(si)
            x_min = min(x_min, float(arr.min()))
            x_max = max(x_max, float(arr.max()))
        x_range = x_max - x_min
        print(f"  Normalization stats from {len(self.sessions)} sessions "
              f"({len(self)} epochs): x_min={x_min:.6f}, x_max={x_max:.6f}, "
              f"x_range={x_range:.6f}")
        return x_min, x_max, x_range

    def set_normalization(self, x_min: float, x_max: float, x_range: float) -> None:
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.x_range = float(x_range)

    # ------------------------------------------------------------ Dataset API
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Data:
        si, ei = self._index[idx]
        arr = self._get_mmap(si)
        # Copy out of the mmap region into a plain ndarray, then to tensor.
        x_np = np.array(arr[ei], copy=True, dtype=np.float32)
        if self.x_range is not None:
            if self.x_range > 0:
                x_np = (2.0 * (x_np - self.x_min) / self.x_range - 1.0).astype(np.float32, copy=False)
            else:
                x_np = np.zeros_like(x_np)
        x = torch.from_numpy(x_np)
        sess = self.sessions[si]
        data = Data(x=x, edge_index=self.edge_index)
        data.subject_id = sess['subject_id']
        data.session_num = sess['session_num']
        data.matrix_idx = ei
        data.electrode_labels = self.electrode_labels
        # Cohort (e.g. 'DOC' or 'control') passed through when available;
        # makes per-cohort cluster interpretation downstream trivial.
        cohort = sess.get('cohort')
        if cohort is not None:
            data.cohort = cohort
        return data

    # ----------------------------------------------------------- introspection
    @property
    def subject_ids(self) -> List[str]:
        """One subject_id per __getitem__ index (for GroupKFold)."""
        return self._subject_ids

    @property
    def data(self):
        """Backward-compat alias: existing call sites do `dataset.data` to get
        an iterable of graphs. We are already such an iterable.
        """
        return self

    def subset(self, indices) -> "_WsmiSubset":
        """View of the dataset over a subset of indices (no copy)."""
        return _WsmiSubset(self, list(indices))

    # ------------------------------------------------------------------ pickle
    def __getstate__(self):
        # Don't pickle the mmap dict — handles aren't picklable and workers
        # rebuild their own anyway.
        state = self.__dict__.copy()
        state['_mmaps'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mmaps = {}


class _WsmiSubset(torch.utils.data.Dataset):
    """Subset view that preserves the on-disk dataset API the rest of the
    pipeline expects (subject_ids, data, indexing, etc.).
    """

    def __init__(self, parent: WsmiOnDiskDataset, indices: List[int]):
        super().__init__()
        self.parent = parent
        self.indices = list(indices)
        # Pre-extract subject ids for fast leakage checks.
        self._subject_ids = [parent.subject_ids[i] for i in self.indices]
        # Cache the set of session indices this subset spans (used for
        # streaming normalization stats without touching out-of-split sessions).
        self._session_indices = sorted({parent._index[i][0] for i in self.indices})

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Data:
        return self.parent[self.indices[i]]

    @property
    def subject_ids(self) -> List[str]:
        return self._subject_ids

    @property
    def data(self):
        return self

    @property
    def edge_index(self):
        return self.parent.edge_index

    @property
    def electrode_labels(self):
        return self.parent.electrode_labels

    # Polymorphism hooks used by train.compute_normalization_stats /
    # apply_normalization. Without these, the subset would fall through to the
    # in-memory torch.cat path and materialize the entire split — the exact
    # OOM trap we built this class to avoid.
    def compute_train_stats(self) -> tuple:
        """Streaming min/max over only the sessions in this subset."""
        x_min = float('inf')
        x_max = float('-inf')
        for si in self._session_indices:
            arr = self.parent._get_mmap(si)
            x_min = min(x_min, float(arr.min()))
            x_max = max(x_max, float(arr.max()))
        x_range = x_max - x_min
        print(f"  Normalization stats from {len(self._session_indices)} sessions "
              f"({len(self)} epochs): x_min={x_min:.6f}, x_max={x_max:.6f}, "
              f"x_range={x_range:.6f}")
        return x_min, x_max, x_range

    def set_normalization(self, x_min: float, x_max: float, x_range: float) -> None:
        # Subsets share the same underlying mmap; normalize on the parent.
        self.parent.set_normalization(x_min, x_max, x_range)

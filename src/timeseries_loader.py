"""
TIME-SERIES LOADER
==================
Load per-electrode EEG time-series (raw .fif epochs) into PyTorch Geometric
graphs. Mirrors `wsmi_loader.load_wsmi_dataset` so the rest of the pipeline
(splits, train, latent extraction, clustering, decoder eval) is unchanged.

Each epoch becomes ONE graph:
- node features x: shape (n_channels, n_timepoints) — the raw time-series
  for each electrode in the epoch window
- edge_index: k-NN over the biosemi64 electrode coordinates (shared adjacency,
  same as the wSMI pipeline)
- attached metadata: subject_id, session_num, acq, matrix_idx (epoch index),
  diagnosis, diagnosis_group, group ("patient"/"control"), electrode_labels.

Window selection:
- target_sfreq defaults to 100 Hz to match the existing wSMI pipeline
  (`nice_epochs_sfreq-100Hz_recombine-biosemi64_dur-16s`).
- window_sec defaults to 16 s -> 1600 samples per node, aligning 1:1 with the
  pre-computed wSMI matrices.
- If the .fif epoch is longer than window_sec * target_sfreq, it is cropped
  from the start.
"""
import os
import re
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from wsmi_loader import (
    DIAGNOSIS_GROUP_MAP, DROP_GROUP,
    _build_adjacency, _is_control, _iter_wsmi_files, _load_diagnosis_lookup,
    _load_pkl_data, _resolve_diagnosis, _symmetrize_and_clean,
)


FNAME_RE = re.compile(
    r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_task-(?P<task>[^_]+)_acq-(?P<acq>[^_]+)_epo\.fif$"
)


def _channel_correlation_matrix(ts: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-graph Pearson correlation matrix between channels.

    Args:
        ts: shape (n_channels, n_timepoints).
    Returns:
        corr: shape (n_channels, n_channels), float32.
    """
    centered = ts - ts.mean(axis=1, keepdims=True)
    std = centered.std(axis=1, keepdims=True).clip(min=eps)
    normed = centered / std
    return ((normed @ normed.T) / ts.shape[1]).astype(np.float32)


def _iter_fif_files(root: str):
    """Yield (subject_id, session, task, acq, fif_path) for every epoch file under `root`."""
    for sub_folder in sorted(os.listdir(root)):
        if not sub_folder.startswith("sub-"):
            continue
        sub_id = sub_folder[4:]
        sub_path = os.path.join(root, sub_folder)
        if not os.path.isdir(sub_path):
            continue
        for ses_folder in sorted(os.listdir(sub_path)):
            if not ses_folder.startswith("ses-"):
                continue
            ses = ses_folder[4:]
            eeg_dir = os.path.join(sub_path, ses_folder, "eeg")
            if not os.path.isdir(eeg_dir):
                continue
            for fname in sorted(os.listdir(eeg_dir)):
                m = FNAME_RE.match(fname)
                if not m:
                    continue
                yield (sub_id, ses, m.group("task"), m.group("acq"),
                       os.path.join(eeg_dir, fname))


def _load_epochs_array(fif_path: str, target_sfreq: float,
                       window_samples: int, expected_n_channels: int,
                       electrode_labels: List[str],
                       crop: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Read a .fif epoch file, resample if needed, crop to window_samples.

    If `crop=(tmin, tmax)` is given (seconds, relative to the epoch's time axis),
    the epoch is time-cropped to that window via MNE before slicing — this is the
    local-global path (e.g. -0.2..0.6 s). Otherwise the first `window_samples`
    samples are kept from the start (the resting-state path).

    Returns array of shape (n_epochs, n_channels, window_samples) ordered to
    match `electrode_labels`, or None on failure (caller decides what to do).
    """
    epochs = mne.read_epochs(fif_path, verbose="error", preload=True)
    if epochs.info["sfreq"] != target_sfreq:
        epochs = epochs.resample(target_sfreq, verbose="error")
    if crop is not None:
        epochs = epochs.crop(tmin=crop[0], tmax=crop[1], verbose="error")

    # Reorder channels to match electrode_labels (drop extras, keep order).
    present = [ch for ch in electrode_labels if ch in epochs.ch_names]
    if len(present) != expected_n_channels:
        return None
    epochs = epochs.pick_channels(present, ordered=True)

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    if data.shape[2] < window_samples:
        return None
    data = data[:, :, :window_samples]
    return data.astype(np.float32)


def _build_wsmi_path_index(wsmi_dir: Optional[str], subject_filter: Optional[set] = None,
                           ) -> Dict[Tuple[str, str, str], str]:
    """Map (subject, session, acq) -> wSMI pkl path (cheap; no array loaded).

    The matched wSMI stack is loaded per-recording on demand (see
    `load_timeseries_dataset`) so we never hold every recording's stack in RAM at
    once. Returns an empty dict if `wsmi_dir` is None/missing.
    """
    index: Dict[Tuple[str, str, str], str] = {}
    if wsmi_dir is None or not os.path.isdir(wsmi_dir):
        return index
    for sub_id, ses, acq, pkl_path in _iter_wsmi_files(wsmi_dir):
        if subject_filter is not None and sub_id not in subject_filter:
            continue
        index[(sub_id, ses, acq)] = pkl_path
    return index


def _load_wsmi_stack(pkl_path: str) -> Optional[np.ndarray]:
    """Load one recording's wSMI stack (n_epochs, C, C) from its pkl, or None."""
    try:
        arr = _load_pkl_data(pkl_path)  # (1, n_epochs, C, C)
    except Exception:
        return None
    if arr.ndim != 4:
        return None
    return arr[0]


def load_timeseries_dataset(
    patient_dir: str,
    control_dir: Optional[str],
    diagnosis_csv: str,
    coords_file: str,
    k: int = 6,
    target_sfreq: float = 100.0,
    window_sec: float = 16.0,
    crop: Optional[Tuple[float, float]] = None,
    wsmi_patient_dir: Optional[str] = None,
    wsmi_control_dir: Optional[str] = None,
    subject_filter: Optional[set] = None,
    task: Optional[str] = None,
    granularity: str = "coarse",
    max_epochs_per_recording: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[Data], List[str], List[str]]:
    """Mirror of wsmi_loader.load_wsmi_dataset but with raw time-series node features.

    Returns:
        graphs: list of torch_geometric.data.Data, each with x of shape
                (n_channels, window_sec * target_sfreq) and the same metadata
                attributes as the wSMI loader's output (subject_id,
                session_num, acq, matrix_idx, diagnosis, diagnosis_group,
                group, electrode_labels).
        subject_ids: parallel list of subject IDs.
        diagnosis_groups: parallel list of diagnosis_group strings.
    """
    lookup = _load_diagnosis_lookup(diagnosis_csv)
    adjacency, electrode_labels = _build_adjacency(coords_file, k=k)
    edge_index = torch.tensor(
        np.array(adjacency.tocoo().nonzero()), dtype=torch.long
    )
    n_channels = len(electrode_labels)
    if crop is not None:
        # Samples spanned by [tmin, tmax). We deliberately do NOT add the +1 of an
        # end-inclusive crop: rs epochs end at 0.59 s (80 samples) while lg reach
        # 0.6 s (81 samples), so the common, consistent window is floor(dur*sfreq)
        # = 80 @ 100 Hz for -0.2..0.6. Each recording is then sliced to exactly
        # this many samples, so rs and lg graphs share the same feature width.
        window_samples = int(round((crop[1] - crop[0]) * target_sfreq))
    else:
        window_samples = int(round(window_sec * target_sfreq))

    # Matched-wSMI path indexes (one per cohort) for the outlier filter. Stacks
    # are loaded per-recording on demand so we never hold all of them in RAM.
    wsmi_indexes = {
        "patient": _build_wsmi_path_index(wsmi_patient_dir, subject_filter),
        "control": _build_wsmi_path_index(wsmi_control_dir, subject_filter),
    }
    attach_wsmi = any(len(v) > 0 for v in wsmi_indexes.values())
    rng = np.random.default_rng(seed)

    graphs: List[Data] = []
    subject_ids: List[str] = []
    diagnosis_groups: List[str] = []

    for root, group_name in ((patient_dir, "patient"), (control_dir, "control")):
        if root is None:
            if verbose:
                print(f"  skipping {group_name} cohort (control_dir=None)")
            continue
        if not os.path.isdir(root):
            if verbose:
                print(f"  skipping missing folder: {root}")
            continue
        n_files = 0
        n_epochs_total = 0
        n_dropped_recs = 0
        for sub_id, ses, file_task, acq, fif_path in _iter_fif_files(root):
            if subject_filter is not None and sub_id not in subject_filter:
                continue
            if task is not None and file_task != task:
                continue
            n_files += 1
            diag, diag_group = _resolve_diagnosis(sub_id, ses, lookup, granularity)
            if diag_group == DROP_GROUP:
                n_dropped_recs += 1
                continue
            try:
                data = _load_epochs_array(
                    fif_path, target_sfreq=target_sfreq,
                    window_samples=window_samples,
                    expected_n_channels=n_channels,
                    electrode_labels=electrode_labels,
                    crop=crop,
                )
            except Exception as e:
                if verbose:
                    print(f"  WARN failed to read {fif_path}: {e}")
                continue
            if data is None:
                if verbose:
                    print(f"  WARN skipping {fif_path}: channel mismatch or window too long")
                continue

            # Matched wSMI stack for this recording (loaded on demand, then freed).
            wsmi_stack = None
            n_epochs_file = data.shape[0]
            if attach_wsmi:
                pkl_path = wsmi_indexes[group_name].get((sub_id, ses, acq))
                wsmi_stack = _load_wsmi_stack(pkl_path) if pkl_path else None
                if wsmi_stack is None:
                    if verbose:
                        print(f"  WARN no matched wSMI for {sub_id}/{ses}/{acq}; "
                              f"skipping {n_epochs_file} epochs (cannot outlier-filter)")
                    continue
                # Align epoch counts: only keep the common prefix.
                n_common = min(n_epochs_file, wsmi_stack.shape[0])
                if n_common != n_epochs_file and verbose:
                    print(f"  [wsmi-match] {sub_id}/{ses}/{acq}: epoch count "
                          f"fif={n_epochs_file} wsmi={wsmi_stack.shape[0]} -> using {n_common}")
            else:
                n_common = n_epochs_file

            # Optional per-recording subsample (bounds memory / speeds tuning).
            epoch_indices = range(n_common)
            if max_epochs_per_recording is not None and n_common > max_epochs_per_recording:
                epoch_indices = sorted(
                    rng.choice(n_common, size=max_epochs_per_recording, replace=False))

            for epoch_idx in epoch_indices:
                x_arr = data[epoch_idx]  # (n_channels, window_samples)
                x = torch.tensor(x_arr, dtype=torch.float32)
                g = Data(x=x, edge_index=edge_index)
                g.subject_id = sub_id
                g.session_num = ses
                g.acq = acq
                g.matrix_idx = int(epoch_idx)
                g.diagnosis = diag
                g.diagnosis_group = diag_group
                g.group = group_name
                g.electrode_labels = electrode_labels
                # NOTE: raw_matrix (Pearson corr) is NOT stored here to save RAM on
                # large runs; it is recomputed lazily from x at clustering time
                # (corr is invariant to the global affine normalization).
                if attach_wsmi:
                    # Basis for the Mahalanobis outlier filter (real wSMI).
                    g.wsmi_matrix = _symmetrize_and_clean(wsmi_stack[epoch_idx])
                graphs.append(g)
                subject_ids.append(sub_id)
                diagnosis_groups.append(diag_group)
                n_epochs_total += 1
            del wsmi_stack, data
        if verbose:
            dropped_reason = ("unlabeled only; EMCS/COMA kept"
                              if granularity == "fine" else "EMCS/COMA/unlabeled")
            print(f"  {group_name}: {n_files} files, {n_epochs_total} epoch-graphs from {root} "
                  f"(dropped {n_dropped_recs} recordings: {dropped_reason})")

    if verbose:
        from collections import Counter
        print(f"Total graphs: {len(graphs)}; unique subjects: {len(set(subject_ids))}")
        print(f"Diagnosis-group counts (graphs): {dict(Counter(diagnosis_groups))}")
        per_subj_group: Dict[str, str] = {}
        for s, g in zip(subject_ids, diagnosis_groups):
            per_subj_group[s] = g
        print(f"Diagnosis-group counts (subjects): {dict(Counter(per_subj_group.values()))}")
        if crop is not None:
            win_desc = f"crop [{crop[0]}, {crop[1]}] s"
        else:
            win_desc = f"{window_sec} s from start"
        print(f"Per-graph x shape: ({n_channels}, {window_samples}) "
              f"= ({n_channels} channels, {win_desc} @ {target_sfreq} Hz)")

    return graphs, subject_ids, diagnosis_groups


def attach_real_wsmi(
    graphs,
    wsmi_patient_dir: Optional[str],
    wsmi_control_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[int, int]:
    """Populate ``g.raw_matrix`` on time-series graphs with the matched real wSMI.

    Time-series graphs do not store a connectivity matrix (to save RAM), so the
    cluster summaries fall back to a Pearson correlation recomputed from the raw
    voltage traces — a different, near-constant quantity that makes per-cluster
    mean matrices look almost identical. This walks each graph back to the wSMI
    epoch it was built from (via ``subject_id``/``session_num``/``acq`` ->
    pkl path, then ``matrix_idx`` into the stack) and attaches the true 64x64
    wSMI matrix as ``raw_matrix`` so ``cluster_mean_matrices`` averages real wSMI.

    The lookup mirrors the loader: ``matrix_idx`` is the shared epoch index into
    both the .fif epochs and the wSMI stack (aligned on the common prefix). Each
    recording's stack is loaded once. Returns ``(n_attached, n_unmatched)``.
    """
    from collections import defaultdict

    index: Dict[Tuple[str, str, str], str] = {}
    index.update(_build_wsmi_path_index(wsmi_patient_dir))
    if wsmi_control_dir:
        index.update(_build_wsmi_path_index(wsmi_control_dir))
    if not index:
        if verbose:
            print("[wsmi-attach] no wSMI directories given; nothing attached")
        return 0, len(list(graphs))

    by_rec: Dict[Tuple[str, str, str], list] = defaultdict(list)
    for g in graphs:
        key = (str(getattr(g, "subject_id", "")),
               str(getattr(g, "session_num", "")),
               str(getattr(g, "acq", "")))
        by_rec[key].append(g)

    n_ok = n_miss = 0
    stack_cache: Dict[str, Optional[np.ndarray]] = {}
    for key, recs in by_rec.items():
        pkl = index.get(key)
        if pkl is None:
            n_miss += len(recs)
            continue
        if pkl not in stack_cache:
            stack_cache[pkl] = _load_wsmi_stack(pkl)
        stack = stack_cache[pkl]
        if stack is None:
            n_miss += len(recs)
            continue
        for g in recs:
            idx = int(getattr(g, "matrix_idx", -1))
            if 0 <= idx < stack.shape[0]:
                g.raw_matrix = _symmetrize_and_clean(stack[idx])
                n_ok += 1
            else:
                n_miss += 1
    if verbose:
        print(f"[wsmi-attach] attached real wSMI to {n_ok} graphs "
              f"({n_miss} unmatched -> Pearson fallback)")
    return n_ok, n_miss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_dir", required=True)
    parser.add_argument("--control_dir", required=True)
    parser.add_argument("--diagnosis_csv", required=True)
    parser.add_argument("--coords_file", required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--sfreq", type=float, default=100.0)
    parser.add_argument("--window_sec", type=float, default=16.0)
    args = parser.parse_args()

    graphs, subj, grp = load_timeseries_dataset(
        args.patient_dir, args.control_dir, args.diagnosis_csv,
        args.coords_file, k=args.k,
        target_sfreq=args.sfreq, window_sec=args.window_sec,
    )
    print(f"\nSummary: {len(graphs)} graphs across {len(set(subj))} subjects")

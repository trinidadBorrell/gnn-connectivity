"""
wSMI LOADER
===========
Load wSMI-theta connectivity matrices (junifer output) into PyTorch Geometric graphs.

Inputs:
- Two BIDS-style folders of `*_desc-wsmi_connectivity.pkl` files (one per
  subject-session-acquisition). Each pickle holds a dict with a single
  marker entry whose `data` is shape (1, n_epochs, 64, 64).
- A diagnosis CSV mapping subject -> diagnosis_crs_final.

Each epoch becomes ONE graph:
- node features x: the wSMI matrix (each row = a node's 64-d connectivity profile)
- edge_index: k-NN over biosemi64 electrode coordinates (shared adjacency)
- attached metadata: subject_id, session_num, acq, matrix_idx (epoch index),
  diagnosis, diagnosis_group, group ("patient"/"control").

Subject IDs:
- Numeric (sub-001 ... sub-014) -> control -> diagnosis_group="HC".
- Alphanumeric (sub-AA048, sub-BM059, ...) -> patient -> diagnosis_group looked up.
"""
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

# Reuse adjacency utilities from the existing preprocessing module
from preprocessing import EEGtoGraph


FNAME_RE = re.compile(
    r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_acq-(?P<acq>[^_]+)_desc-wsmi_connectivity\.pkl$"
)

# diagnostic_crs_final -> coarse group
DIAGNOSIS_GROUP_MAP = {
    "UWS": "UWS",
    "VS": "UWS",
    "MCS-": "MCS",
    "MCS+": "MCS",
    "MCS": "MCS",
    # EMCS / COMA intentionally absent -> resolved to "DROP" and filtered out
}

# Fine granularity: keep the raw diagnostic_crs_final label (so MCS-, MCS+, EMCS,
# COMA stay distinct and are NOT dropped). VS is the single merge -> UWS.
FINE_DIAGNOSIS_GROUP_MAP = {"VS": "UWS"}

DROP_GROUP = "DROP"


def _load_diagnosis_lookup(csv_path: str) -> Dict[Tuple[str, str], str]:
    """Return {(subject, session_str): diagnostic_crs_final}.

    session is normalized to a zero-padded string ('1' -> '01') to match BIDS
    'ses-01' folder naming.
    """
    df = pd.read_csv(csv_path)
    lookup: Dict[Tuple[str, str], str] = {}
    for _, row in df.iterrows():
        subj = str(row["subject"]).strip()
        sess_raw = row["session"]
        if pd.isna(sess_raw):
            continue
        try:
            sess = f"{int(float(sess_raw)):02d}"
        except (ValueError, TypeError):
            sess = str(sess_raw).zfill(2)
        diag = row.get("diagnostic_crs_final")
        if isinstance(diag, str) and diag.strip():
            lookup[(subj, sess)] = diag.strip()
    return lookup


def _load_pkl_data(pkl_path: str) -> np.ndarray:
    """Return the ndarray from a junifer wSMI .pkl, shape (1, n_epochs, 64, 64)."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if len(d) != 1:
        raise ValueError(f"Expected one marker key in {pkl_path}, found {list(d.keys())}")
    inner = next(iter(d.values()))
    arr = inner["data"]
    return arr


def _is_control(subject_id: str) -> bool:
    """Numeric subject IDs are controls (HC). Alphanumeric are patients."""
    return subject_id.isdigit()


def _resolve_diagnosis(
    subject_id: str, session: str, lookup: Dict[Tuple[str, str], str],
    granularity: str = "coarse",
) -> Tuple[str, str]:
    """Return (diagnosis, diagnosis_group).

    granularity="coarse" -> diagnosis_group in {CONTROL, UWS, MCS, DROP}; the DROP
    sentinel covers EMCS, COMA, and unlabeled patients (callers filter these out).
    granularity="fine" -> diagnosis_group is the raw diagnostic_crs_final label
    (UWS, MCS-, MCS+, EMCS, COMA, ...) with the single merge VS -> UWS; EMCS/COMA
    are kept. Controls are CONTROL; only missing/unknown diagnoses are DROP.
    """
    if _is_control(subject_id):
        return "HC", "CONTROL"
    diag = lookup.get((subject_id, session))
    if diag is None:
        for (s, _), v in lookup.items():
            if s == subject_id:
                diag = v
                break
    if diag is None:
        return "UNK", DROP_GROUP
    if granularity == "fine":
        return diag, FINE_DIAGNOSIS_GROUP_MAP.get(diag, diag)
    return diag, DIAGNOSIS_GROUP_MAP.get(diag, DROP_GROUP)


def _build_adjacency(coords_file: str, k: int = 6) -> Tuple[sp.csr_matrix, List[str]]:
    """Build the shared biosemi64 k-NN adjacency once."""
    with open(coords_file) as f:
        first_line = f.readline().split()
    if len(first_line) >= 4:
        coords_df = pd.read_csv(
            coords_file, sep=r"\s+", header=None,
            names=["label", "x", "y", "z"], usecols=[0, 1, 2, 3],
        )
        coords_df = coords_df[~coords_df["label"].str.startswith("Fid")].reset_index(drop=True)
    else:
        from eeg_positions import get_elec_coords
        labels = np.loadtxt(coords_file, usecols=(0,), dtype=str)
        coords = get_elec_coords(system="1005", as_mne_montage=False)
        coords_df = coords[coords["label"].isin(labels)].copy()
        coords_df = coords_df.set_index("label").loc[labels].reset_index()

    k_nearest, _ = EEGtoGraph.find_k_nearest_sensors(coords_df, k=k)
    labels = list(coords_df["label"].values)
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    rows, cols = [], []
    for i, l in enumerate(labels):
        for neigh, _ in k_nearest[l]:
            j = label_to_idx[neigh]
            rows.append(i)
            cols.append(j)
    A = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=np.float32
    )
    A = A + A.T
    A = (A > 0).astype(np.float32)
    return A, labels


def _symmetrize_and_clean(mat: np.ndarray) -> np.ndarray:
    """Symmetrize wSMI matrix, zero diagonal, replace NaN/Inf with 0."""
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 0.0)
    return mat.astype(np.float32)


def _iter_wsmi_files(root: str):
    """Yield (subject_id, session, acq, pkl_path) for every wSMI file under `root`."""
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
                yield sub_id, ses, m.group("acq"), os.path.join(eeg_dir, fname)


def load_wsmi_dataset(
    patient_dir: str,
    control_dir: Optional[str],
    diagnosis_csv: str,
    coords_file: str,
    k: int = 6,
    subject_filter: Optional[set] = None,
    granularity: str = "coarse",
    max_epochs_per_recording: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[Data], List[str], List[str]]:
    """
    Load both wSMI directories and produce a list of per-epoch graphs.

    Returns:
        graphs: list of torch_geometric.data.Data, each with attributes
                subject_id, session_num, acq, matrix_idx, diagnosis,
                diagnosis_group, group ('patient' or 'control'),
                electrode_labels, raw_matrix (float32 64x64).
        subject_ids: parallel list of subject IDs (e.g. 'AA048', '001').
        diagnosis_groups: parallel list of diagnosis_group strings.
    """
    lookup = _load_diagnosis_lookup(diagnosis_csv)
    adjacency, electrode_labels = _build_adjacency(coords_file, k=k)
    edge_index = torch.tensor(
        np.array(adjacency.tocoo().nonzero()), dtype=torch.long
    )
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
        n_epochs = 0
        n_dropped = 0
        for sub_id, ses, acq, pkl_path in _iter_wsmi_files(root):
            if subject_filter is not None and sub_id not in subject_filter:
                continue
            n_files += 1
            try:
                arr = _load_pkl_data(pkl_path)  # (1, n_epochs, 64, 64)
            except Exception as e:
                if verbose:
                    print(f"  WARN failed to read {pkl_path}: {e}")
                continue
            if arr.ndim != 4 or arr.shape[2] != 64 or arr.shape[3] != 64:
                if verbose:
                    print(f"  WARN unexpected shape {arr.shape} in {pkl_path}")
                continue
            epoch_mats = arr[0]
            diag, diag_group = _resolve_diagnosis(sub_id, ses, lookup, granularity)
            if diag_group == DROP_GROUP:
                n_dropped += 1
                continue
            n_ep = epoch_mats.shape[0]
            epoch_indices = range(n_ep)
            if max_epochs_per_recording is not None and n_ep > max_epochs_per_recording:
                epoch_indices = sorted(
                    rng.choice(n_ep, size=max_epochs_per_recording, replace=False))
            for epoch_idx in epoch_indices:
                mat = _symmetrize_and_clean(epoch_mats[epoch_idx])
                x = torch.tensor(mat, dtype=torch.float32)
                data = Data(x=x, edge_index=edge_index)
                data.subject_id = sub_id
                data.session_num = ses
                data.acq = acq
                data.matrix_idx = int(epoch_idx)
                data.diagnosis = diag
                data.diagnosis_group = diag_group
                data.group = group_name
                data.electrode_labels = electrode_labels
                # raw_matrix doubles as the Mahalanobis outlier-filter basis here
                # (the filter falls back to raw_matrix when wsmi_matrix is absent),
                # so we keep a single copy rather than two.
                data.raw_matrix = mat
                graphs.append(data)
                subject_ids.append(sub_id)
                diagnosis_groups.append(diag_group)
                n_epochs += 1
        if verbose:
            dropped_reason = ("unlabeled only; EMCS/COMA kept"
                              if granularity == "fine" else "EMCS/COMA/unlabeled")
            print(f"  {group_name}: {n_files} files, {n_epochs} epoch-graphs from {root} "
                  f"(dropped {n_dropped} recordings: {dropped_reason})")

    if verbose:
        print(f"Total graphs: {len(graphs)}; unique subjects: {len(set(subject_ids))}")
        from collections import Counter
        per_group = Counter(diagnosis_groups)
        print(f"Diagnosis-group counts (graphs): {dict(per_group)}")
        per_subj_group = {}
        for s, g in zip(subject_ids, diagnosis_groups):
            per_subj_group[s] = g
        from collections import Counter as C
        print(f"Diagnosis-group counts (subjects): {dict(C(per_subj_group.values()))}")

    return graphs, subject_ids, diagnosis_groups


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_dir", required=True)
    parser.add_argument("--control_dir", required=True)
    parser.add_argument("--diagnosis_csv", required=True)
    parser.add_argument("--coords_file", required=True)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    graphs, subj, grp = load_wsmi_dataset(
        args.patient_dir, args.control_dir, args.diagnosis_csv,
        args.coords_file, k=args.k,
    )
    print(f"\nSummary: {len(graphs)} graphs across {len(set(subj))} subjects")

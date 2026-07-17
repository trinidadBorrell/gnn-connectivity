"""
Brain-state dynamics over a cluster sequence.

A "state sequence" here is the cluster id assigned to each consecutive epoch of
a single recording (subject, session, acq), ordered by `matrix_idx`. From that
sequence we derive:

- transition_matrix: row-normalized state-to-state transition probs.
- occupancy_entropy: Shannon entropy of state-occupancy fractions.
- entropy_rate: H(X_{t+1} | X_t) under the empirical Markov chain.
- weighted_entropy: occupancy weighted by per-state variance of raw wSMI
  matrices (the "how spread is each state internally" signal).

Functions take a graph list with the metadata attached by wsmi_loader (subject_id,
session_num, acq, matrix_idx, raw_matrix, diagnosis_group) plus a parallel
array of cluster labels, and return tidy DataFrames suitable for downstream
group comparisons.

Wire from `cookbook/run_analysis.py` after running `extract_latents_from_graphs`
+ a clusterer from `cluster_analysis.py`.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _shannon_entropy_bits(probs: np.ndarray) -> float:
    """Entropy in bits; matches clustering-wsmi convention (log2)."""
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def transition_matrix(labels: Sequence[int], n_states: int) -> np.ndarray:
    """Row-normalized state-to-state transition probabilities.

    Rows with zero outgoing observations are left as zeros — a sentinel that
    "we never saw state i in this sequence except possibly at the end".
    """
    out = np.zeros((n_states, n_states), dtype=np.float64)
    if len(labels) < 2:
        return out
    arr = np.asarray(labels, dtype=int)
    for src, dst in zip(arr[:-1], arr[1:]):
        if 0 <= src < n_states and 0 <= dst < n_states:
            out[src, dst] += 1.0
    row_sums = out.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 0
    out[valid] /= row_sums[valid]
    return out


def occupancy_probs(labels: Sequence[int], n_states: int) -> np.ndarray:
    arr = np.asarray(labels, dtype=int)
    counts = np.zeros(n_states, dtype=np.float64)
    for s in arr:
        if 0 <= s < n_states:
            counts[s] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def occupancy_entropy(labels: Sequence[int], n_states: int) -> float:
    return _shannon_entropy_bits(occupancy_probs(labels, n_states))


def entropy_rate(labels: Sequence[int], n_states: int) -> float:
    """H(X_{t+1} | X_t) = -sum_i pi_i * sum_j P_ij log P_ij."""
    pi = occupancy_probs(labels, n_states)
    P = transition_matrix(labels, n_states)
    total = 0.0
    for i in range(n_states):
        row = P[i]
        nz = row > 0
        if nz.any():
            total += float(-pi[i] * (row[nz] * np.log2(row[nz])).sum())
    return total


def weighted_entropy(
    labels: Sequence[int], n_states: int, per_state_variance: Sequence[float],
) -> float:
    """sum_i pi_i * var_i  — occupancy weighted by per-state matrix variance."""
    pi = occupancy_probs(labels, n_states)
    v = np.asarray(per_state_variance, dtype=np.float64)
    if v.shape[0] != n_states:
        raise ValueError(f"per_state_variance length {v.shape[0]} != n_states {n_states}")
    return float((pi * v).sum())


def _per_state_variance(
    labels: np.ndarray, raw_matrices: Sequence[np.ndarray], n_states: int,
) -> np.ndarray:
    """Mean (over edges) of per-edge variance, computed within each state."""
    out = np.zeros(n_states, dtype=np.float64)
    for s in range(n_states):
        idx = np.where(labels == s)[0]
        mats = [raw_matrices[i] for i in idx if raw_matrices[i] is not None]
        if len(mats) > 1:
            out[s] = float(np.var(np.stack(mats, axis=0), axis=0).mean())
    return out


def build_recording_sequences(
    graphs: Sequence,
    cluster_labels: np.ndarray,
) -> Dict[Tuple[str, str, str], List[int]]:
    """Group cluster labels by (subject_id, session_num, acq), ordered by matrix_idx.

    Returns {(subject, session, acq): [label_per_epoch_in_order]}.
    Relies on the metadata attached by wsmi_loader.load_wsmi_dataset.
    """
    bag: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)
    for g, c in zip(graphs, cluster_labels):
        key = (
            getattr(g, "subject_id", "UNK"),
            str(getattr(g, "session_num", "UNK")),
            str(getattr(g, "acq", "UNK")),
        )
        bag[key].append((int(getattr(g, "matrix_idx", 0)), int(c)))
    seqs: Dict[Tuple[str, str, str], List[int]] = {}
    for key, pairs in bag.items():
        pairs.sort(key=lambda t: t[0])
        seqs[key] = [c for _, c in pairs]
    return seqs


def per_recording_dynamics(
    graphs: Sequence,
    cluster_labels: np.ndarray,
    n_states: int,
    diagnosis_attr: str = "diagnosis_group",
) -> pd.DataFrame:
    """One row per (subject, session, acq).

    Columns:
      subject_id, session, acq, diagnosis_group, n_epochs, n_states_seen,
      occupancy_entropy_bits, entropy_rate_bits, weighted_entropy,
      occupancy_p_<i>, transition_<i>_<j>
    """
    arr_labels = np.asarray(cluster_labels, dtype=int)
    raw_matrices = [getattr(g, "raw_matrix", None) for g in graphs]
    global_per_state_var = _per_state_variance(arr_labels, raw_matrices, n_states)

    diag_lookup = {}
    for g in graphs:
        key = (
            getattr(g, "subject_id", "UNK"),
            str(getattr(g, "session_num", "UNK")),
            str(getattr(g, "acq", "UNK")),
        )
        diag_lookup[key] = getattr(g, diagnosis_attr, "UNK")

    sequences = build_recording_sequences(graphs, arr_labels)
    rows = []
    for key, seq in sequences.items():
        sub, ses, acq = key
        seq_arr = np.asarray(seq, dtype=int)
        pi = occupancy_probs(seq_arr, n_states)
        P = transition_matrix(seq_arr, n_states)
        H_occ = _shannon_entropy_bits(pi)
        H_rate = entropy_rate(seq_arr, n_states)
        H_w = weighted_entropy(seq_arr, n_states, global_per_state_var)
        row = {
            "subject_id": sub, "session": ses, "acq": acq,
            "diagnosis_group": diag_lookup.get(key, "UNK"),
            "n_epochs": int(seq_arr.size),
            "n_states_seen": int(len(set(seq))),
            "occupancy_entropy_bits": H_occ,
            "entropy_rate_bits": H_rate,
            "weighted_entropy": H_w,
        }
        for i in range(n_states):
            row[f"occupancy_p_{i}"] = float(pi[i])
        for i in range(n_states):
            for j in range(n_states):
                row[f"transition_{i}_{j}"] = float(P[i, j])
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["diagnosis_group", "subject_id", "session", "acq"]).reset_index(drop=True)


def per_subject_aggregate(per_recording_df: pd.DataFrame) -> pd.DataFrame:
    """Mean of numeric features per (subject_id, diagnosis_group).

    Matches the convention in clustering-wsmi/README.md:113-119 — subjects with
    multiple recordings are represented by the mean before any LOSO step.
    """
    if per_recording_df.empty:
        return per_recording_df
    num_cols = per_recording_df.select_dtypes(include="number").columns.tolist()
    grp = per_recording_df.groupby(["subject_id", "diagnosis_group"], as_index=False)
    return grp[num_cols].mean()


def group_summary(per_recording_df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of the headline dynamics columns by diagnosis_group."""
    if per_recording_df.empty:
        return per_recording_df
    cols = ["occupancy_entropy_bits", "entropy_rate_bits", "weighted_entropy", "n_epochs"]
    cols = [c for c in cols if c in per_recording_df.columns]
    g = per_recording_df.groupby("diagnosis_group")
    return g[cols].agg(["mean", "std", "count"]).reset_index()

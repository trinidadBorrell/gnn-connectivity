"""Backfill `prevalence_proportion_boxplots.png` for already-computed clustering dirs.

Reads each clustering dir's `per_subject_within_cluster_share.csv` +
`prevalence_counts.csv` and renders the new boxplot (per-subject within-cluster
share, with the epoch-pooled prevalence_proportion = group-sum starred). Does NOT
re-run the pipeline. Safe to run while the matrix is still going.

Usage:
  python gnn_connectivity/scripts/add_prevalence_boxplots.py [output_root]
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))
from cluster_analysis import (  # noqa: E402
    plot_prevalence_proportion_boxplots,
    per_diagnosis_cluster_occupancy,
    plot_state_occupancy_by_diagnosis,
)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "gnn_connectivity/output"
    share_csvs = glob.glob(
        os.path.join(root, "*", "clustering", "*", "*", "per_subject_within_cluster_share.csv")
    )
    n_ok = 0
    for share_path in sorted(share_csvs):
        out_dir = os.path.dirname(share_path)
        counts_path = os.path.join(out_dir, "prevalence_counts.csv")
        if not os.path.exists(counts_path):
            continue
        try:
            share_df = pd.read_csv(share_path)
            counts = pd.read_csv(counts_path, index_col=0)
            prevalence_prop = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
            clusters = sorted(share_df["cluster"].unique().tolist())
            name = "/".join(out_dir.split(os.sep)[-3:])
            plot_prevalence_proportion_boxplots(
                share_df, prevalence_prop, clusters,
                os.path.join(out_dir, "prevalence_proportion_boxplots.png"),
                title=f"{name}: prevalence proportions (★) with per-subject spread",
            )
            # MATRIX-LEVEL occupancy P(cluster|diagnosis) from assignments.csv.
            assign_path = os.path.join(out_dir, "assignments.csv")
            if os.path.exists(assign_path):
                a = pd.read_csv(assign_path)
                occ = per_diagnosis_cluster_occupancy(
                    a["cluster"].values, a["diagnosis_group"].tolist())
                occ.to_csv(os.path.join(out_dir, "state_occupancy_by_diagnosis.csv"))
                plot_state_occupancy_by_diagnosis(
                    occ, os.path.join(out_dir, "state_occupancy_by_diagnosis.png"),
                    title=f"{name}: state occupancy P(cluster|diagnosis)")
            n_ok += 1
        except Exception as e:
            print(f"  SKIP {out_dir}: {e}")
    print(f"Backfilled prevalence-boxplot + occupancy for {n_ok} clustering dirs.")


if __name__ == "__main__":
    main()

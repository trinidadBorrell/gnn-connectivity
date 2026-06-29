"""Backfill `prevalence_proportion_boxplots.png` for already-computed clustering dirs.

Reads each clustering dir's `per_subject_cluster_proportions.csv` and re-renders
the boxplot in the current style: per-subject cluster proportions (each in [0, 1])
with the group MEAN marked as a ★, so the per-subject points spread around the
marker. Does NOT re-run the pipeline — cheap, CSV-only. Safe to run while the
matrix is still going (it overwrites the PNG in place).

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
    prop_csvs = glob.glob(
        os.path.join(root, "*", "clustering", "*", "*", "per_subject_cluster_proportions.csv")
    )
    n_ok = 0
    for prop_path in sorted(prop_csvs):
        out_dir = os.path.dirname(prop_path)
        try:
            prop_df = pd.read_csv(prop_path)
            clusters = sorted(prop_df["cluster"].unique().tolist())
            name = "/".join(out_dir.split(os.sep)[-3:])
            plot_prevalence_proportion_boxplots(
                prop_df, clusters,
                os.path.join(out_dir, "prevalence_proportion_boxplots.png"),
                title=f"{name}: per-subject cluster proportions (★ = group mean) by diagnosis",
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

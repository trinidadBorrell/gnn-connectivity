#!/usr/bin/env python3
"""Pick a balanced held-out subject pool that the contrastive encoder
will never see (no epochs, no labels). Saved once and reused.

Output: data/holdout_subjects.json with:
    {"holdout": [sub_id, ...], "train_pool": [sub_id, ...],
     "config": {"frac": 0.2, "random_state": 42}}
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import EEGtoGraph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/wsmi_res')
    ap.add_argument('--labels_csv',
                    default='data/wsmi_res/256_electrodes-20260526T091011Z-3-001/256_electrodes/wsmi_res_DOC/patient_labels.csv')
    ap.add_argument('--out', default='data/holdout_subjects.json')
    ap.add_argument('--frac', type=float, default=0.20,
                    help='Fraction of each diagnosis class to hold out.')
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    sessions = EEGtoGraph.enumerate_matrix_sessions(args.data_dir)
    labels = pd.read_csv(args.labels_csv, dtype=str)
    labels['session_z'] = labels['session'].str.zfill(2)
    lookup = {(r['subject'], r['session_z']): r['diagnostic_crs_final']
              for _, r in labels.iterrows()}

    # Subject → diagnosis (take modal across sessions, controls = "control")
    subj_dx = {}
    for sid, snum, source in sessions:
        if source.get('cohort') == 'control':
            subj_dx[sid] = 'control'
            continue
        dx = lookup.get((sid, str(snum).zfill(2)))
        if dx and dx in ('UWS', 'MCS-', 'MCS+', 'EMCS', 'COMA'):
            subj_dx.setdefault(sid, dx)  # first session's dx, simple
    print(f"  {len(subj_dx)} usable subjects "
          f"(after dropping unknown diagnoses)")

    rng = np.random.default_rng(args.random_state)
    by_dx = defaultdict(list)
    for sid, dx in subj_dx.items():
        by_dx[dx].append(sid)

    holdout = []
    for dx, subjects in by_dx.items():
        n_hold = max(1, int(round(len(subjects) * args.frac)))
        subjects = sorted(subjects)  # determinism
        rng_sample = rng.choice(subjects, size=n_hold, replace=False)
        holdout.extend(rng_sample.tolist())
        print(f"    {dx:>7s}: {n_hold} held out of {len(subjects)}")

    holdout = sorted(holdout)
    train_pool = sorted(set(subj_dx) - set(holdout))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'holdout': holdout,
            'train_pool': train_pool,
            'config': {'frac': args.frac, 'random_state': args.random_state},
            'n_subjects_total': len(subj_dx),
            'n_holdout': len(holdout),
            'n_train_pool': len(train_pool),
        }, f, indent=2)
    print(f"\n  saved to {args.out}")
    print(f"  held out: {len(holdout)} subjects ({len(holdout)/len(subj_dx):.0%})")
    print(f"  train pool: {len(train_pool)} subjects")


if __name__ == '__main__':
    main()

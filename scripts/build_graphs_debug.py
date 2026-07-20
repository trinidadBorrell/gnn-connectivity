#!/usr/bin/env python3
"""Instrumented, CPU-ONLY builder for the wSMI graphs.pt cache.

Purpose: (1) find EXACTLY where the eager load hangs (heavy flushed logging at
every step, with timestamps), and (2) build the cache off the GPU — graph
construction is pure CPU, so there is no reason to hold a GPU (and importing
torch on a GPU node triggers CUDA init, a prime suspect for the silent hang).

CUDA is disabled up-front so `import torch` cannot block on CUDA init. Run on a
CPU partition (no --gres). Every print is flushed so `tail -f` shows live progress.
"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # <- no CUDA init; isolates the GPU-hang hypothesis
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# matplotlib pyplot (imported by preprocessing.py -> mne) builds a font cache in
# $HOME/.cache on first run -> a classic silent hang on NFS/headless nodes.
# Force headless backend + node-local cache dir BEFORE any matplotlib import.
os.environ["MPLBACKEND"] = "Agg"
_u = os.environ.get('USER', 'x')
os.environ["MPLCONFIGDIR"] = f"/tmp/mplcfg_{_u}"
# fontconfig caches to $XDG_CACHE_HOME (default ~/.cache on NFS $HOME) — that is
# what actually hangs during matplotlib's font scan. Redirect it to node-local /tmp.
os.environ["XDG_CACHE_HOME"] = f"/tmp/xdgcache_{_u}"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log(f"START  CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}  "
    f"MPLCONFIGDIR={os.environ['MPLCONFIGDIR']}  host={os.uname().nodename}")
t = time.time(); import numpy as np;            log(f"import numpy            {time.time()-t:5.1f}s")
t = time.time(); import pandas as pd;           log(f"import pandas           {time.time()-t:5.1f}s")
t = time.time(); import torch;                  log(f"import torch            {time.time()-t:5.1f}s  cuda_avail={torch.cuda.is_available()}")
t = time.time(); import torch_geometric;        log(f"import torch_geometric  {time.time()-t:5.1f}s")
t = time.time(); import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; log(f"import matplotlib.pyplot {time.time()-t:5.1f}s")
t = time.time(); import mne;                     log(f"import mne              {time.time()-t:5.1f}s")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
t = time.time(); from wsmi_loader import load_wsmi_dataset_npz; log(f"import wsmi_loader      {time.time()-t:5.1f}s")
from data_loaders import split_by_subject_stratified

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default='data/wsmi_res')
ap.add_argument('--coords_file', default='data_scalp/GSN-HydroCel-257.txt')
ap.add_argument('--labels_csv', required=True)
ap.add_argument('--run_name', required=True)
ap.add_argument('--output_root', default='output')
ap.add_argument('--max_epochs_per_recording', type=int, default=300)
ap.add_argument('--granularity', default='fine')
ap.add_argument('--k', type=int, default=6)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--no_save', action='store_true', help='diagnose only, do not save graphs.pt')
args = ap.parse_args()

log(f"CALL load_wsmi_dataset_npz(max_epochs={args.max_epochs_per_recording}, gran={args.granularity})")
t = time.time()
graphs, subjects, dgroups = load_wsmi_dataset_npz(
    data_dir=args.data_dir, control_dir=args.data_dir, include_controls=True,
    diagnosis_csv=args.labels_csv, coords_file=args.coords_file, k=args.k,
    granularity=args.granularity,
    max_epochs_per_recording=args.max_epochs_per_recording,
    seed=args.seed, verbose=True)
log(f"LOAD DONE  {len(graphs)} graphs  {len(set(subjects))} subjects  {time.time()-t:.1f}s")

if args.no_save:
    log("DONE (no_save)"); sys.exit(0)

log("split_by_subject_stratified ...")
out_root = os.path.join(args.output_root, args.run_name)
splits_dir = os.path.join(out_root, 'splits'); os.makedirs(splits_dir, exist_ok=True)
train_g, val_g, test_g, subject_split = split_by_subject_stratified(
    graphs=graphs, subject_ids=subjects, diagnosis_groups=dgroups,
    test_frac=0.15, val_frac=0.15, random_state=args.seed, persist_dir=splits_dir)
log(f"split  train={len(train_g)} val={len(val_g)} test={len(test_g)}")

log("min-max normalize (train stats) ...")
x_min = min(float(g.x.min()) for g in train_g)
x_max = max(float(g.x.max()) for g in train_g)
x_range = (x_max - x_min) or 1.0
for split in (train_g, val_g, test_g):
    for g in split:
        g.x = 2.0 * (g.x - x_min) / x_range - 1.0
norm = {"x_min": x_min, "x_max": x_max, "x_range": x_range}

log("torch.save graphs.pt ...")
t = time.time()
torch.save({"train": train_g, "val": val_g, "test": test_g,
            "subject_split": subject_split, "normalization": norm},
           os.path.join(splits_dir, 'graphs.pt'))
log(f"SAVED graphs.pt  {time.time()-t:.1f}s  -> {splits_dir}/graphs.pt")
log("ALL DONE")

"""
Convert junifer wSMI .pkl outputs to .npz for the clustering-wsmi pipeline.

Junifer writes one .pkl per (subject, session, acq) under
  <root>/<dataset>/sub-XXX/ses-YY/eeg/sub-XXX_ses-YY_acq-ZZ_desc-wsmi_connectivity.pkl
Each pickle holds {marker_key: {"data": np.ndarray of shape (1, n_epochs, 64, 64), ...}}.

clustering-wsmi expects .npz files with a "data" array of shape (n_epochs, 64, 64),
each matrix symmetric with zero diagonal.

Usage:
  python scripts/convert_wsmi_pkl_to_npz.py \\
      --root /path/to/data/markers/wsmi_theta \\
      --datasets control_bids_biosemi64_dur-16s nice_epochs_sfreq-100Hz_recombine-biosemi64_dur-16s \\
      --out-root /path/to/data/markers/wsmi_theta_npz

Helpers below mirror the originals in src/wsmi_loader.py; inlined here so this
script has no torch / torch_geometric dependency.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np


FNAME_RE = re.compile(
    r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_acq-(?P<acq>[^_]+)_desc-wsmi_connectivity\.pkl$"
)


def _load_pkl_data(pkl_path: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if len(d) != 1:
        raise ValueError(f"Expected one marker key in {pkl_path}, found {list(d.keys())}")
    inner = next(iter(d.values()))
    return inner["data"]


def _symmetrize_and_clean(mat: np.ndarray) -> np.ndarray:
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 0.0)
    return mat.astype(np.float32)


def _iter_wsmi_files(root: str) -> Iterator[Tuple[str, str, str, str]]:
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


def convert_dataset(dataset_dir: Path, out_dataset_dir: Path, overwrite: bool) -> tuple[int, int]:
    n_done = 0
    n_skip = 0
    for sub_id, ses, acq, pkl_path in _iter_wsmi_files(str(dataset_dir)):
        out_eeg = out_dataset_dir / f"sub-{sub_id}" / f"ses-{ses}" / "eeg"
        out_eeg.mkdir(parents=True, exist_ok=True)
        out_path = out_eeg / f"sub-{sub_id}_ses-{ses}_acq-{acq}_desc-wsmi_connectivity.npz"

        if out_path.exists() and not overwrite:
            n_skip += 1
            continue

        try:
            arr = _load_pkl_data(pkl_path)
        except Exception as e:
            print(f"  SKIP {pkl_path}: {e}")
            n_skip += 1
            continue

        if arr.ndim != 4 or arr.shape[2] != 64 or arr.shape[3] != 64:
            print(f"  SKIP {pkl_path}: unexpected shape {arr.shape}")
            n_skip += 1
            continue

        epoch_mats = arr[0]
        cleaned = np.stack(
            [_symmetrize_and_clean(epoch_mats[i]) for i in range(epoch_mats.shape[0])],
            axis=0,
        ).astype(np.float32)

        np.savez_compressed(out_path, data=cleaned)
        n_done += 1

    return n_done, n_skip


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", required=True, type=Path,
                        help="Parent directory containing dataset folders, e.g. data/markers/wsmi_theta")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset subdirectory names to convert")
    parser.add_argument("--out-root", required=True, type=Path,
                        help="Output root; each dataset becomes <out_root>/<dataset>/...")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing .npz files")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"--root does not exist or is not a directory: {args.root}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    grand_done = 0
    grand_skip = 0
    for ds in args.datasets:
        ds_in = args.root / ds
        if not ds_in.is_dir():
            print(f"WARN dataset directory missing, skipping: {ds_in}")
            continue
        ds_out = args.out_root / ds
        print(f"\n=== {ds} ===")
        print(f"  in : {ds_in}")
        print(f"  out: {ds_out}")
        n_done, n_skip = convert_dataset(ds_in, ds_out, overwrite=args.overwrite)
        print(f"  wrote {n_done} npz, skipped {n_skip}")
        grand_done += n_done
        grand_skip += n_skip

    print(f"\nTotal: wrote {grand_done} npz, skipped {grand_skip}")


if __name__ == "__main__":
    main()

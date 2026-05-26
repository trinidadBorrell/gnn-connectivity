"""
Build a clustering-wsmi manifest TSV from junifer wSMI outputs.

Walks one control dataset directory and one patient dataset directory,
joins each patient (subject, session) with metadata_patient_labels.csv to
get the `diagnostic_crs_final` label, and emits a TSV with the columns
clustering-wsmi expects:
    sample_id, subject, session, path, diagnosis_canonical

Diagnosis mapping (clustering-wsmi uses the 3-class taxonomy {CONTROL, UWS, MCS}):
    numeric subject -> CONTROL
    UWS             -> UWS
    MCS- / MCS / MCS+ -> MCS
    EMCS / COMA     -> EMCS / COMA  (kept verbatim; dropped by default
                                     so the 3-class pipeline works unchanged)

Usage:
  python scripts/build_wsmi_manifest.py \\
      --root /path/to/data/markers/wsmi_theta \\
      --control-dataset control_bids_biosemi64_dur-16s \\
      --patient-dataset nice_epochs_sfreq-100Hz_recombine-biosemi64_dur-16s \\
      --metadata /path/to/metadata/DoC_metadata/metadata_patient_labels.csv \\
      --out /path/to/data/markers/wsmi_theta_npz/manifest_dur-16s.tsv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


FNAME_RE = re.compile(
    r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_acq-(?P<acq>[^_]+)_desc-wsmi_connectivity\.(?P<ext>pkl|npz)$"
)

DIAG_TO_CANONICAL = {
    "UWS": "UWS",
    "VS": "UWS",
    "MCS-": "MCS",
    "MCS": "MCS",
    "MCS+": "MCS",
    "EMCS": "EMCS",
    "COMA": "COMA",
}

CANONICAL_3CLASS = {"CONTROL", "UWS", "MCS"}


def load_diagnosis_lookup(csv_path: Path) -> Dict[Tuple[str, str], str]:
    """{(subject, session_padded_str): diagnostic_crs_final} from the metadata CSV."""
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


def iter_recordings(dataset_dir: Path, prefer_ext: str):
    """Yield (sub_id, ses, acq, path) tuples from a BIDS-style folder.

    prefer_ext is "pkl" or "npz". We pick the file matching the preferred
    extension if both exist for a given (sub, ses, acq).
    """
    for sub_folder in sorted(os.listdir(dataset_dir)):
        if not sub_folder.startswith("sub-"):
            continue
        sub_id = sub_folder[4:]
        sub_path = dataset_dir / sub_folder
        if not sub_path.is_dir():
            continue
        for ses_folder in sorted(os.listdir(sub_path)):
            if not ses_folder.startswith("ses-"):
                continue
            ses = ses_folder[4:]
            eeg_dir = sub_path / ses_folder / "eeg"
            if not eeg_dir.is_dir():
                continue
            # group files by (acq, ext)
            by_acq: Dict[str, Dict[str, Path]] = {}
            for fname in os.listdir(eeg_dir):
                m = FNAME_RE.match(fname)
                if not m:
                    continue
                by_acq.setdefault(m.group("acq"), {})[m.group("ext")] = eeg_dir / fname
            for acq in sorted(by_acq):
                exts = by_acq[acq]
                path = exts.get(prefer_ext) or exts.get("npz") or exts.get("pkl")
                if path is None:
                    continue
                yield sub_id, ses, acq, path


def diagnosis_canonical(
    sub_id: str,
    session: str,
    lookup: Dict[Tuple[str, str], str],
    group: str,
) -> str:
    if group == "control":
        return "CONTROL"
    diag = lookup.get((sub_id, session))
    if diag is None:
        for (s, _), v in lookup.items():
            if s == sub_id:
                diag = v
                break
    if diag is None:
        return "UNK"
    return DIAG_TO_CANONICAL.get(diag, diag)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", required=True, type=Path,
                        help="Parent directory containing dataset folders")
    parser.add_argument("--control-dataset", required=True,
                        help="Control dataset subdirectory name")
    parser.add_argument("--patient-dataset", required=True,
                        help="Patient dataset subdirectory name")
    parser.add_argument("--metadata", required=True, type=Path,
                        help="metadata_patient_labels.csv")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output manifest TSV path")
    parser.add_argument("--prefer-ext", choices=["pkl", "npz"], default="pkl",
                        help="Preferred file extension when both are present (default: pkl)")
    parser.add_argument("--drop-non-canonical", action="store_true", default=True,
                        help="Drop EMCS/COMA/UNK rows so the 3-class pipeline works (default: True)")
    parser.add_argument("--keep-all", dest="drop_non_canonical", action="store_false",
                        help="Keep all rows including EMCS/COMA/UNK")
    args = parser.parse_args()

    lookup = load_diagnosis_lookup(args.metadata)
    print(f"Loaded {len(lookup)} (subject, session) -> diagnosis entries from {args.metadata}")

    rows = []
    counts_per_group_diag: Counter = Counter()

    for ds_name, group in (
        (args.control_dataset, "control"),
        (args.patient_dataset, "patient"),
    ):
        ds_dir = args.root / ds_name
        if not ds_dir.is_dir():
            print(f"WARN missing dataset directory: {ds_dir}")
            continue
        for sub_id, ses, acq, path in iter_recordings(ds_dir, args.prefer_ext):
            diag = diagnosis_canonical(sub_id, ses, lookup, group=group)
            sample_id = f"sub-{sub_id}_ses-{ses}_acq-{acq}"
            rows.append({
                "sample_id": sample_id,
                "subject": sub_id,
                "session": ses,
                "path": str(path.resolve()),
                "diagnosis_canonical": diag,
            })
            counts_per_group_diag[(group, diag)] += 1

    print("\nRow counts by (group, diagnosis_canonical):")
    for (group, diag), n in sorted(counts_per_group_diag.items()):
        print(f"  {group:8s} {diag:8s}  {n}")

    if args.drop_non_canonical:
        before = len(rows)
        rows = [r for r in rows if r["diagnosis_canonical"] in CANONICAL_3CLASS]
        print(f"\nDropped {before - len(rows)} non-canonical rows; kept {len(rows)} for 3-class pipeline.")
    else:
        print(f"\nKeeping all {len(rows)} rows (including EMCS/COMA/UNK).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "subject", "session", "path", "diagnosis_canonical"]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

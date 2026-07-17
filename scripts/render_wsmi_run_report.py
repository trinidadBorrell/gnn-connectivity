"""
Render an HTML + TSV report for a single wsmi_run_* output directory.

Combines, for each model present (gae, vgae):
  - run metadata (args.json)
  - pretraining hyperparameter sweep (tuning/<model>/trials.csv + best_config.json)
  - final test metrics (final_test/<model>/test_report.json)
  - clustering results — one row per (model, clusterer); columns include
    silhouette, ARI/AMI/V-measure/purity vs CONTROL/UWS/MCS diagnosis,
    shannon entropy, variance-weighted entropy
  - brain-state dynamics by diagnosis class (state occupancy / entropy rate
    / weighted entropy aggregated over diagnosis_group)
  - LOSO decoder metrics (best row by macro-AUC)
  - VGAE-only diagnostics: posterior collapse fraction, KL per dim summary

Usage:
  .venv/bin/python scripts/render_wsmi_run_report.py \\
      --run-dir gnn_connectivity/output/wsmi_run_main \\
      --output-html gnn_connectivity/output/wsmi_run_main/report.html \\
      --output-tsv gnn_connectivity/output/wsmi_run_main/report.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #222; }
h1 { margin-bottom: 4px; }
h2 { margin-top: 28px; }
h3 { margin-top: 18px; color: #444; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) { text-align: left; }
th { background: #f3f4f6; }
.note { color: #555; max-width: 900px; line-height: 1.45; }
.warn { color: #b00; background: #ffeaea; padding: 8px 12px; border-left: 4px solid #b00; }
pre { background: #f7f7f9; padding: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto; }
"""

CANONICAL_3 = ("CONTROL", "UWS", "MCS")


# ---------- helpers ----------


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _safe_purity(cluster_labels: np.ndarray, diag_labels: np.ndarray) -> float:
    if cluster_labels.size == 0:
        return float("nan")
    purities = []
    for c in np.unique(cluster_labels):
        idx = cluster_labels == c
        if idx.sum() == 0:
            continue
        from collections import Counter
        ctr = Counter(diag_labels[idx])
        purities.append(max(ctr.values()) / idx.sum())
    return float(np.mean(purities)) if purities else float("nan")


def _diag_metrics(assignments_df: pd.DataFrame) -> Dict[str, float]:
    """ARI/AMI/V-measure/purity from an assignments.csv frame."""
    diag = assignments_df["diagnosis_group"].astype(str).to_numpy()
    cluster = assignments_df["cluster"].to_numpy()
    keep = (diag != "UNK") & (diag != "DROP") & (cluster != -1)
    if keep.sum() == 0 or len(set(diag[keep])) < 2 or len(set(cluster[keep])) < 2:
        return {k: float("nan") for k in ("ari", "ami", "homogeneity", "completeness", "v_measure", "purity")}
    d = diag[keep]
    c = cluster[keep]
    homo, comp, vmes = homogeneity_completeness_v_measure(d, c)
    return {
        "ari": float(adjusted_rand_score(d, c)),
        "ami": float(adjusted_mutual_info_score(d, c)),
        "homogeneity": float(homo),
        "completeness": float(comp),
        "v_measure": float(vmes),
        "purity": _safe_purity(c, d),
    }


def _chosen_k_metric(metric_curve: Dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    """Pick the per-clusterer selection metric (silhouette / BIC / validity)
    at the chosen k. Returns (metric_name, value) so the renderer can label it.
    """
    if metric_curve is None:
        return None, None
    ks = metric_curve.get("k")
    best = metric_curve.get("best_k") or metric_curve.get("best_min_cluster_size")
    for name in ("silhouette", "bic", "validity"):
        vals = metric_curve.get(name)
        if vals is None:
            continue
        if ks and best in ks:
            return name, float(vals[ks.index(best)])
        if best is not None:
            return name, float(vals[0])
        return name, float(max(vals))
    return None, None


def _read_state_dynamics(metrics_dir: Path) -> Optional[pd.DataFrame]:
    p = metrics_dir / "state_dynamics_per_recording.tsv"
    if p.exists():
        return pd.read_csv(p, sep="\t")
    return None


# ---------- section builders ----------


def build_run_metadata(run_dir: Path) -> str:
    args = _read_json(run_dir / "args.json") or {}
    rows = [{"key": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else str(v)} for k, v in args.items()]
    df = pd.DataFrame(rows)
    return df.to_html(index=False)


def build_pretraining(run_dir: Path, model: str) -> str:
    tuning_dir = run_dir / "tuning" / model
    parts = [f"<h3>{model.upper()}</h3>"]

    best_cfg = _read_json(tuning_dir / "best_config.json")
    if best_cfg:
        parts.append("<h4>Final chosen config</h4>")
        parts.append(f"<pre>{json.dumps(best_cfg, indent=2)}</pre>")
    else:
        parts.append('<p class="warn">No best_config.json found</p>')

    trials_path = tuning_dir / "trials.csv"
    if trials_path.exists():
        trials = pd.read_csv(trials_path)
        # Each trial may have multiple rows (one per epoch). Take the min val_mse per trial_id.
        if "val_mse" in trials.columns and "trial_id" in trials.columns:
            best_per_trial = (trials.sort_values("val_mse")
                              .groupby("trial_id", as_index=False).first())
        else:
            best_per_trial = trials.copy()
        keep_cols = [c for c in [
            "trial_id", "val_mse", "train_loss", "recon_loss", "kl_loss",
            "config/latent_dim", "config/hidden_dims", "config/lr",
            "config/dropout", "config/batch_size", "config/weight_decay",
            "training_iteration",
        ] if c in best_per_trial.columns]
        top = best_per_trial[keep_cols].sort_values(
            "val_mse" if "val_mse" in keep_cols else keep_cols[0]
        ).head(10)
        parts.append("<h4>Top 10 trials by best val_mse</h4>")
        parts.append(top.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        parts.append('<p class="warn">No trials.csv found</p>')

    return "\n".join(parts)


def build_final_test(run_dir: Path, model: str) -> Optional[Dict[str, Any]]:
    return _read_json(run_dir / "final_test" / model / "test_report.json")


def build_clustering_results(run_dir: Path, model: str) -> List[Dict[str, Any]]:
    """One row per clusterer present under clustering/<model>/<clusterer>/.

    Computes silhouette (from metric_curve.json), shannon entropy + variance-
    weighted entropy (from entropy.json), and ARI/AMI/V-measure/purity vs the
    canonical 3-class diagnosis (computed at render time from assignments.csv).
    """
    clu_root = run_dir / "clustering" / model
    if not clu_root.is_dir():
        return []
    rows = []
    for sub in sorted(clu_root.iterdir()):
        if not sub.is_dir():
            continue
        clusterer = sub.name
        metric_curve = _read_json(sub / "metric_curve.json")
        entropy = _read_json(sub / "entropy.json") or {}
        assignments_path = sub / "assignments.csv"
        metric_name, metric_val = _chosen_k_metric(metric_curve)

        row: Dict[str, Any] = {
            "model": model,
            "clusterer": clusterer,
            "n_states": (metric_curve or {}).get("best_k")
                        or (metric_curve or {}).get("best_min_cluster_size")
                        or (len(entropy.get("cluster_ids", [])) if entropy else None),
            "selection_metric": metric_name,
            "selection_metric_value": metric_val,
            "shannon_entropy": entropy.get("shannon"),
            "variance_weighted_entropy": entropy.get("variance_weighted_entropy"),
        }
        if assignments_path.exists():
            asn = pd.read_csv(assignments_path)
            row.update(_diag_metrics(asn))
        else:
            row.update({k: float("nan") for k in ("ari", "ami", "homogeneity", "completeness", "v_measure", "purity")})
        rows.append(row)
    return rows


def build_dynamics_by_class(metrics_dir: Path) -> Optional[pd.DataFrame]:
    df = _read_state_dynamics(metrics_dir)
    if df is None or "diagnosis_group" not in df.columns:
        return None
    metrics = [c for c in ("occupancy_entropy_bits", "entropy_rate_bits", "weighted_entropy") if c in df.columns]
    if not metrics:
        return None
    g = df.groupby("diagnosis_group")
    out = pd.concat([
        g[m].agg(["mean", "std", "count"]).rename(columns={"mean": f"{m}_mean", "std": f"{m}_std", "count": f"{m}_n"})
        for m in metrics
    ], axis=1)
    out = out.reset_index()
    # Sort canonical order
    out["__order"] = out["diagnosis_group"].map({c: i for i, c in enumerate(CANONICAL_3)}).fillna(len(CANONICAL_3))
    out = out.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    return out


def build_decoder_table(metrics_dir: Path) -> Optional[pd.DataFrame]:
    p = metrics_dir / "decoder_metrics.tsv"
    if not p.exists():
        return None
    return pd.read_csv(p, sep="\t")


def build_vgae_diagnostics(metrics_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    coll = _read_json(metrics_dir / "posterior_collapse.json")
    if coll:
        out["posterior_collapse"] = coll
    kl_path = metrics_dir / "kl_per_dim.npy"
    if kl_path.exists():
        kl = np.load(kl_path)
        out["kl_per_dim_summary"] = {
            "n_dims": int(kl.size),
            "mean": float(kl.mean()),
            "min": float(kl.min()),
            "max": float(kl.max()),
            "below_1e-2": int((kl < 1e-2).sum()),
        }
    return out


def build_recon_summary(metrics_dir: Path) -> Optional[Dict[str, Any]]:
    p = metrics_dir / "per_edge_mse.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    flat = arr.reshape(-1)
    worst = int(np.argmax(flat))
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "worst_node_feature": [worst // arr.shape[1], worst % arr.shape[1]] if arr.ndim == 2 else None,
    }


# ---------- top-level ----------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"run-dir not found: {run_dir}")

    models = [m for m in ("gae", "vgae")
              if (run_dir / "clustering" / m).exists()
              or (run_dir / "models" / m).exists()
              or (run_dir / "tuning" / m).exists()]
    if not models:
        raise SystemExit(f"No gae/vgae artifacts found under {run_dir}")

    # ---- per-model sections ----
    pretraining_html_parts = []
    test_rows = []
    clustering_rows: List[Dict[str, Any]] = []
    dynamics_by_class_parts = []
    decoder_parts = []
    vgae_parts = []
    recon_parts = []

    for model in models:
        pretraining_html_parts.append(build_pretraining(run_dir, model))

        test = build_final_test(run_dir, model)
        if test is not None:
            row = {"model": model, "test_mse": test.get("test_mse"), "n_test_graphs": test.get("n_test_graphs")}
            cfg = test.get("config", {}) or {}
            for k in ("latent_dim", "hidden_dims", "lr", "dropout", "batch_size", "weight_decay", "kl_warmup_epochs", "beta"):
                if k in cfg:
                    v = cfg[k]
                    row[f"final_{k}"] = json.dumps(v) if isinstance(v, list) else v
            test_rows.append(row)

        clustering_rows.extend(build_clustering_results(run_dir, model))

        metrics_dir = run_dir / "metrics" / model
        dyn_class = build_dynamics_by_class(metrics_dir) if metrics_dir.exists() else None
        if dyn_class is not None and not dyn_class.empty:
            dynamics_by_class_parts.append(
                f"<h3>{model.upper()}</h3>" + dyn_class.to_html(index=False, float_format=lambda x: f"{x:.4f}")
            )

        dec = build_decoder_table(metrics_dir) if metrics_dir.exists() else None
        if dec is not None and not dec.empty:
            decoder_parts.append(
                f"<h3>{model.upper()}</h3>" + dec.to_html(index=False, float_format=lambda x: f"{x:.4f}")
            )

        recon = build_recon_summary(metrics_dir) if metrics_dir.exists() else None
        if recon is not None:
            recon_parts.append(f"<h3>{model.upper()}</h3><pre>{json.dumps(recon, indent=2)}</pre>")

        if model == "vgae" and metrics_dir.exists():
            vg = build_vgae_diagnostics(metrics_dir)
            if vg:
                vgae_parts.append(f"<pre>{json.dumps(vg, indent=2)}</pre>")

    # ---- flat tables for TSV + HTML ----
    test_df = pd.DataFrame(test_rows)
    clustering_df = pd.DataFrame(clustering_rows)

    # ---- assemble HTML ----
    chunks: List[str] = []
    chunks.append('<h2>Run metadata</h2>' + build_run_metadata(run_dir))

    chunks.append('<h2>Pretraining hyperparameter sweep</h2>')
    chunks.extend(pretraining_html_parts)

    chunks.append('<h2>Final test metrics</h2>')
    if not test_df.empty:
        chunks.append(test_df.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        chunks.append('<p class="warn">No final_test/*/test_report.json found.</p>')

    chunks.append('<h2>Clustering results — by (model, clusterer)</h2>')
    if not clustering_df.empty:
        chunks.append(clustering_df.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        chunks.append('<p class="warn">No clustering/*/<clusterer>/ artifacts found.</p>')

    chunks.append('<h2>Brain-state dynamics by diagnosis class</h2>')
    if dynamics_by_class_parts:
        chunks.extend(dynamics_by_class_parts)
    else:
        chunks.append('<p class="warn">No state_dynamics_per_recording.tsv found.</p>')

    chunks.append('<h2>LOSO decoder metrics</h2>')
    if decoder_parts:
        chunks.extend(decoder_parts)
    else:
        chunks.append('<p class="warn">No decoder_metrics.tsv found.</p>')

    chunks.append('<h2>Reconstruction diagnostics</h2>')
    if recon_parts:
        chunks.extend(recon_parts)
    else:
        chunks.append('<p class="warn">No per_edge_mse.npy found.</p>')

    if vgae_parts:
        chunks.append('<h2>VGAE-only diagnostics</h2>')
        chunks.extend(vgae_parts)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>wSMI Run Report — {run_dir.name}</title>
<style>{CSS}</style></head><body>
<h1>wSMI Run Report — {run_dir.name}</h1>
<p class="note">Models present: {", ".join(m.upper() for m in models)}.
Diagnoses constrained to CONTROL / UWS / MCS (EMCS / COMA / unlabeled dropped at load time).</p>
{"".join(chunks)}
</body></html>
"""
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html)

    # ---- flat TSV: one row per (model, clusterer) ----
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    if not clustering_df.empty:
        clustering_df.to_csv(args.output_tsv, sep="\t", index=False)
    else:
        args.output_tsv.write_text("model\tclusterer\tnote\n")

    print(f"Wrote HTML: {args.output_html}")
    print(f"Wrote TSV : {args.output_tsv}")


if __name__ == "__main__":
    main()

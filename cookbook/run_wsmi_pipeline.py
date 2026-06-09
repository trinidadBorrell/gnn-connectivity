"""
wSMI end-to-end pipeline.

Stages (each runnable independently via --stage):
  load     -> build per-epoch graphs from both wSMI folders, save splits.
  tune     -> random-search hyperparameters on train (early-stop on val) for GAE and VGAE.
  train    -> train final model with best config; report VAL MSE only.
  test     -> evaluate frozen model on held-out TEST (one-shot).
  latents  -> extract graph-level latents + raw matrices for train/val/test.
  cluster  -> run K-Means / GMM / HDBSCAN with k-sweep, save prevalence,
              per-cluster mean matrices, entropy.
  all      -> load -> tune -> train -> test -> latents -> cluster for both models.

Outputs land under <output_root>/wsmi_run_<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(THIS_DIR), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import GAE, GAEVAE, VGAE, GNNEncoder, kl_divergence  # noqa: E402
from train import (  # noqa: E402
    _is_variational, _select_model_class,
    apply_normalization, compute_mse_on_graphs,
    compute_normalization_stats, run_ray_tune_wsmi, train_one_epoch,
)
from wsmi_loader import load_wsmi_dataset  # noqa: E402
from timeseries_loader import load_timeseries_dataset  # noqa: E402
from data_loaders import split_by_subject_stratified  # noqa: E402
from outlier_filter import (  # noqa: E402
    apply_outlier_model, fit_outlier_model, summarize as summarize_outliers,
    plot_outlier_diagnostics,
)
from cebra_loss import CebraPairDataset, InfoNCE, collate_pairs  # noqa: E402
from cluster_analysis import (  # noqa: E402
    extract_embeddings_encoder, extract_latents_from_graphs, run_all_clusterers,
    diagnosis_clustering_metrics, kmeans_best_k,
)
from state_dynamics import per_recording_dynamics, per_subject_aggregate, group_summary  # noqa: E402
from decoder_eval import (  # noqa: E402
    aggregate_latents_per_subject, build_subject_feature_table, loso_decoder,
)
from latent_diagnostics import (  # noqa: E402
    per_edge_reconstruction_error, per_subject_recon_mse,
    kl_per_dim, posterior_collapse_fraction, mu_logvar_stats,
)


# Default data/marker roots per --type_data. Filled into the dir flags in main()
# only when the user leaves them unset, so explicit --*_dir overrides still win.
DATA_DEFAULTS = {
    "rs": {
        "wsmi_patient": "data/markers/wsmi_theta/"
                        "nice_epochs_sfreq-100Hz_recombine-biosemi64_dur-16s",
        "wsmi_control": "data/markers/wsmi_theta/control_bids_biosemi64_dur-16_tau10",
        # task-rs .fif live in the same pic-nic root as lg; the task filter selects them.
        "ts_patient": "data/fif/pic-nic/nice_epochs_sfreq-100Hz_recombine-biosemi64",
        # Control rs time-series: biosemi64 controls (250 Hz; the loader resamples to
        # --sfreq). Omitted when --patients_only is set. Epochs are cropped to
        # [crop_tmin, crop_tmax] (-0.2..0.6 s) just like lg.
        "ts_control": "data/fif/pic-nic/control_bids_biosemi64-rs",
    },
    "lg": {
        "wsmi_patient": "data/markers/wsmi_theta_lg/"
                        "nice_epochs_sfreq-100Hz_recombine-biosemi64_lg",
        # Healthy-control lg wSMI: EGI256 controls recombined to biosemi64 and
        # resampled to 100 Hz (tau=4), so it matches the patient lg config exactly.
        # Omitted automatically when --patients_only is set.
        "wsmi_control": "data/markers/wsmi_theta_lg/"
                        "control_bids_sfreq-100Hz_recombine-biosemi64-lg",
        "ts_patient": "data/fif/pic-nic/nice_epochs_sfreq-100Hz_recombine-biosemi64",
        "ts_control": "data/fif/pic-nic/control_bids_sfreq-100Hz_recombine-biosemi64-lg",
    },
}


def _is_encoder_loss(loss: str) -> bool:
    """The cebra loss drives the encoder-only (enc_gae_fc) path."""
    return loss == "cebra"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(model_kind: str, config: Dict) -> torch.nn.Module:
    Cls = _select_model_class(model_kind)
    return Cls(
        in_channels=config["in_channels"],
        hidden_dims=config["hidden_dims"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
    )


# ---------------- stages ----------------


def _apply_outlier_filter(args, train_g, val_g, test_g, splits_dir):
    """Fit the Mahalanobis 2-sigma model on TRAIN and filter all three splits.

    Bounds come from raw (pre-normalization) train wSMI matrices and are reused
    verbatim on val/test. Writes splits/outlier_report.json.
    """
    print(f"\n[outlier] fitting diagonal-Gaussian model on TRAIN "
          f"(n_sigma={args.outlier_n_sigma}, threshold={args.outlier_threshold})")
    omodel = fit_outlier_model(
        train_g, n_sigma=args.outlier_n_sigma,
        threshold=args.outlier_threshold,
    )
    pre = {"train": list(train_g), "val": list(val_g), "test": list(test_g)}
    train_g, ktr, d2tr = apply_outlier_model(train_g, omodel)
    val_g, kva, d2va = apply_outlier_model(val_g, omodel)
    test_g, kte, d2te = apply_outlier_model(test_g, omodel)
    report = summarize_outliers(
        omodel, {"train": ktr, "val": kva, "test": kte}, pre)
    os.makedirs(splits_dir, exist_ok=True)
    # Visual sanity checks (PCA scatter of inliers/outliers, d2 histograms, drop
    # fraction by diagnosis). Must run BEFORE we free wsmi_matrix below.
    try:
        plot_outlier_diagnostics(
            omodel, pre,
            {"train": ktr, "val": kva, "test": kte},
            {"train": d2tr, "val": d2va, "test": d2te},
            os.path.join(splits_dir, "outlier_diagnostics.png"),
        )
    except Exception as e:
        print(f"  [outlier-plot] skipped (error: {e})")
    # The wSMI basis is only needed for the filter; free it to save RAM on large
    # (lg) runs. wSMI mode keeps raw_matrix (a separate attribute) intact.
    for g in (*train_g, *val_g, *test_g):
        if hasattr(g, "wsmi_matrix"):
            del g.wsmi_matrix
    with open(os.path.join(splits_dir, "outlier_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"  n_dims={report['n_dims']}  d2_max={report['d2_max']:.2f} "
          f"(conf={report['conf']:.4f})")
    for name in ("train", "val", "test"):
        s = report["per_split"][name]
        print(f"  {name}: kept {s['n_kept']}/{s['n_total']} "
              f"(dropped {s['n_dropped']})")
    return train_g, val_g, test_g


def stage_load(args, out_root: str) -> dict:
    cohort_tag = "patients-only" if args.patients_only else "patients+controls"
    subject_filter = (
        {s.strip() for s in args.subject_filter.split(",") if s.strip()}
        if args.subject_filter else None
    )
    print(f"\n=== STAGE: LOAD ({args.input_mode}, type_data={args.type_data}, "
          f"{cohort_tag}{', subjects=' + str(sorted(subject_filter)) if subject_filter else ''}) ===")
    if args.input_mode == "wsmi":
        graphs, subjects, dgroups = load_wsmi_dataset(
            patient_dir=args.patient_dir,
            control_dir=None if args.patients_only else args.control_dir,
            diagnosis_csv=args.diagnosis_csv,
            coords_file=args.coords_file,
            k=args.k,
            subject_filter=subject_filter,
            granularity=args.diagnosis_granularity,
            max_epochs_per_recording=args.max_epochs_per_recording,
            seed=args.seed,
        )
    elif args.input_mode == "timeseries":
        # Both rs and lg epochs span -0.2..0.6 s, so crop identically for both.
        # The window is a property of the DATA (task), not the model.
        crop = (args.crop_tmin, args.crop_tmax)
        # When filtering, attach the matched real wSMI (args.patient/control_dir
        # are the wSMI roots) as the Mahalanobis basis.
        wsmi_patient = args.patient_dir if args.filter_outliers else None
        wsmi_control = (None if (args.patients_only or not args.filter_outliers)
                        else args.control_dir)
        graphs, subjects, dgroups = load_timeseries_dataset(
            patient_dir=args.timeseries_patient_dir,
            control_dir=None if args.patients_only else args.timeseries_control_dir,
            diagnosis_csv=args.diagnosis_csv,
            coords_file=args.coords_file,
            k=args.k,
            target_sfreq=args.sfreq,
            window_sec=args.window_sec,
            crop=crop,
            wsmi_patient_dir=wsmi_patient,
            wsmi_control_dir=wsmi_control,
            subject_filter=subject_filter,
            task=args.type_data,  # 'lg' or 'rs' -> only load that task's .fif files
            granularity=args.diagnosis_granularity,
            max_epochs_per_recording=args.max_epochs_per_recording,
            seed=args.seed,
        )
    else:
        raise ValueError(f"unknown input_mode={args.input_mode!r}")
    splits_dir = os.path.join(out_root, "splits")
    train_g, val_g, test_g, subject_split = split_by_subject_stratified(
        graphs=graphs, subject_ids=subjects, diagnosis_groups=dgroups,
        test_frac=args.test_frac, val_frac=args.val_frac,
        random_state=args.seed, persist_dir=splits_dir,
    )

    if args.filter_outliers:
        train_g, val_g, test_g = _apply_outlier_filter(
            args, train_g, val_g, test_g, splits_dir)

    print("\n[normalize] computing stats from TRAIN only and applying in-place")
    x_min, x_max, x_range = compute_normalization_stats(train_g)
    for split in (train_g, val_g, test_g):
        apply_normalization(split, x_min, x_max, x_range, inplace=True)
    norm = {"x_min": float(x_min), "x_max": float(x_max), "x_range": float(x_range)}
    with open(os.path.join(splits_dir, "normalization.json"), "w") as f:
        json.dump(norm, f, indent=2)

    payload = {"train": train_g, "val": val_g, "test": test_g,
               "subject_split": subject_split, "normalization": norm}
    cache = os.path.join(out_root, "splits", "graphs.pt")
    torch.save(payload, cache)
    print(f"\n[load] cached graphs+split to {cache}")
    return payload


def _load_cached(out_root: str) -> dict:
    cache = os.path.join(out_root, "splits", "graphs.pt")
    print(f"[load-cached] {cache}")
    return torch.load(cache, weights_only=False)


# ---------- tuning: Ray Tune ASHA search ----------


def stage_tune(args, out_root: str, data: dict, model_kind: str) -> Dict:
    tag = model_kind
    print(f"\n=== STAGE: TUNE ({tag}) — Ray Tune ASHA ===")
    tune_dir = os.path.join(out_root, "tuning", tag)
    os.makedirs(tune_dir, exist_ok=True)

    in_channels = data["train"][0].x.shape[1]
    best_config, trials_df = run_ray_tune_wsmi(
        train_graphs=data["train"],
        val_graphs=data["val"],
        in_channels=in_channels,
        model_kind=model_kind,
        num_samples=args.num_trials,
        n_epochs=args.tune_epochs,
        grace_period=max(2, args.tune_epochs // 4),
        cpus_per_trial=args.cpus_per_trial,
        storage_path=os.path.join(tune_dir, "ray_results"),
        corr_lambda_search=(args.input_mode == "timeseries"),
        max_concurrent_trials=args.max_concurrent_trials,
    )

    # Persist trials and best config
    trials_df.to_csv(os.path.join(tune_dir, "trials.csv"), index=False)
    serializable = {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in best_config.items()}
    with open(os.path.join(tune_dir, "best_config.json"), "w") as f:
        json.dump(serializable, f, indent=2, default=float)
    print(f"\n[tune {tag}] best config saved to {tune_dir}")

    # Sweep summary plot: val_mse vs trial index, colour by latent_dim
    try:
        mse_col = "val_mse" if "val_mse" in trials_df.columns else None
        if mse_col is None:
            for c in trials_df.columns:
                if c.endswith("val_mse"):
                    mse_col = c
                    break
        if mse_col is not None:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(range(len(trials_df)), trials_df[mse_col].values, alpha=0.7)
            ax.set_xlabel("trial")
            ax.set_ylabel("val_mse (final reported)")
            ax.set_title(f"Ray Tune trials — {tag}")
            plt.tight_layout()
            plt.savefig(os.path.join(tune_dir, "sweep_plot.png"),
                        dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"  [warn] sweep plot failed: {e}")

    return serializable


def stage_train(args, out_root: str, data: dict, best: Dict, model_kind: str) -> Tuple[torch.nn.Module, Dict]:
    tag = model_kind
    variational = _is_variational(model_kind)
    print(f"\n=== STAGE: TRAIN FINAL ({tag}) ===")
    model_dir = os.path.join(out_root, "models", tag)
    os.makedirs(model_dir, exist_ok=True)

    device = select_device(prefer_gpu=not args.cpu)
    in_channels = data["train"][0].x.shape[1]
    drop = {"val_mse", "n_epochs_run", "wall_seconds", "n_epochs"}
    config = {k: v for k, v in best.items() if k not in drop}
    config["in_channels"] = in_channels
    config["model_kind"] = model_kind
    # batch_size may be missing if loaded from older runs; default
    config.setdefault("batch_size", 64)
    config.setdefault("weight_decay", 0.0)
    # Rename: tune calls it beta_kl, train_one_epoch expects 'beta'
    if "beta_kl" in config and "beta" not in config:
        config["beta"] = config.pop("beta_kl")
    config.setdefault("beta", 1.0 if variational else 0.0)
    config.setdefault("kl_warmup_epochs", 10)
    # Correlation regularizer, governed by --loss:
    #   mse      -> corr_lambda = 0
    #   mse_corr -> corr_lambda from --lambda_corr / tuning (forced > 0)
    if args.lambda_corr is not None:
        config["corr_lambda"] = args.lambda_corr
    config.setdefault("corr_lambda", 0.0)
    if args.loss == "mse":
        config["corr_lambda"] = 0.0
    elif args.loss == "mse_corr" and config["corr_lambda"] <= 0:
        config["corr_lambda"] = 1.0  # ensure the regularizer is actually active
    set_seed(args.seed)
    model = _build_model(model_kind, config).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                              weight_decay=config.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=15)
    criterion = torch.nn.MSELoss()
    loader = PyGDataLoader(data["train"], batch_size=config["batch_size"],
                           shuffle=True, num_workers=0, pin_memory=False)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since = 0
    history = {"train": [], "val": [], "kl": [], "beta": [], "corr": []}
    for ep in range(args.train_epochs):
        if variational:
            beta_max = config.get("beta", 1.0)
            warm = max(1, config.get("kl_warmup_epochs", 1))
            beta = beta_max * min(1.0, ep / warm)
        else:
            beta = 0.0
        tl, _, kl, c_loss = train_one_epoch(
            model, optim, criterion, loader, device=device,
            variational=variational, beta=beta,
            corr_lambda=config["corr_lambda"],
        )
        val = compute_mse_on_graphs(model, data["val"], device=device, variational=variational)
        history["train"].append(tl)
        history["val"].append(val)
        history["kl"].append(kl)
        history["beta"].append(beta)
        history["corr"].append(c_loss)
        sched.step(val)
        if val < best_val - 1e-6:
            best_val = val
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since = 0
        else:
            epochs_since += 1
        if ep % 10 == 0:
            print(f"  ep {ep:03d}  train={tl:.6f}  val={val:.6f}  "
                  f"best_val={best_val:.6f}  kl={kl:.4f}  beta={beta:.3f}  "
                  f"corr={c_loss:.6f}")
        if epochs_since >= args.train_patience and ep > config.get("kl_warmup_epochs", 0):
            print(f"  early stop at ep {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    torch.save({"model_state_dict": model.state_dict(), "config": config,
                "best_epoch": best_epoch, "best_val": best_val},
               os.path.join(model_dir, "model.pt"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=float)
    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=float)

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train"], label="train", linewidth=1.5)
    ax.plot(history["val"], label="val", linewidth=1.5)
    ax.axvline(best_epoch, linestyle="--", color="grey", label=f"best ep={best_epoch}")
    ax.set_title(f"{tag.upper()} reconstruction MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    if variational:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history["kl"], color="purple", label="KL")
        ax2 = ax.twinx()
        ax2.plot(history["beta"], color="orange", label="beta", linestyle="--")
        ax.set_xlabel("epoch")
        ax.set_ylabel("KL")
        ax2.set_ylabel("beta")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.title("VGAE KL and beta schedule")
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "kl_curve.png"), dpi=200, bbox_inches="tight")
        plt.close()

    _plot_recon_grid(model, data["val"], variational, config,
                     os.path.join(model_dir, "val_recon_grid.png"),
                     device=device, n_samples=5)
    return model, config


def _plot_recon_grid(model, graphs, variational, config, out_path,
                     device, n_samples=5):
    model.eval()
    if len(graphs) == 0:
        return
    n_samples = min(n_samples, len(graphs))
    idxs = np.random.choice(len(graphs), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples, 4, figsize=(18, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    latent_dim = config.get("latent_dim", 2)
    with torch.no_grad():
        for r, i in enumerate(idxs):
            g = graphs[i].to(device)
            out = model(g.x, g.edge_index)
            x_recon, z = out[0], out[1]
            orig = g.x.cpu().numpy()
            rec = x_recon.cpu().numpy()
            err = np.abs(orig - rec)
            lat = z.cpu().numpy()
            for col, (mat, ttl, cmap, lim) in enumerate([
                (orig, f"Original (sub={getattr(g, 'subject_id', '?')})", "viridis", (-1, 1)),
                (rec, "Reconstruction", "viridis", (-1, 1)),
                (err, f"|error|  MAE={err.mean():.4f}", "Reds", (None, None)),
                (None, "latent", None, None),
            ]):
                ax = axes[r, col]
                if col < 3:
                    im = ax.imshow(mat, aspect="auto", cmap=cmap,
                                   vmin=lim[0], vmax=lim[1])
                    plt.colorbar(im, ax=ax, fraction=0.046)
                else:
                    if latent_dim == 1:
                        ax.imshow(lat, aspect="auto", cmap="coolwarm")
                    elif latent_dim == 2:
                        ax.scatter(lat[:, 0], lat[:, 1], c=range(len(lat)),
                                   cmap="coolwarm", s=20)
                    else:
                        ax.imshow(lat, aspect="auto", cmap="coolwarm")
                ax.set_title(ttl, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def stage_test(args, out_root: str, model, config, data, model_kind: str) -> Dict:
    tag = model_kind
    variational = _is_variational(model_kind)
    print(f"\n=== STAGE: TEST (held-out, one-shot) ({tag}) ===")
    test_dir = os.path.join(out_root, "final_test", tag)
    os.makedirs(test_dir, exist_ok=True)
    device = next(model.parameters()).device
    test_mse = compute_mse_on_graphs(model, data["test"], device=device,
                                     variational=variational)
    report = {"test_mse": float(test_mse), "n_test_graphs": len(data["test"]),
              "config": config}
    with open(os.path.join(test_dir, "test_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    _plot_recon_grid(model, data["test"], variational, config,
                     os.path.join(test_dir, "test_recon_grid.png"),
                     device=device, n_samples=5)
    print(f"  test MSE = {test_mse:.6f}; report saved to {test_dir}")
    return report


def stage_latents_and_cluster(args, out_root: str, model, data, model_kind: str):
    tag = model_kind
    is_encoder = (model_kind == "enc_gae_fc")
    variational = _is_variational(model_kind)
    print(f"\n=== STAGE: LATENTS + CLUSTER ({tag}) ===")
    cluster_root = os.path.join(out_root, "clustering", tag)
    os.makedirs(cluster_root, exist_ok=True)

    # Combine all graphs with their split label, in deterministic order
    all_graphs, splits = [], []
    for split_name in ("train", "val", "test"):
        for g in data[split_name]:
            all_graphs.append(g)
            splits.append(split_name)

    device = next(model.parameters()).device
    if is_encoder:
        bundle = extract_embeddings_encoder(model, all_graphs, splits, device=device)
    else:
        bundle = extract_latents_from_graphs(
            model, all_graphs, splits, device=device,
            aggregate=args.graph_latent_agg)
    electrode_labels = getattr(all_graphs[0], "electrode_labels", None)
    run_all_clusterers(bundle, cluster_root, electrode_labels=electrode_labels)

    # ---- New metrics: state dynamics, LOSO decoder, latent diagnostics ----
    metrics_root = os.path.join(out_root, "metrics", tag)
    os.makedirs(metrics_root, exist_ok=True)

    # Pick a single canonical clustering (KMeans, k chosen by silhouette) to
    # drive the dynamics + decoder, so the dynamics/decoder reuse exactly the
    # latents we trained.
    print(f"  [{tag}] fitting KMeans (silhouette-best k) on latents for dynamics + decoder")
    labels, best_k, _ = kmeans_best_k(bundle.embeds, k_range=range(2, 11))
    n_states = int(best_k)

    # C.1: diagnosis-aware clustering quality
    diag_metrics = diagnosis_clustering_metrics(labels, bundle.diagnosis_groups)
    diag_metrics["clusterer"] = f"kmeans_k{n_states}"
    pd_diag = __import__("pandas").DataFrame([diag_metrics])
    pd_diag.to_csv(os.path.join(metrics_root, "diagnosis_clustering_quality.tsv"),
                   sep="\t", index=False)
    print(f"  [{tag}] diagnosis clustering quality: "
          f"ARI={diag_metrics['ari']:.3f} AMI={diag_metrics['ami']:.3f} "
          f"purity={diag_metrics['purity']:.3f}")

    # C.2: state dynamics per recording, per subject, group summary
    per_rec = per_recording_dynamics(all_graphs, labels, n_states=n_states)
    per_rec.to_csv(os.path.join(metrics_root, "state_dynamics_per_recording.tsv"),
                   sep="\t", index=False)
    subj_dyn = per_subject_aggregate(per_rec)
    subj_dyn.to_csv(os.path.join(metrics_root, "state_dynamics_per_subject.tsv"),
                    sep="\t", index=False)
    group_summary(per_rec).to_csv(
        os.path.join(metrics_root, "state_dynamics_group_summary.tsv"),
        sep="\t", index=False,
    )
    print(f"  [{tag}] wrote state_dynamics_*.tsv  ({len(per_rec)} recordings, "
          f"{len(subj_dyn)} subjects, n_states={n_states})")

    # C.3: LOSO decoder on (latent_means, state_probs, transitions, combined)
    latent_per_sub = aggregate_latents_per_subject(
        bundle.embeds, bundle.subject_ids, bundle.diagnosis_groups,
    )
    subject_table = build_subject_feature_table(
        per_rec, latent_per_subject_df=latent_per_sub, label_col="diagnosis_group",
    )
    if subject_table["diagnosis_group"].nunique() >= 2:
        dec_metrics, dec_preds = loso_decoder(subject_table, label_col="diagnosis_group")
        dec_metrics.to_csv(os.path.join(metrics_root, "decoder_metrics.tsv"), sep="\t", index=False)
        dec_preds.to_csv(os.path.join(metrics_root, "decoder_predictions.tsv"), sep="\t", index=False)
        best = dec_metrics.iloc[0]
        print(f"  [{tag}] best LOSO decoder: {best['feature_set']} "
              f"AUC={best['macro_auc_ovr']:.3f} acc={best['accuracy']:.3f} F1={best['macro_f1']:.3f}")
    else:
        print(f"  [{tag}] skipping LOSO decoder: fewer than 2 diagnosis groups present")

    # C.4: reconstruction & latent diagnostics (autoencoders only — the
    # encoder-only enc_gae_fc has no reconstruction to score).
    if is_encoder:
        print(f"  [{tag}] encoder-only model: skipping reconstruction diagnostics")
    else:
        edge_mse = per_edge_reconstruction_error(model, all_graphs, device=device)
        __import__("numpy").save(os.path.join(metrics_root, "per_edge_mse.npy"), edge_mse)
        subj_mse = per_subject_recon_mse(model, all_graphs, device=device)
        subj_mse.to_csv(os.path.join(metrics_root, "per_subject_recon_mse.tsv"), sep="\t", index=False)
        print(f"  [{tag}] per-edge MSE shape={edge_mse.shape}, "
              f"global mean={edge_mse.mean():.4f}")

    if variational:
        per_dim = kl_per_dim(model, all_graphs, device=device)
        __import__("numpy").save(os.path.join(metrics_root, "kl_per_dim.npy"), per_dim)
        coll = posterior_collapse_fraction(model, all_graphs, device=device)
        import json as _json
        with open(os.path.join(metrics_root, "posterior_collapse.json"), "w") as f:
            _json.dump(coll, f, indent=2)
        mu_lv = mu_logvar_stats(model, all_graphs, device=device)
        mu_lv.to_csv(os.path.join(metrics_root, "mu_logvar_stats.tsv"), sep="\t", index=False)
        print(f"  [{tag}] posterior collapse: {coll['n_collapsed']}/{coll['latent_dim']} "
              f"dims under tol={coll['tol']} (fraction={coll['fraction_collapsed']:.3f})")

    print(f"  [{tag}] metrics written to {metrics_root}")


# ---------------- enc_gae_fc + cebra (contrastive) path ----------------


def _build_cebra_config(args, in_channels: int, overrides: Optional[Dict] = None) -> Dict:
    config = {
        "in_channels": in_channels,
        "hidden_dims": [64, 64, 32, 16],
        "latent_dim": args.cebra_latent_dim,
        "dropout": args.cebra_dropout,
        "lr": args.cebra_lr,
        "weight_decay": args.cebra_weight_decay,
        "batch_size": args.cebra_batch_size,
        "model_kind": "enc_gae_fc",
        "loss": "cebra",
        "temperature": args.cebra_temperature,
        "min_temp": args.cebra_min_temp,
        "learn_temperature": not args.cebra_fixed_temp,
    }
    if overrides:
        config.update(overrides)
    return config


def stage_tune_cebra(args, out_root: str, data: dict) -> Dict:
    """Lightweight grid search for enc_gae_fc/cebra (no Ray).

    Sweeps temperature x lr x latent_dim, trains a few epochs on the train pairs,
    scores val InfoNCE, and returns the best {temperature, lr, latent_dim}.
    """
    from itertools import product
    from torch.utils.data import DataLoader as TorchDataLoader
    tag = "enc_gae_fc"
    print(f"\n=== STAGE: TUNE ({tag}, cebra grid) ===")
    tune_dir = os.path.join(out_root, "tuning", tag)
    os.makedirs(tune_dir, exist_ok=True)
    device = select_device(prefer_gpu=not args.cpu)
    in_channels = data["train"][0].x.shape[1]
    bs = args.cebra_batch_size

    train_ds = CebraPairDataset(data["train"])
    val_ds = CebraPairDataset(data["val"])
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("  [cebra-tune] not enough pairs; falling back to defaults")
        return {}
    train_loader = TorchDataLoader(train_ds, batch_size=bs, shuffle=True,
                                   collate_fn=collate_pairs)
    val_loader = TorchDataLoader(val_ds, batch_size=bs, shuffle=False,
                                 collate_fn=collate_pairs)

    grid = list(product(args.cebra_tune_temperatures, args.cebra_tune_lrs,
                        args.cebra_tune_latent_dims))
    results = []
    best = None
    best_val = float("inf")
    for temp, lr, ld in grid:
        set_seed(args.seed)
        model = GNNEncoder(in_channels=in_channels, hidden_dims=[64, 64, 32, 16],
                           latent_dim=ld, dropout=args.cebra_dropout).to(device)
        criterion = InfoNCE(temperature=temp, learn_temperature=not args.cebra_fixed_temp,
                            min_temp=args.cebra_min_temp).to(device)
        params = list(model.parameters()) + list(criterion.parameters())
        optim = torch.optim.AdamW(params, lr=lr, weight_decay=args.cebra_weight_decay)
        for _ in range(args.cebra_tune_epochs):
            model.train()
            for ref_b, pos_b in train_loader:
                ref_b = ref_b.to(device); pos_b = pos_b.to(device)
                optim.zero_grad(set_to_none=True)
                rz = model(ref_b.x, ref_b.edge_index, ref_b.batch)
                pz = model(pos_b.x, pos_b.edge_index, pos_b.batch)
                loss, _ = criterion(rz, pz)
                loss.backward(); optim.step()
        val_loss, val_align = _cebra_eval(model, criterion, val_loader, device)
        results.append({"temperature": temp, "lr": lr, "latent_dim": ld,
                        "val_infonce": val_loss, "val_alignment": val_align})
        print(f"  temp={temp} lr={lr} latent_dim={ld} -> val_infonce={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            best = {"temperature": temp, "lr": lr, "latent_dim": ld}
        del model, criterion, optim
    with open(os.path.join(tune_dir, "best_config.json"), "w") as f:
        json.dump({"best": best, "grid_results": results}, f, indent=2, default=float)
    print(f"  [cebra-tune] best={best} (val_infonce={best_val:.5f})")
    return best or {}


def _cebra_eval(model, criterion, loader, device):
    """Mean contrastive loss + alignment over a pair-loader (no grad)."""
    model.eval()
    tot_loss, tot_align, n = 0.0, 0.0, 0
    with torch.no_grad():
        for ref_b, pos_b in loader:
            ref_b = ref_b.to(device)
            pos_b = pos_b.to(device)
            ref_z = model(ref_b.x, ref_b.edge_index, ref_b.batch)
            pos_z = model(pos_b.x, pos_b.edge_index, pos_b.batch)
            loss, align = criterion(ref_z, pos_z)
            tot_loss += float(loss)
            tot_align += float(align)
            n += 1
    if n == 0:
        return 0.0, 0.0
    return tot_loss / n, tot_align / n


def stage_train_cebra(args, out_root: str, data: dict, overrides: Optional[Dict] = None
                      ) -> Tuple[torch.nn.Module, Dict]:
    from torch.utils.data import DataLoader as TorchDataLoader
    tag = "enc_gae_fc"
    print(f"\n=== STAGE: TRAIN FINAL ({tag}, cebra/InfoNCE) ===")
    model_dir = os.path.join(out_root, "models", tag)
    os.makedirs(model_dir, exist_ok=True)

    device = select_device(prefer_gpu=not args.cpu)
    in_channels = data["train"][0].x.shape[1]
    config = _build_cebra_config(args, in_channels, overrides=overrides)
    set_seed(args.seed)

    train_ds = CebraPairDataset(data["train"])
    val_ds = CebraPairDataset(data["val"])
    if len(train_ds) == 0:
        raise RuntimeError(
            "no temporal (ref, pos) pairs in TRAIN — every recording has < 2 "
            "epochs, so the time-contrastive loss has nothing to contrast.")
    print(f"  contrastive pairs: train={len(train_ds)} val={len(val_ds)}")
    train_loader = TorchDataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_pairs, num_workers=0)
    val_loader = TorchDataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        collate_fn=collate_pairs, num_workers=0)

    model = GNNEncoder(
        in_channels=config["in_channels"], hidden_dims=config["hidden_dims"],
        latent_dim=config["latent_dim"], dropout=config["dropout"]).to(device)
    criterion = InfoNCE(
        temperature=config["temperature"],
        learn_temperature=config["learn_temperature"],
        min_temp=config["min_temp"]).to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    optim = torch.optim.AdamW(params, lr=config["lr"],
                              weight_decay=config["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=15)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since = 0
    history = {"train": [], "val": [], "align": [], "temp": []}
    for ep in range(args.train_epochs):
        model.train()
        tl, n = 0.0, 0
        for ref_b, pos_b in train_loader:
            ref_b = ref_b.to(device)
            pos_b = pos_b.to(device)
            optim.zero_grad(set_to_none=True)
            ref_z = model(ref_b.x, ref_b.edge_index, ref_b.batch)
            pos_z = model(pos_b.x, pos_b.edge_index, pos_b.batch)
            loss, _ = criterion(ref_z, pos_z)
            loss.backward()
            optim.step()
            tl += float(loss)
            n += 1
        tl = tl / max(1, n)
        val, align = _cebra_eval(model, criterion, val_loader, device)
        sched.step(val)
        history["train"].append(tl)
        history["val"].append(val)
        history["align"].append(align)
        history["temp"].append(float(criterion.temperature))
        if val < best_val - 1e-6:
            best_val = val
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since = 0
        else:
            epochs_since += 1
        if ep % 10 == 0:
            print(f"  ep {ep:03d}  train={tl:.6f}  val={val:.6f}  "
                  f"best_val={best_val:.6f}  align={align:.4f}  "
                  f"temp={criterion.temperature:.3f}")
        if epochs_since >= args.train_patience:
            print(f"  early stop at ep {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    torch.save({"model_state_dict": model.state_dict(), "config": config,
                "best_epoch": best_epoch, "best_val": best_val},
               os.path.join(model_dir, "model.pt"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=float)
    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train"], label="train", linewidth=1.5)
    ax.plot(history["val"], label="val", linewidth=1.5)
    ax.axvline(best_epoch, linestyle="--", color="grey", label=f"best ep={best_epoch}")
    ax.set_title("enc_gae_fc InfoNCE (contrastive) loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("InfoNCE loss")
    ax2 = ax.twinx()
    ax2.plot(history["align"], color="green", alpha=0.6, label="alignment")
    ax2.set_ylabel("positive alignment (cosine)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()
    return model, config


def stage_test_cebra(args, out_root: str, model, config, data) -> Dict:
    from torch.utils.data import DataLoader as TorchDataLoader
    tag = "enc_gae_fc"
    print(f"\n=== STAGE: TEST (held-out, one-shot) ({tag}, cebra) ===")
    test_dir = os.path.join(out_root, "final_test", tag)
    os.makedirs(test_dir, exist_ok=True)
    device = next(model.parameters()).device
    criterion = InfoNCE(
        temperature=config["temperature"],
        learn_temperature=False, min_temp=config["min_temp"]).to(device)
    bs = config.get("batch_size", 256)
    val_loss, val_align = _cebra_eval(
        model, criterion,
        TorchDataLoader(CebraPairDataset(data["val"]), batch_size=bs,
                        shuffle=False, collate_fn=collate_pairs), device)
    test_loss, test_align = _cebra_eval(
        model, criterion,
        TorchDataLoader(CebraPairDataset(data["test"]), batch_size=bs,
                        shuffle=False, collate_fn=collate_pairs), device)
    report = {
        "val_infonce": float(val_loss), "val_alignment": float(val_align),
        "test_infonce": float(test_loss), "test_alignment": float(test_align),
        "n_test_graphs": len(data["test"]), "config": config,
    }
    with open(os.path.join(test_dir, "test_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"  test InfoNCE = {test_loss:.6f}  alignment = {test_align:.4f}; "
          f"report saved to {test_dir}")
    return report


# ---------------- driver ----------------


def main():
    parser = argparse.ArgumentParser()
    # Data dirs default to None and are resolved from --type_data in main()
    # (so an explicit path always overrides the type_data default).
    parser.add_argument("--patient_dir", default=None,
                        help="wSMI patient root. In timeseries mode this is the "
                             "wSMI basis used by the outlier filter.")
    parser.add_argument("--control_dir", default=None,
                        help="wSMI control root (None for patients-only).")
    parser.add_argument("--diagnosis_csv", default=(
        "metadata/DoC_metadata/metadata_patient_labels.csv"))
    parser.add_argument("--coords_file", default=(
        "gnn_connectivity/data_scalp/biosemi64.txt"))
    parser.add_argument("--output_root", default="gnn_connectivity/output")
    parser.add_argument("--run_name", default=None,
                        help="Defaults to wsmi_run_<timestamp>")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_trials", type=int, default=30,
                        help="Ray Tune ASHA samples per model")
    parser.add_argument("--tune_epochs", type=int, default=80,
                        help="Max epochs per ASHA trial (max_t)")
    parser.add_argument("--cpus_per_trial", type=int, default=2)
    parser.add_argument("--max_concurrent_trials", type=int, default=None,
                        help="Cap simultaneous Ray Tune trials. Each concurrent "
                             "trial holds its own copy of the dataset, so peak RAM "
                             "~ dataset*(1+this). Lower it on memory-limited machines "
                             "(default: Ray uses cpus/cpus_per_trial).")
    parser.add_argument("--train_epochs", type=int, default=150)
    parser.add_argument("--train_patience", type=int, default=25)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--stage", default="all",
                        choices=["load", "tune", "train", "test",
                                 "latents", "cluster", "all"])
    parser.add_argument("--models", nargs="+", default=["gae", "vgae"],
                        choices=["gae", "vgae", "gae_vae", "enc_gae_fc"])
    parser.add_argument("--input_mode", default="wsmi",
                        choices=["wsmi", "timeseries"],
                        help="Node-feature modality: pre-computed wSMI matrices "
                             "or raw per-electrode time-series.")
    parser.add_argument("--type_data", default="rs", choices=["lg", "rs"],
                        help="Dataset/task: 'lg' (local-global) or 'rs' (resting-state). "
                             "Sets default data dirs. Timeseries epochs are cropped to "
                             "[crop_tmin, crop_tmax] for BOTH tasks (same 0.8 s window).")
    # Timeseries crop window (seconds), applied to BOTH rs and lg. Both cohorts'
    # epochs span -0.2..0.6 s after our preprocessing, so the window depends only on
    # the data, not on the model (gae_vae / cebra all see the same 0.8 s).
    parser.add_argument("--crop_tmin", "--lg_tmin", dest="crop_tmin",
                        type=float, default=-0.2,
                        help="Timeseries crop start (s). Default -0.2. "
                             "(--lg_tmin kept as a deprecated alias.)")
    parser.add_argument("--crop_tmax", "--lg_tmax", dest="crop_tmax",
                        type=float, default=0.6,
                        help="Timeseries crop end (s). Default 0.6. "
                             "(--lg_tmax kept as a deprecated alias.)")
    parser.add_argument("--loss", default="mse",
                        choices=["mse", "mse_corr", "cebra"],
                        help="Training objective. mse/mse_corr need a decoder "
                             "model (gae/vgae/gae_vae); cebra needs enc_gae_fc.")
    parser.add_argument("--graph_latent_agg", default="flatten",
                        choices=["mean", "flatten"],
                        help="How per-node (per-electrode) latents become ONE vector "
                             "per graph for clustering (autoencoder models only). "
                             "'flatten' (default) = concatenate electrodes in fixed order "
                             "(size 64*latent_dim; block k = electrode k, so the "
                             "spatial/electrode notion is preserved). 'mean' = average over "
                             "electrodes (size latent_dim; loses electrode identity).")
    # --- outlier filter ---
    parser.add_argument("--filter_outliers", action="store_true",
                        help="Drop epochs whose wSMI matrix is a Mahalanobis 2-sigma "
                             "outlier under a Gaussian fit on TRAIN only.")
    parser.add_argument("--outlier_n_sigma", type=float, default=2.0)
    parser.add_argument("--outlier_threshold", default="empirical",
                        choices=["empirical", "chi2"],
                        help="2-sigma boundary: 'empirical' (conf-quantile of TRAIN "
                             "d2, robust to feature correlation) or 'chi2' "
                             "(theoretical chi2 quantile assuming independence).")
    # --- enc_gae_fc / cebra hyperparameters ---
    parser.add_argument("--cebra_latent_dim", type=int, default=8)
    parser.add_argument("--cebra_dropout", type=float, default=0.1)
    parser.add_argument("--cebra_lr", type=float, default=1e-3)
    parser.add_argument("--cebra_weight_decay", type=float, default=1e-5)
    parser.add_argument("--cebra_batch_size", type=int, default=256)
    parser.add_argument("--cebra_temperature", type=float, default=1.0)
    parser.add_argument("--cebra_min_temp", type=float, default=0.1)
    parser.add_argument("--cebra_fixed_temp", action="store_true",
                        help="Freeze the InfoNCE temperature (default: learnable).")
    parser.add_argument("--cebra_tune_epochs", type=int, default=8,
                        help="Epochs per grid point in the cebra tune.")
    parser.add_argument("--cebra_tune_temperatures", type=float, nargs="+",
                        default=[0.5, 1.0])
    parser.add_argument("--cebra_tune_lrs", type=float, nargs="+",
                        default=[1e-3, 3e-4])
    parser.add_argument("--cebra_tune_latent_dims", type=int, nargs="+",
                        default=[8, 16])
    parser.add_argument("--timeseries_patient_dir", default=None,
                        help="Root of .fif epoch files for patients "
                             "(only used when --input_mode=timeseries)")
    parser.add_argument("--timeseries_control_dir", default=None,
                        help="Root of .fif epoch files for controls "
                             "(only used when --input_mode=timeseries)")
    parser.add_argument("--sfreq", type=float, default=100.0,
                        help="Target sampling rate (Hz) for time-series mode; "
                             "epochs are resampled if needed.")
    parser.add_argument("--window_sec", type=float, default=16.0,
                        help="Window length (seconds) per graph in time-series mode. "
                             "Each graph carries x of shape (n_channels, "
                             "window_sec * sfreq).")
    parser.add_argument("--lambda_corr", type=float, default=None,
                        help="Override the Pearson-correlation regularizer weight "
                             "for stage_train (skip Ray Tune's value). Set 0 to disable.")
    parser.add_argument("--patients_only", action="store_true",
                        help="Skip the healthy-control directory and train/cluster "
                             "only on DoC patients. Diagnosis groups become {UWS, MCS}.")
    parser.add_argument("--subject_filter", default=None,
                        help="Comma-separated subject IDs to load (e.g. 'AA048,AA069'). "
                             "Limits memory for smoke tests / quick runs.")
    parser.add_argument("--diagnosis_granularity", default="coarse",
                        choices=["coarse", "fine"],
                        help="coarse -> {CONTROL, UWS, MCS}; fine -> raw labels "
                             "(UWS, MCS-, MCS+, EMCS, COMA, CONTROL) with VS->UWS, "
                             "keeping EMCS/COMA instead of dropping them.")
    parser.add_argument("--max_epochs_per_recording", type=int, default=None,
                        help="Randomly subsample at most N epochs per recording "
                             "(seeded). Bounds RAM / speeds tuning on large (lg) "
                             "datasets. Default None = use all epochs.")
    parser.add_argument("--cebra_tune", action="store_true",
                        help="Run a small grid search (temperature/lr/latent_dim) "
                             "for enc_gae_fc/cebra before training.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny config for fast smoke test")
    args = parser.parse_args()

    # --- resolve data-dir defaults from --type_data (explicit paths win) ---
    d = DATA_DEFAULTS[args.type_data]
    if args.patient_dir is None:
        args.patient_dir = d["wsmi_patient"]
    if args.control_dir is None and not args.patients_only:
        args.control_dir = d["wsmi_control"]
    if args.timeseries_patient_dir is None:
        args.timeseries_patient_dir = d["ts_patient"]
    if args.timeseries_control_dir is None and not args.patients_only:
        args.timeseries_control_dir = d["ts_control"]

    # --- loss/model compatibility ---
    decoder_models = {"gae", "vgae", "gae_vae"}
    if args.loss == "cebra":
        bad = [m for m in args.models if m != "enc_gae_fc"]
        if bad:
            parser.error(f"--loss cebra requires --models enc_gae_fc; got {bad}")
    else:
        if "enc_gae_fc" in args.models:
            parser.error("--models enc_gae_fc requires --loss cebra "
                         "(it is encoder-only, no reconstruction for MSE)")
    if args.loss == "mse_corr" and args.input_mode != "timeseries":
        parser.error("--loss mse_corr is only meaningful with "
                     "--input_mode timeseries (the Pearson regularizer)")

    if args.input_mode == "timeseries":
        required = ["timeseries_patient_dir"]
        if not args.patients_only:
            required.append("timeseries_control_dir")
        missing = [n for n in required if getattr(args, n) is None]
        if missing:
            parser.error(
                "--input_mode=timeseries requires: "
                + ", ".join(f"--{n}" for n in missing)
            )

    if args.smoke:
        args.num_trials = 2
        args.tune_epochs = 3
        args.train_epochs = 5
        args.train_patience = 3

    run_name = args.run_name or f"wsmi_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root = os.path.join(args.output_root, run_name)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    if args.stage in ("load", "all"):
        data = stage_load(args, out_root)
    else:
        data = _load_cached(out_root)

    cebra_path = _is_encoder_loss(args.loss)
    for model_kind in args.models:
        tag = model_kind
        best = None
        model = None
        config = None

        # GAE-family uses Ray Tune; the contrastive encoder uses an optional
        # lightweight grid (stage_tune_cebra), else fixed defaults.
        if args.stage in ("tune", "all"):
            if cebra_path:
                if args.cebra_tune:
                    best = stage_tune_cebra(args, out_root, data)
            else:
                best = stage_tune(args, out_root, data, model_kind)
        else:
            best_path = os.path.join(out_root, "tuning", tag, "best_config.json")
            if os.path.exists(best_path):
                with open(best_path) as f:
                    best = json.load(f)
                if cebra_path and isinstance(best, dict):
                    best = best.get("best", best)

        if args.stage in ("train", "all"):
            if cebra_path:
                model, config = stage_train_cebra(args, out_root, data, overrides=best)
            else:
                if best is None:
                    raise RuntimeError(f"No best config found for {tag} - run --stage tune first")
                model, config = stage_train(args, out_root, data, best, model_kind)
        else:
            model_path = os.path.join(out_root, "models", tag, "model.pt")
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
                config = ckpt["config"]
                in_channels = data["train"][0].x.shape[1]
                config["in_channels"] = in_channels
                # legacy checkpoints may not carry model_kind
                config.setdefault("model_kind", model_kind)
                model = _build_model(model_kind, config)
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(select_device(prefer_gpu=not args.cpu))

        if args.stage in ("test", "all"):
            if model is None:
                print(f"  no {tag} model loaded - skipping test")
            elif cebra_path:
                stage_test_cebra(args, out_root, model, config, data)
            else:
                stage_test(args, out_root, model, config, data, model_kind)

        if args.stage in ("latents", "cluster", "all"):
            if model is None:
                print(f"  no {tag} model loaded - skipping clustering")
            else:
                stage_latents_and_cluster(args, out_root, model, data, model_kind)

    print(f"\n=== DONE ===\nOutputs at: {out_root}")


if __name__ == "__main__":
    main()

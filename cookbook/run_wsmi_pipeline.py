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

from model import GAE, VGAE, kl_divergence  # noqa: E402
from train import (  # noqa: E402
    apply_normalization, compute_mse_on_graphs,
    compute_normalization_stats, run_ray_tune_wsmi, train_one_epoch,
)
from wsmi_loader import load_wsmi_dataset  # noqa: E402
from data_loaders import split_by_subject_stratified  # noqa: E402
from cluster_analysis import (  # noqa: E402
    extract_latents_from_graphs, run_all_clusterers,
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


def _build_model(variational: bool, config: Dict) -> torch.nn.Module:
    Cls = VGAE if variational else GAE
    return Cls(
        in_channels=config["in_channels"],
        hidden_dims=config["hidden_dims"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
    )


# ---------------- stages ----------------


def stage_load(args, out_root: str) -> dict:
    print("\n=== STAGE: LOAD ===")
    graphs, subjects, dgroups = load_wsmi_dataset(
        patient_dir=args.patient_dir,
        control_dir=args.control_dir,
        diagnosis_csv=args.diagnosis_csv,
        coords_file=args.coords_file,
        k=args.k,
    )
    splits_dir = os.path.join(out_root, "splits")
    train_g, val_g, test_g, subject_split = split_by_subject_stratified(
        graphs=graphs, subject_ids=subjects, diagnosis_groups=dgroups,
        test_frac=args.test_frac, val_frac=args.val_frac,
        random_state=args.seed, persist_dir=splits_dir,
    )

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


def stage_tune(args, out_root: str, data: dict, variational: bool) -> Dict:
    tag = "vgae" if variational else "gae"
    print(f"\n=== STAGE: TUNE ({tag}) — Ray Tune ASHA ===")
    tune_dir = os.path.join(out_root, "tuning", tag)
    os.makedirs(tune_dir, exist_ok=True)

    in_channels = data["train"][0].x.shape[1]
    best_config, trials_df = run_ray_tune_wsmi(
        train_graphs=data["train"],
        val_graphs=data["val"],
        in_channels=in_channels,
        variational=variational,
        num_samples=args.num_trials,
        n_epochs=args.tune_epochs,
        grace_period=max(2, args.tune_epochs // 4),
        cpus_per_trial=args.cpus_per_trial,
        storage_path=os.path.join(tune_dir, "ray_results"),
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


def stage_train(args, out_root: str, data: dict, best: Dict, variational: bool) -> Tuple[torch.nn.Module, Dict]:
    tag = "vgae" if variational else "gae"
    print(f"\n=== STAGE: TRAIN FINAL ({tag}) ===")
    model_dir = os.path.join(out_root, "models", tag)
    os.makedirs(model_dir, exist_ok=True)

    device = select_device(prefer_gpu=not args.cpu)
    in_channels = data["train"][0].x.shape[1]
    drop = {"val_mse", "n_epochs_run", "wall_seconds", "n_epochs"}
    config = {k: v for k, v in best.items() if k not in drop}
    config["in_channels"] = in_channels
    # batch_size may be missing if loaded from older runs; default
    config.setdefault("batch_size", 64)
    config.setdefault("weight_decay", 0.0)
    # Rename: tune calls it beta_kl, train_one_epoch expects 'beta'
    if "beta_kl" in config and "beta" not in config:
        config["beta"] = config.pop("beta_kl")
    config.setdefault("beta", 1.0 if config.get("variational", False) else 0.0)
    config.setdefault("kl_warmup_epochs", 10)
    set_seed(args.seed)
    model = _build_model(variational, config).to(device)
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
    history = {"train": [], "val": [], "kl": [], "beta": []}
    for ep in range(args.train_epochs):
        if variational:
            beta_max = config.get("beta", 1.0)
            warm = max(1, config.get("kl_warmup_epochs", 1))
            beta = beta_max * min(1.0, ep / warm)
        else:
            beta = 0.0
        tl, _, kl = train_one_epoch(model, optim, criterion, loader, device=device,
                                    variational=variational, beta=beta)
        val = compute_mse_on_graphs(model, data["val"], device=device, variational=variational)
        history["train"].append(tl)
        history["val"].append(val)
        history["kl"].append(kl)
        history["beta"].append(beta)
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
                  f"best_val={best_val:.6f}  kl={kl:.4f}  beta={beta:.3f}")
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


def stage_test(args, out_root: str, model, config, data, variational: bool) -> Dict:
    tag = "vgae" if variational else "gae"
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


def stage_latents_and_cluster(args, out_root: str, model, data, variational: bool):
    tag = "vgae" if variational else "gae"
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
    bundle = extract_latents_from_graphs(model, all_graphs, splits, device=device)
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

    # C.4: reconstruction & latent diagnostics
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


# ---------------- driver ----------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_dir", default=(
        "data/markers/wsmi_theta/"
        "nice_epochs_sfreq-100Hz_recombine-biosemi64_dur-16s"))
    parser.add_argument("--control_dir", default=(
        "data/markers/wsmi_theta/control_bids_biosemi64_dur-16_tau10"))
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
    parser.add_argument("--train_epochs", type=int, default=150)
    parser.add_argument("--train_patience", type=int, default=25)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--stage", default="all",
                        choices=["load", "tune", "train", "test",
                                 "latents", "cluster", "all"])
    parser.add_argument("--models", nargs="+", default=["gae", "vgae"],
                        choices=["gae", "vgae"])
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny config for fast smoke test")
    args = parser.parse_args()

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

    for variational in [m == "vgae" for m in args.models]:
        tag = "vgae" if variational else "gae"
        best = None
        model = None
        config = None

        if args.stage in ("tune", "all"):
            best = stage_tune(args, out_root, data, variational)
        else:
            best_path = os.path.join(out_root, "tuning", tag, "best_config.json")
            if os.path.exists(best_path):
                with open(best_path) as f:
                    best = json.load(f)

        if args.stage in ("train", "all"):
            if best is None:
                raise RuntimeError(f"No best config found for {tag} - run --stage tune first")
            model, config = stage_train(args, out_root, data, best, variational)
        else:
            model_path = os.path.join(out_root, "models", tag, "model.pt")
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
                config = ckpt["config"]
                in_channels = data["train"][0].x.shape[1]
                config["in_channels"] = in_channels
                model = _build_model(variational, config)
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(select_device(prefer_gpu=not args.cpu))

        if args.stage in ("test", "all"):
            if model is None:
                print(f"  no {tag} model loaded - skipping test")
            else:
                stage_test(args, out_root, model, config, data, variational)

        if args.stage in ("latents", "cluster", "all"):
            if model is None:
                print(f"  no {tag} model loaded - skipping clustering")
            else:
                stage_latents_and_cluster(args, out_root, model, data, variational)

    print(f"\n=== DONE ===\nOutputs at: {out_root}")


if __name__ == "__main__":
    main()

# Lab notebook — GNN connectivity / raw wSMI clustering

> This directory is **gitignored**. It's a working notebook of the
> exploratory analyses I'm running on the wSMI matrices that Greta sent
> on 2026-05-26 (256 EGI electrodes, theta band, 178 sessions,
> 144 unique subjects after dropping the 2 with unknown CRS diagnosis).
>
> Everything here is the **raw-wSMI baseline** — no GAE / VGAE yet. The
> goal of the notebook is to nail down (a) what signal is already in the
> data, (b) how to interpret it, (c) what numbers the learned
> representation must beat.

---

**See also**: [paper_draft.md](./paper_draft.md) — section-by-section
journal-paper skeleton (≤ 3–4 sentences per section), with pointers
back into the chapters for the full numbers and figures.

---

## Read in this order

1. **[methodology.md](./methodology.md)** — data, subject splits,
   per-fold pipeline, **how we guarantee no leakage**, reproducibility
   notes. Read this first; the chapters assume it.

2. **[chapter_01_clustering_ablation.md](./chapter_01_clustering_ablation.md)**
   — exploratory ablation of clustering methods (KMeans / GMM /
   Spectral / Louvain) at K ∈ {4, 6, 8} on the raw wSMI features.
   **Winner: GMM K=4** with Cramér's V = 0.183 on the full dataset.

3. **[chapter_02_interpretability.md](./chapter_02_interpretability.md)**
   — what do these clusters mean? Centroid wSMI matrices, manifold
   visualisations (PCA(2), UMAP(2)), per-subject trajectories. **Headline
   result: PC1 of the wSMI feature space recovers the clinical
   consciousness gradient as a strictly monotone axis** — no
   supervision involved.

4. **[chapter_03_optimisations.md](./chapter_03_optimisations.md)** —
   data-engineering improvements: take only the last 100 epochs per
   session, balance across diagnoses, drop to K=3 instead of K=4.
   **New benchmark: V = 0.280 in-sample, 0.612 3-class held-out
   balanced accuracy** vs 0.333 random.

5. **[chapter_04_regional_interpretation.md](./chapter_04_regional_interpretation.md)**
   — anatomical interpretation of the K=3 model with literature
   comparison. Recovers the canonical "posterior hot zone" + long-range
   coupling signature of consciousness (Sitt 2014 / King 2013 / Casarotto
   2016). **Most consciousness-sensitive coupling: P–O (0.018)**;
   least: T–T (0.001).

6. **[chapter_05_contrastive_pretraining.md](./chapter_05_contrastive_pretraining.md)**
   — Why reconstruction is the wrong loss for wSMI (same argument as
   CoCoEEG-fmri); MoCo-v2 + SAGEConv encoder design that outputs a
   256-D per-epoch embedding; wSMI-specific augmentation pipeline.
   **Negative result**: three augmentation-strength variants all
   underperform the raw-PCA baseline; lower pretrain loss correlates
   with worse downstream accuracy → augmentation invariance is the
   wrong inductive bias for this signal.

7. **[chapter_06_supervised_and_binary_eval.md](./chapter_06_supervised_and_binary_eval.md)**
   — Pivot to supervised end-to-end on the same encoder + 3-class
   diagnosis head, plus a full binary-discrimination ROC AUC benchmark
   across the four best models (GMM K=3, K=4, supervised GCN, MoCo
   GCN wave-1) on *control vs any-other* and *MCS vs UWS/COMA*.
   **Headline**: the supervised GCN wins control-vs-DOC decisively
   (AUC 0.931 ± 0.083 vs the GMM baseline 0.815 ± 0.095, +0.116) but
   ties on MCS-vs-UWS (~0.7). The 3-class bal_acc metric averages
   these and so favours the GMMs (0.611 ± 0.131 vs 0.527 ± 0.118 for
   the supervised GCN, 0.535 ± 0.086 for MoCo). MoCo has the
   tightest std on every metric (0.069 / 0.127 / 0.086) but lower
   mean AUC than the unsupervised baseline.

---

## One-screen summary

| stage | best result | source |
|---|---|---|
| Method ablation | GMM K=4, V = 0.183 (full data) | [ch 1](./chapter_01_clustering_ablation.md) |
| Interpretability (K=4) | PC1 monotone in clinical consciousness order | [ch 2](./chapter_02_interpretability.md) |
| Partition optimisation | last_100 × balanced → V = 0.252 | [ch 3 §3.4](./chapter_03_optimisations.md#34-stacking-the-two-levers-last_100--balanced) |
| K optimisation | K=3 best for held-out | [ch 3 §3.5–3.6](./chapter_03_optimisations.md#35-the-tiny-outlier-cluster-question--k3-vs-k4) |
| Held-out 3-class bal_acc | **0.611 ± 0.131** (K=3, 5-fold) / 0.590 pooled LOOCV | [ch 3 §3.6](./chapter_03_optimisations.md#36-held-out-validation-does-the-v0252-actually-mean-anything) / [ch 6 §6.3](./chapter_06_supervised_and_binary_eval.md#63-headline-result-per-fold-3-class-balanced-accuracy) |
| Regional + literature | **P–O most consciousness-sensitive (0.018)** | [ch 4](./chapter_04_regional_interpretation.md) |
| MoCo contrastive (binary AUC) | control 0.772 ± 0.069, MCS-vs-UWS 0.696 ± 0.127 (lowest std but lowest mean) | [ch 5](./chapter_05_contrastive_pretraining.md) / [ch 6](./chapter_06_supervised_and_binary_eval.md) |
| Supervised end-to-end | control-vs-DOC AUC **0.931 ± 0.083** (+0.116 vs baseline) | [ch 6](./chapter_06_supervised_and_binary_eval.md) |
| Binary ROC AUC benchmark | **GMM K=3 control 0.815 ± 0.095, MCS-vs-UWS 0.713 ± 0.155** | [ch 6 §6.4](./chapter_06_supervised_and_binary_eval.md#64-binary-discrimination--the-structure-3-class-bal_acc-hides) |

---

## The benchmark the GAE will need to beat

Same `last_100 × balanced` partition, same `GroupKFold` per-subject
splits, three prediction strategies (see
[methodology §3.3](./methodology.md#33-three-prediction-strategies-held-out-evaluation)).
Replace raw-wSMI → PCA(50) features with a learned GAE / VGAE latent
representation, then re-score:

| metric | raw-wSMI floor | aspirational |
|---|---|---|
| in-sample Cramér's V | 0.280 | > 0.30 |
| held-out 3-class bal_acc (mean) | 0.612 | > 0.65 |
| held-out 6-class bal_acc (mean) | 0.334 | > 0.40 |

If the GAE matches or barely exceeds these, the learned representation
is recapitulating raw-wSMI structure. If it pulls notably ahead, we've
found a useful encoding.

---

## Where the artefacts live

Per the [methodology](./methodology.md), all plots / models / metrics
are under `output/` in the repo root (gitignored). Key directories:

| dir | contents |
|---|---|
| [output/full_cluster/](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/output/full_cluster/) | full-data KMeans + GMM at K ∈ {4,6,8}, X_pca, meta |
| [output/raw_cluster_sanity/](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/output/raw_cluster_sanity/) | subsample ablation: KMeans + GMM + Spectral + Louvain |
| [output/partition_search/](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/output/partition_search/) | 7 partition strategies |
| [output/time_within_session/](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/output/time_within_session/) | quintile + window-size sweeps |
| [output/combined_partition/](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/output/combined_partition/) | last_N × balanced × K sweep |
| [output/centroid_last_100_balanced/](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/output/centroid_last_100_balanced/) | 4 centroid wSMI matrices |
| [output/manifold_viz/](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/output/manifold_viz/) | PCA(2), UMAP(2), consciousness axis |
| [output/holdout/](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/output/holdout/) | 5-fold K=4 |
| [output/holdout_K3/](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/output/holdout_K3/) | 5-fold K=3 |
| [output/holdout_regcov/](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/output/holdout_regcov/) | 5-fold K=4 reg_covar=1e-2 |
| [output/holdout_loocv/](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/output/holdout_loocv/) | LOOCV K=4 (in flight) |

Scripts under `scripts/`, sbatch wrappers under `slurm/`, logs under
`slurm/logs/` (gitignored). Every script accepts `--random_state 42`
and is otherwise hyperparameter-frozen.

---

## Provenance / housekeeping

- Data load timestamp on Margaret: 2026-05-26
- All numbers in this notebook reproduce with `random_state=42` across
  every fitting step (StandardScaler → PCA → GMM → LogisticRegression).
- Compute: SLURM batch jobs on `normal-best` partition (CPU-only at this
  stage; no GPU needed for raw-wSMI clustering).
- Notebook last updated: 2026-06-01.

---

## Status / TODOs (snapshot 2026-06-01)

- [x] Methodology + reproducibility doc
- [x] Chapter 1 — clustering ablation
- [x] Chapter 2 — interpretability for GMM K=4
- [x] Chapter 3 — optimisations + held-out validation (5-fold)
- [x] K=3 centroid plot — see [chapter 3 §3.7](./chapter_03_optimisations.md#37-the-k3-model--centroids-and-manifold)
- [x] K=3 manifold viz — see [chapter 3 §3.7c](./chapter_03_optimisations.md#37c-k3-manifold--pca2-and-umap2)
- [x] Chapter 4 — regional interpretation + literature comparison
- [x] LOOCV K=4 — V = 0.254 ± 0.009; pooled 3-class hard-FP bal_acc = **0.601**
- [x] LOOCV K=3 — V = 0.283 ± 0.010; pooled 3-class soft-FP bal_acc = **0.590** (essentially tied with K=4 LOOCV; see [chapter 3 §3.6e](./chapter_03_optimisations.md#36e-loocv--the-tightest-held-out-estimate))
- [x] Chapter 5 — MoCo contrastive pretraining (negative result)
- [x] Chapter 6 — supervised end-to-end + binary ROC AUC benchmarks (3-class bal_acc 0.527 ± 0.118, control-vs-DOC AUC 0.931 ± 0.083)
- [ ] Multi-band / multi-feature extension (theta + α + β + δ + γ wSMI)
- [ ] Outcome prediction (`cs_{6m,1y,2y}` from baseline cluster occupancy)
